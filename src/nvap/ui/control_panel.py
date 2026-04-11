"""Simplified NVAP control panel with green-channel pass-through mode."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from nvap.analysis.microglia_vessel_report import MICROGLIA_CELL_REPORT_COLUMNS, MicrogliaCellReport
from nvap.config.types import MeshExportConfig, PSFConfig, PreprocessConfig, RenderConfig


class ControlPanel(QWidget):
    load_requested = Signal()
    apply_psf_requested = Signal()
    render_config_changed = Signal(object)
    psf_config_changed = Signal(object)
    microglia_view_changed = Signal()
    run_microglia_analysis_requested = Signal()
    export_microglia_analysis_requested = Signal()
    export_metrics_requested = Signal()
    export_snapshot_requested = Signal()
    export_mesh_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # Debounce timer to batch rapid changes
        self._render_update_timer = QTimer(self)
        self._render_update_timer.setSingleShot(True)
        self._render_update_timer.setInterval(250)  # 250ms delay
        self._render_update_timer.timeout.connect(self._emit_render_config_delayed)

        self._microglia_update_timer = QTimer(self)
        self._microglia_update_timer.setSingleShot(True)
        self._microglia_update_timer.setInterval(400)  # 400ms delay for heavier operation
        self._microglia_update_timer.timeout.connect(self._emit_microglia_view_change_delayed)

        self._auto_apply_enabled = True
        self._pending_render_update = False
        self._pending_microglia_update = False
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        load_btn = QPushButton("Load Dataset")
        load_btn.setObjectName("primaryAction")
        load_btn.setToolTip("Load a dataset from disk (Ctrl+L)")
        load_btn.clicked.connect(self.load_requested.emit)
        root.addWidget(load_btn)

        # Auto-apply controls
        apply_group = QGroupBox("Update Mode")
        apply_layout = QVBoxLayout(apply_group)
        apply_layout.setContentsMargins(6, 6, 6, 6)

        self.auto_apply_checkbox = QCheckBox("Auto-apply changes")
        self.auto_apply_checkbox.setChecked(True)
        self.auto_apply_checkbox.setToolTip(
            "When enabled, changes update the view automatically after a brief delay.\n"
            "When disabled, click Apply to update the view.\n"
            "Toggle with Ctrl+A"
        )
        self.auto_apply_checkbox.toggled.connect(self._on_auto_apply_toggled)
        apply_layout.addWidget(self.auto_apply_checkbox)

        self.apply_btn = QPushButton("Apply Changes")
        self.apply_btn.setObjectName("primaryAction")
        self.apply_btn.setToolTip("Apply pending changes to the view (F5 or Return)")
        self.apply_btn.setVisible(False)
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        apply_layout.addWidget(self.apply_btn)

        self.pending_label = QLabel("")
        self.pending_label.setObjectName("pendingWarning")
        self.pending_label.setVisible(False)
        apply_layout.addWidget(self.pending_label)

        root.addWidget(apply_group)

        self.show_advanced = QCheckBox("Show advanced controls")
        self.show_advanced.setChecked(False)
        self.show_advanced.toggled.connect(self._set_advanced_visible)
        root.addWidget(self.show_advanced)
        _mode_hint = QLabel("Mode: green channel pass-through (no masking / denoising)")
        _mode_hint.setObjectName("modeHint")
        root.addWidget(_mode_hint)

        self.render_group = self._build_render_group()
        root.addWidget(self.render_group)

        self.preprocess_group = self._build_preprocess_group()
        root.addWidget(self.preprocess_group)

        self.microglia_group = self._build_microglia_group()
        root.addWidget(self.microglia_group)

        self.microglia_analysis_group = self._build_microglia_analysis_group()
        root.addWidget(self.microglia_analysis_group)

        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout(export_group)
        csv_btn = QPushButton("Metrics CSV")
        csv_btn.setToolTip("Export metrics to CSV file (Ctrl+E)")
        csv_btn.clicked.connect(self.export_metrics_requested.emit)
        png_btn = QPushButton("Snapshot PNG")
        png_btn.setToolTip("Export current 3D view as PNG (Ctrl+S)")
        png_btn.clicked.connect(self.export_snapshot_requested.emit)
        mesh_btn = QPushButton("3D Mesh")
        mesh_btn.setObjectName("primaryAction")
        mesh_btn.setToolTip("Export 3D mesh files (Ctrl+M)")
        mesh_btn.clicked.connect(self.export_mesh_requested.emit)
        export_layout.addWidget(csv_btn)
        export_layout.addWidget(png_btn)
        export_layout.addWidget(mesh_btn)
        root.addWidget(export_group)

        plugin_group = QGroupBox("Plugins")
        plugin_layout = QVBoxLayout(plugin_group)
        self.plugin_text = QTextEdit()
        self.plugin_text.setReadOnly(True)
        self.plugin_text.setMaximumHeight(80)
        plugin_layout.addWidget(self.plugin_text)
        root.addWidget(plugin_group)

        metrics_group = QGroupBox("Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(130)
        metrics_layout.addWidget(self.metrics_text)
        root.addWidget(metrics_group)

        self.debug_group = QGroupBox("Log")
        self.debug_group.setCheckable(True)
        self.debug_group.setChecked(False)
        debug_layout = QVBoxLayout(self.debug_group)
        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        self.debug_text.setMaximumHeight(120)
        debug_layout.addWidget(self.debug_text)
        self.debug_group.toggled.connect(lambda checked: self.debug_text.setVisible(checked))
        self.debug_text.setVisible(False)
        root.addWidget(self.debug_group)

        self._set_advanced_visible(False)
        root.addStretch(1)
        self._emit_render_config()
        self._emit_psf_config()

    def _build_render_group(self) -> QGroupBox:
        group = QGroupBox("Rendering")
        form = QFormLayout(group)

        row = QHBoxLayout()
        self.show_green = QCheckBox("Green")
        self.show_green.setChecked(True)
        self.show_green.setObjectName("channelGreen")
        self.show_green.stateChanged.connect(self._on_render_setting_changed)
        self.show_red = QCheckBox("Red")
        self.show_red.setChecked(True)
        self.show_red.setObjectName("channelRed")
        self.show_red.stateChanged.connect(self._on_render_setting_changed)
        row.addWidget(self.show_green)
        row.addWidget(self.show_red)
        form.addRow("Channels", row)

        iso_row = QHBoxLayout()
        self.show_iso_green = QCheckBox("Green")
        self.show_iso_green.setObjectName("channelGreen")
        self.show_iso_green.stateChanged.connect(self._on_render_setting_changed)
        self.show_iso_red = QCheckBox("Red")
        self.show_iso_red.setObjectName("channelRed")
        self.show_iso_red.stateChanged.connect(self._on_render_setting_changed)
        iso_row.addWidget(self.show_iso_green)
        iso_row.addWidget(self.show_iso_red)
        form.addRow("Isosurfaces", iso_row)

        self.threshold_green = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.15)
        self.threshold_green.setToolTip("Minimum intensity to visualize green channel (microglia)")
        self.threshold_green.valueChanged.connect(self._on_render_setting_changed)
        _lbl_thr_g = QLabel("Threshold G")
        _lbl_thr_g.setObjectName("channelGreen")
        form.addRow(_lbl_thr_g, self.threshold_green)

        self.threshold_red = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.15)
        self.threshold_red.setToolTip("Minimum intensity to visualize red channel (vasculature)")
        self.threshold_red.valueChanged.connect(self._on_render_setting_changed)
        _lbl_thr_r = QLabel("Threshold R")
        _lbl_thr_r.setObjectName("channelRed")
        form.addRow(_lbl_thr_r, self.threshold_red)

        self.opacity_green = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.40)
        self.opacity_green.valueChanged.connect(self._on_render_setting_changed)
        _lbl_op_g = QLabel("Opacity G")
        _lbl_op_g.setObjectName("channelGreen")
        form.addRow(_lbl_op_g, self.opacity_green)

        self.opacity_red = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.40)
        self.opacity_red.valueChanged.connect(self._on_render_setting_changed)
        _lbl_op_r = QLabel("Opacity R")
        _lbl_op_r.setObjectName("channelRed")
        form.addRow(_lbl_op_r, self.opacity_red)

        self.iso_green = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.20)
        self.iso_green.valueChanged.connect(self._on_render_setting_changed)
        _lbl_iso_g = QLabel("Iso level G")
        _lbl_iso_g.setObjectName("channelGreen")
        form.addRow(_lbl_iso_g, self.iso_green)

        self.iso_red = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.20)
        self.iso_red.valueChanged.connect(self._on_render_setting_changed)
        _lbl_iso_r = QLabel("Iso level R")
        _lbl_iso_r.setObjectName("channelRed")
        form.addRow(_lbl_iso_r, self.iso_red)

        self.display_z_scale = self._make_unit_spinbox(0.2, 3.0, 0.05, 2.0 / 3.0)
        self.display_z_scale.valueChanged.connect(self._on_render_setting_changed)
        form.addRow("Z height scale", self.display_z_scale)

        self.trim_first_slices = QSpinBox()
        self.trim_first_slices.setRange(0, 9999)
        self.trim_first_slices.setValue(20)
        self.trim_first_slices.valueChanged.connect(self._on_render_setting_changed)
        form.addRow("Trim first Z", self.trim_first_slices)

        self.trim_last_slices = QSpinBox()
        self.trim_last_slices.setRange(0, 9999)
        self.trim_last_slices.setValue(20)
        self.trim_last_slices.valueChanged.connect(self._on_render_setting_changed)
        form.addRow("Trim last Z", self.trim_last_slices)

        self.offset_x = self._make_offset_spinbox()
        self.offset_x.valueChanged.connect(self._on_render_setting_changed)
        form.addRow("Offset X um", self.offset_x)

        self.offset_y = self._make_offset_spinbox()
        self.offset_y.valueChanged.connect(self._on_render_setting_changed)
        form.addRow("Offset Y um", self.offset_y)

        self.offset_z = self._make_offset_spinbox()
        self.offset_z.valueChanged.connect(self._on_render_setting_changed)
        form.addRow("Offset Z um", self.offset_z)
        return group

    def _build_preprocess_group(self) -> QGroupBox:
        group = QGroupBox("Green Channel")
        form = QFormLayout(group)

        self.preprocess_enabled = QCheckBox("Enable preprocessing")
        self.preprocess_enabled.setChecked(True)
        form.addRow(self.preprocess_enabled)

        self.noise_strength = self._make_unit_spinbox(0.002, 0.06, 0.002, 0.012)
        form.addRow("Noise strength", self.noise_strength)

        self.noise_multiplier = self._make_unit_spinbox(0.5, 4.0, 0.1, 1.9)
        form.addRow("Noise multiplier", self.noise_multiplier)

        self.branch_protection = self._make_unit_spinbox(0.0, 1.0, 0.05, 0.72)
        form.addRow("Branch protect", self.branch_protection)

        self.speckle_attenuation = self._make_unit_spinbox(0.0, 1.0, 0.02, 0.12)
        form.addRow("Speckle attenuation", self.speckle_attenuation)

        form.addRow(
            QLabel(
                "Single mode: green pass-through.\n"
                "Green is pass-through (input used as-is).\n"
                "Red keeps the existing workflow."
            )
        )

        self.reprocess_btn = QPushButton("Reprocess Dataset")
        self.reprocess_btn.setToolTip("Run the preprocessing / processing pipeline again.")
        self.reprocess_btn.clicked.connect(self.apply_psf_requested.emit)
        form.addRow(self.reprocess_btn)
        return group

    def _build_microglia_group(self) -> QGroupBox:
        group = QGroupBox("Microglia Viewer")
        form = QFormLayout(group)

        self.microglia_isolate = QCheckBox("View one microglia")
        self.microglia_isolate.toggled.connect(self._on_microglia_setting_changed)
        form.addRow(self.microglia_isolate)

        row = QHBoxLayout()
        self.microglia_prev = QPushButton("<")
        self.microglia_prev.setFixedWidth(28)
        self.microglia_prev.clicked.connect(self._on_microglia_prev)
        row.addWidget(self.microglia_prev)

        self.microglia_index = QSpinBox()
        self.microglia_index.setRange(0, 0)
        self.microglia_index.setSpecialValueText("All")
        self.microglia_index.valueChanged.connect(self._on_microglia_setting_changed)
        row.addWidget(self.microglia_index)

        self.microglia_next = QPushButton(">")
        self.microglia_next.setFixedWidth(28)
        self.microglia_next.clicked.connect(self._on_microglia_next)
        row.addWidget(self.microglia_next)
        form.addRow("Cell", row)

        self.microglia_info = QLabel("Load a dataset to detect microglia components.")
        form.addRow(self.microglia_info)

        self.microglia_branch_sensitivity = self._make_unit_spinbox(0.4, 2.0, 0.05, 1.0)
        self.microglia_branch_sensitivity.valueChanged.connect(self._on_microglia_setting_changed)
        form.addRow("Branch sensitivity", self.microglia_branch_sensitivity)

        self._set_microglia_navigation_enabled(False)
        return group

    def _build_microglia_analysis_group(self) -> QGroupBox:
        group = QGroupBox("Microglia Analysis")
        layout = QVBoxLayout(group)

        controls_row = QHBoxLayout()
        self.microglia_analysis_threshold_mode = QComboBox()
        self.microglia_analysis_threshold_mode.addItem("Adaptive thresholds", "adaptive")
        self.microglia_analysis_threshold_mode.addItem("Use render thresholds", "render")
        controls_row.addWidget(QLabel("Thresholds"))
        controls_row.addWidget(self.microglia_analysis_threshold_mode)
        layout.addLayout(controls_row)

        action_row = QHBoxLayout()
        self.run_microglia_analysis_btn = QPushButton("Run Analysis")
        self.run_microglia_analysis_btn.clicked.connect(self.run_microglia_analysis_requested.emit)
        action_row.addWidget(self.run_microglia_analysis_btn)

        self.export_microglia_analysis_btn = QPushButton("Export Analysis CSV")
        self.export_microglia_analysis_btn.setEnabled(False)
        self.export_microglia_analysis_btn.clicked.connect(
            self.export_microglia_analysis_requested.emit
        )
        action_row.addWidget(self.export_microglia_analysis_btn)
        layout.addLayout(action_row)

        self.microglia_analysis_table = QTableWidget(0, len(MICROGLIA_CELL_REPORT_COLUMNS))
        self.microglia_analysis_table.setHorizontalHeaderLabels(MICROGLIA_CELL_REPORT_COLUMNS)
        self.microglia_analysis_table.setAlternatingRowColors(True)
        self.microglia_analysis_table.setMinimumHeight(140)
        layout.addWidget(self.microglia_analysis_table)
        return group

    @staticmethod
    def _make_unit_spinbox(minimum: float, maximum: float, step: float, value: float):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(3)
        spin.setValue(value)
        spin.setKeyboardTracking(False)  # Only emit valueChanged when editing finishes
        return spin

    @staticmethod
    def _make_offset_spinbox():
        spin = QDoubleSpinBox()
        spin.setRange(-1000.0, 1000.0)
        spin.setSingleStep(0.1)
        spin.setDecimals(3)
        spin.setValue(0.0)
        spin.setKeyboardTracking(False)  # Only emit valueChanged when editing finishes
        return spin

    def _emit_render_config(self) -> None:
        self.render_config_changed.emit(self.current_render_config())
        self._pending_render_update = False
        self._update_pending_label()

    def _emit_psf_config(self) -> None:
        self.psf_config_changed.emit(self.current_psf_config())

    def _emit_microglia_view_change(self) -> None:
        self.microglia_view_changed.emit()
        self._pending_microglia_update = False
        self._update_pending_label()

    def _on_render_setting_changed(self) -> None:
        """Handle any render setting change with debouncing or manual apply."""
        if self._auto_apply_enabled:
            self._pending_render_update = True
            self._update_pending_label()
            # Restart timer on each change (debounce)
            self._render_update_timer.start()
        else:
            self._pending_render_update = True
            self._update_pending_label()

    def _on_microglia_setting_changed(self) -> None:
        """Handle microglia setting change with debouncing or manual apply."""
        if self._auto_apply_enabled:
            self._pending_microglia_update = True
            self._update_pending_label()
            # Restart timer on each change (debounce)
            self._microglia_update_timer.start()
        else:
            self._pending_microglia_update = True
            self._update_pending_label()

    def _emit_render_config_delayed(self) -> None:
        """Called by timer after debounce delay."""
        if self._pending_render_update:
            self._emit_render_config()

    def _emit_microglia_view_change_delayed(self) -> None:
        """Called by timer after debounce delay."""
        if self._pending_microglia_update:
            self._emit_microglia_view_change()

    def _on_auto_apply_toggled(self, checked: bool) -> None:
        """Handle auto-apply checkbox toggle."""
        self._auto_apply_enabled = bool(checked)
        self.apply_btn.setVisible(not checked)
        if checked:
            # When re-enabling auto-apply, immediately apply any pending changes
            if self._pending_render_update or self._pending_microglia_update:
                self._on_apply_clicked()
        self._update_pending_label()

    def _on_apply_clicked(self) -> None:
        """Handle manual Apply button click."""
        # Stop any running timers
        self._render_update_timer.stop()
        self._microglia_update_timer.stop()

        # Apply pending changes immediately
        if self._pending_render_update:
            self._emit_render_config()
        if self._pending_microglia_update:
            self._emit_microglia_view_change()

    def _update_pending_label(self) -> None:
        """Update the pending changes label."""
        if not self._auto_apply_enabled and (self._pending_render_update or self._pending_microglia_update):
            changes = []
            if self._pending_render_update:
                changes.append("rendering")
            if self._pending_microglia_update:
                changes.append("microglia")
            self.pending_label.setText(f"Pending: {', '.join(changes)}")
            self.pending_label.setVisible(True)
            self.apply_btn.setStyleSheet("background-color: #f0c040; color: #111318;")
        else:
            self.pending_label.setVisible(False)
            self.apply_btn.setStyleSheet("")

    def _on_microglia_prev(self) -> None:
        current = int(self.microglia_index.value())
        if current > 0:
            self.microglia_index.setValue(current - 1)

    def _on_microglia_next(self) -> None:
        current = int(self.microglia_index.value())
        if current < int(self.microglia_index.maximum()):
            self.microglia_index.setValue(current + 1)

    def _set_microglia_navigation_enabled(self, enabled: bool) -> None:
        self.microglia_prev.setEnabled(enabled)
        self.microglia_next.setEnabled(enabled)
        self.microglia_index.setEnabled(enabled)

    def _set_advanced_visible(self, visible: bool) -> None:
        self.preprocess_group.setVisible(visible)
        self.debug_group.setVisible(visible)

    def current_render_config(self) -> RenderConfig:
        return RenderConfig(
            threshold_green=float(self.threshold_green.value()),
            threshold_red=float(self.threshold_red.value()),
            opacity_green=float(self.opacity_green.value()),
            opacity_red=float(self.opacity_red.value()),
            iso_green=float(self.iso_green.value()),
            iso_red=float(self.iso_red.value()),
            display_z_scale=float(self.display_z_scale.value()),
            trim_first_slices=int(self.trim_first_slices.value()),
            trim_last_slices=int(self.trim_last_slices.value()),
            offset_x_um=float(self.offset_x.value()),
            offset_y_um=float(self.offset_y.value()),
            offset_z_um=float(self.offset_z.value()),
            show_green=self.show_green.isChecked(),
            show_red=self.show_red.isChecked(),
            show_iso_green=self.show_iso_green.isChecked(),
            show_iso_red=self.show_iso_red.isChecked(),
        )

    def current_psf_config(self) -> PSFConfig:
        return PSFConfig(enabled=False, iterations=0, tv_regularization=False)

    def current_preprocess_config(self) -> PreprocessConfig:
        return PreprocessConfig(
            enabled=self.preprocess_enabled.isChecked(),
            denoise_strength=float(self.noise_strength.value()),
            green_denoise_multiplier=float(self.noise_multiplier.value()),
            green_branch_protection=float(self.branch_protection.value()),
            green_speckle_attenuation=float(self.speckle_attenuation.value()),
        )

    def current_mesh_config(self) -> MeshExportConfig:
        return MeshExportConfig()

    def current_preview_z_index(self) -> int:
        return 0

    def set_threshold_defaults(self, green: float, red: float) -> None:
        self.threshold_green.blockSignals(True)
        self.threshold_red.blockSignals(True)
        self.iso_green.blockSignals(True)
        self.iso_red.blockSignals(True)
        self.threshold_green.setValue(green)
        self.threshold_red.setValue(red)
        self.iso_green.setValue(min(1.0, max(0.0, green + 0.02)))
        self.iso_red.setValue(min(1.0, max(0.0, red + 0.05)))
        self.threshold_green.blockSignals(False)
        self.threshold_red.blockSignals(False)
        self.iso_green.blockSignals(False)
        self.iso_red.blockSignals(False)
        self._emit_render_config()

    def set_metrics_text(self, text: str) -> None:
        self.metrics_text.setPlainText(text)

    def microglia_view_state(self) -> tuple[bool, int]:
        return self.microglia_isolate.isChecked(), int(self.microglia_index.value())

    def current_microglia_branch_sensitivity(self) -> float:
        return float(self.microglia_branch_sensitivity.value())

    def current_microglia_analysis_threshold_mode(self) -> str:
        return str(self.microglia_analysis_threshold_mode.currentData())

    def set_microglia_component_summary(
        self,
        count: int,
        selected_index: int = 0,
        selected_voxels: int = 0,
    ) -> None:
        total = max(0, int(count))
        selected = int(np.clip(selected_index, 0, total)) if total > 0 else 0

        self.microglia_index.blockSignals(True)
        self.microglia_index.setRange(0, total)
        self.microglia_index.setValue(selected)
        self.microglia_index.blockSignals(False)
        self._set_microglia_navigation_enabled(total > 0)

        if total <= 0:
            self.microglia_info.setText("No microglia component found above threshold.")
            return
        if selected <= 0:
            self.microglia_info.setText(f"{total} components detected. Showing all.")
            return
        self.microglia_info.setText(f"Component {selected}/{total} - voxels={int(selected_voxels)}")

    def set_plugin_text(self, text: str) -> None:
        self.plugin_text.setPlainText(text)

    def set_microglia_analysis_report(self, report: MicrogliaCellReport) -> None:
        self.microglia_analysis_table.setRowCount(0)
        self.microglia_analysis_table.setRowCount(len(report.rows))
        for row_idx, row in enumerate(report.rows):
            values = {
                "cell_index": row.cell_index,
                "component_id": row.component_id,
                "segmentation_engine_used": row.segmentation_engine_used,
                "voxel_count": row.voxel_count,
                "volume_um3": row.volume_um3,
                "branch_endpoint_count": row.branch_endpoint_count,
                "branch_junction_count": row.branch_junction_count,
                "distance_to_vasculature_um": row.distance_to_vasculature_um,
                "microglia_closest_x_um": row.microglia_closest_x_um,
                "microglia_closest_y_um": row.microglia_closest_y_um,
                "microglia_closest_z_um": row.microglia_closest_z_um,
                "vessel_closest_x_um": row.vessel_closest_x_um,
                "vessel_closest_y_um": row.vessel_closest_y_um,
                "vessel_closest_z_um": row.vessel_closest_z_um,
                "threshold_green_used": row.threshold_green_used,
                "threshold_red_used": row.threshold_red_used,
            }
            for col_idx, key in enumerate(MICROGLIA_CELL_REPORT_COLUMNS):
                self.microglia_analysis_table.setItem(
                    row_idx,
                    col_idx,
                    QTableWidgetItem(str(values[key])),
                )
        self.export_microglia_analysis_btn.setEnabled(len(report.rows) > 0)

    def clear_microglia_analysis_table(self) -> None:
        self.microglia_analysis_table.setRowCount(0)
        self.export_microglia_analysis_btn.setEnabled(False)

    def append_debug_text(self, text: str) -> None:
        self.debug_text.append(text)

    def clear_debug_text(self) -> None:
        self.debug_text.clear()

    def force_apply_pending_changes(self) -> None:
        """Force immediate application of any pending changes.

        This is useful when loading a new dataset or performing operations
        that require the latest settings to be applied.
        """
        self._render_update_timer.stop()
        self._microglia_update_timer.stop()

        # Clear pending flags without emitting signals
        # (the caller will handle the update)
        self._pending_render_update = False
        self._pending_microglia_update = False
        self._update_pending_label()
