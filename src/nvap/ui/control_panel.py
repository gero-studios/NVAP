"""Simplified NVAP control panel with green-channel pass-through mode."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from nvap.config.types import MeshExportConfig, PSFConfig, PreprocessConfig, RenderConfig


class ControlPanel(QWidget):
    load_requested = Signal()
    apply_psf_requested = Signal()
    render_config_changed = Signal(object)
    psf_config_changed = Signal(object)
    microglia_view_changed = Signal()
    export_metrics_requested = Signal()
    export_snapshot_requested = Signal()
    export_mesh_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        load_btn = QPushButton("Load Dataset")
        load_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        load_btn.clicked.connect(self.load_requested.emit)
        root.addWidget(load_btn)

        self.show_advanced = QCheckBox("Show advanced controls")
        self.show_advanced.setChecked(False)
        self.show_advanced.toggled.connect(self._set_advanced_visible)
        root.addWidget(self.show_advanced)
        root.addWidget(QLabel("Mode: green pass-through (no green masking/denoise)"))

        self.render_group = self._build_render_group()
        root.addWidget(self.render_group)

        self.preprocess_group = self._build_preprocess_group()
        root.addWidget(self.preprocess_group)

        self.microglia_group = self._build_microglia_group()
        root.addWidget(self.microglia_group)

        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout(export_group)
        csv_btn = QPushButton("Metrics CSV")
        csv_btn.clicked.connect(self.export_metrics_requested.emit)
        png_btn = QPushButton("Snapshot PNG")
        png_btn.clicked.connect(self.export_snapshot_requested.emit)
        mesh_btn = QPushButton("3D Mesh")
        mesh_btn.setStyleSheet("font-weight: bold;")
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
        self.show_green.stateChanged.connect(self._emit_render_config)
        self.show_red = QCheckBox("Red")
        self.show_red.setChecked(True)
        self.show_red.stateChanged.connect(self._emit_render_config)
        row.addWidget(self.show_green)
        row.addWidget(self.show_red)
        form.addRow("Channels", row)

        iso_row = QHBoxLayout()
        self.show_iso_green = QCheckBox("Green")
        self.show_iso_green.stateChanged.connect(self._emit_render_config)
        self.show_iso_red = QCheckBox("Red")
        self.show_iso_red.stateChanged.connect(self._emit_render_config)
        iso_row.addWidget(self.show_iso_green)
        iso_row.addWidget(self.show_iso_red)
        form.addRow("Isosurfaces", iso_row)

        self.threshold_green = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.15)
        self.threshold_green.valueChanged.connect(self._emit_render_config)
        form.addRow("Threshold G", self.threshold_green)

        self.threshold_red = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.15)
        self.threshold_red.valueChanged.connect(self._emit_render_config)
        form.addRow("Threshold R", self.threshold_red)

        self.opacity_green = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.40)
        self.opacity_green.valueChanged.connect(self._emit_render_config)
        form.addRow("Opacity G", self.opacity_green)

        self.opacity_red = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.40)
        self.opacity_red.valueChanged.connect(self._emit_render_config)
        form.addRow("Opacity R", self.opacity_red)

        self.iso_green = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.20)
        self.iso_green.valueChanged.connect(self._emit_render_config)
        form.addRow("Iso level G", self.iso_green)

        self.iso_red = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.20)
        self.iso_red.valueChanged.connect(self._emit_render_config)
        form.addRow("Iso level R", self.iso_red)

        self.display_z_scale = self._make_unit_spinbox(0.2, 3.0, 0.05, 2.0 / 3.0)
        self.display_z_scale.valueChanged.connect(self._emit_render_config)
        form.addRow("Z height scale", self.display_z_scale)

        self.offset_x = self._make_offset_spinbox()
        self.offset_x.valueChanged.connect(self._emit_render_config)
        form.addRow("Offset X um", self.offset_x)

        self.offset_y = self._make_offset_spinbox()
        self.offset_y.valueChanged.connect(self._emit_render_config)
        form.addRow("Offset Y um", self.offset_y)

        self.offset_z = self._make_offset_spinbox()
        self.offset_z.valueChanged.connect(self._emit_render_config)
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
        return group

    def _build_microglia_group(self) -> QGroupBox:
        group = QGroupBox("Microglia Viewer")
        form = QFormLayout(group)

        self.microglia_isolate = QCheckBox("View one microglia")
        self.microglia_isolate.toggled.connect(self._emit_microglia_view_change)
        form.addRow(self.microglia_isolate)

        row = QHBoxLayout()
        self.microglia_prev = QPushButton("<")
        self.microglia_prev.setFixedWidth(28)
        self.microglia_prev.clicked.connect(self._on_microglia_prev)
        row.addWidget(self.microglia_prev)

        self.microglia_index = QSpinBox()
        self.microglia_index.setRange(0, 0)
        self.microglia_index.setSpecialValueText("All")
        self.microglia_index.valueChanged.connect(self._emit_microglia_view_change)
        row.addWidget(self.microglia_index)

        self.microglia_next = QPushButton(">")
        self.microglia_next.setFixedWidth(28)
        self.microglia_next.clicked.connect(self._on_microglia_next)
        row.addWidget(self.microglia_next)
        form.addRow("Cell", row)

        self.microglia_info = QLabel("Load a dataset to detect microglia components.")
        form.addRow(self.microglia_info)

        self.microglia_branch_sensitivity = self._make_unit_spinbox(0.4, 2.0, 0.05, 1.0)
        self.microglia_branch_sensitivity.valueChanged.connect(self._emit_microglia_view_change)
        form.addRow("Branch sensitivity", self.microglia_branch_sensitivity)

        self._set_microglia_navigation_enabled(False)
        return group

    @staticmethod
    def _make_unit_spinbox(minimum: float, maximum: float, step: float, value: float):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(3)
        spin.setValue(value)
        return spin

    @staticmethod
    def _make_offset_spinbox():
        spin = QDoubleSpinBox()
        spin.setRange(-1000.0, 1000.0)
        spin.setSingleStep(0.1)
        spin.setDecimals(3)
        spin.setValue(0.0)
        return spin

    def _emit_render_config(self) -> None:
        self.render_config_changed.emit(self.current_render_config())

    def _emit_psf_config(self) -> None:
        self.psf_config_changed.emit(self.current_psf_config())

    def _emit_microglia_view_change(self) -> None:
        self.microglia_view_changed.emit()

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

    def append_debug_text(self, text: str) -> None:
        self.debug_text.append(text)

    def clear_debug_text(self) -> None:
        self.debug_text.clear()
