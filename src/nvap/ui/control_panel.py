"""NVAP control panel — collapsible-section inspector.

Replaces the flat QTabWidget layout with a scrollable accordion of named
sections.  All public widget attributes and signals are preserved so
main_window.py and existing tests require zero changes.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from nvap.analysis.microglia_vessel_report import (
    MICROGLIA_CELL_REPORT_COLUMNS,
    MicrogliaCellReport,
)
from nvap.config.types import MeshExportConfig, PSFConfig, PreprocessConfig, RenderConfig
from nvap.ui.design import COLOR, ICON_MD, ICON_SM, SPACE
from nvap.ui.icons import icon, icon_pixmap


# ─── Accordion primitives ──────────────────────────────────────────────────

class _SectionHeader(QFrame):
    """Clickable QFrame that emits clicked() on left-mouse press."""

    clicked = Signal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class _CollapsibleSection(QWidget):
    """A titled, collapsible inspector section with icon + chevron header."""

    def __init__(
        self,
        title: str,
        icon_name: str = "",
        expanded: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._expanded = expanded

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── header ──────────────────────────────────────────────────────
        self._hdr = _SectionHeader(self)
        self._hdr.setObjectName("sectionHeader")
        self._hdr.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hdr.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        hrow = QHBoxLayout(self._hdr)
        hrow.setContentsMargins(10, 7, 10, 7)
        hrow.setSpacing(7)

        if icon_name:
            self._icon_lbl = QLabel(self._hdr)
            self._icon_lbl.setPixmap(icon_pixmap(icon_name, 12, COLOR.accent))
            self._icon_lbl.setFixedSize(14, 14)
            self._icon_lbl.setScaledContents(True)
            hrow.addWidget(self._icon_lbl)

        self._title_lbl = QLabel(title.upper(), self._hdr)
        self._title_lbl.setObjectName("sectionHeaderTitle")
        hrow.addWidget(self._title_lbl, 1)

        self._chevron = QLabel(self._hdr)
        self._chevron.setFixedSize(14, 14)
        self._chevron.setScaledContents(True)
        hrow.addWidget(self._chevron)
        self._refresh_chevron()

        root.addWidget(self._hdr)

        # ── body ────────────────────────────────────────────────────────
        self._body = QFrame(self)
        self._body.setObjectName("sectionBody")
        self._body_lo = QVBoxLayout(self._body)
        self._body_lo.setContentsMargins(10, 8, 8, 10)
        self._body_lo.setSpacing(6)
        self._body.setVisible(expanded)
        root.addWidget(self._body)

        self._hdr.clicked.connect(self._toggle)

    # ── public helpers ─────────────────────────────────────────────────
    def add_widget(self, w: QWidget) -> None:
        self._body_lo.addWidget(w)

    def add_layout(self, lo) -> None:
        self._body_lo.addLayout(lo)

    def add_spacing(self, n: int) -> None:
        self._body_lo.addSpacing(n)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._body.setVisible(expanded)
        self._refresh_chevron()

    def is_expanded(self) -> bool:
        return self._expanded

    @property
    def body_layout(self) -> QVBoxLayout:
        return self._body_lo

    # ── private ────────────────────────────────────────────────────────
    def _toggle(self) -> None:
        self.set_expanded(not self._expanded)

    def _refresh_chevron(self) -> None:
        name = "chevron-up" if self._expanded else "chevron-down"
        self._chevron.setPixmap(icon_pixmap(name, 12, COLOR.text_tertiary))


# ─── Main ControlPanel ─────────────────────────────────────────────────────

class ControlPanel(QWidget):
    """Vertical accordion inspector panel — left half of the workspace view."""

    # ── signals (unchanged) ────────────────────────────────────────────
    load_requested                    = Signal()
    apply_psf_requested               = Signal()
    render_config_changed             = Signal(object)
    psf_config_changed                = Signal(object)
    microglia_view_changed            = Signal()
    enhance_microglia_requested       = Signal()
    run_microglia_analysis_requested  = Signal()
    export_microglia_analysis_requested = Signal()
    microglia_analysis_overlay_changed  = Signal()
    export_metrics_requested          = Signal()
    export_snapshot_requested         = Signal()
    export_mesh_requested             = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Debounce timers
        self._render_update_timer = QTimer(self)
        self._render_update_timer.setSingleShot(True)
        self._render_update_timer.setInterval(250)
        self._render_update_timer.timeout.connect(self._emit_render_config_delayed)

        self._microglia_update_timer = QTimer(self)
        self._microglia_update_timer.setSingleShot(True)
        self._microglia_update_timer.setInterval(400)
        self._microglia_update_timer.timeout.connect(self._emit_microglia_view_change_delayed)

        self._auto_apply_enabled      = True
        self._pending_render_update   = False
        self._pending_microglia_update = False

        # ── Root layout ──────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_fixed_header())

        sep = QFrame(self)
        sep.setObjectName("inspectorSeparator")
        sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        # ── Scrollable section stack ─────────────────────────────────────
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        sections_w = QWidget()
        sections_w.setObjectName("inspectorSections")
        slo = QVBoxLayout(sections_w)
        slo.setContentsMargins(8, 8, 8, 16)
        slo.setSpacing(4)

        self._build_rendering_section(slo)
        self._build_microglia_viewer_section(slo)
        self._build_analysis_section(slo)
        self._processing_section = self._build_processing_section(slo)
        self._build_metrics_export_section(slo)
        self._build_system_section(slo)

        slo.addStretch(1)
        scroll.setWidget(sections_w)
        root.addWidget(scroll, 1)

        # ── Init ────────────────────────────────────────────────────────
        self._set_microglia_navigation_enabled(False)
        self._emit_render_config()
        self._emit_psf_config()

    # ══════════════════════════════════════════════════════════════════════════
    # Fixed header (title, load button, update-mode bar)
    # ══════════════════════════════════════════════════════════════════════════

    def _build_fixed_header(self) -> QWidget:
        w = QWidget(self)
        w.setObjectName("inspectorHeader")
        lo = QVBoxLayout(w)
        lo.setContentsMargins(14, 14, 14, 10)
        lo.setSpacing(6)

        title = QLabel("NVAP Workbench")
        title.setObjectName("panelTitle")
        lo.addWidget(title)

        subtitle = QLabel("Volumetric microglia and vascular analysis")
        subtitle.setObjectName("panelSubtitle")
        lo.addWidget(subtitle)

        # Open dataset
        load_btn = QPushButton("  Open Dataset")
        load_btn.setObjectName("primaryAction")
        load_btn.setIcon(icon("folder-open", ICON_MD, COLOR.text_inverse))
        load_btn.setToolTip("Load a dataset from disk (Ctrl+L)")
        load_btn.clicked.connect(self.load_requested.emit)
        lo.addWidget(load_btn)

        # Auto-apply row
        update_row = QHBoxLayout()
        update_row.setSpacing(8)

        self.auto_apply_checkbox = QCheckBox("Auto-apply changes")
        self.auto_apply_checkbox.setChecked(True)
        self.auto_apply_checkbox.setToolTip(
            "When enabled, changes update the view automatically after a brief delay.\n"
            "When disabled, click Apply to update the view. (Ctrl+A)"
        )
        self.auto_apply_checkbox.toggled.connect(self._on_auto_apply_toggled)
        update_row.addWidget(self.auto_apply_checkbox, 1)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setObjectName("primaryAction")
        self.apply_btn.setToolTip("Apply pending changes (F5 or Return)")
        self.apply_btn.setVisible(False)
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        update_row.addWidget(self.apply_btn)
        lo.addLayout(update_row)

        self.pending_label = QLabel("")
        self.pending_label.setObjectName("pendingWarning")
        self.pending_label.setVisible(False)
        lo.addWidget(self.pending_label)

        # show_advanced — hidden QCheckBox kept for API / test compat.
        # Toggling it expands / collapses the Processing section.
        self.show_advanced = QCheckBox("Show advanced controls")
        self.show_advanced.setChecked(False)
        self.show_advanced.setVisible(False)
        self.show_advanced.toggled.connect(self._on_show_advanced_toggled)
        lo.addWidget(self.show_advanced)

        # mode hint
        hint = QLabel("Mode: green channel pass-through (no masking / denoising)")
        hint.setObjectName("modeHint")
        lo.addWidget(hint)

        return w

    # ══════════════════════════════════════════════════════════════════════════
    # Section builders
    # ══════════════════════════════════════════════════════════════════════════

    def _build_rendering_section(self, parent_lo: QVBoxLayout) -> _CollapsibleSection:
        sec = _CollapsibleSection("Channels & Rendering", "sliders", expanded=True, parent=self)
        parent_lo.addWidget(sec)

        form = QFormLayout()
        form.setSpacing(6)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # Channel visibility
        vis_row = QHBoxLayout()
        vis_row.setSpacing(12)
        self.show_green = QCheckBox("Green")
        self.show_green.setChecked(True)
        self.show_green.setObjectName("channelGreen")
        self.show_green.stateChanged.connect(self._on_render_setting_changed)
        self.show_red = QCheckBox("Red")
        self.show_red.setChecked(True)
        self.show_red.setObjectName("channelRed")
        self.show_red.stateChanged.connect(self._on_render_setting_changed)
        vis_row.addWidget(self.show_green)
        vis_row.addWidget(self.show_red)
        vis_row.addStretch(1)
        form.addRow("Channels", vis_row)

        # Thresholds
        self.threshold_green = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.15)
        self.threshold_green.setToolTip("Min intensity to render green channel (microglia)")
        self.threshold_green.valueChanged.connect(self._on_render_setting_changed)
        _g = QLabel("Threshold G")
        _g.setObjectName("channelGreen")
        form.addRow(_g, self.threshold_green)

        self.threshold_red = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.15)
        self.threshold_red.setToolTip("Min intensity to render red channel (vasculature)")
        self.threshold_red.valueChanged.connect(self._on_render_setting_changed)
        _r = QLabel("Threshold R")
        _r.setObjectName("channelRed")
        form.addRow(_r, self.threshold_red)

        # Opacity
        self.opacity_green = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.40)
        self.opacity_green.valueChanged.connect(self._on_render_setting_changed)
        _og = QLabel("Opacity G")
        _og.setObjectName("channelGreen")
        form.addRow(_og, self.opacity_green)

        self.opacity_red = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.40)
        self.opacity_red.valueChanged.connect(self._on_render_setting_changed)
        _or = QLabel("Opacity R")
        _or.setObjectName("channelRed")
        form.addRow(_or, self.opacity_red)

        # Isosurfaces
        iso_row = QHBoxLayout()
        iso_row.setSpacing(12)
        self.show_iso_green = QCheckBox("Green")
        self.show_iso_green.setObjectName("channelGreen")
        self.show_iso_green.stateChanged.connect(self._on_render_setting_changed)
        self.show_iso_red = QCheckBox("Red")
        self.show_iso_red.setObjectName("channelRed")
        self.show_iso_red.stateChanged.connect(self._on_render_setting_changed)
        iso_row.addWidget(self.show_iso_green)
        iso_row.addWidget(self.show_iso_red)
        iso_row.addStretch(1)
        form.addRow("Isosurfaces", iso_row)

        self.iso_green = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.20)
        self.iso_green.valueChanged.connect(self._on_render_setting_changed)
        _ig = QLabel("Iso level G")
        _ig.setObjectName("channelGreen")
        form.addRow(_ig, self.iso_green)

        self.iso_red = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.20)
        self.iso_red.valueChanged.connect(self._on_render_setting_changed)
        _ir = QLabel("Iso level R")
        _ir.setObjectName("channelRed")
        form.addRow(_ir, self.iso_red)

        # Z / trim / offset
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
        form.addRow("Offset X μm", self.offset_x)

        self.offset_y = self._make_offset_spinbox()
        self.offset_y.valueChanged.connect(self._on_render_setting_changed)
        form.addRow("Offset Y μm", self.offset_y)

        self.offset_z = self._make_offset_spinbox()
        self.offset_z.valueChanged.connect(self._on_render_setting_changed)
        form.addRow("Offset Z μm", self.offset_z)

        sec.add_layout(form)
        return sec

    def _build_microglia_viewer_section(
        self, parent_lo: QVBoxLayout
    ) -> _CollapsibleSection:
        sec = _CollapsibleSection("Microglia Viewer", "eye", expanded=True, parent=self)
        parent_lo.addWidget(sec)

        form = QFormLayout()
        form.setSpacing(6)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.microglia_isolate = QCheckBox("View one microglia")
        self.microglia_isolate.toggled.connect(self._on_microglia_setting_changed)
        form.addRow(self.microglia_isolate)

        # Cell navigation
        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)

        self.microglia_prev = QPushButton()
        self.microglia_prev.setIcon(icon("chevron-left", ICON_SM, COLOR.text_secondary))
        self.microglia_prev.setFixedSize(28, 28)
        self.microglia_prev.setToolTip("Previous microglia")
        self.microglia_prev.clicked.connect(self._on_microglia_prev)
        nav_row.addWidget(self.microglia_prev)

        self.microglia_index = QSpinBox()
        self.microglia_index.setRange(0, 0)
        self.microglia_index.setSpecialValueText("All")
        self.microglia_index.valueChanged.connect(self._on_microglia_setting_changed)
        nav_row.addWidget(self.microglia_index, 1)

        self.microglia_next = QPushButton()
        self.microglia_next.setIcon(icon("chevron-right", ICON_SM, COLOR.text_secondary))
        self.microglia_next.setFixedSize(28, 28)
        self.microglia_next.setToolTip("Next microglia")
        self.microglia_next.clicked.connect(self._on_microglia_next)
        nav_row.addWidget(self.microglia_next)

        form.addRow("Cell", nav_row)

        self.microglia_info = QLabel("Load a dataset to detect microglia components.")
        self.microglia_info.setWordWrap(True)
        form.addRow(self.microglia_info)

        self.microglia_branch_sensitivity = self._make_unit_spinbox(0.4, 2.0, 0.05, 1.0)
        self.microglia_branch_sensitivity.valueChanged.connect(
            self._on_microglia_setting_changed
        )
        form.addRow("Branch sensitivity", self.microglia_branch_sensitivity)

        self.microglia_enhancement_method = QComboBox()
        self.microglia_enhancement_method.addItem(
            "Microglia-preserving combined", "microglia_preserve"
        )
        self.microglia_enhancement_method.addItem(
            "Microscopy clean soma/branch", "microscopy_clean"
        )
        self.microglia_enhancement_method.addItem(
            "ImageJ/Fiji rolling ball", "imagej_rolling_ball"
        )
        self.microglia_enhancement_method.addItem("BaSiC-style correction", "basic")
        self.microglia_enhancement_method.addItem("CIDRE-style correction", "cidre")
        self.microglia_enhancement_method.addItem(
            "scikit-image white top-hat", "white_tophat"
        )
        self.microglia_enhancement_method.addItem("scikit-image CLAHE", "clahe")
        form.addRow("Enhancement", self.microglia_enhancement_method)

        sec.add_layout(form)

        self.enhance_microglia_btn = QPushButton("  Enhance Microglia")
        self.enhance_microglia_btn.setIcon(icon("sparkles", ICON_SM, COLOR.accent))
        self.enhance_microglia_btn.setEnabled(False)
        self.enhance_microglia_btn.setToolTip(
            "Remove green-channel background after loading while preserving somas and branches."
        )
        self.enhance_microglia_btn.clicked.connect(self.enhance_microglia_requested.emit)
        sec.add_widget(self.enhance_microglia_btn)

        return sec

    def _build_analysis_section(self, parent_lo: QVBoxLayout) -> _CollapsibleSection:
        sec = _CollapsibleSection("Analysis", "activity", expanded=True, parent=self)
        parent_lo.addWidget(sec)

        # Threshold mode row
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        mode_row.addWidget(QLabel("Thresholds"))
        self.microglia_analysis_threshold_mode = QComboBox()
        self.microglia_analysis_threshold_mode.addItem("Adaptive thresholds", "adaptive")
        self.microglia_analysis_threshold_mode.addItem(
            "Use render thresholds", "render"
        )
        mode_row.addWidget(self.microglia_analysis_threshold_mode, 1)
        sec.add_layout(mode_row)

        # Action row
        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.run_microglia_analysis_btn = QPushButton("  Run Analysis")
        self.run_microglia_analysis_btn.setObjectName("primaryAction")
        self.run_microglia_analysis_btn.setIcon(icon("zap", ICON_SM, COLOR.text_inverse))
        self.run_microglia_analysis_btn.clicked.connect(
            self.run_microglia_analysis_requested.emit
        )
        action_row.addWidget(self.run_microglia_analysis_btn)

        self.export_microglia_analysis_btn = QPushButton("  Export Bundle")
        self.export_microglia_analysis_btn.setIcon(
            icon("download", ICON_SM, COLOR.text_secondary)
        )
        self.export_microglia_analysis_btn.setEnabled(False)
        self.export_microglia_analysis_btn.setToolTip(
            "Export cell, branch, tip, and vessel-crossing CSVs plus debug outputs."
        )
        self.export_microglia_analysis_btn.clicked.connect(
            self.export_microglia_analysis_requested.emit
        )
        action_row.addWidget(self.export_microglia_analysis_btn)
        sec.add_layout(action_row)

        # Debug overlay toggles — inside a compact sub-section
        overlay_lbl = QLabel("Visual Debuggers")
        overlay_lbl.setObjectName("sectionHeaderTitle")
        sec.add_widget(overlay_lbl)

        self.debug_overlay_soma        = QCheckBox("Soma cores")
        self.debug_overlay_branches    = QCheckBox("Branch skeletons")
        self.debug_overlay_tips        = QCheckBox("Branch tips")
        self.debug_overlay_connectors  = QCheckBox("Nearest-vessel connectors")
        self.debug_overlay_vessels     = QCheckBox("Vessel points")
        self.debug_overlay_diameter    = QCheckBox("Diameter samples")
        self.debug_overlay_crossings   = QCheckBox("Vessel crossings")

        overlay_grid = QHBoxLayout()
        col_a = QVBoxLayout()
        col_b = QVBoxLayout()
        overlays = [
            self.debug_overlay_soma,
            self.debug_overlay_branches,
            self.debug_overlay_tips,
            self.debug_overlay_connectors,
        ]
        overlays_b = [
            self.debug_overlay_vessels,
            self.debug_overlay_diameter,
            self.debug_overlay_crossings,
        ]
        for cb in overlays:
            cb.setChecked(True)
            cb.toggled.connect(lambda _: self.microglia_analysis_overlay_changed.emit())
            col_a.addWidget(cb)
        for cb in overlays_b:
            cb.setChecked(True)
            cb.toggled.connect(lambda _: self.microglia_analysis_overlay_changed.emit())
            col_b.addWidget(cb)

        overlay_grid.addLayout(col_a)
        overlay_grid.addLayout(col_b)
        sec.add_layout(overlay_grid)

        # Results table
        self.microglia_analysis_table = QTableWidget(
            0, len(MICROGLIA_CELL_REPORT_COLUMNS)
        )
        self.microglia_analysis_table.setHorizontalHeaderLabels(
            MICROGLIA_CELL_REPORT_COLUMNS
        )
        self.microglia_analysis_table.setAlternatingRowColors(True)
        self.microglia_analysis_table.setMinimumHeight(140)
        sec.add_widget(self.microglia_analysis_table)

        return sec

    def _build_processing_section(self, parent_lo: QVBoxLayout) -> _CollapsibleSection:
        """Processing section — collapsed by default; show_advanced expands it."""
        sec = _CollapsibleSection("Processing", "cpu", expanded=False, parent=self)
        parent_lo.addWidget(sec)

        # Keep a reference as preprocess_group for backward compat
        self.preprocess_group = sec

        form = QFormLayout()
        form.setSpacing(6)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

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

        note = QLabel(
            "Single mode: green pass-through.\n"
            "Green is pass-through (input used as-is).\n"
            "Red keeps the existing workflow."
        )
        note.setWordWrap(True)
        form.addRow(note)

        sec.add_layout(form)

        self.reprocess_btn = QPushButton("  Reprocess Dataset")
        self.reprocess_btn.setIcon(icon("rotate-ccw", ICON_SM, COLOR.text_secondary))
        self.reprocess_btn.setToolTip("Run the preprocessing / processing pipeline again.")
        self.reprocess_btn.clicked.connect(self.apply_psf_requested.emit)
        sec.add_widget(self.reprocess_btn)

        return sec

    def _build_metrics_export_section(
        self, parent_lo: QVBoxLayout
    ) -> _CollapsibleSection:
        sec = _CollapsibleSection("Metrics & Export", "download", expanded=True, parent=self)
        parent_lo.addWidget(sec)

        # Metrics readout
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(120)
        self.metrics_text.setPlaceholderText("Load a dataset to compute metrics.")
        sec.add_widget(self.metrics_text)

        # Export buttons row
        exp_row = QHBoxLayout()
        exp_row.setSpacing(6)

        csv_btn = QPushButton("  CSV")
        csv_btn.setIcon(icon("activity", ICON_SM, COLOR.text_secondary))
        csv_btn.setToolTip("Export metrics to CSV file (Ctrl+E)")
        csv_btn.clicked.connect(self.export_metrics_requested.emit)
        exp_row.addWidget(csv_btn)

        png_btn = QPushButton("  Snapshot")
        png_btn.setIcon(icon("camera", ICON_SM, COLOR.text_secondary))
        png_btn.setToolTip("Export current 3D view as PNG (Ctrl+S)")
        png_btn.clicked.connect(self.export_snapshot_requested.emit)
        exp_row.addWidget(png_btn)

        mesh_btn = QPushButton("  3D Mesh")
        mesh_btn.setObjectName("primaryAction")
        mesh_btn.setIcon(icon("box", ICON_SM, COLOR.text_inverse))
        mesh_btn.setToolTip("Export 3D mesh files (Ctrl+M)")
        mesh_btn.clicked.connect(self.export_mesh_requested.emit)
        exp_row.addWidget(mesh_btn)

        sec.add_layout(exp_row)
        return sec

    def _build_system_section(self, parent_lo: QVBoxLayout) -> _CollapsibleSection:
        sec = _CollapsibleSection("System & Log", "settings", expanded=False, parent=self)
        parent_lo.addWidget(sec)

        # Plugins
        plugin_lbl = QLabel("Plugins")
        plugin_lbl.setObjectName("sectionHeaderTitle")
        sec.add_widget(plugin_lbl)

        self.plugin_text = QTextEdit()
        self.plugin_text.setReadOnly(True)
        self.plugin_text.setMaximumHeight(72)
        sec.add_widget(self.plugin_text)

        # Debug log — keep debug_group as a reference to this section for compat
        self.debug_group = sec
        log_lbl = QLabel("Debug Log")
        log_lbl.setObjectName("sectionHeaderTitle")
        sec.add_widget(log_lbl)

        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        self.debug_text.setMaximumHeight(110)
        sec.add_widget(self.debug_text)

        return sec

    # ══════════════════════════════════════════════════════════════════════════
    # Widget factories
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _make_unit_spinbox(
        minimum: float, maximum: float, step: float, value: float
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(3)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        return spin

    @staticmethod
    def _make_offset_spinbox() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1000.0, 1000.0)
        spin.setSingleStep(0.1)
        spin.setDecimals(3)
        spin.setValue(0.0)
        spin.setKeyboardTracking(False)
        return spin

    # ══════════════════════════════════════════════════════════════════════════
    # Debounce / emit helpers
    # ══════════════════════════════════════════════════════════════════════════

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
        self._pending_render_update = True
        self._update_pending_label()
        if self._auto_apply_enabled:
            self._render_update_timer.start()

    def _on_microglia_setting_changed(self) -> None:
        self._pending_microglia_update = True
        self._update_pending_label()
        if self._auto_apply_enabled:
            self._microglia_update_timer.start()

    def _emit_render_config_delayed(self) -> None:
        if self._pending_render_update:
            self._emit_render_config()

    def _emit_microglia_view_change_delayed(self) -> None:
        if self._pending_microglia_update:
            self._emit_microglia_view_change()

    def _on_auto_apply_toggled(self, checked: bool) -> None:
        self._auto_apply_enabled = bool(checked)
        self.apply_btn.setVisible(not checked)
        if checked and (self._pending_render_update or self._pending_microglia_update):
            self._on_apply_clicked()
        self._update_pending_label()

    def _on_apply_clicked(self) -> None:
        self._render_update_timer.stop()
        self._microglia_update_timer.stop()
        if self._pending_render_update:
            self._emit_render_config()
        if self._pending_microglia_update:
            self._emit_microglia_view_change()

    def _update_pending_label(self) -> None:
        if not self._auto_apply_enabled and (
            self._pending_render_update or self._pending_microglia_update
        ):
            changes = []
            if self._pending_render_update:
                changes.append("rendering")
            if self._pending_microglia_update:
                changes.append("microglia")
            self.pending_label.setText(f"Pending: {', '.join(changes)}")
            self.pending_label.setVisible(True)
            self.apply_btn.setStyleSheet(
                f"background-color: {COLOR.warning}; color: {COLOR.text_inverse};"
            )
        else:
            self.pending_label.setVisible(False)
            self.apply_btn.setStyleSheet("")

    # ══════════════════════════════════════════════════════════════════════════
    # Microglia navigation
    # ══════════════════════════════════════════════════════════════════════════

    def _on_microglia_prev(self) -> None:
        v = int(self.microglia_index.value())
        if v > 0:
            self.microglia_index.setValue(v - 1)

    def _on_microglia_next(self) -> None:
        v = int(self.microglia_index.value())
        if v < int(self.microglia_index.maximum()):
            self.microglia_index.setValue(v + 1)

    def _set_microglia_navigation_enabled(self, enabled: bool) -> None:
        self.microglia_prev.setEnabled(enabled)
        self.microglia_next.setEnabled(enabled)
        self.microglia_index.setEnabled(enabled)

    # ══════════════════════════════════════════════════════════════════════════
    # Advanced / show_advanced compat shim
    # ══════════════════════════════════════════════════════════════════════════

    def _on_show_advanced_toggled(self, checked: bool) -> None:
        """Expand / collapse the Processing section when show_advanced changes."""
        self._processing_section.set_expanded(checked)

    def _set_advanced_visible(self, visible: bool) -> None:
        """Legacy shim: show_advanced.toggled previously called this."""
        self.show_advanced.setChecked(visible)

    # ══════════════════════════════════════════════════════════════════════════
    # Public read-out API
    # ══════════════════════════════════════════════════════════════════════════

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

    def current_microglia_enhancement_method(self) -> str:
        return str(self.microglia_enhancement_method.currentData())

    def microglia_view_state(self) -> tuple[bool, int]:
        return self.microglia_isolate.isChecked(), int(self.microglia_index.value())

    def current_microglia_branch_sensitivity(self) -> float:
        return float(self.microglia_branch_sensitivity.value())

    def current_microglia_analysis_threshold_mode(self) -> str:
        return str(self.microglia_analysis_threshold_mode.currentData())

    def current_microglia_debug_overlay_state(self) -> dict[str, bool]:
        return {
            "soma":       self.debug_overlay_soma.isChecked(),
            "branches":   self.debug_overlay_branches.isChecked(),
            "tips":       self.debug_overlay_tips.isChecked(),
            "connectors": self.debug_overlay_connectors.isChecked(),
            "vessels":    self.debug_overlay_vessels.isChecked(),
            "diameter":   self.debug_overlay_diameter.isChecked(),
            "crossings":  self.debug_overlay_crossings.isChecked(),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Public update API (called from main_window)
    # ══════════════════════════════════════════════════════════════════════════

    def set_threshold_defaults(self, green: float, red: float) -> None:
        for spin in (
            self.threshold_green,
            self.threshold_red,
            self.iso_green,
            self.iso_red,
        ):
            spin.blockSignals(True)
        self.threshold_green.setValue(green)
        self.threshold_red.setValue(red)
        self.iso_green.setValue(min(1.0, max(0.0, green + 0.02)))
        self.iso_red.setValue(min(1.0, max(0.0, red + 0.05)))
        for spin in (
            self.threshold_green,
            self.threshold_red,
            self.iso_green,
            self.iso_red,
        ):
            spin.blockSignals(False)
        self._emit_render_config()

    def set_metrics_text(self, text: str) -> None:
        self.metrics_text.setPlainText(text)

    def set_microglia_enhancement_enabled(self, enabled: bool) -> None:
        self.enhance_microglia_btn.setEnabled(bool(enabled))

    def set_microglia_component_summary(
        self,
        count: int,
        selected_index: int = 0,
        selected_voxels: int = 0,
    ) -> None:
        total    = max(0, int(count))
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
        self.microglia_info.setText(
            f"Component {selected}/{total} · voxels={int(selected_voxels)}"
        )

    def set_plugin_text(self, text: str) -> None:
        self.plugin_text.setPlainText(text)

    def set_microglia_analysis_report(self, report: MicrogliaCellReport) -> None:
        self.microglia_analysis_table.setRowCount(0)
        self.microglia_analysis_table.setRowCount(len(report.rows))
        for row_idx, row in enumerate(report.rows):
            values = row.__dict__
            for col_idx, key in enumerate(MICROGLIA_CELL_REPORT_COLUMNS):
                self.microglia_analysis_table.setItem(
                    row_idx,
                    col_idx,
                    QTableWidgetItem(str(values.get(key, ""))),
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
        """Force immediate application of any pending changes (e.g. before a load)."""
        self._render_update_timer.stop()
        self._microglia_update_timer.stop()
        self._pending_render_update   = False
        self._pending_microglia_update = False
        self._update_pending_label()
