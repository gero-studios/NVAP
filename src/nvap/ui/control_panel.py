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
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from nvap.config.types import (
    DEFAULT_SPACING,
    MeshExportConfig,
    PSFConfig,
    PreprocessConfig,
    RenderConfig,
    VoxelSpacing,
)
from nvap.ui.design import COLOR, ICON_MD, ICON_SM
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
    spacing_changed                   = Signal(object)
    microglia_view_changed            = Signal()
    run_microglia_segmentation_requested = Signal()
    run_microglia_analysis_requested  = Signal()
    enhance_microglia_requested       = Signal()
    enhance_vasculature_requested     = Signal()
    auto_thresholds_requested         = Signal()
    wipe_specks_requested             = Signal()
    wipe_vasculature_blobs_requested  = Signal()
    export_metrics_requested          = Signal()
    export_project_analytics_requested = Signal()
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

        self._build_voxel_spacing_section(slo)
        self._build_rendering_section(slo)
        self._build_microglia_viewer_section(slo)
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

    def _build_voxel_spacing_section(self, parent_lo: QVBoxLayout) -> _CollapsibleSection:
        sec = _CollapsibleSection("Voxel Spacing", "ruler", expanded=True, parent=self)
        parent_lo.addWidget(sec)

        help_text = QLabel(
            "Physical pixel/voxel size used by metrics and 3D rendering. "
            "CZI values are read from file metadata and remain editable."
        )
        help_text.setWordWrap(True)
        help_text.setObjectName("sectionHint")
        sec.add_widget(help_text)

        form = QFormLayout()
        form.setSpacing(6)
        self.spacing_x_um = self._make_spacing_spinbox(DEFAULT_SPACING.x_um)
        self.spacing_y_um = self._make_spacing_spinbox(DEFAULT_SPACING.y_um)
        self.spacing_z_um = self._make_spacing_spinbox(DEFAULT_SPACING.z_um)
        form.addRow("X spacing", self.spacing_x_um)
        form.addRow("Y spacing", self.spacing_y_um)
        form.addRow("Z spacing", self.spacing_z_um)
        sec.add_layout(form)

        self.spacing_source_label = QLabel("Manual values (fallback defaults shown)")
        self.spacing_source_label.setWordWrap(True)
        self.spacing_source_label.setObjectName("modeHint")
        sec.add_widget(self.spacing_source_label)
        for spin in (self.spacing_x_um, self.spacing_y_um, self.spacing_z_um):
            spin.valueChanged.connect(self._emit_spacing_changed)
        return sec

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
        self.threshold_green = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.80)
        self.threshold_green.setToolTip("Min intensity to render green channel (microglia)")
        self.threshold_green.valueChanged.connect(self._on_render_setting_changed)
        _g = QLabel("Threshold G")
        _g.setObjectName("channelGreen")
        form.addRow(_g, self.threshold_green)

        self.threshold_red = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.60)
        self.threshold_red.setToolTip("Min intensity to render red channel (vasculature)")
        self.threshold_red.valueChanged.connect(self._on_render_setting_changed)
        _r = QLabel("Threshold R")
        _r.setObjectName("channelRed")
        form.addRow(_r, self.threshold_red)

        self.auto_thresholds_btn = QPushButton("Auto Thresholds")
        self.auto_thresholds_btn.setObjectName("workbenchSecondaryAction")
        self.auto_thresholds_btn.setIcon(icon("sparkles", ICON_SM, COLOR.accent))
        self.auto_thresholds_btn.setEnabled(False)
        self.auto_thresholds_btn.setToolTip(
            "Estimate green and red thresholds from the currently loaded dataset."
        )
        self.auto_thresholds_btn.clicked.connect(self.auto_thresholds_requested.emit)
        form.addRow("Automatic", self.auto_thresholds_btn)

        # Opacity
        self.opacity_green = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.40)
        self.opacity_green.valueChanged.connect(self._on_render_setting_changed)
        _og = QLabel("Opacity G")
        _og.setObjectName("channelGreen")
        form.addRow(_og, self.opacity_green)

        self.opacity_red = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.80)
        self.opacity_red.valueChanged.connect(self._on_render_setting_changed)
        _or = QLabel("Opacity R")
        _or.setObjectName("channelRed")
        form.addRow(_or, self.opacity_red)

        # Isosurfaces
        iso_row = QHBoxLayout()
        iso_row.setSpacing(12)
        self.show_iso_green = QCheckBox("Green")
        self.show_iso_green.setChecked(True)
        self.show_iso_green.setObjectName("channelGreen")
        self.show_iso_green.setToolTip("Render the green channel as a 3D surface.")
        self.show_iso_green.stateChanged.connect(self._on_render_setting_changed)
        self.show_iso_red = QCheckBox("Red")
        self.show_iso_red.setChecked(True)
        self.show_iso_red.setObjectName("channelRed")
        self.show_iso_red.setToolTip("Render the red channel as a 3D surface.")
        self.show_iso_red.stateChanged.connect(self._on_render_setting_changed)
        iso_row.addWidget(self.show_iso_green)
        iso_row.addWidget(self.show_iso_red)
        iso_row.addStretch(1)
        form.addRow("Isosurfaces", iso_row)

        # Iso level is retired from the UI: the Threshold sliders now drive both
        # the 3D surface level and the analysis, so a separate iso control was
        # redundant and confusing. The spinboxes are kept (hidden) so render
        # config / saved projects stay backward compatible.
        self.iso_green = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.20)
        self.iso_red = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.20)
        self.iso_green.hide()
        self.iso_red.hide()

        # Z / trim / offset
        self.display_z_scale = self._make_unit_spinbox(0.2, 3.0, 0.05, 0.70)
        self.display_z_scale.setToolTip("Visual-only Z scale (depth). 1.0 matches physical spacing; higher exaggerates depth. Metrics stay in physical units.")
        self.display_z_scale.valueChanged.connect(self._on_render_setting_changed)
        form.addRow("Z height scale", self.display_z_scale)

        self.trim_first_slices = QSpinBox()
        self.trim_first_slices.setRange(0, 9999)
        self.trim_first_slices.setValue(0)
        self.trim_first_slices.valueChanged.connect(self._on_render_setting_changed)
        form.addRow("Trim first Z", self.trim_first_slices)

        self.trim_last_slices = QSpinBox()
        self.trim_last_slices.setRange(0, 9999)
        self.trim_last_slices.setValue(0)
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
        sec = _CollapsibleSection("Microglia Workbench", "eye", expanded=True, parent=self)
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
        self.microglia_branch_sensitivity.setVisible(False)

        self.microglia_analysis_debug = QCheckBox("Show analysis debug")
        self.microglia_analysis_debug.toggled.connect(self._set_microglia_debug_layers_enabled)
        self.microglia_analysis_debug.toggled.connect(self._on_microglia_setting_changed)
        form.addRow(self.microglia_analysis_debug)

        debug_layers = QWidget()
        debug_layers_layout = QGridLayout(debug_layers)
        debug_layers_layout.setContentsMargins(18, 0, 0, 0)
        debug_layers_layout.setHorizontalSpacing(10)
        debug_layers_layout.setVerticalSpacing(4)

        self.microglia_debug_voxels = QCheckBox("Voxels")
        self.microglia_debug_branches = QCheckBox("Branches")
        self.microglia_debug_soma = QCheckBox("Soma")
        self.microglia_debug_tips = QCheckBox("Tips")
        self.microglia_debug_tip_distance = QCheckBox("Tip distance")
        self.microglia_debug_soma_distance = QCheckBox("Soma distance")
        self.microglia_debug_cell_distance = QCheckBox("Cell distance")

        for checkbox in (
            self.microglia_debug_voxels,
            self.microglia_debug_branches,
            self.microglia_debug_soma,
            self.microglia_debug_tips,
            self.microglia_debug_tip_distance,
            self.microglia_debug_soma_distance,
            self.microglia_debug_cell_distance,
        ):
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._on_microglia_setting_changed)

        debug_layers_layout.addWidget(self.microglia_debug_voxels, 0, 0)
        debug_layers_layout.addWidget(self.microglia_debug_branches, 0, 1)
        debug_layers_layout.addWidget(self.microglia_debug_soma, 1, 0)
        debug_layers_layout.addWidget(self.microglia_debug_tips, 1, 1)
        debug_layers_layout.addWidget(self.microglia_debug_tip_distance, 2, 0)
        debug_layers_layout.addWidget(self.microglia_debug_soma_distance, 2, 1)
        debug_layers_layout.addWidget(self.microglia_debug_cell_distance, 3, 0, 1, 2)
        form.addRow("Layers", debug_layers)
        self._set_microglia_debug_layers_enabled(False)

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
        _clean_idx = self.microglia_enhancement_method.findData("microscopy_clean")
        if _clean_idx >= 0:
            self.microglia_enhancement_method.setCurrentIndex(_clean_idx)
        form.addRow("Enhancement", self.microglia_enhancement_method)

        sec.add_layout(form)

        action_panel = QFrame(self)
        action_panel.setObjectName("workbenchActionPanel")
        action_panel_layout = QVBoxLayout(action_panel)
        action_panel_layout.setContentsMargins(8, 8, 8, 8)
        action_panel_layout.setSpacing(6)

        action_grid = QGridLayout()
        action_grid.setContentsMargins(0, 0, 0, 0)
        action_grid.setHorizontalSpacing(6)
        action_grid.setVerticalSpacing(6)

        self.enhance_microglia_btn = QPushButton("  Enhance Microglia")
        self.enhance_microglia_btn.setObjectName("workbenchPrimaryAction")
        self.enhance_microglia_btn.setIcon(icon("sparkles", ICON_SM, COLOR.accent))
        self.enhance_microglia_btn.setEnabled(False)
        self.enhance_microglia_btn.setToolTip(
            "Remove green-channel background after loading while preserving somas and branches."
        )
        self.enhance_microglia_btn.clicked.connect(self.enhance_microglia_requested.emit)
        action_grid.addWidget(self.enhance_microglia_btn, 0, 0)

        self.enhance_vasculature_btn = QPushButton("  Enhance Vasculature")
        self.enhance_vasculature_btn.setObjectName("workbenchPrimaryAction")
        self.enhance_vasculature_btn.setIcon(icon("sparkles", ICON_SM, COLOR.accent))
        self.enhance_vasculature_btn.setEnabled(False)
        self.enhance_vasculature_btn.setToolTip(
            "Remove uneven red-channel background while preserving vessel-like detail."
        )
        self.enhance_vasculature_btn.clicked.connect(self.enhance_vasculature_requested.emit)
        action_grid.addWidget(self.enhance_vasculature_btn, 0, 1)

        # Speck wipe — remove tiny isolated blobs from both channels.
        wipe_form = QFormLayout()
        wipe_form.setSpacing(6)
        wipe_form.setContentsMargins(0, 0, 0, 0)
        wipe_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        wipe_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.wipe_speck_max_voxels = QSpinBox()
        self.wipe_speck_max_voxels.setRange(1, 1_000_000)
        self.wipe_speck_max_voxels.setValue(128)
        self.wipe_speck_max_voxels.setSuffix(" vox")
        self.wipe_speck_max_voxels.setKeyboardTracking(False)
        self.wipe_speck_max_voxels.setToolTip(
            "Largest isolated blob (in voxels) treated as a speck.\n"
            "Connected components smaller than this are wiped from both channels;\n"
            "larger structures (vessels, somas, branches) are kept."
        )
        wipe_form.addRow("Speck size <", self.wipe_speck_max_voxels)

        self.vascular_blob_max_voxels = QSpinBox()
        # Values below 64 only clear isolated one- or two-pixel noise, which
        # is not useful for the visibly persistent vascular blobs this control
        # is meant to target.
        self.vascular_blob_max_voxels.setRange(64, 1_000_000)
        self.vascular_blob_max_voxels.setSingleStep(64)
        self.vascular_blob_max_voxels.setValue(2_048)
        self.vascular_blob_max_voxels.setSuffix(" vox")
        self.vascular_blob_max_voxels.setKeyboardTracking(False)
        self.vascular_blob_max_voxels.setToolTip(
            "Largest compact red-channel component treated as vascular debris.\n"
            "Elongated vessel segments are retained, even below this size."
        )
        wipe_form.addRow("Remove red blobs up to", self.vascular_blob_max_voxels)
        action_panel_layout.addLayout(wipe_form)

        # "Automatic on load" toggles — read every time a dataset finishes
        # loading (including each dataset in a multi-stack project set). They are
        # always enabled so they can be set before loading anything.
        auto_header = QLabel("Automatic on load")
        auto_header.setObjectName("sectionSubheading")
        auto_header.setToolTip(
            "Steps run automatically each time a dataset loads. Default thresholds "
            "are always applied; the two toggles below control the cleanup passes."
        )
        action_panel_layout.addWidget(auto_header)

        self.auto_enhance_on_load = QCheckBox("Enhance microglia (clean) on load")
        self.auto_enhance_on_load.setChecked(True)
        self.auto_enhance_on_load.setToolTip(
            "Automatically run the selected microglia enhancement (default:\n"
            "Microscopy clean soma/branch) on the green channel right after load.\n"
            "Skipped when the dataset already has a cached enhancement."
        )
        action_panel_layout.addWidget(self.auto_enhance_on_load)

        self.auto_wipe_on_load = QCheckBox("Wipe specks on load")
        self.auto_wipe_on_load.setChecked(True)
        self.auto_wipe_on_load.setToolTip(
            "Automatically remove specks (below the size above) from both channels\n"
            "right after a dataset finishes loading. Uncheck to load the raw stacks\n"
            "and wipe manually with the button below."
        )
        action_panel_layout.addWidget(self.auto_wipe_on_load)

        # "Automatic on edit" toggle — read every time the green/red threshold
        # sliders settle on a new value (debounced), independent of the
        # on-load toggles above.
        auto_edit_header = QLabel("Automatic on edit")
        auto_edit_header.setObjectName("sectionSubheading")
        auto_edit_header.setToolTip(
            "Steps run automatically in response to edits you make while working\n"
            "with an already-loaded dataset."
        )
        action_panel_layout.addWidget(auto_edit_header)

        self.auto_wipe_on_threshold_edit = QCheckBox("Wipe specks when threshold changes")
        self.auto_wipe_on_threshold_edit.setChecked(False)
        self.auto_wipe_on_threshold_edit.setToolTip(
            "Automatically re-run the speck wipe (using the size above) on both\n"
            "channels whenever you change the green or red threshold. Off by\n"
            "default since it re-wipes on every threshold edit, not just on load."
        )
        action_panel_layout.addWidget(self.auto_wipe_on_threshold_edit)

        self.wipe_specks_btn = QPushButton("  Wipe Specks")
        self.wipe_specks_btn.setObjectName("workbenchSecondaryAction")
        self.wipe_specks_btn.setIcon(icon("trash", ICON_SM, COLOR.accent))
        self.wipe_specks_btn.setEnabled(False)
        self.wipe_specks_btn.setToolTip(
            "Remove small isolated specks from both the microglia (green) and "
            "vasculature (red) channels. Larger structures are preserved."
        )
        self.wipe_specks_btn.clicked.connect(self.wipe_specks_requested.emit)
        action_grid.addWidget(self.wipe_specks_btn, 1, 0)

        self.wipe_vasculature_blobs_btn = QPushButton("  Wipe Vascular Blobs")
        self.wipe_vasculature_blobs_btn.setObjectName("workbenchSecondaryAction")
        self.wipe_vasculature_blobs_btn.setIcon(icon("trash", ICON_SM, COLOR.accent))
        self.wipe_vasculature_blobs_btn.setEnabled(False)
        self.wipe_vasculature_blobs_btn.setToolTip(
            "Remove compact red-channel blobs using the dedicated vascular size limit. "
            "Elongated vessel fragments are kept."
        )
        self.wipe_vasculature_blobs_btn.clicked.connect(self.wipe_vasculature_blobs_requested.emit)
        action_grid.addWidget(self.wipe_vasculature_blobs_btn, 1, 1)
        action_panel_layout.addLayout(action_grid)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)

        self.run_microglia_segmentation_btn = QPushButton("Run Segmentation")
        self.run_microglia_segmentation_btn.setObjectName("workbenchSecondaryAction")
        self.run_microglia_segmentation_btn.setEnabled(False)
        self.run_microglia_segmentation_btn.setToolTip(
            "Separate visible microglia into individual components using the current green threshold."
        )
        self.run_microglia_segmentation_btn.clicked.connect(
            self.run_microglia_segmentation_requested.emit
        )
        action_row.addWidget(self.run_microglia_segmentation_btn)

        self.run_microglia_analysis_btn = QPushButton("Run Analysis")
        self.run_microglia_analysis_btn.setObjectName("workbenchPrimaryAction")
        self.run_microglia_analysis_btn.setEnabled(False)
        self.run_microglia_analysis_btn.setToolTip(
            "Analyze the current visible microglia and vasculature view and open Analytics."
        )
        self.run_microglia_analysis_btn.clicked.connect(
            self.run_microglia_analysis_requested.emit
        )
        action_row.addWidget(self.run_microglia_analysis_btn)

        action_panel_layout.addLayout(action_row)
        sec.add_widget(action_panel)

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

        project_btn = QPushButton("  Project CSV")
        project_btn.setIcon(icon("bar-chart", ICON_SM, COLOR.text_secondary))
        project_btn.setToolTip(
            "For project sets: apply this sample's thresholds/cleanup to every sample, "
            "then export individual and cumulative analytics."
        )
        project_btn.clicked.connect(self.export_project_analytics_requested.emit)
        exp_row.addWidget(project_btn)

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
    def _make_spacing_spinbox(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.000001, 10000.0)
        spin.setSingleStep(0.01)
        spin.setDecimals(6)
        spin.setSuffix(" µm")
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        spin.setToolTip("Enter a positive physical spacing in micrometres.")
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

    def _emit_spacing_changed(self) -> None:
        self.spacing_source_label.setText("Manual override")
        self.spacing_changed.emit(self.current_voxel_spacing())

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

    def _set_microglia_debug_layers_enabled(self, enabled: bool) -> None:
        active = bool(enabled)
        for checkbox in (
            self.microglia_debug_voxels,
            self.microglia_debug_branches,
            self.microglia_debug_soma,
            self.microglia_debug_tips,
            self.microglia_debug_tip_distance,
            self.microglia_debug_soma_distance,
            self.microglia_debug_cell_distance,
        ):
            checkbox.setEnabled(active)

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

    def current_voxel_spacing(self) -> VoxelSpacing:
        return VoxelSpacing(
            x_um=float(self.spacing_x_um.value()),
            y_um=float(self.spacing_y_um.value()),
            z_um=float(self.spacing_z_um.value()),
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

    def microglia_analysis_debug_enabled(self) -> bool:
        return bool(self.microglia_analysis_debug.isChecked())

    def microglia_analysis_debug_layers(self) -> set[str]:
        layers: set[str] = set()
        if self.microglia_debug_voxels.isChecked():
            layers.add("voxels")
        if self.microglia_debug_branches.isChecked():
            layers.add("branches")
        if self.microglia_debug_soma.isChecked():
            layers.add("soma")
        if self.microglia_debug_tips.isChecked():
            layers.add("tips")
        if self.microglia_debug_tip_distance.isChecked():
            layers.add("tip_distance")
        if self.microglia_debug_soma_distance.isChecked():
            layers.add("soma_distance")
        if self.microglia_debug_cell_distance.isChecked():
            layers.add("cell_distance")
        return layers

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
        self.enhance_vasculature_btn.setEnabled(bool(enabled))
        self.auto_thresholds_btn.setEnabled(bool(enabled))
        self.wipe_specks_btn.setEnabled(bool(enabled))
        self.wipe_vasculature_blobs_btn.setEnabled(bool(enabled))

    def current_wipe_speck_max_voxels(self) -> int:
        return int(self.wipe_speck_max_voxels.value())

    def current_vascular_blob_max_voxels(self) -> int:
        return int(self.vascular_blob_max_voxels.value())

    def auto_wipe_specks_on_load_enabled(self) -> bool:
        return bool(self.auto_wipe_on_load.isChecked())

    def auto_enhance_microglia_on_load_enabled(self) -> bool:
        return bool(self.auto_enhance_on_load.isChecked())

    def auto_wipe_specks_on_threshold_edit_enabled(self) -> bool:
        return bool(self.auto_wipe_on_threshold_edit.isChecked())

    def set_microglia_workflow_enabled(self, enabled: bool) -> None:
        workflow_enabled = bool(enabled)
        self.run_microglia_segmentation_btn.setEnabled(workflow_enabled)
        self.run_microglia_analysis_btn.setEnabled(workflow_enabled)

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


    def set_voxel_spacing(
        self,
        spacing: VoxelSpacing,
        *,
        source: str = "Manual override",
        emit: bool = True,
    ) -> None:
        spins_and_values = (
            (self.spacing_x_um, spacing.x_um),
            (self.spacing_y_um, spacing.y_um),
            (self.spacing_z_um, spacing.z_um),
        )
        for spin, value in spins_and_values:
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)
        self.spacing_source_label.setText(source)
        if emit:
            self.spacing_changed.emit(self.current_voxel_spacing())
