from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import logging
import os
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.ndimage as ndi
from PySide6.QtCore import QObject, QThread, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nvap.analysis.microglia_analysis import (
    MicrogliaAnalysisResult,
    analyze_microglia_cells,
    build_microglia_cell_debug,
    microglia_analysis_to_csv_rows,
)
from nvap.analysis.metrics import compute_metrics, metrics_to_csv_rows
from nvap.analysis.vascular_analysis import (
    analyze_vasculature,
    vascular_analysis_to_csv_rows,
)
from nvap.analysis.neurovascular import (
    neurovascular_association_to_csv_rows,
    summarize_neurovascular_association,
)
from nvap.cache.processed_cache import (
    build_dataset_signature,
    build_processed_cache_key,
    has_processed_cache,
    load_enhanced_green,
    load_processed_dataset,
    load_processed_thresholds,
    save_enhanced_dataset,
    save_processed_dataset,
    save_processed_metadata,
)
from nvap.config.types import (
    ChannelVolume,
    DEFAULT_SPACING,
    DatasetVolume,
    MeshExportConfig,
    MetricsComputation,
    PSFConfig,
    PreprocessConfig,
    RenderConfig,
    VoxelSpacing,
)
from nvap.export.exporters import export_metrics_csv
from nvap.export.mesh_export import export_dataset_meshes, reconstruct_combined_mesh
from nvap.io.stack_loader import (
    DatasetProjectCandidate,
    discover_dataset_projects,
    inspect_dataset_stats,
    load_dataset,
    resolve_channel_dirs,
)
from nvap.pipeline import (
    apply_psf_to_dataset,
    fill_and_sync_dataset,
    prepare_dataset_for_mesh,
)
from nvap.analysis.microglia_components import (
    compute_component_labels,
    filter_components_by_preferred_voxel_floor,
    isolate_component,
)
from nvap.preprocess.enhancement import (
    enhance_microglia_background,
    preprocess_dataset,
    wipe_small_specks,
)
from nvap.plugins.registry import discover_plugins
from nvap.render.vtk_scene import MicrogliaDebugOverlay, VTKScene
from nvap.ui.control_panel import ControlPanel
from nvap.ui.design import COLOR, ICON_MD, ICON_SM, SIDEBAR_WIDTH
from nvap.ui.dialogs.about import AboutDialog
from nvap.ui.home_page import HomePage
from nvap.ui.icons import icon, icon_pixmap
from nvap.ui.metrics_panel import MetricsPanel
from nvap.ui.services.recent_projects import RecentProjectsStore
from nvap.samples import register_bundled_samples
from nvap.ui.services.project_files import (
    load_project_state,
    project_channel_sources,
    save_project_state,
)
from nvap.ui.services.system_status import SystemStatus, gpu_status, memory_status
from nvap.update_check import UpdateInfo, check_for_update_async
from nvap.ui.sidebar_pages import SettingsPage

logger = logging.getLogger(__name__)
_METRICS_BACKGROUND_MIN_VOXELS = 20_000_000

# Default render/analysis thresholds chosen for the microscopy workbench.
_DEFAULT_GREEN_THRESHOLD = 0.80
_DEFAULT_RED_THRESHOLD = 0.60


def _green_no_psf_mode(_config: PreprocessConfig) -> bool:
    return True


@dataclass
class _LoadTaskResult:
    synced_dataset: DatasetVolume
    raw_dataset: DatasetVolume
    processed_dataset: DatasetVolume
    visual_dataset: DatasetVolume
    threshold_green: float
    threshold_red: float
    enhancement_method: str | None = None


@dataclass
class _MicrogliaComponentsTaskResult:
    labels: np.ndarray
    order: np.ndarray
    sizes: np.ndarray
    threshold: float
    branch_sensitivity: float
    shape: tuple[int, int, int]


@dataclass
class _MetricsTaskResult:
    cache_key: tuple[int, float, float, int, int, float, float, float]
    metrics: MetricsComputation
    vascular_cache_key: tuple[float, int, int] | None = None
    vascular_line: str | None = None


@dataclass(frozen=True)
class _ProjectAnalyticsConfig:
    render: RenderConfig
    psf: PSFConfig
    preprocess: PreprocessConfig
    apply_enhancement: bool
    enhancement_method: str
    apply_wipe: bool
    wipe_max_voxels: int
    branch_sensitivity: float


@dataclass(frozen=True)
class _ProjectAnalyticsResult:
    base_path: Path
    files: list[Path]
    sample_count: int
    row_count: int


def _vascular_cache_key_for(render: RenderConfig) -> tuple[float, int, int]:
    return (
        float(render.threshold_red),
        int(render.trim_first_slices),
        int(render.trim_last_slices),
    )


def _compute_vascular_summary_line(
    dataset: DatasetVolume | None,
    render: RenderConfig,
) -> str | None:
    """Heavy vascular morphometry summary. MUST run off the UI thread.

    analyze_vasculature skeletonizes the full-resolution red volume and can take
    well over a minute on large stacks; calling it on the Qt main thread freezes
    the window (Windows AppHang) the instant the user pans. Compute it in a
    worker and cache the formatted line for the UI to read.
    """
    if dataset is None:
        return None
    try:
        vascular = analyze_vasculature(
            dataset.red.data,
            threshold=float(render.threshold_red),
            spacing=dataset.red.spacing,
            render=render,
        )
    except Exception:  # pragma: no cover - defensive background path
        return None
    return (
        f"vasculature: vol_fraction={vascular.vessel_volume_fraction:.4f}, "
        f"length_density={vascular.length_density_mm_per_mm3:.2f} mm/mm3, "
        f"mean_diameter_um={vascular.mean_diameter_um:.2f}, "
        f"junctions={vascular.junction_count}, "
        f"tortuosity={vascular.mean_tortuosity:.3f}"
    )


class _AnalyticsMetricCard(QFrame):
    def __init__(self, label: str, value: str = "--", helper: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("analyticsMetricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(6)

        self._value = QLabel(value)
        self._value.setObjectName("analyticsMetricValue")
        self._value.setWordWrap(True)
        layout.addWidget(self._value)

        caption = QLabel(label)
        caption.setObjectName("analyticsMetricLabel")
        caption.setWordWrap(True)
        layout.addWidget(caption)

        self._helper = QLabel(helper)
        self._helper.setObjectName("analyticsMetricHelper")
        self._helper.setWordWrap(True)
        layout.addWidget(self._helper)

    def set_value(self, value: str, helper: str = "") -> None:
        self._value.setText(value)
        self._helper.setText(helper)


class _LogBridge(QObject):
    message = Signal(str)


class _ControlPanelLogHandler(logging.Handler):
    def __init__(self, emit_message: Callable[[str], None]) -> None:
        super().__init__()
        self._emit_message = emit_message

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            self._emit_message(message)
        except Exception:
            # Never let logging errors break UI flow.
            pass


class _FunctionThread(QThread):
    result_ready = Signal(object)
    error_raised = Signal(str)

    def __init__(self, fn: Callable[[], object], parent=None) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn()
            self.result_ready.emit(result)
        except Exception:
            self.error_raised.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    # Emitted from the background update-check thread; Qt marshals delivery
    # onto this (GUI) thread automatically since emitter and receiver differ.
    _update_available = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NVAP — NeuroVascular Analytics Program")
        self.resize(1600, 960)

        # ── Core analysis widgets ────────────────────────────────────────
        self.scene = VTKScene(self)
        self.controls = ControlPanel(self)
        self.controls_scroll = QScrollArea(self)
        self.controls_scroll.setObjectName("controlPanelScrollArea")
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.controls_scroll.setWidget(self.controls)

        self._metrics_panel = MetricsPanel()

        self._viewport_shell = QFrame(self)
        self._viewport_shell.setObjectName("viewportShell")
        viewport_layout = QVBoxLayout(self._viewport_shell)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(0)
        self._project_set_bar = self._build_project_set_bar()
        viewport_layout.addWidget(self._project_set_bar)
        viewport_layout.addWidget(self.scene.widget())

        # Workspace splitter: inspector | VTK viewport | metrics
        self._workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._workspace_splitter.setObjectName("workspaceSplitter")
        self._workspace_splitter.addWidget(self.controls_scroll)
        self._workspace_splitter.addWidget(self._viewport_shell)
        self._workspace_splitter.addWidget(self._metrics_panel)
        self._workspace_splitter.setStretchFactor(0, 0)
        self._workspace_splitter.setStretchFactor(1, 1)
        self._workspace_splitter.setStretchFactor(2, 0)
        self._workspace_splitter.setSizes([320, 980, 260])

        # ── Project store + pages ────────────────────────────────────────
        self._project_store = RecentProjectsStore()
        try:
            register_bundled_samples(self._project_store)
        except Exception as exc:  # pragma: no cover - sample registration is best-effort
            logger.warning("Bundled sample registration failed: %s", exc)
        self._home_page = HomePage(self._project_store)
        self._settings_page = SettingsPage()
        self._analytics_page = self._build_analytics_placeholder()

        self._page_stack = QStackedWidget()
        self._page_stack.addWidget(self._home_page)           # 0
        self._page_stack.addWidget(self._workspace_splitter)  # 1
        self._page_stack.addWidget(self._analytics_page)      # 2
        self._page_stack.addWidget(self._settings_page)       # 3

        # ── Sidebar + shell ──────────────────────────────────────────────
        self._sidebar = self._build_sidebar()
        shell = QFrame()
        shell.setObjectName("appShell")
        shell_lo = QHBoxLayout(shell)
        shell_lo.setContentsMargins(0, 0, 0, 0)
        shell_lo.setSpacing(0)
        shell_lo.addWidget(self._sidebar)
        shell_lo.addWidget(self._page_stack, 1)
        self.setCentralWidget(shell)

        # Wire home page actions
        self._home_page.new_project_requested.connect(self._on_load_requested)
        self._home_page.open_project_requested.connect(self._on_load_requested)
        self._home_page.import_stack_requested.connect(self._on_load_requested)
        self._home_page.browse_samples_requested.connect(self._on_load_requested)
        self._home_page.project_open_requested.connect(self._open_recent_project)
        # project_remove_requested is handled internally by HomePage; we only
        # need to wire pin-toggle so the store stays in sync.
        self._home_page.project_pin_toggled.connect(self._pin_recent_project)

        # Wire settings page actions
        self._settings_page.open_workspace_requested.connect(
            lambda: self._nav_to(1)
        )
        self._settings_page.open_analytics_requested.connect(
            lambda: self._nav_to(2)
        )

        # Wire metrics panel
        self._metrics_panel.view_details_requested.connect(lambda: self._nav_to(2))

        # Start on home page
        self._nav_to(0)
        self._home_page.refresh_projects()

        self.spacing = DEFAULT_SPACING
        self.preprocess_config = PreprocessConfig(enabled=True)
        self.synced_dataset: DatasetVolume | None = None
        self.raw_dataset: DatasetVolume | None = None
        self.processed_dataset: DatasetVolume | None = None
        self.visual_dataset: DatasetVolume | None = None
        self.current_psf = self.controls.current_psf_config()
        self.current_render = self.controls.current_render_config()
        self.latest_metrics: MetricsComputation | None = None
        self.latest_microglia_analysis: MicrogliaAnalysisResult | None = None
        self._metrics_revision = 0
        self._microglia_analysis_revision = 0
        self._metrics_cache_key: tuple[int, float, float, int, int, float, float, float] | None = None
        self._metrics_thread: _FunctionThread | None = None
        self._metrics_task_key: tuple[int, float, float, int, int, float, float, float] | None = None
        self._metrics_refresh_pending = False
        self._vascular_summary_cache: tuple[tuple[float, int, int], str | None] | None = None
        self._microglia_analysis_cache_key: tuple[int, tuple[int, int, int], float, float, int, int, float, float, float, float] | None = None
        self.dataset_root: Path | None = None
        self._dataset_signature: str | None = None
        self._current_channel_sources: dict[str, str] | None = None
        self._current_load_mode = "folder"
        self._project_set_root: Path | None = None
        self._project_set_entries: list[DatasetProjectCandidate] = []
        self._project_set_index = -1
        self._last_processed_cache_key: str | None = None
        self._current_microglia_enhancement_method: str | None = None
        self._current_speck_wipe_applied = False
        self._busy_dialog: QProgressDialog | None = None
        self._busy_start = 0.0
        self._busy_base_message = ""
        self._busy_title = ""
        self._busy_eta_total: float | None = None
        self._busy_progress_percent = 0.0
        self._busy_progress_message = ""
        self._busy_progress_eta_total: float | None = None
        self._busy_progress_lock = threading.Lock()
        self._eta_scale_load = 1.0
        self._eta_scale_psf = 1.0
        self._eta_scale_microglia_separation = 1.0
        self._eta_scale_mesh_export = 1.0
        self._display_z_scale = float(max(0.05, self.current_render.display_z_scale))
        self._busy_timer = QTimer(self)
        self._busy_timer.setInterval(1000)
        self._busy_timer.timeout.connect(self._on_busy_tick)
        self._active_thread: _FunctionThread | None = None
        self._green_component_labels: np.ndarray | None = None
        self._green_component_order = np.empty((0,), dtype=np.int32)
        self._green_component_sizes: np.ndarray | None = None
        self._green_component_threshold: float | None = None
        self._green_component_branch_sensitivity: float | None = None
        self._green_component_shape: tuple[int, int, int] | None = None
        self._green_component_sparse: dict[int, tuple[tuple[slice, slice, slice], np.ndarray]] = {}
        self._green_component_coloring_active = False
        self._microglia_isolate_active = bool(self.controls.microglia_view_state()[0])
        self._microglia_label_cache_max_bytes = 256 * 1024 * 1024
        self._analytics_selection_sync = False

        self._log_bridge = _LogBridge(self)
        self._log_bridge.message.connect(self.controls.append_debug_text)
        self._log_handler = _ControlPanelLogHandler(self._log_bridge.message.emit)
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        nvap_logger = logging.getLogger("nvap")
        nvap_logger.setLevel(logging.INFO)
        nvap_logger.addHandler(self._log_handler)

        self.controls.load_requested.connect(self._on_load_requested)
        self.controls.apply_psf_requested.connect(self._on_apply_psf_requested)
        self.controls.psf_config_changed.connect(self._on_psf_config_changed)
        self.controls.render_config_changed.connect(self._on_render_config_changed)
        self.controls.microglia_view_changed.connect(self._on_microglia_view_changed)
        self.controls.run_microglia_segmentation_requested.connect(
            self._on_run_microglia_segmentation_requested
        )
        self.controls.run_microglia_analysis_requested.connect(
            self._on_run_microglia_analysis_requested
        )
        self.controls.enhance_microglia_requested.connect(self._on_enhance_microglia_requested)
        self.controls.wipe_specks_requested.connect(self._on_wipe_specks_requested)
        self.controls.export_metrics_requested.connect(self._on_export_metrics_requested)
        self.controls.export_project_analytics_requested.connect(
            self._on_export_project_analytics_requested
        )
        self.controls.export_snapshot_requested.connect(self._on_export_snapshot_requested)
        self.controls.export_mesh_requested.connect(self._on_export_mesh_requested)

        self._setup_keyboard_shortcuts()

        self._refresh_plugin_panel()
        self._home_page.refresh_projects()
        QTimer.singleShot(0, self._refresh_system_indicators)
        self.statusBar().showMessage("Load a dataset to begin. Preprocessing is ON by default.")
        self._log_info("NVAP UI initialized: green pass-through mode active.")
        self._refresh_section_state()
        self._update_available.connect(self._on_update_available)
        check_for_update_async(self._update_available.emit)

    def _on_update_available(self, info: object) -> None:
        if not isinstance(info, UpdateInfo):
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Update available")
        box.setText(
            "A newer NVAP build is available:\n"
            f"{info.build_description}\n\n"
            "This does not update automatically — open the download page to get it."
        )
        open_btn = box.addButton("Open Download Page", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl(info.download_url))

    def closeEvent(self, event) -> None:
        try:
            self.controls.force_apply_pending_changes()
        except Exception:
            pass
        if self._active_thread is not None and self._active_thread.isRunning():
            self._log_info("Waiting briefly for active background task to finish before close.")
            self._active_thread.wait(2000)
        if self._metrics_thread is not None and self._metrics_thread.isRunning():
            self._log_info("Waiting briefly for metrics task to finish before close.")
            self._metrics_thread.wait(1000)
        logging.getLogger("nvap").removeHandler(self._log_handler)
        # Clean up VTK resources to avoid GPU/memory leaks.
        try:
            self.scene.cleanup()
        except Exception:
            pass
        super().closeEvent(event)

    def _setup_keyboard_shortcuts(self) -> None:
        """Setup keyboard shortcuts for common operations."""
        # Load dataset
        load_action = QAction("Load Dataset", self)
        load_action.setShortcut(QKeySequence("Ctrl+L"))
        load_action.triggered.connect(self._on_load_requested)
        self.addAction(load_action)

        # Export metrics
        export_metrics_action = QAction("Export Metrics", self)
        export_metrics_action.setShortcut(QKeySequence("Ctrl+E"))
        export_metrics_action.triggered.connect(self._on_export_metrics_requested)
        self.addAction(export_metrics_action)

        # Export snapshot
        snapshot_action = QAction("Export Snapshot", self)
        snapshot_action.setShortcut(QKeySequence("Ctrl+S"))
        snapshot_action.triggered.connect(self._on_export_snapshot_requested)
        self.addAction(snapshot_action)

        # Export mesh
        mesh_action = QAction("Export Mesh", self)
        mesh_action.setShortcut(QKeySequence("Ctrl+M"))
        mesh_action.triggered.connect(self._on_export_mesh_requested)
        self.addAction(mesh_action)

        # Toggle auto-apply
        toggle_auto_action = QAction("Toggle Auto-Apply", self)
        toggle_auto_action.setShortcut(QKeySequence("Ctrl+A"))
        toggle_auto_action.triggered.connect(
            lambda: self.controls.auto_apply_checkbox.setChecked(
                not self.controls.auto_apply_checkbox.isChecked()
            )
        )
        self.addAction(toggle_auto_action)

        # Apply changes (F5 or Enter when apply button is visible)
        apply_action = QAction("Apply Changes", self)
        apply_action.setShortcut(QKeySequence("F5"))
        apply_action.triggered.connect(lambda: self.controls._on_apply_clicked())
        self.addAction(apply_action)

        # Apply changes with Return
        apply_return_action = QAction("Apply Changes (Return)", self)
        apply_return_action.setShortcut(QKeySequence("Return"))
        apply_return_action.triggered.connect(
            lambda: self.controls._on_apply_clicked()
            if self.controls.apply_btn.isVisible()
            else None
        )
        self.addAction(apply_return_action)

    def _log_info(self, message: str) -> None:
        logger.info(message)

    def _log_debug(self, message: str) -> None:
        logger.debug(message)

    # ══════════════════════════════════════════════════════════════════════════
    # Sidebar & navigation
    # ══════════════════════════════════════════════════════════════════════════

    def _build_sidebar(self) -> QFrame:
        """Construct the left navigation sidebar."""
        sidebar = QFrame()
        sidebar.setObjectName("appSidebar")
        sidebar.setFixedWidth(SIDEBAR_WIDTH)

        lo = QVBoxLayout(sidebar)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(0)

        # ── Logotype ────────────────────────────────────────────────────
        logo_row = QFrame()
        logo_row.setObjectName("sidebarLogo")
        logo_lo = QHBoxLayout(logo_row)
        logo_lo.setContentsMargins(16, 18, 16, 14)
        logo_lo.setSpacing(8)

        logo_icon = QLabel()
        logo_icon.setPixmap(icon_pixmap("logo", 22, COLOR.accent))
        logo_icon.setFixedSize(24, 24)
        logo_icon.setScaledContents(True)
        logo_lo.addWidget(logo_icon)

        app_lbl = QLabel("NVAP")
        app_lbl.setObjectName("sidebarAppName")
        logo_lo.addWidget(app_lbl, 1)
        lo.addWidget(logo_row)

        # ── Nav separator ────────────────────────────────────────────────
        sep = QFrame()
        sep.setObjectName("sidebarSep")
        sep.setFrameShape(QFrame.Shape.HLine)
        lo.addWidget(sep)

        # ── Navigation buttons ───────────────────────────────────────────
        nav_lo = QVBoxLayout()
        nav_lo.setContentsMargins(8, 8, 8, 8)
        nav_lo.setSpacing(2)

        _nav_items = [
            ("home",      "Home",      0),
            ("layers",    "Workspace", 1),
            ("bar-chart", "Analytics", 2),
            ("settings",  "Settings",  3),
        ]
        self._nav_buttons: list[QPushButton] = []
        for icon_name, label, page_idx in _nav_items:
            btn = QPushButton(f"  {label}")
            btn.setObjectName("navButton")
            btn.setIcon(icon(icon_name, ICON_MD, COLOR.text_tertiary))
            btn.setCheckable(True)
            btn.setFlat(True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setFixedHeight(40)
            _idx = page_idx  # capture for lambda
            btn.clicked.connect(lambda _checked, i=_idx: self._nav_to(i))
            nav_lo.addWidget(btn)
            self._nav_buttons.append(btn)

        lo.addLayout(nav_lo)
        lo.addStretch(1)

        # ── Bottom strip ─────────────────────────────────────────────────
        sep2 = QFrame()
        sep2.setObjectName("sidebarSep")
        sep2.setFrameShape(QFrame.Shape.HLine)
        lo.addWidget(sep2)

        bottom_lo = QVBoxLayout()
        bottom_lo.setContentsMargins(8, 8, 8, 8)
        bottom_lo.setSpacing(2)

        # GPU / memory status pills
        self._gpu_pill = self._make_status_pill("GPU", "unknown")
        self._mem_pill = self._make_status_pill("MEM", "unknown")
        bottom_lo.addWidget(self._gpu_pill)
        bottom_lo.addWidget(self._mem_pill)

        bottom_sep = QFrame()
        bottom_sep.setObjectName("sidebarSep")
        bottom_sep.setFrameShape(QFrame.Shape.HLine)
        bottom_lo.addWidget(bottom_sep)

        about_btn = QPushButton("  About")
        about_btn.setObjectName("navButton")
        about_btn.setIcon(icon("info", ICON_SM, COLOR.text_tertiary))
        about_btn.setFlat(True)
        about_btn.setFixedHeight(36)
        about_btn.clicked.connect(self._open_about)
        bottom_lo.addWidget(about_btn)

        try:
            import importlib.metadata as _imeta
            _ver = _imeta.version("nvap")
        except Exception:
            _ver = "dev"
        ver_lbl = QLabel(f"v{_ver}")
        ver_lbl.setObjectName("sidebarVersion")
        ver_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bottom_lo.addWidget(ver_lbl)

        lo.addLayout(bottom_lo)
        return sidebar

    def _build_project_set_bar(self) -> QFrame:
        bar = QFrame(self)
        bar.setObjectName("projectSetBar")
        bar.setVisible(False)
        lo = QHBoxLayout(bar)
        lo.setContentsMargins(12, 8, 12, 8)
        lo.setSpacing(8)

        label = QLabel("Project set")
        label.setObjectName("metricLabel")
        lo.addWidget(label)

        self._project_set_combo = QComboBox(bar)
        self._project_set_combo.setObjectName("projectSetCombo")
        self._project_set_combo.setMinimumWidth(260)
        self._project_set_combo.currentIndexChanged.connect(self._on_project_set_combo_changed)
        lo.addWidget(self._project_set_combo, 1)

        self._project_set_count_label = QLabel("")
        self._project_set_count_label.setObjectName("metricLabel")
        lo.addWidget(self._project_set_count_label)

        prev_btn = QPushButton("Previous")
        prev_btn.setObjectName("secondaryActionHome")
        prev_btn.clicked.connect(lambda: self._move_project_set_selection(-1))
        lo.addWidget(prev_btn)
        self._project_set_prev_btn = prev_btn

        next_btn = QPushButton("Next")
        next_btn.setObjectName("secondaryActionHome")
        next_btn.clicked.connect(lambda: self._move_project_set_selection(1))
        lo.addWidget(next_btn)
        self._project_set_next_btn = next_btn
        return bar

    def _build_analytics_placeholder(self) -> QWidget:
        """Analytics overview for dataset and per-cell microglia metrics."""
        w = QWidget()
        w.setObjectName("sectionPage")
        lo = QVBoxLayout(w)
        lo.setContentsMargins(40, 36, 40, 40)
        lo.setSpacing(0)

        eyebrow = QLabel("ANALYTICS")
        eyebrow.setObjectName("pageEyebrow")
        lo.addWidget(eyebrow)

        title = QLabel("Analytics")
        title.setObjectName("pageTitle")
        lo.addWidget(title)

        intro = QLabel("Dataset-scale channel metrics from the current render thresholds.")
        intro.setObjectName("pageIntro")
        intro.setWordWrap(True)
        lo.addWidget(intro)
        lo.addSpacing(20)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        self._project_analytics_btn = QPushButton("Apply Current Settings to Project Set + Export CSV")
        self._project_analytics_btn.setObjectName("primaryAction")
        self._project_analytics_btn.setIcon(icon("bar-chart", ICON_SM, COLOR.text_inverse))
        self._project_analytics_btn.setToolTip(
            "Apply this sample's thresholds, clean, and wipe settings to every sample "
            "in the project set, then export individual and cumulative analytics."
        )
        self._project_analytics_btn.clicked.connect(self._on_export_project_analytics_requested)
        action_row.addWidget(self._project_analytics_btn)
        action_row.addStretch(1)
        lo.addLayout(action_row)
        lo.addSpacing(18)

        card_grid = QGridLayout()
        card_grid.setHorizontalSpacing(12)
        card_grid.setVerticalSpacing(12)
        self._analytics_cards: dict[str, _AnalyticsMetricCard] = {
            "volume": _AnalyticsMetricCard("Green volume", "--", "Thresholded green channel."),
            "overlap": _AnalyticsMetricCard("Channel overlap", "--", "Green/red shared voxels."),
            "green_components": _AnalyticsMetricCard("Green components", "--", "Connected thresholded objects."),
            "red_components": _AnalyticsMetricCard("Red components", "--", "Connected thresholded objects."),
        }
        for idx, card in enumerate(self._analytics_cards.values()):
            card_grid.addWidget(card, idx // 2, idx % 2)
        lo.addLayout(card_grid)

        section = QLabel("MICROGLIA ANALYSIS")
        section.setObjectName("pageEyebrow")
        lo.addSpacing(24)
        lo.addWidget(section)

        self._analytics_microglia_hint = QLabel(
            "Open Analytics on a rendered dataset to measure separated microglia against visible vasculature."
        )
        self._analytics_microglia_hint.setObjectName("pageIntro")
        self._analytics_microglia_hint.setWordWrap(True)
        lo.addWidget(self._analytics_microglia_hint)
        lo.addSpacing(16)

        analysis_grid = QGridLayout()
        analysis_grid.setHorizontalSpacing(12)
        analysis_grid.setVerticalSpacing(12)
        self._analytics_microglia_cards: dict[str, _AnalyticsMetricCard] = {
            "cells": _AnalyticsMetricCard("Cells analyzed", "--", "Visible separated microglia."),
            "branches": _AnalyticsMetricCard("Avg branches", "--", "Branch tips per visible cell."),
            "soma": _AnalyticsMetricCard("Avg soma volume", "--", "Non-branched soma body volume."),
            "soma_shape": _AnalyticsMetricCard("Avg soma roundness", "--", "Soma shape: 1.0 is round."),
            "distance": _AnalyticsMetricCard("Closest vessel distance", "--", "Shortest cell-to-vessel distance."),
            "tip_vessels": _AnalyticsMetricCard("Tips near vessels", "--", "Cells whose tips are near multiple vessel components."),
        }
        for idx, card in enumerate(self._analytics_microglia_cards.values()):
            analysis_grid.addWidget(card, idx // 2, idx % 2)
        lo.addLayout(analysis_grid)
        lo.addSpacing(16)

        self._analytics_cell_table = QTableWidget(0, 10)
        self._analytics_cell_table.setObjectName("analyticsCellTable")
        self._analytics_cell_table.setHorizontalHeaderLabels(
            [
                "Cell",
                "Branches",
                "Branch tortuosity",
                "Soma (um^3)",
                "Soma diameter (um)",
                "Soma roundness",
                "Tip -> Vessel (um)",
                "Cell -> Vessel (um)",
                "Soma -> Vessel (um)",
                "Soma center -> Vessel (um)",
            ]
        )
        self._analytics_cell_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._analytics_cell_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._analytics_cell_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._analytics_cell_table.verticalHeader().setVisible(False)
        self._analytics_cell_table.horizontalHeader().setStretchLastSection(True)
        self._analytics_cell_table.setMinimumHeight(260)
        self._analytics_cell_table.itemSelectionChanged.connect(self._on_analytics_cell_selected)
        lo.addWidget(self._analytics_cell_table)
        lo.addStretch(1)
        return w

    def _refresh_analytics_metrics(self) -> None:
        if not hasattr(self, "_analytics_cards"):
            return

        # Whole-volume cards (green volume, overlap, component counts) come from
        # the slower background metrics task. Show a clear pending state while it
        # runs rather than leaving stale numbers.
        if self.latest_metrics is None:
            pending = (
                "Computing metrics…"
                if self.processed_dataset is not None
                else "Load a dataset to populate."
            )
            for card in self._analytics_cards.values():
                card.set_value("--", pending)
        else:
            for card in self._analytics_cards.values():
                card.set_value("--", "Load a dataset to populate.")
            by_channel = {item.channel.lower(): item for item in self.latest_metrics.channel_results}
            green = by_channel.get("green")
            red = by_channel.get("red")
            if green is not None:
                self._analytics_cards["volume"].set_value(
                    f"{green.volume_um3:,.1f}",
                    f"{green.voxel_count:,} voxels",
                )
                self._analytics_cards["green_components"].set_value(
                    f"{green.component_count:,}",
                    f"largest {green.largest_component_voxels:,} voxels",
                )
            if red is not None:
                self._analytics_cards["red_components"].set_value(
                    f"{red.component_count:,}",
                    f"largest {red.largest_component_voxels:,} voxels",
                )
            self._analytics_cards["overlap"].set_value(
                f"{self.latest_metrics.overlap_volume_um3:,.1f}",
                f"{self.latest_metrics.overlap_voxel_count:,} voxels",
            )

        # Per-cell microglia analysis is independent of the whole-volume metrics
        # above, so refresh it as soon as Analytics is open instead of gating it
        # behind the (much slower) vascular morphometry. Previously this sat
        # behind an early `return` when metrics were still pending, so the
        # microglia cards showed "--" for minutes after a load or a Wipe Specks.
        if self.visual_dataset is None:
            self._clear_analytics_microglia_widgets("Load a dataset to populate.")
        elif self._page_stack.currentIndex() == 2 or self.latest_microglia_analysis is not None:
            self._refresh_microglia_analysis()
        else:
            self._clear_analytics_microglia_widgets(
                "Open Analytics to compute per-cell microglia measurements for the current view."
            )

    def _refresh_microglia_analysis(self) -> None:
        if self.visual_dataset is None:
            self._clear_analytics_microglia_widgets("Load a dataset to analyze separated microglia.")
            return

        cache_key = self._microglia_analysis_cache_key_for_render(self.current_render)
        if self.latest_microglia_analysis is None or self._microglia_analysis_cache_key != cache_key:
            # Runs synchronously and can take ~1 minute on large volumes; show a
            # wait cursor + status so the UI doesn't look frozen or stuck on "--".
            self._analytics_microglia_hint.setText(
                "Analyzing visible microglia… this can take up to a minute on large volumes."
            )
            self.statusBar().showMessage("Analyzing microglia cells…")
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()
            try:
                labels, order, _sizes = self._ensure_microglia_components_current()
                branch_sensitivity = float(self.controls.current_microglia_branch_sensitivity())
                self.latest_microglia_analysis = analyze_microglia_cells(
                    self.visual_dataset.green.data,
                    self.visual_dataset.red.data,
                    labels,
                    order,
                    spacing=self.visual_dataset.green.spacing,
                    render=self.current_render,
                    branch_sensitivity=branch_sensitivity,
                )
                self._microglia_analysis_cache_key = cache_key
            finally:
                QApplication.restoreOverrideCursor()
                self.statusBar().showMessage("Microglia analysis updated.", 3000)

        self._set_analytics_microglia_result(self.latest_microglia_analysis)

    def _ensure_microglia_components_current(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.visual_dataset is None:
            raise RuntimeError("No visual dataset loaded.")

        threshold = float(self.current_render.threshold_green)
        branch_sensitivity = float(self.controls.current_microglia_branch_sensitivity())
        green = np.asarray(self.visual_dataset.green.data, dtype=np.float32)
        if (
            self._green_component_sizes is not None
            and self._green_component_threshold is not None
            and self._green_component_branch_sensitivity is not None
            and np.isclose(self._green_component_threshold, threshold, atol=1.0e-6)
            and np.isclose(self._green_component_branch_sensitivity, branch_sensitivity, atol=1.0e-6)
            and self._green_component_shape == green.shape
            and self._green_component_labels is not None
        ):
            return (
                np.asarray(self._green_component_labels, dtype=np.int32),
                np.asarray(self._green_component_order, dtype=np.int32),
                np.asarray(self._green_component_sizes, dtype=np.int64),
            )

        labels, order, sizes = self._compute_microglia_components(green, threshold)
        selected_index = int(self.controls.microglia_view_state()[1])
        self._cache_microglia_components(
            labels,
            order,
            sizes,
            threshold=threshold,
            branch_sensitivity=branch_sensitivity,
            shape=green.shape,
            selected_index=selected_index,
        )
        return labels, order, sizes

    def _set_analytics_microglia_result(self, analysis: MicrogliaAnalysisResult) -> None:
        if not hasattr(self, "_analytics_microglia_cards"):
            return

        self._analytics_microglia_cards["cells"].set_value(
            f"{analysis.analyzed_cell_count:,}",
            "Visible separated cells in the current view.",
        )
        self._analytics_microglia_cards["branches"].set_value(
            f"{analysis.mean_branch_count:,.2f}",
            f"Avg process branches per cell ({analysis.mean_tip_count:,.1f} tips, "
            f"{analysis.mean_process_length_um:,.1f} um length, "
            f"{analysis.mean_branch_tortuosity:,.2f} tortuosity).",
        )
        self._analytics_microglia_cards["soma"].set_value(
            f"{analysis.mean_soma_volume_um3:,.1f}",
            f"Avg soma diameter {analysis.mean_soma_equivalent_diameter_um:,.1f} um.",
        )
        self._analytics_microglia_cards["soma_shape"].set_value(
            f"{analysis.mean_soma_roundness:,.2f}",
            "1.0 is rounder; lower values are elongated/rectangular.",
        )
        closest_distance = (
            "--"
            if analysis.min_cell_to_vessel_um is None
            else f"{analysis.min_cell_to_vessel_um:,.2f}"
        )
        self._analytics_microglia_cards["distance"].set_value(
            closest_distance,
            "Shortest visible cell-to-vessel distance.",
        )
        self._analytics_microglia_cards["tip_vessels"].set_value(
            f"{analysis.cells_with_tips_near_multiple_vessels:,}",
            "Cells with one or more tips within 5 um of multiple vessel components.",
        )

        self._analytics_cell_table.setRowCount(int(len(analysis.cells)))
        for row, cell in enumerate(analysis.cells):
            values = [
                str(row + 1),
                str(cell.branch_count),
                f"{cell.mean_branch_tortuosity:,.2f}",
                f"{cell.soma_volume_um3:,.1f}",
                f"{cell.soma_equivalent_diameter_um:,.1f}",
                f"{cell.soma_roundness:,.2f}",
                self._format_optional_analytics_value(cell.nearest_tip_to_vessel_um),
                self._format_optional_analytics_value(cell.nearest_cell_to_vessel_um),
                self._format_optional_analytics_value(cell.soma_to_vessel_um),
                self._format_optional_analytics_value(cell.soma_centroid_to_vessel_um),
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if col == 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                else:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._analytics_cell_table.setItem(row, col, item)

        if analysis.analyzed_cell_count <= 0:
            self._analytics_microglia_hint.setText(
                "No visible separated microglia were found for the current thresholds and trim."
            )
        else:
            self._analytics_microglia_hint.setText(
                "Selecting a row isolates the same microglia in the viewer and keeps the measurements aligned with the current view."
            )
        self._analytics_cell_table.resizeColumnsToContents()
        self._sync_analytics_table_selection()

    def _clear_analytics_microglia_widgets(self, message: str) -> None:
        if not hasattr(self, "_analytics_microglia_cards"):
            return
        for card in self._analytics_microglia_cards.values():
            card.set_value("--", "Load a dataset to populate.")
        self._analytics_microglia_hint.setText(message)
        self._analytics_cell_table.clearSelection()
        self._analytics_cell_table.setRowCount(0)

    def _format_optional_analytics_value(self, value: float | None) -> str:
        if value is None:
            return "--"
        return f"{float(value):,.2f}"

    def _refresh_microglia_analysis_debug(self) -> None:
        if self.visual_dataset is None:
            self.scene.set_microglia_analysis_debug(None)
            return
        if not self.controls.microglia_analysis_debug_enabled():
            self.scene.set_microglia_analysis_debug(None)
            return

        isolate_enabled, selected_index = self.controls.microglia_view_state()
        if not isolate_enabled or selected_index <= 0:
            self.scene.set_microglia_analysis_debug(None)
            return

        labels, order, _sizes = self._ensure_microglia_components_current()
        if selected_index > int(len(order)):
            self.scene.set_microglia_analysis_debug(None)
            return

        component_id = int(order[selected_index - 1])
        known_tip_distance = None
        known_soma_distance = None
        known_cell_distance = None
        if self.latest_microglia_analysis is not None:
            for cell in self.latest_microglia_analysis.cells:
                if int(cell.component_id) == component_id:
                    known_tip_distance = cell.nearest_tip_to_vessel_um
                    known_soma_distance = cell.soma_to_vessel_um
                    known_cell_distance = cell.nearest_cell_to_vessel_um
                    break

        debug = build_microglia_cell_debug(
            self.visual_dataset.green.data,
            self.visual_dataset.red.data,
            labels,
            component_id,
            spacing=self.visual_dataset.green.spacing,
            render=self.current_render,
            branch_sensitivity=float(self.controls.current_microglia_branch_sensitivity()),
            known_tip_distance_um=known_tip_distance,
            known_soma_distance_um=known_soma_distance,
            known_cell_distance_um=known_cell_distance,
        )
        if debug is None:
            self.scene.set_microglia_analysis_debug(None)
            return

        debug_layers = self.controls.microglia_analysis_debug_layers()
        green_spacing = self._display_spacing(self.visual_dataset.green.spacing)
        red_spacing = self._display_spacing(self.visual_dataset.red.spacing)
        green_offset = (
            float(self.current_render.offset_x_um),
            float(self.current_render.offset_y_um),
            float(self.current_render.offset_z_um),
        )
        overlay = MicrogliaDebugOverlay(
            voxel_points_xyz=(
                self._coords_zyx_to_world_xyz(
                    debug.voxel_sample_coords_zyx,
                    spacing=green_spacing,
                    offset_xyz=green_offset,
                )
                if "voxels" in debug_layers
                else np.empty((0, 3), dtype=np.float32)
            ),
            branch_points_xyz=(
                self._coords_zyx_to_world_xyz(
                    debug.branch_sample_coords_zyx,
                    spacing=green_spacing,
                    offset_xyz=green_offset,
                )
                if "branches" in debug_layers
                else np.empty((0, 3), dtype=np.float32)
            ),
            soma_points_xyz=(
                self._coords_zyx_to_world_xyz(
                    debug.soma_sample_coords_zyx,
                    spacing=green_spacing,
                    offset_xyz=green_offset,
                )
                if "soma" in debug_layers
                else np.empty((0, 3), dtype=np.float32)
            ),
            tip_points_xyz=(
                self._coords_zyx_to_world_xyz(
                    debug.tip_coords_zyx,
                    spacing=green_spacing,
                    offset_xyz=green_offset,
                )
                if "tips" in debug_layers
                else np.empty((0, 3), dtype=np.float32)
            ),
            tip_segments_xyz=(
                self._segment_zyx_to_world_xyz(
                    debug.nearest_tip_segment_zyx,
                    source_spacing=green_spacing,
                    target_spacing=red_spacing,
                    source_offset_xyz=green_offset,
                )
                if "tip_distance" in debug_layers
                else np.empty((0, 2, 3), dtype=np.float32)
            ),
            soma_segments_xyz=(
                self._segment_zyx_to_world_xyz(
                    debug.nearest_soma_segment_zyx,
                    source_spacing=green_spacing,
                    target_spacing=red_spacing,
                    source_offset_xyz=green_offset,
                )
                if "soma_distance" in debug_layers
                else np.empty((0, 2, 3), dtype=np.float32)
            ),
            cell_segments_xyz=(
                self._segment_zyx_to_world_xyz(
                    debug.nearest_cell_segment_zyx,
                    source_spacing=green_spacing,
                    target_spacing=red_spacing,
                    source_offset_xyz=green_offset,
                )
                if "cell_distance" in debug_layers
                else np.empty((0, 2, 3), dtype=np.float32)
            ),
        )
        self.scene.set_microglia_analysis_debug(overlay)

    @staticmethod
    def _coords_zyx_to_world_xyz(
        coords_zyx: np.ndarray,
        *,
        spacing: VoxelSpacing,
        offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        coords = np.asarray(coords_zyx, dtype=np.float32)
        if coords.size <= 0:
            return np.empty((0, 3), dtype=np.float32)
        world = np.empty((coords.shape[0], 3), dtype=np.float32)
        world[:, 0] = (coords[:, 2] + 0.5) * float(spacing.x_um) + float(offset_xyz[0])
        world[:, 1] = (coords[:, 1] + 0.5) * float(spacing.y_um) + float(offset_xyz[1])
        world[:, 2] = (coords[:, 0] + 0.5) * float(spacing.z_um) + float(offset_xyz[2])
        return world

    def _segment_zyx_to_world_xyz(
        self,
        segment_zyx: np.ndarray | None,
        *,
        source_spacing: VoxelSpacing,
        target_spacing: VoxelSpacing,
        source_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        if segment_zyx is None:
            return np.empty((0, 2, 3), dtype=np.float32)
        segment = np.asarray(segment_zyx, dtype=np.float32)
        if segment.shape != (2, 3):
            return np.empty((0, 2, 3), dtype=np.float32)
        source = self._coords_zyx_to_world_xyz(
            segment[:1],
            spacing=source_spacing,
            offset_xyz=source_offset_xyz,
        )
        target = self._coords_zyx_to_world_xyz(segment[1:], spacing=target_spacing)
        return np.asarray([[source[0], target[0]]], dtype=np.float32)

    def _on_analytics_cell_selected(self) -> None:
        if self._analytics_selection_sync or self.latest_microglia_analysis is None:
            return
        row = int(self._analytics_cell_table.currentRow())
        if row < 0 or row >= int(len(self.latest_microglia_analysis.cells)):
            return

        selected_index = row + 1
        self._analytics_selection_sync = True
        try:
            self.controls.microglia_isolate.blockSignals(True)
            self.controls.microglia_index.blockSignals(True)
            self.controls.microglia_isolate.setChecked(True)
            self.controls.microglia_index.setValue(selected_index)
        finally:
            self.controls.microglia_isolate.blockSignals(False)
            self.controls.microglia_index.blockSignals(False)
            self._analytics_selection_sync = False

        self._on_microglia_view_changed()

    def _sync_analytics_table_selection(self) -> None:
        if not hasattr(self, "_analytics_cell_table") or self.latest_microglia_analysis is None:
            return
        isolate_enabled, selected_index = self.controls.microglia_view_state()
        target_row = int(selected_index) - 1 if isolate_enabled and selected_index > 0 else -1
        self._analytics_selection_sync = True
        try:
            if target_row < 0 or target_row >= int(self._analytics_cell_table.rowCount()):
                self._analytics_cell_table.clearSelection()
            else:
                self._analytics_cell_table.selectRow(target_row)
        finally:
            self._analytics_selection_sync = False

    def _mark_microglia_analysis_dirty(self) -> None:
        self._microglia_analysis_revision += 1
        self._microglia_analysis_cache_key = None
        self.latest_microglia_analysis = None

    def _microglia_analysis_cache_key_for_render(
        self,
        render: RenderConfig,
    ) -> tuple[int, tuple[int, int, int], float, float, int, int, float, float, float, float]:
        if self.visual_dataset is None:
            raise RuntimeError("No visual dataset loaded.")
        return (
            int(self._microglia_analysis_revision),
            tuple(int(v) for v in self.visual_dataset.green.data.shape),
            float(render.threshold_green),
            float(render.threshold_red),
            int(render.trim_first_slices),
            int(render.trim_last_slices),
            float(render.offset_x_um),
            float(render.offset_y_um),
            float(render.offset_z_um),
            float(self.controls.current_microglia_branch_sensitivity()),
        )

    def _clear_project_set(self) -> None:
        self._project_set_root = None
        self._project_set_entries = []
        self._project_set_index = -1
        if hasattr(self, "_project_set_combo"):
            self._project_set_combo.blockSignals(True)
            try:
                self._project_set_combo.clear()
            finally:
                self._project_set_combo.blockSignals(False)
        if hasattr(self, "_project_set_count_label"):
            self._project_set_count_label.setText("")
        if hasattr(self, "_project_set_bar"):
            self._project_set_bar.setVisible(False)

    def _set_project_set(
        self,
        root: Path,
        entries: list[DatasetProjectCandidate],
        *,
        selected_index: int = 0,
    ) -> None:
        self._project_set_root = root.resolve()
        self._project_set_entries = list(entries)
        self._project_set_index = -1

        self._project_set_combo.blockSignals(True)
        try:
            self._project_set_combo.clear()
            for idx, entry in enumerate(self._project_set_entries):
                self._project_set_combo.addItem(entry.name, idx)
        finally:
            self._project_set_combo.blockSignals(False)

        self._project_set_bar.setVisible(bool(self._project_set_entries))
        self._project_set_count_label.setText(f"{len(self._project_set_entries)} datasets")
        self._load_project_set_entry(selected_index)

    def _load_project_set_entry(self, index: int) -> None:
        if not self._project_set_entries:
            return
        bounded = max(0, min(int(index), len(self._project_set_entries) - 1))
        entry = self._project_set_entries[bounded]
        self._project_set_index = bounded
        self._project_set_combo.blockSignals(True)
        try:
            self._project_set_combo.setCurrentIndex(bounded)
        finally:
            self._project_set_combo.blockSignals(False)
        self._project_set_count_label.setText(
            f"{bounded + 1} of {len(self._project_set_entries)}"
        )
        self._project_set_prev_btn.setEnabled(bounded > 0)
        self._project_set_next_btn.setEnabled(bounded < len(self._project_set_entries) - 1)
        self._log_info(
            f"Project set: opening {entry.name} ({bounded + 1}/{len(self._project_set_entries)})."
        )
        self._start_dataset_load(
            entry.root,
            channel_overrides=None,
            channel_dirs=entry.channel_dirs,
            load_mode="project_set",
        )

    def _on_project_set_combo_changed(self, index: int) -> None:
        if index < 0 or index == self._project_set_index:
            return
        self._load_project_set_entry(index)

    def _move_project_set_selection(self, delta: int) -> None:
        if not self._project_set_entries:
            return
        self._load_project_set_entry(self._project_set_index + int(delta))

    def _nav_to(self, page_idx: int) -> None:
        """Switch the page stack and update nav button checked states."""
        self._page_stack.setCurrentIndex(page_idx)
        for i, btn in enumerate(self._nav_buttons):
            btn.setChecked(i == page_idx)
            # Refresh icon color to reflect active state
            icon_names = ["home", "layers", "bar-chart", "settings"]
            color = COLOR.accent if i == page_idx else COLOR.text_tertiary
            btn.setIcon(icon(icon_names[i], ICON_MD, color))
        if page_idx == 2:
            self._refresh_analytics_metrics()

    def _open_about(self) -> None:
        AboutDialog(self).exec()

    # ── Status pills ───────────────────────────────────────────────────

    def _make_status_pill(self, label: str, status: str) -> QFrame:
        pill = QFrame()
        pill.setObjectName("statusPill")
        pill_lo = QHBoxLayout(pill)
        pill_lo.setContentsMargins(10, 4, 10, 4)
        pill_lo.setSpacing(6)

        dot = QLabel("●")
        dot.setObjectName(f"statusDot_{status}")
        dot.setFixedWidth(10)
        pill_lo.addWidget(dot)

        text = QLabel(label)
        text.setObjectName("statusPillLabel")
        pill_lo.addWidget(text, 1)

        pill._nvap_dot = dot   # type: ignore[attr-defined]
        pill._nvap_text = text  # type: ignore[attr-defined]
        return pill

    def _update_status_pill(self, pill: QFrame, status: SystemStatus) -> None:
        dot: QLabel = pill._nvap_dot  # type: ignore[attr-defined]
        text: QLabel = pill._nvap_text  # type: ignore[attr-defined]
        color_map = {
            "good":    COLOR.success,
            "warn":    COLOR.warning,
            "bad":     COLOR.danger,
            "idle":    COLOR.text_disabled,
            "unknown": COLOR.text_disabled,
        }
        bg_map = {
            "good":    COLOR.channel_green_subtle,
            "warn":    COLOR.accent_subtle,
            "bad":     COLOR.channel_red_subtle,
            "idle":    COLOR.bg_surface,
            "unknown": COLOR.bg_surface,
        }
        color = color_map.get(status.status, COLOR.text_disabled)
        bg = bg_map.get(status.status, COLOR.bg_surface)
        dot.setStyleSheet(f"color: {color};")
        text.setText(status.label)
        text.setStyleSheet(f"color: {color};")
        pill.setStyleSheet(
            f"QFrame#statusPill {{ background-color: {bg}; border: 1px solid {color}; border-radius: 2px; }}"
        )
        if status.detail:
            pill.setToolTip(status.detail)
        else:
            pill.setToolTip("")

    def _refresh_system_indicators(self) -> None:
        try:
            self._update_status_pill(self._gpu_pill, gpu_status())
        except Exception:
            pass
        try:
            self._update_status_pill(self._mem_pill, memory_status())
        except Exception:
            pass

    # ── Recent project management ──────────────────────────────────────

    def _record_current_project(
        self,
        *,
        samples: int | None = None,
        status: str | None = None,
    ) -> None:
        if self.dataset_root is None:
            return
        name = self.dataset_root.name
        self._project_store.upsert(
            str(self.dataset_root),
            name=name,
            samples=samples if samples is not None else 0,
            status=status if status is not None else "Open",
        )
        self._home_page.refresh_projects()

    def _save_current_project_state(self) -> None:
        if (
            self.dataset_root is None
            or self._current_channel_sources is None
            or self._dataset_signature is None
        ):
            return
        try:
            save_project_state(
                self.dataset_root,
                channel_sources=self._current_channel_sources,
                dataset_signature=self._dataset_signature,
                load_mode=self._current_load_mode,
                spacing=self.spacing,
                psf_config=self.current_psf,
                preprocess_config=self.preprocess_config,
                cache_key=self._last_processed_cache_key,
                enhancement_method=self._current_microglia_enhancement_method,
            )
        except OSError as exc:
            self._log_info(f"Project metadata save failed: {exc}")

    def _project_cache_base_dir(self) -> Path:
        return self.dataset_root or Path.cwd()

    def _refresh_section_state(self) -> None:
        dataset_name = self.dataset_root.name if self.dataset_root is not None else "No dataset loaded"
        set_detail = ""
        if self._project_set_entries and 0 <= self._project_set_index < len(self._project_set_entries):
            dataset_name = (
                f"{dataset_name} ({self._project_set_index + 1}/"
                f"{len(self._project_set_entries)})"
            )
            if self._project_set_root is not None:
                set_detail = f"\nProject set: {self._project_set_root}"
        cache_root = str(self._project_cache_base_dir() / ".nvap_cache")
        plugin_summary = self.controls.plugin_text.toPlainText().strip() or "No plugins discovered"
        auto_apply = bool(self.controls.auto_apply_checkbox.isChecked())
        has_dataset = bool(self.processed_dataset is not None and self.visual_dataset is not None)
        has_project_set = bool(len(self._project_set_entries) > 1)
        self._settings_page.set_runtime_details(
            dataset_name=dataset_name,
            auto_apply_enabled=auto_apply,
            plugin_summary=plugin_summary,
            cache_root=cache_root,
        )
        self.controls.set_microglia_workflow_enabled(has_dataset)
        if hasattr(self, "_project_analytics_btn"):
            self._project_analytics_btn.setEnabled(has_dataset and has_project_set)
        if self.dataset_root is not None:
            self._home_page.set_preview_summary(
                "ACTIVE",
                self.dataset_root.name,
                f"{self.dataset_root}{set_detail}\nCache: {cache_root}",
            )
        self._home_page.refresh_projects()

    def _on_run_microglia_segmentation_requested(self) -> None:
        if self.visual_dataset is None or self.processed_dataset is None:
            self._show_error("No dataset", "Load and render a dataset before running segmentation.")
            return
        if self._active_thread is not None and self._active_thread.isRunning():
            self.statusBar().showMessage("Another operation is still running.", 3000)
            return

        self.controls.microglia_isolate.blockSignals(True)
        try:
            self.controls.microglia_isolate.setChecked(True)
        finally:
            self.controls.microglia_isolate.blockSignals(False)
        self._microglia_isolate_active = True
        self._invalidate_microglia_components()
        self.statusBar().showMessage("Running microglia segmentation...", 3000)
        self._start_microglia_refresh_task()

    def _on_run_microglia_analysis_requested(self) -> None:
        if self.visual_dataset is None or self.processed_dataset is None:
            self._show_error("No dataset", "Load and render a dataset before running analysis.")
            return
        if self._active_thread is not None and self._active_thread.isRunning():
            self.statusBar().showMessage("Another operation is still running.", 3000)
            return

        self._nav_to(2)
        self._refresh_metrics()
        self.statusBar().showMessage("Microglia analysis updated.", 3000)

    def _open_recent_project(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            QMessageBox.warning(
                self,
                "Project Not Found",
                f"The dataset folder no longer exists:\n{path}",
            )
            self._project_store.remove(path)
            self._home_page.refresh_projects()
            return
        try:
            state = load_project_state(p)
            channel_overrides = project_channel_sources(state) if state is not None else None
            channel_dirs = resolve_channel_dirs(p, channel_overrides=channel_overrides)
        except Exception as exc:
            self._show_error("Project load failed", str(exc))
            return
        self._clear_project_set()
        self.dataset_root = p.resolve()
        self._nav_to(1)
        self._start_dataset_load(
            self.dataset_root,
            channel_overrides=channel_overrides,
            channel_dirs=channel_dirs,
            load_mode=str(state.get("load_mode", "project")) if state is not None else "project",
        )

    def _pin_recent_project(self, path: str, pinned: bool) -> None:
        self._project_store.set_pinned(path, pinned)
        self._home_page.refresh_projects()

    def _display_spacing(self, spacing: VoxelSpacing) -> VoxelSpacing:
        # Visual-only Z squeeze for less depth exaggeration. Metrics stay in physical units.
        z_scale = float(max(0.05, self._display_z_scale))
        return VoxelSpacing(
            x_um=spacing.x_um,
            y_um=spacing.y_um,
            z_um=float(spacing.z_um) * z_scale,
        )

    @staticmethod
    def _apply_render_trim(
        volume: np.ndarray,
        trim_first_slices: int,
        trim_last_slices: int,
    ) -> np.ndarray:
        arr = np.asarray(volume)
        if arr.ndim != 3 or arr.shape[0] <= 0:
            return arr
        trim_first = max(0, int(trim_first_slices))
        trim_last = max(0, int(trim_last_slices))
        if trim_first <= 0 and trim_last <= 0:
            return arr
        if trim_first + trim_last >= int(arr.shape[0]):
            return np.zeros_like(arr)
        out = np.array(arr, copy=True)
        if trim_first > 0:
            out[:trim_first] = 0
        if trim_last > 0:
            out[-trim_last:] = 0
        return out

    @staticmethod
    def _render_trim_changed(previous: RenderConfig, current: RenderConfig) -> bool:
        return bool(
            int(previous.trim_first_slices) != int(current.trim_first_slices)
            or int(previous.trim_last_slices) != int(current.trim_last_slices)
        )

    @staticmethod
    def _render_config_affects_metrics(previous: RenderConfig, current: RenderConfig) -> bool:
        return bool(
            not np.isclose(float(previous.threshold_green), float(current.threshold_green), atol=1.0e-6)
            or not np.isclose(float(previous.threshold_red), float(current.threshold_red), atol=1.0e-6)
            or int(previous.trim_first_slices) != int(current.trim_first_slices)
            or int(previous.trim_last_slices) != int(current.trim_last_slices)
            or not np.isclose(float(previous.offset_x_um), float(current.offset_x_um), atol=1.0e-6)
            or not np.isclose(float(previous.offset_y_um), float(current.offset_y_um), atol=1.0e-6)
            or not np.isclose(float(previous.offset_z_um), float(current.offset_z_um), atol=1.0e-6)
        )

    def _mark_metrics_dirty(self) -> None:
        self._metrics_revision += 1
        self._metrics_cache_key = None
        self.latest_metrics = None
        self._vascular_summary_cache = None
        self._mark_microglia_analysis_dirty()

    def _metrics_cache_key_for_render(
        self,
        render: RenderConfig,
    ) -> tuple[int, float, float, int, int, float, float, float]:
        return (
            int(self._metrics_revision),
            float(render.threshold_green),
            float(render.threshold_red),
            int(render.trim_first_slices),
            int(render.trim_last_slices),
            float(render.offset_x_um),
            float(render.offset_y_um),
            float(render.offset_z_um),
        )

    def _set_metrics_text_from_result(self, metrics: MetricsComputation) -> None:
        lines = []
        for item in metrics.channel_results:
            lines.append(
                (
                    f"{item.channel}: voxels={item.voxel_count}, "
                    f"volume_um3={item.volume_um3:.3f}, "
                    f"components={item.component_count}, "
                    f"largest_component={item.largest_component_voxels}"
                )
            )
        lines.append(
            f"overlap: voxels={metrics.overlap_voxel_count}, "
            f"volume_um3={metrics.overlap_volume_um3:.3f}"
        )
        vascular_line = self._vascular_summary_line()
        if vascular_line:
            lines.append(vascular_line)
        self.controls.set_metrics_text("\n".join(lines))

    def _vascular_summary_line(self) -> str | None:
        """Return the cached vascular morphometry line, if current.

        The heavy analyze_vasculature call is performed off the UI thread (in
        the metrics worker) and the formatted line cached; this reader never
        computes synchronously, so it can be called from the main thread without
        risking an AppHang freeze. Returns None until the worker fills the cache.
        """
        if self.processed_dataset is None:
            return None
        cache_key = _vascular_cache_key_for(self.current_render)
        cache = self._vascular_summary_cache
        if (
            cache is not None
            and np.isclose(cache[0][0], cache_key[0], atol=1.0e-6)
            and cache[0][1:] == cache_key[1:]
        ):
            return cache[1]
        return None

    def _invalidate_microglia_components(self) -> None:
        self._green_component_labels = None
        self._green_component_order = np.empty((0,), dtype=np.int32)
        self._green_component_sizes = None
        self._green_component_threshold = None
        self._green_component_branch_sensitivity = None
        self._green_component_shape = None
        self._green_component_sparse = {}
        self._green_component_coloring_active = False
        self._mark_microglia_analysis_dirty()
        self.scene.set_microglia_analysis_debug(None)
        self.scene.set_channel_component_coloring("green", False)
        self.controls.set_microglia_component_summary(0, 0, 0)
        self.controls.microglia_info.setText(
            "Use 'Run Segmentation' or enable 'View one microglia' to detect components."
        )

    def _compute_microglia_components(
        self,
        green: np.ndarray,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        branch_sense = float(self.controls.current_microglia_branch_sensitivity())
        base_min_voxels = max(64, int(self.preprocess_config.green_speckle_min_voxels) * 4)
        spacing_zyx: tuple[float, float, float] | None = None
        if self.visual_dataset is not None:
            spacing = self.visual_dataset.green.spacing
            spacing_zyx = (float(spacing.z_um), float(spacing.y_um), float(spacing.x_um))
        return self._compute_microglia_components_from_params(
            green,
            threshold=threshold,
            branch_sense=branch_sense,
            base_min_voxels=base_min_voxels,
            spacing=spacing_zyx,
        )

    def _compute_microglia_components_from_params(
        self,
        green: np.ndarray,
        *,
        threshold: float,
        branch_sense: float,
        base_min_voxels: int,
        spacing: tuple[float, float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        min_voxels = int(round(base_min_voxels / (0.85 + (0.35 * branch_sense))))
        min_voxels = max(32, min_voxels)
        if green.size >= 120 * 1024 * 1024:
            min_voxels = max(min_voxels, 256)
        labels, order, sizes = compute_component_labels(
            green,
            threshold=threshold,
            min_voxels=min_voxels,
            max_components=256,
            smooth_sigma=(0.2, 0.45, 0.45),
            branch_sensitivity=branch_sense,
            spacing=spacing,
        )
        return filter_components_by_preferred_voxel_floor(labels, order, sizes)

    def _cache_microglia_components(
        self,
        labels: np.ndarray,
        order: np.ndarray,
        sizes: np.ndarray,
        *,
        threshold: float,
        branch_sensitivity: float,
        shape: tuple[int, int, int],
        selected_index: int,
    ) -> None:
        label_count = int(len(order))
        if label_count <= int(np.iinfo(np.uint8).max):
            label_dtype = np.uint8
        elif label_count <= int(np.iinfo(np.uint16).max):
            label_dtype = np.uint16
        else:
            label_dtype = np.uint32
        estimated_bytes = int(np.prod(shape) * np.dtype(label_dtype).itemsize)
        self._green_component_sparse = {}
        if estimated_bytes <= self._microglia_label_cache_max_bytes:
            self._green_component_labels = labels.astype(label_dtype, copy=False)
        else:
            # Avoid keeping a very large dense label volume resident in memory.
            self._green_component_labels = None
            if label_count > 0:
                objects = ndi.find_objects(np.asarray(labels, dtype=np.int32))
                for component_id in order:
                    idx = int(component_id) - 1
                    if idx < 0 or idx >= len(objects):
                        continue
                    comp_slice = objects[idx]
                    if comp_slice is None:
                        continue
                    local_mask = np.asarray(labels[comp_slice] == int(component_id), dtype=bool)
                    if not np.any(local_mask):
                        continue
                    self._green_component_sparse[int(component_id)] = (comp_slice, local_mask)
        self._green_component_order = order
        self._green_component_sizes = sizes
        self._green_component_threshold = threshold
        self._green_component_branch_sensitivity = branch_sensitivity
        self._green_component_shape = shape
        selected = min(selected_index, int(len(order)))
        selected_voxels = int(sizes[int(order[selected - 1])]) if selected > 0 else 0
        self.controls.set_microglia_component_summary(
            count=int(len(order)),
            selected_index=selected,
            selected_voxels=selected_voxels,
        )

    def _start_microglia_refresh_task(self) -> None:
        if self.visual_dataset is None:
            return
        if self._active_thread is not None and self._active_thread.isRunning():
            self.statusBar().showMessage("Another operation is still running.", 3000)
            return

        threshold = float(self.current_render.threshold_green)
        branch_sensitivity = float(self.controls.current_microglia_branch_sensitivity())
        selected_index = int(self.controls.microglia_view_state()[1])
        green = self.visual_dataset.green.data
        if green.dtype != np.float32:
            green = np.asarray(green, dtype=np.float32)
        spacing = self.visual_dataset.green.spacing
        spacing_zyx = (float(spacing.z_um), float(spacing.y_um), float(spacing.x_um))
        base_min_voxels = max(64, int(self.preprocess_config.green_speckle_min_voxels) * 4)
        eta_seconds = self._estimate_microglia_separation_eta_seconds(
            green.shape,
            branch_sensitivity,
        )

        if (
            self._green_component_sizes is not None
            and self._green_component_threshold is not None
            and self._green_component_branch_sensitivity is not None
            and np.isclose(self._green_component_threshold, threshold, atol=1.0e-6)
            and np.isclose(self._green_component_branch_sensitivity, branch_sensitivity, atol=1.0e-6)
            and self._green_component_shape == green.shape
        ):
            self._push_scene_channels(green_only=True)
            self.scene.apply_render_config(self.current_render)
            self._refresh_microglia_analysis_debug()
            return

        def _compute_task() -> _MicrogliaComponentsTaskResult:
            self._publish_busy_progress(percent=10.0, message="Detecting microglia components...")
            labels, order, sizes = self._compute_microglia_components_from_params(
                green,
                threshold=threshold,
                branch_sense=branch_sensitivity,
                base_min_voxels=base_min_voxels,
                spacing=spacing_zyx,
            )
            self._publish_busy_progress(percent=75.0, message="Preparing isolate cache...")
            return _MicrogliaComponentsTaskResult(
                labels=labels,
                order=order,
                sizes=sizes,
                threshold=threshold,
                branch_sensitivity=branch_sensitivity,
                shape=green.shape,
            )

        def _on_success(result: object) -> None:
            if not isinstance(result, _MicrogliaComponentsTaskResult):
                raise TypeError("Invalid microglia task result payload.")
            isolate_enabled, current_selected = self.controls.microglia_view_state()
            if not isolate_enabled:
                return
            self._publish_busy_progress(percent=85.0, message="Updating component cache...")
            self._cache_microglia_components(
                result.labels,
                result.order,
                result.sizes,
                threshold=result.threshold,
                branch_sensitivity=result.branch_sensitivity,
                shape=result.shape,
                selected_index=int(current_selected if current_selected > 0 else selected_index),
            )
            self._publish_busy_progress(percent=92.0, message="Updating 3D view...")
            self._push_scene_channels(green_only=True)
            self.scene.apply_render_config(self.current_render)
            self._refresh_microglia_analysis_debug()
            self._publish_busy_progress(percent=100.0, message="Microglia view updated.")

        self._start_background_task(
            title="Microglia Separation",
            message="Separating microglia components...",
            fn=_compute_task,
            on_success=_on_success,
            error_title="Microglia separation failed",
            success_status="Microglia view updated.",
            eta_total_seconds=eta_seconds,
            eta_kind="microglia-separation",
        )

    def _refresh_microglia_components_if_needed(self) -> None:
        if self.visual_dataset is None:
            self._invalidate_microglia_components()
            return
        isolate_enabled, selected_index = self.controls.microglia_view_state()
        if not isolate_enabled:
            self._invalidate_microglia_components()
            return
        threshold = float(self.current_render.threshold_green)
        branch_sensitivity = float(self.controls.current_microglia_branch_sensitivity())
        green = np.asarray(self.visual_dataset.green.data, dtype=np.float32)
        if (
            self._green_component_sizes is not None
            and self._green_component_threshold is not None
            and self._green_component_branch_sensitivity is not None
            and np.isclose(self._green_component_threshold, threshold, atol=1.0e-6)
            and np.isclose(self._green_component_branch_sensitivity, branch_sensitivity, atol=1.0e-6)
            and self._green_component_shape == green.shape
        ):
            return

        labels, order, sizes = self._compute_microglia_components(green, threshold)
        self._cache_microglia_components(
            labels,
            order,
            sizes,
            threshold=threshold,
            branch_sensitivity=branch_sensitivity,
            shape=green.shape,
            selected_index=selected_index,
        )

    def _current_green_volume_for_view(self) -> np.ndarray:
        if self.visual_dataset is None:
            raise RuntimeError("No visual dataset loaded.")
        base = np.asarray(self.visual_dataset.green.data, dtype=np.float32)
        enabled, selected_index = self.controls.microglia_view_state()
        self._green_component_coloring_active = False
        if not enabled:
            return base

        self._refresh_microglia_components_if_needed()
        if self._green_component_sizes is None:
            return base
        count = int(len(self._green_component_order))
        if selected_index <= 0:
            self.controls.set_microglia_component_summary(
                count=count,
                selected_index=0,
                selected_voxels=0,
            )
            # Preserve the enhanced intensity volume in the "All" view.
            # Rendering label IDs here makes the segmented result look like it
            # lost thresholding / cleanup even though the underlying data did not.
            return base
        if selected_index > int(len(self._green_component_order)):
            return base
        component_id = int(self._green_component_order[selected_index - 1])
        selected_voxels = int(self._green_component_sizes[component_id])
        self.controls.set_microglia_component_summary(
            count=int(len(self._green_component_order)),
            selected_index=selected_index,
            selected_voxels=selected_voxels,
        )
        if self._green_component_labels is not None:
            return isolate_component(base, self._green_component_labels, component_id)

        sparse = self._green_component_sparse.get(component_id)
        if sparse is None:
            return base
        comp_slice, local_mask = sparse
        out = np.zeros_like(base, dtype=np.float32)
        out_region = out[comp_slice]
        base_region = base[comp_slice]
        out_region[local_mask] = base_region[local_mask]
        return out

    def _push_scene_channels(self, *, green_only: bool = False) -> None:
        if self.visual_dataset is None:
            return
        trim_first = int(self.current_render.trim_first_slices)
        trim_last = int(self.current_render.trim_last_slices)
        self._set_busy_message("Uploading green channel to VTK...")
        green_volume = self._apply_render_trim(
            self._current_green_volume_for_view(),
            trim_first,
            trim_last,
        )
        self.scene.set_channel_component_coloring(
            "green",
            self._green_component_coloring_active,
            label_count=int(len(self._green_component_order)),
        )
        self.scene.set_channel_data(
            channel="green",
            volume=green_volume,
            spacing=self._display_spacing(self.visual_dataset.green.spacing),
        )
        if green_only:
            return
        self._set_busy_message("Uploading red channel to VTK...")
        red_volume = self._apply_render_trim(
            self.visual_dataset.red.data,
            trim_first,
            trim_last,
        )
        self.scene.set_channel_data(
            channel="red",
            volume=red_volume,
            spacing=self._display_spacing(self.visual_dataset.red.spacing),
        )

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        total = max(0, int(round(seconds)))
        mins, secs = divmod(total, 60)
        return f"{mins:02d}:{secs:02d}"

    def _compose_busy_label(self, elapsed: float, progress_percent: float) -> str:
        title = (self._busy_title or "NVAP").upper()
        message = self._busy_base_message.strip() or "Working..."
        status_parts = [
            f"Progress {int(round(progress_percent))}%",
            f"Elapsed {self._format_seconds(elapsed)}",
        ]
        with self._busy_progress_lock:
            eta_total = self._busy_eta_total
        if eta_total is not None:
            remaining = max(0.0, float(eta_total) - elapsed)
            status_parts.append(f"ETA {self._format_seconds(remaining)}")
        lines = [
            title,
            message,
            "",
            " / ".join(status_parts),
        ]
        return "\n".join(lines)

    def _publish_busy_progress(
        self,
        percent: float | None = None,
        message: str | None = None,
        eta_total_seconds: float | None = None,
    ) -> None:
        with self._busy_progress_lock:
            if percent is not None:
                self._busy_progress_percent = float(np.clip(percent, 0.0, 100.0))
            if message is not None:
                self._busy_progress_message = message
            if eta_total_seconds is not None:
                self._busy_progress_eta_total = float(max(0.0, eta_total_seconds))

    def _eta_scale_for_kind(self, eta_kind: str | None) -> float:
        kind = (eta_kind or "").strip().lower()
        if kind == "load":
            return float(self._eta_scale_load)
        if kind == "psf":
            return float(self._eta_scale_psf)
        if kind == "microglia-separation":
            return float(self._eta_scale_microglia_separation)
        if kind == "mesh-export":
            return float(self._eta_scale_mesh_export)
        return 1.0

    def _update_eta_scale_for_kind(self, eta_kind: str | None, ratio: float) -> None:
        kind = (eta_kind or "").strip().lower()
        if kind not in {
            "load",
            "psf",
            "microglia-separation",
            "mesh-export",
        }:
            return
        if not np.isfinite(ratio) or ratio <= 0.0:
            return
        current = self._eta_scale_for_kind(kind)
        updated = max(0.45, min(2.75, (0.85 * current) + (0.15 * float(ratio))))
        if kind == "load":
            self._eta_scale_load = updated
        elif kind == "psf":
            self._eta_scale_psf = updated
        elif kind == "microglia-separation":
            self._eta_scale_microglia_separation = updated
        elif kind == "mesh-export":
            self._eta_scale_mesh_export = updated
        self._log_debug(
            f"{kind} ETA calibration updated: ratio={ratio:.2f}, scale={updated:.2f}"
        )

    def _update_busy_eta_estimate(
        self,
        elapsed: float,
        progress_percent: float,
        eta_hint: float | None,
    ) -> None:
        model_total: float | None = None
        if eta_hint is not None and np.isfinite(eta_hint) and eta_hint > 0.0:
            model_total = float(max(eta_hint, elapsed + 0.5))

        observed_total: float | None = None
        if progress_percent >= 3.0 and elapsed >= 0.8:
            observed_total = float(elapsed / max(1.0e-3, progress_percent / 100.0))

        if model_total is None and observed_total is None:
            return
        if model_total is None:
            candidate_total = observed_total
        elif observed_total is None:
            candidate_total = model_total
        else:
            trust_observed = float(np.clip((progress_percent - 5.0) / 70.0, 0.12, 0.85))
            candidate_total = ((1.0 - trust_observed) * model_total) + (
                trust_observed * observed_total
            )

        if candidate_total is None or not np.isfinite(candidate_total):
            return

        previous_total = self._busy_eta_total
        if previous_total is None or not np.isfinite(previous_total):
            smoothed_total = float(candidate_total)
        else:
            responsiveness = float(
                np.clip(0.16 + (0.26 * (progress_percent / 100.0)), 0.16, 0.42)
            )
            smoothed_total = ((1.0 - responsiveness) * float(previous_total)) + (
                responsiveness * float(candidate_total)
            )
            step_ratio = 0.30 if progress_percent < 25.0 else 0.18
            lower = float(previous_total) * (1.0 - step_ratio)
            upper = float(previous_total) * (1.0 + step_ratio)
            smoothed_total = float(np.clip(smoothed_total, lower, upper))

        self._busy_eta_total = max(float(elapsed) + 0.5, float(smoothed_total))

    def _estimate_microglia_separation_eta_seconds(
        self,
        green_shape: tuple[int, int, int],
        branch_sensitivity: float,
    ) -> float | None:
        try:
            total_voxels = int(np.prod(green_shape))
        except Exception:
            return None
        if total_voxels <= 0:
            return None

        branch_factor = float(np.clip(1.0 + (0.24 * (branch_sensitivity - 1.0)), 0.8, 1.35))
        segment_seconds = total_voxels * 4.2e-8
        filtering_seconds = total_voxels * 1.8e-8
        scale = self._eta_scale_for_kind("microglia-separation")
        total = max(3.0, (2.5 + ((segment_seconds + filtering_seconds) * branch_factor)) * scale)
        self._log_info(
            "Estimated microglia separation ETA="
            f"{total:.1f}s (scale={scale:.2f}, voxels={total_voxels}, "
            f"branch_sensitivity={branch_sensitivity:.2f})"
        )
        return total

    def _estimate_mesh_export_eta_seconds(
        self,
        dataset: DatasetVolume,
        mesh_cfg: MeshExportConfig,
    ) -> float | None:
        green_voxels = int(dataset.green.data.size)
        red_voxels = int(dataset.red.data.size)
        total_voxels = green_voxels + red_voxels
        if total_voxels <= 0:
            return None

        smoothing_factor = 1.0 + (
            float(np.clip(mesh_cfg.smooth_iterations, 0, 120)) / 40.0
        ) * 0.35
        decimate_factor = 1.0 + (float(np.clip(mesh_cfg.decimate_fraction, 0.0, 0.95)) * 0.45)
        poisson_factor = 1.35 if mesh_cfg.use_poisson else 1.0
        extract_seconds = total_voxels * 8.0e-8
        post_seconds = total_voxels * 2.5e-8
        scale = self._eta_scale_for_kind("mesh-export")
        total = max(
            8.0,
            (6.0 + ((extract_seconds + post_seconds) * smoothing_factor * decimate_factor * poisson_factor))
            * scale,
        )
        self._log_info(
            "Estimated mesh export ETA="
            f"{total:.1f}s (scale={scale:.2f}, voxels={total_voxels}, "
            f"smooth_iter={mesh_cfg.smooth_iterations}, decimate={mesh_cfg.decimate_fraction:.2f}, "
            f"poisson={mesh_cfg.use_poisson})"
        )
        return total

    def _estimate_load_eta_seconds(
        self,
        root: Path,
        channel_overrides: dict[str, str | Path] | None,
        psf_cfg: PSFConfig,
        preprocess_cfg: PreprocessConfig,
        dataset_signature: str | None = None,
    ) -> float | None:
        try:
            stats = inspect_dataset_stats(root, channel_overrides=channel_overrides)
        except Exception as exc:
            self._log_debug(f"ETA estimation unavailable: {exc}")
            return None

        total_voxels = int(stats.total_full_voxels)
        cache_hit = False
        if dataset_signature:
            cache_key = build_processed_cache_key(
                dataset_signature,
                self.spacing,
                psf_cfg,
                preprocess_config=preprocess_cfg,
            )
            cache_hit = has_processed_cache(cache_key, base_dir=root)
            self._log_debug(f"ETA cache key={cache_key} hit={cache_hit}")

        total_slices = int(stats.green.slice_count + stats.red.slice_count)
        load_seconds = 0.120 * total_slices
        interpolate_seconds = 0.100 * stats.total_missing_slices
        if cache_hit:
            preprocess_seconds = 0.0
            psf_seconds = 0.0
            cache_restore_seconds = total_voxels * 3.0e-8
        else:
            cache_restore_seconds = 0.0
            preprocess_seconds = (
                total_voxels * 2.1e-7 if preprocess_cfg.enabled else 0.0
            )
            psf_seconds = 0.0
            if (not _green_no_psf_mode(preprocess_cfg)) and psf_cfg.enabled and psf_cfg.iterations > 0:
                psf_seconds = total_voxels * psf_cfg.iterations * 1.2e-7
        threshold_seconds = total_voxels * 5.0e-8
        resample_seconds = (
            total_voxels * 2.0e-8 if preprocess_cfg.resample_for_mesh else 0.0
        )
        render_seconds = 8.0
        total = (
            load_seconds
            + interpolate_seconds
            + preprocess_seconds
            + psf_seconds
            + threshold_seconds
            + resample_seconds
            + cache_restore_seconds
            + render_seconds
        )
        scale = self._eta_scale_for_kind("load")
        total *= scale
        total = max(5.0, total)
        self._log_info(
            "Estimated load ETA="
            f"{total:.1f}s (scale={scale:.2f}, "
            f"slices={stats.green.slice_count}/{stats.red.slice_count}, "
            f"iterations={psf_cfg.iterations}, cache_hit={cache_hit})"
        )
        return total

    def _estimate_psf_eta_seconds(
        self,
        dataset: DatasetVolume,
        psf_cfg: PSFConfig,
        preprocess_cfg: PreprocessConfig,
        dataset_signature: str | None = None,
    ) -> float | None:
        scale = self._eta_scale_for_kind("psf")
        if _green_no_psf_mode(preprocess_cfg) or not psf_cfg.enabled or psf_cfg.iterations <= 0:
            total_voxels = int(dataset.green.data.size + dataset.red.data.size)
            preprocess_seconds = total_voxels * 1.8e-7
            threshold_seconds = total_voxels * 5.0e-8
            resample_seconds = total_voxels * 2.0e-8 if preprocess_cfg.resample_for_mesh else 0.0
            total = (preprocess_seconds + threshold_seconds + resample_seconds + 8.0) * scale
            total = max(8.0, total)
            self._log_info(
                "Estimated reprocess ETA="
                f"{total:.1f}s (scale={scale:.2f}, mode={preprocess_cfg.green_denoise_strategy})"
            )
            return total
        cache_hit = False
        if dataset_signature:
            cache_key = build_processed_cache_key(
                dataset_signature,
                self.spacing,
                psf_cfg,
                preprocess_config=preprocess_cfg,
            )
            cache_hit = has_processed_cache(cache_key, base_dir=self._project_cache_base_dir())
        if cache_hit:
            self._log_info("Estimated PSF ETA=6.0s (cache hit).")
            return 6.0
        total_voxels = int(dataset.green.data.size + dataset.red.data.size)
        preprocess_seconds = total_voxels * 1.3e-7 if preprocess_cfg.enabled else 0.0
        psf_seconds = total_voxels * psf_cfg.iterations * 1.2e-7
        threshold_seconds = total_voxels * 5.0e-8
        resample_seconds = total_voxels * 2.0e-8 if preprocess_cfg.resample_for_mesh else 0.0
        total = (preprocess_seconds + psf_seconds + threshold_seconds + resample_seconds + 6.0) * scale
        total = max(6.0, total)
        self._log_info(
            "Estimated PSF ETA="
            f"{total:.1f}s (scale={scale:.2f}, "
            f"iterations={psf_cfg.iterations}, cache_hit={cache_hit})"
        )
        return total

    def _detect_measured_psf_path(self) -> str:
        if self.dataset_root is None:
            return ""
        candidates = [
            self.dataset_root / "psf.npy",
            self.dataset_root / "psf.npz",
            self.dataset_root / "psf.tif",
            self.dataset_root / "psf.tiff",
            self.dataset_root / "Input" / "psf.npy",
            self.dataset_root / "Input" / "psf.npz",
            self.dataset_root / "Input" / "psf.tif",
            self.dataset_root / "Input" / "psf.tiff",
        ]
        for path in candidates:
            if path.exists():
                self._log_info(f"Detected measured PSF file: {path}")
                return str(path.resolve())
        return ""

    def _effective_psf_config(self, config: PSFConfig) -> PSFConfig:
        if config.measured_psf_path.strip():
            return config
        detected = self._detect_measured_psf_path()
        if detected:
            return replace(config, measured_psf_path=detected, use_measured_psf=True)
        return config

    def _on_busy_tick(self) -> None:
        if self._busy_dialog is None:
            return
        with self._busy_progress_lock:
            progress_percent = float(np.clip(self._busy_progress_percent, 0.0, 100.0))
            message = self._busy_progress_message
            eta_hint = self._busy_progress_eta_total
        if message:
            self._busy_base_message = message
        elapsed = time.perf_counter() - self._busy_start
        self._update_busy_eta_estimate(elapsed, progress_percent, eta_hint)
        self._busy_dialog.setValue(int(round(progress_percent)))
        self._busy_dialog.setLabelText(self._compose_busy_label(elapsed, progress_percent))

    def _show_busy(
        self,
        title: str,
        message: str,
        eta_total_seconds: float | None = None,
        allow_cancel: bool = False,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        if self._busy_dialog is None:
            self._busy_dialog = QProgressDialog(message, "", 0, 100, self)
            self._busy_dialog.setObjectName("busyDialog")
            self._busy_dialog.setMinimumDuration(0)
            self._busy_dialog.setWindowModality(Qt.WindowModal)
            self._busy_dialog.setAutoClose(False)
            self._busy_dialog.setAutoReset(False)
            self._busy_dialog.setMinimumWidth(420)
            self._busy_dialog.setMinimumHeight(180)
        else:
            self._busy_dialog.setRange(0, 100)
        try:
            self._busy_dialog.canceled.disconnect()
        except Exception:
            pass
        if allow_cancel:
            self._busy_dialog.setCancelButtonText("Cancel")
            if on_cancel is not None:
                self._busy_dialog.canceled.connect(on_cancel)
        else:
            self._busy_dialog.setCancelButton(None)
        self._busy_title = title
        self._busy_base_message = message
        initial_eta = None
        if eta_total_seconds is not None and np.isfinite(eta_total_seconds):
            initial_eta = float(max(0.0, eta_total_seconds))
        self._busy_eta_total = initial_eta
        self._busy_start = time.perf_counter()
        with self._busy_progress_lock:
            self._busy_progress_percent = 0.0
            self._busy_progress_message = message
            self._busy_progress_eta_total = initial_eta
        self._busy_dialog.setWindowTitle(title)
        self._busy_dialog.setValue(0)
        self._busy_dialog.setLabelText(self._compose_busy_label(0.0, 0.0))
        self._busy_dialog.show()
        self._busy_timer.start()
        QApplication.processEvents()
        self._log_info(f"{title} - started")

    def _hide_busy(self) -> None:
        elapsed = time.perf_counter() - self._busy_start
        finished_title = self._busy_title
        self._busy_timer.stop()
        if self._busy_dialog is not None:
            self._busy_dialog.close()
            self._busy_dialog = None
        self._busy_eta_total = None
        self._busy_title = ""
        self._busy_base_message = ""
        with self._busy_progress_lock:
            self._busy_progress_percent = 0.0
            self._busy_progress_message = ""
            self._busy_progress_eta_total = None
        QApplication.processEvents()
        if finished_title:
            self._log_info(f"{finished_title} - finished in {elapsed:.2f}s")

    @contextmanager
    def _busy(self, title: str, message: str):
        self._show_busy(title, message)
        try:
            yield
        finally:
            self._hide_busy()

    def _set_busy_message(self, message: str) -> None:
        self._publish_busy_progress(message=message)
        if self._busy_dialog is not None:
            with self._busy_progress_lock:
                progress_percent = float(np.clip(self._busy_progress_percent, 0.0, 100.0))
            elapsed = time.perf_counter() - self._busy_start
            self._busy_dialog.setValue(int(round(progress_percent)))
            self._busy_dialog.setLabelText(self._compose_busy_label(elapsed, progress_percent))
            QApplication.processEvents()
        self._log_debug(message)

    def _set_stage_progress(
        self,
        start_percent: float,
        end_percent: float,
        fraction: float,
        message: str | None = None,
    ) -> None:
        frac = float(np.clip(fraction, 0.0, 1.0))
        start = float(start_percent)
        end = float(end_percent)
        percent = start + ((end - start) * frac)
        self._publish_busy_progress(percent=percent, message=message)

    def _start_background_task(
        self,
        title: str,
        message: str,
        fn: Callable[[], object],
        on_success: Callable[[object], None],
        error_title: str,
        success_status: str | None = None,
        eta_total_seconds: float | None = None,
        eta_kind: str | None = None,
        allow_cancel: bool = False,
        cancel_event: threading.Event | None = None,
        canceled_status: str | None = None,
    ) -> None:
        if self._active_thread is not None and self._active_thread.isRunning():
            QMessageBox.warning(self, "Task Running", "Another operation is still running.")
            return

        if allow_cancel and cancel_event is None:
            cancel_event = threading.Event()

        def on_cancel_request() -> None:
            if cancel_event is not None:
                cancel_event.set()
            self._publish_busy_progress(message="Cancel requested. Finishing current iteration...")
            self._set_busy_message("Cancel requested. Finishing current iteration...")
            self._log_info("Cancellation requested by user.")

        self.controls.setEnabled(False)
        self._show_busy(
            title,
            message,
            eta_total_seconds=eta_total_seconds,
            allow_cancel=allow_cancel,
            on_cancel=on_cancel_request if allow_cancel else None,
        )
        self._publish_busy_progress(percent=0.0, message=message, eta_total_seconds=eta_total_seconds)
        thread = _FunctionThread(fn, self)
        self._active_thread = thread

        def cleanup() -> None:
            elapsed = time.perf_counter() - self._busy_start
            if eta_total_seconds is not None and eta_total_seconds > 0:
                ratio = elapsed / eta_total_seconds
                self._update_eta_scale_for_kind(eta_kind, ratio)
            self.controls.setEnabled(True)
            self._publish_busy_progress(percent=100.0)
            self._hide_busy()
            if self._active_thread is thread:
                self._active_thread = None
            thread.deleteLater()

        def handle_success(result: object) -> None:
            try:
                on_success(result)
                if success_status:
                    self.statusBar().showMessage(success_status, 5000)
            except Exception as exc:
                self._show_error(error_title, str(exc))
            finally:
                cleanup()

        def handle_error(error_text: str) -> None:
            logger.error("Background task error:\n%s", error_text)
            if "OperationCanceledError" in error_text:
                if canceled_status:
                    self.statusBar().showMessage(canceled_status, 5000)
                self._log_info("Background task canceled.")
                cleanup()
                return
            concise = error_text.strip().splitlines()[-1] if error_text.strip() else "Unknown error"
            self._show_error(error_title, concise)
            cleanup()

        thread.result_ready.connect(handle_success)
        thread.error_raised.connect(handle_error)
        thread.start()

    def _refresh_plugin_panel(self) -> None:
        self._log_debug("Discovering plugins from entry point group 'nvap.plugins'.")
        plugins = discover_plugins()
        if not plugins:
            self.controls.set_plugin_text("No plugins discovered in 'nvap.plugins'.")
            self._log_info("No plugins discovered.")
            self._refresh_section_state()
            return
        lines = []
        for plugin in plugins:
            if plugin.status == "loaded":
                lines.append(f"- {plugin.plugin_id} ({plugin.target_channel}) loaded")
            else:
                lines.append(f"- {plugin.plugin_id} error: {plugin.error}")
        self._log_info(f"Discovered {len(plugins)} plugin descriptor(s).")
        self.controls.set_plugin_text("\n".join(lines))
        self._refresh_section_state()

    def _prompt_channel_source(self, channel_label: str, start_dir: Path) -> str | None:
        chooser = QMessageBox(self)
        chooser.setWindowTitle("Select Channel Source Type")
        chooser.setText(f"Choose source type for {channel_label}.")
        chooser.setInformativeText(
            "Use a single TIFF/PNG stack file, or a folder of sequenced images."
        )
        file_btn = chooser.addButton("Single Stack File", QMessageBox.ButtonRole.AcceptRole)
        chooser.addButton("Image Sequence Folder", QMessageBox.ButtonRole.ActionRole)
        cancel_btn = chooser.addButton(QMessageBox.StandardButton.Cancel)
        chooser.exec()

        clicked = chooser.clickedButton()
        if clicked is None or clicked == cancel_btn:
            return None
        if clicked == file_btn:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                f"Select {channel_label} Stack File",
                str(start_dir),
                "Image files (*.tif *.tiff *.png);;All files (*.*)",
            )
            return file_path or None

        folder_path = QFileDialog.getExistingDirectory(
            self,
            f"Select {channel_label} Image Sequence Folder",
            str(start_dir),
        )
        return folder_path or None

    def _prompt_channel_sources_in_order(self, start_dir: Path) -> dict[str, str] | None:
        QMessageBox.information(
            self,
            "Select Image Sequences",
            (
                "Select channels in this order:\n\n"
                "1. Vasculature (Red)\n"
                "2. Microglia (Green)\n\n"
                "Each channel can be either:\n"
                "- a single TIFF/PNG stack file, or\n"
                "- a folder of sequenced images."
            ),
        )

        red_source = self._prompt_channel_source("Vasculature (Red)", start_dir)
        if not red_source:
            return None

        red_path = Path(red_source).resolve()
        green_start = red_path.parent if red_path.is_file() else red_path
        green_source = self._prompt_channel_source("Microglia (Green)", green_start)
        if not green_source:
            return None

        return {"red": red_source, "green": green_source}

    def _prompt_load_source_mode(self) -> str | None:
        chooser = QMessageBox(self)
        chooser.setWindowTitle("Open Dataset")
        chooser.setText("Choose how to load images.")
        chooser.setInformativeText(
            "Open one auto-detected dataset folder, a parent folder containing multiple datasets, "
            "or select individual red/green TIFF files or sequence folders."
        )
        folder_btn = chooser.addButton("Folder Process", QMessageBox.ButtonRole.AcceptRole)
        set_btn = chooser.addButton("Project Set Folder", QMessageBox.ButtonRole.ActionRole)
        manual_btn = chooser.addButton("Individual TIFF/Sequence", QMessageBox.ButtonRole.ActionRole)
        cancel_btn = chooser.addButton(QMessageBox.StandardButton.Cancel)
        chooser.exec()

        clicked = chooser.clickedButton()
        if clicked is None or clicked == cancel_btn:
            return None
        if clicked == set_btn:
            return "project_set"
        if clicked == manual_btn:
            return "manual"
        if clicked == folder_btn:
            return "folder"
        return None

    def _start_dataset_load(
        self,
        root: Path,
        *,
        channel_overrides: dict[str, str | Path] | None,
        channel_dirs: dict[str, Path],
        load_mode: str,
    ) -> None:
        root = root.resolve()
        self.dataset_root = root
        self._current_load_mode = str(load_mode or "folder")
        self._current_channel_sources = {
            "green": str(channel_dirs["green"].resolve()),
            "red": str(channel_dirs["red"].resolve()),
        }
        self._dataset_signature = build_dataset_signature(channel_dirs)
        self._log_debug(f"Dataset signature set: {self._dataset_signature}")

        preprocess_cfg = self.controls.current_preprocess_config()
        self.preprocess_config = preprocess_cfg
        psf_cfg = self._effective_psf_config(self.current_psf)
        self.current_psf = psf_cfg
        eta_seconds = self._estimate_load_eta_seconds(
            root,
            channel_overrides,
            psf_cfg,
            preprocess_cfg,
            dataset_signature=self._dataset_signature,
        )
        self._refresh_section_state()
        self._start_background_task(
            title="Load Dataset",
            message="Loading stacks and processing...",
            fn=lambda: self._background_load_dataset(
                root,
                channel_overrides,
                psf_cfg,
                preprocess_cfg,
            ),
            on_success=self._on_load_task_success,
            error_title="Dataset load failed",
            success_status=f"Loaded dataset: {root}",
            eta_total_seconds=eta_seconds,
            eta_kind="load",
        )

    def _on_load_requested(self) -> None:
        base = self.dataset_root or (Path.cwd() / "Input")
        load_mode = self._prompt_load_source_mode()
        if load_mode is None:
            return
        channel_overrides: dict[str, str] | None = None
        channel_dirs: dict[str, Path]

        if load_mode == "project_set":
            selected = QFileDialog.getExistingDirectory(
                self,
                "Select Project Set Folder",
                str(base),
            )
            if not selected:
                return
            root = Path(selected).resolve()
            entries = discover_dataset_projects(root)
            if not entries:
                self._show_error(
                    "No Datasets Found",
                    (
                        "NVAP could not find any loadable red/green image-series datasets "
                        f"inside:\n{root}"
                    ),
                )
                return
            self._log_info(
                f"Project set discovered: root={root} datasets={len(entries)}"
            )
            self.statusBar().showMessage(
                f"Project set loaded with {len(entries)} dataset(s). Opening first dataset...",
                8000,
            )
            self._set_project_set(root, entries, selected_index=0)
            return

        self._clear_project_set()

        if load_mode == "folder":
            selected = QFileDialog.getExistingDirectory(
                self,
                "Select Dataset Folder",
                str(base),
            )
            if not selected:
                return
            root = Path(selected).resolve()
            try:
                channel_dirs = resolve_channel_dirs(root)
                self.dataset_root = root
                self._log_dataset_detection(root, channel_dirs)
            except FileNotFoundError as exc:
                self._log_info(f"Auto-detection failed for {root}: {exc}")
                use_manual = QMessageBox.question(
                    self,
                    "Dataset Channels Not Found",
                    (
                        "NVAP could not auto-detect red/green channels in the selected folder.\n\n"
                        "Select individual red/green TIFF files or sequence folders instead?"
                    ),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if use_manual != QMessageBox.StandardButton.Yes:
                    return
                load_mode = "manual"
                base = root

        if load_mode == "manual":
            channel_overrides = self._prompt_channel_sources_in_order(base)
            if channel_overrides is None:
                return
            first_red = Path(channel_overrides["red"]).resolve()
            self.dataset_root = first_red.parent if first_red.is_file() else first_red
            self._log_info(
                f"Manual channel sources selected: red={Path(channel_overrides['red']).resolve()} "
                f"green={Path(channel_overrides['green']).resolve()}"
            )
            channel_dirs = resolve_channel_dirs(self.dataset_root, channel_overrides=channel_overrides)
            self._log_dataset_detection(self.dataset_root, channel_dirs, manual=True)

        root = self.dataset_root
        if root is None:
            return
        self._start_dataset_load(
            root,
            channel_overrides=channel_overrides,
            channel_dirs=channel_dirs,
            load_mode=load_mode,
        )

    def _log_dataset_detection(
        self,
        root: Path,
        channel_dirs: dict[str, Path],
        *,
        manual: bool = False,
    ) -> None:
        mode = "Manual" if manual else "Auto"
        self._log_info(
            f"{mode} dataset sources: root={root} "
            f"green={channel_dirs['green']} red={channel_dirs['red']}"
        )
        try:
            stats = inspect_dataset_stats(root, channel_overrides=channel_dirs)
        except Exception as exc:
            self._log_info(f"{mode} dataset stats unavailable: {exc}")
            self.statusBar().showMessage(f"{mode}-detected dataset sources.", 5000)
            return
        summary = (
            f"{mode}-detected dataset: green/microglia {stats.green.slice_count} slice(s), "
            f"red/vasculature {stats.red.slice_count} slice(s), "
            f"shared z {stats.shared_z_range[0]}-{stats.shared_z_range[1]}"
        )
        self._log_info(summary)
        self.statusBar().showMessage(summary, 8000)

    def _background_load_dataset(
        self,
        root: Path,
        channel_overrides: dict[str, str | Path] | None,
        psf_cfg: PSFConfig,
        preprocess_cfg: PreprocessConfig,
    ) -> _LoadTaskResult:
        t0 = time.perf_counter()
        self._publish_busy_progress(percent=2.0, message="Reading channel stacks...")
        self._log_info("Load step 1/5: reading channel stacks...")
        dataset = load_dataset(root, spacing=self.spacing, channel_overrides=channel_overrides)
        self._log_info("Load step 1/5 complete.")
        self._publish_busy_progress(percent=16.0, message="Synchronizing channels...")

        self._log_info("Load step 2/5: filling missing slices...")
        synced_dataset = fill_and_sync_dataset(dataset)
        self._log_info("Load step 2/5 complete.")
        self._publish_busy_progress(percent=26.0, message="Resolving processed cache and pipeline...")

        self._log_info("Load step 3/5: resolving processed cache and pipeline...")
        self.preprocess_config = preprocess_cfg
        processed_dataset = self._get_processed_dataset_with_cache(
            synced_dataset,
            psf_cfg,
            preprocess_cfg,
            self._dataset_signature,
            cancel_event=None,
            progress_bounds=(26.0, 74.0),
        )
        self._log_info("Load step 3/5 complete.")
        self._publish_busy_progress(percent=76.0, message="Preparing mesh dataset...")

        enhancement_method: str | None = None
        saved_state = load_project_state(root)
        saved_method = str(saved_state.get("microglia_enhancement_method") or "") if saved_state else ""
        if saved_method and self._last_processed_cache_key:
            enhanced_green = load_enhanced_green(
                self._last_processed_cache_key,
                saved_method,
                base_dir=self._project_cache_base_dir(),
            )
            if enhanced_green is not None and enhanced_green.shape == processed_dataset.green.data.shape:
                processed_dataset = DatasetVolume(
                    green=ChannelVolume(
                        name="green",
                        data=enhanced_green,
                        z_indices=list(processed_dataset.green.z_indices),
                        spacing=processed_dataset.green.spacing,
                    ),
                    red=processed_dataset.red,
                    shared_z_range=processed_dataset.shared_z_range,
                )
                enhancement_method = saved_method
                self._log_info(f"Restored microglia enhancement '{saved_method}' from cache.")

        self._log_info("Load step 4/5: preparing mesh dataset...")
        visual_dataset = prepare_dataset_for_mesh(processed_dataset, preprocess_cfg)
        self._log_info("Load step 4/5 complete.")
        self._publish_busy_progress(percent=86.0, message="Computing default thresholds...")

        self._log_info("Load step 5/5: computing thresholds...")
        cached_thresholds = (
            load_processed_thresholds(
                self._last_processed_cache_key,
                base_dir=self._project_cache_base_dir(),
            )
            if self._last_processed_cache_key
            else None
        )
        if cached_thresholds is not None and enhancement_method is None:
            threshold_green, threshold_red = cached_thresholds
            self._log_info(
                "Load step 5/5: using cached thresholds "
                f"(green={threshold_green:.4f}, red={threshold_red:.4f})."
            )
        else:
            threshold_green = _DEFAULT_GREEN_THRESHOLD
            threshold_red = cached_thresholds[1] if cached_thresholds is not None else _DEFAULT_RED_THRESHOLD
            if self._last_processed_cache_key and enhancement_method is None:
                try:
                    save_processed_metadata(
                        self._last_processed_cache_key,
                        threshold_green=threshold_green,
                        threshold_red=threshold_red,
                        base_dir=self._project_cache_base_dir(),
                    )
                except OSError as exc:
                    self._log_info(f"Processed metadata save failed: {exc}")
        self._publish_busy_progress(percent=92.0, message="Finalizing load...")
        self._log_info(
            "Load complete. "
            f"thresholds=(green={threshold_green:.4f}, red={threshold_red:.4f}) "
            f"total_dt={time.perf_counter() - t0:.2f}s"
        )
        return _LoadTaskResult(
            synced_dataset=synced_dataset,
            raw_dataset=synced_dataset,
            processed_dataset=processed_dataset,
            visual_dataset=visual_dataset,
            threshold_green=threshold_green,
            threshold_red=threshold_red,
            enhancement_method=enhancement_method,
        )

    def _on_load_task_success(self, result: object) -> None:
        if not isinstance(result, _LoadTaskResult):
            raise TypeError("Invalid load task result payload.")
        self._publish_busy_progress(percent=93.0, message="Uploading channels to VTK...")
        self._microglia_isolate_active = bool(self.controls.microglia_view_state()[0])
        self.synced_dataset = result.synced_dataset
        self.raw_dataset = result.raw_dataset
        self.processed_dataset = result.processed_dataset
        self._mark_metrics_dirty()
        self._current_microglia_enhancement_method = result.enhancement_method
        self._current_speck_wipe_applied = False
        if self._current_microglia_enhancement_method:
            idx = self.controls.microglia_enhancement_method.findData(self._current_microglia_enhancement_method)
            if idx >= 0:
                self.controls.microglia_enhancement_method.setCurrentIndex(idx)

        self.visual_dataset = result.visual_dataset
        self._invalidate_microglia_components()
        self._push_scene_channels()
        self._publish_busy_progress(percent=97.0, message="Applying initial thresholds...")
        self._set_busy_message("Applying initial thresholds...")

        # Block signals so set_threshold_defaults doesn't trigger _refresh_metrics
        # synchronously and freeze the progress dialog at 97%.
        self.controls.blockSignals(True)
        self.controls.set_threshold_defaults(result.threshold_green, result.threshold_red)
        self.controls.set_microglia_enhancement_enabled(True)
        self.controls.blockSignals(False)
        self.current_render = self.controls.current_render_config()
        self.scene.apply_render_config(self.current_render)
        self._publish_busy_progress(percent=100.0, message="Load complete.")
        self._set_busy_message("Load complete.")
        self._log_info("Dataset load and initial render completed.")
        # Navigate to workspace and record in recent projects
        self._nav_to(1)
        self._save_current_project_state()
        self._record_current_project(samples=1, status="Open")
        self._refresh_section_state()
        # After the busy dialog closes, run the automatic on-load pipeline (clean
        # enhancement + speck wipe) or, if both are disabled, just compute metrics.
        # Deferring via a zero timer lets the load task's cleanup clear
        # _active_thread first, so the pipeline's "another operation running" guard
        # does not block it. This runs for single datasets and for every dataset
        # in a multi-stack project set, since both paths land here.
        QTimer.singleShot(0, self._run_auto_load_pipeline)

    def _run_auto_load_pipeline(self) -> None:
        """Automatically clean each freshly loaded dataset.

        Runs the selected microglia enhancement (unless one was already applied
        from cache) followed by the speck wipe, in a single background task, then
        computes metrics. Each step is gated by its own on-load toggle so the user
        can disable them before loading. With everything off this just refreshes
        metrics, so behaviour is unchanged.
        """
        if self.processed_dataset is None:
            self._refresh_metrics()
            return

        want_enhance = (
            self.controls.auto_enhance_microglia_on_load_enabled()
            and not self._current_microglia_enhancement_method
        )
        want_wipe = self.controls.auto_wipe_specks_on_load_enabled()
        if not want_enhance and not want_wipe:
            self._refresh_metrics()
            return

        dataset = self.processed_dataset
        preprocess_cfg = self.preprocess_config
        method = self.controls.current_microglia_enhancement_method()
        max_voxels = int(self.controls.current_wipe_speck_max_voxels())
        threshold_green = float(self.current_render.threshold_green)
        threshold_red = float(self.current_render.threshold_red)
        steps = [name for name, on in (("enhance", want_enhance), ("wipe", want_wipe)) if on]
        self._log_info(
            "Auto on-load pipeline: "
            f"{' + '.join(steps)} (method={method if want_enhance else '-'}, "
            f"speck<{max_voxels} vox)."
        )

        def _report_auto_enhance_progress(completed: int, total: int) -> None:
            if total <= 0:
                return
            fraction = min(1.0, completed / total)
            self._publish_busy_progress(
                percent=10.0 + (fraction * 50.0),
                message=f"Enhancing microglia ({completed}/{total} slices)...",
            )

        def _auto_task() -> DatasetVolume:
            green = np.asarray(dataset.green.data, dtype=np.float32)
            red = np.asarray(dataset.red.data, dtype=np.float32)
            if want_enhance:
                self._publish_busy_progress(percent=10.0, message="Enhancing microglia (clean)...")
                green = enhance_microglia_background(
                    green,
                    preprocess_cfg,
                    method=method,
                    progress_callback=_report_auto_enhance_progress,
                )
                # Persist so the next time this project opens, the restore path in
                # _background_load_dataset finds a real cache file instead of
                # silently missing and re-running the enhancement from scratch.
                if self._last_processed_cache_key:
                    try:
                        save_enhanced_dataset(
                            self._last_processed_cache_key,
                            method,
                            green,
                            base_dir=self._project_cache_base_dir(),
                        )
                    except OSError as exc:
                        self._log_info(f"Enhanced cache save failed: {exc}")
            if want_wipe:
                self._publish_busy_progress(percent=60.0, message="Wiping microglia specks...")
                green = wipe_small_specks(green, threshold=threshold_green, min_voxels=max_voxels)
                self._publish_busy_progress(percent=80.0, message="Wiping vasculature specks...")
                red = wipe_small_specks(red, threshold=threshold_red, min_voxels=max_voxels)
            self._publish_busy_progress(percent=92.0, message="Updating dataset...")
            return DatasetVolume(
                green=ChannelVolume(
                    name="green",
                    data=green,
                    z_indices=list(dataset.green.z_indices),
                    spacing=dataset.green.spacing,
                ),
                red=ChannelVolume(
                    name="red",
                    data=red,
                    z_indices=list(dataset.red.z_indices),
                    spacing=dataset.red.spacing,
                ),
                shared_z_range=dataset.shared_z_range,
            )

        def _on_auto_success(result: object) -> None:
            if want_enhance:
                self._current_microglia_enhancement_method = method
            self._on_wipe_specks_success(result)
            self._current_speck_wipe_applied = bool(want_wipe)

        self._start_background_task(
            title="Auto-processing dataset",
            message="Cleaning microglia and vasculature...",
            fn=_auto_task,
            on_success=_on_auto_success,
            error_title="Automatic on-load processing failed",
            success_status="Automatic cleanup complete.",
        )

    def _on_psf_task_success(self, result: object) -> None:
        if not isinstance(result, DatasetVolume):
            raise TypeError("Invalid processing task result payload.")
        self._publish_busy_progress(percent=90.0, message="Preparing render dataset...")
        self._microglia_isolate_active = bool(self.controls.microglia_view_state()[0])
        self.processed_dataset = result
        self._mark_metrics_dirty()
        self.visual_dataset = prepare_dataset_for_mesh(self.processed_dataset, self.preprocess_config)
        self._publish_busy_progress(percent=94.0, message="Uploading channels to VTK...")
        self._invalidate_microglia_components()
        self._push_scene_channels()
        self._publish_busy_progress(percent=98.0, message="Refreshing render...")
        self._set_busy_message("Refreshing render...")
        self.controls.set_microglia_enhancement_enabled(True)
        self.scene.apply_render_config(self.current_render)
        self._refresh_microglia_analysis_debug()
        self._publish_busy_progress(percent=100.0, message="Processing complete.")
        self._set_busy_message("Processing complete.")
        self._log_info("Processing applied and scene refreshed.")
        self._save_current_project_state()
        self._record_current_project(samples=1, status="Open")
        self._refresh_section_state()
        QTimer.singleShot(0, self._refresh_metrics)

    def _get_processed_dataset_with_cache(
        self,
        synced_dataset: DatasetVolume,
        psf_cfg: PSFConfig,
        preprocess_cfg: PreprocessConfig,
        dataset_signature: str | None,
        cancel_event: threading.Event | None,
        progress_bounds: tuple[float, float] | None = None,
    ) -> DatasetVolume:
        progress_start = float(progress_bounds[0]) if progress_bounds is not None else 8.0
        progress_end = float(progress_bounds[1]) if progress_bounds is not None else 86.0
        progress_end = max(progress_start + 1.0, progress_end)

        def publish_process_progress(fraction: float, message: str | None = None) -> None:
            self._set_stage_progress(progress_start, progress_end, fraction, message=message)

        publish_process_progress(0.0, "Checking processed cache...")
        cache_key = None
        if dataset_signature:
            cache_key = build_processed_cache_key(
                dataset_signature,
                self.spacing,
                psf_cfg,
                preprocess_config=preprocess_cfg,
            )
            self._log_info(f"Cache pipeline: lookup key={cache_key}")
            self._last_processed_cache_key = cache_key
            cached = load_processed_dataset(
                cache_key,
                self.spacing,
                base_dir=self._project_cache_base_dir(),
            )
            if cached is not None:
                publish_process_progress(1.0, "Loaded processed dataset from cache.")
                self._log_info("Using cached processed dataset.")
                return cached
            self._log_info(f"Cache pipeline: miss key={cache_key}")
        working_dataset = synced_dataset
        if preprocess_cfg.enabled:
            publish_process_progress(0.08, "Running preprocessing...")
            self._log_info(
                "Cache pipeline: preprocessing start "
                f"(strategy={preprocess_cfg.green_denoise_strategy})."
            )
            t_pre = time.perf_counter()
            working_dataset = preprocess_dataset(synced_dataset, preprocess_cfg)
            self._log_info(
                f"Cache pipeline: preprocessing complete dt={time.perf_counter() - t_pre:.2f}s"
            )
        else:
            self._log_info("Cache pipeline: preprocessing skipped (disabled).")
        publish_process_progress(0.18, "Running processing pipeline...")

        self._log_info(
            "Cache pipeline: PSF start "
            f"(enabled={psf_cfg.enabled} iterations={psf_cfg.iterations})."
        )
        t_psf = time.perf_counter()

        progress_seen: dict[str, int] = {"green": 0, "red": 0}
        channel_progress: dict[str, float] = {"green": 0.0, "red": 0.0}
        progress_lock = threading.Lock()

        def on_psf_progress(channel: str, current: int, total: int) -> None:
            if total <= 0:
                return
            with progress_lock:
                frac = float(np.clip(current / total, 0.0, 1.0))
                channel_progress[channel] = frac
                avg_frac = 0.5 * (channel_progress.get("green", 0.0) + channel_progress.get("red", 0.0))
                percent = int((100 * current) / total)
                last = progress_seen.get(channel, -1)
                emit_log = percent >= 100 or percent >= (last + 10)
                if emit_log:
                    progress_seen[channel] = percent
            publish_process_progress(
                0.22 + (0.68 * avg_frac),
                message=f"Processing {channel}: {current}/{total}",
            )
            if emit_log:
                self._log_info(f"Load progress: PSF {channel} {current}/{total} ({percent}%)")

        processed = apply_psf_to_dataset(
            working_dataset,
            psf_cfg,
            preprocess_config=preprocess_cfg,
            cancel_event=cancel_event,
            progress_callback=on_psf_progress,
        )
        publish_process_progress(0.92, "Finalizing processed dataset...")
        self._log_info(f"Cache pipeline: PSF complete dt={time.perf_counter() - t_psf:.2f}s")
        if cache_key is not None and (cancel_event is None or not cancel_event.is_set()):
            publish_process_progress(0.96, "Saving processed dataset to cache...")
            self._log_info("Cache pipeline: saving processed dataset to cache.")
            save_processed_dataset(
                cache_key,
                processed,
                base_dir=self._project_cache_base_dir(),
            )
            self._log_info(f"Cache pipeline: cache save complete key={cache_key}")
        publish_process_progress(1.0, "Processed dataset ready.")
        return processed

    def _on_psf_config_changed(self, config: PSFConfig) -> None:
        self.current_psf = config
        if config.iterations >= 8:
            self.statusBar().showMessage(
                "High RL iterations can take several minutes on large stacks.",
                4000,
            )

    def _on_apply_psf_requested(self) -> None:
        if self.synced_dataset is None:
            self._show_error("No dataset", "Load a dataset before reprocessing.")
            return
        synced_dataset = self.synced_dataset
        self.preprocess_config = self.controls.current_preprocess_config()
        preprocess_cfg = self.preprocess_config
        psf_cfg = self._effective_psf_config(self.current_psf)
        self.current_psf = psf_cfg
        dataset_signature = self._dataset_signature
        cancel_event = threading.Event()
        eta_seconds = self._estimate_psf_eta_seconds(
            synced_dataset,
            psf_cfg,
            preprocess_cfg,
            dataset_signature=dataset_signature,
        )
        no_psf_mode = _green_no_psf_mode(preprocess_cfg)

        def _reprocess_task():
            return self._get_processed_dataset_with_cache(
                synced_dataset,
                psf_cfg,
                preprocess_cfg,
                dataset_signature,
                cancel_event=cancel_event,
                progress_bounds=(8.0, 88.0),
            )

        self._start_background_task(
            title="Reprocess Dataset",
            message=(
                "Applying pass-through pipeline (green unchanged, red processed)..."
                if no_psf_mode
                else (
                    f"Running Richardson-Lucy (iterations={psf_cfg.iterations})...\n"
                    "This can take several minutes for large volumes."
                )
            ),
            fn=_reprocess_task,
            on_success=self._on_psf_task_success,
            error_title="Dataset processing failed",
            success_status="Processing complete.",
            eta_total_seconds=eta_seconds,
            eta_kind="psf",
            allow_cancel=True,
            cancel_event=cancel_event,
            canceled_status="Processing canceled. Previous rendering kept.",
        )

    def _on_render_config_changed(self, config: RenderConfig) -> None:
        previous = self.current_render
        self.current_render = config
        new_z_scale = float(max(0.05, config.display_z_scale))
        z_scale_changed = not np.isclose(
            float(previous.display_z_scale),
            new_z_scale,
            atol=1.0e-6,
        )
        trim_changed = self._render_trim_changed(previous, config)
        data_upload_changed = z_scale_changed or trim_changed
        metrics_changed = self._render_config_affects_metrics(previous, config)
        self._display_z_scale = new_z_scale
        if self.processed_dataset is None:
            return
        green_threshold_changed = not np.isclose(
            float(previous.threshold_green),
            float(config.threshold_green),
            atol=1.0e-6,
        )
        red_threshold_changed = not np.isclose(
            float(previous.threshold_red),
            float(config.threshold_red),
            atol=1.0e-6,
        )
        should_auto_wipe = (
            (green_threshold_changed or red_threshold_changed)
            and self.controls.auto_wipe_specks_on_threshold_edit_enabled()
        )
        if green_threshold_changed:
            self._invalidate_microglia_components()
        isolate_enabled, _ = self.controls.microglia_view_state()
        if isolate_enabled and green_threshold_changed:
            # Start background microglia refresh which will also push scene
            # channels when done — avoid synchronous watershed here.
            if data_upload_changed:
                self._push_scene_channels()
            self.scene.set_microglia_analysis_debug(None)
            self._start_microglia_refresh_task()
            self.scene.apply_render_config(config)
            if metrics_changed:
                self._refresh_metrics()
            if should_auto_wipe:
                self._on_wipe_specks_requested()
            return
        if isolate_enabled:
            self._refresh_microglia_components_if_needed()
        if data_upload_changed:
            self._push_scene_channels()
        self.scene.apply_render_config(config)
        self._refresh_microglia_analysis_debug()
        if metrics_changed:
            self._refresh_metrics()
        if should_auto_wipe:
            self._on_wipe_specks_requested()

    def _on_microglia_view_changed(self) -> None:
        if self.visual_dataset is None or self.processed_dataset is None:
            return
        isolate_enabled, _ = self.controls.microglia_view_state()
        was_enabled = bool(self._microglia_isolate_active)
        self._microglia_isolate_active = bool(isolate_enabled)
        if isolate_enabled:
            self._sync_analytics_table_selection()
            if self._page_stack.currentIndex() == 2 or self.latest_microglia_analysis is not None:
                self._refresh_analytics_metrics()
            self._start_microglia_refresh_task()
            return
        elif was_enabled:
            # Only push full green when isolate mode is being turned OFF.
            self._push_scene_channels(green_only=True)
        else:
            # Ignore slider/index changes while isolate mode is already disabled.
            self._sync_analytics_table_selection()
            if self._page_stack.currentIndex() == 2 or self.latest_microglia_analysis is not None:
                self._refresh_analytics_metrics()
            self._refresh_microglia_analysis_debug()
            return
        self.scene.apply_render_config(self.current_render)
        self._refresh_microglia_analysis_debug()
        if self._page_stack.currentIndex() == 2 or self.latest_microglia_analysis is not None:
            self._refresh_analytics_metrics()
        self._sync_analytics_table_selection()

    def _on_enhance_microglia_requested(self) -> None:
        if self.processed_dataset is None:
            self._show_error("No dataset", "Load and render a dataset before enhancing microglia.")
            return
        if self._active_thread is not None and self._active_thread.isRunning():
            self.statusBar().showMessage("Another operation is still running.", 3000)
            return

        dataset = self.processed_dataset
        preprocess_cfg = self.preprocess_config
        enhancement_method = self.controls.current_microglia_enhancement_method()
        eta_seconds = self._estimate_microglia_separation_eta_seconds(
            dataset.green.data.shape,
            float(self.controls.current_microglia_branch_sensitivity()),
        )

        def _report_enhance_progress(completed: int, total: int) -> None:
            if total <= 0:
                return
            fraction = min(1.0, completed / total)
            self._publish_busy_progress(
                percent=8.0 + (fraction * 78.0),
                message=f"Enhancing microglia ({completed}/{total} slices)...",
            )

        def _enhance_task() -> DatasetVolume:
            self._publish_busy_progress(percent=8.0, message="Estimating green background...")
            enhanced_green = enhance_microglia_background(
                dataset.green.data,
                preprocess_cfg,
                method=enhancement_method,
                progress_callback=_report_enhance_progress,
            )
            self._publish_busy_progress(percent=86.0, message="Updating enhanced dataset...")
            return DatasetVolume(
                green=ChannelVolume(
                    name="green",
                    data=enhanced_green,
                    z_indices=list(dataset.green.z_indices),
                    spacing=dataset.green.spacing,
                ),
                red=dataset.red,
                shared_z_range=dataset.shared_z_range,
            )

        def _on_enhance_success(result: object) -> None:
            self._current_microglia_enhancement_method = enhancement_method
            self._on_enhance_microglia_success(result)

        self._start_background_task(
            title="Enhance Microglia",
            message=f"Enhancing microglia with {enhancement_method}...",
            fn=_enhance_task,
            on_success=_on_enhance_success,
            error_title="Microglia enhancement failed",
            success_status="Microglia enhancement complete.",
            eta_total_seconds=eta_seconds,
            eta_kind="microglia-separation",
        )

    def _on_enhance_microglia_success(self, result: object) -> None:
        if not isinstance(result, DatasetVolume):
            raise TypeError("Invalid microglia enhancement result payload.")
        self._publish_busy_progress(percent=90.0, message="Preparing enhanced render...")
        self.processed_dataset = result
        self._mark_metrics_dirty()
        self.visual_dataset = prepare_dataset_for_mesh(result, self.preprocess_config)
        self._invalidate_microglia_components()
        self._push_scene_channels()
        threshold_green = _DEFAULT_GREEN_THRESHOLD
        threshold_red = _DEFAULT_RED_THRESHOLD
        # Block signals so set_threshold_defaults doesn't trigger _refresh_metrics
        # synchronously and freeze the progress dialog before it can show 100%.
        self.controls.blockSignals(True)
        self.controls.set_threshold_defaults(threshold_green, threshold_red)
        self.controls.blockSignals(False)
        self.current_render = self.controls.current_render_config()
        self._publish_busy_progress(percent=96.0, message="Refreshing enhanced render...")
        self.scene.apply_render_config(self.current_render)
        self._refresh_microglia_analysis_debug()
        # Persist the enhanced green channel so the project can restore it on next open.
        if self._last_processed_cache_key and self._current_microglia_enhancement_method:
            try:
                save_enhanced_dataset(
                    self._last_processed_cache_key,
                    self._current_microglia_enhancement_method,
                    result.green.data,
                    base_dir=self._project_cache_base_dir(),
                )
            except OSError as exc:
                self._log_info(f"Enhanced cache save failed: {exc}")
        self._publish_busy_progress(percent=100.0, message="Microglia enhancement complete.")
        self._log_info(
            "Microglia enhancement applied: "
            f"green_shape={result.green.data.shape} thresholds=(green={threshold_green:.4f}, red={threshold_red:.4f})"
        )
        self._save_current_project_state()
        QTimer.singleShot(0, self._refresh_metrics)

    def _on_wipe_specks_requested(self) -> None:
        if self.processed_dataset is None:
            self._show_error("No dataset", "Load and render a dataset before wiping specks.")
            return
        if self._active_thread is not None and self._active_thread.isRunning():
            self.statusBar().showMessage("Another operation is still running.", 3000)
            return

        dataset = self.processed_dataset
        max_voxels = int(self.controls.current_wipe_speck_max_voxels())
        # Detect specks among the voxels that are currently visible, so the wipe
        # removes exactly the small dots the user sees in each channel.
        threshold_green = float(self.current_render.threshold_green)
        threshold_red = float(self.current_render.threshold_red)

        def _wipe_task() -> DatasetVolume:
            self._publish_busy_progress(percent=10.0, message="Wiping microglia specks...")
            green = wipe_small_specks(
                dataset.green.data,
                threshold=threshold_green,
                min_voxels=max_voxels,
            )
            self._publish_busy_progress(percent=55.0, message="Wiping vasculature specks...")
            red = wipe_small_specks(
                dataset.red.data,
                threshold=threshold_red,
                min_voxels=max_voxels,
            )
            self._publish_busy_progress(percent=85.0, message="Updating dataset...")
            return DatasetVolume(
                green=ChannelVolume(
                    name="green",
                    data=green,
                    z_indices=list(dataset.green.z_indices),
                    spacing=dataset.green.spacing,
                ),
                red=ChannelVolume(
                    name="red",
                    data=red,
                    z_indices=list(dataset.red.z_indices),
                    spacing=dataset.red.spacing,
                ),
                shared_z_range=dataset.shared_z_range,
            )

        self._start_background_task(
            title="Wipe Specks",
            message=f"Removing isolated specks smaller than {max_voxels} voxels...",
            fn=_wipe_task,
            on_success=self._on_wipe_specks_success,
            error_title="Speck wipe failed",
            success_status="Speck wipe complete.",
        )

    def _on_wipe_specks_success(self, result: object) -> None:
        if not isinstance(result, DatasetVolume):
            raise TypeError("Invalid speck-wipe result payload.")
        self._publish_busy_progress(percent=90.0, message="Preparing render...")
        self.processed_dataset = result
        self._mark_metrics_dirty()
        self.visual_dataset = prepare_dataset_for_mesh(result, self.preprocess_config)
        self._invalidate_microglia_components()
        self._push_scene_channels()
        self._publish_busy_progress(percent=96.0, message="Refreshing render...")
        self.scene.apply_render_config(self.current_render)
        self._refresh_microglia_analysis_debug()
        self._publish_busy_progress(percent=100.0, message="Speck wipe complete.")
        self._log_info(
            "Speck wipe applied: "
            f"green_shape={result.green.data.shape} red_shape={result.red.data.shape}"
        )
        self._current_speck_wipe_applied = True
        self._save_current_project_state()
        QTimer.singleShot(0, self._refresh_metrics)

    def _refresh_metrics(self) -> None:
        if self.processed_dataset is None:
            return
        cache_key = self._metrics_cache_key_for_render(self.current_render)
        if self.latest_metrics is not None and self._metrics_cache_key == cache_key:
            self._set_metrics_text_from_result(self.latest_metrics)
            self._metrics_panel.update_from_metrics(self.latest_metrics)
            self._refresh_analytics_metrics()
            self._log_debug("Metrics updated from cache.")
            return

        total_voxels = int(self.processed_dataset.green.data.size + self.processed_dataset.red.data.size)
        if total_voxels <= _METRICS_BACKGROUND_MIN_VOXELS:
            self.latest_metrics = compute_metrics(self.processed_dataset, self.current_render)
            self._metrics_cache_key = cache_key
            self._vascular_summary_cache = (
                _vascular_cache_key_for(self.current_render),
                _compute_vascular_summary_line(self.processed_dataset, self.current_render),
            )
            self._set_metrics_text_from_result(self.latest_metrics)
            self._metrics_panel.update_from_metrics(self.latest_metrics)
            self._refresh_analytics_metrics()
            self._log_debug("Metrics updated synchronously for small dataset.")
            return

        if self._metrics_thread is not None and self._metrics_thread.isRunning():
            self._metrics_refresh_pending = self._metrics_task_key != cache_key
            self.statusBar().showMessage("Computing metrics in background...", 3000)
            self._log_debug("Metrics refresh queued while previous metrics task is running.")
            return

        dataset = self.processed_dataset
        render = self.current_render
        self._metrics_task_key = cache_key
        self._metrics_refresh_pending = False
        self.statusBar().showMessage("Computing metrics in background...", 3000)
        self._log_info("Metrics computation started in background.")

        def _metrics_task() -> _MetricsTaskResult:
            # Both compute_metrics and the vascular morphometry run here, off the
            # UI thread, so the window keeps pumping events during the (often
            # >60s) analysis instead of freezing into a Windows AppHang.
            metrics = compute_metrics(dataset, render)
            return _MetricsTaskResult(
                cache_key=cache_key,
                metrics=metrics,
                vascular_cache_key=_vascular_cache_key_for(render),
                vascular_line=_compute_vascular_summary_line(dataset, render),
            )

        thread = _FunctionThread(_metrics_task, self)
        self._metrics_thread = thread

        def cleanup() -> None:
            if self._metrics_thread is thread:
                self._metrics_thread = None
                self._metrics_task_key = None
            thread.deleteLater()
            if self._metrics_refresh_pending:
                self._metrics_refresh_pending = False
                QTimer.singleShot(0, self._refresh_metrics)

        def handle_success(result: object) -> None:
            try:
                if not isinstance(result, _MetricsTaskResult):
                    raise TypeError("Invalid metrics task result payload.")
                current_key = (
                    self._metrics_cache_key_for_render(self.current_render)
                    if self.processed_dataset is not None
                    else None
                )
                if result.cache_key != current_key:
                    self._metrics_refresh_pending = True
                    self._log_debug("Ignored stale metrics result; scheduling refresh.")
                    return
                self.latest_metrics = result.metrics
                self._metrics_cache_key = result.cache_key
                if result.vascular_cache_key is not None:
                    self._vascular_summary_cache = (
                        result.vascular_cache_key,
                        result.vascular_line,
                    )
                self._set_metrics_text_from_result(result.metrics)
                self._metrics_panel.update_from_metrics(result.metrics)
                self._refresh_analytics_metrics()
                self.statusBar().showMessage("Metrics updated.", 3000)
                self._log_info("Metrics computation completed.")
            except Exception as exc:
                self._log_info(f"Metrics update failed: {exc}")
            finally:
                cleanup()

        def handle_error(error_text: str) -> None:
            logger.error("Metrics task error:\n%s", error_text)
            concise = error_text.strip().splitlines()[-1] if error_text.strip() else "Unknown error"
            self.statusBar().showMessage(f"Metrics failed: {concise}", 5000)
            self._log_info(f"Metrics computation failed: {concise}")
            cleanup()

        thread.result_ready.connect(handle_success)
        thread.error_raised.connect(handle_error)
        thread.start()

    def _on_export_project_analytics_requested(self) -> None:
        if len(self._project_set_entries) <= 1 or self._project_set_root is None:
            self._show_error(
                "No project set",
                "Load a Project Set Folder before exporting whole-project analytics.",
            )
            return
        if self.processed_dataset is None:
            self._show_error("No current sample", "Open one sample in the project set first.")
            return
        if self._active_thread is not None and self._active_thread.isRunning():
            self.statusBar().showMessage("Another operation is still running.", 3000)
            return
        self.controls.force_apply_pending_changes()

        start = str(self._project_set_root / "project_analytics.csv")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Project Analytics CSV",
            start,
            "CSV files (*.csv)",
        )
        if not file_path:
            return

        config = _ProjectAnalyticsConfig(
            render=self.current_render,
            psf=self._effective_psf_config(self.current_psf),
            preprocess=self.preprocess_config,
            apply_enhancement=bool(
                self._current_microglia_enhancement_method
                or self.controls.auto_enhance_microglia_on_load_enabled()
            ),
            enhancement_method=str(
                self._current_microglia_enhancement_method
                or self.controls.current_microglia_enhancement_method()
            ),
            apply_wipe=bool(
                self._current_speck_wipe_applied
                or self.controls.auto_wipe_specks_on_load_enabled()
            ),
            wipe_max_voxels=int(self.controls.current_wipe_speck_max_voxels()),
            branch_sensitivity=float(self.controls.current_microglia_branch_sensitivity()),
        )
        entries = list(self._project_set_entries)
        base_path = Path(file_path).resolve()
        root = self._project_set_root

        confirm = QMessageBox.question(
            self,
            "Export Project Analytics",
            (
                f"Analyze {len(entries)} samples using the current sample settings?\n\n"
                f"Green threshold: {config.render.threshold_green:.3f}\n"
                f"Red threshold: {config.render.threshold_red:.3f}\n"
                f"Clean: {'yes (' + config.enhancement_method + ')' if config.apply_enhancement else 'no'}\n"
                f"Wipe specks: {'yes (< ' + str(config.wipe_max_voxels) + ' vox)' if config.apply_wipe else 'no'}"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        def _batch_task() -> _ProjectAnalyticsResult:
            return self._export_project_analytics_batch(root, entries, config, base_path)

        self._start_background_task(
            title="Project Analytics Export",
            message="Analyzing all project-set samples...",
            fn=_batch_task,
            on_success=self._on_project_analytics_export_success,
            error_title="Project analytics export failed",
            success_status="Project analytics exported.",
            eta_kind="microglia-separation",
        )

    def _export_project_analytics_batch(
        self,
        project_root: Path,
        entries: list[DatasetProjectCandidate],
        config: _ProjectAnalyticsConfig,
        base_path: Path,
    ) -> _ProjectAnalyticsResult:
        individual_rows: list[dict[str, object]] = []
        provenance_rows: list[dict[str, object]] = []
        total = max(1, len(entries))
        run_started = datetime.now().isoformat(timespec="seconds")

        for idx, entry in enumerate(entries, start=1):
            sample_start = time.perf_counter()
            base_percent = 100.0 * float(idx - 1) / float(total)
            span = 100.0 / float(total)

            def stage(fraction: float, message: str) -> None:
                self._publish_busy_progress(
                    percent=base_percent + span * float(np.clip(fraction, 0.0, 1.0)),
                    message=f"{entry.name} ({idx}/{total}): {message}",
                )

            stage(0.03, "loading stacks...")
            dataset = load_dataset(entry.root, spacing=self.spacing)
            stage(0.12, "synchronizing slices...")
            synced = fill_and_sync_dataset(dataset)
            working = synced
            if config.preprocess.enabled:
                stage(0.22, "preprocessing...")
                working = preprocess_dataset(working, config.preprocess)
            stage(0.36, "processing channels...")
            processed = apply_psf_to_dataset(
                working,
                config.psf,
                preprocess_config=config.preprocess,
                cancel_event=None,
            )
            green = np.asarray(processed.green.data, dtype=np.float32)
            red = np.asarray(processed.red.data, dtype=np.float32)
            if config.apply_enhancement:
                stage(0.52, "enhancing microglia...")

                def _report_enhance_progress(completed: int, total_slices: int) -> None:
                    if total_slices <= 0:
                        return
                    fraction = 0.52 + (min(1.0, completed / total_slices) * 0.10)
                    stage(fraction, f"enhancing microglia ({completed}/{total_slices} slices)...")

                green = enhance_microglia_background(
                    green,
                    config.preprocess,
                    method=config.enhancement_method,
                    progress_callback=_report_enhance_progress,
                )
            if config.apply_wipe:
                stage(0.62, "wiping specks...")
                green = wipe_small_specks(
                    green,
                    threshold=float(config.render.threshold_green),
                    min_voxels=int(config.wipe_max_voxels),
                )
                red = wipe_small_specks(
                    red,
                    threshold=float(config.render.threshold_red),
                    min_voxels=int(config.wipe_max_voxels),
                )
            processed = DatasetVolume(
                green=ChannelVolume(
                    name="green",
                    data=green,
                    z_indices=list(processed.green.z_indices),
                    spacing=processed.green.spacing,
                ),
                red=ChannelVolume(
                    name="red",
                    data=red,
                    z_indices=list(processed.red.z_indices),
                    spacing=processed.red.spacing,
                ),
                shared_z_range=processed.shared_z_range,
            )
            visual = prepare_dataset_for_mesh(processed, config.preprocess)

            stage(0.70, "computing whole-volume metrics...")
            metrics = compute_metrics(processed, config.render)
            individual_rows.extend(
                self._stamp_project_rows(
                    metrics_to_csv_rows(metrics),
                    entry=entry,
                    sample_index=idx,
                    metric_family="basic",
                )
            )

            stage(0.78, "computing vascular analytics...")
            vascular = analyze_vasculature(
                processed.red.data,
                threshold=float(config.render.threshold_red),
                spacing=processed.red.spacing,
                render=config.render,
            )
            individual_rows.extend(
                self._stamp_project_rows(
                    vascular_analysis_to_csv_rows(vascular),
                    entry=entry,
                    sample_index=idx,
                    metric_family="vascular",
                )
            )

            stage(0.88, "computing microglia analytics...")
            spacing = visual.green.spacing
            spacing_zyx = (float(spacing.z_um), float(spacing.y_um), float(spacing.x_um))
            base_min_voxels = max(64, int(config.preprocess.green_speckle_min_voxels) * 4)
            labels, order, _sizes = self._compute_microglia_components_from_params(
                visual.green.data,
                threshold=float(config.render.threshold_green),
                branch_sense=float(config.branch_sensitivity),
                base_min_voxels=base_min_voxels,
                spacing=spacing_zyx,
            )
            microglia = analyze_microglia_cells(
                visual.green.data,
                visual.red.data,
                labels,
                order,
                spacing=visual.green.spacing,
                render=config.render,
                branch_sensitivity=float(config.branch_sensitivity),
            )
            individual_rows.extend(
                self._stamp_project_rows(
                    microglia_analysis_to_csv_rows(microglia),
                    entry=entry,
                    sample_index=idx,
                    metric_family="microglia_cell",
                )
            )
            assoc = summarize_neurovascular_association(microglia)
            individual_rows.extend(
                self._stamp_project_rows(
                    neurovascular_association_to_csv_rows(assoc),
                    entry=entry,
                    sample_index=idx,
                    metric_family="neurovascular",
                )
            )

            provenance_rows.extend(
                self._project_sample_provenance_rows(
                    entry=entry,
                    sample_index=idx,
                    project_root=project_root,
                    config=config,
                    processed=processed,
                    analyzed_cell_count=int(microglia.analyzed_cell_count),
                    run_started=run_started,
                    elapsed_seconds=time.perf_counter() - sample_start,
                )
            )
            stage(1.0, "complete.")

        cumulative_rows = self._build_project_cumulative_rows(individual_rows, len(entries))
        all_rows = [*individual_rows, *cumulative_rows]
        self._publish_busy_progress(percent=98.0, message="Writing project CSV files...")
        main_path = export_metrics_csv(all_rows, base_path)
        stem = main_path.stem
        individual_path = export_metrics_csv(
            individual_rows,
            main_path.with_name(f"{stem}_individual.csv"),
        )
        cumulative_path = export_metrics_csv(
            cumulative_rows,
            main_path.with_name(f"{stem}_cumulative.csv"),
        )
        provenance_path = export_metrics_csv(
            provenance_rows,
            main_path.with_name(f"{stem}_provenance.csv"),
        )
        self._publish_busy_progress(percent=100.0, message="Project analytics export complete.")
        return _ProjectAnalyticsResult(
            base_path=main_path,
            files=[main_path, individual_path, cumulative_path, provenance_path],
            sample_count=len(entries),
            row_count=len(all_rows),
        )

    def _stamp_project_rows(
        self,
        rows: list[dict[str, object]],
        *,
        entry: DatasetProjectCandidate,
        sample_index: int,
        metric_family: str,
    ) -> list[dict[str, object]]:
        stamped: list[dict[str, object]] = []
        for row in rows:
            out: dict[str, object] = {
                "row_scope": "individual",
                "metric_family": metric_family,
                "sample_index": int(sample_index),
                "sample_name": entry.name,
                "sample_path": str(entry.root),
            }
            out.update(row)
            stamped.append(out)
        return stamped

    def _project_sample_provenance_rows(
        self,
        *,
        entry: DatasetProjectCandidate,
        sample_index: int,
        project_root: Path,
        config: _ProjectAnalyticsConfig,
        processed: DatasetVolume,
        analyzed_cell_count: int,
        run_started: str,
        elapsed_seconds: float,
    ) -> list[dict[str, object]]:
        spacing = processed.green.spacing
        values: dict[str, object] = {
            "run_started_at": run_started,
            "project_root": str(project_root),
            "sample_name": entry.name,
            "sample_path": str(entry.root),
            "threshold_green": float(config.render.threshold_green),
            "threshold_red": float(config.render.threshold_red),
            "trim_first_slices": int(config.render.trim_first_slices),
            "trim_last_slices": int(config.render.trim_last_slices),
            "offset_x_um": float(config.render.offset_x_um),
            "offset_y_um": float(config.render.offset_y_um),
            "offset_z_um": float(config.render.offset_z_um),
            "microglia_enhancement_applied": bool(config.apply_enhancement),
            "microglia_enhancement_method": config.enhancement_method if config.apply_enhancement else "",
            "wipe_specks_applied": bool(config.apply_wipe),
            "speck_max_voxels": int(config.wipe_max_voxels),
            "branch_sensitivity": float(config.branch_sensitivity),
            "spacing_x_um": float(spacing.x_um),
            "spacing_y_um": float(spacing.y_um),
            "spacing_z_um": float(spacing.z_um),
            "voxel_volume_um3": float(spacing.voxel_volume_um3),
            "green_shape_zyx": "x".join(str(int(v)) for v in processed.green.data.shape),
            "red_shape_zyx": "x".join(str(int(v)) for v in processed.red.data.shape),
            "shared_z_range": f"{processed.shared_z_range[0]}-{processed.shared_z_range[1]}",
            "microglia_cell_count": int(analyzed_cell_count),
            "elapsed_seconds": float(elapsed_seconds),
        }
        return [
            {
                "sample_index": int(sample_index),
                "sample_name": entry.name,
                "setting": key,
                "value": value,
            }
            for key, value in values.items()
        ]

    def _build_project_cumulative_rows(
        self,
        individual_rows: list[dict[str, object]],
        sample_count: int,
    ) -> list[dict[str, object]]:
        cumulative: list[dict[str, object]] = [
            {
                "row_scope": "cumulative",
                "metric_family": "project_summary",
                "metric": "sample_count",
                "value": int(sample_count),
            }
        ]

        def numeric(value: object) -> float | None:
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float, np.integer, np.floating)):
                val = float(value)
                return val if np.isfinite(val) else None
            return None

        basic_groups: dict[str, dict[str, float]] = {}
        basic_counts: dict[str, int] = {}
        for row in individual_rows:
            if row.get("metric_family") != "basic":
                continue
            channel = str(row.get("channel", "unknown"))
            group = basic_groups.setdefault(channel, {})
            basic_counts[channel] = basic_counts.get(channel, 0) + 1
            for key, value in row.items():
                if key in {"sample_index", "metric_family", "row_scope"}:
                    continue
                val = numeric(value)
                if val is not None:
                    group[key] = group.get(key, 0.0) + val
        for channel, sums in sorted(basic_groups.items()):
            count = max(1, basic_counts.get(channel, 1))
            for key, total in sorted(sums.items()):
                cumulative.append(
                    {
                        "row_scope": "cumulative",
                        "metric_family": "basic_sum",
                        "channel": channel,
                        "metric": key,
                        "value": total,
                    }
                )
                cumulative.append(
                    {
                        "row_scope": "cumulative",
                        "metric_family": "basic_mean_per_sample",
                        "channel": channel,
                        "metric": key,
                        "value": total / float(count),
                    }
                )

        metric_value_groups: dict[tuple[str, str], list[float]] = {}
        microglia_groups: dict[str, list[float]] = {}
        for row in individual_rows:
            family = str(row.get("metric_family", ""))
            if family in {"vascular", "neurovascular"}:
                metric = str(row.get("metric", ""))
                val = numeric(row.get("value"))
                if metric and val is not None:
                    metric_value_groups.setdefault((family, metric), []).append(val)
            elif family == "microglia_cell":
                for key, value in row.items():
                    if key in {"sample_index", "sample_name", "sample_path", "metric_family", "row_scope"}:
                        continue
                    val = numeric(value)
                    if val is not None:
                        microglia_groups.setdefault(key, []).append(val)

        for (family, metric), values in sorted(metric_value_groups.items()):
            arr = np.asarray(values, dtype=np.float64)
            cumulative.extend(
                [
                    {
                        "row_scope": "cumulative",
                        "metric_family": f"{family}_mean_per_sample",
                        "metric": metric,
                        "value": float(np.mean(arr)),
                    },
                    {
                        "row_scope": "cumulative",
                        "metric_family": f"{family}_sum",
                        "metric": metric,
                        "value": float(np.sum(arr)),
                    },
                    {
                        "row_scope": "cumulative",
                        "metric_family": f"{family}_min",
                        "metric": metric,
                        "value": float(np.min(arr)),
                    },
                    {
                        "row_scope": "cumulative",
                        "metric_family": f"{family}_max",
                        "metric": metric,
                        "value": float(np.max(arr)),
                    },
                ]
            )

        for metric, values in sorted(microglia_groups.items()):
            arr = np.asarray(values, dtype=np.float64)
            cumulative.extend(
                [
                    {
                        "row_scope": "cumulative",
                        "metric_family": "microglia_cell_mean",
                        "metric": metric,
                        "value": float(np.mean(arr)),
                    },
                    {
                        "row_scope": "cumulative",
                        "metric_family": "microglia_cell_sum",
                        "metric": metric,
                        "value": float(np.sum(arr)),
                    },
                    {
                        "row_scope": "cumulative",
                        "metric_family": "microglia_cell_min",
                        "metric": metric,
                        "value": float(np.min(arr)),
                    },
                    {
                        "row_scope": "cumulative",
                        "metric_family": "microglia_cell_max",
                        "metric": metric,
                        "value": float(np.max(arr)),
                    },
                ]
            )
        return cumulative

    def _on_project_analytics_export_success(self, result: object) -> None:
        if not isinstance(result, _ProjectAnalyticsResult):
            raise TypeError("Invalid project analytics export result payload.")
        files = ", ".join(str(path) for path in result.files)
        self.statusBar().showMessage(
            f"Project analytics exported for {result.sample_count} samples.",
            7000,
        )
        self._log_info(
            f"Project analytics exported: samples={result.sample_count} "
            f"rows={result.row_count} files={files}"
        )
        QMessageBox.information(
            self,
            "Project Analytics Exported",
            (
                f"Analyzed {result.sample_count} samples and wrote {result.row_count} rows.\n\n"
                f"Main CSV:\n{result.base_path}\n\n"
                "Companion files include individual, cumulative, and provenance CSVs."
            ),
        )

    def _on_export_metrics_requested(self) -> None:
        if self.latest_metrics is None:
            self._show_error("No metrics", "Compute metrics before exporting.")
            return
        start = str((self.dataset_root or Path.cwd()) / "metrics.csv")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Metrics CSV",
            start,
            "CSV files (*.csv)",
        )
        if not file_path:
            return
        try:
            with self._busy("Export Metrics", "Writing CSV..."):
                self._publish_busy_progress(percent=10.0, message="Collecting metrics rows...")
                rows = metrics_to_csv_rows(self.latest_metrics)
                self._publish_busy_progress(percent=50.0, message="Writing CSV file...")
                out = export_metrics_csv(rows, file_path)
                self._publish_busy_progress(percent=70.0, message="Computing vascular morphometry...")
                companions = self._export_extended_analytics(out)
                self._publish_busy_progress(percent=100.0, message="Metrics export complete.")
            extra = f" (+{len(companions)} analytics files)" if companions else ""
            self.statusBar().showMessage(f"Metrics exported to {out}{extra}", 5000)
            self._log_info(f"Metrics exported: {out}{extra}")
        except Exception as exc:
            self._log_info(f"Metrics export failed: {exc}")
            self._show_error("Export failed", str(exc))

    def _export_extended_analytics(self, base_path: Path) -> list[Path]:
        """Write vascular + neurovascular companion CSVs next to ``metrics.csv``.

        These carry the quantitative vasculature morphometry and the population
        neurovascular-association patterns that don't fit the per-channel metrics
        schema. Failures are logged but never block the primary metrics export.
        """
        written: list[Path] = []
        base = Path(base_path)
        stem = base.stem

        # Provenance: record exactly which settings produced these numbers so the
        # CSVs are self-describing and reproducible.
        try:
            ppath = base.with_name(f"{stem}_provenance.csv")
            export_metrics_csv(self._provenance_rows(), ppath)
            written.append(ppath)
        except Exception as exc:  # pragma: no cover - defensive UI path
            self._log_info(f"Provenance export skipped: {exc}")

        dataset = self.processed_dataset
        if dataset is not None:
            try:
                vascular = analyze_vasculature(
                    dataset.red.data,
                    threshold=float(self.current_render.threshold_red),
                    spacing=dataset.red.spacing,
                    render=self.current_render,
                )
                vpath = base.with_name(f"{stem}_vascular.csv")
                export_metrics_csv(vascular_analysis_to_csv_rows(vascular), vpath)
                written.append(vpath)
            except Exception as exc:  # pragma: no cover - defensive UI path
                self._log_info(f"Vascular metrics export skipped: {exc}")

        if self.latest_microglia_analysis is not None:
            try:
                mpath = base.with_name(f"{stem}_microglia.csv")
                export_metrics_csv(microglia_analysis_to_csv_rows(self.latest_microglia_analysis), mpath)
                written.append(mpath)

                assoc = summarize_neurovascular_association(self.latest_microglia_analysis)
                npath = base.with_name(f"{stem}_neurovascular.csv")
                export_metrics_csv(neurovascular_association_to_csv_rows(assoc), npath)
                written.append(npath)
            except Exception as exc:  # pragma: no cover - defensive UI path
                self._log_info(f"Microglia/neurovascular metrics export skipped: {exc}")

        return written

    def _provenance_rows(self) -> list[dict[str, object]]:
        """Long-format (setting, value) rows describing how the metrics were made."""
        render = self.current_render
        dataset = self.processed_dataset
        spacing = dataset.green.spacing if dataset is not None else None
        cells = (
            self.latest_microglia_analysis.analyzed_cell_count
            if self.latest_microglia_analysis is not None
            else None
        )
        values: dict[str, object] = {
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_name": self.dataset_root.name if self.dataset_root else "",
            "dataset_path": str(self.dataset_root) if self.dataset_root else "",
            "compute_backend": os.environ.get("NVAP_GPU_BACKEND", "auto"),
            "threshold_green": float(render.threshold_green),
            "threshold_red": float(render.threshold_red),
            "trim_first_slices": int(render.trim_first_slices),
            "trim_last_slices": int(render.trim_last_slices),
            "offset_x_um": float(render.offset_x_um),
            "offset_y_um": float(render.offset_y_um),
            "offset_z_um": float(render.offset_z_um),
            "display_z_scale": float(render.display_z_scale),
            "microglia_enhancement_method": self._current_microglia_enhancement_method or "",
            "wipe_specks_on_load": bool(self.controls.auto_wipe_specks_on_load_enabled()),
            "enhance_microglia_on_load": bool(self.controls.auto_enhance_microglia_on_load_enabled()),
            "wipe_specks_on_threshold_edit": bool(
                self.controls.auto_wipe_specks_on_threshold_edit_enabled()
            ),
            "speck_max_voxels": int(self.controls.current_wipe_speck_max_voxels()),
        }
        if spacing is not None:
            values.update(
                {
                    "spacing_x_um": float(spacing.x_um),
                    "spacing_y_um": float(spacing.y_um),
                    "spacing_z_um": float(spacing.z_um),
                    "voxel_volume_um3": float(spacing.voxel_volume_um3),
                }
            )
        if dataset is not None:
            values["green_shape_zyx"] = "x".join(str(int(v)) for v in dataset.green.data.shape)
            values["red_shape_zyx"] = "x".join(str(int(v)) for v in dataset.red.data.shape)
            values["shared_z_range"] = f"{dataset.shared_z_range[0]}-{dataset.shared_z_range[1]}"
        if cells is not None:
            values["microglia_cell_count"] = int(cells)
        return [{"setting": key, "value": value} for key, value in values.items()]

    def _on_export_snapshot_requested(self) -> None:
        start = str((self.dataset_root or Path.cwd()) / "snapshot.png")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Snapshot PNG",
            start,
            "PNG files (*.png)",
        )
        if not file_path:
            return
        try:
            with self._busy("Export Snapshot", "Rendering and writing PNG..."):
                self._publish_busy_progress(percent=20.0, message="Capturing render buffer...")
                out = self.scene.capture_snapshot(file_path)
                self._publish_busy_progress(percent=100.0, message="Snapshot export complete.")
            self.statusBar().showMessage(f"Snapshot exported to {out}", 5000)
            self._log_info(f"Snapshot exported: {out}")
        except Exception as exc:
            self._log_info(f"Snapshot export failed: {exc}")
            self._show_error("Snapshot export failed", str(exc))

    def _show_error(self, title: str, details: str) -> None:
        logger.error("%s: %s", title, details)
        QMessageBox.critical(self, title, details)

    def _on_export_mesh_requested(self) -> None:
        if self.processed_dataset is None:
            self._show_error("No dataset", "Load and process a dataset before exporting meshes.")
            return
        start = str((self.dataset_root or Path.cwd()) / "meshes")
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Mesh Export Directory",
            start,
        )
        if not dir_path:
            return
        dataset = self.visual_dataset or self.processed_dataset
        mesh_cfg = self.controls.current_mesh_config()
        eta_seconds = self._estimate_mesh_export_eta_seconds(dataset, mesh_cfg)

        def _do_export() -> dict[str, Path]:
            self._publish_busy_progress(percent=10.0, message="Extracting per-channel meshes...")
            results = export_dataset_meshes(dataset, mesh_cfg, dir_path)
            # Also export combined mesh
            self._publish_busy_progress(percent=75.0, message="Reconstructing combined mesh...")
            combined_path = Path(dir_path) / "combined_mesh.ply"
            combined = reconstruct_combined_mesh(dataset, mesh_cfg, combined_path)
            if combined is not None:
                results["combined"] = combined
            self._publish_busy_progress(percent=100.0, message="Mesh export complete.")
            return results

        self._start_background_task(
            title="Export 3D Meshes",
            message="Extracting isosurfaces and exporting meshes...",
            fn=_do_export,
            on_success=self._on_mesh_export_success,
            error_title="Mesh export failed",
            success_status="3D meshes exported successfully.",
            eta_total_seconds=eta_seconds,
            eta_kind="mesh-export",
        )

    def _on_mesh_export_success(self, result: object) -> None:
        if isinstance(result, dict):
            paths = [str(p) for p in result.values()]
            self._log_info(f"Meshes exported: {', '.join(paths)}")
            self.statusBar().showMessage(f"Exported {len(result)} mesh file(s)", 5000)
