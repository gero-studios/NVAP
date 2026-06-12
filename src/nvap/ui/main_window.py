from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.ndimage as ndi
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
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

from nvap.analysis.microglia_analysis import MicrogliaAnalysisResult, analyze_microglia_cells
from nvap.analysis.microglia_analysis import build_microglia_cell_debug
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
    load_processed_dataset,
    save_processed_dataset,
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
from nvap.io.stack_loader import inspect_dataset_stats, load_dataset, resolve_channel_dirs
from nvap.pipeline import (
    apply_psf_to_dataset,
    default_green_threshold,
    default_threshold,
    fill_and_sync_dataset,
    prepare_dataset_for_mesh,
)
from nvap.analysis.microglia_components import (
    compute_component_labels,
    filter_components_by_preferred_voxel_floor,
    isolate_component,
)
from nvap.preprocess.enhancement import enhance_microglia_background, preprocess_dataset
from nvap.plugins.registry import discover_plugins
from nvap.render.vtk_scene import MicrogliaDebugOverlay, VTKScene
from nvap.ui.control_panel import ControlPanel
from nvap.ui.design import COLOR, ICON_MD, ICON_SM, SIDEBAR_WIDTH
from nvap.ui.dialogs.about import AboutDialog
from nvap.ui.home_page import HomePage
from nvap.ui.icons import icon, icon_pixmap
from nvap.ui.metrics_panel import MetricsPanel
from nvap.ui.services.recent_projects import RecentProjectsStore
from nvap.ui.services.project_files import (
    load_project_state,
    project_channel_sources,
    save_project_state,
)
from nvap.ui.services.system_status import SystemStatus, gpu_status, memory_status
from nvap.ui.sidebar_pages import SettingsPage

logger = logging.getLogger(__name__)


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


@dataclass
class _MicrogliaComponentsTaskResult:
    labels: np.ndarray
    order: np.ndarray
    sizes: np.ndarray
    threshold: float
    branch_sensitivity: float
    shape: tuple[int, int, int]


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
    @staticmethod
    def _build_process_task_result(
        processed_dataset: DatasetVolume,
        preprocess_config: PreprocessConfig,
    ) -> _LoadTaskResult:
        return _LoadTaskResult(
            synced_dataset=processed_dataset,
            raw_dataset=processed_dataset,
            processed_dataset=processed_dataset,
            visual_dataset=processed_dataset,
            threshold_green=default_green_threshold(processed_dataset.green.data),
            threshold_red=default_threshold(processed_dataset.red.data),
        )

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

        # Workspace splitter: inspector | VTK viewport | metrics
        self._workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._workspace_splitter.addWidget(self.controls_scroll)
        self._workspace_splitter.addWidget(self.scene.widget())
        self._workspace_splitter.setStretchFactor(0, 0)
        self._workspace_splitter.setStretchFactor(1, 1)

        # ── Project store + pages ────────────────────────────────────────
        self._project_store = RecentProjectsStore()
        self._home_page = HomePage(self._project_store)
        self._settings_page = SettingsPage()
        self._metrics_panel = MetricsPanel()
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
        self._metrics_cache_key: tuple[int, float, float, float, float, float] | None = None
        self._vascular_summary_cache: tuple[float, str | None] | None = None
        self._microglia_analysis_cache_key: tuple[int, tuple[int, int, int], float, float, int, int, float, float, float, float] | None = None
        self.dataset_root: Path | None = None
        self._dataset_signature: str | None = None
        self._current_channel_sources: dict[str, str] | None = None
        self._current_load_mode = "folder"
        self._last_processed_cache_key: str | None = None
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
        self.controls.export_metrics_requested.connect(self._on_export_metrics_requested)
        self.controls.export_snapshot_requested.connect(self._on_export_snapshot_requested)
        self.controls.export_mesh_requested.connect(self._on_export_mesh_requested)

        self._setup_keyboard_shortcuts()

        self._refresh_plugin_panel()
        self._home_page.refresh_projects()
        QTimer.singleShot(0, self._refresh_system_indicators)
        self.statusBar().showMessage("Load a dataset to begin. Preprocessing is ON by default.")
        self._log_info("NVAP UI initialized: green pass-through mode active.")
        self._refresh_section_state()

    def closeEvent(self, event) -> None:
        if self._active_thread is not None and self._active_thread.isRunning():
            self._log_info("Waiting briefly for active background task to finish before close.")
            self._active_thread.wait(2000)
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

    def _build_analytics_placeholder(self) -> QWidget:
        """Analytics overview for dataset and per-cell microglia metrics."""
        w = QWidget()
        w.setObjectName("sectionPage")
        lo = QVBoxLayout(w)
        lo.setContentsMargins(56, 48, 56, 48)
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
        lo.addSpacing(28)

        card_grid = QGridLayout()
        card_grid.setHorizontalSpacing(16)
        card_grid.setVerticalSpacing(16)
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
        lo.addSpacing(28)
        lo.addWidget(section)

        self._analytics_microglia_hint = QLabel(
            "Open Analytics on a rendered dataset to measure separated microglia against visible vasculature."
        )
        self._analytics_microglia_hint.setObjectName("pageIntro")
        self._analytics_microglia_hint.setWordWrap(True)
        lo.addWidget(self._analytics_microglia_hint)
        lo.addSpacing(20)

        analysis_grid = QGridLayout()
        analysis_grid.setHorizontalSpacing(16)
        analysis_grid.setVerticalSpacing(16)
        self._analytics_microglia_cards: dict[str, _AnalyticsMetricCard] = {
            "cells": _AnalyticsMetricCard("Cells analyzed", "--", "Visible separated microglia."),
            "branches": _AnalyticsMetricCard("Avg branches", "--", "Branch tips per visible cell."),
            "soma": _AnalyticsMetricCard("Avg soma volume", "--", "Non-branched soma body volume."),
            "distance": _AnalyticsMetricCard("Closest vessel distance", "--", "Shortest cell-to-vessel distance."),
        }
        for idx, card in enumerate(self._analytics_microglia_cards.values()):
            analysis_grid.addWidget(card, idx // 2, idx % 2)
        lo.addLayout(analysis_grid)
        lo.addSpacing(20)

        self._analytics_cell_table = QTableWidget(0, 6)
        self._analytics_cell_table.setObjectName("analyticsCellTable")
        self._analytics_cell_table.setHorizontalHeaderLabels(
            [
                "Cell",
                "Branches",
                "Soma (um^3)",
                "Tip -> Vessel (um)",
                "Cell -> Vessel (um)",
                "Soma -> Vessel (um)",
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

        for card in self._analytics_cards.values():
            card.set_value("--", "Load a dataset to populate.")

        if self.latest_metrics is None:
            return

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
        if self._page_stack.currentIndex() == 2 or self.latest_microglia_analysis is not None:
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
            f"{analysis.mean_process_length_um:,.1f} um length).",
        )
        self._analytics_microglia_cards["soma"].set_value(
            f"{analysis.mean_soma_volume_um3:,.1f}",
            "Average soma-body volume in um^3.",
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

        self._analytics_cell_table.setRowCount(int(len(analysis.cells)))
        for row, cell in enumerate(analysis.cells):
            values = [
                str(row + 1),
                str(cell.branch_count),
                f"{cell.soma_volume_um3:,.1f}",
                self._format_optional_analytics_value(cell.nearest_tip_to_vessel_um),
                self._format_optional_analytics_value(cell.nearest_cell_to_vessel_um),
                self._format_optional_analytics_value(cell.soma_to_vessel_um),
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
        known_cell_distance = None
        if self.latest_microglia_analysis is not None:
            for cell in self.latest_microglia_analysis.cells:
                if int(cell.component_id) == component_id:
                    known_tip_distance = cell.nearest_tip_to_vessel_um
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
        color = color_map.get(status.status, COLOR.text_disabled)
        dot.setStyleSheet(f"color: {color};")
        text.setText(status.label)
        if status.detail:
            pill.setToolTip(status.detail)

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
            )
        except OSError as exc:
            self._log_info(f"Project metadata save failed: {exc}")

    def _project_cache_base_dir(self) -> Path:
        return self.dataset_root or Path.cwd()

    def _refresh_section_state(self) -> None:
        dataset_name = self.dataset_root.name if self.dataset_root is not None else "No dataset loaded"
        cache_root = str(self._project_cache_base_dir() / ".nvap_cache")
        plugin_summary = self.controls.plugin_text.toPlainText().strip() or "No plugins discovered"
        auto_apply = bool(self.controls.auto_apply_checkbox.isChecked())
        has_dataset = bool(self.processed_dataset is not None and self.visual_dataset is not None)
        self._settings_page.set_runtime_details(
            dataset_name=dataset_name,
            auto_apply_enabled=auto_apply,
            plugin_summary=plugin_summary,
            cache_root=cache_root,
        )
        self.controls.set_microglia_workflow_enabled(has_dataset)
        if self.dataset_root is not None:
            self._home_page.set_preview_summary(
                "ACTIVE",
                self.dataset_root.name,
                f"{self.dataset_root}\nCache: {cache_root}",
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

    def _sync_recent_project_pages(self) -> None:
        """Compat shim — kept so any old call sites still work."""
        self._home_page.refresh_projects()

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
    ) -> tuple[int, float, float, float, float, float]:
        return (
            int(self._metrics_revision),
            float(render.threshold_green),
            float(render.threshold_red),
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
        """Compact vascular morphometry line for the metrics panel (best-effort)."""
        dataset = self.processed_dataset
        if dataset is None:
            return None
        threshold_red = float(self.current_render.threshold_red)
        if (
            self._vascular_summary_cache is not None
            and np.isclose(self._vascular_summary_cache[0], threshold_red, atol=1.0e-6)
        ):
            return self._vascular_summary_cache[1]
        try:
            vascular = analyze_vasculature(
                dataset.red.data,
                threshold=threshold_red,
                spacing=dataset.red.spacing,
                render=self.current_render,
            )
        except Exception as exc:  # pragma: no cover - defensive UI path
            self._log_debug(f"Vascular summary skipped: {exc}")
            self._vascular_summary_cache = (threshold_red, None)
            return None
        result = (
            f"vasculature: vol_fraction={vascular.vessel_volume_fraction:.4f}, "
            f"length_density={vascular.length_density_mm_per_mm3:.2f} mm/mm3, "
            f"mean_diameter_um={vascular.mean_diameter_um:.2f}, "
            f"junctions={vascular.junction_count}, "
            f"tortuosity={vascular.mean_tortuosity:.3f}"
        )
        self._vascular_summary_cache = (threshold_red, result)
        return result

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
        lines = [
            self._busy_base_message,
            f"Progress: {int(round(progress_percent))}%",
            f"Elapsed: {self._format_seconds(elapsed)}",
        ]

        with self._busy_progress_lock:
            eta_total = self._busy_eta_total
        if eta_total is not None:
            remaining = max(0.0, float(eta_total) - elapsed)
            lines.append(f"ETA: {self._format_seconds(remaining)}")
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
            self._busy_dialog.setMinimumDuration(0)
            self._busy_dialog.setWindowModality(Qt.WindowModal)
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
            "Use folder process for an auto-detected dataset folder, or select individual red/green TIFF files or sequence folders."
        )
        folder_btn = chooser.addButton("Folder Process", QMessageBox.ButtonRole.AcceptRole)
        manual_btn = chooser.addButton("Individual TIFF/Sequence", QMessageBox.ButtonRole.ActionRole)
        cancel_btn = chooser.addButton(QMessageBox.StandardButton.Cancel)
        chooser.exec()

        clicked = chooser.clickedButton()
        if clicked is None or clicked == cancel_btn:
            return None
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

        self._log_info("Load step 4/5: preparing mesh dataset...")
        visual_dataset = prepare_dataset_for_mesh(processed_dataset, preprocess_cfg)
        self._log_info("Load step 4/5 complete.")
        self._publish_busy_progress(percent=86.0, message="Computing default thresholds...")

        self._log_info("Load step 5/5: computing thresholds...")
        threshold_green = default_green_threshold(processed_dataset.green.data)
        threshold_red = default_threshold(processed_dataset.red.data)
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
        self.visual_dataset = result.visual_dataset
        self._invalidate_microglia_components()
        self._push_scene_channels()
        self._publish_busy_progress(percent=97.0, message="Applying initial thresholds + metrics...")
        self._set_busy_message("Applying initial thresholds and computing metrics...")
        # set_threshold_defaults emits render_config_changed → _on_render_config_changed which
        # already calls apply_render_config + _refresh_metrics.  Do not repeat them here.
        self.controls.set_threshold_defaults(result.threshold_green, result.threshold_red)
        self.controls.set_microglia_enhancement_enabled(True)
        self._publish_busy_progress(percent=100.0, message="Load complete.")
        self._log_info("Dataset load and initial render completed.")
        # Navigate to workspace and record in recent projects
        self._nav_to(1)
        self._save_current_project_state()
        self._record_current_project(samples=1, status="Open")
        self._refresh_section_state()

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
        self._publish_busy_progress(percent=98.0, message="Refreshing render + metrics...")
        self._set_busy_message("Refreshing render + metrics...")
        self.controls.set_microglia_enhancement_enabled(True)
        self.scene.apply_render_config(self.current_render)
        self._refresh_microglia_analysis_debug()
        self._refresh_metrics()
        self._publish_busy_progress(percent=100.0, message="Processing complete.")
        self._log_info("Processing applied and scene refreshed.")
        self._save_current_project_state()
        self._record_current_project(samples=1, status="Open")
        self._refresh_section_state()

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

    def _load_dataset_with_manual_fallback(self, root: Path):
        # Kept for compatibility with older code paths.
        try:
            self._log_debug("Attempting dataset auto-detection.")
            return load_dataset(root, spacing=self.spacing)
        except FileNotFoundError:
            self._log_info("Auto-detection failed; requesting channel sources in explicit order.")
            channel_overrides = self._prompt_channel_sources_in_order(root)
            if channel_overrides is None:
                raise RuntimeError("Channel source selection canceled.")
            return load_dataset(
                root,
                spacing=self.spacing,
                channel_overrides=channel_overrides,
            )

    def _on_psf_config_changed(self, config: PSFConfig) -> None:
        self.current_psf = config
        if config.iterations >= 8:
            self.statusBar().showMessage(
                "High RL iterations can take several minutes on large stacks.",
                4000,
            )

    def _on_preprocess_config_changed(self, config: PreprocessConfig) -> None:
        self._log_info("Preprocessing controls are disabled; configuration change ignored.")

    def _on_preview_green_denoise_requested(self) -> None:
        self.statusBar().showMessage("Green pass-through mode is always active.", 5000)

    def _on_apply_green_denoise_requested(self) -> None:
        self.statusBar().showMessage("Green denoise/masking is disabled in pass-through mode.", 5000)

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

    def _apply_psf_and_refresh(self, update_thresholds: bool) -> None:
        # Kept for compatibility with older code paths.
        assert self.synced_dataset is not None
        self._set_busy_message("Applying pass-through/processing pipeline...")
        psf_cfg = self._effective_psf_config(self.current_psf)
        preprocessed = preprocess_dataset(self.synced_dataset, self.preprocess_config) if self.preprocess_config.enabled else self.synced_dataset
        self.processed_dataset = apply_psf_to_dataset(
            preprocessed,
            psf_cfg,
            preprocess_config=self.preprocess_config,
        )
        self._mark_metrics_dirty()
        self.visual_dataset = prepare_dataset_for_mesh(self.processed_dataset, self.preprocess_config)
        self._invalidate_microglia_components()
        self._log_debug(
            f"Processed dataset shapes - green={self.processed_dataset.green.data.shape}, "
            f"red={self.processed_dataset.red.data.shape}"
        )

        self._push_scene_channels()

        if update_thresholds:
            self._set_busy_message("Computing initial Otsu thresholds...")
            tg = default_green_threshold(self.processed_dataset.green.data)
            tr = default_threshold(self.processed_dataset.red.data)
            self.controls.set_threshold_defaults(tg, tr)
            self._log_info(f"Default thresholds set: green={tg:.4f}, red={tr:.4f}")

        self._set_busy_message("Refreshing render + metrics...")
        self.scene.apply_render_config(self.current_render)
        self._refresh_metrics()

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
            return
        if isolate_enabled:
            self._refresh_microglia_components_if_needed()
        if data_upload_changed:
            self._push_scene_channels()
        self.scene.apply_render_config(config)
        self._refresh_microglia_analysis_debug()
        if metrics_changed:
            self._refresh_metrics()

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

        def _enhance_task() -> DatasetVolume:
            self._publish_busy_progress(percent=8.0, message="Estimating green background...")
            enhanced_green = enhance_microglia_background(
                dataset.green.data,
                preprocess_cfg,
                method=enhancement_method,
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

        self._start_background_task(
            title="Enhance Microglia",
            message=f"Enhancing microglia with {enhancement_method}...",
            fn=_enhance_task,
            on_success=self._on_enhance_microglia_success,
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
        threshold_green = default_green_threshold(result.green.data)
        threshold_red = default_threshold(result.red.data)
        self.controls.set_threshold_defaults(threshold_green, threshold_red)
        self._publish_busy_progress(percent=98.0, message="Refreshing enhanced render...")
        self.scene.apply_render_config(self.current_render)
        self._refresh_microglia_analysis_debug()
        self._refresh_metrics()
        self._publish_busy_progress(percent=100.0, message="Microglia enhancement complete.")
        self._log_info(
            "Microglia enhancement applied: "
            f"green_shape={result.green.data.shape} thresholds=(green={threshold_green:.4f}, red={threshold_red:.4f})"
        )

    def _refresh_metrics(self) -> None:
        if self.processed_dataset is None:
            return
        cache_key = self._metrics_cache_key_for_render(self.current_render)
        if self.latest_metrics is None or self._metrics_cache_key != cache_key:
            self.latest_metrics = compute_metrics(self.processed_dataset, self.current_render)
            self._metrics_cache_key = cache_key
        self._set_metrics_text_from_result(self.latest_metrics)
        self._metrics_panel.update_from_metrics(self.latest_metrics)
        self._refresh_analytics_metrics()
        self._log_debug("Metrics updated.")

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
                assoc = summarize_neurovascular_association(self.latest_microglia_analysis)
                npath = base.with_name(f"{stem}_neurovascular.csv")
                export_metrics_csv(neurovascular_association_to_csv_rows(assoc), npath)
                written.append(npath)
            except Exception as exc:  # pragma: no cover - defensive UI path
                self._log_info(f"Neurovascular metrics export skipped: {exc}")

        return written

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
