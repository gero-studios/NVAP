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
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSplitter,
)

from nvap.analysis.metrics import compute_metrics, metrics_to_csv_rows
from nvap.cache.processed_cache import (
    build_dataset_signature,
    build_processed_cache_key,
    has_processed_cache,
    load_processed_dataset,
    save_processed_dataset,
)
from nvap.config.types import (
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
from nvap.analysis.microglia_components import compute_component_labels, isolate_component
from nvap.preprocess.enhancement import preprocess_dataset
from nvap.plugins.registry import discover_plugins
from nvap.render.vtk_scene import VTKScene
from nvap.ui.control_panel import ControlPanel

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
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NVAP - NeuroVascular Analytics Program")
        self.resize(1540, 920)

        self.scene = VTKScene(self)
        self.controls = ControlPanel(self)
        splitter = QSplitter(self)
        splitter.addWidget(self.controls)
        splitter.addWidget(self.scene.widget())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.spacing = DEFAULT_SPACING
        self.preprocess_config = PreprocessConfig(enabled=True)
        self.synced_dataset: DatasetVolume | None = None
        self.raw_dataset: DatasetVolume | None = None
        self.processed_dataset: DatasetVolume | None = None
        self.visual_dataset: DatasetVolume | None = None
        self.current_psf = self.controls.current_psf_config()
        self.current_render = self.controls.current_render_config()
        self.latest_metrics: MetricsComputation | None = None
        self.dataset_root: Path | None = None
        self._dataset_signature: str | None = None
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
        self._microglia_isolate_active = bool(self.controls.microglia_view_state()[0])
        self._microglia_label_cache_max_bytes = 256 * 1024 * 1024

        self._log_bridge = _LogBridge(self)
        self._log_bridge.message.connect(self.controls.append_debug_text)
        self._log_handler = _ControlPanelLogHandler(self._log_bridge.message.emit)
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logging.getLogger("nvap").addHandler(self._log_handler)

        self.controls.load_requested.connect(self._on_load_requested)
        self.controls.apply_psf_requested.connect(self._on_apply_psf_requested)
        self.controls.psf_config_changed.connect(self._on_psf_config_changed)
        self.controls.render_config_changed.connect(self._on_render_config_changed)
        self.controls.microglia_view_changed.connect(self._on_microglia_view_changed)
        self.controls.export_metrics_requested.connect(self._on_export_metrics_requested)
        self.controls.export_snapshot_requested.connect(self._on_export_snapshot_requested)
        self.controls.export_mesh_requested.connect(self._on_export_mesh_requested)

        self._refresh_plugin_panel()
        self.statusBar().showMessage("Load a dataset to begin. Preprocessing is ON by default.")
        self._log_info("NVAP UI initialized: green pass-through mode active.")

    def closeEvent(self, event) -> None:
        if self._active_thread is not None and self._active_thread.isRunning():
            self._log_info("Waiting briefly for active background task to finish before close.")
            self._active_thread.wait(2000)
        logging.getLogger("nvap").removeHandler(self._log_handler)
        super().closeEvent(event)

    def _log_info(self, message: str) -> None:
        logger.info(message)

    def _log_debug(self, message: str) -> None:
        logger.debug(message)

    def _display_spacing(self, spacing: VoxelSpacing) -> VoxelSpacing:
        # Visual-only Z squeeze for less depth exaggeration. Metrics stay in physical units.
        z_scale = float(max(0.05, self._display_z_scale))
        return VoxelSpacing(
            x_um=spacing.x_um,
            y_um=spacing.y_um,
            z_um=float(spacing.z_um) * z_scale,
        )

    def _invalidate_microglia_components(self) -> None:
        self._green_component_labels = None
        self._green_component_order = np.empty((0,), dtype=np.int32)
        self._green_component_sizes = None
        self._green_component_threshold = None
        self._green_component_branch_sensitivity = None
        self._green_component_shape = None
        self._green_component_sparse = {}
        self.controls.set_microglia_component_summary(0, 0, 0)
        self.controls.microglia_info.setText("Enable 'View one microglia' to detect components.")

    def _compute_microglia_components(
        self,
        green: np.ndarray,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        branch_sense = float(self.controls.current_microglia_branch_sensitivity())
        base_min_voxels = max(64, int(self.preprocess_config.green_speckle_min_voxels) * 4)
        min_voxels = int(round(base_min_voxels / (0.85 + (0.35 * branch_sense))))
        min_voxels = max(32, min_voxels)
        if green.size >= 120 * 1024 * 1024:
            min_voxels = max(min_voxels, 256)
        return compute_component_labels(
            green,
            threshold=threshold,
            min_voxels=min_voxels,
            max_components=256,
            smooth_sigma=(0.2, 0.45, 0.45),
            branch_sensitivity=branch_sense,
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
        label_count = int(len(order))
        if label_count <= int(np.iinfo(np.uint8).max):
            label_dtype = np.uint8
        elif label_count <= int(np.iinfo(np.uint16).max):
            label_dtype = np.uint16
        else:
            label_dtype = np.uint32
        estimated_bytes = int(green.size * np.dtype(label_dtype).itemsize)
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
        self._green_component_shape = green.shape
        selected = min(selected_index, int(len(order)))
        selected_voxels = int(sizes[int(order[selected - 1])]) if selected > 0 else 0
        self.controls.set_microglia_component_summary(
            count=int(len(order)),
            selected_index=selected,
            selected_voxels=selected_voxels,
        )

    def _current_green_volume_for_view(self) -> np.ndarray:
        if self.visual_dataset is None:
            raise RuntimeError("No visual dataset loaded.")
        base = np.asarray(self.visual_dataset.green.data, dtype=np.float32)
        enabled, selected_index = self.controls.microglia_view_state()
        if not enabled or selected_index <= 0:
            return base

        self._refresh_microglia_components_if_needed()
        if self._green_component_sizes is None:
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
        self._set_busy_message("Uploading green channel to VTK...")
        self.scene.set_channel_data(
            channel="green",
            volume=self._current_green_volume_for_view(),
            spacing=self._display_spacing(self.visual_dataset.green.spacing),
        )
        if green_only:
            return
        self._set_busy_message("Uploading red channel to VTK...")
        self.scene.set_channel_data(
            channel="red",
            volume=self.visual_dataset.red.data,
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

        eta_total = self._busy_eta_total
        if progress_percent >= 1.0 and elapsed >= 1.0:
            pace_total = elapsed / max(1.0e-3, progress_percent / 100.0)
            if eta_total is None:
                eta_total = pace_total
            else:
                # Blend model estimate with observed pace so ETA can both increase and decrease.
                eta_total = (0.35 * float(eta_total)) + (0.65 * float(pace_total))
        if eta_total is not None:
            remaining = max(0.0, eta_total - elapsed)
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
            cache_hit = has_processed_cache(cache_key)
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
        total *= self._eta_scale_load
        total = max(5.0, total)
        self._log_info(
            "Estimated load ETA="
            f"{total:.1f}s (scale={self._eta_scale_load:.2f}, "
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
        if _green_no_psf_mode(preprocess_cfg) or not psf_cfg.enabled or psf_cfg.iterations <= 0:
            total_voxels = int(dataset.green.data.size + dataset.red.data.size)
            preprocess_seconds = total_voxels * 1.8e-7
            threshold_seconds = total_voxels * 5.0e-8
            resample_seconds = total_voxels * 2.0e-8 if preprocess_cfg.resample_for_mesh else 0.0
            total = (preprocess_seconds + threshold_seconds + resample_seconds + 8.0) * self._eta_scale_psf
            total = max(8.0, total)
            self._log_info(
                "Estimated reprocess ETA="
                f"{total:.1f}s (scale={self._eta_scale_psf:.2f}, mode={preprocess_cfg.green_denoise_strategy})"
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
            cache_hit = has_processed_cache(cache_key)
        if cache_hit:
            self._log_info("Estimated PSF ETA=6.0s (cache hit).")
            return 6.0
        total_voxels = int(dataset.green.data.size + dataset.red.data.size)
        preprocess_seconds = total_voxels * 1.3e-7 if preprocess_cfg.enabled else 0.0
        psf_seconds = total_voxels * psf_cfg.iterations * 1.2e-7
        threshold_seconds = total_voxels * 5.0e-8
        resample_seconds = total_voxels * 2.0e-8 if preprocess_cfg.resample_for_mesh else 0.0
        total = (preprocess_seconds + psf_seconds + threshold_seconds + resample_seconds + 6.0) * self._eta_scale_psf
        total = max(6.0, total)
        self._log_info(
            "Estimated PSF ETA="
            f"{total:.1f}s (scale={self._eta_scale_psf:.2f}, "
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
        if eta_hint is not None:
            self._busy_eta_total = eta_hint
        elapsed = time.perf_counter() - self._busy_start
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
        self._busy_eta_total = eta_total_seconds
        self._busy_start = time.perf_counter()
        with self._busy_progress_lock:
            self._busy_progress_percent = 0.0
            self._busy_progress_message = message
            self._busy_progress_eta_total = eta_total_seconds
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
                if eta_kind == "load":
                    scale = self._eta_scale_load
                    scale = max(0.5, min(2.5, (0.8 * scale) + (0.2 * ratio)))
                    self._eta_scale_load = scale
                    self._log_debug(
                        f"Load ETA calibration updated: ratio={ratio:.2f}, scale={self._eta_scale_load:.2f}"
                    )
                elif eta_kind == "psf":
                    scale = self._eta_scale_psf
                    scale = max(0.5, min(2.5, (0.8 * scale) + (0.2 * ratio)))
                    self._eta_scale_psf = scale
                    self._log_debug(
                        f"PSF ETA calibration updated: ratio={ratio:.2f}, scale={self._eta_scale_psf:.2f}"
                    )
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
            return
        lines = []
        for plugin in plugins:
            if plugin.status == "loaded":
                lines.append(f"- {plugin.plugin_id} ({plugin.target_channel}) loaded")
            else:
                lines.append(f"- {plugin.plugin_id} error: {plugin.error}")
        self._log_info(f"Discovered {len(plugins)} plugin descriptor(s).")
        self.controls.set_plugin_text("\n".join(lines))

    def _prompt_channel_source(self, channel_label: str, start_dir: Path) -> str | None:
        chooser = QMessageBox(self)
        chooser.setWindowTitle("Select Channel Source Type")
        chooser.setText(f"Choose source type for {channel_label}.")
        chooser.setInformativeText(
            "Use a single TIFF/PNG stack file, or a folder of sequenced images."
        )
        file_btn = chooser.addButton("Single Stack File", QMessageBox.ButtonRole.AcceptRole)
        folder_btn = chooser.addButton("Image Sequence Folder", QMessageBox.ButtonRole.ActionRole)
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

    def _on_load_requested(self) -> None:
        base = self.dataset_root or (Path.cwd() / "Input")
        channel_overrides = self._prompt_channel_sources_in_order(base)
        if channel_overrides is None:
            return

        channel_dirs: dict[str, Path]
        first_red = Path(channel_overrides["red"]).resolve()
        self.dataset_root = first_red.parent
        self._log_info(
            f"Channel sources selected: red={Path(channel_overrides['red']).resolve()} "
            f"green={Path(channel_overrides['green']).resolve()}"
        )
        channel_dirs = resolve_channel_dirs(self.dataset_root, channel_overrides=channel_overrides)

        self._dataset_signature = build_dataset_signature(channel_dirs)
        self._log_debug(f"Dataset signature set: {self._dataset_signature}")

        root = self.dataset_root
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
        self._start_background_task(
            title="Load Dataset",
            message="Loading stacks and processing...",
            fn=lambda: self._background_load_dataset(root, channel_overrides, psf_cfg, preprocess_cfg),
            on_success=self._on_load_task_success,
            error_title="Dataset load failed",
            success_status=f"Loaded dataset: {self.dataset_root}",
            eta_total_seconds=eta_seconds,
            eta_kind="load",
        )

    def _background_load_dataset(
        self,
        root: Path,
        channel_overrides: dict[str, str | Path] | None,
        psf_cfg: PSFConfig,
        preprocess_cfg: PreprocessConfig,
    ) -> _LoadTaskResult:
        t0 = time.perf_counter()
        self._publish_busy_progress(percent=2.0, message="Reading channel stacks...")
        self._log_info("Load step 1/6: reading channel stacks...")
        dataset = load_dataset(root, spacing=self.spacing, channel_overrides=channel_overrides)
        self._log_info("Load step 1/6 complete.")
        self._publish_busy_progress(percent=16.0, message="Synchronizing channels...")

        self._log_info("Load step 2/6: filling missing slices...")
        synced_dataset = fill_and_sync_dataset(dataset)
        self._log_info("Load step 2/6 complete.")
        self._publish_busy_progress(percent=26.0, message="Preprocessing (green pass-through + red cleanup)...")

        self._log_info("Load step 3/6: preprocessing (green pass-through + red cleanup)...")
        self.preprocess_config = preprocess_cfg
        if preprocess_cfg.enabled:
            preprocessed_dataset = preprocess_dataset(synced_dataset, preprocess_cfg)
            self._log_info("Load step 3/6 complete (preprocessing applied).")
        else:
            preprocessed_dataset = synced_dataset
            self._log_info("Load step 3/6 skipped (preprocessing disabled).")
        self._publish_busy_progress(percent=54.0, message="Applying cached/processed volume...")

        self._log_info("Load step 4/6: applying cached/processed volume...")
        processed_dataset = self._get_processed_dataset_with_cache(
            preprocessed_dataset,
            psf_cfg,
            preprocess_cfg,
            self._dataset_signature,
            cancel_event=None,
            progress_bounds=(54.0, 74.0),
        )
        self._log_info("Load step 4/6 complete.")
        self._publish_busy_progress(percent=76.0, message="Preparing mesh dataset...")

        self._log_info("Load step 5/6: preparing mesh dataset...")
        visual_dataset = prepare_dataset_for_mesh(processed_dataset, preprocess_cfg)
        self._log_info("Load step 5/6 complete.")
        self._publish_busy_progress(percent=86.0, message="Computing default thresholds...")

        self._log_info("Load step 6/6: computing thresholds...")
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
            raw_dataset=preprocessed_dataset,
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
        self.visual_dataset = result.visual_dataset
        self._invalidate_microglia_components()
        self._push_scene_channels()
        self._publish_busy_progress(percent=97.0, message="Applying initial thresholds + metrics...")
        self._set_busy_message("Applying initial thresholds and computing metrics...")
        self.controls.set_threshold_defaults(result.threshold_green, result.threshold_red)
        self.scene.apply_render_config(self.current_render)
        self._refresh_metrics()
        self._publish_busy_progress(percent=100.0, message="Load complete.")
        self._log_info("Dataset load and initial render completed.")

    def _on_psf_task_success(self, result: object) -> None:
        if not isinstance(result, DatasetVolume):
            raise TypeError("Invalid processing task result payload.")
        self._publish_busy_progress(percent=90.0, message="Preparing render dataset...")
        self._microglia_isolate_active = bool(self.controls.microglia_view_state()[0])
        self.processed_dataset = result
        self.visual_dataset = prepare_dataset_for_mesh(self.processed_dataset, self.preprocess_config)
        self._publish_busy_progress(percent=94.0, message="Uploading channels to VTK...")
        self._invalidate_microglia_components()
        self._push_scene_channels()
        self._publish_busy_progress(percent=98.0, message="Refreshing render + metrics...")
        self._set_busy_message("Refreshing render + metrics...")
        self.scene.apply_render_config(self.current_render)
        self._refresh_metrics()
        self._publish_busy_progress(percent=100.0, message="Processing complete.")
        self._log_info("Processing applied and scene refreshed.")

    def _get_processed_dataset_with_cache(
        self,
        raw_dataset: DatasetVolume,
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
            cached = load_processed_dataset(cache_key, self.spacing)
            if cached is not None:
                publish_process_progress(1.0, "Loaded processed dataset from cache.")
                self._log_info("Using cached processed dataset.")
                return cached
            self._log_info(f"Cache pipeline: miss key={cache_key}")
        publish_process_progress(0.08, "Running processing pipeline...")

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
                0.12 + (0.74 * avg_frac),
                message=f"Processing {channel}: {current}/{total}",
            )
            if emit_log:
                self._log_info(f"Load progress: PSF {channel} {current}/{total} ({percent}%)")

        processed = apply_psf_to_dataset(
            raw_dataset,
            psf_cfg,
            preprocess_config=preprocess_cfg,
            cancel_event=cancel_event,
            progress_callback=on_psf_progress,
        )
        publish_process_progress(0.90, "Finalizing processed dataset...")
        self._log_info(f"Cache pipeline: PSF complete dt={time.perf_counter() - t_psf:.2f}s")
        if cache_key is not None and (cancel_event is None or not cancel_event.is_set()):
            publish_process_progress(0.95, "Saving processed dataset to cache...")
            self._log_info("Cache pipeline: saving processed dataset to cache.")
            save_processed_dataset(cache_key, processed)
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
            preprocessed = preprocess_dataset(synced_dataset, preprocess_cfg) if preprocess_cfg.enabled else synced_dataset
            return self._get_processed_dataset_with_cache(
                preprocessed,
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
        self._display_z_scale = new_z_scale
        if self.processed_dataset is None:
            return
        threshold_changed = not np.isclose(
            float(previous.threshold_green),
            float(config.threshold_green),
            atol=1.0e-6,
        )
        if threshold_changed:
            self._invalidate_microglia_components()
        isolate_enabled, _ = self.controls.microglia_view_state()
        if isolate_enabled:
            self._refresh_microglia_components_if_needed()
        if z_scale_changed:
            self._push_scene_channels()
        elif isolate_enabled and threshold_changed:
            self._push_scene_channels(green_only=True)
        self.scene.apply_render_config(config)
        self._refresh_metrics()

    def _on_microglia_view_changed(self) -> None:
        if self.visual_dataset is None or self.processed_dataset is None:
            return
        isolate_enabled, _ = self.controls.microglia_view_state()
        was_enabled = bool(self._microglia_isolate_active)
        self._microglia_isolate_active = bool(isolate_enabled)
        if isolate_enabled:
            self._refresh_microglia_components_if_needed()
            self._push_scene_channels(green_only=True)
        elif was_enabled:
            # Only push full green when isolate mode is being turned OFF.
            self._push_scene_channels(green_only=True)
        else:
            # Ignore slider/index changes while isolate mode is already disabled.
            return
        self.scene.apply_render_config(self.current_render)

    def _refresh_metrics(self) -> None:
        if self.processed_dataset is None:
            return
        self.latest_metrics = compute_metrics(self.processed_dataset, self.current_render)
        lines = []
        for item in self.latest_metrics.channel_results:
            lines.append(
                (
                    f"{item.channel}: voxels={item.voxel_count}, "
                    f"volume_um3={item.volume_um3:.3f}, "
                    f"components={item.component_count}, "
                    f"largest_component={item.largest_component_voxels}"
                )
            )
        lines.append(
            f"overlap: voxels={self.latest_metrics.overlap_voxel_count}, "
            f"volume_um3={self.latest_metrics.overlap_volume_um3:.3f}"
        )
        self.controls.set_metrics_text("\n".join(lines))
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
                self._publish_busy_progress(percent=60.0, message="Writing CSV file...")
                out = export_metrics_csv(rows, file_path)
                self._publish_busy_progress(percent=100.0, message="Metrics export complete.")
            self.statusBar().showMessage(f"Metrics exported to {out}", 5000)
            self._log_info(f"Metrics exported: {out}")
        except Exception as exc:
            self._log_info(f"Metrics export failed: {exc}")
            self._show_error("Export failed", str(exc))

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
        )

    def _on_mesh_export_success(self, result: object) -> None:
        if isinstance(result, dict):
            paths = [str(p) for p in result.values()]
            self._log_info(f"Meshes exported: {', '.join(paths)}")
            self.statusBar().showMessage(f"Exported {len(result)} mesh file(s)", 5000)
