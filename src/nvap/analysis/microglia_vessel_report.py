from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import logging
from typing import Literal

import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import skeletonize

try:
    from skimage.morphology import skeletonize_3d as _skeletonize_3d
except ImportError:  # scikit-image>=0.25 removed skeletonize_3d
    _skeletonize_3d = None

from nvap.analysis.microglia_components import compute_component_labels
from nvap.config.types import DatasetVolume, PreprocessConfig, RenderConfig, VoxelSpacing
from nvap.pipeline import default_green_threshold, default_threshold
from nvap.preprocess.microglia_masking import mask_green_volume_with_microglia_bundle

logger = logging.getLogger(__name__)

SegmentationEngine = Literal["internal", "fiji"]
SegmentationMode = Literal["auto", "internal", "fiji"]
AnalysisThresholdSource = Literal["adaptive", "render"]

MICROGLIA_CELL_REPORT_COLUMNS = [
    "cell_index",
    "component_id",
    "segmentation_engine_used",
    "voxel_count",
    "volume_um3",
    "branch_endpoint_count",
    "branch_junction_count",
    "distance_to_vasculature_um",
    "microglia_closest_x_um",
    "microglia_closest_y_um",
    "microglia_closest_z_um",
    "vessel_closest_x_um",
    "vessel_closest_y_um",
    "vessel_closest_z_um",
    "threshold_green_used",
    "threshold_red_used",
]

_CC_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)
_DEGREE_KERNEL = np.ones((3, 3, 3), dtype=np.uint8)
_AUTO_FIJI_FALLBACK_MAX_VOXELS = 96 * 1024 * 1024


@dataclass(frozen=True)
class MicrogliaCellReportRow:
    cell_index: int
    component_id: int
    segmentation_engine_used: SegmentationEngine
    voxel_count: int
    volume_um3: float
    branch_endpoint_count: int
    branch_junction_count: int
    distance_to_vasculature_um: float
    microglia_closest_x_um: float
    microglia_closest_y_um: float
    microglia_closest_z_um: float
    vessel_closest_x_um: float
    vessel_closest_y_um: float
    vessel_closest_z_um: float
    threshold_green_used: float
    threshold_red_used: float


@dataclass(frozen=True)
class MicrogliaCellReport:
    rows: list[MicrogliaCellReportRow] = field(default_factory=list)
    segmentation_engine_used: SegmentationEngine = "internal"
    threshold_green_used: float = 0.0
    threshold_red_used: float = 0.0

    @property
    def cell_count(self) -> int:
        return int(len(self.rows))


def _min_voxels_for_analysis(preprocess_config: PreprocessConfig, branch_sensitivity: float = 1.0) -> int:
    base_min_voxels = max(24, int(preprocess_config.green_speckle_min_voxels) * 2)
    sens = float(np.clip(branch_sensitivity, 0.4, 2.0))
    min_voxels = int(round(base_min_voxels / (0.90 + (0.40 * sens))))
    return max(12, min_voxels)


def _component_labels_from_volume(
    green_volume: np.ndarray,
    threshold: float,
    preprocess_config: PreprocessConfig,
    spacing: VoxelSpacing | None = None,
    branch_sensitivity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_voxels = _min_voxels_for_analysis(preprocess_config, branch_sensitivity=branch_sensitivity)
    if green_volume.size >= 120 * 1024 * 1024:
        min_voxels = max(min_voxels, 256)
    sp = (
        (float(spacing.z_um), float(spacing.y_um), float(spacing.x_um))
        if spacing is not None
        else None
    )
    return compute_component_labels(
        np.asarray(green_volume, dtype=np.float32),
        threshold=float(threshold),
        min_voxels=min_voxels,
        max_components=512,
        smooth_sigma=(0.2, 0.45, 0.45),
        branch_sensitivity=float(branch_sensitivity),
        spacing=sp,
    )


def _surface_mask(mask: np.ndarray) -> np.ndarray:
    filled = np.asarray(mask, dtype=bool)
    if not np.any(filled):
        return np.zeros_like(filled, dtype=bool)
    eroded = ndi.binary_erosion(filled, structure=_CC_STRUCTURE, border_value=0)
    return filled & (~eroded)


def _voxel_to_xyz_um(
    voxel_zyx: tuple[int, int, int],
    spacing: VoxelSpacing,
    z_values: list[int] | None = None,
) -> tuple[float, float, float]:
    z, y, x = voxel_zyx
    if z_values is not None and 0 <= int(z) < len(z_values):
        z_coord = float(z_values[int(z)])
    else:
        z_coord = float(z)
    return (
        float(x) * float(spacing.x_um),
        float(y) * float(spacing.y_um),
        z_coord * float(spacing.z_um),
    )


def _branch_counts(component_mask: np.ndarray) -> tuple[int, int]:
    binary = np.asarray(component_mask, dtype=bool)
    if not np.any(binary):
        return 0, 0
    if _skeletonize_3d is not None:
        skeleton = np.asarray(_skeletonize_3d(binary), dtype=bool)
    else:
        skeleton = np.asarray(skeletonize(binary), dtype=bool)
    if not np.any(skeleton):
        return 0, 0
    degree_map = ndi.convolve(
        skeleton.astype(np.uint8, copy=False),
        _DEGREE_KERNEL,
        mode="constant",
        cval=0,
    ).astype(np.int16, copy=False)
    degree_map = degree_map - skeleton.astype(np.int16, copy=False)
    skeleton_degrees = degree_map[skeleton]
    endpoints = int(np.count_nonzero(skeleton_degrees == 1))
    junctions = int(np.count_nonzero(skeleton_degrees >= 3))
    return endpoints, junctions


def _coords_from_local_mask(
    local_mask: np.ndarray,
    comp_slice: tuple[slice, slice, slice],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = np.nonzero(local_mask)
    if len(local[0]) == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty, empty
    z0 = int(comp_slice[0].start or 0)
    y0 = int(comp_slice[1].start or 0)
    x0 = int(comp_slice[2].start or 0)
    return (
        local[0].astype(np.int64, copy=False) + z0,
        local[1].astype(np.int64, copy=False) + y0,
        local[2].astype(np.int64, copy=False) + x0,
    )


def _closest_surface_pair_from_coords(
    component_coords: tuple[np.ndarray, np.ndarray, np.ndarray],
    component_surface_coords: tuple[np.ndarray, np.ndarray, np.ndarray],
    red_mask: np.ndarray,
    vessel_surface_dist_um: np.ndarray | None,
    vessel_surface_indices: np.ndarray | None,
) -> tuple[float, tuple[int, int, int], tuple[int, int, int] | None]:
    if component_coords[0].size <= 0:
        return float("nan"), (0, 0, 0), None

    touching = red_mask[component_coords]
    if np.any(touching):
        touch_idx = int(np.flatnonzero(touching)[0])
        micro_idx = (
            int(component_coords[0][touch_idx]),
            int(component_coords[1][touch_idx]),
            int(component_coords[2][touch_idx]),
        )
        return 0.0, micro_idx, micro_idx

    if vessel_surface_dist_um is None or vessel_surface_indices is None:
        if component_surface_coords[0].size > 0:
            micro_idx = (
                int(component_surface_coords[0][0]),
                int(component_surface_coords[1][0]),
                int(component_surface_coords[2][0]),
            )
        else:
            micro_idx = (
                int(component_coords[0][0]),
                int(component_coords[1][0]),
                int(component_coords[2][0]),
            )
        return float("nan"), micro_idx, None

    candidate_coords = component_surface_coords
    if candidate_coords[0].size <= 0:
        candidate_coords = component_coords
    candidate_dist = vessel_surface_dist_um[candidate_coords]
    best_local = int(np.argmin(candidate_dist))
    micro_idx = (
        int(candidate_coords[0][best_local]),
        int(candidate_coords[1][best_local]),
        int(candidate_coords[2][best_local]),
    )
    best_distance = float(candidate_dist[best_local])
    vessel_idx = (
        int(vessel_surface_indices[0][micro_idx]),
        int(vessel_surface_indices[1][micro_idx]),
        int(vessel_surface_indices[2][micro_idx]),
    )
    return best_distance, micro_idx, vessel_idx


def _run_internal_segmentation(
    green_volume: np.ndarray,
    threshold_green: float,
    preprocess_config: PreprocessConfig,
    spacing: VoxelSpacing | None = None,
    branch_sensitivity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels, order, sizes = _component_labels_from_volume(
        green_volume, threshold_green, preprocess_config,
        spacing=spacing, branch_sensitivity=branch_sensitivity,
    )
    return labels, order, sizes


def _run_fiji_segmentation(
    green_volume: np.ndarray,
    threshold_green: float,
    preprocess_config: PreprocessConfig,
    spacing: VoxelSpacing | None = None,
    branch_sensitivity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masked = mask_green_volume_with_microglia_bundle(np.asarray(green_volume, dtype=np.float32))
    labels, order, sizes = _component_labels_from_volume(
        masked, threshold_green, preprocess_config,
        spacing=spacing, branch_sensitivity=branch_sensitivity,
    )
    return labels, order, sizes


def _compute_vessel_surface_distance_maps(
    red_mask: np.ndarray,
    spacing: VoxelSpacing,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    vessel_surface = _surface_mask(red_mask)
    if not np.any(vessel_surface):
        return None, None
    distances, indices = ndi.distance_transform_edt(
        ~vessel_surface,
        sampling=(float(spacing.z_um), float(spacing.y_um), float(spacing.x_um)),
        return_distances=True,
        return_indices=True,
    )
    return np.asarray(distances, dtype=np.float32), np.asarray(indices, dtype=np.int64)


def _aligned_shared_volumes(dataset: DatasetVolume) -> tuple[np.ndarray, np.ndarray, list[int]]:
    z0, z1 = dataset.shared_z_range
    shared_values = list(range(int(z0), int(z1) + 1))
    g_map = {int(z): i for i, z in enumerate(dataset.green.z_indices)}
    r_map = {int(z): i for i, z in enumerate(dataset.red.z_indices)}

    green_shared = np.stack(
        [np.asarray(dataset.green.data[g_map[z]], dtype=np.float32) for z in shared_values],
        axis=0,
    )
    red_shared = np.stack(
        [np.asarray(dataset.red.data[r_map[z]], dtype=np.float32) for z in shared_values],
        axis=0,
    )
    return green_shared, red_shared, shared_values


def _estimate_seed_count(
    green_volume: np.ndarray,
    threshold_green: float,
) -> int:
    arr = np.asarray(green_volume, dtype=np.float32)
    positive = arr[np.isfinite(arr) & (arr > 0.0)]
    if positive.size <= 0:
        return 0
    seed_source = ndi.gaussian_filter(arr, sigma=(0.35, 0.85, 0.85), mode="nearest")
    seed_floor = float(max(threshold_green, np.quantile(positive, 0.90)))
    peak_mask = seed_source >= seed_floor
    if not np.any(peak_mask):
        return 0
    peak_max = ndi.maximum_filter(seed_source, size=(3, 5, 5), mode="nearest")
    np.subtract(peak_max, 1.0e-6, out=peak_max)
    np.greater_equal(seed_source, peak_max, out=peak_mask, where=peak_mask)
    _, count = ndi.label(peak_mask, structure=_CC_STRUCTURE)
    return int(count)


def _internal_result_looks_merged(
    green_volume: np.ndarray,
    threshold_green: float,
    order: np.ndarray,
    sizes: np.ndarray,
) -> bool:
    if int(len(order)) != 1:
        return False
    component_id = int(order[0])
    component_voxels = int(sizes[component_id]) if component_id < int(sizes.shape[0]) else 0
    if component_voxels <= 0:
        return False
    seed_count = _estimate_seed_count(green_volume, threshold_green)
    if seed_count >= 2:
        logger.info(
            "Internal segmentation looks merged: component_voxels=%d seed_count=%d",
            component_voxels,
            seed_count,
        )
        return True
    return False


def resolve_microglia_analysis_render(
    dataset: DatasetVolume,
    render: RenderConfig,
    threshold_source: AnalysisThresholdSource = "adaptive",
) -> RenderConfig:
    source = str(threshold_source).strip().lower()
    if source == "render":
        return render
    if source != "adaptive":
        raise ValueError("threshold_source must be one of: adaptive, render.")

    green, red, _ = _aligned_shared_volumes(dataset)
    return replace(
        render,
        threshold_green=default_green_threshold(green, fallback=float(render.threshold_green)),
        threshold_red=default_threshold(red, fallback=float(render.threshold_red)),
    )


def analyze_microglia_vessel(
    dataset: DatasetVolume,
    render: RenderConfig,
    preprocess_config: PreprocessConfig,
    segmentation_mode: SegmentationMode = "auto",
    branch_sensitivity: float = 1.0,
    threshold_source: AnalysisThresholdSource = "adaptive",
) -> MicrogliaCellReport:
    mode = str(segmentation_mode).strip().lower()
    if mode not in {"auto", "internal", "fiji"}:
        raise ValueError("segmentation_mode must be one of: auto, internal, fiji.")

    resolved_render = resolve_microglia_analysis_render(
        dataset,
        render,
        threshold_source=threshold_source,
    )
    threshold_green = float(resolved_render.threshold_green)
    threshold_red = float(resolved_render.threshold_red)
    green, red, shared_z_values = _aligned_shared_volumes(dataset)
    spacing = dataset.green.spacing

    labels: np.ndarray
    order: np.ndarray
    sizes: np.ndarray
    engine: SegmentationEngine

    if mode == "fiji":
        labels, order, sizes = _run_fiji_segmentation(
            green, threshold_green, preprocess_config,
            spacing=spacing, branch_sensitivity=branch_sensitivity,
        )
        engine = "fiji"
    else:
        labels, order, sizes = _run_internal_segmentation(
            green, threshold_green, preprocess_config,
            spacing=spacing, branch_sensitivity=branch_sensitivity,
        )
        engine = "internal"
        should_retry_with_fiji = (
            mode == "auto"
            and (
                int(len(order)) == 0
                or _internal_result_looks_merged(green, threshold_green, order, sizes)
            )
        )
        if should_retry_with_fiji:
            if int(green.size) > _AUTO_FIJI_FALLBACK_MAX_VOXELS:
                logger.info(
                    "Skipping auto fiji fallback for large volume (voxels=%d > %d). "
                    "Use segmentation_mode='fiji' to force fallback.",
                    int(green.size),
                    int(_AUTO_FIJI_FALLBACK_MAX_VOXELS),
                )
            else:
                try:
                    labels, order, sizes = _run_fiji_segmentation(
                        green, threshold_green, preprocess_config,
                        spacing=spacing, branch_sensitivity=branch_sensitivity,
                    )
                    engine = "fiji"
                    logger.info("Microglia report segmentation fallback: internal->fiji.")
                except Exception as exc:
                    logger.warning("Microglia report fallback to fiji failed: %s", exc)

    red_mask = red >= threshold_red
    vessel_surface_dist_um, vessel_surface_indices = _compute_vessel_surface_distance_maps(red_mask, spacing)

    rows: list[MicrogliaCellReportRow] = []
    voxel_volume_um3 = float(spacing.voxel_volume_um3)
    component_objects = ndi.find_objects(np.asarray(labels))
    for cell_index, component_id in enumerate(order.tolist(), start=1):
        comp_id = int(component_id)
        obj_idx = comp_id - 1
        if obj_idx < 0 or obj_idx >= len(component_objects):
            continue
        comp_slice = component_objects[obj_idx]
        if comp_slice is None:
            continue
        component_local = np.asarray(labels[comp_slice] == comp_id, dtype=bool)
        if not np.any(component_local):
            continue

        component_surface_local = _surface_mask(component_local)
        component_coords = _coords_from_local_mask(component_local, comp_slice)
        component_surface_coords = _coords_from_local_mask(component_surface_local, comp_slice)
        distance_um, micro_idx, vessel_idx = _closest_surface_pair_from_coords(
            component_coords=component_coords,
            component_surface_coords=component_surface_coords,
            red_mask=red_mask,
            vessel_surface_dist_um=vessel_surface_dist_um,
            vessel_surface_indices=vessel_surface_indices,
        )
        micro_xyz = _voxel_to_xyz_um(micro_idx, spacing, z_values=shared_z_values)
        if vessel_idx is None:
            vessel_xyz = (float("nan"), float("nan"), float("nan"))
        else:
            vessel_xyz = _voxel_to_xyz_um(vessel_idx, spacing, z_values=shared_z_values)

        endpoint_count, junction_count = _branch_counts(component_local)
        voxel_count = int(sizes[comp_id]) if comp_id < int(sizes.shape[0]) else int(np.count_nonzero(component_local))
        rows.append(
            MicrogliaCellReportRow(
                cell_index=int(cell_index),
                component_id=comp_id,
                segmentation_engine_used=engine,
                voxel_count=voxel_count,
                volume_um3=float(voxel_count * voxel_volume_um3),
                branch_endpoint_count=int(endpoint_count),
                branch_junction_count=int(junction_count),
                distance_to_vasculature_um=float(distance_um),
                microglia_closest_x_um=float(micro_xyz[0]),
                microglia_closest_y_um=float(micro_xyz[1]),
                microglia_closest_z_um=float(micro_xyz[2]),
                vessel_closest_x_um=float(vessel_xyz[0]),
                vessel_closest_y_um=float(vessel_xyz[1]),
                vessel_closest_z_um=float(vessel_xyz[2]),
                threshold_green_used=threshold_green,
                threshold_red_used=threshold_red,
            )
        )

    return MicrogliaCellReport(
        rows=rows,
        segmentation_engine_used=engine,
        threshold_green_used=threshold_green,
        threshold_red_used=threshold_red,
    )


def microglia_cell_report_to_csv_rows(
    report: MicrogliaCellReport,
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for item in report.rows:
        raw = asdict(item)
        rows.append({key: raw[key] for key in MICROGLIA_CELL_REPORT_COLUMNS})
    return rows
