from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
import json
import logging
import os
from pathlib import Path
import time
from typing import Callable, Literal, Mapping

import imageio.v3 as iio
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import skeletonize

try:
    from skimage.morphology import skeletonize_3d as _skeletonize_3d
except ImportError:  # scikit-image>=0.25 removed skeletonize_3d
    _skeletonize_3d = None

from nvap.analysis.microglia_components import (
    compute_component_labels,
    filter_components_by_preferred_voxel_floor,
)
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
    "soma_voxel_count",
    "soma_volume_um3",
    "soma_center_x_um",
    "soma_center_y_um",
    "soma_center_z_um",
    "soma_distance_to_vasculature_um",
    "soma_equivalent_diameter_um",
    "soma_bbox_x_um",
    "soma_bbox_y_um",
    "soma_bbox_z_um",
    "soma_roundness",
    "soma_rectangularity",
    "branch_count",
    "branch_endpoint_count",
    "branch_junction_count",
    "distance_to_vasculature_um",
    "nearest_vessel_id",
    "nearest_vessel_diameter_um",
    "tip_near_multiple_vessel_count",
    "microglia_closest_x_um",
    "microglia_closest_y_um",
    "microglia_closest_z_um",
    "vessel_closest_x_um",
    "vessel_closest_y_um",
    "vessel_closest_z_um",
    "threshold_green_used",
    "threshold_red_used",
]

MICROGLIA_BRANCH_REPORT_COLUMNS = [
    "cell_index",
    "component_id",
    "branch_id",
    "start_x_um",
    "start_y_um",
    "start_z_um",
    "end_x_um",
    "end_y_um",
    "end_z_um",
    "path_length_um",
    "chord_length_um",
    "tortuosity",
    "nearest_vessel_id",
    "nearest_vessel_distance_um",
    "nearest_vessel_diameter_um",
]

MICROGLIA_TIP_REPORT_COLUMNS = [
    "cell_index",
    "component_id",
    "branch_id",
    "tip_id",
    "tip_x_um",
    "tip_y_um",
    "tip_z_um",
    "nearest_vessel_id",
    "nearest_vessel_distance_um",
    "nearest_vessel_diameter_um",
    "nearby_vessel_count",
    "nearby_vessel_ids",
]

VESSEL_CROSSING_REPORT_COLUMNS = [
    "crossing_id",
    "x_um",
    "y_um",
    "z_um",
    "vessel_id_a",
    "vessel_id_b",
    "mean_z_a_um",
    "mean_z_b_um",
    "over_vessel_id",
    "confidence",
    "status",
]

_CC_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)
_DEGREE_KERNEL = np.ones((3, 3, 3), dtype=np.uint8)
_AUTO_FIJI_FALLBACK_MAX_VOXELS = 96 * 1024 * 1024
_TIP_NEAR_VESSEL_RADIUS_UM = 10.0
_BRANCH_OVERLAY_MAX_POINTS = 192


@dataclass
class _ComponentAnalysisResult:
    cell_index: int
    component_id: int
    component_slice: tuple[slice, slice, slice]
    row: MicrogliaCellReportRow
    branch_rows: list[MicrogliaBranchReportRow]
    tip_payloads: list[dict[str, object]]
    soma_local: np.ndarray
    branch_labels_local: np.ndarray
    overlay_points: list[dict[str, object]]
    overlay_lines: list[dict[str, object]]


@dataclass(frozen=True)
class MicrogliaCellReportRow:
    cell_index: int
    component_id: int
    segmentation_engine_used: SegmentationEngine
    voxel_count: int
    volume_um3: float
    soma_voxel_count: int = 0
    soma_volume_um3: float = float("nan")
    soma_center_x_um: float = float("nan")
    soma_center_y_um: float = float("nan")
    soma_center_z_um: float = float("nan")
    soma_distance_to_vasculature_um: float = float("nan")
    soma_equivalent_diameter_um: float = float("nan")
    soma_bbox_x_um: float = float("nan")
    soma_bbox_y_um: float = float("nan")
    soma_bbox_z_um: float = float("nan")
    soma_roundness: float = float("nan")
    soma_rectangularity: float = float("nan")
    branch_count: int = 0
    branch_endpoint_count: int = 0
    branch_junction_count: int = 0
    distance_to_vasculature_um: float = float("nan")
    nearest_vessel_id: int = 0
    nearest_vessel_diameter_um: float = float("nan")
    tip_near_multiple_vessel_count: int = 0
    microglia_closest_x_um: float = float("nan")
    microglia_closest_y_um: float = float("nan")
    microglia_closest_z_um: float = float("nan")
    vessel_closest_x_um: float = float("nan")
    vessel_closest_y_um: float = float("nan")
    vessel_closest_z_um: float = float("nan")
    threshold_green_used: float = 0.0
    threshold_red_used: float = 0.0


@dataclass(frozen=True)
class MicrogliaBranchReportRow:
    cell_index: int
    component_id: int
    branch_id: int
    start_x_um: float
    start_y_um: float
    start_z_um: float
    end_x_um: float
    end_y_um: float
    end_z_um: float
    path_length_um: float
    chord_length_um: float
    tortuosity: float
    nearest_vessel_id: int = 0
    nearest_vessel_distance_um: float = float("nan")
    nearest_vessel_diameter_um: float = float("nan")


@dataclass(frozen=True)
class MicrogliaTipReportRow:
    cell_index: int
    component_id: int
    branch_id: int
    tip_id: int
    tip_x_um: float
    tip_y_um: float
    tip_z_um: float
    nearest_vessel_id: int = 0
    nearest_vessel_distance_um: float = float("nan")
    nearest_vessel_diameter_um: float = float("nan")
    nearby_vessel_count: int = 0
    nearby_vessel_ids: str = ""


@dataclass(frozen=True)
class VesselCrossingReportRow:
    crossing_id: int
    x_um: float
    y_um: float
    z_um: float
    vessel_id_a: int
    vessel_id_b: int
    mean_z_a_um: float
    mean_z_b_um: float
    over_vessel_id: int
    confidence: float
    status: str


@dataclass(frozen=True)
class MicrogliaCellReport:
    rows: list[MicrogliaCellReportRow] = field(default_factory=list)
    branch_rows: list[MicrogliaBranchReportRow] = field(default_factory=list)
    tip_rows: list[MicrogliaTipReportRow] = field(default_factory=list)
    crossing_rows: list[VesselCrossingReportRow] = field(default_factory=list)
    segmentation_engine_used: SegmentationEngine = "internal"
    threshold_green_used: float = 0.0
    threshold_red_used: float = 0.0
    debug_volumes: dict[str, np.ndarray] = field(default_factory=dict)
    debug_measurements: dict[str, object] = field(default_factory=dict)

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
    labels, order, sizes = filter_components_by_preferred_voxel_floor(labels, order, sizes)
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
    labels, order, sizes = filter_components_by_preferred_voxel_floor(labels, order, sizes)
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


def _skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        return np.zeros_like(binary, dtype=bool)
    if _skeletonize_3d is not None:
        return np.asarray(_skeletonize_3d(binary), dtype=bool)
    return np.asarray(skeletonize(binary), dtype=bool)


def _skeleton_degree_map(skeleton: np.ndarray) -> np.ndarray:
    skel = np.asarray(skeleton, dtype=bool)
    if not np.any(skel):
        return np.zeros(skel.shape, dtype=np.int16)
    degree_map = ndi.convolve(
        skel.astype(np.uint8, copy=False),
        _DEGREE_KERNEL,
        mode="constant",
        cval=0,
    ).astype(np.int16, copy=False)
    return degree_map - skel.astype(np.int16, copy=False)


def _spacing_vector_um(spacing: VoxelSpacing) -> np.ndarray:
    return np.asarray(
        [float(spacing.z_um), float(spacing.y_um), float(spacing.x_um)],
        dtype=np.float64,
    )


def _path_length_um(path: list[tuple[int, int, int]], spacing: VoxelSpacing) -> float:
    if len(path) <= 1:
        return 0.0
    scale = _spacing_vector_um(spacing)
    total = 0.0
    for left, right in zip(path[:-1], path[1:]):
        delta = (np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)) * scale
        total += float(np.linalg.norm(delta))
    return total


def _chord_length_um(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    spacing: VoxelSpacing,
) -> float:
    delta = (np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)) * _spacing_vector_um(spacing)
    return float(np.linalg.norm(delta))


def _neighbor_offsets() -> list[tuple[int, int, int]]:
    return [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]


_NEIGHBOR_OFFSETS = _neighbor_offsets()


def _downsample_overlay_path(
    path: list[tuple[int, int, int]],
    max_points: int,
) -> list[tuple[int, int, int]]:
    if len(path) <= int(max_points):
        return path
    target = max(2, int(max_points))
    sample_idx = np.linspace(0, len(path) - 1, num=target, dtype=np.int64)
    sample_idx = np.unique(sample_idx)
    sampled = [path[int(i)] for i in sample_idx.tolist()]
    if sampled[0] != path[0]:
        sampled.insert(0, path[0])
    if sampled[-1] != path[-1]:
        sampled.append(path[-1])
    return sampled


def _skeleton_neighbors(
    point: tuple[int, int, int],
    skeleton: np.ndarray,
) -> list[tuple[int, int, int]]:
    z, y, x = point
    out: list[tuple[int, int, int]] = []
    max_z, max_y, max_x = skeleton.shape
    for dz, dy, dx in _NEIGHBOR_OFFSETS:
        nz = z + dz
        ny = y + dy
        nx = x + dx
        if 0 <= nz < max_z and 0 <= ny < max_y and 0 <= nx < max_x and bool(skeleton[nz, ny, nx]):
            out.append((int(nz), int(ny), int(nx)))
    return out


def _trace_skeleton_branches(
    skeleton: np.ndarray,
    spacing: VoxelSpacing,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    skel = np.asarray(skeleton, dtype=bool)
    branch_labels = np.zeros(skel.shape, dtype=np.int32)
    tip_labels = np.zeros(skel.shape, dtype=np.int32)
    degree_map = _skeleton_degree_map(skel)
    junction_mask = skel & (degree_map >= 3)
    endpoints = [tuple(int(v) for v in row) for row in np.argwhere(skel & (degree_map == 1))]
    nodes = set(endpoints)
    nodes.update(tuple(int(v) for v in row) for row in np.argwhere(junction_mask))

    branches: list[dict[str, object]] = []
    visited_edges: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()

    def edge_key(a: tuple[int, int, int], b: tuple[int, int, int]):
        return (a, b) if a <= b else (b, a)

    starts = list(nodes)
    if not starts and np.any(skel):
        # Closed loop fallback: pick one skeleton point as a pseudo-node so it
        # still receives one branch row instead of disappearing.
        starts = [tuple(int(v) for v in np.argwhere(skel)[0])]
        nodes.add(starts[0])

    branch_id = 0
    for start in starts:
        for next_point in _skeleton_neighbors(start, skel):
            first_edge = edge_key(start, next_point)
            if first_edge in visited_edges:
                continue
            path = [start, next_point]
            visited_edges.add(first_edge)
            prev = start
            cur = next_point
            while cur not in nodes:
                neighbors = [p for p in _skeleton_neighbors(cur, skel) if p != prev]
                if not neighbors:
                    break
                nxt = neighbors[0]
                key = edge_key(cur, nxt)
                if key in visited_edges:
                    break
                visited_edges.add(key)
                path.append(nxt)
                prev, cur = cur, nxt

            if len(path) < 2:
                continue
            branch_id += 1
            for z, y, x in path:
                branch_labels[z, y, x] = branch_id
            start_pt = path[0]
            end_pt = path[-1]
            chord = _chord_length_um(start_pt, end_pt, spacing)
            length = _path_length_um(path, spacing)
            branches.append(
                {
                    "branch_id": branch_id,
                    "path": path,
                    "start": start_pt,
                    "end": end_pt,
                    "path_length_um": length,
                    "chord_length_um": chord,
                    "tortuosity": float(length / chord) if chord > 1.0e-9 else float("nan"),
                }
            )

    for tip_id, tip in enumerate(endpoints, start=1):
        tip_labels[tip] = tip_id
    return branches, branch_labels, tip_labels, junction_mask.astype(np.uint8, copy=False)


def _soma_core_mask(
    component_local: np.ndarray,
    green_local: np.ndarray,
    skeleton_local: np.ndarray,
    threshold_green: float,
) -> np.ndarray:
    comp = np.asarray(component_local, dtype=bool)
    if not np.any(comp):
        return np.zeros_like(comp, dtype=bool)
    values = np.asarray(green_local, dtype=np.float32)
    comp_values = values[comp]
    if comp_values.size <= 0:
        return comp.copy()
    dist = ndi.distance_transform_edt(comp)
    core_floor = float(max(threshold_green, np.quantile(comp_values, 0.70)))
    thick_floor = float(np.quantile(dist[comp], 0.60)) if np.any(dist[comp] > 0) else 0.0
    soma = comp & (~np.asarray(skeleton_local, dtype=bool)) & (values >= core_floor) & (dist >= max(0.0, thick_floor))
    if not np.any(soma):
        soma = comp & (values >= core_floor)
    if not np.any(soma):
        max_pos = np.unravel_index(int(np.argmax(np.where(comp, values, -np.inf))), values.shape)
        soma = np.zeros_like(comp, dtype=bool)
        soma[max_pos] = True
    return np.asarray(soma, dtype=bool)


def _mask_centroid_voxel(mask: np.ndarray, comp_slice: tuple[slice, slice, slice]) -> tuple[int, int, int]:
    coords = _coords_from_local_mask(mask, comp_slice)
    if coords[0].size <= 0:
        return (0, 0, 0)
    return (
        int(round(float(np.mean(coords[0])))),
        int(round(float(np.mean(coords[1])))),
        int(round(float(np.mean(coords[2])))),
    )


def _mask_bbox_um(mask: np.ndarray, spacing: VoxelSpacing) -> tuple[float, float, float]:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size <= 0:
        return (float("nan"), float("nan"), float("nan"))
    extent = coords.max(axis=0) - coords.min(axis=0) + 1
    return (
        float(extent[2]) * float(spacing.x_um),
        float(extent[1]) * float(spacing.y_um),
        float(extent[0]) * float(spacing.z_um),
    )


def _soma_shape_metrics(
    soma_local: np.ndarray,
    spacing: VoxelSpacing,
) -> tuple[float, float, float, float, float]:
    voxel_count = int(np.count_nonzero(soma_local))
    if voxel_count <= 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    volume = float(voxel_count) * float(spacing.voxel_volume_um3)
    equiv_diameter = float((6.0 * volume / np.pi) ** (1.0 / 3.0)) if volume > 0.0 else float("nan")
    bbox_x, bbox_y, bbox_z = _mask_bbox_um(soma_local, spacing)
    dims = np.asarray([bbox_x, bbox_y, bbox_z], dtype=np.float64)
    finite_dims = dims[np.isfinite(dims) & (dims > 0.0)]
    if finite_dims.size <= 0:
        roundness = float("nan")
        rectangularity = float("nan")
    else:
        roundness = float(np.min(finite_dims) / np.max(finite_dims))
        rectangularity = float(volume / max(1.0e-9, float(np.prod(finite_dims))))
    return equiv_diameter, bbox_x, bbox_y, bbox_z, float(np.clip(roundness, 0.0, 1.0)), float(np.clip(rectangularity, 0.0, 1.0))


def _nearest_vessel_info(
    voxel_zyx: tuple[int, int, int],
    vessel_surface_dist_um: np.ndarray | None,
    vessel_surface_indices: np.ndarray | None,
    vessel_labels: np.ndarray,
    vessel_radius_um: np.ndarray,
) -> tuple[float, tuple[int, int, int] | None, int, float]:
    if vessel_surface_dist_um is None or vessel_surface_indices is None:
        return float("nan"), None, 0, float("nan")
    z, y, x = voxel_zyx
    if not (0 <= z < vessel_labels.shape[0] and 0 <= y < vessel_labels.shape[1] and 0 <= x < vessel_labels.shape[2]):
        return float("nan"), None, 0, float("nan")
    vessel_idx = (
        int(vessel_surface_indices[0][z, y, x]),
        int(vessel_surface_indices[1][z, y, x]),
        int(vessel_surface_indices[2][z, y, x]),
    )
    vessel_id = int(vessel_labels[vessel_idx])
    diameter = _local_vessel_diameter_um(vessel_idx, vessel_id, vessel_labels, vessel_radius_um)
    return float(vessel_surface_dist_um[z, y, x]), vessel_idx, vessel_id, diameter


def _local_vessel_diameter_um(
    vessel_idx: tuple[int, int, int],
    vessel_id: int,
    vessel_labels: np.ndarray,
    vessel_radius_um: np.ndarray,
) -> float:
    if int(vessel_id) <= 0:
        return float("nan")
    z, y, x = vessel_idx
    z0, z1 = max(0, z - 4), min(vessel_labels.shape[0], z + 5)
    y0, y1 = max(0, y - 4), min(vessel_labels.shape[1], y + 5)
    x0, x1 = max(0, x - 4), min(vessel_labels.shape[2], x + 5)
    local_labels = vessel_labels[z0:z1, y0:y1, x0:x1]
    local_radius = vessel_radius_um[z0:z1, y0:y1, x0:x1]
    owned = local_labels == int(vessel_id)
    if not np.any(owned):
        return float("nan")
    return float(np.max(local_radius[owned]) * 2.0)


def _nearby_vessel_ids(
    voxel_zyx: tuple[int, int, int],
    vessel_labels: np.ndarray,
    spacing: VoxelSpacing,
    radius_um: float = _TIP_NEAR_VESSEL_RADIUS_UM,
) -> list[int]:
    z, y, x = voxel_zyx
    rz = int(np.ceil(radius_um / max(1.0e-9, float(spacing.z_um))))
    ry = int(np.ceil(radius_um / max(1.0e-9, float(spacing.y_um))))
    rx = int(np.ceil(radius_um / max(1.0e-9, float(spacing.x_um))))
    z0, z1 = max(0, z - rz), min(vessel_labels.shape[0], z + rz + 1)
    y0, y1 = max(0, y - ry), min(vessel_labels.shape[1], y + ry + 1)
    x0, x1 = max(0, x - rx), min(vessel_labels.shape[2], x + rx + 1)
    local = vessel_labels[z0:z1, y0:y1, x0:x1]
    if local.size <= 0:
        return []
    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    phys = (
        ((zz - z) * float(spacing.z_um)) ** 2
        + ((yy - y) * float(spacing.y_um)) ** 2
        + ((xx - x) * float(spacing.x_um)) ** 2
    )
    ids = np.unique(local[phys <= float(radius_um) ** 2])
    return [int(v) for v in ids.tolist() if int(v) > 0]


def _detect_vessel_crossings(
    vessel_skeleton: np.ndarray,
    vessel_labels: np.ndarray,
    spacing: VoxelSpacing,
    z_values: list[int],
) -> tuple[list[VesselCrossingReportRow], np.ndarray]:
    crossing_markers = np.zeros(vessel_labels.shape, dtype=np.int32)
    skel_coords = np.argwhere(np.asarray(vessel_skeleton, dtype=bool))
    if skel_coords.size <= 0:
        return [], crossing_markers
    by_xy: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for z, y, x in skel_coords:
        vessel_id = int(vessel_labels[int(z), int(y), int(x)])
        if vessel_id <= 0:
            continue
        by_xy.setdefault((int(y), int(x)), []).append((int(z), vessel_id))

    rows: list[VesselCrossingReportRow] = []
    seen_xy: set[tuple[int, int]] = set()
    for (y, x), items in by_xy.items():
        z_by_id: dict[int, list[int]] = {}
        for z, vessel_id in items:
            z_by_id.setdefault(vessel_id, []).append(z)
        if len(z_by_id) < 2 and len({z for z, _ in items}) < 2:
            continue
        ids = sorted(z_by_id)
        if len(ids) >= 2:
            a, b = ids[:2]
            mean_a = float(np.mean(z_by_id[a]))
            mean_b = float(np.mean(z_by_id[b]))
        else:
            sorted_items = sorted(items)
            a = int(sorted_items[0][1])
            b = int(sorted_items[-1][1])
            mean_a = float(sorted_items[0][0])
            mean_b = float(sorted_items[-1][0])
        sep_um = abs(mean_a - mean_b) * float(spacing.z_um)
        mean_a_um = (float(z_values[int(round(mean_a))]) if z_values and 0 <= int(round(mean_a)) < len(z_values) else mean_a) * float(spacing.z_um)
        mean_b_um = (float(z_values[int(round(mean_b))]) if z_values and 0 <= int(round(mean_b)) < len(z_values) else mean_b) * float(spacing.z_um)
        over = 0
        status = "ambiguous"
        confidence = 0.0
        if sep_um >= float(spacing.z_um) and a != b:
            over = int(a if mean_a_um > mean_b_um else b)
            status = "over_under"
            confidence = float(np.clip(sep_um / max(float(spacing.z_um) * 3.0, 1.0e-9), 0.0, 1.0))
        if (y, x) in seen_xy:
            continue
        seen_xy.add((y, x))
        crossing_id = len(rows) + 1
        z_mid = int(round((mean_a + mean_b) * 0.5))
        z_mid = int(np.clip(z_mid, 0, vessel_labels.shape[0] - 1))
        crossing_markers[z_mid, y, x] = crossing_id
        xyz = _voxel_to_xyz_um((z_mid, y, x), spacing, z_values=z_values)
        rows.append(
            VesselCrossingReportRow(
                crossing_id=crossing_id,
                x_um=float(xyz[0]),
                y_um=float(xyz[1]),
                z_um=float(xyz[2]),
                vessel_id_a=int(a),
                vessel_id_b=int(b),
                mean_z_a_um=float(mean_a_um),
                mean_z_b_um=float(mean_b_um),
                over_vessel_id=int(over),
                confidence=float(confidence),
                status=status,
            )
        )
    return rows, crossing_markers


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


def _resolve_analysis_workers(
    preprocess_config: PreprocessConfig,
    *,
    total_jobs: int,
    max_workers: int = 8,
) -> int:
    if int(total_jobs) <= 1:
        return 1
    requested = int(preprocess_config.cpu_worker_threads)
    if requested > 0:
        return max(1, min(requested, int(total_jobs), int(max_workers)))
    cpus = os.cpu_count() or 1
    return max(1, min(int(cpus), int(total_jobs), int(max_workers)))


def _compute_vessel_labels_and_radius(
    red_mask: np.ndarray,
    spacing: VoxelSpacing,
) -> tuple[np.ndarray, np.ndarray]:
    labels, _ = ndi.label(red_mask, structure=_CC_STRUCTURE)
    radius = ndi.distance_transform_edt(
        red_mask,
        sampling=(float(spacing.z_um), float(spacing.y_um), float(spacing.x_um)),
    )
    return np.asarray(labels, dtype=np.int32), np.asarray(radius, dtype=np.float32)


def _analyze_single_microglia_component(
    *,
    cell_index: int,
    component_id: int,
    component_slice: tuple[slice, slice, slice],
    labels: np.ndarray,
    sizes: np.ndarray,
    green: np.ndarray,
    red_mask: np.ndarray,
    threshold_green: float,
    threshold_red: float,
    spacing: VoxelSpacing,
    shared_z_values: list[int],
    engine: SegmentationEngine,
    vessel_labels: np.ndarray,
    vessel_radius_um: np.ndarray,
    vessel_surface_dist_um: np.ndarray | None,
    vessel_surface_indices: np.ndarray | None,
    overlay_state: Mapping[str, bool],
    max_branch_overlay_points: int,
) -> _ComponentAnalysisResult | None:
    comp_id = int(component_id)
    comp_slice = component_slice
    component_local = np.asarray(labels[comp_slice] == comp_id, dtype=bool)
    if not np.any(component_local):
        return None

    green_local = np.asarray(green[comp_slice], dtype=np.float32)
    skeleton_local = _skeletonize_mask(component_local)
    degree_local = _skeleton_degree_map(skeleton_local)
    branch_defs, branch_labels_local, _tip_labels_local, _junction_mask = _trace_skeleton_branches(
        skeleton_local,
        spacing,
    )
    endpoint_count = int(np.count_nonzero(skeleton_local & (degree_local == 1)))
    junction_count = int(np.count_nonzero(skeleton_local & (degree_local >= 3)))

    soma_local = _soma_core_mask(
        component_local,
        green_local,
        skeleton_local,
        threshold_green,
    )
    soma_voxel_count = int(np.count_nonzero(soma_local))
    soma_center_idx = _mask_centroid_voxel(soma_local, comp_slice)
    soma_xyz = _voxel_to_xyz_um(soma_center_idx, spacing, z_values=shared_z_values)
    soma_dist_um, soma_vessel_idx, _soma_vessel_id, _soma_vessel_diam = _nearest_vessel_info(
        soma_center_idx,
        vessel_surface_dist_um,
        vessel_surface_indices,
        vessel_labels,
        vessel_radius_um,
    )
    soma_equiv_diameter, soma_bbox_x, soma_bbox_y, soma_bbox_z, soma_roundness, soma_rectangularity = _soma_shape_metrics(
        soma_local,
        spacing,
    )

    overlay_points: list[dict[str, object]] = []
    overlay_lines: list[dict[str, object]] = []
    if overlay_state["soma"]:
        overlay_points.append(
            {
                "kind": "soma_center",
                "cell_index": int(cell_index),
                "component_id": comp_id,
                "xyz_um": [float(v) for v in soma_xyz],
            }
        )

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
        nearest_vessel_id = 0
        nearest_vessel_diameter_um = float("nan")
    else:
        vessel_xyz = _voxel_to_xyz_um(vessel_idx, spacing, z_values=shared_z_values)
        nearest_vessel_id = int(vessel_labels[vessel_idx])
        nearest_vessel_diameter_um = _local_vessel_diameter_um(
            vessel_idx,
            nearest_vessel_id,
            vessel_labels,
            vessel_radius_um,
        )
        if overlay_state["vessels"]:
            overlay_points.append(
                {
                    "kind": "nearest_vessel_point",
                    "cell_index": int(cell_index),
                    "component_id": comp_id,
                    "vessel_id": nearest_vessel_id,
                    "xyz_um": [float(v) for v in vessel_xyz],
                }
            )
        if overlay_state["connectors"]:
            overlay_lines.append(
                {
                    "kind": "cell_to_vessel",
                    "cell_index": int(cell_index),
                    "component_id": comp_id,
                    "points_xyz_um": [[float(v) for v in micro_xyz], [float(v) for v in vessel_xyz]],
                }
            )

    tip_payloads: list[dict[str, object]] = []
    branch_rows: list[MicrogliaBranchReportRow] = []
    tip_near_multiple_vessel_count = 0
    for branch in branch_defs:
        branch_id = int(branch["branch_id"])
        path_local = [tuple(int(v) for v in p) for p in branch["path"]]  # type: ignore[index]
        path_global = [
            (
                int(p[0]) + int(comp_slice[0].start or 0),
                int(p[1]) + int(comp_slice[1].start or 0),
                int(p[2]) + int(comp_slice[2].start or 0),
            )
            for p in path_local
        ]
        if not path_global:
            continue

        if vessel_surface_dist_um is not None and vessel_surface_indices is not None:
            path_arr = np.asarray(path_global, dtype=np.int64)
            z_path = path_arr[:, 0]
            y_path = path_arr[:, 1]
            x_path = path_arr[:, 2]
            branch_dist = np.asarray(vessel_surface_dist_um[z_path, y_path, x_path], dtype=np.float32)
            finite_mask = np.isfinite(branch_dist)
            if np.any(finite_mask):
                finite_idx = np.flatnonzero(finite_mask)
                best_local = int(finite_idx[int(np.argmin(branch_dist[finite_mask]))])
                best_point = (
                    int(z_path[best_local]),
                    int(y_path[best_local]),
                    int(x_path[best_local]),
                )
                best_branch_distance = float(branch_dist[best_local])
                best_branch_vessel_idx = (
                    int(vessel_surface_indices[0][best_point]),
                    int(vessel_surface_indices[1][best_point]),
                    int(vessel_surface_indices[2][best_point]),
                )
                branch_vessel_id = int(vessel_labels[best_branch_vessel_idx])
                branch_vessel_diam = _local_vessel_diameter_um(
                    best_branch_vessel_idx,
                    branch_vessel_id,
                    vessel_labels,
                    vessel_radius_um,
                )
            else:
                best_branch_distance, branch_vessel_id, branch_vessel_diam = float("nan"), 0, float("nan")
        else:
            best_branch_distance, branch_vessel_id, branch_vessel_diam = float("nan"), 0, float("nan")

        start_global = path_global[0]
        end_global = path_global[-1]
        start_xyz = _voxel_to_xyz_um(start_global, spacing, z_values=shared_z_values)
        end_xyz = _voxel_to_xyz_um(end_global, spacing, z_values=shared_z_values)
        branch_rows.append(
            MicrogliaBranchReportRow(
                cell_index=int(cell_index),
                component_id=comp_id,
                branch_id=branch_id,
                start_x_um=float(start_xyz[0]),
                start_y_um=float(start_xyz[1]),
                start_z_um=float(start_xyz[2]),
                end_x_um=float(end_xyz[0]),
                end_y_um=float(end_xyz[1]),
                end_z_um=float(end_xyz[2]),
                path_length_um=float(branch["path_length_um"]),
                chord_length_um=float(branch["chord_length_um"]),
                tortuosity=float(branch["tortuosity"]),
                nearest_vessel_id=int(branch_vessel_id),
                nearest_vessel_distance_um=float(best_branch_distance),
                nearest_vessel_diameter_um=float(branch_vessel_diam),
            )
        )

        if overlay_state["branches"]:
            overlay_path = _downsample_overlay_path(path_global, max_branch_overlay_points)
            overlay_lines.append(
                {
                    "kind": "branch_path",
                    "cell_index": int(cell_index),
                    "component_id": comp_id,
                    "branch_id": branch_id,
                    "points_xyz_um": [
                        [float(v) for v in _voxel_to_xyz_um(point, spacing, z_values=shared_z_values)]
                        for point in overlay_path
                    ],
                }
            )

        for tip_global in (start_global, end_global):
            local_tip = (
                int(tip_global[0]) - int(comp_slice[0].start or 0),
                int(tip_global[1]) - int(comp_slice[1].start or 0),
                int(tip_global[2]) - int(comp_slice[2].start or 0),
            )
            if not bool(skeleton_local[local_tip]) or int(degree_local[local_tip]) != 1:
                continue
            near_ids = _nearby_vessel_ids(tip_global, vessel_labels, spacing)
            tip_near_multiple_vessel_count = max(tip_near_multiple_vessel_count, int(len(near_ids)))
            tip_dist, tip_vessel_idx, tip_vessel_id, tip_vessel_diameter = _nearest_vessel_info(
                tip_global,
                vessel_surface_dist_um,
                vessel_surface_indices,
                vessel_labels,
                vessel_radius_um,
            )
            tip_xyz = _voxel_to_xyz_um(tip_global, spacing, z_values=shared_z_values)
            tip_payloads.append(
                {
                    "cell_index": int(cell_index),
                    "component_id": comp_id,
                    "branch_id": branch_id,
                    "tip_global": tuple(int(v) for v in tip_global),
                    "tip_xyz": tuple(float(v) for v in tip_xyz),
                    "tip_dist": float(tip_dist),
                    "tip_vessel_idx": (
                        tuple(int(v) for v in tip_vessel_idx)
                        if tip_vessel_idx is not None
                        else None
                    ),
                    "tip_vessel_id": int(tip_vessel_id),
                    "tip_vessel_diameter": float(tip_vessel_diameter),
                    "near_ids": [int(v) for v in near_ids],
                }
            )

    voxel_count = int(sizes[comp_id]) if comp_id < int(sizes.shape[0]) else int(np.count_nonzero(component_local))
    voxel_volume_um3 = float(spacing.voxel_volume_um3)
    row = MicrogliaCellReportRow(
        cell_index=int(cell_index),
        component_id=comp_id,
        segmentation_engine_used=engine,
        voxel_count=voxel_count,
        volume_um3=float(voxel_count * voxel_volume_um3),
        soma_voxel_count=soma_voxel_count,
        soma_volume_um3=float(soma_voxel_count * voxel_volume_um3),
        soma_center_x_um=float(soma_xyz[0]),
        soma_center_y_um=float(soma_xyz[1]),
        soma_center_z_um=float(soma_xyz[2]),
        soma_distance_to_vasculature_um=float(soma_dist_um),
        soma_equivalent_diameter_um=float(soma_equiv_diameter),
        soma_bbox_x_um=float(soma_bbox_x),
        soma_bbox_y_um=float(soma_bbox_y),
        soma_bbox_z_um=float(soma_bbox_z),
        soma_roundness=float(soma_roundness),
        soma_rectangularity=float(soma_rectangularity),
        branch_count=int(len(branch_defs)),
        branch_endpoint_count=int(endpoint_count),
        branch_junction_count=int(junction_count),
        distance_to_vasculature_um=float(distance_um),
        nearest_vessel_id=int(nearest_vessel_id),
        nearest_vessel_diameter_um=float(nearest_vessel_diameter_um),
        tip_near_multiple_vessel_count=int(tip_near_multiple_vessel_count),
        microglia_closest_x_um=float(micro_xyz[0]),
        microglia_closest_y_um=float(micro_xyz[1]),
        microglia_closest_z_um=float(micro_xyz[2]),
        vessel_closest_x_um=float(vessel_xyz[0]),
        vessel_closest_y_um=float(vessel_xyz[1]),
        vessel_closest_z_um=float(vessel_xyz[2]),
        threshold_green_used=float(threshold_green),
        threshold_red_used=float(threshold_red),
    )

    return _ComponentAnalysisResult(
        cell_index=int(cell_index),
        component_id=comp_id,
        component_slice=comp_slice,
        row=row,
        branch_rows=branch_rows,
        tip_payloads=tip_payloads,
        soma_local=np.asarray(soma_local, dtype=bool),
        branch_labels_local=np.asarray(branch_labels_local, dtype=np.int32),
        overlay_points=overlay_points,
        overlay_lines=overlay_lines,
    )


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
    progress_callback: Callable[[float, str], None] | None = None,
    overlay_options: Mapping[str, bool] | None = None,
    max_branch_overlay_points: int = _BRANCH_OVERLAY_MAX_POINTS,
) -> MicrogliaCellReport:
    total_start = time.perf_counter()

    def _publish_progress(progress: float, message: str, *, log: bool = False) -> None:
        frac = float(np.clip(progress, 0.0, 1.0))
        if progress_callback is not None:
            try:
                progress_callback(frac, message)
            except Exception:
                # Progress updates are best-effort and must never fail analysis.
                pass
        if log:
            logger.info("Microglia analysis: %s", message)

    mode = str(segmentation_mode).strip().lower()
    if mode not in {"auto", "internal", "fiji"}:
        raise ValueError("segmentation_mode must be one of: auto, internal, fiji.")

    overlay_state = {
        "soma": True,
        "branches": True,
        "tips": True,
        "connectors": True,
        "vessels": True,
        "diameter": True,
        "crossings": True,
    }
    if overlay_options is not None:
        for key in overlay_state:
            if key in overlay_options:
                overlay_state[key] = bool(overlay_options[key])

    max_branch_overlay_points = max(8, int(max_branch_overlay_points))

    _publish_progress(0.02, "Resolving thresholds and shared volume...", log=True)

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

    _publish_progress(0.12, f"Segmenting microglia (mode={mode})...", log=True)

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
                    _publish_progress(0.20, "Retrying segmentation with Fiji fallback...", log=True)
                    labels, order, sizes = _run_fiji_segmentation(
                        green, threshold_green, preprocess_config,
                        spacing=spacing, branch_sensitivity=branch_sensitivity,
                    )
                    engine = "fiji"
                    logger.info("Microglia report segmentation fallback: internal->fiji.")
                except Exception as exc:
                    logger.warning("Microglia report fallback to fiji failed: %s", exc)

    _publish_progress(
        0.28,
        f"Segmentation complete: engine={engine}, components={int(len(order))}.",
        log=True,
    )

    vessel_start = time.perf_counter()
    _publish_progress(0.34, "Computing vessel distance maps and skeletons...", log=True)
    red_mask = red >= threshold_red
    vessel_workers = _resolve_analysis_workers(preprocess_config, total_jobs=3, max_workers=3)
    if vessel_workers > 1:
        logger.info("Microglia analysis vessel preprocessing: parallel workers=%d", vessel_workers)
        with ThreadPoolExecutor(max_workers=vessel_workers, thread_name_prefix="nvap-vessel") as pool:
            labels_future = pool.submit(_compute_vessel_labels_and_radius, red_mask, spacing)
            skeleton_future = pool.submit(_skeletonize_mask, red_mask)
            surface_future = pool.submit(_compute_vessel_surface_distance_maps, red_mask, spacing)
            vessel_labels, vessel_radius_um = labels_future.result()
            vessel_skeleton = np.asarray(skeleton_future.result(), dtype=bool)
            vessel_surface_dist_um, vessel_surface_indices = surface_future.result()
    else:
        vessel_labels, vessel_radius_um = _compute_vessel_labels_and_radius(red_mask, spacing)
        vessel_skeleton = _skeletonize_mask(red_mask)
        vessel_surface_dist_um, vessel_surface_indices = _compute_vessel_surface_distance_maps(red_mask, spacing)
    crossing_rows, vessel_crossing_markers = _detect_vessel_crossings(
        vessel_skeleton,
        np.asarray(vessel_labels, dtype=np.int32),
        spacing,
        shared_z_values,
    )
    _publish_progress(
        0.44,
        f"Vessel preprocessing complete in {time.perf_counter() - vessel_start:.2f}s.",
        log=True,
    )

    rows: list[MicrogliaCellReportRow] = []
    branch_rows: list[MicrogliaBranchReportRow] = []
    tip_rows: list[MicrogliaTipReportRow] = []
    soma_core_labels = np.zeros(labels.shape, dtype=np.int32)
    branch_skeleton_labels = np.zeros(labels.shape, dtype=np.int32)
    tip_marker_labels = np.zeros(labels.shape, dtype=np.int32)
    overlay_points: list[dict[str, object]] = []
    overlay_lines: list[dict[str, object]] = []
    component_objects = ndi.find_objects(np.asarray(labels))
    component_jobs: list[tuple[int, int, tuple[slice, slice, slice]]] = []
    for cell_index, component_id in enumerate(order.tolist(), start=1):
        comp_id = int(component_id)
        obj_idx = comp_id - 1
        if obj_idx < 0 or obj_idx >= len(component_objects):
            continue
        comp_slice = component_objects[obj_idx]
        if comp_slice is None:
            continue
        component_jobs.append((int(cell_index), comp_id, comp_slice))

    total_cells = int(len(component_jobs))
    if total_cells <= 0:
        _publish_progress(0.90, "No microglia components found above threshold.", log=True)
    progress_stride = max(1, total_cells // 10) if total_cells > 0 else 1
    component_results: list[_ComponentAnalysisResult] = []
    cell_workers = _resolve_analysis_workers(preprocess_config, total_jobs=total_cells, max_workers=8)

    if total_cells > 0 and cell_workers > 1:
        logger.info("Microglia analysis cell processing: parallel workers=%d", cell_workers)
        with ThreadPoolExecutor(max_workers=cell_workers, thread_name_prefix="nvap-cell") as pool:
            futures = {
                pool.submit(
                    _analyze_single_microglia_component,
                    cell_index=cell_index,
                    component_id=component_id,
                    component_slice=component_slice,
                    labels=labels,
                    sizes=sizes,
                    green=green,
                    red_mask=red_mask,
                    threshold_green=threshold_green,
                    threshold_red=threshold_red,
                    spacing=spacing,
                    shared_z_values=shared_z_values,
                    engine=engine,
                    vessel_labels=vessel_labels,
                    vessel_radius_um=vessel_radius_um,
                    vessel_surface_dist_um=vessel_surface_dist_um,
                    vessel_surface_indices=vessel_surface_indices,
                    overlay_state=overlay_state,
                    max_branch_overlay_points=max_branch_overlay_points,
                ): int(cell_index)
                for cell_index, component_id, component_slice in component_jobs
            }
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                if result is not None:
                    component_results.append(result)
                if completed == 1 or completed == total_cells or (completed % progress_stride == 0):
                    frac = 0.46 + (0.44 * (float(completed - 1) / max(1.0, float(total_cells))))
                    _publish_progress(frac, f"Analyzing microglia cell {completed}/{total_cells}...", log=True)
    else:
        for completed, (cell_index, component_id, component_slice) in enumerate(component_jobs, start=1):
            if completed == 1 or completed == total_cells or (completed % progress_stride == 0):
                frac = 0.46 + (0.44 * (float(completed - 1) / max(1.0, float(total_cells))))
                _publish_progress(frac, f"Analyzing microglia cell {completed}/{total_cells}...", log=True)
            result = _analyze_single_microglia_component(
                cell_index=cell_index,
                component_id=component_id,
                component_slice=component_slice,
                labels=labels,
                sizes=sizes,
                green=green,
                red_mask=red_mask,
                threshold_green=threshold_green,
                threshold_red=threshold_red,
                spacing=spacing,
                shared_z_values=shared_z_values,
                engine=engine,
                vessel_labels=vessel_labels,
                vessel_radius_um=vessel_radius_um,
                vessel_surface_dist_um=vessel_surface_dist_um,
                vessel_surface_indices=vessel_surface_indices,
                overlay_state=overlay_state,
                max_branch_overlay_points=max_branch_overlay_points,
            )
            if result is not None:
                component_results.append(result)

    component_results.sort(key=lambda item: int(item.cell_index))
    for result in component_results:
        rows.append(result.row)
        soma_core_labels[result.component_slice][result.soma_local] = int(result.cell_index)
        overlay_points.extend(result.overlay_points)
        overlay_lines.extend(result.overlay_lines)

        local_branch_labels = np.asarray(result.branch_labels_local, dtype=np.int32)
        for branch_row in result.branch_rows:
            branch_rows.append(branch_row)
            branch_global_id = int(len(branch_rows))
            branch_skeleton_labels[result.component_slice][local_branch_labels == int(branch_row.branch_id)] = branch_global_id

        for payload in result.tip_payloads:
            tip_id = int(len(tip_rows) + 1)
            tip_global = tuple(int(v) for v in payload["tip_global"])
            tip_xyz = tuple(float(v) for v in payload["tip_xyz"])
            near_ids = [int(v) for v in payload["near_ids"]]
            tip_marker_labels[tip_global] = tip_id
            tip_rows.append(
                MicrogliaTipReportRow(
                    cell_index=int(payload["cell_index"]),
                    component_id=int(payload["component_id"]),
                    branch_id=int(payload["branch_id"]),
                    tip_id=tip_id,
                    tip_x_um=float(tip_xyz[0]),
                    tip_y_um=float(tip_xyz[1]),
                    tip_z_um=float(tip_xyz[2]),
                    nearest_vessel_id=int(payload["tip_vessel_id"]),
                    nearest_vessel_distance_um=float(payload["tip_dist"]),
                    nearest_vessel_diameter_um=float(payload["tip_vessel_diameter"]),
                    nearby_vessel_count=int(len(near_ids)),
                    nearby_vessel_ids=";".join(str(v) for v in near_ids),
                )
            )
            if overlay_state["tips"]:
                overlay_points.append(
                    {
                        "kind": "tip",
                        "cell_index": int(payload["cell_index"]),
                        "component_id": int(payload["component_id"]),
                        "branch_id": int(payload["branch_id"]),
                        "tip_id": tip_id,
                        "nearby_vessel_count": int(len(near_ids)),
                        "xyz_um": [float(v) for v in tip_xyz],
                    }
                )
            tip_vessel_idx = payload["tip_vessel_idx"]
            if overlay_state["connectors"] and tip_vessel_idx is not None:
                tip_vessel_xyz = _voxel_to_xyz_um(
                    tuple(int(v) for v in tip_vessel_idx),
                    spacing,
                    z_values=shared_z_values,
                )
                overlay_lines.append(
                    {
                        "kind": "tip_to_vessel",
                        "cell_index": int(payload["cell_index"]),
                        "component_id": int(payload["component_id"]),
                        "branch_id": int(payload["branch_id"]),
                        "tip_id": tip_id,
                        "points_xyz_um": [[float(v) for v in tip_xyz], [float(v) for v in tip_vessel_xyz]],
                    }
                )

    if overlay_state["crossings"]:
        for crossing in crossing_rows:
            overlay_points.append(
                {
                    "kind": "vessel_crossing",
                    "crossing_id": int(crossing.crossing_id),
                    "vessel_id_a": int(crossing.vessel_id_a),
                    "vessel_id_b": int(crossing.vessel_id_b),
                    "over_vessel_id": int(crossing.over_vessel_id),
                    "status": crossing.status,
                    "confidence": float(crossing.confidence),
                    "xyz_um": [float(crossing.x_um), float(crossing.y_um), float(crossing.z_um)],
                }
            )

    _publish_progress(0.96, "Packaging analysis outputs...", log=True)
    debug_volumes = {
        "microglia_component_labels": np.asarray(labels, dtype=np.int32),
        "soma_core_labels": np.asarray(soma_core_labels, dtype=np.int32),
        "branch_skeleton_labels": np.asarray(branch_skeleton_labels, dtype=np.int32),
        "tip_marker_labels": np.asarray(tip_marker_labels, dtype=np.int32),
        "vessel_component_labels": np.asarray(vessel_labels, dtype=np.int32),
        "vessel_crossing_markers": np.asarray(vessel_crossing_markers, dtype=np.int32),
    }
    debug_measurements: dict[str, object] = {
        "threshold_green_used": float(threshold_green),
        "threshold_red_used": float(threshold_red),
        "parallel_workers": {
            "vessel_preprocessing": int(vessel_workers),
            "cell_analysis": int(cell_workers),
        },
        "spacing_um": {
            "x": float(spacing.x_um),
            "y": float(spacing.y_um),
            "z": float(spacing.z_um),
        },
        "tip_near_vessel_radius_um": float(_TIP_NEAR_VESSEL_RADIUS_UM),
        "cell_rows": [asdict(row) for row in rows],
        "branch_rows": [asdict(row) for row in branch_rows],
        "tip_rows": [asdict(row) for row in tip_rows],
        "vessel_crossing_rows": [asdict(row) for row in crossing_rows],
        "overlay_points": overlay_points,
        "overlay_lines": overlay_lines,
    }

    elapsed = time.perf_counter() - total_start
    _publish_progress(
        1.0,
        (
            "Analysis complete "
            f"dt={elapsed:.2f}s cells={len(rows)} branches={len(branch_rows)} "
            f"tips={len(tip_rows)} crossings={len(crossing_rows)}"
        ),
        log=True,
    )

    return MicrogliaCellReport(
        rows=rows,
        branch_rows=branch_rows,
        tip_rows=tip_rows,
        crossing_rows=crossing_rows,
        segmentation_engine_used=engine,
        threshold_green_used=threshold_green,
        threshold_red_used=threshold_red,
        debug_volumes=debug_volumes,
        debug_measurements=debug_measurements,
    )


def microglia_cell_report_to_csv_rows(
    report: MicrogliaCellReport,
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for item in report.rows:
        raw = asdict(item)
        rows.append({key: raw[key] for key in MICROGLIA_CELL_REPORT_COLUMNS})
    return rows


def microglia_branch_report_to_csv_rows(
    report: MicrogliaCellReport,
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for item in report.branch_rows:
        raw = asdict(item)
        rows.append({key: raw[key] for key in MICROGLIA_BRANCH_REPORT_COLUMNS})
    return rows


def microglia_tip_report_to_csv_rows(
    report: MicrogliaCellReport,
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for item in report.tip_rows:
        raw = asdict(item)
        rows.append({key: raw[key] for key in MICROGLIA_TIP_REPORT_COLUMNS})
    return rows


def vessel_crossing_report_to_csv_rows(
    report: MicrogliaCellReport,
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for item in report.crossing_rows:
        raw = asdict(item)
        rows.append({key: raw[key] for key in VESSEL_CROSSING_REPORT_COLUMNS})
    return rows


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _normalize_projection(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    if data.size <= 0:
        return np.zeros(data.shape, dtype=np.uint8)
    finite = data[np.isfinite(data)]
    if finite.size <= 0:
        return np.zeros(data.shape, dtype=np.uint8)
    vmax = float(np.max(finite))
    if vmax <= 0.0:
        return np.zeros(data.shape, dtype=np.uint8)
    return np.asarray(np.clip(data / vmax, 0.0, 1.0) * 255.0, dtype=np.uint8)


def _label_projection(labels: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.size <= 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.asarray(np.max(arr, axis=axis) > 0, dtype=np.uint8)


def _overlay_projection_png(
    green: np.ndarray,
    red: np.ndarray,
    debug_volumes: dict[str, np.ndarray],
    axis: int,
) -> np.ndarray:
    green_proj = _normalize_projection(np.max(np.asarray(green, dtype=np.float32), axis=axis))
    red_proj = _normalize_projection(np.max(np.asarray(red, dtype=np.float32), axis=axis))
    rgb = np.zeros((*green_proj.shape, 3), dtype=np.uint8)
    rgb[..., 1] = green_proj
    rgb[..., 0] = red_proj

    soma = _label_projection(debug_volumes.get("soma_core_labels", np.zeros_like(green)), axis=axis)
    branches = _label_projection(debug_volumes.get("branch_skeleton_labels", np.zeros_like(green)), axis=axis)
    tips = _label_projection(debug_volumes.get("tip_marker_labels", np.zeros_like(green)), axis=axis)
    crossings = _label_projection(debug_volumes.get("vessel_crossing_markers", np.zeros_like(green)), axis=axis)

    rgb[soma > 0] = np.asarray([255, 255, 0], dtype=np.uint8)
    rgb[branches > 0] = np.asarray([0, 180, 255], dtype=np.uint8)
    rgb[tips > 0] = np.asarray([255, 0, 255], dtype=np.uint8)
    rgb[crossings > 0] = np.asarray([255, 255, 255], dtype=np.uint8)
    return rgb


def export_microglia_analysis_bundle(
    report: MicrogliaCellReport,
    output_path: str | Path,
    *,
    green_volume: np.ndarray | None = None,
    red_volume: np.ndarray | None = None,
) -> dict[str, Path]:
    """Export the multi-CSV analysis report and visual debug bundle."""
    import pandas as pd

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    suffix = path.suffix or ".csv"

    outputs: dict[str, Path] = {}
    tables = {
        "cells": (
            path if suffix.lower() == ".csv" else path.with_suffix(".csv"),
            microglia_cell_report_to_csv_rows(report),
            MICROGLIA_CELL_REPORT_COLUMNS,
        ),
        "branches": (
            path.with_name(f"{stem}_branches.csv"),
            microglia_branch_report_to_csv_rows(report),
            MICROGLIA_BRANCH_REPORT_COLUMNS,
        ),
        "tips": (
            path.with_name(f"{stem}_tips.csv"),
            microglia_tip_report_to_csv_rows(report),
            MICROGLIA_TIP_REPORT_COLUMNS,
        ),
        "vessel_crossings": (
            path.with_name(f"{stem}_vessel_crossings.csv"),
            vessel_crossing_report_to_csv_rows(report),
            VESSEL_CROSSING_REPORT_COLUMNS,
        ),
    }
    for name, (csv_path, rows, columns) in tables.items():
        pd.DataFrame(rows, columns=list(columns)).to_csv(csv_path, index=False)
        outputs[name] = csv_path

    debug_dir = path.with_name(f"{stem}_debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    if report.debug_volumes:
        volume_path = debug_dir / "debug_label_volumes.npz"
        np.savez_compressed(volume_path, **report.debug_volumes)
        outputs["debug_label_volumes"] = volume_path

    measurements_path = debug_dir / "debug_measurements.json"
    measurements_path.write_text(
        json.dumps(_json_safe(report.debug_measurements), indent=2),
        encoding="utf-8",
    )
    outputs["debug_measurements"] = measurements_path

    if green_volume is not None and red_volume is not None and report.debug_volumes:
        projections = {
            "xy": 0,
            "xz": 1,
            "yz": 2,
        }
        for name, axis in projections.items():
            png_path = debug_dir / f"overlay_{name}.png"
            iio.imwrite(
                png_path,
                _overlay_projection_png(green_volume, red_volume, report.debug_volumes, axis=axis),
            )
            outputs[f"overlay_{name}"] = png_path

    return outputs
