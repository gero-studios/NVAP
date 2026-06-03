from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import logging

import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import remove_small_holes, skeletonize

from nvap.config.types import RenderConfig, VoxelSpacing

logger = logging.getLogger(__name__)

# skimage renamed remove_small_holes' size kwarg (area_threshold -> max_size) in
# 0.26; pick whichever the installed version exposes so we stay warning-free and
# compatible across the supported range.
_REMOVE_SMALL_HOLES_KW = (
    "max_size"
    if "max_size" in inspect.signature(remove_small_holes).parameters
    else "area_threshold"
)

# Default terminal-branch length below which a skeleton branch is treated as a
# spurious spur (microns). Microglia primary/secondary processes comfortably
# exceed this; sub-resolution skeletonisation bumps do not.
_DEFAULT_MIN_BRANCH_LENGTH_UM = 3.0
# Thickness-aware spur removal: a terminal branch sprouting from a region of
# radius r is only kept when it extends well beyond that surface, i.e. its
# length must exceed ``spur_factor * r`` (capped so genuine long processes off
# thick trunks always survive).
_SPUR_RADIUS_FACTOR = 1.25
_SPUR_LENGTH_CAP_UM = 6.0
# Interior holes smaller than this (in voxels) are filled before skeletonising
# so they do not create spurious skeleton loops / branches.
_FILL_HOLE_VOXELS = 64
# Endpoints within this physical radius collapse to a single process terminal.
# Microglia terminals are flat lamellar "fans" whose skeletons fragment into
# many endpoints; clustering converts those clusters back into one tip each.
_DEFAULT_TIP_MERGE_RADIUS_UM = 5.0
# A real process tip is thin: drop endpoints sitting in voxels thicker than this
# (they lie inside the soma / cell body, not at a process end). Expressed as a
# fraction of the soma radius with an absolute floor.
_TIP_THICKNESS_SOMA_FRACTION = 0.45
_TIP_THICKNESS_FLOOR_UM = 1.5
# Intensity margin above the display threshold at which the volume render
# becomes visible (matches vtk_scene's green opacity ramp ~ threshold + knee*2.8).
# Tips are only kept on signal at least this far above threshold, so they land on
# structure the user can actually see rather than on the near-invisible halo.
_TIP_VISIBILITY_MARGIN = 0.05

try:  # skan provides robust, junction-aware skeleton topology (3D, anisotropy-aware).
    from skan import Skeleton as _SkanSkeleton, summarize as _skan_summarize

    _HAS_SKAN = True
except Exception:  # pragma: no cover - exercised only when optional dep missing
    _SkanSkeleton = None  # type: ignore[assignment]
    _skan_summarize = None  # type: ignore[assignment]
    _HAS_SKAN = False

_FULL_STRUCTURE = ndi.generate_binary_structure(3, 3).astype(np.uint8, copy=False)
# 3x3x3 neighbour-count kernel (centre excluded) for the dependency-free fallback.
_DEGREE_KERNEL = np.ones((3, 3, 3), dtype=np.uint8)
_DEGREE_KERNEL[1, 1, 1] = 0

# skan branch-type codes: 0 endpoint-to-endpoint, 1 junction-to-endpoint,
# 2 junction-to-junction, 3 closed loop. Terminal branches (those owning a free
# endpoint) are the ones eligible for spur pruning.
_TERMINAL_BRANCH_TYPES = (0, 1)


@dataclass(frozen=True)
class MicrogliaCellAnalysis:
    component_id: int
    voxel_count: int
    volume_um3: float
    soma_voxel_count: int
    soma_volume_um3: float
    branch_count: int
    tip_count: int
    branch_point_count: int
    total_process_length_um: float
    mean_branch_length_um: float
    sholl_max_intersections: int
    sholl_critical_radius_um: float
    sholl_enclosing_radius_um: float
    nearest_tip_to_vessel_um: float | None
    nearest_cell_to_vessel_um: float | None
    soma_to_vessel_um: float | None


@dataclass(frozen=True)
class MicrogliaAnalysisResult:
    cells: list[MicrogliaCellAnalysis] = field(default_factory=list)
    analyzed_cell_count: int = 0
    mean_branch_count: float = 0.0
    mean_tip_count: float = 0.0
    mean_branch_point_count: float = 0.0
    mean_process_length_um: float = 0.0
    mean_soma_volume_um3: float = 0.0
    mean_sholl_max_intersections: float = 0.0
    min_cell_to_vessel_um: float | None = None
    min_soma_to_vessel_um: float | None = None


@dataclass(frozen=True)
class MicrogliaCellDebug:
    component_id: int
    voxel_sample_coords_zyx: np.ndarray
    branch_sample_coords_zyx: np.ndarray
    soma_sample_coords_zyx: np.ndarray
    tip_coords_zyx: np.ndarray
    nearest_tip_segment_zyx: np.ndarray | None
    nearest_cell_segment_zyx: np.ndarray | None


@dataclass(frozen=True)
class _BranchTopology:
    tip_coords: np.ndarray
    branch_mask: np.ndarray
    branch_count: int
    branch_point_count: int
    total_length_um: float
    mean_length_um: float


@dataclass(frozen=True)
class _ShollMetrics:
    max_intersections: int
    critical_radius_um: float
    enclosing_radius_um: float


@dataclass(frozen=True)
class _ComponentShapeDebug:
    soma_mask: np.ndarray
    branch_mask: np.ndarray
    tip_coords: np.ndarray
    branch_count: int
    branch_point_count: int
    total_process_length_um: float
    mean_branch_length_um: float
    sholl_max_intersections: int
    sholl_critical_radius_um: float
    sholl_enclosing_radius_um: float


def analyze_microglia_cells(
    green_volume: np.ndarray,
    red_volume: np.ndarray,
    labels: np.ndarray,
    order: np.ndarray,
    *,
    spacing: VoxelSpacing | tuple[float, float, float],
    render: RenderConfig,
    min_branch_length_um: float = _DEFAULT_MIN_BRANCH_LENGTH_UM,
    branch_sensitivity: float = 1.0,
) -> MicrogliaAnalysisResult:
    green = np.asarray(green_volume, dtype=np.float32)
    red = np.asarray(red_volume, dtype=np.float32)
    lbl = np.asarray(labels, dtype=np.int32)
    ordered = np.asarray(order, dtype=np.int32)

    if green.shape != red.shape or green.shape != lbl.shape:
        raise ValueError("green_volume, red_volume, and labels must share the same shape.")
    if green.ndim != 3:
        raise ValueError("green_volume must be 3D in (z, y, x) order.")

    spacing_zyx = _spacing_zyx(spacing)
    voxel_volume_um3 = float(np.prod(spacing_zyx))

    trimmed_labels = _apply_render_trim(
        lbl,
        int(render.trim_first_slices),
        int(render.trim_last_slices),
    )
    vessel_mask = _apply_render_trim(
        red >= float(render.threshold_red),
        int(render.trim_first_slices),
        int(render.trim_last_slices),
    )

    vessel_dist = _distance_to_vessel(vessel_mask, spacing_zyx)
    dz, dy, dx = _offset_shift_zyx(render, spacing_zyx)

    vessel_voxels = int(np.count_nonzero(vessel_mask))
    vessel_occupancy = float(vessel_voxels) / float(max(1, vessel_mask.size))
    if vessel_dist is None:
        logger.info(
            "Microglia vessel distance: no vasculature above threshold_red=%.4f "
            "(red max=%.4f); all cell/tip distances will be undefined.",
            float(render.threshold_red),
            float(red.max()) if red.size else 0.0,
        )
    elif vessel_occupancy >= 0.5:
        logger.warning(
            "Microglia vessel distance: vasculature mask fills %.1f%% of the volume "
            "(threshold_red=%.4f). Cell-to-vessel distances will collapse toward 0; "
            "raise the red threshold for a meaningful measurement.",
            100.0 * vessel_occupancy,
            float(render.threshold_red),
        )

    full_shape = green.shape
    shift = np.asarray([dz, dy, dx], dtype=np.int64).reshape(1, 3)
    tip_intensity_floor = float(render.threshold_green) + _TIP_VISIBILITY_MARGIN
    # Per-component bounding boxes in a single pass, so each cell is processed
    # only within its own box instead of rescanning the whole volume per label.
    objects = ndi.find_objects(trimmed_labels)
    n_objects = len(objects)

    cells: list[MicrogliaCellAnalysis] = []
    for component_id in ordered.tolist():
        cid = int(component_id)
        if cid <= 0 or cid > n_objects:
            continue
        bbox = objects[cid - 1]
        if bbox is None:
            continue
        local_mask = np.asarray(
            trimmed_labels[bbox] == cid,
            dtype=bool,
        )
        if not np.any(local_mask):
            continue
        bbox_start = np.asarray(
            [bbox[0].start or 0, bbox[1].start or 0, bbox[2].start or 0],
            dtype=np.int64,
        )

        shape_debug = _describe_component_shape(
            local_mask,
            spacing_zyx,
            min_branch_length_um=float(min_branch_length_um),
            branch_sensitivity=float(branch_sensitivity),
            intensity=green[bbox],
            tip_intensity_floor=tip_intensity_floor,
        )
        soma_mask = shape_debug.soma_mask  # bbox-local frame
        tip_coords = shape_debug.tip_coords  # bbox-local frame

        comp_coords = np.argwhere(local_mask).astype(np.int64) + bbox_start
        nearest_cell_to_vessel = _min_distance_at_coords(
            vessel_dist, comp_coords, shift, full_shape
        )

        if np.any(soma_mask):
            soma_coords = np.argwhere(soma_mask).astype(np.int64) + bbox_start
            soma_to_vessel = _min_distance_at_coords(vessel_dist, soma_coords, shift, full_shape)
        else:
            soma_to_vessel = None

        if tip_coords.size > 0:
            tip_coords_global = tip_coords.astype(np.int64) + bbox_start
            nearest_tip_to_vessel = _min_distance_at_coords(
                vessel_dist, tip_coords_global, shift, full_shape
            )
        else:
            nearest_tip_to_vessel = None

        voxel_count = int(np.count_nonzero(local_mask))
        soma_voxel_count = int(np.count_nonzero(soma_mask))
        tip_count = int(tip_coords.shape[0])
        cells.append(
            MicrogliaCellAnalysis(
                component_id=int(component_id),
                voxel_count=voxel_count,
                volume_um3=float(voxel_count * voxel_volume_um3),
                soma_voxel_count=soma_voxel_count,
                soma_volume_um3=float(soma_voxel_count * voxel_volume_um3),
                branch_count=int(shape_debug.branch_count),
                tip_count=tip_count,
                branch_point_count=int(shape_debug.branch_point_count),
                total_process_length_um=float(shape_debug.total_process_length_um),
                mean_branch_length_um=float(shape_debug.mean_branch_length_um),
                sholl_max_intersections=int(shape_debug.sholl_max_intersections),
                sholl_critical_radius_um=float(shape_debug.sholl_critical_radius_um),
                sholl_enclosing_radius_um=float(shape_debug.sholl_enclosing_radius_um),
                nearest_tip_to_vessel_um=nearest_tip_to_vessel,
                nearest_cell_to_vessel_um=nearest_cell_to_vessel,
                soma_to_vessel_um=soma_to_vessel,
            )
        )

    distances = [
        float(cell.nearest_cell_to_vessel_um)
        for cell in cells
        if cell.nearest_cell_to_vessel_um is not None
    ]
    soma_distances = [
        float(cell.soma_to_vessel_um) for cell in cells if cell.soma_to_vessel_um is not None
    ]
    if cells and vessel_dist is not None:
        zero_cells = int(sum(1 for d in distances if d <= 1.0e-6))
        logger.info(
            "Microglia vessel distance: cells=%d vessel_occupancy=%.2f%% "
            "cell->vessel min=%.3f median=%.3f zero_contacts=%d soma->vessel min=%.3f",
            len(cells),
            100.0 * vessel_occupancy,
            float(min(distances)) if distances else float("nan"),
            float(np.median(distances)) if distances else float("nan"),
            zero_cells,
            float(min(soma_distances)) if soma_distances else float("nan"),
        )
    return MicrogliaAnalysisResult(
        cells=cells,
        analyzed_cell_count=int(len(cells)),
        mean_branch_count=float(np.mean([cell.branch_count for cell in cells])) if cells else 0.0,
        mean_tip_count=float(np.mean([cell.tip_count for cell in cells])) if cells else 0.0,
        mean_branch_point_count=(
            float(np.mean([cell.branch_point_count for cell in cells])) if cells else 0.0
        ),
        mean_process_length_um=(
            float(np.mean([cell.total_process_length_um for cell in cells])) if cells else 0.0
        ),
        mean_soma_volume_um3=float(np.mean([cell.soma_volume_um3 for cell in cells])) if cells else 0.0,
        mean_sholl_max_intersections=(
            float(np.mean([cell.sholl_max_intersections for cell in cells])) if cells else 0.0
        ),
        min_cell_to_vessel_um=float(min(distances)) if distances else None,
        min_soma_to_vessel_um=float(min(soma_distances)) if soma_distances else None,
    )


def microglia_analysis_to_csv_rows(
    analysis: MicrogliaAnalysisResult,
) -> list[dict[str, int | float | str | None]]:
    rows: list[dict[str, int | float | str | None]] = []
    for cell in analysis.cells:
        rows.append(
            {
                "component_id": int(cell.component_id),
                "voxel_count": int(cell.voxel_count),
                "volume_um3": float(cell.volume_um3),
                "soma_voxel_count": int(cell.soma_voxel_count),
                "soma_volume_um3": float(cell.soma_volume_um3),
                "branch_count": int(cell.branch_count),
                "tip_count": int(cell.tip_count),
                "branch_point_count": int(cell.branch_point_count),
                "total_process_length_um": float(cell.total_process_length_um),
                "mean_branch_length_um": float(cell.mean_branch_length_um),
                "sholl_max_intersections": int(cell.sholl_max_intersections),
                "sholl_critical_radius_um": float(cell.sholl_critical_radius_um),
                "sholl_enclosing_radius_um": float(cell.sholl_enclosing_radius_um),
                "nearest_tip_to_vessel_um": cell.nearest_tip_to_vessel_um,
                "nearest_cell_to_vessel_um": cell.nearest_cell_to_vessel_um,
                "soma_to_vessel_um": cell.soma_to_vessel_um,
            }
        )
    return rows


def build_microglia_cell_debug(
    green_volume: np.ndarray,
    red_volume: np.ndarray,
    labels: np.ndarray,
    component_id: int,
    *,
    spacing: VoxelSpacing | tuple[float, float, float],
    render: RenderConfig,
    min_branch_length_um: float = _DEFAULT_MIN_BRANCH_LENGTH_UM,
    branch_sensitivity: float = 1.0,
    max_soma_samples: int = 48,
    known_tip_distance_um: float | None = None,
    known_cell_distance_um: float | None = None,
) -> MicrogliaCellDebug | None:
    green = np.asarray(green_volume, dtype=np.float32)
    red = np.asarray(red_volume, dtype=np.float32)
    lbl = np.asarray(labels, dtype=np.int32)
    component = int(component_id)

    if green.shape != red.shape or green.shape != lbl.shape:
        raise ValueError("green_volume, red_volume, and labels must share the same shape.")
    if component <= 0:
        return None

    spacing_zyx = _spacing_zyx(spacing)
    trimmed_labels = _apply_render_trim(
        lbl,
        int(render.trim_first_slices),
        int(render.trim_last_slices),
    )
    vessel_mask = _apply_render_trim(
        red >= float(render.threshold_red),
        int(render.trim_first_slices),
        int(render.trim_last_slices),
    )
    component_mask = np.asarray(
        trimmed_labels == component,
        dtype=bool,
    )
    if not np.any(component_mask):
        return None

    shape_debug = _describe_component_shape(
        component_mask,
        spacing_zyx,
        min_branch_length_um=float(min_branch_length_um),
        branch_sensitivity=float(branch_sensitivity),
        intensity=green,
        tip_intensity_floor=float(render.threshold_green) + _TIP_VISIBILITY_MARGIN,
    )
    soma_sample_coords = _sample_mask_coords(
        shape_debug.soma_mask,
        max_points=int(max_soma_samples),
    )
    voxel_sample_coords = _sample_mask_coords(component_mask, max_points=1200)
    branch_sample_coords = _sample_mask_coords(shape_debug.branch_mask, max_points=512)
    tip_coords = shape_debug.tip_coords
    sample_shift_zyx = np.asarray(_offset_shift_zyx(render, spacing_zyx), dtype=np.int32)

    cell_segment = _nearest_segment_between_masks(
        component_mask,
        vessel_mask,
        spacing_zyx=spacing_zyx,
        known_distance_um=known_cell_distance_um,
        sample_shift_zyx=sample_shift_zyx,
    )

    tip_mask = np.zeros(component_mask.shape, dtype=bool)
    if tip_coords.size > 0:
        tip_mask[tuple(tip_coords.T)] = True
    tip_segment = _nearest_segment_between_masks(
        tip_mask,
        vessel_mask,
        spacing_zyx=spacing_zyx,
        known_distance_um=known_tip_distance_um,
        sample_shift_zyx=sample_shift_zyx,
    )

    return MicrogliaCellDebug(
        component_id=component,
        voxel_sample_coords_zyx=voxel_sample_coords,
        branch_sample_coords_zyx=branch_sample_coords,
        soma_sample_coords_zyx=soma_sample_coords,
        tip_coords_zyx=np.asarray(tip_coords, dtype=np.int32),
        nearest_tip_segment_zyx=tip_segment,
        nearest_cell_segment_zyx=cell_segment,
    )


def _describe_component_shape(
    component_mask: np.ndarray,
    spacing_zyx: np.ndarray,
    *,
    min_branch_length_um: float,
    branch_sensitivity: float,
    intensity: np.ndarray | None = None,
    tip_intensity_floor: float = 0.0,
) -> _ComponentShapeDebug:
    bounds = _mask_bounds(component_mask)
    if bounds is None:
        return _empty_shape_debug(component_mask)

    local_mask = np.asarray(component_mask[bounds], dtype=bool)
    local_intensity = (
        np.asarray(intensity[bounds], dtype=np.float32) if intensity is not None else None
    )
    # Fill small interior cavities so the skeleton does not grow loops/spurs
    # around them. This only adds enclosed background voxels (safe for thin
    # processes) and is used for soma/skeleton shape only, not voxel counts.
    if local_mask.size > 0:
        local_mask = np.asarray(
            remove_small_holes(local_mask, **{_REMOVE_SMALL_HOLES_KW: int(_FILL_HOLE_VOXELS)}),
            dtype=bool,
        )
    dist = ndi.distance_transform_edt(local_mask, sampling=tuple(float(v) for v in spacing_zyx))
    soma_local = _segment_soma_body(local_mask, dist, spacing_zyx)
    # Skeletonise only the *visible* signal (intensity at/above the render's
    # visibility floor) so processes/tips track what the user can actually see.
    # Faint halo above the raw threshold but below visibility is excluded, which
    # naturally places endpoints where the visible structure ends instead of out
    # in the near-invisible fringe. Soma/volume stay on the full thresholded mask.
    skeleton_source = local_mask
    if local_intensity is not None and float(tip_intensity_floor) > 0.0:
        visible = local_mask & (local_intensity >= float(tip_intensity_floor))
        if np.any(visible):
            skeleton_source = visible
    skeleton = np.asarray(skeletonize(skeleton_source), dtype=bool)
    offset = np.asarray([s.start or 0 for s in bounds], dtype=np.int32)

    if not np.any(skeleton):
        # Degenerate component (too small to skeletonise): soma only, no processes.
        return _ComponentShapeDebug(
            soma_mask=_embed_local_mask(component_mask.shape, bounds, soma_local),
            branch_mask=np.zeros_like(component_mask, dtype=bool),
            tip_coords=np.empty((0, 3), dtype=np.int32),
            branch_count=0,
            branch_point_count=0,
            total_process_length_um=0.0,
            mean_branch_length_um=0.0,
            sholl_max_intersections=0,
            sholl_critical_radius_um=0.0,
            sholl_enclosing_radius_um=0.0,
        )

    soma_exclusion = ndi.binary_dilation(soma_local, structure=_FULL_STRUCTURE, iterations=1)
    # A branch endpoint that sits right where the soma was carved out is a process
    # root, not a tip; flag any branch voxel touching the soma exclusion zone.
    soma_contact = ndi.binary_dilation(soma_exclusion, structure=_FULL_STRUCTURE, iterations=1)
    branch_skeleton = np.asarray(skeleton & (~soma_exclusion), dtype=bool)

    topology = _branch_topology(
        branch_skeleton,
        soma_contact=soma_contact,
        spacing_zyx=spacing_zyx,
        min_branch_length_um=float(min_branch_length_um),
        branch_sensitivity=float(branch_sensitivity),
        dist_local=dist,
    )

    soma_centroid = _mask_centroid(soma_local)
    sholl = _sholl_metrics(topology.branch_mask, soma_centroid, spacing_zyx)

    # Convert raw skeleton endpoints into biological process terminals: drop
    # endpoints buried in the thick cell body, then collapse the endpoint
    # cluster of each lamellar "fan" into a single representative tip.
    gated_tips = _gate_and_cluster_tips(
        topology.tip_coords,
        dist_local=dist,
        soma_local=soma_local,
        spacing_zyx=spacing_zyx,
        branch_sensitivity=branch_sensitivity,
        intensity_local=local_intensity,
        intensity_floor=float(tip_intensity_floor),
    )
    tip_coords = gated_tips + offset if gated_tips.size > 0 else gated_tips
    return _ComponentShapeDebug(
        soma_mask=_embed_local_mask(component_mask.shape, bounds, soma_local),
        branch_mask=_embed_local_mask(component_mask.shape, bounds, topology.branch_mask),
        tip_coords=np.asarray(tip_coords, dtype=np.int32),
        branch_count=int(topology.branch_count),
        branch_point_count=int(topology.branch_point_count),
        total_process_length_um=float(topology.total_length_um),
        mean_branch_length_um=float(topology.mean_length_um),
        sholl_max_intersections=int(sholl.max_intersections),
        sholl_critical_radius_um=float(sholl.critical_radius_um),
        sholl_enclosing_radius_um=float(sholl.enclosing_radius_um),
    )


def _empty_shape_debug(component_mask: np.ndarray) -> _ComponentShapeDebug:
    zeros = np.zeros_like(component_mask, dtype=bool)
    return _ComponentShapeDebug(
        soma_mask=zeros,
        branch_mask=np.zeros_like(component_mask, dtype=bool),
        tip_coords=np.empty((0, 3), dtype=np.int32),
        branch_count=0,
        branch_point_count=0,
        total_process_length_um=0.0,
        mean_branch_length_um=0.0,
        sholl_max_intersections=0,
        sholl_critical_radius_um=0.0,
        sholl_enclosing_radius_um=0.0,
    )


def _segment_soma_body(
    component_mask: np.ndarray,
    dist: np.ndarray,
    spacing_zyx: np.ndarray,
) -> np.ndarray:
    if not np.any(component_mask):
        return np.zeros_like(component_mask, dtype=bool)

    max_flat = int(np.argmax(dist))
    center = np.unravel_index(max_flat, dist.shape)
    max_dist = float(dist[center])
    if max_dist <= 0.0:
        soma = np.zeros_like(component_mask, dtype=bool)
        soma[center] = True
        return soma

    min_spacing = float(np.min(spacing_zyx))
    core_floor = float(max(max_dist * 0.45, min_spacing * 1.15))
    body_floor = float(max(max_dist * 0.35, min_spacing * 1.02))
    core_mask = np.asarray(component_mask & (dist >= core_floor), dtype=bool)
    if not core_mask[center]:
        core_mask[center] = True
    core_labels, _ = ndi.label(core_mask, structure=_FULL_STRUCTURE)
    core_id = int(core_labels[center])
    if core_id > 0:
        core_mask = np.asarray(core_labels == core_id, dtype=bool)

    body_mask = np.asarray(component_mask & (dist >= body_floor), dtype=bool)
    if np.any(body_mask):
        soma = np.asarray(
            ndi.binary_propagation(core_mask, mask=body_mask, structure=_FULL_STRUCTURE),
            dtype=bool,
        )
        if np.any(soma):
            return soma
    return core_mask


def _branch_topology(
    branch_skeleton: np.ndarray,
    *,
    soma_contact: np.ndarray,
    spacing_zyx: np.ndarray,
    min_branch_length_um: float,
    branch_sensitivity: float = 1.0,
    dist_local: np.ndarray | None = None,
) -> _BranchTopology:
    """Decompose the soma-excluded skeleton into branches/tips/junctions.

    Uses skan for junction-cluster-aware topology and geodesic, spacing-correct
    branch lengths in 3D; falls back to a local-degree estimate if skan is
    unavailable or cannot build a graph for a tiny skeleton.
    """
    skeleton = np.asarray(branch_skeleton, dtype=bool)
    if not np.any(skeleton):
        return _empty_topology(skeleton)

    if _HAS_SKAN:
        result = _branch_topology_skan(
            skeleton,
            soma_contact=soma_contact,
            spacing_zyx=spacing_zyx,
            min_branch_length_um=min_branch_length_um,
            branch_sensitivity=branch_sensitivity,
            dist_local=dist_local,
        )
        if result is not None:
            return result

    return _branch_topology_fallback(
        skeleton,
        soma_contact=soma_contact,
        spacing_zyx=spacing_zyx,
    )


def _terminal_spur_indices(
    skel,
    *,
    min_branch_length_um: float,
    branch_sensitivity: float,
    dist_local: np.ndarray | None,
) -> np.ndarray:
    """Indices of terminal branches that are too short to be real processes.

    A terminal branch is pruned when its geodesic length falls below
    ``max(min_branch_length_um, spur_factor * root_radius)``, where ``root_radius``
    is the structure thickness (EDT) at the branch's attachment point. The
    resulting threshold is scaled by ``1 / branch_sensitivity`` so higher
    sensitivity keeps shorter tips while lower sensitivity prunes more.
    """
    try:
        lengths = np.asarray(skel.path_lengths(), dtype=np.float64)
        summary = _skan_summarize(skel, separator="-")
        branch_types = np.asarray(summary["branch-type"].to_numpy(), dtype=np.int64)
        src_ids = np.asarray(summary["node-id-src"].to_numpy(), dtype=np.int64)
        dst_ids = np.asarray(summary["node-id-dst"].to_numpy(), dtype=np.int64)
    except Exception:
        return np.empty((0,), dtype=np.int64)

    sensitivity = float(np.clip(branch_sensitivity, 0.4, 2.0))
    length_scale = 1.0 / sensitivity
    floor = float(max(0.0, min_branch_length_um)) * length_scale
    thresholds = np.full(lengths.shape, floor, dtype=np.float64)

    if dist_local is not None:
        coords = np.rint(np.asarray(skel.coordinates, dtype=np.float64)).astype(np.int64)
        degrees = np.asarray(skel.degrees, dtype=np.int64)
        shape = np.asarray(dist_local.shape, dtype=np.int64)
        for i in range(lengths.shape[0]):
            s, d = int(src_ids[i]), int(dst_ids[i])
            # Attachment node = the higher-degree (junction) end of the branch.
            root = s if degrees[s] >= degrees[d] else d
            rc = coords[root]
            if np.all((rc >= 0) & (rc < shape)):
                root_radius = float(dist_local[rc[0], rc[1], rc[2]])
                radius_thr = (
                    min(_SPUR_RADIUS_FACTOR * root_radius, _SPUR_LENGTH_CAP_UM)
                    * length_scale
                )
                thresholds[i] = max(thresholds[i], radius_thr)

    terminal = np.isin(branch_types, _TERMINAL_BRANCH_TYPES)
    return np.nonzero(terminal & (lengths < thresholds))[0]


def _branch_topology_skan(
    skeleton: np.ndarray,
    *,
    soma_contact: np.ndarray,
    spacing_zyx: np.ndarray,
    min_branch_length_um: float,
    branch_sensitivity: float = 1.0,
    dist_local: np.ndarray | None = None,
) -> _BranchTopology | None:
    sampling = tuple(float(v) for v in spacing_zyx)
    try:
        skel = _SkanSkeleton(skeleton, spacing=sampling)
    except Exception:
        # skan raises for skeletons with no traceable path (e.g. a single voxel).
        return None

    if float(max(0.0, min_branch_length_um)) > 0.0 or dist_local is not None:
        for _ in range(12):
            if skel.n_paths <= 0:
                break
            drop = _terminal_spur_indices(
                skel,
                min_branch_length_um=min_branch_length_um,
                branch_sensitivity=branch_sensitivity,
                dist_local=dist_local,
            )
            if drop.size == 0:
                break
            if int(drop.size) >= int(skel.n_paths):
                # Every remaining branch is a sub-threshold spur: no real processes.
                return _empty_topology(skeleton)
            try:
                skel = skel.prune_paths(drop.tolist())
            except Exception:
                break

    if skel.n_paths <= 0:
        return _empty_topology(skeleton)

    coords = np.rint(np.asarray(skel.coordinates, dtype=np.float64)).astype(np.int32)
    degrees = np.asarray(skel.degrees, dtype=np.int64)
    if coords.shape[0] != degrees.shape[0]:
        return None

    branch_mask = np.zeros(skeleton.shape, dtype=bool)
    in_bounds = _coords_in_bounds(coords, skeleton.shape)
    branch_mask[tuple(coords[in_bounds].T)] = True

    endpoint_coords = coords[(degrees == 1) & in_bounds]
    tip_coords = _filter_non_contact_coords(endpoint_coords, soma_contact)

    junction_coords = coords[(degrees >= 3) & in_bounds]
    branch_point_count = _connected_node_count(junction_coords, skeleton.shape)

    try:
        total_length = float(np.sum(skel.path_lengths()))
    except Exception:
        total_length = 0.0
    branch_count = int(skel.n_paths)
    mean_length = float(total_length / branch_count) if branch_count > 0 else 0.0

    return _BranchTopology(
        tip_coords=tip_coords,
        branch_mask=branch_mask,
        branch_count=branch_count,
        branch_point_count=branch_point_count,
        total_length_um=total_length,
        mean_length_um=mean_length,
    )


def _branch_topology_fallback(
    skeleton: np.ndarray,
    *,
    soma_contact: np.ndarray,
    spacing_zyx: np.ndarray,
) -> _BranchTopology:
    degree = ndi.convolve(
        skeleton.astype(np.uint8), _DEGREE_KERNEL, mode="constant", cval=0
    )
    endpoint_mask = skeleton & (degree <= 1)
    endpoint_coords = np.argwhere(endpoint_mask).astype(np.int32, copy=False)
    tip_coords = _filter_non_contact_coords(endpoint_coords, soma_contact)

    junction_coords = np.argwhere(skeleton & (degree >= 3)).astype(np.int32, copy=False)
    branch_point_count = _connected_node_count(junction_coords, skeleton.shape)
    branch_count, total_length = _fallback_branch_segments(
        skeleton,
        degree=degree,
        spacing_zyx=spacing_zyx,
    )

    mean_length = float(total_length / branch_count) if branch_count > 0 else 0.0

    return _BranchTopology(
        tip_coords=tip_coords,
        branch_mask=np.asarray(skeleton, dtype=bool),
        branch_count=branch_count,
        branch_point_count=branch_point_count,
        total_length_um=total_length,
        mean_length_um=mean_length,
    )


def _empty_topology(skeleton: np.ndarray) -> _BranchTopology:
    return _BranchTopology(
        tip_coords=np.empty((0, 3), dtype=np.int32),
        branch_mask=np.zeros(skeleton.shape, dtype=bool),
        branch_count=0,
        branch_point_count=0,
        total_length_um=0.0,
        mean_length_um=0.0,
    )


def _fallback_branch_segments(
    skeleton: np.ndarray,
    *,
    degree: np.ndarray,
    spacing_zyx: np.ndarray,
) -> tuple[int, float]:
    """Count endpoint/junction paths when skan is unavailable."""
    skel = np.asarray(skeleton, dtype=bool)
    coords = np.argwhere(skel).astype(np.int32, copy=False)
    if coords.size == 0:
        return 0, 0.0

    node_mask = skel & (degree != 2)
    node_coords = np.argwhere(node_mask).astype(np.int32, copy=False)
    if node_coords.size == 0:
        # Closed loop: one branch, approximate length from skeleton voxels.
        return 1, float(coords.shape[0] * float(np.mean(spacing_zyx)))

    node_labels, _ = ndi.label(node_mask, structure=_FULL_STRUCTURE)
    coord_to_node = {
        tuple(int(v) for v in row): int(node_labels[int(row[0]), int(row[1]), int(row[2])])
        for row in node_coords.tolist()
    }
    skel_set = {tuple(int(v) for v in row) for row in coords.tolist()}
    visited_edges: set[frozenset[tuple[int, int, int]]] = set()
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    branch_count = 0
    total_length = 0.0

    for node, start_node_id in coord_to_node.items():
        for neighbor in _skeleton_neighbors(node, skel_set):
            neighbor_node_id = coord_to_node.get(neighbor, 0)
            if neighbor_node_id == start_node_id:
                continue
            edge = frozenset((node, neighbor))
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            prev = node
            current = neighbor
            length = _neighbor_step_length(prev, current, spacing)
            while current not in coord_to_node:
                next_neighbors = [
                    n for n in _skeleton_neighbors(current, skel_set) if n != prev
                ]
                if not next_neighbors:
                    break
                nxt = next_neighbors[0]
                visited_edges.add(frozenset((current, nxt)))
                length += _neighbor_step_length(current, nxt, spacing)
                prev, current = current, nxt
            branch_count += 1
            total_length += float(length)

    if branch_count <= 0:
        _, components = ndi.label(skel, structure=_FULL_STRUCTURE)
        branch_count = int(components)
        total_length = float(coords.shape[0] * float(np.mean(spacing_zyx)))
    return branch_count, float(total_length)


def _skeleton_neighbors(
    coord: tuple[int, int, int],
    skel_set: set[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    z, y, x = coord
    out: list[tuple[int, int, int]] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                n = (z + dz, y + dy, x + dx)
                if n in skel_set:
                    out.append(n)
    return out


def _neighbor_step_length(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    spacing_zyx: np.ndarray,
) -> float:
    delta = np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)
    return float(np.linalg.norm(delta * spacing_zyx))


def _sholl_metrics(
    branch_skeleton: np.ndarray,
    soma_centroid_zyx: np.ndarray | None,
    spacing_zyx: np.ndarray,
) -> _ShollMetrics:
    """3D Sholl ramification anchored at the soma centroid.

    Counts process intersections (connected components of the skeleton) within
    concentric spherical shells in physical units, so anisotropic spacing and
    full 3D topology are respected.
    """
    coords = np.argwhere(np.asarray(branch_skeleton, dtype=bool)).astype(np.float64, copy=False)
    if coords.shape[0] == 0 or soma_centroid_zyx is None:
        return _ShollMetrics(0, 0.0, 0.0)

    spacing = np.asarray(spacing_zyx, dtype=np.float64).reshape(1, 3)
    centroid = np.asarray(soma_centroid_zyx, dtype=np.float64).reshape(1, 3)
    radii = np.linalg.norm((coords - centroid) * spacing, axis=1)
    enclosing = float(np.max(radii))
    if enclosing <= 0.0:
        return _ShollMetrics(0, 0.0, enclosing)

    step = float(max(float(np.min(spacing)) * 2.0, 1.0))
    n_shells = int(np.ceil(enclosing / step))
    int_coords = np.rint(coords).astype(np.int32)

    best_count = 0
    best_radius = 0.0
    for k in range(n_shells):
        r0 = k * step
        r1 = r0 + step
        in_shell = (radii >= r0) & (radii < r1)
        if not np.any(in_shell):
            continue
        shell_mask = np.zeros(branch_skeleton.shape, dtype=bool)
        shell_coords = int_coords[in_shell]
        in_bounds = _coords_in_bounds(shell_coords, branch_skeleton.shape)
        if not np.any(in_bounds):
            continue
        shell_mask[tuple(shell_coords[in_bounds].T)] = True
        _, intersections = ndi.label(shell_mask, structure=_FULL_STRUCTURE)
        if int(intersections) > best_count:
            best_count = int(intersections)
            best_radius = float(0.5 * (r0 + r1))

    return _ShollMetrics(best_count, best_radius, enclosing)


def _filter_non_contact_coords(coords: np.ndarray, contact_mask: np.ndarray) -> np.ndarray:
    pts = np.asarray(coords, dtype=np.int32)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    keep = ~np.asarray(contact_mask, dtype=bool)[tuple(pts.T)]
    filtered = pts[keep]
    if filtered.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(filtered, dtype=np.int32)


def _gate_and_cluster_tips(
    tips_local: np.ndarray,
    *,
    dist_local: np.ndarray,
    soma_local: np.ndarray,
    spacing_zyx: np.ndarray,
    branch_sensitivity: float,
    intensity_local: np.ndarray | None = None,
    intensity_floor: float = 0.0,
) -> np.ndarray:
    """Turn raw skeleton endpoints into biological process terminals.

    Three corrections:

    * Visibility gate - the volume render ramps opacity from zero at the display
      threshold and only becomes visible somewhat above it, so endpoints sitting
      on signal below ``intensity_floor`` look like they float in empty space.
      They are dropped so tips land only on structure the user can actually see.
    * Thickness gate - a genuine process tip is thin, so endpoints sitting in
      voxels thicker than a soma-relative limit (i.e. inside the cell body) are
      discarded rather than counted as tips.
    * Terminal clustering - microglia processes end in flat lamellar "fans"
      whose skeletons fragment into many adjacent endpoints. Endpoints within a
      merge radius collapse to the single most distal representative, so one fan
      counts as one tip. The radius scales inversely with branch sensitivity.
    """
    pts = np.asarray(tips_local, dtype=np.int32)
    if pts.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int32)

    shape = np.asarray(dist_local.shape, dtype=np.int64).reshape(1, 3)
    pts = pts[np.all((pts >= 0) & (pts < shape), axis=1)]
    if pts.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int32)

    if intensity_local is not None and float(intensity_floor) > 0.0:
        values = np.asarray(intensity_local, dtype=np.float32)[pts[:, 0], pts[:, 1], pts[:, 2]]
        pts = pts[values >= float(intensity_floor)]
        if pts.shape[0] == 0:
            return np.empty((0, 3), dtype=np.int32)

    soma_mask = np.asarray(soma_local, dtype=bool)
    soma_radius = float(dist_local[soma_mask].max()) if np.any(soma_mask) else 0.0
    max_thickness = float(max(_TIP_THICKNESS_FLOOR_UM, _TIP_THICKNESS_SOMA_FRACTION * soma_radius))
    radii = np.asarray(dist_local[pts[:, 0], pts[:, 1], pts[:, 2]], dtype=np.float64)
    pts = pts[radii <= max_thickness]
    if pts.shape[0] <= 1:
        return np.asarray(pts, dtype=np.int32)

    sensitivity = float(np.clip(branch_sensitivity, 0.4, 2.0))
    merge_radius = float(np.clip(_DEFAULT_TIP_MERGE_RADIUS_UM / sensitivity, 1.5, 12.0))
    spacing = np.asarray(spacing_zyx, dtype=np.float64).reshape(1, 3)
    pts_um = pts.astype(np.float64) * spacing

    centroid = _mask_centroid(soma_mask)
    if centroid is not None:
        soma_um = np.asarray(centroid, dtype=np.float64).reshape(1, 3) * spacing
        order = np.argsort(np.linalg.norm(pts_um - soma_um, axis=1))[::-1]
    else:
        order = np.arange(pts.shape[0])

    merge_sq = merge_radius * merge_radius
    kept_idx: list[int] = []
    kept_um: list[np.ndarray] = []
    for i in order.tolist():
        candidate = pts_um[i]
        if any(float(np.dot(candidate - k, candidate - k)) < merge_sq for k in kept_um):
            continue
        kept_idx.append(int(i))
        kept_um.append(candidate)
    return np.asarray(pts[kept_idx], dtype=np.int32)


def _connected_node_count(coords: np.ndarray, shape: tuple[int, int, int]) -> int:
    pts = np.asarray(coords, dtype=np.int32)
    if pts.size == 0:
        return 0
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(pts.T)] = True
    _, count = ndi.label(mask, structure=_FULL_STRUCTURE)
    return int(count)


def _coords_in_bounds(coords: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    pts = np.asarray(coords, dtype=np.int64)
    if pts.size == 0:
        return np.zeros((0,), dtype=bool)
    bounds = np.asarray(shape, dtype=np.int64).reshape(1, 3)
    return np.all((pts >= 0) & (pts < bounds), axis=1)


def _mask_centroid(mask: np.ndarray) -> np.ndarray | None:
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        return None
    return np.asarray(ndi.center_of_mass(binary), dtype=np.float64)


def _sample_mask_coords(mask: np.ndarray, *, max_points: int) -> np.ndarray:
    coords = np.argwhere(np.asarray(mask, dtype=bool)).astype(np.int32, copy=False)
    if coords.size <= 0:
        return np.empty((0, 3), dtype=np.int32)
    if coords.shape[0] <= int(max_points):
        return np.asarray(coords, dtype=np.int32)
    positions = np.linspace(0, coords.shape[0] - 1, num=int(max_points), dtype=np.int32)
    return np.asarray(coords[positions], dtype=np.int32)


def _nearest_segment_between_masks(
    sample_mask: np.ndarray,
    vessel_mask: np.ndarray,
    *,
    spacing_zyx: np.ndarray,
    known_distance_um: float | None,
    sample_shift_zyx: np.ndarray | None = None,
) -> np.ndarray | None:
    sample = np.asarray(sample_mask, dtype=bool)
    vessel = np.asarray(vessel_mask, dtype=bool)
    if not np.any(sample) or not np.any(vessel):
        return None

    sample_coords = np.argwhere(sample).astype(np.int32, copy=False)
    if sample_shift_zyx is None:
        sample_shift = np.zeros((1, 3), dtype=np.int32)
    else:
        sample_shift = np.asarray(sample_shift_zyx, dtype=np.int32).reshape(1, 3)
    shifted_sample_coords = sample_coords + sample_shift
    shape = np.asarray(vessel.shape, dtype=np.int32).reshape(1, 3)
    in_bounds = np.all((shifted_sample_coords >= 0) & (shifted_sample_coords < shape), axis=1)
    if not np.any(in_bounds):
        return None
    sample_coords = np.asarray(sample_coords[in_bounds], dtype=np.int32)
    shifted_sample_coords = np.asarray(shifted_sample_coords[in_bounds], dtype=np.int32)

    search_bounds = _expanded_search_bounds(
        shifted_sample_coords,
        vessel.shape,
        spacing_zyx=spacing_zyx,
        known_distance_um=known_distance_um,
    )
    local_vessel = np.asarray(vessel[search_bounds], dtype=bool)
    vessel_coords_local = np.argwhere(local_vessel).astype(np.int32, copy=False)
    if vessel_coords_local.size <= 0:
        vessel_coords = np.argwhere(vessel).astype(np.int32, copy=False)
    else:
        offset = np.asarray([sl.start or 0 for sl in search_bounds], dtype=np.int32)
        vessel_coords = vessel_coords_local + offset
    if vessel_coords.size <= 0:
        return None

    sample_um = shifted_sample_coords.astype(np.float32, copy=False) * spacing_zyx.reshape(1, 3)
    vessel_um = vessel_coords.astype(np.float32, copy=False) * spacing_zyx.reshape(1, 3)
    tree = cKDTree(vessel_um)
    dist, idx = tree.query(sample_um, k=1)
    best = int(np.argmin(dist))
    return np.asarray(
        [sample_coords[best], vessel_coords[int(idx[best])]],
        dtype=np.int32,
    )


def _expanded_search_bounds(
    sample_coords: np.ndarray,
    shape: tuple[int, int, int],
    *,
    spacing_zyx: np.ndarray,
    known_distance_um: float | None,
) -> tuple[slice, slice, slice]:
    mins = np.min(sample_coords, axis=0)
    maxs = np.max(sample_coords, axis=0)
    if known_distance_um is None or not np.isfinite(float(known_distance_um)):
        margin = np.asarray([8, 24, 24], dtype=np.int32)
    else:
        radius = float(max(0.0, float(known_distance_um)))
        margin = np.ceil(radius / np.maximum(spacing_zyx, 1.0e-6)).astype(np.int32) + 2
    start = np.maximum(0, mins - margin)
    stop = np.minimum(np.asarray(shape, dtype=np.int32), maxs + margin + 1)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(start.tolist(), stop.tolist(), strict=True))


def _distance_to_vessel(vessel_mask: np.ndarray, spacing_zyx: np.ndarray) -> np.ndarray | None:
    binary = np.asarray(vessel_mask, dtype=bool)
    if not np.any(binary):
        return None
    sampling = tuple(float(v) for v in spacing_zyx)
    try:
        dt = ndi.distance_transform_edt(~binary, sampling=sampling)
    except MemoryError:
        # Keep physical (micron) units even under memory pressure: process the
        # volume slice-by-slice rather than dropping the spacing, which would
        # silently switch the reported distances to voxel units.
        dt = np.empty(binary.shape, dtype=np.float32)
        for z in range(binary.shape[0]):
            dt[z] = ndi.distance_transform_edt(~binary[z], sampling=sampling[1:])
    return np.asarray(dt, dtype=np.float32)


def _min_distance_at_coords(
    vessel_distance: np.ndarray | None,
    coords_zyx: np.ndarray,
    shift_zyx: np.ndarray,
    shape: tuple[int, int, int],
) -> float | None:
    """Shortest vessel distance over a set of (optionally shifted) voxel coords.

    Equivalent to shifting a mask and sampling ``vessel_distance`` at the in-bounds
    voxels, but works directly on a coordinate list so no full-volume mask has to
    be allocated per component.
    """
    if vessel_distance is None:
        return None
    coords = np.asarray(coords_zyx, dtype=np.int64)
    if coords.shape[0] == 0:
        return None
    shifted = coords + np.asarray(shift_zyx, dtype=np.int64).reshape(1, 3)
    bounds = np.asarray(shape, dtype=np.int64).reshape(1, 3)
    in_bounds = np.all((shifted >= 0) & (shifted < bounds), axis=1)
    if not np.any(in_bounds):
        return None
    s = shifted[in_bounds]
    return float(np.min(vessel_distance[s[:, 0], s[:, 1], s[:, 2]]))


def _apply_render_trim(volume: np.ndarray, trim_first_slices: int, trim_last_slices: int) -> np.ndarray:
    arr = np.asarray(volume)
    if arr.ndim != 3 or arr.shape[0] <= 0:
        return np.array(arr, copy=True)
    trim_first = max(0, int(trim_first_slices))
    trim_last = max(0, int(trim_last_slices))
    if trim_first <= 0 and trim_last <= 0:
        return np.array(arr, copy=True)
    if trim_first + trim_last >= int(arr.shape[0]):
        return np.zeros_like(arr)
    out = np.array(arr, copy=True)
    if trim_first > 0:
        out[:trim_first] = 0
    if trim_last > 0:
        out[-trim_last:] = 0
    return out


def _offset_shift_zyx(render: RenderConfig, spacing_zyx: np.ndarray) -> tuple[int, int, int]:
    dz = int(round(float(render.offset_z_um) / float(max(spacing_zyx[0], 1.0e-6))))
    dy = int(round(float(render.offset_y_um) / float(max(spacing_zyx[1], 1.0e-6))))
    dx = int(round(float(render.offset_x_um) / float(max(spacing_zyx[2], 1.0e-6))))
    return dz, dy, dx


def _mask_bounds(mask: np.ndarray) -> tuple[slice, slice, slice] | None:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size <= 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins.tolist(), maxs.tolist(), strict=True))


def _embed_local_mask(
    shape: tuple[int, int, int],
    bounds: tuple[slice, slice, slice],
    local_mask: np.ndarray,
) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[bounds] = np.asarray(local_mask, dtype=bool)
    return out


def _spacing_zyx(spacing: VoxelSpacing | tuple[float, float, float]) -> np.ndarray:
    if isinstance(spacing, VoxelSpacing):
        return np.asarray((spacing.z_um, spacing.y_um, spacing.x_um), dtype=np.float32)
    arr = np.asarray(spacing, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError("spacing must provide (z_um, y_um, x_um).")
    return np.maximum(arr, 1.0e-6)
