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
_SPUR_LENGTH_CAP_UM = 4.5
# Interior holes smaller than this (in voxels) are filled before skeletonising
# so they do not create spurious skeleton loops / branches.
_FILL_HOLE_VOXELS = 128
# Endpoints within this physical radius collapse to a single process terminal.
# Microglia terminals are flat lamellar "fans" whose skeletons fragment into
# many endpoints; clustering converts those clusters back into one tip each.
_DEFAULT_TIP_MERGE_RADIUS_UM = 4.5
# A real process tip is thin: drop endpoints sitting in voxels thicker than this.
# The cutoff is derived from the soma *boundary* EDT (minimum EDT inside the
# detected soma), NOT the soma centre (maximum EDT).  Using the boundary EDT
# closes the "soma shell" gap — without this, endpoints in the 35–45% EDT band
# (inside the soma surface but outside the detected soma core) pass every filter
# and appear as hundreds of false tips all over the cell body.
# max_thickness = max(floor, soma_boundary_edt × _TIP_THICKNESS_BOUNDARY_FACTOR)
_TIP_THICKNESS_BOUNDARY_FACTOR = 0.80   # keeps cutoff strictly below body_floor
_TIP_THICKNESS_FLOOR_UM = 1.5
# Intensity margin above the display threshold for tip detection. Kept at 0 so
# the tip skeleton and the voxel-debug cloud both track exactly the rendered
# structure (raw intensity >= render threshold). A positive margin raised the tip
# threshold above the render/voxel threshold, which made tips retract inward and
# fall short of the visible branch ends.
_TIP_VISIBILITY_MARGIN = 0.0
# Geodesic tip detection uses this fraction of the branch-length floor as the
# h-maxima prominence gate. The full branch-length floor was too conservative:
# it kept long terminals accurate, but missed shorter visible terminal twigs.
_TIP_PROMINENCE_LENGTH_FACTOR = 0.50
# Raw graph endpoints can land just outside the thresholded component after
# physical-coordinate rounding. Snap tiny misses back onto the component, but
# discard anything farther away so tips cannot appear in empty space.
_TIP_SUPPORT_SNAP_FLOOR_UM = 1.25
_TIP_SUPPORT_SNAP_SPACING_FACTOR = 2.0

try:  # skan provides robust, junction-aware skeleton topology (3D, anisotropy-aware).
    from skan import Skeleton as _SkanSkeleton, summarize as _skan_summarize

    _HAS_SKAN = True
except Exception:  # pragma: no cover - exercised only when optional dep missing
    _SkanSkeleton = None  # type: ignore[assignment]
    _skan_summarize = None  # type: ignore[assignment]
    _HAS_SKAN = False

try:  # active tip path: geodesic distance from the soma through the voxel cloud.
    from skimage.graph import MCP_Geometric as _MCP_Geometric
    from skimage.morphology import h_maxima as _h_maxima

    _HAS_GEODESIC_TIPS = True
except Exception:  # pragma: no cover - exercised only when optional dep missing
    _MCP_Geometric = None  # type: ignore[assignment]
    _h_maxima = None  # type: ignore[assignment]
    _HAS_GEODESIC_TIPS = False

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
    soma_centroid_to_vessel_um: float | None = None
    soma_equivalent_diameter_um: float = 0.0
    soma_roundness: float = 0.0
    soma_elongation: float = 0.0
    mean_branch_tortuosity: float = 0.0
    tip_near_vessel_component_count: int = 0
    tips_near_multiple_vessels: bool = False


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
    mean_soma_centroid_to_vessel_um: float | None = None
    mean_soma_equivalent_diameter_um: float = 0.0
    mean_soma_roundness: float = 0.0
    mean_branch_tortuosity: float = 0.0
    cells_with_tips_near_multiple_vessels: int = 0


@dataclass(frozen=True)
class MicrogliaCellDebug:
    component_id: int
    voxel_sample_coords_zyx: np.ndarray
    branch_sample_coords_zyx: np.ndarray
    soma_sample_coords_zyx: np.ndarray
    tip_coords_zyx: np.ndarray
    nearest_tip_segment_zyx: np.ndarray | None
    nearest_soma_segment_zyx: np.ndarray | None
    nearest_cell_segment_zyx: np.ndarray | None


@dataclass(frozen=True)
class _BranchTopology:
    tip_coords: np.ndarray
    branch_mask: np.ndarray
    branch_count: int
    branch_point_count: int
    total_length_um: float
    mean_length_um: float
    mean_tortuosity: float


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
    soma_equivalent_diameter_um: float
    soma_roundness: float
    soma_elongation: float
    mean_branch_tortuosity: float


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
    vessel_labels: np.ndarray | None = None
    if vessel_voxels > 0:
        vessel_labels, _ = ndi.label(vessel_mask, structure=_FULL_STRUCTURE)
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
            soma_centroid = _mask_centroid(soma_mask)
            if soma_centroid is not None:
                soma_centroid_to_vessel = _distance_at_coord(
                    vessel_dist,
                    np.rint(soma_centroid).astype(np.int64) + bbox_start,
                    shift,
                    full_shape,
                )
            else:
                soma_centroid_to_vessel = None
        else:
            soma_to_vessel = None
            soma_centroid_to_vessel = None

        if tip_coords.size > 0:
            tip_coords_global = tip_coords.astype(np.int64) + bbox_start
            nearest_tip_to_vessel = _min_distance_at_coords(
                vessel_dist, tip_coords_global, shift, full_shape
            )
            tip_near_vessel_component_count = _nearby_vessel_component_count(
                vessel_labels,
                tip_coords_global,
                shift,
                spacing_zyx,
                full_shape,
            )
        else:
            nearest_tip_to_vessel = None
            tip_near_vessel_component_count = 0

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
                soma_centroid_to_vessel_um=soma_centroid_to_vessel,
                soma_equivalent_diameter_um=float(shape_debug.soma_equivalent_diameter_um),
                soma_roundness=float(shape_debug.soma_roundness),
                soma_elongation=float(shape_debug.soma_elongation),
                mean_branch_tortuosity=float(shape_debug.mean_branch_tortuosity),
                tip_near_vessel_component_count=int(tip_near_vessel_component_count),
                tips_near_multiple_vessels=bool(tip_near_vessel_component_count >= 2),
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
    soma_centroid_distances = [
        float(cell.soma_centroid_to_vessel_um)
        for cell in cells
        if cell.soma_centroid_to_vessel_um is not None
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
        mean_soma_centroid_to_vessel_um=(
            float(np.mean(soma_centroid_distances)) if soma_centroid_distances else None
        ),
        mean_soma_equivalent_diameter_um=(
            float(np.mean([cell.soma_equivalent_diameter_um for cell in cells])) if cells else 0.0
        ),
        mean_soma_roundness=(
            float(np.mean([cell.soma_roundness for cell in cells])) if cells else 0.0
        ),
        mean_branch_tortuosity=(
            float(np.mean([cell.mean_branch_tortuosity for cell in cells])) if cells else 0.0
        ),
        cells_with_tips_near_multiple_vessels=int(
            sum(1 for cell in cells if cell.tips_near_multiple_vessels)
        ),
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
                "soma_centroid_to_vessel_um": cell.soma_centroid_to_vessel_um,
                "soma_equivalent_diameter_um": float(cell.soma_equivalent_diameter_um),
                "soma_roundness": float(cell.soma_roundness),
                "soma_elongation": float(cell.soma_elongation),
                "mean_branch_tortuosity": float(cell.mean_branch_tortuosity),
                "tip_near_vessel_component_count": int(cell.tip_near_vessel_component_count),
                "tips_near_multiple_vessels": int(bool(cell.tips_near_multiple_vessels)),
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
    known_soma_distance_um: float | None = None,
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
    # Sample the voxel-debug cloud only where the cell is actually rendered: raw
    # intensity at/above the render threshold within the component. The component
    # is detected on a smoothed volume, so component_mask alone includes
    # sub-threshold fringe/bridge voxels that show up as debug spheres floating in
    # space where nothing is rendered. The denser cap fills the cloud.
    visible_mask = component_mask & (green >= float(render.threshold_green))
    if not np.any(visible_mask):
        visible_mask = component_mask
    voxel_sample_coords = _sample_mask_coords(visible_mask, max_points=5000)
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
    if not np.any(tip_mask):
        tip_mask = np.asarray(shape_debug.branch_mask, dtype=bool)
    if not np.any(tip_mask):
        tip_mask = component_mask
    tip_segment = _nearest_segment_between_masks(
        tip_mask,
        vessel_mask,
        spacing_zyx=spacing_zyx,
        known_distance_um=known_tip_distance_um,
        sample_shift_zyx=sample_shift_zyx,
    )
    soma_segment = _nearest_segment_between_masks(
        shape_debug.soma_mask,
        vessel_mask,
        spacing_zyx=spacing_zyx,
        known_distance_um=known_soma_distance_um,
        sample_shift_zyx=sample_shift_zyx,
    )

    return MicrogliaCellDebug(
        component_id=component,
        voxel_sample_coords_zyx=voxel_sample_coords,
        branch_sample_coords_zyx=branch_sample_coords,
        soma_sample_coords_zyx=soma_sample_coords,
        tip_coords_zyx=np.asarray(tip_coords, dtype=np.int32),
        nearest_tip_segment_zyx=tip_segment,
        nearest_soma_segment_zyx=soma_segment,
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

    component_local = np.asarray(component_mask[bounds], dtype=bool)
    local_mask = component_local
    local_intensity = (
        np.asarray(intensity[bounds], dtype=np.float32) if intensity is not None else None
    )
    # Fill small interior cavities so the skeleton does not grow loops/spurs
    # around them. This only adds enclosed background voxels (safe for thin
    # processes) and is used for soma/skeleton shape only, not voxel counts.
    if local_mask.size > 0:
        filled_mask = np.asarray(
            remove_small_holes(local_mask, **{_REMOVE_SMALL_HOLES_KW: int(_FILL_HOLE_VOXELS)}),
            dtype=bool,
        )
        added = int(np.count_nonzero(filled_mask) - np.count_nonzero(local_mask))
        if 0 < added <= int(_FILL_HOLE_VOXELS):
            local_mask = filled_mask
    dist = ndi.distance_transform_edt(component_local, sampling=tuple(float(v) for v in spacing_zyx))
    soma_local = _segment_soma_body(component_local, dist, spacing_zyx)
    soma_local &= component_local
    soma_shape = _soma_shape_metrics(soma_local, spacing_zyx)
    # Tips must land at the branch ends the user actually sees, and align with
    # the Voxels debug cloud. Both the isosurface render and the voxel cloud show
    # raw intensity >= the render threshold, so skeletonise exactly that "visible"
    # mask. With _TIP_VISIBILITY_MARGIN == 0 the floor equals the render
    # threshold, so the skeleton spans the full rendered branch (no inward
    # retraction) without trailing into the smoothed sub-threshold fringe that the
    # component mask alone would include.
    skeleton_source = local_mask
    if local_intensity is not None and float(tip_intensity_floor) > 0.0:
        visible = local_mask & (local_intensity >= float(tip_intensity_floor))
        if np.any(visible):
            skeleton_source = visible

    offset = np.asarray([s.start or 0 for s in bounds], dtype=np.int32)

    # Soma exclusion: carve soma out of the skeleton so process roots aren't counted
    # as tips.
    soma_exclusion = ndi.binary_dilation(soma_local, structure=_FULL_STRUCTURE, iterations=1)
    # One additional dilation catches process roots just outside the carved zone.
    soma_contact = ndi.binary_dilation(soma_exclusion, structure=_FULL_STRUCTURE, iterations=1)

    # Dense voxel skeleton first. skan may prune and measure the skeleton, but
    # tip placement comes from degree-1 voxels on this voxelized representation,
    # not from sparse graph vertices.
    skeleton = np.asarray(skeletonize(skeleton_source), dtype=bool)
    if not np.any(skeleton):
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
            soma_equivalent_diameter_um=float(soma_shape[0]),
            soma_roundness=float(soma_shape[1]),
            soma_elongation=float(soma_shape[2]),
            mean_branch_tortuosity=0.0,
        )
    branch_skeleton = np.asarray(skeleton & (~soma_exclusion) & component_local, dtype=bool)
    topology = _branch_topology(
        branch_skeleton,
        soma_contact=soma_contact,
        spacing_zyx=spacing_zyx,
        min_branch_length_um=float(min_branch_length_um),
        branch_sensitivity=float(branch_sensitivity),
        dist_local=dist,
    )

    # Degenerate: no process voxels outside the soma.
    if not np.any(topology.branch_mask):
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
            soma_equivalent_diameter_um=float(soma_shape[0]),
            soma_roundness=float(soma_shape[1]),
            soma_elongation=float(soma_shape[2]),
            mean_branch_tortuosity=0.0,
        )

    soma_centroid = _mask_centroid(soma_local)
    sholl = _sholl_metrics(topology.branch_mask, soma_centroid, spacing_zyx)

    # Tips come straight off the visible voxel cloud (the same voxels the "Voxels"
    # debug layer renders): grow a geodesic distance field outward from the soma
    # through those voxels and take its regional maxima — each process terminus is
    # one maximum. This replaces the old skeleton-endpoint -> thickness/visibility
    # gate -> fan-cluster chain, which was brittle around lamellar fans and
    # near-soma stubs. The skeleton above is still used for branch/length/Sholl
    # topology; only tip *placement* moved to the voxel cloud. The gating path is
    # retained as a fallback for environments without skimage.graph.
    voxel_endpoint_tips = _voxel_endpoint_coords(topology.branch_mask, soma_contact)
    endpoint_tips = _gate_and_cluster_tips(
        voxel_endpoint_tips,
        dist_local=dist,
        soma_local=soma_local,
        spacing_zyx=spacing_zyx,
        branch_sensitivity=branch_sensitivity,
        intensity_local=None,
        intensity_floor=0.0,
        support_mask=component_local,
    )
    if _HAS_GEODESIC_TIPS:
        geodesic_tips = _voxel_tip_coords_geodesic(
            skeleton_source,
            soma_local=soma_local,
            dist_local=dist,
            spacing_zyx=spacing_zyx,
            min_process_length_um=float(min_branch_length_um),
            branch_sensitivity=float(branch_sensitivity),
        )
        local_tips = _merge_tip_candidate_sets(
            geodesic_tips,
            endpoint_tips,
            soma_local=soma_local,
            spacing_zyx=spacing_zyx,
            branch_sensitivity=branch_sensitivity,
        )
    else:
        local_tips = endpoint_tips
        if local_tips.size == 0:
            local_tips = _gate_and_cluster_tips(
                topology.tip_coords,
                dist_local=dist,
                soma_local=soma_local,
                spacing_zyx=spacing_zyx,
                branch_sensitivity=branch_sensitivity,
                intensity_local=None,
                intensity_floor=0.0,
                support_mask=component_local,
            )
    tip_coords = local_tips + offset if local_tips.size > 0 else local_tips
    return _ComponentShapeDebug(
        soma_mask=_embed_local_mask(component_mask.shape, bounds, soma_local),
        branch_mask=_embed_local_mask(
            component_mask.shape,
            bounds,
            np.asarray(topology.branch_mask & component_local, dtype=bool),
        ),
        tip_coords=np.asarray(tip_coords, dtype=np.int32),
        branch_count=int(topology.branch_count),
        branch_point_count=int(topology.branch_point_count),
        total_process_length_um=float(topology.total_length_um),
        mean_branch_length_um=float(topology.mean_length_um),
        sholl_max_intersections=int(sholl.max_intersections),
        sholl_critical_radius_um=float(sholl.critical_radius_um),
        sholl_enclosing_radius_um=float(sholl.enclosing_radius_um),
        soma_equivalent_diameter_um=float(soma_shape[0]),
        soma_roundness=float(soma_shape[1]),
        soma_elongation=float(soma_shape[2]),
        mean_branch_tortuosity=float(topology.mean_tortuosity),
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
        soma_equivalent_diameter_um=0.0,
        soma_roundness=0.0,
        soma_elongation=0.0,
        mean_branch_tortuosity=0.0,
    )


def _segment_soma_body(
    component_mask: np.ndarray,
    dist: np.ndarray,
    spacing_zyx: np.ndarray,
) -> np.ndarray:
    if not np.any(component_mask):
        return np.zeros_like(component_mask, dtype=bool)

    masked_dist = np.where(np.asarray(component_mask, dtype=bool), dist, -np.inf)
    max_flat = int(np.argmax(masked_dist))
    center = np.unravel_index(max_flat, dist.shape)
    max_dist = float(masked_dist[center])
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

    n_paths_initial = int(skel.n_paths)
    if float(max(0.0, min_branch_length_um)) > 0.0 or dist_local is not None:
        for round_idx in range(12):
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
                logger.debug("Spur pruning: all %d paths removed as spurs", skel.n_paths)
                return _empty_topology(skeleton)
            try:
                skel = skel.prune_paths(drop.tolist())
            except Exception:
                break
    logger.debug(
        "Spur pruning: %d → %d paths (dropped %d spurs, sensitivity=%.2f)",
        n_paths_initial, int(skel.n_paths), n_paths_initial - int(skel.n_paths), branch_sensitivity,
    )

    if skel.n_paths <= 0:
        return _empty_topology(skeleton)

    coords = np.rint(np.asarray(skel.coordinates, dtype=np.float64)).astype(np.int32)
    degrees = np.asarray(skel.degrees, dtype=np.int64)
    if coords.shape[0] != degrees.shape[0]:
        return None

    branch_mask = np.zeros(skeleton.shape, dtype=bool)
    in_bounds = _coords_in_bounds(coords, skeleton.shape)
    branch_mask[tuple(coords[in_bounds].T)] = True

    junction_coords = coords[(degrees >= 3) & in_bounds]
    branch_point_count = _connected_node_count(junction_coords, skeleton.shape)
    has_junctions = bool(np.any(degrees >= 3))

    # Default: all degree-1 nodes are tip candidates.
    raw_endpoint_coords = coords[(degrees == 1) & in_bounds]
    endpoint_coords = raw_endpoint_coords
    total_length = 0.0
    branch_count = int(skel.n_paths)
    mean_length = 0.0
    all_lengths = np.zeros((0,), dtype=np.float64)
    euclid = np.zeros((0,), dtype=np.float64)

    try:
        all_lengths = np.asarray(skel.path_lengths(), dtype=np.float64)
        total_length = float(np.sum(all_lengths))
        final_summary = _skan_summarize(skel, separator="-")
        euclid = np.asarray(final_summary["euclidean-distance"].to_numpy(), dtype=np.float64)
        final_types = np.asarray(final_summary["branch-type"].to_numpy(), dtype=np.int64)
        src_ids = np.asarray(final_summary["node-id-src"].to_numpy(), dtype=np.int64)
        dst_ids = np.asarray(final_summary["node-id-dst"].to_numpy(), dtype=np.int64)

        if all_lengths.shape == final_types.shape:
            terminal_mask = np.isin(final_types, _TERMINAL_BRANCH_TYPES)
            branch_count = int(np.sum(terminal_mask))
            terminal_lengths = all_lengths[terminal_mask]
            mean_length = float(np.mean(terminal_lengths)) if terminal_lengths.size > 0 else 0.0
            if branch_count == 0:
                branch_count = int(skel.n_paths)
                mean_length = float(total_length / branch_count) if branch_count > 0 else 0.0
        else:
            branch_count = int(skel.n_paths)
            mean_length = float(total_length / branch_count) if branch_count > 0 else 0.0

        # For branched cells restrict tips to type-1 (junction-to-endpoint) path
        # endpoints.  Type-0 (endpoint-to-endpoint) isolated paths are usually
        # soma-carving disconnection fragments; their two degree-1 nodes sit in the
        # cell interior rather than at real process terminals.
        if has_junctions and len(final_types) > 0:
            n_nodes = len(degrees)
            t1_node_ids: set[int] = set()
            for _i in range(len(final_types)):
                if int(final_types[_i]) == 1:
                    _s, _d = int(src_ids[_i]), int(dst_ids[_i])
                    if 0 <= _s < n_nodes and degrees[_s] == 1:
                        t1_node_ids.add(_s)
                    if 0 <= _d < n_nodes and degrees[_d] == 1:
                        t1_node_ids.add(_d)
            if t1_node_ids:
                t1_ids = np.array(sorted(t1_node_ids), dtype=np.int64)
                t1_coords = coords[t1_ids]
                t1_ib = _coords_in_bounds(t1_coords, skeleton.shape)
                endpoint_coords = t1_coords[t1_ib]
    except Exception:
        pass  # endpoint_coords stays as raw_endpoint_coords

    tip_coords = _filter_non_contact_coords(endpoint_coords, soma_contact)
    logger.debug(
        "Skan endpoints: raw=%d type1=%d → contact_filtered=%d (junctions=%s)",
        raw_endpoint_coords.shape[0], endpoint_coords.shape[0], tip_coords.shape[0], has_junctions,
    )

    return _BranchTopology(
        tip_coords=tip_coords,
        branch_mask=branch_mask,
        branch_count=branch_count,
        branch_point_count=branch_point_count,
        total_length_um=total_length,
        mean_length_um=mean_length,
        mean_tortuosity=_mean_tortuosity_from_lengths(all_lengths, euclid),
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
        mean_tortuosity=1.0 if branch_count > 0 else 0.0,
    )


def _empty_topology(skeleton: np.ndarray) -> _BranchTopology:
    return _BranchTopology(
        tip_coords=np.empty((0, 3), dtype=np.int32),
        branch_mask=np.zeros(skeleton.shape, dtype=bool),
        branch_count=0,
        branch_point_count=0,
        total_length_um=0.0,
        mean_length_um=0.0,
        mean_tortuosity=0.0,
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

    Counts process intersections (connected components of the skeleton, 26-
    connectivity) within concentric spherical shells in physical units, so
    anisotropic spacing and full 3D topology are respected. The shell thickness is
    set to one voxel diagonal: thinner shells leave radial gaps (a process running
    straight outward would skip shells and be undercounted), while much thicker
    shells merge distinct nearby processes. ``max_intersections`` is the peak count
    over all shells and ``critical_radius_um`` the radius at which it occurs.
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

    # One voxel diagonal guarantees a radially-oriented process deposits at least
    # one skeleton voxel in every shell it crosses, so no crossing is missed.
    step = float(max(float(np.linalg.norm(spacing)), 0.5))
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


def _voxel_endpoint_coords(branch_skeleton: np.ndarray, contact_mask: np.ndarray) -> np.ndarray:
    skeleton = np.asarray(branch_skeleton, dtype=bool)
    if not np.any(skeleton):
        return np.empty((0, 3), dtype=np.int32)

    degree = ndi.convolve(skeleton.astype(np.uint8), _DEGREE_KERNEL, mode="constant", cval=0)
    endpoint_mask = skeleton & (degree <= 1)
    coords = np.argwhere(endpoint_mask).astype(np.int32, copy=False)
    return _filter_non_contact_coords(coords, contact_mask)


def _voxel_tip_coords_geodesic(
    visible_mask: np.ndarray,
    *,
    soma_local: np.ndarray,
    dist_local: np.ndarray,
    spacing_zyx: np.ndarray,
    min_process_length_um: float,
    branch_sensitivity: float,
) -> np.ndarray:
    """Process tips read directly off the visible voxel cloud — no skeleton.

    A geodesic distance field is grown from the soma outward through
    ``visible_mask`` (the same voxels the Voxels debug layer shows), so every
    process terminus becomes a regional maximum of that field. ``h_maxima`` keeps
    only maxima whose height rises at least ``min_process_length_um`` microns
    (scaled by ``1 / branch_sensitivity``) above their surroundings, so a
    protrusion must extend that far to count as a tip. Maxima from one lamellar
    "fan" that fall within the merge radius collapse to their most distal voxel.

    Because every candidate is, by construction, a voxel inside the cloud, tips
    can never land in empty space — the old support-snap / thickness / visibility
    gates are unnecessary here. Returns bbox-local (z, y, x) integer coords.
    """
    mask = np.asarray(visible_mask, dtype=bool)
    if not _HAS_GEODESIC_TIPS or not np.any(mask):
        return np.empty((0, 3), dtype=np.int32)

    # Seed from soma voxels inside the passable cloud; if the soma is entirely
    # sub-threshold, fall back to the thickest visible voxel (the cloud centre).
    seed_mask = np.asarray(soma_local, dtype=bool) & mask
    if not np.any(seed_mask):
        masked_dist = np.where(mask, np.asarray(dist_local, dtype=np.float64), -np.inf)
        seed_mask = np.zeros_like(mask)
        seed_mask[np.unravel_index(int(np.argmax(masked_dist)), mask.shape)] = True

    sampling = tuple(float(v) for v in np.asarray(spacing_zyx, dtype=np.float64))
    # cost 1 inside the cloud, +inf outside -> accumulated cost is the geodesic
    # (anisotropy-aware) distance through the structure, in microns.
    cost = np.where(mask, 1.0, np.inf).astype(np.float64)
    try:
        mcp = _MCP_Geometric(cost, sampling=sampling)
        starts = [list(map(int, c)) for c in np.argwhere(seed_mask)]
        cumulative, _ = mcp.find_costs(starts)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Geodesic tip detection failed (%s); no tips emitted.", exc)
        return np.empty((0, 3), dtype=np.int32)

    cumulative = np.asarray(cumulative, dtype=np.float64)
    reachable = np.isfinite(cumulative) & mask
    if not np.any(reachable):
        return np.empty((0, 3), dtype=np.int32)
    geo_field = np.where(reachable, cumulative, 0.0)

    sensitivity = float(np.clip(branch_sensitivity, 0.4, 2.0))
    min_spacing = float(min(sampling)) if sampling else 1.0
    h = float(
        max(float(min_process_length_um) * _TIP_PROMINENCE_LENGTH_FACTOR, min_spacing)
    ) / sensitivity
    peaks = np.asarray(_h_maxima(geo_field, h), dtype=bool) & reachable
    if not np.any(peaks):
        return np.empty((0, 3), dtype=np.int32)

    # One representative per connected maximum region: its most distal voxel.
    peak_labels, n_peaks = ndi.label(peaks, structure=_FULL_STRUCTURE)
    reps = np.empty((n_peaks, 3), dtype=np.int32)
    for i in range(1, n_peaks + 1):
        region = np.argwhere(peak_labels == i)
        vals = geo_field[region[:, 0], region[:, 1], region[:, 2]]
        reps[i - 1] = region[int(np.argmax(vals))]

    merge_radius = float(np.clip(_DEFAULT_TIP_MERGE_RADIUS_UM / sensitivity, 1.5, 12.0))
    result = _merge_tip_points(reps, geo_field, spacing_zyx, merge_radius)
    logger.debug(
        "Geodesic tips: visible=%d peaks=%d -> tips=%d (h=%.2fμm merge=%.1fμm)",
        int(np.count_nonzero(mask)), int(n_peaks), int(result.shape[0]), h, merge_radius,
    )
    return result


def _merge_tip_candidate_sets(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    soma_local: np.ndarray,
    spacing_zyx: np.ndarray,
    branch_sensitivity: float,
) -> np.ndarray:
    pieces = [
        np.asarray(arr, dtype=np.int32).reshape(-1, 3)
        for arr in (primary, secondary)
        if np.asarray(arr).size > 0
    ]
    if not pieces:
        return np.empty((0, 3), dtype=np.int32)
    pts = np.vstack(pieces)
    _, unique_idx = np.unique(pts, axis=0, return_index=True)
    unique_idx.sort()
    pts = np.asarray(pts[unique_idx], dtype=np.int32)
    if pts.shape[0] <= 1:
        return pts

    centroid = _mask_centroid(np.asarray(soma_local, dtype=bool))
    spacing = np.asarray(spacing_zyx, dtype=np.float64).reshape(1, 3)
    if centroid is None:
        ranks = np.arange(pts.shape[0], dtype=np.float64)
    else:
        soma_um = np.asarray(centroid, dtype=np.float64).reshape(1, 3) * spacing
        pts_um = pts.astype(np.float64) * spacing
        ranks = np.linalg.norm(pts_um - soma_um, axis=1)

    sensitivity = float(np.clip(branch_sensitivity, 0.4, 2.0))
    merge_radius = float(np.clip(_DEFAULT_TIP_MERGE_RADIUS_UM / sensitivity, 1.25, 10.0))
    return _merge_tip_points_by_rank(pts, ranks, spacing_zyx, merge_radius)


def _merge_tip_points(
    coords: np.ndarray,
    rank_field: np.ndarray,
    spacing_zyx: np.ndarray,
    merge_radius_um: float,
) -> np.ndarray:
    """Collapse tips within ``merge_radius_um`` to their most distal member.

    ``rank_field`` ranks candidates (higher = more distal); within each cluster
    the farthest tip is kept, matching the lamellar-fan behaviour of the legacy
    clustering path.
    """
    pts = np.asarray(coords, dtype=np.int32)
    if pts.shape[0] <= 1:
        return pts
    ranks = np.asarray(rank_field, dtype=np.float64)[pts[:, 0], pts[:, 1], pts[:, 2]]
    return _merge_tip_points_by_rank(pts, ranks, spacing_zyx, merge_radius_um)


def _merge_tip_points_by_rank(
    coords: np.ndarray,
    ranks: np.ndarray,
    spacing_zyx: np.ndarray,
    merge_radius_um: float,
) -> np.ndarray:
    pts = np.asarray(coords, dtype=np.int32)
    if pts.shape[0] <= 1:
        return pts
    spacing = np.asarray(spacing_zyx, dtype=np.float64).reshape(1, 3)
    pts_um = pts.astype(np.float64) * spacing
    ranks = np.asarray(ranks, dtype=np.float64).reshape(-1)
    if ranks.shape[0] != pts.shape[0]:
        ranks = np.zeros((pts.shape[0],), dtype=np.float64)
    order = np.argsort(ranks)[::-1]
    merge_sq = float(merge_radius_um) * float(merge_radius_um)
    kept_idx: list[int] = []
    kept_um: list[np.ndarray] = []
    for i in order.tolist():
        candidate = pts_um[i]
        if any(float(np.dot(candidate - k, candidate - k)) < merge_sq for k in kept_um):
            continue
        kept_idx.append(int(i))
        kept_um.append(candidate)
    return np.asarray(pts[kept_idx], dtype=np.int32)


def _gate_and_cluster_tips(
    tips_local: np.ndarray,
    *,
    dist_local: np.ndarray,
    soma_local: np.ndarray,
    spacing_zyx: np.ndarray,
    branch_sensitivity: float,
    intensity_local: np.ndarray | None = None,
    intensity_floor: float = 0.0,
    support_mask: np.ndarray | None = None,
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
    n_raw = pts.shape[0]
    if n_raw == 0:
        return np.empty((0, 3), dtype=np.int32)

    shape = np.asarray(dist_local.shape, dtype=np.int64).reshape(1, 3)
    pts = pts[np.all((pts >= 0) & (pts < shape), axis=1)]
    if pts.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int32)

    if support_mask is not None:
        pts = _snap_coords_to_mask(
            pts,
            support_mask,
            spacing_zyx=spacing_zyx,
            max_distance_um=_tip_support_snap_radius_um(spacing_zyx),
        )
        if pts.shape[0] == 0:
            logger.debug("Tip gate: raw=%d -> support=0 (no endpoints near component voxels)", n_raw)
            return np.empty((0, 3), dtype=np.int32)
    n_after_support = pts.shape[0]

    if intensity_local is not None and float(intensity_floor) > 0.0:
        values = np.asarray(intensity_local, dtype=np.float32)[pts[:, 0], pts[:, 1], pts[:, 2]]
        pts = pts[values >= float(intensity_floor)]
        if pts.shape[0] == 0:
            logger.debug("Tip gate: raw=%d → vis=0 (all below floor %.3f)", n_raw, intensity_floor)
            return np.empty((0, 3), dtype=np.int32)
    n_after_vis = pts.shape[0]

    soma_mask = np.asarray(soma_local, dtype=bool)
    if np.any(soma_mask):
        # Anchor to the soma BOUNDARY EDT (minimum EDT inside soma_mask) rather
        # than the soma CENTRE EDT (maximum).  The soma is detected where EDT ≥
        # body_floor (≈ 0.35 × max_dist); min(dist[soma_mask]) ≈ body_floor.
        # Scaling by _TIP_THICKNESS_BOUNDARY_FACTOR (0.80) keeps max_thickness
        # strictly below body_floor, closing the shell gap that produced hundreds
        # of false soma-surface tips.
        soma_boundary_edt = float(dist_local[soma_mask].min())
        soma_center_edt = float(dist_local[soma_mask].max())
        max_thickness = float(max(_TIP_THICKNESS_FLOOR_UM, soma_boundary_edt * _TIP_THICKNESS_BOUNDARY_FACTOR))
    else:
        soma_boundary_edt = 0.0
        soma_center_edt = 0.0
        max_thickness = _TIP_THICKNESS_FLOOR_UM
    radii = np.asarray(dist_local[pts[:, 0], pts[:, 1], pts[:, 2]], dtype=np.float64)
    pts = pts[radii <= max_thickness]
    n_after_thickness = pts.shape[0]
    if pts.shape[0] <= 1:
        logger.debug(
            "Tip gate: raw=%d support=%d vis=%d thickness=%d (soma_boundary=%.1f soma_center=%.1f max_thick=%.1f) cluster=%d",
            n_raw, n_after_support, n_after_vis, n_after_thickness,
            soma_boundary_edt, soma_center_edt, max_thickness, pts.shape[0],
        )
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

    result = np.asarray(pts[kept_idx], dtype=np.int32)
    logger.debug(
        "Tip gate: raw=%d vis=%d thickness=%d (soma_boundary=%.1f soma_center=%.1f max_thick=%.1f) cluster=%d (r=%.1fμm)",
        n_raw, n_after_vis, n_after_thickness, soma_boundary_edt, soma_center_edt, max_thickness, result.shape[0], merge_radius,
    )
    return result


def _tip_support_snap_radius_um(spacing_zyx: np.ndarray) -> float:
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    if spacing.size == 0:
        return float(_TIP_SUPPORT_SNAP_FLOOR_UM)
    return float(
        max(
            _TIP_SUPPORT_SNAP_FLOOR_UM,
            _TIP_SUPPORT_SNAP_SPACING_FACTOR * float(np.max(spacing)),
        )
    )


def _snap_coords_to_mask(
    coords: np.ndarray,
    support_mask: np.ndarray,
    *,
    spacing_zyx: np.ndarray,
    max_distance_um: float,
) -> np.ndarray:
    pts = np.asarray(coords, dtype=np.int32)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.int32)

    support = np.asarray(support_mask, dtype=bool)
    if support.ndim != 3 or not np.any(support):
        return np.empty((0, 3), dtype=np.int32)

    shape = np.asarray(support.shape, dtype=np.int64).reshape(1, 3)
    pts = pts[np.all((pts >= 0) & (pts < shape), axis=1)]
    if pts.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int32)

    distances, nearest = ndi.distance_transform_edt(
        ~support,
        sampling=tuple(float(v) for v in np.asarray(spacing_zyx, dtype=np.float64)),
        return_indices=True,
    )
    lookup = tuple(pts.T)
    keep = np.asarray(distances[lookup], dtype=np.float64) <= float(max_distance_um)
    if not np.any(keep):
        return np.empty((0, 3), dtype=np.int32)

    kept = pts[keep]
    kept_lookup = tuple(kept.T)
    snapped = np.stack(
        [np.asarray(nearest[axis][kept_lookup], dtype=np.int32) for axis in range(3)],
        axis=1,
    )
    if snapped.shape[0] <= 1:
        return np.asarray(snapped, dtype=np.int32)

    _, unique_idx = np.unique(snapped, axis=0, return_index=True)
    unique_idx.sort()
    return np.asarray(snapped[unique_idx], dtype=np.int32)


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


def _soma_shape_metrics(mask: np.ndarray, spacing_zyx: np.ndarray) -> tuple[float, float, float]:
    soma = np.asarray(mask, dtype=bool)
    count = int(np.count_nonzero(soma))
    if count <= 0:
        return 0.0, 0.0, 0.0

    spacing = np.asarray(spacing_zyx, dtype=np.float64).reshape(1, 3)
    voxel_volume = float(np.prod(spacing))
    volume_um3 = float(count) * voxel_volume
    equivalent_diameter = float((6.0 * volume_um3 / np.pi) ** (1.0 / 3.0))

    coords = np.argwhere(soma).astype(np.float64) * spacing
    if coords.shape[0] < 3:
        return equivalent_diameter, 1.0, 1.0

    centered = coords - np.mean(coords, axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eig = np.linalg.eigvalsh(np.asarray(cov, dtype=np.float64))
    axes = np.sqrt(np.maximum(eig, 0.0))
    max_axis = float(np.max(axes))
    min_axis = float(np.min(axes))
    if max_axis <= 1.0e-9:
        return equivalent_diameter, 1.0, 1.0
    roundness = float(np.clip(min_axis / max_axis, 0.0, 1.0))
    elongation = float(max_axis / max(min_axis, 1.0e-9))
    return equivalent_diameter, roundness, elongation


def _mean_tortuosity_from_lengths(lengths: np.ndarray, euclidean: np.ndarray) -> float:
    branch_dist = np.asarray(lengths, dtype=np.float64)
    chord = np.asarray(euclidean, dtype=np.float64)
    if branch_dist.shape != chord.shape or branch_dist.size == 0:
        return 0.0
    valid = chord > 1.0e-6
    if not np.any(valid):
        return 1.0 if branch_dist.size > 0 else 0.0
    tort = branch_dist[valid] / chord[valid]
    tort = tort[np.isfinite(tort) & (tort >= 1.0) & (tort < 50.0)]
    return float(np.mean(tort)) if tort.size > 0 else 1.0


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
    dist = np.asarray(dist, dtype=np.float64)
    positive = dist > 1.0e-6
    if np.any(positive):
        if known_distance_um is not None and np.isfinite(float(known_distance_um)) and float(known_distance_um) > 1.0e-6:
            # Match the numeric analysis value when available. This keeps the
            # debug line aligned with the reported nearest non-overlapping
            # distance instead of selecting an overlapping zero-length pair.
            score = np.full(dist.shape, np.inf, dtype=np.float64)
            score[positive] = np.abs(dist[positive] - float(known_distance_um))
            best = int(np.argmin(score))
        else:
            positive_idx = np.flatnonzero(positive)
            best = int(positive_idx[int(np.argmin(dist[positive_idx]))])
    else:
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
    dist_vals = vessel_distance[s[:, 0], s[:, 1], s[:, 2]]
    # Exclude voxels that sit inside the vessel (EDT = 0) — those positions are
    # at-or-inside the vessel surface and would collapse the minimum to 0 whenever
    # there is any voxel-level overlap between channels (e.g. spectral bleedthrough).
    # The biologically meaningful distance is from the nearest non-overlapping cell
    # voxel to the vessel; only return 0 when the cell is entirely within the mask.
    outside = dist_vals[dist_vals > 0.0]
    return float(np.min(outside)) if outside.size > 0 else 0.0


def _distance_at_coord(
    vessel_distance: np.ndarray | None,
    coord_zyx: np.ndarray,
    shift_zyx: np.ndarray,
    shape: tuple[int, int, int],
) -> float | None:
    if vessel_distance is None:
        return None
    coord = np.asarray(coord_zyx, dtype=np.int64).reshape(1, 3)
    shifted = coord + np.asarray(shift_zyx, dtype=np.int64).reshape(1, 3)
    bounds = np.asarray(shape, dtype=np.int64).reshape(1, 3)
    if not np.all((shifted >= 0) & (shifted < bounds)):
        return None
    z, y, x = (int(v) for v in shifted[0])
    return float(vessel_distance[z, y, x])


def _nearby_vessel_component_count(
    vessel_labels: np.ndarray | None,
    coords_zyx: np.ndarray,
    shift_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
    shape: tuple[int, int, int],
    *,
    radius_um: float = 5.0,
) -> int:
    if vessel_labels is None:
        return 0
    coords = np.asarray(coords_zyx, dtype=np.int64)
    if coords.shape[0] == 0:
        return 0
    shifted = coords + np.asarray(shift_zyx, dtype=np.int64).reshape(1, 3)
    bounds = np.asarray(shape, dtype=np.int64).reshape(1, 3)
    shifted = shifted[np.all((shifted >= 0) & (shifted < bounds), axis=1)]
    if shifted.shape[0] == 0:
        return 0

    spacing = np.asarray(spacing_zyx, dtype=np.float64).reshape(3)
    radius = float(max(0.0, radius_um))
    voxel_radius = np.ceil(radius / np.maximum(spacing, 1.0e-6)).astype(np.int64)
    labels_seen: set[int] = set()
    for coord in shifted:
        start = np.maximum(0, coord - voxel_radius)
        stop = np.minimum(np.asarray(shape, dtype=np.int64), coord + voxel_radius + 1)
        slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(start, stop, strict=True))
        local = np.asarray(vessel_labels[slices], dtype=np.int32)
        if local.size == 0:
            continue
        local_coords = np.argwhere(local > 0).astype(np.float64)
        if local_coords.shape[0] == 0:
            continue
        offset = start.astype(np.float64).reshape(1, 3)
        dist = np.linalg.norm(((local_coords + offset) - coord.reshape(1, 3)) * spacing, axis=1)
        nearby_labels = np.unique(local[tuple(local_coords[dist <= radius].astype(np.int64).T)])
        labels_seen.update(int(label) for label in nearby_labels.tolist() if int(label) > 0)
    return int(len(labels_seen))


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
