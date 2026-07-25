"""Quantitative vascular morphometry for the red (vasculature) channel.

Historically NVAP only reported coarse vessel metrics (voxel count, physical
volume, connected-component count, microglia overlap). For a neurovascular
analytics tool that is a large scientific gap: the vasculature carries most of
the spatial structure microglia organise around, yet none of its geometry was
quantified.

This module computes anisotropy-aware, physically-calibrated vascular
morphometry from a thresholded red volume. Red fluorescence is treated as a
vessel-wall signal, so the analysis keeps two masks:

* a cleaned wall mask for staining coverage and vessel-contact distances
* a reconstructed solid vessel mask for anatomical volume, centerline length,
  radius/diameter, topology, tortuosity, decussation candidates, and surface

* vascular volume fraction (fraction of imaged tissue occupied by vessels)
* total vessel (centreline) length and length density (mm vessel / mm^3 tissue)
* vessel radius / diameter distribution from the Euclidean distance transform
  sampled along the medial axis (true 3D, spacing-correct)
* topology: junction (branch-point) count + density, free-end count, segment
  count, mean segment length
* mean tortuosity (geodesic / Euclidean ratio per segment)
* surface-area estimate and vessel surface-to-volume ratio

All lengths are in micrometres, volumes in cubic micrometres, unless the field
name says otherwise. Topology uses :mod:`skan` (junction-cluster aware, 3D);
a dependency-free fallback keeps the headline density/radius metrics available
if skan cannot build a graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging

import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import skeletonize

from nvap.config.types import RenderConfig, VoxelSpacing

logger = logging.getLogger(__name__)

try:  # skan: robust junction-aware skeleton topology in anisotropic 3D.
    from skan import Skeleton as _SkanSkeleton, summarize as _skan_summarize

    _HAS_SKAN = True
except Exception:  # pragma: no cover - skan is a hard dependency, guarded anyway
    _SkanSkeleton = None  # type: ignore[assignment]
    _skan_summarize = None  # type: ignore[assignment]
    _HAS_SKAN = False

_FULL_STRUCTURE = ndi.generate_binary_structure(3, 3).astype(np.uint8, copy=False)
_WALL_MIN_COMPONENT_VOLUME_UM3 = 0.10
_SOLID_MIN_COMPONENT_VOLUME_UM3 = 1.0
# Wall/lumen reconstruction tolerance. Two micrometres closes small acquisition
# gaps in the supplied vascular-wall stacks without the component loss and
# centreline shortening observed with a 2.5 um pass.
_SOLID_CLOSE_RADIUS_UM = 2.00
_SOLID_3D_CLOSE_RADIUS_UM = 1.50
_SOLID_STACKED_CROSSING_SAFE_CLOSE_UM = 1.99
_TERMINAL_SPUR_PRUNE_LENGTH_UM = 2.0
_MIN_LUMEN_FILL_FRACTION = 0.20
_MIN_RADIUS_ESTIMATOR_RATIO = 0.75
_MAX_RADIUS_ESTIMATOR_RATIO = 1.25
_SHEET_GLOBAL_FRACTION = 0.10
_MAX_ACQUISITION_SLICE_FRACTION = 0.50
_RADIUS_RIDGE_SEARCH_UM = 1.50


@dataclass(frozen=True)
class VascularMasks:
    """Analysis masks derived from red-channel vessel-wall fluorescence."""

    wall_mask: np.ndarray
    solid_mask: np.ndarray


@dataclass(frozen=True)
class VascularAnalysisResult:
    """Aggregate vascular morphometry over the (trimmed) red channel."""

    vessel_voxel_count: int
    vessel_volume_um3: float
    tissue_volume_um3: float
    vessel_volume_fraction: float
    component_count: int

    total_length_um: float
    length_density_mm_per_mm3: float

    mean_radius_um: float
    median_radius_um: float
    max_radius_um: float
    mean_diameter_um: float

    junction_count: int
    junction_density_per_mm3: float
    endpoint_count: int
    segment_count: int
    mean_segment_length_um: float

    mean_tortuosity: float

    surface_area_um2: float
    surface_to_volume_ratio_per_um: float
    decussation_candidate_count: int
    mean_decussation_z_separation_um: float

    wall_voxel_count: int = 0
    wall_volume_um3: float = 0.0
    wall_volume_fraction: float = 0.0
    wall_component_count: int = 0
    wall_surface_area_um2: float = 0.0
    red_positive_voxel_count: int = 0
    red_positive_volume_um3: float = 0.0
    red_positive_volume_fraction: float = 0.0
    solid_vessel_voxel_count: int = 0
    solid_vessel_volume_um3: float = 0.0
    solid_vessel_volume_fraction: float = 0.0
    solid_component_count: int = 0
    solid_surface_area_um2: float = 0.0
    terminal_spur_prune_length_um: float = _TERMINAL_SPUR_PRUNE_LENGTH_UM
    mean_reconstructed_mask_radius_um: float = 0.0
    median_reconstructed_mask_radius_um: float = 0.0
    max_reconstructed_mask_radius_um: float = 0.0
    mean_reconstructed_mask_diameter_um: float = 0.0
    radius_p10_um: float = 0.0
    radius_p25_um: float = 0.0
    radius_p75_um: float = 0.0
    radius_p90_um: float = 0.0
    diameter_p10_um: float = 0.0
    diameter_p90_um: float = 0.0
    volume_length_equivalent_radius_um: float = 0.0
    volume_length_equivalent_diameter_um: float = 0.0
    radius_estimator_ratio: float = 0.0
    max_principal_slice_fraction: float = 0.0
    sheet_like_mask: bool = False
    radius_ridge_search_um: float = _RADIUS_RIDGE_SEARCH_UM
    mean_wall_distance_um: float = 0.0
    median_wall_distance_um: float = 0.0
    max_wall_distance_um: float = 0.0
    solid_to_wall_volume_ratio: float = 0.0
    reconstructed_lumen_fill_fraction: float = 0.0
    anatomical_radius_reliable: bool = False


def _spacing_zyx(spacing: VoxelSpacing | tuple[float, float, float]) -> np.ndarray:
    if isinstance(spacing, VoxelSpacing):
        arr = np.asarray((spacing.z_um, spacing.y_um, spacing.x_um), dtype=np.float64)
    else:
        arr = np.asarray(spacing, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError("spacing must provide (z_um, y_um, x_um).")
    return np.maximum(arr, 1.0e-6)


def _apply_render_trim(mask: np.ndarray, trim_first: int, trim_last: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 3 or arr.shape[0] <= 0:
        return arr
    tf = max(0, int(trim_first))
    tl = max(0, int(trim_last))
    if tf <= 0 and tl <= 0:
        return arr
    if tf + tl >= int(arr.shape[0]):
        return np.zeros_like(arr)
    out = arr.copy()
    if tf > 0:
        out[:tf] = False
    if tl > 0:
        out[-tl:] = False
    return out


def _empty_result(tissue_volume_um3: float) -> VascularAnalysisResult:
    return VascularAnalysisResult(
        vessel_voxel_count=0,
        vessel_volume_um3=0.0,
        tissue_volume_um3=float(tissue_volume_um3),
        vessel_volume_fraction=0.0,
        component_count=0,
        total_length_um=0.0,
        length_density_mm_per_mm3=0.0,
        mean_radius_um=0.0,
        median_radius_um=0.0,
        max_radius_um=0.0,
        mean_diameter_um=0.0,
        junction_count=0,
        junction_density_per_mm3=0.0,
        endpoint_count=0,
        segment_count=0,
        mean_segment_length_um=0.0,
        mean_tortuosity=0.0,
        surface_area_um2=0.0,
        surface_to_volume_ratio_per_um=0.0,
        decussation_candidate_count=0,
        mean_decussation_z_separation_um=0.0,
    )


def build_vascular_masks(
    red_volume: np.ndarray,
    *,
    threshold: float,
    spacing: VoxelSpacing | tuple[float, float, float],
    render: RenderConfig | None = None,
    reconstruct_solid: bool = True,
    closing_radius_um: float = _SOLID_CLOSE_RADIUS_UM,
    wall_min_component_volume_um3: float = _WALL_MIN_COMPONENT_VOLUME_UM3,
    solid_min_component_volume_um3: float = _SOLID_MIN_COMPONENT_VOLUME_UM3,
) -> VascularMasks:
    """Build separate red wall and reconstructed solid-vessel masks.

    ``wall_mask`` preserves the thresholded fluorescent vessel wall after small
    component cleanup. ``solid_mask`` closes small wall gaps and fills 2D lumen
    cross-sections in all three orientations before 3D reconciliation, so
    centerline and radius metrics are computed from a solid vessel volume rather
    than from the hollow wall shell.
    """
    red = np.asarray(red_volume, dtype=np.float32)
    if red.ndim != 3:
        raise ValueError("red_volume must be 3D in (z, y, x) order.")

    spacing_zyx = _spacing_zyx(spacing)
    if render is not None:
        wall = _apply_render_trim(
            red >= float(threshold),
            max(0, int(render.trim_first_slices)),
            max(0, int(render.trim_last_slices)),
        )
    else:
        wall = np.asarray(red >= float(threshold), dtype=bool)

    wall_min_voxels = _volume_um3_to_voxels(
        wall_min_component_volume_um3,
        spacing_zyx,
        minimum=1,
    )
    wall = _remove_small_components(wall, wall_min_voxels)
    solid = (
        _reconstruct_solid_vessel_mask(
            wall,
            spacing_zyx,
            closing_radius_um=closing_radius_um,
            min_component_volume_um3=solid_min_component_volume_um3,
        )
        if reconstruct_solid
        else wall.copy()
    )
    return VascularMasks(
        wall_mask=np.asarray(wall, dtype=bool),
        solid_mask=np.asarray(solid, dtype=bool),
    )


def analyze_vasculature(
    red_volume: np.ndarray,
    *,
    threshold: float,
    spacing: VoxelSpacing | tuple[float, float, float],
    render: RenderConfig | None = None,
    fill_cavities: bool = True,
    prune_terminal_spurs_um: float = _TERMINAL_SPUR_PRUNE_LENGTH_UM,
) -> VascularAnalysisResult:
    """Compute vascular morphometry from a red volume and a threshold.

    Parameters
    ----------
    red_volume:
        3D (z, y, x) intensity volume for the vasculature channel.
    threshold:
        Binarisation threshold; voxels ``>= threshold`` are vessel.
    spacing:
        Physical voxel spacing (micrometres).
    render:
        Optional render config; only ``trim_first_slices`` / ``trim_last_slices``
        are honoured so the metrics match the trimmed 3D view and microglia
        analytics.
    fill_cavities:
        When true, reconstruct a solid anatomical vessel mask from the hollow
        fluorescent wall signal before skeleton, radius, topology and surface
        measurements. Wall/staining metrics are reported separately either way.
    """
    red = np.asarray(red_volume, dtype=np.float32)
    if red.ndim != 3:
        raise ValueError("red_volume must be 3D in (z, y, x) order.")
    spacing_zyx = _spacing_zyx(spacing)
    voxel_volume = float(np.prod(spacing_zyx))

    trim_first = max(0, int(render.trim_first_slices)) if render is not None else 0
    trim_last = max(0, int(render.trim_last_slices)) if render is not None else 0
    if render is not None:
        raw_wall_mask = _apply_render_trim(red >= float(threshold), trim_first, trim_last)
    else:
        raw_wall_mask = np.asarray(red >= float(threshold), dtype=bool)
    masks = build_vascular_masks(
        red,
        threshold=float(threshold),
        spacing=spacing_zyx,
        render=render,
        reconstruct_solid=bool(fill_cavities),
    )
    wall_mask = masks.wall_mask
    mask = masks.solid_mask

    # Tissue volume is the *retained* field of view. Trimmed slices are zeroed in
    # the mask (so they never contribute vessel voxels); counting them in the
    # denominator too would bias volume fraction / length density / junction
    # density low in proportion to the trim.
    if mask.ndim == 3:
        retained_slices = max(0, int(mask.shape[0]) - trim_first - trim_last)
        tissue_voxels = int(retained_slices * mask.shape[1] * mask.shape[2])
    else:
        tissue_voxels = int(mask.size)
    tissue_volume_um3 = float(tissue_voxels) * voxel_volume
    red_positive_voxels = int(np.count_nonzero(raw_wall_mask))
    red_positive_volume_um3 = float(red_positive_voxels) * voxel_volume
    red_positive_volume_fraction = (
        float(red_positive_voxels) / float(tissue_voxels) if tissue_voxels else 0.0
    )
    wall_voxels = int(np.count_nonzero(wall_mask))
    if wall_voxels == 0:
        logger.info("Vascular analysis: no vasculature above threshold=%.4f", float(threshold))
        data = asdict(_empty_result(tissue_volume_um3))
        data.update(
            {
                "red_positive_voxel_count": red_positive_voxels,
                "red_positive_volume_um3": red_positive_volume_um3,
                "red_positive_volume_fraction": red_positive_volume_fraction,
                "terminal_spur_prune_length_um": float(prune_terminal_spurs_um),
            }
        )
        return VascularAnalysisResult(**data)

    _, component_count = ndi.label(mask, structure=_FULL_STRUCTURE)
    _, wall_component_count = ndi.label(wall_mask, structure=_FULL_STRUCTURE)
    vessel_voxels = int(np.count_nonzero(mask))
    if vessel_voxels == 0:
        logger.info(
            "Vascular analysis: all red wall components were removed during solid-mask cleanup."
        )
        base = _empty_result(tissue_volume_um3)
        wall_volume_um3 = float(wall_voxels) * voxel_volume
        wall_surface = _surface_area_um2(wall_mask, spacing_zyx)
        data = asdict(base)
        data.update(
            {
                "wall_voxel_count": wall_voxels,
                "wall_volume_um3": wall_volume_um3,
                "wall_volume_fraction": (
                    float(wall_voxels) / float(tissue_voxels) if tissue_voxels else 0.0
                ),
                "wall_component_count": int(wall_component_count),
                "wall_surface_area_um2": float(wall_surface),
                "red_positive_voxel_count": red_positive_voxels,
                "red_positive_volume_um3": red_positive_volume_um3,
                "red_positive_volume_fraction": red_positive_volume_fraction,
                "terminal_spur_prune_length_um": float(prune_terminal_spurs_um),
            }
        )
        return VascularAnalysisResult(**data)

    # Radius field: distance from each vessel voxel to the nearest background,
    # in physical units. Sampled along the medial axis it gives vessel radius.
    dist_um = ndi.distance_transform_edt(mask, sampling=tuple(spacing_zyx))
    # Lee skeletonisation preserves topology but, on hollow/irregular anisotropic
    # masks, its centreline can sit several fine XY voxels away from the true EDT
    # ridge. A fixed 3-voxel filter represented only ~0.2 um in common datasets
    # and systematically under-read vessel radius. Search a physical (micron)
    # neighbourhood instead, while keeping the radius itself spacing-correct.
    ridge_um = _physical_maximum_filter(
        dist_um,
        spacing_zyx,
        radius_um=_RADIUS_RIDGE_SEARCH_UM,
    )

    skeleton = np.asarray(skeletonize(mask), dtype=bool)
    skeleton = _prune_terminal_spurs(
        skeleton,
        spacing_zyx,
        min_length_um=float(prune_terminal_spurs_um),
    )
    skel_coords = np.argwhere(skeleton)
    if skel_coords.shape[0] == 0:
        # Blob too small / flat to skeletonise: report density + radius only.
        radii = _correct_edt_radius(ridge_um[mask], spacing_zyx)
        return _radius_only_result(
            mask=mask,
            wall_mask=wall_mask,
            raw_wall_mask=raw_wall_mask,
            radii=radii,
            spacing_zyx=spacing_zyx,
            voxel_volume=voxel_volume,
            tissue_volume_um3=tissue_volume_um3,
            tissue_voxels=tissue_voxels,
            component_count=component_count,
            wall_component_count=wall_component_count,
            prune_terminal_spurs_um=float(prune_terminal_spurs_um),
            reconstructed_solid=bool(fill_cavities),
        )

    radii = _correct_edt_radius(
        np.asarray(ridge_um[tuple(skel_coords.T)], dtype=np.float64), spacing_zyx
    )
    radii = radii[radii > 0.0]
    if radii.size == 0:
        radii = np.asarray([float(np.mean(spacing_zyx))], dtype=np.float64)

    topo = _skeleton_topology(skeleton, spacing_zyx)
    decussation_count, decussation_z_sep = _decussation_candidates(
        skeleton,
        spacing_zyx,
    )

    vessel_volume_um3 = float(vessel_voxels) * voxel_volume
    wall_volume_um3 = float(wall_voxels) * voxel_volume
    vessel_volume_fraction = (
        float(vessel_voxels) / float(tissue_voxels) if tissue_voxels else 0.0
    )
    wall_volume_fraction = (
        float(wall_voxels) / float(tissue_voxels) if tissue_voxels else 0.0
    )
    # Length density expressed as mm of centreline per mm^3 of tissue.
    tissue_volume_mm3 = tissue_volume_um3 / 1.0e9
    total_length_mm = topo.total_length_um / 1.0e3
    length_density = (
        total_length_mm / tissue_volume_mm3 if tissue_volume_mm3 > 0.0 else 0.0
    )
    junction_density = (
        topo.junction_count / tissue_volume_mm3 if tissue_volume_mm3 > 0.0 else 0.0
    )

    surface_area = _surface_area_um2(mask, spacing_zyx)
    wall_surface_area = _surface_area_um2(wall_mask, spacing_zyx)
    s2v = surface_area / vessel_volume_um3 if vessel_volume_um3 > 0.0 else 0.0
    wall_distance = _wall_distance_values(wall_mask, spacing_zyx)
    solid_to_wall_ratio = float(vessel_voxels) / float(wall_voxels) if wall_voxels else 0.0
    lumen_fill_fraction = (
        float(max(0, vessel_voxels - wall_voxels)) / float(vessel_voxels)
        if vessel_voxels
        else 0.0
    )
    mean_reconstructed_radius = float(np.mean(radii))
    median_reconstructed_radius = float(np.median(radii))
    max_reconstructed_radius = float(np.max(radii))
    radius_percentiles = np.percentile(radii, [10.0, 25.0, 75.0, 90.0])
    equivalent_radius = (
        float(np.sqrt(vessel_volume_um3 / (np.pi * topo.total_length_um)))
        if topo.total_length_um > 0.0 and vessel_volume_um3 > 0.0
        else 0.0
    )
    radius_estimator_ratio = (
        mean_reconstructed_radius / equivalent_radius
        if equivalent_radius > 0.0
        else 0.0
    )
    max_slice_fraction = _max_principal_slice_fraction(mask)
    sheet_like_mask = _is_sheet_like_mask(mask)
    anatomical_radius_reliable = _anatomical_radius_is_reliable(
        radii,
        lumen_fill_fraction=lumen_fill_fraction,
        reconstructed_solid=bool(fill_cavities),
        equivalent_radius_um=equivalent_radius,
        solid_mask=mask,
    )
    mean_radius = mean_reconstructed_radius if anatomical_radius_reliable else float("nan")
    median_radius = (
        median_reconstructed_radius if anatomical_radius_reliable else float("nan")
    )
    max_radius = max_reconstructed_radius if anatomical_radius_reliable else float("nan")

    result = VascularAnalysisResult(
        vessel_voxel_count=vessel_voxels,
        vessel_volume_um3=vessel_volume_um3,
        tissue_volume_um3=tissue_volume_um3,
        vessel_volume_fraction=vessel_volume_fraction,
        component_count=int(component_count),
        total_length_um=float(topo.total_length_um),
        length_density_mm_per_mm3=float(length_density),
        mean_radius_um=float(mean_radius),
        median_radius_um=float(median_radius),
        max_radius_um=float(max_radius),
        mean_diameter_um=float(2.0 * mean_radius),
        junction_count=int(topo.junction_count),
        junction_density_per_mm3=float(junction_density),
        endpoint_count=int(topo.endpoint_count),
        segment_count=int(topo.segment_count),
        mean_segment_length_um=float(topo.mean_segment_length_um),
        mean_tortuosity=float(topo.mean_tortuosity),
        surface_area_um2=float(surface_area),
        surface_to_volume_ratio_per_um=float(s2v),
        decussation_candidate_count=int(decussation_count),
        mean_decussation_z_separation_um=float(decussation_z_sep),
        wall_voxel_count=wall_voxels,
        wall_volume_um3=wall_volume_um3,
        wall_volume_fraction=wall_volume_fraction,
        wall_component_count=int(wall_component_count),
        wall_surface_area_um2=float(wall_surface_area),
        red_positive_voxel_count=red_positive_voxels,
        red_positive_volume_um3=red_positive_volume_um3,
        red_positive_volume_fraction=red_positive_volume_fraction,
        solid_vessel_voxel_count=vessel_voxels,
        solid_vessel_volume_um3=vessel_volume_um3,
        solid_vessel_volume_fraction=vessel_volume_fraction,
        solid_component_count=int(component_count),
        solid_surface_area_um2=float(surface_area),
        terminal_spur_prune_length_um=float(prune_terminal_spurs_um),
        mean_reconstructed_mask_radius_um=mean_reconstructed_radius,
        median_reconstructed_mask_radius_um=median_reconstructed_radius,
        max_reconstructed_mask_radius_um=max_reconstructed_radius,
        mean_reconstructed_mask_diameter_um=float(2.0 * mean_reconstructed_radius),
        radius_p10_um=float(radius_percentiles[0]),
        radius_p25_um=float(radius_percentiles[1]),
        radius_p75_um=float(radius_percentiles[2]),
        radius_p90_um=float(radius_percentiles[3]),
        diameter_p10_um=float(2.0 * radius_percentiles[0]),
        diameter_p90_um=float(2.0 * radius_percentiles[3]),
        volume_length_equivalent_radius_um=equivalent_radius,
        volume_length_equivalent_diameter_um=float(2.0 * equivalent_radius),
        radius_estimator_ratio=float(radius_estimator_ratio),
        max_principal_slice_fraction=float(max_slice_fraction),
        sheet_like_mask=bool(sheet_like_mask),
        radius_ridge_search_um=float(_RADIUS_RIDGE_SEARCH_UM),
        mean_wall_distance_um=float(np.mean(wall_distance)) if wall_distance.size else 0.0,
        median_wall_distance_um=(
            float(np.median(wall_distance)) if wall_distance.size else 0.0
        ),
        max_wall_distance_um=float(np.max(wall_distance)) if wall_distance.size else 0.0,
        solid_to_wall_volume_ratio=solid_to_wall_ratio,
        reconstructed_lumen_fill_fraction=lumen_fill_fraction,
        anatomical_radius_reliable=bool(anatomical_radius_reliable),
    )
    logger.info(
        "Vascular analysis: vol_frac=%.4f length=%.1fum density=%.2f mm/mm3 "
        "mean_diam=%.2fum junctions=%d segments=%d tortuosity=%.3f",
        result.vessel_volume_fraction,
        result.total_length_um,
        result.length_density_mm_per_mm3,
        result.mean_diameter_um,
        result.junction_count,
        result.segment_count,
        result.mean_tortuosity,
    )
    return result


@dataclass(frozen=True)
class _SkeletonTopology:
    total_length_um: float
    segment_count: int
    mean_segment_length_um: float
    junction_count: int
    endpoint_count: int
    mean_tortuosity: float


def _skeleton_topology(skeleton: np.ndarray, spacing_zyx: np.ndarray) -> _SkeletonTopology:
    if _HAS_SKAN:
        result = _skeleton_topology_skan(skeleton, spacing_zyx)
        if result is not None:
            return result
    return _skeleton_topology_fallback(skeleton, spacing_zyx)


def _skeleton_topology_skan(
    skeleton: np.ndarray, spacing_zyx: np.ndarray
) -> _SkeletonTopology | None:
    sampling = tuple(float(v) for v in spacing_zyx)
    try:
        skel = _SkanSkeleton(skeleton, spacing=sampling)
        if skel.n_paths <= 0:
            return None
        summary = _skan_summarize(skel, separator="-")
        branch_dist = np.asarray(summary["branch-distance"].to_numpy(), dtype=np.float64)
        euclid = np.asarray(summary["euclidean-distance"].to_numpy(), dtype=np.float64)
        degrees = np.asarray(skel.degrees, dtype=np.int64)
    except Exception:
        return None

    total_length = float(np.sum(branch_dist))
    segment_count = int(branch_dist.shape[0])
    mean_segment = float(total_length / segment_count) if segment_count > 0 else 0.0

    # Tortuosity per segment: geodesic / straight-line. Only meaningful for
    # segments with a finite chord (skip loops where euclidean ~ 0).
    valid = euclid > 1.0e-6
    if np.any(valid):
        tort = branch_dist[valid] / euclid[valid]
        # Clamp pathological ratios from sub-voxel chords.
        tort = tort[np.isfinite(tort) & (tort >= 1.0) & (tort < 50.0)]
        mean_tortuosity = float(np.mean(tort)) if tort.size else 1.0
    else:
        mean_tortuosity = 1.0

    junction_count = _cluster_count(skel.coordinates, degrees >= 3, skeleton.shape)
    endpoint_count = int(np.count_nonzero(degrees == 1))

    return _SkeletonTopology(
        total_length_um=total_length,
        segment_count=segment_count,
        mean_segment_length_um=mean_segment,
        junction_count=junction_count,
        endpoint_count=endpoint_count,
        mean_tortuosity=mean_tortuosity,
    )


def _cluster_count(
    coordinates: np.ndarray, selector: np.ndarray, shape: tuple[int, int, int]
) -> int:
    """Count connected clusters of selected skeleton nodes.

    Adjacent junction voxels belong to one anatomical branch point, so we label
    them with full 3D connectivity rather than counting raw voxels.
    """
    coords = np.rint(np.asarray(coordinates, dtype=np.float64)).astype(np.int64)
    sel = np.asarray(selector, dtype=bool)
    if coords.shape[0] != sel.shape[0] or not np.any(sel):
        return 0
    pts = coords[sel]
    bounds = np.asarray(shape, dtype=np.int64).reshape(1, 3)
    inb = np.all((pts >= 0) & (pts < bounds), axis=1)
    pts = pts[inb]
    if pts.shape[0] == 0:
        return 0
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(pts.T)] = True
    _, count = ndi.label(mask, structure=_FULL_STRUCTURE)
    return int(count)


def _skeleton_topology_fallback(
    skeleton: np.ndarray, spacing_zyx: np.ndarray
) -> _SkeletonTopology:
    skel = np.asarray(skeleton, dtype=bool)
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    degree = ndi.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    endpoint_count = int(np.count_nonzero(skel & (degree == 1)))
    junction_count = _cluster_count(
        np.argwhere(skel & (degree >= 3)).astype(np.float64),
        np.ones(int(np.count_nonzero(skel & (degree >= 3))), dtype=bool),
        skeleton.shape,
    )
    # Length approximation: each skeleton voxel contributes the mean spacing.
    voxel_count = int(np.count_nonzero(skel))
    total_length = float(voxel_count) * float(np.mean(spacing_zyx))
    _, segment_count = ndi.label(skel & (degree != 2), structure=_FULL_STRUCTURE)
    segment_count = max(1, int(segment_count))
    return _SkeletonTopology(
        total_length_um=total_length,
        segment_count=segment_count,
        mean_segment_length_um=total_length / segment_count,
        junction_count=junction_count,
        endpoint_count=endpoint_count,
        mean_tortuosity=1.0,
    )


def _prune_terminal_spurs(
    skeleton: np.ndarray,
    spacing_zyx: np.ndarray,
    *,
    min_length_um: float,
) -> np.ndarray:
    """Remove short endpoint-to-junction paths from a skeleton."""
    skel = np.asarray(skeleton, dtype=bool)
    min_len = float(min_length_um)
    if not np.any(skel) or min_len <= 0.0 or not _HAS_SKAN:
        return skel
    sampling = tuple(float(v) for v in spacing_zyx)
    current = skel
    for _ in range(8):
        try:
            skan_skel = _SkanSkeleton(current, spacing=sampling)
            if skan_skel.n_paths <= 0:
                return current
            summary = _skan_summarize(skan_skel, separator="-")
            branch_dist = np.asarray(summary["branch-distance"].to_numpy(), dtype=np.float64)
            branch_type = np.asarray(summary["branch-type"].to_numpy(), dtype=np.int64)
            prune_ids = np.flatnonzero((branch_type == 1) & (branch_dist < min_len)).tolist()
            if not prune_ids:
                return current
            current = np.asarray(skan_skel.prune_paths(prune_ids).skeleton_image, dtype=bool)
        except Exception:
            return current
    return current


def _correct_edt_radius(radii: np.ndarray, spacing_zyx: np.ndarray) -> np.ndarray:
    """Correct the systematic half-voxel underestimate of the Euclidean EDT.

    The EDT measures voxel-centre to nearest background *voxel centre*, so the true
    vessel surface sits ~half a voxel beyond the last foreground centre. Add half
    the mean in-plane spacing so reported radii/diameters are physically calibrated
    rather than biased low by the discretisation.
    """
    arr = np.asarray(radii, dtype=np.float64)
    if arr.size == 0:
        return arr
    # In-plane (y, x) spacing governs the boundary offset for the typical vessel
    # cross-section; averaging the two keeps it robust to mild anisotropy.
    half_voxel = 0.5 * float(np.mean(np.asarray(spacing_zyx, dtype=np.float64)[1:]))
    return arr + half_voxel


def _physical_maximum_filter(
    values: np.ndarray,
    spacing_zyx: np.ndarray,
    *,
    radius_um: float,
) -> np.ndarray:
    """Return a local maximum field using a physically sized 3D footprint."""
    radius = float(max(0.0, radius_um))
    if radius <= 0.0:
        return np.asarray(values)
    footprint = _ellipsoid_structure(radius, spacing_zyx)
    return ndi.maximum_filter(values, footprint=footprint, mode="nearest")


def _wall_distance_values(wall_mask: np.ndarray, spacing_zyx: np.ndarray) -> np.ndarray:
    """Return diagnostic EDT distances inside the red-positive wall mask."""
    wall = np.asarray(wall_mask, dtype=bool)
    if not np.any(wall):
        return np.asarray([], dtype=np.float64)
    dist_um = ndi.distance_transform_edt(wall, sampling=tuple(spacing_zyx))
    ridge_um = ndi.maximum_filter(dist_um, size=3, mode="nearest")
    skeleton = np.asarray(skeletonize(wall), dtype=bool)
    values = ridge_um[skeleton] if np.any(skeleton) else ridge_um[wall]
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values) & (values > 0.0)]


def _max_principal_slice_fraction(mask: np.ndarray) -> float:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3 or binary.size == 0:
        return 0.0
    maximum = 0.0
    for axis in (0, 1, 2):
        other_axes = tuple(index for index in (0, 1, 2) if index != axis)
        slice_area = float(np.prod([binary.shape[index] for index in other_axes]))
        if slice_area <= 0.0:
            continue
        occupancy = np.sum(binary, axis=other_axes, dtype=np.int64) / slice_area
        maximum = max(maximum, float(np.max(occupancy, initial=0.0)))
    return float(maximum)


def _is_sheet_like_mask(mask: np.ndarray) -> bool:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3 or binary.size == 0:
        return False
    global_fraction = float(np.mean(binary))
    xy_fraction = np.mean(binary, axis=(1, 2))
    max_xy_fraction = float(np.max(xy_fraction, initial=0.0))
    return bool(
        global_fraction >= _SHEET_GLOBAL_FRACTION
        and max_xy_fraction > _MAX_ACQUISITION_SLICE_FRACTION
    )


def _anatomical_radius_is_reliable(
    reconstructed_radii_um: np.ndarray,
    *,
    lumen_fill_fraction: float,
    reconstructed_solid: bool,
    equivalent_radius_um: float = 0.0,
    solid_mask: np.ndarray | None = None,
) -> bool:
    """Gate radius reporting using fill, estimator agreement, and anti-sheet QC."""
    if not bool(reconstructed_solid):
        return False
    radii = np.asarray(reconstructed_radii_um, dtype=np.float64)
    radii = radii[np.isfinite(radii) & (radii > 0.0)]
    if radii.size == 0:
        return False

    if solid_mask is not None:
        binary = np.asarray(solid_mask, dtype=bool)
        if _is_sheet_like_mask(binary):
            return False

    fill_fraction = float(max(0.0, lumen_fill_fraction))
    if fill_fraction >= _MIN_LUMEN_FILL_FRACTION or fill_fraction < 0.02:
        return True

    equivalent = float(equivalent_radius_um)
    if not np.isfinite(equivalent) or equivalent <= 0.0:
        return False
    ratio = float(np.mean(radii)) / equivalent
    return _MIN_RADIUS_ESTIMATOR_RATIO <= ratio <= _MAX_RADIUS_ESTIMATOR_RATIO


def _radius_only_result(
    *,
    mask: np.ndarray,
    wall_mask: np.ndarray,
    raw_wall_mask: np.ndarray,
    radii: np.ndarray,
    spacing_zyx: np.ndarray,
    voxel_volume: float,
    tissue_volume_um3: float,
    tissue_voxels: int,
    component_count: int,
    wall_component_count: int,
    prune_terminal_spurs_um: float,
    reconstructed_solid: bool,
) -> VascularAnalysisResult:
    radii = np.asarray(radii, dtype=np.float64)
    radii = radii[radii > 0.0]
    if radii.size == 0:
        radii = np.asarray([float(np.mean(spacing_zyx))], dtype=np.float64)
    vessel_voxels = int(np.count_nonzero(mask))
    wall_voxels = int(np.count_nonzero(wall_mask))
    red_positive_voxels = int(np.count_nonzero(raw_wall_mask))
    vessel_volume_um3 = float(vessel_voxels) * voxel_volume
    wall_volume_um3 = float(wall_voxels) * voxel_volume
    red_positive_volume_um3 = float(red_positive_voxels) * voxel_volume
    surface_area = _surface_area_um2(mask, spacing_zyx)
    wall_surface_area = _surface_area_um2(wall_mask, spacing_zyx)
    s2v = surface_area / vessel_volume_um3 if vessel_volume_um3 > 0.0 else 0.0
    wall_distance = _wall_distance_values(wall_mask, spacing_zyx)
    solid_to_wall_ratio = float(vessel_voxels) / float(wall_voxels) if wall_voxels else 0.0
    lumen_fill_fraction = (
        float(max(0, vessel_voxels - wall_voxels)) / float(vessel_voxels)
        if vessel_voxels
        else 0.0
    )
    anatomical_radius_reliable = _anatomical_radius_is_reliable(
        radii,
        lumen_fill_fraction=lumen_fill_fraction,
        reconstructed_solid=bool(reconstructed_solid),
        solid_mask=mask,
    )
    mean_reconstructed_radius = float(np.mean(radii))
    median_reconstructed_radius = float(np.median(radii))
    max_reconstructed_radius = float(np.max(radii))
    radius_percentiles = np.percentile(radii, [10.0, 25.0, 75.0, 90.0])
    mean_radius = mean_reconstructed_radius if anatomical_radius_reliable else float("nan")
    median_radius = (
        median_reconstructed_radius if anatomical_radius_reliable else float("nan")
    )
    max_radius = max_reconstructed_radius if anatomical_radius_reliable else float("nan")
    return VascularAnalysisResult(
        vessel_voxel_count=vessel_voxels,
        vessel_volume_um3=vessel_volume_um3,
        tissue_volume_um3=tissue_volume_um3,
        vessel_volume_fraction=(
            float(vessel_voxels) / float(tissue_voxels) if tissue_voxels else 0.0
        ),
        component_count=int(component_count),
        total_length_um=0.0,
        length_density_mm_per_mm3=0.0,
        mean_radius_um=float(mean_radius),
        median_radius_um=float(median_radius),
        max_radius_um=float(max_radius),
        mean_diameter_um=float(2.0 * mean_radius),
        junction_count=0,
        junction_density_per_mm3=0.0,
        endpoint_count=0,
        segment_count=0,
        mean_segment_length_um=0.0,
        mean_tortuosity=0.0,
        surface_area_um2=float(surface_area),
        surface_to_volume_ratio_per_um=float(s2v),
        decussation_candidate_count=0,
        mean_decussation_z_separation_um=0.0,
        wall_voxel_count=wall_voxels,
        wall_volume_um3=wall_volume_um3,
        wall_volume_fraction=(
            float(wall_voxels) / float(tissue_voxels) if tissue_voxels else 0.0
        ),
        wall_component_count=int(wall_component_count),
        wall_surface_area_um2=float(wall_surface_area),
        red_positive_voxel_count=red_positive_voxels,
        red_positive_volume_um3=red_positive_volume_um3,
        red_positive_volume_fraction=(
            float(red_positive_voxels) / float(tissue_voxels) if tissue_voxels else 0.0
        ),
        solid_vessel_voxel_count=vessel_voxels,
        solid_vessel_volume_um3=vessel_volume_um3,
        solid_vessel_volume_fraction=(
            float(vessel_voxels) / float(tissue_voxels) if tissue_voxels else 0.0
        ),
        solid_component_count=int(component_count),
        solid_surface_area_um2=float(surface_area),
        terminal_spur_prune_length_um=float(prune_terminal_spurs_um),
        mean_reconstructed_mask_radius_um=mean_reconstructed_radius,
        median_reconstructed_mask_radius_um=median_reconstructed_radius,
        max_reconstructed_mask_radius_um=max_reconstructed_radius,
        mean_reconstructed_mask_diameter_um=float(2.0 * mean_reconstructed_radius),
        radius_p10_um=float(radius_percentiles[0]),
        radius_p25_um=float(radius_percentiles[1]),
        radius_p75_um=float(radius_percentiles[2]),
        radius_p90_um=float(radius_percentiles[3]),
        diameter_p10_um=float(2.0 * radius_percentiles[0]),
        diameter_p90_um=float(2.0 * radius_percentiles[3]),
        radius_ridge_search_um=float(_RADIUS_RIDGE_SEARCH_UM),
        mean_wall_distance_um=float(np.mean(wall_distance)) if wall_distance.size else 0.0,
        median_wall_distance_um=(
            float(np.median(wall_distance)) if wall_distance.size else 0.0
        ),
        max_wall_distance_um=float(np.max(wall_distance)) if wall_distance.size else 0.0,
        solid_to_wall_volume_ratio=solid_to_wall_ratio,
        reconstructed_lumen_fill_fraction=lumen_fill_fraction,
        anatomical_radius_reliable=bool(anatomical_radius_reliable),
    )


def _volume_um3_to_voxels(volume_um3: float, spacing_zyx: np.ndarray, *, minimum: int) -> int:
    voxel_volume = float(np.prod(np.asarray(spacing_zyx, dtype=np.float64)))
    if voxel_volume <= 0.0:
        return int(max(1, minimum))
    return int(max(int(minimum), int(np.ceil(float(volume_um3) / voxel_volume))))


def _remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    floor = int(max(1, min_voxels))
    if floor <= 1 or not np.any(binary):
        return binary.copy()
    labels, count = ndi.label(binary, structure=_FULL_STRUCTURE)
    if count <= 0:
        return binary.copy()
    sizes = np.bincount(labels.ravel())
    keep = np.flatnonzero(sizes >= floor)
    keep = keep[keep != 0]
    return np.isin(labels, keep) if keep.size > 0 else np.zeros_like(binary)


def _ellipsoid_structure(radius_um: float, spacing_zyx: np.ndarray) -> np.ndarray:
    radius = float(radius_um)
    if radius <= 0.0:
        return np.ones((1, 1, 1), dtype=bool)
    spacing_arr = np.asarray(spacing_zyx, dtype=np.float64)
    extents = np.maximum(1, np.ceil(radius / spacing_arr).astype(np.int64))
    zz, yy, xx = np.indices(tuple((2 * extents + 1).tolist()))
    center = extents.reshape(3, 1, 1, 1)
    coords = np.asarray([zz, yy, xx], dtype=np.float64)
    scaled = (coords - center) * spacing_arr.reshape(3, 1, 1, 1)
    dist = np.sqrt(np.sum(scaled**2, axis=0))
    return np.asarray(dist <= radius + 1.0e-9, dtype=bool)


def _ellipse_structure_for_slice(
    *,
    sliced_axis: int,
    radius_um: float,
    spacing_zyx: np.ndarray,
) -> np.ndarray:
    """2D elliptical structuring element for a slice plane.

    The volume is stored as (z, y, x), so a slice perpendicular to z is an
    anisotropic y/x plane, while slices perpendicular to y or x are z/x and z/y
    planes. Building the element from physical spacing lets one lumen-sealing
    radius behave consistently across all orientations.
    """
    radius = float(radius_um)
    if radius <= 0.0:
        return np.ones((1, 1), dtype=bool)
    axes = [axis for axis in (0, 1, 2) if axis != int(sliced_axis)]
    spacing_arr = np.asarray(spacing_zyx, dtype=np.float64)[axes]
    extents = np.maximum(1, np.ceil(radius / spacing_arr).astype(np.int64))
    aa, bb = np.indices(tuple((2 * extents + 1).tolist()))
    center = extents.reshape(2, 1, 1)
    coords = np.asarray([aa, bb], dtype=np.float64)
    scaled = (coords - center) * spacing_arr.reshape(2, 1, 1)
    dist = np.sqrt(np.sum(scaled**2, axis=0))
    return np.asarray(dist <= radius + 1.0e-9, dtype=bool)


def _fill_lumen_slicewise_with_closing(
    mask: np.ndarray,
    spacing_zyx: np.ndarray,
    *,
    closing_radius_um: float,
) -> np.ndarray:
    """Seal broken wall cross-sections and fill the enclosed lumen.

    Hollow vascular stains usually appear as rings or arcs in individual
    cross-sections. A small 3D closing can miss those lumens because gaps and
    field-of-view cuts keep the background connected in 3D. Filling after
    physically calibrated 2D closing on all principal planes is deliberately
    more aggressive about reconstructing tube interiors while still avoiding a
    blanket dilation of the whole volume.
    """
    wall = np.asarray(mask, dtype=bool)
    if not np.any(wall):
        return wall.copy()
    solid = wall.copy()
    for axis in (0, 1, 2):
        # In planes containing z, exclude the exact 2 um axial offset. This tiny
        # cap keeps vessels separated by 4 um in z from closing into one another
        # while retaining essentially the full tolerance for anisotropic data.
        slice_close_radius = (
            float(closing_radius_um)
            if axis == 0
            else min(
                float(closing_radius_um),
                _SOLID_STACKED_CROSSING_SAFE_CLOSE_UM,
            )
        )
        structure = _ellipse_structure_for_slice(
            sliced_axis=axis,
            radius_um=slice_close_radius,
            spacing_zyx=spacing_zyx,
        )
        moved = np.moveaxis(wall, axis, 0)
        filled = np.empty_like(moved, dtype=bool)
        for idx in range(moved.shape[0]):
            closed = np.asarray(
                ndi.binary_closing(moved[idx], structure=structure),
                dtype=bool,
            )
            closed |= moved[idx]
            filled[idx] = ndi.binary_fill_holes(closed)
        solid |= np.moveaxis(filled, 0, axis)
    return solid


def _fill_holes_slicewise_all_axes(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    filled = binary.copy()
    for axis in (0, 1, 2):
        moved = np.moveaxis(binary, axis, 0)
        axis_filled = np.empty_like(moved, dtype=bool)
        for idx in range(moved.shape[0]):
            axis_filled[idx] = ndi.binary_fill_holes(moved[idx])
        filled |= np.moveaxis(axis_filled, 0, axis)
    return filled


def _fill_internal_cavities(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed background cavities without filling exterior background.

    A reconstructed vessel mask should be solid. Background connected to the
    field-of-view boundary remains exterior and is never filled.
    """
    binary = np.asarray(mask, dtype=bool)
    background = ~binary
    labels, count = ndi.label(background, structure=_FULL_STRUCTURE)
    if count == 0:
        return binary
    # Background component touching the border is the true exterior; never fill.
    border_ids = set(np.unique(np.concatenate([
        labels[0].ravel(), labels[-1].ravel(),
        labels[:, 0].ravel(), labels[:, -1].ravel(),
        labels[:, :, 0].ravel(), labels[:, :, -1].ravel(),
    ])).tolist())
    fill = binary.copy()
    for comp_id in range(1, count + 1):
        if comp_id in border_ids:
            continue
        fill[labels == comp_id] = True
    return fill


def _reconstruct_solid_vessel_mask(
    wall_mask: np.ndarray,
    spacing_zyx: np.ndarray,
    *,
    closing_radius_um: float,
    min_component_volume_um3: float,
) -> np.ndarray:
    wall = np.asarray(wall_mask, dtype=bool)
    if not np.any(wall):
        return wall.copy()
    solid = _fill_lumen_slicewise_with_closing(
        wall,
        spacing_zyx,
        closing_radius_um=float(closing_radius_um),
    )
    # Principal-plane filling handles most tubes cheaply, but an oblique vessel
    # can have a small wall opening in every XY/XZ/YZ section even when its 3D
    # shell is nearly closed. Seal those orientation-dependent gaps in 3D and
    # fill only cavities that are disconnected from the field-of-view exterior.
    # Unioning this result with the slice reconstruction preserves open ends and
    # avoids eroding the original thresholded wall.
    # Keep the unrestricted estimate in the cross-sectional planes. In 3D, cap
    # it below the separation of plausible stacked vessels so an over/under
    # crossing is not fused into one object and lost as a decussation.
    close_radius = min(
        float(max(0.0, closing_radius_um)),
        _SOLID_3D_CLOSE_RADIUS_UM,
    )
    if close_radius > 0.0:
        closed_3d = np.asarray(
            ndi.binary_closing(
                wall,
                structure=_ellipsoid_structure(close_radius, spacing_zyx),
            ),
            dtype=bool,
        )
        closed_3d |= wall
        solid |= _fill_internal_cavities(closed_3d)
    solid = _fill_internal_cavities(solid)
    min_voxels = _volume_um3_to_voxels(
        min_component_volume_um3,
        spacing_zyx,
        minimum=1,
    )
    return _remove_small_components(solid, min_voxels)


def _surface_area_um2(mask: np.ndarray, spacing_zyx: np.ndarray) -> float:
    """Estimate vessel surface area from exposed voxel faces.

    Each internal vessel↔background face contributes the area of that face (the
    product of the two in-plane spacings). Faces lying on the volume boundary are
    *not* counted: a vessel clipped by the field of view is cut by the imaging
    limit, not bounded by a real tissue surface, so including those faces would
    inflate the surface area and the surface-to-volume ratio.
    """
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        return 0.0
    sz, sy, sx = (float(v) for v in spacing_zyx)
    face_area = {0: sy * sx, 1: sz * sx, 2: sz * sy}
    total = 0.0
    for axis in (0, 1, 2):
        # Count internal vessel/background transitions along this axis (both
        # directions). Boundary faces are intentionally excluded.
        diff = np.diff(binary.astype(np.int8), axis=axis)
        exposed = int(np.count_nonzero(diff == 1)) + int(np.count_nonzero(diff == -1))
        total += exposed * face_area[axis]
    return float(total)


def _decussation_candidates(skeleton: np.ndarray, spacing_zyx: np.ndarray) -> tuple[int, float]:
    """Detect stacked vessel centerline columns that may indicate crossings.

    A single binary vessel channel cannot identify which anatomical segment
    passes over which with certainty. This conservative candidate metric counts
    (y, x) centerline columns containing two or more separated z-runs, which is
    the measurable signature of over/under vessel crossings in the current data.
    """
    skel = np.asarray(skeleton, dtype=bool)
    coords = np.argwhere(skel)
    if coords.shape[0] == 0 or skel.shape[0] <= 1:
        return 0, 0.0

    z_spacing = float(np.asarray(spacing_zyx, dtype=np.float64)[0])
    by_yx: dict[tuple[int, int], list[int]] = {}
    for z, y, x in coords.tolist():
        by_yx.setdefault((int(y), int(x)), []).append(int(z))

    candidate_sep: dict[tuple[int, int], float] = {}
    for yx, z_values in by_yx.items():
        unique_z = sorted(set(z_values))
        if len(unique_z) < 2:
            continue
        groups: list[list[int]] = [[unique_z[0]]]
        for z in unique_z[1:]:
            if z <= groups[-1][-1] + 1:
                groups[-1].append(z)
            else:
                groups.append([z])
        if len(groups) < 2:
            continue
        centers = [0.5 * (g[0] + g[-1]) for g in groups]
        gaps = [abs(b - a) * z_spacing for a, b in zip(centers, centers[1:], strict=False)]
        candidate_sep[yx] = float(max(gaps))

    if not candidate_sep:
        return 0, 0.0
    candidate_mask = np.zeros(skel.shape[1:], dtype=bool)
    for y, x in candidate_sep:
        candidate_mask[int(y), int(x)] = True
    labels, count = ndi.label(candidate_mask, structure=np.ones((3, 3), dtype=np.uint8))
    separations: list[float] = []
    for label_id in range(1, int(count) + 1):
        coords = np.argwhere(labels == label_id)
        values = [candidate_sep[(int(y), int(x))] for y, x in coords.tolist()]
        if values:
            separations.append(float(np.max(values)))
    if not separations:
        return 0, 0.0
    return int(len(separations)), float(np.mean(separations))


def vascular_analysis_to_csv_rows(
    result: VascularAnalysisResult,
) -> list[dict[str, float | int | str]]:
    """Flatten the result into a single-row (metric, value) long-format table.

    Long format keeps the vascular CSV self-describing and easy to merge with
    the existing metrics export, regardless of how many fields evolve later.
    """
    data = asdict(result)
    rows: list[dict[str, float | int | str]] = [
        {"metric": key, "value": value} for key, value in data.items()
    ]
    rows.append(
        {
            "metric": "vascular_wall_mask",
            "value": "clean thresholded red fluorescence; used for wall/staining/contact metrics",
        }
    )
    rows.append(
        {
            "metric": "vascular_solid_mask",
            "value": "reconstructed filled vessel volume; anatomical morphometry only when reliability flag is true",
        }
    )
    if not bool(result.anatomical_radius_reliable):
        rows.append(
            {
                "metric": "vascular_radius_interpretation",
                "value": (
                    "mean_radius_um/mean_diameter_um withheld; reconstructed mask "
                    "failed lumen-fill, independent-estimator, or anti-sheet QC"
                ),
            }
        )
    return rows
