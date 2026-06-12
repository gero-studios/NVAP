"""Quantitative vascular morphometry for the red (vasculature) channel.

Historically NVAP only reported coarse vessel metrics (voxel count, physical
volume, connected-component count, microglia overlap). For a neurovascular
analytics tool that is a large scientific gap: the vasculature carries most of
the spatial structure microglia organise around, yet none of its geometry was
quantified.

This module computes anisotropy-aware, physically-calibrated vascular
morphometry from a thresholded red volume:

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
# Interior cavities below this many voxels are filled before skeletonising so a
# noisy lumen does not seed spurious skeleton loops/branch points.
_FILL_CAVITY_VOXELS = 64


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
    )


def analyze_vasculature(
    red_volume: np.ndarray,
    *,
    threshold: float,
    spacing: VoxelSpacing | tuple[float, float, float],
    render: RenderConfig | None = None,
    fill_cavities: bool = True,
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
    """
    red = np.asarray(red_volume, dtype=np.float32)
    if red.ndim != 3:
        raise ValueError("red_volume must be 3D in (z, y, x) order.")
    spacing_zyx = _spacing_zyx(spacing)
    voxel_volume = float(np.prod(spacing_zyx))

    if render is not None:
        mask = _apply_render_trim(
            red >= float(threshold),
            int(render.trim_first_slices),
            int(render.trim_last_slices),
        )
    else:
        mask = np.asarray(red >= float(threshold), dtype=bool)

    tissue_volume_um3 = float(mask.size) * voxel_volume
    vessel_voxels = int(np.count_nonzero(mask))
    if vessel_voxels == 0:
        logger.info("Vascular analysis: no vasculature above threshold=%.4f", float(threshold))
        return _empty_result(tissue_volume_um3)

    if fill_cavities:
        mask = _fill_small_cavities(mask, _FILL_CAVITY_VOXELS)

    _, component_count = ndi.label(mask, structure=_FULL_STRUCTURE)

    # Radius field: distance from each vessel voxel to the nearest background,
    # in physical units. Sampled along the medial axis it gives vessel radius.
    dist_um = ndi.distance_transform_edt(mask, sampling=tuple(spacing_zyx))

    skeleton = np.asarray(skeletonize(mask), dtype=bool)
    skel_coords = np.argwhere(skeleton)
    if skel_coords.shape[0] == 0:
        # Blob too small / flat to skeletonise: report density + radius only.
        radii = dist_um[mask]
        return _radius_only_result(
            mask=mask,
            radii=radii,
            spacing_zyx=spacing_zyx,
            voxel_volume=voxel_volume,
            tissue_volume_um3=tissue_volume_um3,
            component_count=component_count,
        )

    radii = np.asarray(dist_um[tuple(skel_coords.T)], dtype=np.float64)
    radii = radii[radii > 0.0]
    if radii.size == 0:
        radii = np.asarray([float(np.mean(spacing_zyx))], dtype=np.float64)

    topo = _skeleton_topology(skeleton, spacing_zyx)

    vessel_volume_um3 = float(vessel_voxels) * voxel_volume
    vessel_volume_fraction = (
        float(vessel_voxels) / float(mask.size) if mask.size else 0.0
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
    s2v = surface_area / vessel_volume_um3 if vessel_volume_um3 > 0.0 else 0.0

    result = VascularAnalysisResult(
        vessel_voxel_count=vessel_voxels,
        vessel_volume_um3=vessel_volume_um3,
        tissue_volume_um3=tissue_volume_um3,
        vessel_volume_fraction=vessel_volume_fraction,
        component_count=int(component_count),
        total_length_um=float(topo.total_length_um),
        length_density_mm_per_mm3=float(length_density),
        mean_radius_um=float(np.mean(radii)),
        median_radius_um=float(np.median(radii)),
        max_radius_um=float(np.max(radii)),
        mean_diameter_um=float(2.0 * np.mean(radii)),
        junction_count=int(topo.junction_count),
        junction_density_per_mm3=float(junction_density),
        endpoint_count=int(topo.endpoint_count),
        segment_count=int(topo.segment_count),
        mean_segment_length_um=float(topo.mean_segment_length_um),
        mean_tortuosity=float(topo.mean_tortuosity),
        surface_area_um2=float(surface_area),
        surface_to_volume_ratio_per_um=float(s2v),
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


def _radius_only_result(
    *,
    mask: np.ndarray,
    radii: np.ndarray,
    spacing_zyx: np.ndarray,
    voxel_volume: float,
    tissue_volume_um3: float,
    component_count: int,
) -> VascularAnalysisResult:
    radii = np.asarray(radii, dtype=np.float64)
    radii = radii[radii > 0.0]
    if radii.size == 0:
        radii = np.asarray([float(np.mean(spacing_zyx))], dtype=np.float64)
    vessel_voxels = int(np.count_nonzero(mask))
    vessel_volume_um3 = float(vessel_voxels) * voxel_volume
    surface_area = _surface_area_um2(mask, spacing_zyx)
    s2v = surface_area / vessel_volume_um3 if vessel_volume_um3 > 0.0 else 0.0
    return VascularAnalysisResult(
        vessel_voxel_count=vessel_voxels,
        vessel_volume_um3=vessel_volume_um3,
        tissue_volume_um3=tissue_volume_um3,
        vessel_volume_fraction=(
            float(vessel_voxels) / float(mask.size) if mask.size else 0.0
        ),
        component_count=int(component_count),
        total_length_um=0.0,
        length_density_mm_per_mm3=0.0,
        mean_radius_um=float(np.mean(radii)),
        median_radius_um=float(np.median(radii)),
        max_radius_um=float(np.max(radii)),
        mean_diameter_um=float(2.0 * np.mean(radii)),
        junction_count=0,
        junction_density_per_mm3=0.0,
        endpoint_count=0,
        segment_count=0,
        mean_segment_length_um=0.0,
        mean_tortuosity=0.0,
        surface_area_um2=float(surface_area),
        surface_to_volume_ratio_per_um=float(s2v),
    )


def _fill_small_cavities(mask: np.ndarray, max_voxels: int) -> np.ndarray:
    """Fill enclosed background pockets smaller than ``max_voxels``.

    Removes lumen speckle that would otherwise fragment the medial axis without
    materially changing the vessel surface.
    """
    binary = np.asarray(mask, dtype=bool)
    background = ~binary
    labels, count = ndi.label(background, structure=_FULL_STRUCTURE)
    if count == 0:
        return binary
    sizes = np.bincount(labels.ravel())
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
        if int(sizes[comp_id]) <= int(max_voxels):
            fill[labels == comp_id] = True
    return fill


def _surface_area_um2(mask: np.ndarray, spacing_zyx: np.ndarray) -> float:
    """Estimate vessel surface area from exposed voxel faces.

    Each vessel voxel face adjacent to background contributes the area of that
    face (the product of the two in-plane spacings). This is a fast, bias-aware
    estimate suitable for surface-to-volume ratios.
    """
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        return 0.0
    sz, sy, sx = (float(v) for v in spacing_zyx)
    face_area = {0: sy * sx, 1: sz * sx, 2: sz * sy}
    total = 0.0
    for axis in (0, 1, 2):
        # Count vessel/background transitions along this axis (both directions).
        diff = np.diff(binary.astype(np.int8), axis=axis)
        exposed = int(np.count_nonzero(diff == 1)) + int(np.count_nonzero(diff == -1))
        # Faces on the volume border for vessel voxels are also exposed.
        first = np.take(binary, 0, axis=axis)
        last = np.take(binary, binary.shape[axis] - 1, axis=axis)
        exposed += int(np.count_nonzero(first)) + int(np.count_nonzero(last))
        total += exposed * face_area[axis]
    return float(total)


def vascular_analysis_to_csv_rows(
    result: VascularAnalysisResult,
) -> list[dict[str, float | int | str]]:
    """Flatten the result into a single-row (metric, value) long-format table.

    Long format keeps the vascular CSV self-describing and easy to merge with
    the existing metrics export, regardless of how many fields evolve later.
    """
    data = asdict(result)
    return [{"metric": key, "value": value} for key, value in data.items()]
