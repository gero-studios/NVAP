"""Smooth fused 3D volume reconstruction from stacked 2D slices.

The raw dataset is a stack of anisotropic 2D slices (xy = 0.331 um, z = 0.4 um).
Branch / tip / distance analytics historically ran straight on that staircased
voxel grid, inheriting every slice-stacking artifact.

This module rebuilds the green (microglia) channel as a single smooth, fused
3D shape so downstream skeletonisation and distance maths are genuinely 3D:

1. Isotropic resampling   -> cubic voxels (cubic-spline on green intensity).
2. Gap bridging (fusion)  -> reconnect processes broken across slices with thin
                             connectors, distance-constrained to avoid blunt
                             morphological ballooning.
3. Thin-aware smoothing   -> remove staircasing via a signed-distance field,
                             while protecting thin processes from erosion.

The green channel becomes a smooth occupancy field in ``[0, 1]`` whose shape
surface sits exactly at ``0.5`` (``occupancy >= 0.5`` recovers the binary mask).
The red (vasculature) channel is only geometrically resampled onto the same
isotropic grid -- no smoothing, no fusion -- so overlap and distance-to-vessel
maths stay co-registered with green.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu

from nvap.config.types import (
    ChannelVolume,
    DatasetVolume,
    PreprocessConfig,
    VoxelSpacing,
)

logger = logging.getLogger(__name__)

_FULL_STRUCTURE = ndi.generate_binary_structure(3, 3).astype(np.uint8, copy=False)

# Surface level of the fused green occupancy field. The field is built so that
# ``occupancy >= FUSED_GREEN_SURFACE_LEVEL`` recovers the binary fused shape
# exactly; render, metrics, and analysis all threshold green at this level.
FUSED_GREEN_SURFACE_LEVEL: float = 0.5


@dataclass(frozen=True)
class FusedVolumeParams:
    """Tunables for the smooth-fused-volume reconstruction.

    Defaults encode the decisions agreed for NVAP: finest-resolution isotropic
    target, *moderate* gap bridging (~4.5 um), and thin-branch-preserving
    smoothing.
    """

    enabled: bool = True
    # 0.0 -> auto: finest in-plane spacing, stepped coarser for huge stacks.
    isotropic_target_um: float = 0.0
    max_isotropic_voxels: int = 240_000_000
    # Gap bridging.
    bridge_max_gap_um: float = 4.5
    bridge_tube_radius_um: float = 0.35
    min_fragment_voxels: int = 4
    bridge_min_endpoint_voxels: int = 6
    bridge_max_components: int = 8000
    # Thin-aware smoothing.
    smoothing_sigma_um: float = 0.33
    thin_branch_thickness_um: float = 0.7


DEFAULT_FUSED_PARAMS = FusedVolumeParams()


def fused_params_from_preprocess_config(config: PreprocessConfig) -> FusedVolumeParams:
    """Build :class:`FusedVolumeParams` from a :class:`PreprocessConfig`."""
    return FusedVolumeParams(
        enabled=bool(config.fuse_smooth_enabled),
        isotropic_target_um=float(config.fuse_isotropic_target_um),
        bridge_max_gap_um=float(config.fuse_bridge_max_gap_um),
        bridge_tube_radius_um=float(config.fuse_bridge_tube_radius_um),
        thin_branch_thickness_um=float(config.fuse_thin_branch_thickness_um),
        smoothing_sigma_um=float(config.fuse_smoothing_sigma_um),
    )


# ---------------------------------------------------------------------------
# Isotropic resampling
# ---------------------------------------------------------------------------

def resolve_isotropic_target_um(
    spacing: VoxelSpacing,
    shape: tuple[int, int, int],
    params: FusedVolumeParams = DEFAULT_FUSED_PARAMS,
) -> float:
    """Pick the cubic voxel size to resample to.

    Auto mode anchors on the finest in-plane spacing for best centerline
    accuracy, then steps coarser if the resulting volume would exceed the
    voxel-count budget (keeps large stacks within the analysis memory caps).
    """
    if params.isotropic_target_um and params.isotropic_target_um > 0.0:
        return float(params.isotropic_target_um)

    if len(shape) != 3 or any(int(axis) <= 0 for axis in shape):
        return float(min(spacing.x_um, spacing.y_um))

    finest = float(min(spacing.x_um, spacing.y_um))
    if finest <= 0.0:
        return float(max(spacing.x_um, spacing.y_um, spacing.z_um, 1.0e-3))

    extent_um = (
        float(shape[0]) * float(spacing.z_um),
        float(shape[1]) * float(spacing.y_um),
        float(shape[2]) * float(spacing.x_um),
    )

    def output_voxels(target: float) -> float:
        return float(np.prod([axis / target for axis in extent_um]))

    cap = float(max(1, params.max_isotropic_voxels))
    target = finest
    ceiling = float(max(spacing.x_um, spacing.y_um, spacing.z_um)) * 4.0
    while output_voxels(target) > cap and target < ceiling:
        target *= 1.08
    if target > finest:
        logger.info(
            "Fused volume: stepped isotropic target %.4f um -> %.4f um to respect "
            "%.0fM voxel cap (extent_um=%s).",
            finest, target, cap / 1.0e6, tuple(round(v, 1) for v in extent_um),
        )
    return float(target)


def resample_channel_isotropic(
    channel: ChannelVolume,
    target_um: float,
    *,
    order: int,
) -> ChannelVolume:
    """Resample ``channel`` onto a cubic ``target_um`` grid.

    ``order=3`` (cubic spline) is used for the green intensity volume so the
    inter-slice interpolation is smooth; ``order=1`` (linear) for red, which is
    only co-registered, not smoothed.
    """
    target = float(max(target_um, 1.0e-6))
    sp = channel.spacing
    factors = (
        float(sp.z_um) / target,
        float(sp.y_um) / target,
        float(sp.x_um) / target,
    )
    data = np.asarray(channel.data, dtype=np.float32)
    iso_spacing = VoxelSpacing(x_um=target, y_um=target, z_um=target)

    if all(abs(f - 1.0) < 0.01 for f in factors):
        resampled = data.copy()
    else:
        logger.info(
            "Fused volume: resampling channel '%s' to isotropic %.4f um "
            "factors=(%.3f,%.3f,%.3f) old_shape=%s",
            channel.name, target, factors[0], factors[1], factors[2], data.shape,
        )
        resampled = ndi.zoom(
            data,
            zoom=factors,
            order=int(order),
            mode="nearest",
            prefilter=int(order) > 1,
        ).astype(np.float32, copy=False)
        # Cubic splines can overshoot; clamp back into the original value range.
        if data.size:
            lo = float(min(0.0, float(data.min())))
            hi = float(max(lo + 1.0e-6, float(data.max())))
            np.clip(resampled, lo, hi, out=resampled)

    start = int(min(channel.z_indices)) if channel.z_indices else 1
    z_indices = list(range(start, start + int(resampled.shape[0])))
    return ChannelVolume(
        name=channel.name,
        data=resampled,
        z_indices=z_indices,
        spacing=iso_spacing,
    )


# ---------------------------------------------------------------------------
# Working mask helpers
# ---------------------------------------------------------------------------

def otsu_threshold(volume: np.ndarray, fallback: float = 0.15) -> float:
    """Otsu threshold over finite values, with a safe fallback."""
    arr = np.asarray(volume, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or float(finite.max()) <= 0.0:
        return float(fallback)
    try:
        return float(np.clip(threshold_otsu(finite), 1.0e-4, 1.0))
    except ValueError:
        return float(fallback)


def remove_small_fragments(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    """Drop connected components smaller than ``min_voxels`` (speckle/dust)."""
    binary = np.asarray(mask, dtype=bool)
    if min_voxels <= 1 or not binary.any():
        return binary.copy()
    labels, count = ndi.label(binary, structure=_FULL_STRUCTURE)
    if count == 0:
        return binary.copy()
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_voxels)
    keep[0] = False
    return keep[labels]


# ---------------------------------------------------------------------------
# Gap bridging (fusion)
# ---------------------------------------------------------------------------

def _draw_tube(
    out: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    radius_vox: int,
) -> None:
    """Paint a thin straight connector between two voxel coordinates."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    seg = p1 - p0
    length = float(np.linalg.norm(seg))
    steps = max(2, int(np.ceil(length)) + 1)
    shape = out.shape
    rv = int(max(0, radius_vox))
    ball = [
        (dz, dy, dx)
        for dz in range(-rv, rv + 1)
        for dy in range(-rv, rv + 1)
        for dx in range(-rv, rv + 1)
        if dz * dz + dy * dy + dx * dx <= rv * rv
    ]
    for t in np.linspace(0.0, 1.0, steps):
        center = np.round(p0 + t * seg).astype(np.int64)
        for dz, dy, dx in ball:
            z = int(center[0]) + dz
            y = int(center[1]) + dy
            x = int(center[2]) + dx
            if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
                out[z, y, x] = True


def bridge_component_gaps(
    mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    max_gap_um: float,
    tube_radius_um: float,
    min_endpoint_voxels: int = 6,
    max_components: int = 8000,
) -> tuple[np.ndarray, int]:
    """Reconnect fragments whose surfaces fall within ``max_gap_um``.

    Uses surface-to-surface nearest-pair distances (KD-tree) and draws a thin
    connector for each gap under threshold. A union-find pass dedupes redundant
    connectors so an already-linked cluster is not re-bridged. Returns the
    bridged mask and the number of connectors added.
    """
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return binary.copy(), 0

    labels, count = ndi.label(binary, structure=_FULL_STRUCTURE)
    if count <= 1:
        return binary.copy(), 0
    if count > int(max_components):
        logger.warning(
            "Fused volume: %d components exceeds bridge cap %d; skipping gap bridging.",
            count, max_components,
        )
        return binary.copy(), 0

    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    surface = binary & ~ndi.binary_erosion(binary, structure=_FULL_STRUCTURE, border_value=0)

    comp_coords: dict[int, np.ndarray] = {}
    comp_um: dict[int, np.ndarray] = {}
    comp_bbox: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    objects = ndi.find_objects(labels)
    for lab in range(1, count + 1):
        sl = objects[lab - 1]
        if sl is None:
            continue
        sub = surface[sl] & (labels[sl] == lab)
        local = np.argwhere(sub)
        if local.shape[0] < int(min_endpoint_voxels):
            continue
        offset = np.array([s.start for s in sl], dtype=np.int64)
        coords = local.astype(np.int64) + offset
        pts_um = coords.astype(np.float64) * spacing
        comp_coords[lab] = coords
        comp_um[lab] = pts_um
        comp_bbox[lab] = (pts_um.min(axis=0), pts_um.max(axis=0))

    valid = sorted(comp_coords)
    if len(valid) <= 1:
        return binary.copy(), 0

    trees = {lab: cKDTree(comp_um[lab]) for lab in valid}
    max_gap = float(max_gap_um)

    bridges: list[tuple[float, int, int, np.ndarray, np.ndarray]] = []
    for ii, lab_a in enumerate(valid):
        a_min, a_max = comp_bbox[lab_a]
        for lab_b in valid[ii + 1:]:
            b_min, b_max = comp_bbox[lab_b]
            bbox_gap = np.maximum(0.0, np.maximum(a_min - b_max, b_min - a_max))
            if float(np.linalg.norm(bbox_gap)) > max_gap:
                continue
            dist, idx = trees[lab_b].query(comp_um[lab_a], k=1)
            best_a = int(np.argmin(dist))
            gap = float(dist[best_a])
            if gap > max_gap:
                continue
            best_b = int(idx[best_a])
            bridges.append(
                (gap, lab_a, lab_b, comp_coords[lab_a][best_a], comp_coords[lab_b][best_b])
            )

    if not bridges:
        return binary.copy(), 0

    bridges.sort(key=lambda item: item[0])
    parent = {lab: lab for lab in valid}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    out = binary.copy()
    radius_vox = int(max(1, round(float(tube_radius_um) / float(spacing.min()))))
    added = 0
    for _gap, lab_a, lab_b, pt_a, pt_b in bridges:
        root_a, root_b = find(lab_a), find(lab_b)
        if root_a == root_b:
            continue
        _draw_tube(out, pt_a, pt_b, radius_vox)
        parent[root_a] = root_b
        added += 1

    logger.info(
        "Fused volume: bridged %d gap(s) across %d fragments (max_gap=%.2f um).",
        added, len(valid), max_gap,
    )
    return out, added


# ---------------------------------------------------------------------------
# Thin-aware smoothing
# ---------------------------------------------------------------------------

def smooth_shape_preserving_thin(
    mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    sigma_um: float,
    thin_thickness_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth staircasing while protecting thin processes.

    The shape is smoothed by Gaussian-blurring its signed-distance field and
    re-thresholding at zero -- the standard staircase-free binary smoothing
    trick. Voxels in thin regions (small local thickness) are then unioned back
    so smoothing can never erode a thin process into disconnection.

    Returns ``(binary_mask, occupancy_field)`` where ``occupancy_field`` is a
    float field in ``[0, 1]`` with the shape surface at ``0.5``
    (``occupancy >= 0.5`` exactly recovers ``binary_mask``).
    """
    binary = np.asarray(mask, dtype=bool)
    shape = binary.shape
    if not binary.any():
        return binary.copy(), np.zeros(shape, dtype=np.float32)

    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    inside = ndi.distance_transform_edt(binary, sampling=spacing)
    outside = ndi.distance_transform_edt(~binary, sampling=spacing)
    sdf = np.asarray(inside, dtype=np.float32) - np.asarray(outside, dtype=np.float32)

    sigma_vox = tuple(float(sigma_um) / float(s) for s in spacing)
    sdf_smoothed = ndi.gaussian_filter(sdf, sigma=sigma_vox, mode="nearest")
    smoothed = sdf_smoothed >= 0.0

    # A voxel is "thin" when its surrounding region never gets thick: take the
    # local max of the inside-distance so a soma's surface shell (thick core
    # nearby) is NOT protected, but a slim process (thin everywhere) is.
    thin_radius_vox = max(
        1, int(round(float(thin_thickness_um) / float(spacing.min())))
    )
    local_max_thickness = ndi.maximum_filter(
        np.asarray(inside, dtype=np.float32),
        size=2 * thin_radius_vox + 1,
        mode="nearest",
    )
    thin = binary & (local_max_thickness <= float(thin_thickness_um))
    protect = binary & ndi.binary_dilation(thin, iterations=2)

    result = smoothed | protect

    in2 = ndi.distance_transform_edt(result, sampling=spacing)
    out2 = ndi.distance_transform_edt(~result, sampling=spacing)
    sdf2 = np.asarray(in2, dtype=np.float32) - np.asarray(out2, dtype=np.float32)
    band = float(max(spacing.min(), 1.0e-6))
    occupancy = np.clip(0.5 + sdf2 / (2.0 * band), 0.0, 1.0).astype(np.float32)
    return result, occupancy


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FusedVolumeResult:
    """Outcome of :func:`build_fused_dataset`, with provenance for logging."""

    dataset: DatasetVolume
    isotropic_target_um: float
    green_threshold: float
    bridges_added: int
    green_mask: np.ndarray


def build_fused_green(
    green_iso: ChannelVolume,
    *,
    threshold: float,
    params: FusedVolumeParams = DEFAULT_FUSED_PARAMS,
) -> tuple[ChannelVolume, np.ndarray, int]:
    """Bridge + smooth an already-isotropic green channel.

    Returns ``(green_channel, binary_mask, bridges_added)`` where the channel
    data is the smooth occupancy field.
    """
    target = float(green_iso.spacing.x_um)
    spacing_zyx = (target, target, target)

    mask = np.asarray(green_iso.data, dtype=np.float32) >= float(threshold)
    mask = remove_small_fragments(mask, params.min_fragment_voxels)

    bridged, bridges_added = bridge_component_gaps(
        mask,
        spacing_zyx,
        max_gap_um=params.bridge_max_gap_um,
        tube_radius_um=params.bridge_tube_radius_um,
        min_endpoint_voxels=params.bridge_min_endpoint_voxels,
        max_components=params.bridge_max_components,
    )

    final_mask, occupancy = smooth_shape_preserving_thin(
        bridged,
        spacing_zyx,
        sigma_um=params.smoothing_sigma_um,
        thin_thickness_um=params.thin_branch_thickness_um,
    )

    green_channel = ChannelVolume(
        name="green",
        data=occupancy,
        z_indices=list(green_iso.z_indices),
        spacing=green_iso.spacing,
    )
    return green_channel, final_mask, bridges_added


def build_fused_dataset(
    dataset: DatasetVolume,
    params: FusedVolumeParams = DEFAULT_FUSED_PARAMS,
    *,
    green_threshold: float | None = None,
) -> FusedVolumeResult:
    """Rebuild ``dataset`` as a smooth fused 3D model.

    Green is resampled to an isotropic grid, gap-bridged, and smoothed into an
    occupancy field. Red is resampled onto the *same* isotropic grid only
    (co-registration) so overlap/distance maths stay valid.
    """
    if not params.enabled:
        logger.info("Fused volume: disabled; returning dataset unchanged.")
        return FusedVolumeResult(
            dataset=dataset,
            isotropic_target_um=float(dataset.green.spacing.x_um),
            green_threshold=float("nan"),
            bridges_added=0,
            green_mask=np.zeros((0, 0, 0), dtype=bool),
        )

    target = resolve_isotropic_target_um(
        dataset.green.spacing, dataset.green.data.shape, params
    )

    green_iso = resample_channel_isotropic(dataset.green, target, order=3)
    red_iso = resample_channel_isotropic(dataset.red, target, order=1)

    threshold = (
        float(green_threshold)
        if green_threshold is not None
        else otsu_threshold(green_iso.data)
    )

    green_channel, green_mask, bridges_added = build_fused_green(
        green_iso, threshold=threshold, params=params
    )

    z_start = int(min(green_channel.z_indices)) if green_channel.z_indices else 1
    z_end = int(max(green_channel.z_indices)) if green_channel.z_indices else z_start
    fused = DatasetVolume(
        green=green_channel,
        red=red_iso,
        shared_z_range=(z_start, z_end),
    )
    logger.info(
        "Fused volume: built green=%s red=%s target=%.4f um threshold=%.4f bridges=%d",
        green_channel.data.shape, red_iso.data.shape, target, threshold, bridges_added,
    )
    return FusedVolumeResult(
        dataset=fused,
        isotropic_target_um=target,
        green_threshold=threshold,
        bridges_added=bridges_added,
        green_mask=green_mask,
    )
