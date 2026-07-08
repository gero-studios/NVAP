from __future__ import annotations

from dataclasses import asdict
import logging

import numpy as np
import scipy.ndimage as ndi

from nvap.config.types import DatasetVolume, MetricsComputation, MetricsResult, RenderConfig

logger = logging.getLogger(__name__)


def mask_from_threshold(volume: np.ndarray, threshold: float) -> np.ndarray:
    return volume >= float(threshold)


def _apply_render_trim(mask: np.ndarray, trim_first_slices: int, trim_last_slices: int) -> np.ndarray:
    """Zero the first/last z-slices so metrics match the trimmed render and analysis."""
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 3 or arr.shape[0] <= 0:
        return arr
    trim_first = max(0, int(trim_first_slices))
    trim_last = max(0, int(trim_last_slices))
    if trim_first <= 0 and trim_last <= 0:
        return arr
    if trim_first + trim_last >= int(arr.shape[0]):
        return np.zeros_like(arr)
    out = arr.copy()
    if trim_first > 0:
        out[:trim_first] = False
    if trim_last > 0:
        out[-trim_last:] = False
    return out


def _component_stats(mask: np.ndarray) -> tuple[int, int]:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, count = ndi.label(mask, structure=structure)
    if count == 0:
        return 0, 0
    bincount = np.bincount(labels.flat)
    largest = int(bincount[1:].max()) if bincount.size > 1 else 0
    return int(count), largest


def _shift_slice_in_plane(slice_mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a single (y, x) mask by integer voxels, zero-filling exposed edges."""
    out = np.zeros_like(slice_mask, dtype=bool)
    ny, nx = slice_mask.shape
    src_y0 = max(0, -dy)
    src_y1 = min(ny, ny - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(ny, ny + dy)
    src_x0 = max(0, -dx)
    src_x1 = min(nx, nx - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(nx, nx + dx)
    if src_y0 < src_y1 and src_x0 < src_x1 and dst_y0 < dst_y1 and dst_x0 < dst_x1:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = slice_mask[src_y0:src_y1, src_x0:src_x1]
    return out


def _shifted_overlap_voxels(
    green_mask: np.ndarray,
    red_mask: np.ndarray,
    green_z_indices: list[int],
    red_z_indices: list[int],
    shared_z_range: tuple[int, int],
    *,
    dz: int,
    dy: int,
    dx: int,
) -> int:
    """Voxel overlap of the (dz, dy, dx)-shifted green mask with the red mask.

    The z-shift is resolved in *physical* slice space: for each shared physical z
    aligned with red, the contributing green slice is sampled from physical
    ``z - dz``. That source slice may lie outside the shared range but still
    within green's acquired stack, so voxels shifting across the shared-range
    boundary are handled correctly instead of being dropped (and slices are pulled
    in from outside the boundary rather than zero-filled).
    """
    z0, z1 = shared_z_range
    g_map = {z: i for i, z in enumerate(green_z_indices)}
    r_map = {z: i for i, z in enumerate(red_z_indices)}

    overlap = 0
    for z in range(z0, z1 + 1):
        r_idx = r_map.get(z)
        if r_idx is None:
            continue
        g_idx = g_map.get(z - dz)
        if g_idx is None:
            continue
        green_slice = _shift_slice_in_plane(green_mask[g_idx], dy, dx)
        overlap += int(np.logical_and(green_slice, red_mask[r_idx]).sum())
    return overlap


def compute_metrics(dataset: DatasetVolume, render: RenderConfig) -> MetricsComputation:
    green = dataset.green
    red = dataset.red
    spacing = green.spacing
    voxel_volume_um3 = spacing.voxel_volume_um3

    trim_first = int(render.trim_first_slices)
    trim_last = int(render.trim_last_slices)
    green_mask = _apply_render_trim(
        mask_from_threshold(green.data, render.threshold_green), trim_first, trim_last
    )
    red_mask = _apply_render_trim(
        mask_from_threshold(red.data, render.threshold_red), trim_first, trim_last
    )

    green_components, green_largest = _component_stats(green_mask)
    red_components, red_largest = _component_stats(red_mask)

    green_voxels = int(green_mask.sum())
    red_voxels = int(red_mask.sum())

    # Offsets are applied to green mask for overlap comparisons.
    dz = int(round(render.offset_z_um / spacing.z_um))
    dy = int(round(render.offset_y_um / spacing.y_um))
    dx = int(round(render.offset_x_um / spacing.x_um))

    overlap_voxels = _shifted_overlap_voxels(
        green_mask=green_mask,
        red_mask=red_mask,
        green_z_indices=green.z_indices,
        red_z_indices=red.z_indices,
        shared_z_range=dataset.shared_z_range,
        dz=dz,
        dy=dy,
        dx=dx,
    )

    overlap_volume = overlap_voxels * voxel_volume_um3
    result_green = MetricsResult(
        channel="green",
        voxel_count=green_voxels,
        volume_um3=green_voxels * voxel_volume_um3,
        component_count=green_components,
        largest_component_voxels=green_largest,
        overlap_voxel_count=overlap_voxels,
        overlap_volume_um3=overlap_volume,
    )
    result_red = MetricsResult(
        channel="red",
        voxel_count=red_voxels,
        volume_um3=red_voxels * voxel_volume_um3,
        component_count=red_components,
        largest_component_voxels=red_largest,
        overlap_voxel_count=overlap_voxels,
        overlap_volume_um3=overlap_volume,
    )
    result = MetricsComputation(
        channel_results=[result_green, result_red],
        overlap_voxel_count=overlap_voxels,
        overlap_volume_um3=overlap_volume,
    )
    logger.debug(
        "Metrics computed: green_vox=%d red_vox=%d overlap_vox=%d offsets_um=(%.3f,%.3f,%.3f)",
        green_voxels,
        red_voxels,
        overlap_voxels,
        render.offset_x_um,
        render.offset_y_um,
        render.offset_z_um,
    )
    return result


def metrics_to_csv_rows(metrics: MetricsComputation) -> list[dict[str, int | float | str]]:
    rows = [asdict(item) for item in metrics.channel_results]
    rows.append(
        {
            "channel": "overlap",
            "voxel_count": 0,
            "volume_um3": 0.0,
            "component_count": 0,
            "largest_component_voxels": 0,
            "overlap_voxel_count": metrics.overlap_voxel_count,
            "overlap_volume_um3": metrics.overlap_volume_um3,
        }
    )
    return rows
