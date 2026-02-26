from __future__ import annotations

from dataclasses import asdict
import logging

import numpy as np
import scipy.ndimage as ndi

from nvap.config.types import DatasetVolume, MetricsComputation, MetricsResult, RenderConfig

logger = logging.getLogger(__name__)


def mask_from_threshold(volume: np.ndarray, threshold: float) -> np.ndarray:
    return volume >= float(threshold)


def _component_stats(mask: np.ndarray) -> tuple[int, int]:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, count = ndi.label(mask, structure=structure)
    if count == 0:
        return 0, 0
    bincount = np.bincount(labels.ravel())
    largest = int(bincount[1:].max()) if bincount.size > 1 else 0
    return int(count), largest


def _shift_mask_integer(mask: np.ndarray, dz: int, dy: int, dx: int) -> np.ndarray:
    shifted = np.zeros_like(mask, dtype=bool)

    src_z0 = max(0, -dz)
    src_z1 = min(mask.shape[0], mask.shape[0] - dz)  # exclusive
    dst_z0 = max(0, dz)
    dst_z1 = min(mask.shape[0], mask.shape[0] + dz)  # exclusive

    src_y0 = max(0, -dy)
    src_y1 = min(mask.shape[1], mask.shape[1] - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(mask.shape[1], mask.shape[1] + dy)

    src_x0 = max(0, -dx)
    src_x1 = min(mask.shape[2], mask.shape[2] - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(mask.shape[2], mask.shape[2] + dx)

    if (
        src_z0 < src_z1
        and src_y0 < src_y1
        and src_x0 < src_x1
        and dst_z0 < dst_z1
        and dst_y0 < dst_y1
        and dst_x0 < dst_x1
    ):
        shifted[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = mask[
            src_z0:src_z1, src_y0:src_y1, src_x0:src_x1
        ]

    return shifted


def _shared_subvolumes(
    green_mask: np.ndarray,
    red_mask: np.ndarray,
    green_z_indices: list[int],
    red_z_indices: list[int],
    shared_z_range: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    z0, z1 = shared_z_range
    shared = list(range(z0, z1 + 1))
    g_map = {z: i for i, z in enumerate(green_z_indices)}
    r_map = {z: i for i, z in enumerate(red_z_indices)}

    g_stack = np.stack([green_mask[g_map[z]] for z in shared], axis=0)
    r_stack = np.stack([red_mask[r_map[z]] for z in shared], axis=0)
    return g_stack, r_stack


def compute_metrics(dataset: DatasetVolume, render: RenderConfig) -> MetricsComputation:
    green = dataset.green
    red = dataset.red
    spacing = green.spacing
    voxel_volume_um3 = spacing.voxel_volume_um3

    green_mask = mask_from_threshold(green.data, render.threshold_green)
    red_mask = mask_from_threshold(red.data, render.threshold_red)

    green_components, green_largest = _component_stats(green_mask)
    red_components, red_largest = _component_stats(red_mask)

    green_voxels = int(green_mask.sum())
    red_voxels = int(red_mask.sum())

    # Offsets are applied to green mask for overlap comparisons.
    dz = int(round(render.offset_z_um / spacing.z_um))
    dy = int(round(render.offset_y_um / spacing.y_um))
    dx = int(round(render.offset_x_um / spacing.x_um))

    green_shared, red_shared = _shared_subvolumes(
        green_mask=green_mask,
        red_mask=red_mask,
        green_z_indices=green.z_indices,
        red_z_indices=red.z_indices,
        shared_z_range=dataset.shared_z_range,
    )
    shifted_green = _shift_mask_integer(green_shared, dz=dz, dy=dy, dx=dx)
    overlap_voxels = int(np.logical_and(shifted_green, red_shared).sum())

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
