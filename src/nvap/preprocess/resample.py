from __future__ import annotations

import logging

import numpy as np
import scipy.ndimage as ndi

from nvap.config.types import ChannelVolume, DatasetVolume, PreprocessConfig, VoxelSpacing

logger = logging.getLogger(__name__)


def resample_channel_for_mesh(
    channel: ChannelVolume,
    target_z_um: float,
) -> ChannelVolume:
    target = float(max(target_z_um, 1e-6))
    factor_z = float(channel.spacing.z_um / target)
    if abs(factor_z - 1.0) < 0.02:
        return channel

    zoom_factors = (factor_z, 1.0, 1.0)
    logger.info(
        "Resampling channel '%s' for mesh: z_factor=%.3f old_shape=%s",
        channel.name,
        factor_z,
        channel.data.shape,
    )
    resampled = ndi.zoom(
        channel.data.astype(np.float32, copy=False),
        zoom=zoom_factors,
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32, copy=False)

    spacing = VoxelSpacing(
        x_um=channel.spacing.x_um,
        y_um=channel.spacing.y_um,
        z_um=target,
    )
    z_start = min(channel.z_indices) if channel.z_indices else 1
    z_indices = list(range(z_start, z_start + int(resampled.shape[0])))
    return ChannelVolume(
        name=channel.name,
        data=resampled,
        z_indices=z_indices,
        spacing=spacing,
    )


def prepare_mesh_dataset(dataset: DatasetVolume, config: PreprocessConfig) -> DatasetVolume:
    if not config.resample_for_mesh:
        return dataset
    green = resample_channel_for_mesh(dataset.green, config.mesh_target_z_um)
    red = resample_channel_for_mesh(dataset.red, config.mesh_target_z_um)
    shared_start = max(min(green.z_indices), min(red.z_indices))
    shared_end = min(max(green.z_indices), max(red.z_indices))
    shared = (shared_start, shared_end) if shared_start <= shared_end else dataset.shared_z_range
    return DatasetVolume(green=green, red=red, shared_z_range=shared)
