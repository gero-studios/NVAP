from __future__ import annotations

import logging

import numpy as np

from nvap.config.types import ChannelVolume

logger = logging.getLogger(__name__)


def interpolate_missing_slices(
    volume: np.ndarray, z_indices: list[int]
) -> tuple[np.ndarray, list[int]]:
    if volume.ndim != 3:
        raise ValueError("Volume must be a 3D array with (z, y, x) order.")
    if volume.shape[0] != len(z_indices):
        raise ValueError("z_indices length must match volume z dimension.")
    if len(z_indices) == 0:
        raise ValueError("z_indices cannot be empty.")

    order = np.argsort(np.asarray(z_indices))
    sorted_z = [int(z_indices[i]) for i in order]
    sorted_volume = volume[order].astype(np.float32, copy=False)

    if len(set(sorted_z)) != len(sorted_z):
        raise ValueError("z_indices must be unique.")

    z_min, z_max = sorted_z[0], sorted_z[-1]
    full_z = list(range(z_min, z_max + 1))
    known_set = set(sorted_z)
    missing = [z for z in full_z if z not in known_set]
    if missing:
        logger.info(
            "Interpolating %d missing z-slice(s) between %d and %d.",
            len(missing),
            z_min,
            z_max,
        )
        logger.debug("Missing z indices: %s", missing)
    else:
        logger.debug("No missing z-slices detected.")

    z_to_slice = {z: sorted_volume[idx] for idx, z in enumerate(sorted_z)}
    full_volume = np.empty((len(full_z),) + sorted_volume.shape[1:], dtype=np.float32)

    known = np.array(sorted_z, dtype=np.int32)
    for out_idx, z in enumerate(full_z):
        if z in z_to_slice:
            full_volume[out_idx] = z_to_slice[z]
            continue

        upper_idx = int(np.searchsorted(known, z))
        lower_idx = upper_idx - 1

        if lower_idx < 0:
            full_volume[out_idx] = z_to_slice[int(known[upper_idx])]
        elif upper_idx >= len(known):
            full_volume[out_idx] = z_to_slice[int(known[lower_idx])]
        else:
            z0 = int(known[lower_idx])
            z1 = int(known[upper_idx])
            s0 = z_to_slice[z0]
            s1 = z_to_slice[z1]
            alpha = (z - z0) / float(z1 - z0)
            full_volume[out_idx] = ((1.0 - alpha) * s0) + (alpha * s1)

    return full_volume, full_z


def fill_channel_missing_slices(channel: ChannelVolume) -> ChannelVolume:
    logger.debug("Filling missing slices for channel '%s'.", channel.name)
    full_volume, full_z = interpolate_missing_slices(channel.data, channel.z_indices)
    logger.info(
        "Channel '%s' z-slices: %d -> %d",
        channel.name,
        len(channel.z_indices),
        len(full_z),
    )
    return ChannelVolume(
        name=channel.name,
        data=full_volume,
        z_indices=full_z,
        spacing=channel.spacing,
    )
