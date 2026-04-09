from __future__ import annotations

import numpy as np

from nvap.config.types import ChannelVolume, DatasetVolume, VoxelSpacing
from nvap.pipeline import fill_and_sync_dataset


def _plane(value: float, shape: tuple[int, int] = (4, 4)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


def test_fill_and_sync_extends_red_to_green_max_with_zero_padding() -> None:
    spacing = VoxelSpacing()

    green = ChannelVolume(
        name="green",
        data=np.stack([_plane(0.1), _plane(0.2), _plane(0.3), _plane(0.4)], axis=0),
        z_indices=[1, 2, 3, 4],
        spacing=spacing,
    )
    red = ChannelVolume(
        name="red",
        data=np.stack([_plane(0.7), _plane(0.9)], axis=0),
        z_indices=[1, 2],
        spacing=spacing,
    )
    dataset = DatasetVolume(green=green, red=red, shared_z_range=(1, 2))

    synced = fill_and_sync_dataset(dataset)

    assert synced.green.z_indices == [1, 2, 3, 4]
    assert synced.red.z_indices == [1, 2, 3, 4]
    assert synced.shared_z_range == (1, 4)
    assert np.allclose(synced.red.data[0], 0.7)
    assert np.allclose(synced.red.data[1], 0.9)
    assert np.allclose(synced.red.data[2], 0.0)
    assert np.allclose(synced.red.data[3], 0.0)
