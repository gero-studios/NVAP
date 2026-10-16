from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from nvap.config.types import ChannelVolume, VoxelSpacing
from nvap.preprocess.resample import resample_channel_for_mesh


def test_resample_channel_for_mesh_increases_z_when_target_is_smaller() -> None:
    spacing = VoxelSpacing(x_um=0.331, y_um=0.331, z_um=0.4)
    channel = ChannelVolume(
        name="green",
        data=np.zeros((10, 8, 8), dtype=np.float32),
        z_indices=list(range(1, 11)),
        spacing=spacing,
    )
    out = resample_channel_for_mesh(channel, target_z_um=0.331)
    assert out.data.shape[0] > channel.data.shape[0]
    assert abs(out.spacing.z_um - 0.331) < 1e-6


def test_red_mesh_resample_preserves_thresholded_diagonal_connections() -> None:
    data = np.zeros((10, 16, 16), dtype=np.float32)
    for z in range(10):
        data[z, z + 2, z + 2] = 0.5
    channel = ChannelVolume("red", data, list(range(10)), VoxelSpacing(z_um=0.4))

    out = resample_channel_for_mesh(channel, target_z_um=0.331)

    # Linear interpolation used to sever this one native component into dots.
    structure = ndi.generate_binary_structure(3, 3)
    assert ndi.label(out.data >= 0.4, structure)[1] == 1
    assert set(np.unique(out.data)) == {0.0, 0.5}
