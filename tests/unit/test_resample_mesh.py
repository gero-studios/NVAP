from __future__ import annotations

import numpy as np

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
