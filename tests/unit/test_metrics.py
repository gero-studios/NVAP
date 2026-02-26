from __future__ import annotations

import numpy as np

from nvap.analysis.metrics import compute_metrics
from nvap.config.types import ChannelVolume, DatasetVolume, RenderConfig, VoxelSpacing


def _dataset() -> DatasetVolume:
    spacing = VoxelSpacing()
    green = np.zeros((2, 3, 3), dtype=np.float32)
    red = np.zeros((2, 3, 3), dtype=np.float32)
    green[0, 1, 1] = 1.0
    green[1, 1, 1] = 1.0
    red[0, 1, 1] = 1.0
    red[1, 0, 0] = 1.0
    g = ChannelVolume(name="green", data=green, z_indices=[1, 2], spacing=spacing)
    r = ChannelVolume(name="red", data=red, z_indices=[1, 2], spacing=spacing)
    return DatasetVolume(green=g, red=r, shared_z_range=(1, 2))


def test_metrics_counts_and_overlap() -> None:
    metrics = compute_metrics(
        _dataset(),
        RenderConfig(
            threshold_green=0.5,
            threshold_red=0.5,
            opacity_green=0.3,
            opacity_red=0.3,
        ),
    )
    green = next(item for item in metrics.channel_results if item.channel == "green")
    red = next(item for item in metrics.channel_results if item.channel == "red")
    assert green.voxel_count == 2
    assert red.voxel_count == 2
    assert metrics.overlap_voxel_count == 1
