from __future__ import annotations

import numpy as np

from nvap.config.types import ChannelVolume, PreprocessConfig, VoxelSpacing
from nvap.preprocess.enhancement import preprocess_channel


def test_green_speckle_reduction_preserves_branch() -> None:
    rng = np.random.default_rng(7)
    arr = np.zeros((5, 56, 56), dtype=np.float32)
    arr[2, 28, 12:44] = 0.22
    arr[2, 12, 12] = 1.0
    arr[2, 16, 42] = 0.95
    arr[3, 18, 20] = 0.9
    arr += rng.normal(0.0, 0.02, size=arr.shape).astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)

    channel = ChannelVolume("green", arr, [0, 1, 2, 3, 4], VoxelSpacing())
    cfg = PreprocessConfig(
        green_denoise_strategy="classical_branch_aware",
        green_pre_deconv_strength=1.0,
        green_post_deconv_strength=0.0,
        green_speckle_min_voxels=12,
        green_speckle_attenuation=0.1,
        green_branch_protection=0.7,
    )
    out = preprocess_channel(channel, cfg)
    branch_signal = float(out.data[2, 28, 18:38].mean())
    speckle_peak = float(max(out.data[2, 12, 12], out.data[2, 16, 42], out.data[3, 18, 20]))

    assert branch_signal > speckle_peak
