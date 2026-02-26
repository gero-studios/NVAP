from __future__ import annotations

import numpy as np

from nvap.config.types import ChannelVolume, PreprocessConfig, VoxelSpacing
from nvap.preprocess.enhancement import preprocess_channel


def test_bm4d_backend_falls_back_without_optional_dependency() -> None:
    arr = np.zeros((4, 24, 24), dtype=np.float32)
    arr[1:3, 12, 8:18] = 0.2
    arr += 0.03 * np.random.default_rng(1).random(arr.shape, dtype=np.float32)
    channel = ChannelVolume("green", arr, [0, 1, 2, 3], VoxelSpacing())
    cfg = PreprocessConfig(
        green_denoise_strategy="bm4d",
        green_pre_deconv_strength=1.0,
        green_post_deconv_strength=0.0,
    )
    out = preprocess_channel(channel, cfg)
    assert out.data.shape == channel.data.shape
    assert float(np.mean(out.data)) >= 0.0
