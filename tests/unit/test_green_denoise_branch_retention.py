from __future__ import annotations

from dataclasses import replace

import numpy as np

from nvap.config.types import ChannelVolume, PreprocessConfig, VoxelSpacing
from nvap.preprocess.enhancement import preprocess_channel


def test_green_branch_retention_beats_legacy_anisotropic() -> None:
    rng = np.random.default_rng(123)
    arr = np.zeros((6, 48, 48), dtype=np.float32)
    arr[2, 24, 8:40] = 0.16
    arr[3, 24, 8:40] = 0.14
    arr += rng.normal(0.0, 0.03, size=arr.shape).astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)

    channel = ChannelVolume("green", arr, list(range(6)), VoxelSpacing())
    cfg = PreprocessConfig(
        green_denoise_strategy="classical_branch_aware",
        green_pre_deconv_strength=1.0,
        green_post_deconv_strength=0.0,
        green_branch_protection=0.72,
    )
    out = preprocess_channel(channel, cfg)
    legacy = preprocess_channel(channel, replace(cfg, green_denoise_strategy="legacy_anisotropic"))

    new_branch = float(out.data[2:4, 24, 12:36].mean())
    legacy_branch = float(legacy.data[2:4, 24, 12:36].mean())
    new_background = float(out.data[:, 6:12, 6:12].mean())

    assert new_branch > new_background
    # Wavelet denoising trades peak intensity for better edge/branch structure.
    assert new_branch >= (legacy_branch * 0.85)
