from __future__ import annotations

import numpy as np

from nvap.config.types import ChannelVolume, PreprocessConfig, VoxelSpacing
from nvap.preprocess.enhancement import preprocess_channel


def test_faint_branch_continuity_survives_speckle_control() -> None:
    rng = np.random.default_rng(1234)
    arr = np.zeros((5, 64, 64), dtype=np.float32)
    arr[2, 30, 10:54] = 0.11
    arr[3, 31, 14:50] = 0.10
    # Add isolated bright speckles that should be attenuated.
    arr[2, 14, 14] = 0.95
    arr[3, 44, 46] = 0.9
    arr[1, 40, 20] = 0.85
    arr += rng.normal(0.0, 0.025, size=arr.shape).astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)

    channel = ChannelVolume("green", arr, list(range(arr.shape[0])), VoxelSpacing())
    cfg = PreprocessConfig(
        green_denoise_strategy="classical_branch_aware",
        green_branch_protection=0.78,
        green_speckle_min_voxels=10,
        green_speckle_attenuation=0.1,
    )
    out = preprocess_channel(channel, cfg).data

    branch_line = out[2, 30, 14:50]
    continuity = float(np.mean(branch_line > 0.05))
    background = float(out[:, 4:12, 4:12].mean())
    assert continuity >= 0.72
    assert float(branch_line.mean()) > (background * 1.8)
