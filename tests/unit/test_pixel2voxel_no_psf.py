from __future__ import annotations

import numpy as np

from nvap.config.types import ChannelVolume, DatasetVolume, PSFConfig, PreprocessConfig, VoxelSpacing
from nvap.pipeline import apply_psf_to_dataset
from nvap.preprocess.enhancement import preprocess_channel


def test_pixel2voxel_no_psf_preserves_branch_signal() -> None:
    rng = np.random.default_rng(2026)
    arr = np.zeros((6, 48, 48), dtype=np.float32)
    arr[2, 24, 10:38] = 0.18
    arr[3, 24, 12:36] = 0.16
    arr[2, 8, 8] = 1.0
    arr[4, 39, 40] = 0.95
    arr += rng.normal(0.0, 0.03, size=arr.shape).astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)

    channel = ChannelVolume("green", arr, list(range(arr.shape[0])), VoxelSpacing())
    cfg = PreprocessConfig(
        green_denoise_strategy="pixel2voxel_no_psf",
        green_pre_deconv_strength=1.0,
        green_post_deconv_strength=0.0,
        green_branch_protection=0.76,
    )
    out = preprocess_channel(channel, cfg)

    branch_mean = float(out.data[2:4, 24, 14:34].mean())
    background_mean = float(out.data[:, 5:11, 5:11].mean())
    assert branch_mean > background_mean


def test_psf_pipeline_is_bypassed_for_pixel2voxel_strategy() -> None:
    rng = np.random.default_rng(17)
    green = np.clip(rng.random((4, 24, 24), dtype=np.float32) * 0.2, 0.0, 1.0)
    red = np.clip(rng.random((4, 24, 24), dtype=np.float32) * 0.2, 0.0, 1.0)
    spacing = VoxelSpacing()
    dataset = DatasetVolume(
        green=ChannelVolume("green", green.copy(), [0, 1, 2, 3], spacing),
        red=ChannelVolume("red", red.copy(), [0, 1, 2, 3], spacing),
        shared_z_range=(0, 3),
    )
    preprocess_cfg = PreprocessConfig(green_denoise_strategy="pixel2voxel_no_psf")
    out = apply_psf_to_dataset(
        dataset,
        config=PSFConfig(enabled=True, iterations=3),
        preprocess_config=preprocess_cfg,
    )

    assert np.allclose(out.green.data, green, atol=0.0)
    assert np.allclose(out.red.data, red, atol=0.0)
