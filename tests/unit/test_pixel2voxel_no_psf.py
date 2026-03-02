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

    assert np.allclose(out.data, arr, atol=0.0)


def test_psf_pipeline_skips_green_and_processes_red_for_pixel2voxel_strategy(monkeypatch) -> None:
    def _fake_deconvolve(volume, spacing, config, cancel_event=None, progress_callback=None):
        if progress_callback is not None:
            progress_callback(1, 1)
        return np.asarray(volume, dtype=np.float32) + np.float32(0.123)

    monkeypatch.setattr("nvap.pipeline.deconvolve_volume", _fake_deconvolve)

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
    assert np.allclose(out.red.data, red + np.float32(0.123), atol=0.0)


def test_psf_pipeline_skips_green_and_processes_red_for_microglia_strategy(monkeypatch) -> None:
    def _fake_deconvolve(volume, spacing, config, cancel_event=None, progress_callback=None):
        if progress_callback is not None:
            progress_callback(1, 1)
        return np.asarray(volume, dtype=np.float32) + np.float32(0.05)

    monkeypatch.setattr("nvap.pipeline.deconvolve_volume", _fake_deconvolve)

    rng = np.random.default_rng(23)
    green = np.clip(rng.random((3, 20, 20), dtype=np.float32) * 0.3, 0.0, 1.0)
    red = np.clip(rng.random((3, 20, 20), dtype=np.float32) * 0.3, 0.0, 1.0)
    spacing = VoxelSpacing()
    dataset = DatasetVolume(
        green=ChannelVolume("green", green.copy(), [0, 1, 2], spacing),
        red=ChannelVolume("red", red.copy(), [0, 1, 2], spacing),
        shared_z_range=(0, 2),
    )
    preprocess_cfg = PreprocessConfig(green_denoise_strategy="microglia_masking")
    out = apply_psf_to_dataset(
        dataset,
        config=PSFConfig(enabled=True, iterations=6),
        preprocess_config=preprocess_cfg,
    )

    assert np.allclose(out.green.data, green, atol=0.0)
    assert np.allclose(out.red.data, red + np.float32(0.05), atol=0.0)
