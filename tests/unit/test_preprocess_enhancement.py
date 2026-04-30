from __future__ import annotations

import numpy as np

from nvap.config.types import ChannelVolume, PreprocessConfig, VoxelSpacing
from nvap.preprocess.enhancement import (
    _imagej_rolling_ball_slicewise,
    _rolling_ball_slicewise,
    enhance_microglia_background,
    preprocess_channel,
)


def test_preprocess_preserves_thin_branch_signal() -> None:
    arr = np.zeros((3, 16, 16), dtype=np.float32)
    arr[1, 8, 3:13] = 0.12  # thin microglia-like branch
    arr += 0.01 * np.random.default_rng(0).random(arr.shape, dtype=np.float32)

    channel = ChannelVolume("green", arr, [1, 2, 3], VoxelSpacing())
    cfg = PreprocessConfig(
        denoise_method="anisotropic",
        denoise_strength=0.01,
        preserve_branches=True,
    )
    out = preprocess_channel(channel, cfg)

    # Keep branch-like structure above low-intensity noise floor.
    branch_mean = float(out.data[1, 8, 3:13].mean())
    background_mean = float(out.data[1, 2:6, 2:6].mean())
    assert branch_mean > background_mean


def test_preprocess_green_suppresses_isolated_speckles() -> None:
    arr = np.zeros((3, 24, 24), dtype=np.float32)
    arr[1, 12, 4:20] = 0.2
    arr[1, 5, 5] = 1.0
    arr[1, 6, 19] = 0.95
    arr += 0.01 * np.random.default_rng(7).random(arr.shape, dtype=np.float32)

    channel = ChannelVolume("green", arr, [1, 2, 3], VoxelSpacing())
    cfg = PreprocessConfig(
        denoise_method="anisotropic",
        denoise_strength=0.012,
        preserve_branches=True,
    )
    out = preprocess_channel(channel, cfg)

    branch_signal = float(out.data[1, 12, 8:16].mean())
    speckle_peak = float(max(out.data[1, 5, 5], out.data[1, 6, 19]))
    background = float(out.data[1, 2:6, 2:6].mean())
    assert speckle_peak < 0.40
    assert branch_signal > background


def test_microglia_background_enhancement_preserves_branch_and_soma() -> None:
    yy, xx = np.mgrid[0:48, 0:48]
    smooth_background = (0.08 + 0.04 * (xx / 47.0)).astype(np.float32)
    arr = np.repeat(smooth_background[np.newaxis, ...], 3, axis=0)
    arr[1, 24, 8:40] += 0.16
    arr[1, 20:29, 20:29] += 0.35
    arr += 0.005 * np.random.default_rng(4).random(arr.shape, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)

    cfg = PreprocessConfig(flatfield_sigma_xy=12.0, preserve_branches=True)
    out = enhance_microglia_background(arr, cfg)

    branch_signal = float(out[1, 24, 10:38].mean())
    soma_signal = float(out[1, 22:27, 22:27].mean())
    background_signal = float(out[1, 2:12, 2:12].mean())

    assert out.shape == arr.shape
    assert branch_signal > background_signal * 2.0
    assert soma_signal > branch_signal
    assert background_signal < 0.12


def test_all_microglia_enhancement_methods_run() -> None:
    arr = np.full((2, 32, 32), 0.08, dtype=np.float32)
    arr[:, :, 16:] += 0.04
    arr[0, 16, 6:26] += 0.18
    arr[0, 12:20, 12:20] += 0.32
    arr = np.clip(arr, 0.0, 1.0)
    cfg = PreprocessConfig(flatfield_sigma_xy=10.0, preserve_branches=True)

    for method in (
        "microglia_preserve",
        "imagej_rolling_ball",
        "basic",
        "cidre",
        "white_tophat",
        "clahe",
    ):
        out = enhance_microglia_background(arr, cfg, method=method)
        assert out.shape == arr.shape
        assert out.dtype == np.float32
        assert float(out.max()) > 0.0


def test_imagej_rolling_ball_process_uses_radius_5_and_multiply_1_5() -> None:
    yy, xx = np.mgrid[0:32, 0:32]
    arr = (0.08 + 0.03 * (xx / 31.0)).astype(np.float32)
    stack = np.repeat(arr[np.newaxis, ...], 2, axis=0)
    stack[0, 16, 10:22] += 0.25
    stack[1, 12:20, 12:20] += 0.35
    stack = np.clip(stack, 0.0, 1.0)

    out = _imagej_rolling_ball_slicewise(stack)
    expected = np.clip(_rolling_ball_slicewise(stack, radius=5) * 1.5, 0.0, 1.0)

    assert out.dtype == np.float32
    assert np.allclose(out, expected)


def test_all_microglia_enhancement_methods_preserve_microglia_features() -> None:
    yy, xx = np.mgrid[0:64, 0:64]
    smooth_background = (0.08 + 0.035 * (xx / 63.0) + 0.018 * np.sin(yy / 9.0)).astype(
        np.float32
    )
    arr = np.repeat(smooth_background[np.newaxis, ...], 4, axis=0)
    arr[1:3, 31, 9:55] += 0.14
    arr[2, 22:33, 25:36] += 0.36
    arr += 0.004 * np.random.default_rng(11).random(arr.shape, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)

    cfg = PreprocessConfig(flatfield_sigma_xy=12.0, preserve_branches=True)

    for method in (
        "microglia_preserve",
        "imagej_rolling_ball",
        "basic",
        "cidre",
        "white_tophat",
        "clahe",
    ):
        out = enhance_microglia_background(arr, cfg, method=method)
        branch_signal = float(out[1:3, 31, 12:52].mean())
        soma_signal = float(out[2, 24:31, 27:34].mean())
        background_signal = float(out[:, 3:14, 3:14].mean())

        assert branch_signal > background_signal * 2.0, method
        assert soma_signal >= branch_signal * 0.95, method
