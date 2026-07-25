from __future__ import annotations

import numpy as np
import pytest

from nvap.config.types import ChannelVolume, PreprocessConfig, VoxelSpacing
from nvap.preprocess import enhancement as enhancement_module
from nvap.preprocess.enhancement import (
    _imagej_rolling_ball_slicewise,
    _reconnect_and_denoise_microglia,
    _rolling_ball_slicewise,
    _restore_soma_interiors_after_imagej_rolling_ball,
    enhance_microglia_background,
    enhance_vasculature_background,
    preprocess_channel,
    wipe_vasculature_blobs,
    wipe_small_specks,
)


def test_wipe_small_specks_removes_specks_keeps_large_structures() -> None:
    vol = np.zeros((6, 40, 40), dtype=np.float32)
    # Large structure (e.g. a vessel/soma) — must survive.
    vol[1:5, 5:25, 5:9] = 0.9
    # A handful of tiny isolated specks — must be wiped.
    vol[0, 1, 1] = 0.95
    vol[3, 30, 32] = 0.8
    vol[4, 35, 10:12] = 0.7  # 2-voxel speck

    out = wipe_small_specks(vol, threshold=0.5, min_voxels=16)

    # Large structure left bit-for-bit untouched.
    assert np.array_equal(out[1:5, 5:25, 5:9], vol[1:5, 5:25, 5:9])
    assert float(out[0, 1, 1]) == 0.0
    assert float(out[3, 30, 32]) == 0.0
    assert float(out[4, 35, 10:12].max()) == 0.0
    # Below-threshold voxels are never considered specks and stay put.
    faint = np.zeros((3, 8, 8), dtype=np.float32)
    faint[1, 4, 4] = 0.2
    kept = wipe_small_specks(faint, threshold=0.5, min_voxels=16)
    assert float(kept[1, 4, 4]) == pytest.approx(0.2)


def test_wipe_small_specks_noop_when_nothing_small() -> None:
    vol = np.zeros((4, 20, 20), dtype=np.float32)
    vol[1:3, 4:16, 4:16] = 0.8  # one big blob, no specks
    out = wipe_small_specks(vol, threshold=0.5, min_voxels=8)
    assert np.array_equal(out, vol)


def test_wipe_vasculature_blobs_clears_compact_dim_halo() -> None:
    vol = np.zeros((3, 40, 40), dtype=np.float32)
    vol[1, 8:15, 8:15] = 0.35  # visible halo at the mesh isovalue
    vol[1, 10:13, 10:13] = 0.85  # compact blob at the stricter red threshold

    out = wipe_vasculature_blobs(vol, threshold=0.6, max_voxels=128)

    assert not np.any(out[1, 8:15, 8:15])


def test_wipe_vasculature_blobs_keeps_elongated_fragments_by_default() -> None:
    vol = np.zeros((3, 120, 120), dtype=np.float32)
    vol[1, 10:16, 10:16] = 0.85
    vol[1, 60, 10:106] = 0.85

    out = wipe_vasculature_blobs(vol, threshold=0.6, max_voxels=128)

    assert float(out[1, 10:16, 10:16].max()) == 0.0
    assert np.array_equal(out[1, 60, 10:106], vol[1, 60, 10:106])


def test_vasculature_enhancement_reduces_uniform_background_and_keeps_vessel() -> None:
    arr = np.full((3, 64, 64), 0.20, dtype=np.float32)
    arr[:, 32, 8:56] = 0.90

    out = enhance_vasculature_background(arr, PreprocessConfig(flatfield_sigma_xy=12.0))

    assert out.shape == arr.shape
    assert out.dtype == np.float32
    assert float(out[:, 32, 8:56].mean()) > float(out[:, :8, :8].mean()) * 3.0


def test_reconnect_and_denoise_bridges_near_and_drops_far() -> None:
    vol = np.zeros((5, 64, 64), dtype=np.float32)
    vol[1:4, 8:18, 8:18] = 0.8  # main cell mass
    vol[1:4, 8:18, 21:27] = 0.5  # nearby fragment (2 px gap -> should reconnect)
    vol[1:4, 48:56, 48:56] = 0.6  # far isolated blob (debris -> should be removed)

    out = _reconnect_and_denoise_microglia(vol, bridge_radius_px=2, far_voxels=26.0)

    assert float(out[1, 8:18, 8:18].mean()) > 0.0  # main kept
    assert float(out[1, 8:18, 21:27].mean()) > 0.0  # near fragment kept
    assert float(out[1, 10:16, 19:21].max()) > 0.0  # gap bridged
    assert float(out[:, 48:56, 48:56].max()) == 0.0  # far debris destroyed


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
        "microscopy_clean",
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

    enhanced = enhance_microglia_background(stack, PreprocessConfig(), method="imagej_rolling_ball")
    assert enhanced.shape == expected.shape
    assert enhanced.dtype == np.float32


def test_imagej_rolling_ball_enhancement_fills_hollow_soma_interiors() -> None:
    yy, xx = np.mgrid[0:72, 0:72]
    background = (0.07 + 0.035 * (xx / 71.0)).astype(np.float32)
    stack = np.repeat(background[np.newaxis, ...], 3, axis=0)
    soma = (yy - 36) ** 2 + (xx - 36) ** 2 <= 9**2
    branch = (np.abs(yy - 36) <= 1) & (xx >= 14) & (xx <= 29)
    stack[1, soma] += 0.48
    stack[1, branch] += 0.22
    stack += 0.003 * np.random.default_rng(17).random(stack.shape, dtype=np.float32)
    stack = np.clip(stack, 0.0, 1.0)

    strict = _imagej_rolling_ball_slicewise(stack, radius=5, multiplier=1.5)
    enhanced = enhance_microglia_background(stack, PreprocessConfig(), method="imagej_rolling_ball")
    core = (yy - 36) ** 2 + (xx - 36) ** 2 <= 4**2
    rim = soma & (~core)
    background_roi = (yy < 14) & (xx < 14)

    strict_core = float(strict[1, core].mean())
    enhanced_core = float(enhanced[1, core].mean())
    enhanced_rim = float(enhanced[1, rim].mean())
    enhanced_background = float(enhanced[1, background_roi].mean())

    assert strict_core < 0.12
    assert enhanced_core > strict_core * 3.0
    assert enhanced_core >= enhanced_rim * 0.75
    assert enhanced_background < enhanced_core * 0.35


def test_imagej_rolling_ball_enhancement_recovers_supported_branches() -> None:
    yy, xx = np.mgrid[0:96, 0:96]
    background = (0.09 + 0.06 * (xx / 95.0) + 0.02 * np.sin(yy / 11.0)).astype(np.float32)
    stack = np.repeat(background[np.newaxis, ...], 3, axis=0)
    soma = (yy - 46) ** 2 + (xx - 52) ** 2 <= 10**2
    branch = (np.abs(yy - 46) <= 1) & (xx >= 12) & (xx <= 44)
    stack[1, soma] += 0.42
    stack[1, branch] += 0.11
    stack += 0.006 * np.random.default_rng(21).random(stack.shape, dtype=np.float32)
    stack = np.clip(stack, 0.0, 1.0)

    strict = _imagej_rolling_ball_slicewise(stack, radius=5, multiplier=1.5)
    restored = _restore_soma_interiors_after_imagej_rolling_ball(stack, strict)
    enhanced = enhance_microglia_background(stack, PreprocessConfig(), method="imagej_rolling_ball")
    background_roi = np.s_[:, 4:18, 4:18]

    restored_branch = float(restored[1, branch].mean())
    enhanced_branch = float(enhanced[1, branch].mean())
    enhanced_background = float(enhanced[background_roi].mean())

    assert enhanced_branch > restored_branch * 1.5
    assert enhanced_background < 0.01


def test_imagej_rolling_ball_enhancement_fades_isolated_speckles() -> None:
    yy, xx = np.mgrid[0:88, 0:88]
    background = (0.065 + 0.030 * (xx / 87.0) + 0.010 * np.sin(yy / 11.0)).astype(
        np.float32
    )
    stack = np.repeat(background[np.newaxis, ...], 3, axis=0)
    soma = (yy - 44) ** 2 + (xx - 46) ** 2 <= 8**2
    branch = (np.abs(yy - 44) <= 1) & (xx >= 14) & (xx <= 38)
    branch |= (np.abs((yy - 44) - 0.45 * (xx - 46)) <= 1.0) & (xx >= 50) & (xx <= 78)
    speckles = np.zeros_like(background, dtype=bool)
    speckles[15, 19] = True
    speckles[24, 70] = True
    speckles[68, 31] = True

    stack[1, soma] += 0.46
    stack[1, branch] += 0.16
    stack[1, speckles] += 0.70
    stack += 0.003 * np.random.default_rng(22).random(stack.shape, dtype=np.float32)
    stack = np.clip(stack, 0.0, 1.0)

    strict = _imagej_rolling_ball_slicewise(stack, radius=5, multiplier=1.5)
    enhanced = enhance_microglia_background(stack, PreprocessConfig(), method="imagej_rolling_ball")
    background_roi = (yy < 12) & (xx < 12)

    strict_branch = float(strict[1, branch].mean())
    enhanced_branch = float(enhanced[1, branch].mean())
    enhanced_soma = float(enhanced[1, soma].mean())
    enhanced_background = float(enhanced[1, background_roi].mean())
    strict_speckles = float(strict[1, speckles].max())
    enhanced_speckles = float(enhanced[1, speckles].max())

    assert enhanced_branch > strict_branch * 1.8
    assert enhanced_soma > enhanced_branch
    assert enhanced_background < enhanced_branch * 0.35
    assert enhanced_speckles < strict_speckles * 0.35


def test_imagej_restore_uses_single_distance_transform_per_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    yy, xx = np.mgrid[0:96, 0:96]
    base = np.zeros((2, 96, 96), dtype=np.float32)
    soma_a = (yy - 28) ** 2 + (xx - 28) ** 2 <= 8**2
    soma_b = (yy - 68) ** 2 + (xx - 68) ** 2 <= 10**2
    base[0, soma_a] = 0.85
    base[1, soma_b] = 0.90
    enhanced = _imagej_rolling_ball_slicewise(base, radius=5, multiplier=1.5)

    original_edt = enhancement_module.ndi.distance_transform_edt
    calls = 0

    def counting_edt(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_edt(*args, **kwargs)

    monkeypatch.setattr(enhancement_module.ndi, "distance_transform_edt", counting_edt)
    restored = enhancement_module._restore_soma_interiors_after_imagej_rolling_ball(base, enhanced)

    assert restored.shape == base.shape
    assert calls == base.shape[0]


def test_microscopy_clean_highlights_soma_branches_and_reduces_speckles() -> None:
    yy, xx = np.mgrid[0:96, 0:96]
    background = (0.09 + 0.055 * (xx / 95.0) + 0.020 * np.sin(yy / 10.0)).astype(np.float32)
    stack = np.repeat(background[np.newaxis, ...], 3, axis=0)
    soma = (yy - 47) ** 2 + (xx - 50) ** 2 <= 9**2
    branch = (np.abs(yy - 47) <= 1) & (xx >= 12) & (xx <= 42)
    branch |= (np.abs((yy - 47) + 0.35 * (xx - 50)) <= 1.0) & (xx >= 55) & (xx <= 82)
    speckles = np.zeros_like(background, dtype=bool)
    speckles[18, 22] = True
    speckles[26, 74] = True
    speckles[75, 34] = True

    stack[1, soma] += 0.44
    stack[1, branch] += 0.14
    stack[1, speckles] += 0.75
    stack += 0.004 * np.random.default_rng(33).random(stack.shape, dtype=np.float32)
    stack = np.clip(stack, 0.0, 1.0)

    out = enhance_microglia_background(stack, PreprocessConfig(), method="microscopy_clean")
    background_roi = (yy < 14) & (xx < 14)

    branch_signal = float(out[1, branch].mean())
    soma_signal = float(out[1, soma].mean())
    background_signal = float(out[1, background_roi].mean())
    speckle_peak = float(out[1, speckles].max())

    assert branch_signal > background_signal * 15.0
    assert soma_signal > branch_signal
    assert speckle_peak < soma_signal * 0.55


def test_microscopy_clean_keeps_soma_body_solid_and_rejects_thin_bar() -> None:
    yy, xx = np.mgrid[0:96, 0:96]
    background = (0.09 + 0.02 * np.sin(yy / 10.0)).astype(np.float32)

    # A compact soma blob must survive the morphological opening as a solid body.
    blob = np.repeat(background[np.newaxis, ...], 3, axis=0).copy()
    disk_mask = (yy - 48) ** 2 + (xx - 48) ** 2 <= 8**2
    blob[1, disk_mask] += 0.45
    blob += 0.004 * np.random.default_rng(1).random(blob.shape, dtype=np.float32)
    out_blob = enhance_microglia_background(
        np.clip(blob, 0.0, 1.0), PreprocessConfig(), method="microscopy_clean"
    )
    enhanced = out_blob[1] > 0.1
    coverage = float(np.count_nonzero(enhanced & disk_mask)) / float(np.count_nonzero(disk_mask))
    assert coverage >= 0.8  # opening must not erode the compact soma away

    # A thin bright bar at the same peak intensity must stay process-like (thin),
    # not be fabricated into a wide soma-shaped solid.
    bar = np.repeat(background[np.newaxis, ...], 3, axis=0).copy()
    line = (np.abs(yy - 48) <= 1) & (xx >= 14) & (xx <= 82)
    bar[1, line] += 0.45
    bar += 0.004 * np.random.default_rng(2).random(bar.shape, dtype=np.float32)
    out_bar = enhance_microglia_background(
        np.clip(bar, 0.0, 1.0), PreprocessConfig(), method="microscopy_clean"
    )
    bar_on = int(np.count_nonzero(out_bar[1] > 0.1))
    # The bar should not balloon into a blob many times its own footprint.
    assert bar_on < int(np.count_nonzero(line)) * 4


def test_microscopy_clean_bridges_faint_process_gaps_without_adding_speckles() -> None:
    import scipy.ndimage as ndi

    yy, xx = np.mgrid[0:96, 0:96]
    background = (0.09 + 0.02 * np.sin(yy / 9.0)).astype(np.float32)
    stack = np.repeat(background[np.newaxis, ...], 3, axis=0).copy()

    soma = (yy - 20) ** 2 + (xx - 20) ** 2 <= 6**2
    stack[1, soma] += 0.45
    # Realistic process width (~3 px ≈ 1 µm). The grain de-speckle pass uses a
    # radius-1 opening that intentionally drops sub-resolution 1-px features
    # (noise grains), so the process is modeled at its true ~3-px width.
    line = (np.abs(yy - 20) <= 1) & (xx >= 26) & (xx <= 86)
    stack[1, line] += 0.16
    faint = line & (xx >= 46) & (xx <= 66)  # middle segment dims toward background
    stack[1, faint] -= 0.115
    speckles = np.zeros_like(background, dtype=bool)
    speckles[70, 30] = speckles[80, 75] = speckles[60, 55] = True
    stack[1, speckles] += 0.75
    stack += 0.004 * np.random.default_rng(7).random(stack.shape, dtype=np.float32)

    out = enhance_microglia_background(
        np.clip(stack, 0.0, 1.0), PreprocessConfig(), method="microscopy_clean"
    )[1]

    process_region = (np.abs(yy - 20) <= 1) & (xx >= 26) & (xx <= 86)
    on = out > 0.05
    labels, count = ndi.label(on & process_region)
    largest_frac = 0.0
    if count > 0:
        sizes = np.bincount(labels.ravel())[1:]
        largest_frac = float(sizes.max()) / float(np.count_nonzero(process_region))

    # The dimming middle no longer fragments the process away (was ~0.13 before).
    assert largest_frac >= 0.4
    assert float(out[faint].mean()) > 0.05
    # Isolated speckles are still suppressed (no ridge to grow along).
    assert float(out[speckles].max()) < float(out[soma].mean()) * 0.55


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
        "microscopy_clean",
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
        assert soma_signal >= branch_signal * 0.85, method
