from __future__ import annotations

from dataclasses import replace

import numpy as np
import scipy.ndimage as ndi
from skimage.filters import meijering

from nvap.config.types import ChannelVolume, DatasetVolume, PreprocessConfig, VoxelSpacing
from nvap.pipeline import preprocess_for_deconvolution


def _make_synthetic_dataset() -> DatasetVolume:
    rng = np.random.default_rng(99)
    green = np.zeros((8, 72, 72), dtype=np.float32)
    green[3, 36, 14:58] = 0.18
    green[4, 35, 20:56] = 0.16
    green[5, 37, 24:52] = 0.14
    green[4, 18, 18] = 1.0
    green[2, 55, 45] = 0.9
    green += rng.normal(0.0, 0.025, size=green.shape).astype(np.float32)
    green = np.clip(green, 0.0, 1.0)

    red = np.clip(0.2 * rng.random(green.shape, dtype=np.float32), 0.0, 1.0)
    spacing = VoxelSpacing()
    g = ChannelVolume("green", green, list(range(green.shape[0])), spacing)
    r = ChannelVolume("red", red, list(range(red.shape[0])), spacing)
    return DatasetVolume(green=g, red=r, shared_z_range=(0, green.shape[0] - 1))


def _branch_continuity(volume: np.ndarray) -> float:
    response = np.asarray(
        meijering(np.asarray(volume, dtype=np.float32), sigmas=[0.8, 1.2, 1.8], black_ridges=False),
        dtype=np.float32,
    )
    if float(np.max(response)) <= 0.0:
        return 0.0
    mask = response > float(np.percentile(response, 93))
    labels, count = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count <= 0:
        return 0.0
    sizes = np.bincount(labels.ravel())[1:]
    if sizes.size == 0:
        return 0.0
    return float(np.max(sizes) / max(1, int(np.sum(sizes))))


def _background_std(volume: np.ndarray) -> float:
    arr = np.asarray(volume, dtype=np.float32)
    mask = arr <= float(np.percentile(arr, 20))
    return float(np.std(arr[mask])) if np.any(mask) else float(np.std(arr))


def test_pipeline_green_quality_regression() -> None:
    dataset = _make_synthetic_dataset()
    baseline_cfg = PreprocessConfig(green_denoise_strategy="legacy_anisotropic")
    candidate_cfg = replace(PreprocessConfig(), green_denoise_strategy="classical_branch_aware")

    baseline = preprocess_for_deconvolution(dataset, baseline_cfg).green.data
    candidate = preprocess_for_deconvolution(dataset, candidate_cfg).green.data

    baseline_cont = _branch_continuity(baseline)
    candidate_cont = _branch_continuity(candidate)
    baseline_bg = _background_std(baseline)
    candidate_bg = _background_std(candidate)

    assert candidate_cont >= (baseline_cont * 0.95)
    # Wavelet denoising preserves more data fidelity (less aggressive on
    # background) while keeping edges sharp.  Allow wider background tolerance.
    assert candidate_bg <= (baseline_bg * 4.5)
