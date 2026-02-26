from __future__ import annotations

import os
import time

import numpy as np
import pytest

from nvap.config.types import ChannelVolume, PreprocessConfig, VoxelSpacing
from nvap.preprocess.enhancement import preprocess_channel


@pytest.mark.integration
def test_default_green_denoise_runtime_guard() -> None:
    if os.environ.get("NVAP_RUN_PERF", "0") != "1":
        pytest.skip("Set NVAP_RUN_PERF=1 to run performance guard.")

    rng = np.random.default_rng(5)
    arr = np.zeros((24, 192, 192), dtype=np.float32)
    arr[12, 96, 24:168] = 0.18
    arr += rng.normal(0.0, 0.02, size=arr.shape).astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    channel = ChannelVolume("green", arr, list(range(arr.shape[0])), VoxelSpacing())

    baseline_cfg = PreprocessConfig(green_denoise_strategy="legacy_anisotropic")
    candidate_cfg = PreprocessConfig(green_denoise_strategy="classical_branch_aware")

    t0 = time.perf_counter()
    preprocess_channel(channel, baseline_cfg)
    baseline_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    preprocess_channel(channel, candidate_cfg)
    candidate_sec = time.perf_counter() - t0

    assert candidate_sec <= (baseline_sec * 1.6)
