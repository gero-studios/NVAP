from __future__ import annotations

import numpy as np

from nvap.analysis.microglia_components import compute_component_labels
from nvap.config.types import ChannelVolume, DatasetVolume, PreprocessConfig, VoxelSpacing
from nvap.pipeline import default_green_threshold
from nvap.ui.main_window import MainWindow


def _make_dim_microglia_volume() -> np.ndarray:
    arr = np.zeros((6, 64, 64), dtype=np.float32)
    centers = [(2, 18, 20), (3, 34, 30), (4, 46, 44)]
    for z, y, x in centers:
        arr[z, y, x] = 0.28
        arr[z, y + 1, x] = 0.24
        arr[z, y, x + 1] = 0.22
        arr[z, y, x - 1] = 0.20
        arr[z, y, x + 2 : x + 8] = 0.11
        arr[z, y - 3 : y, x] = 0.10
    rng = np.random.default_rng(5)
    arr += rng.normal(0.0, 0.01, size=arr.shape).astype(np.float32)
    return np.clip(arr, 0.0, 1.0)


def test_default_green_threshold_is_branch_aware_for_dim_microglia() -> None:
    arr = _make_dim_microglia_volume()

    threshold = default_green_threshold(arr)
    assert 0.03 <= threshold < 0.5

    _, order_adaptive, _ = compute_component_labels(
        arr,
        threshold=threshold,
        min_voxels=6,
        max_components=16,
        smooth_sigma=(0.0, 0.0, 0.0),
        branch_sensitivity=1.0,
    )
    _, order_fixed, _ = compute_component_labels(
        arr,
        threshold=0.5,
        min_voxels=6,
        max_components=16,
        smooth_sigma=(0.0, 0.0, 0.0),
        branch_sensitivity=1.0,
    )

    assert len(order_adaptive) >= 3
    assert len(order_fixed) == 0


def test_reprocess_result_recomputes_thresholds_from_processed_green_volume() -> None:
    spacing = VoxelSpacing()
    green = _make_dim_microglia_volume()
    red = np.zeros_like(green, dtype=np.float32)
    dataset = DatasetVolume(
        green=ChannelVolume("green", green, list(range(green.shape[0])), spacing),
        red=ChannelVolume("red", red, list(range(red.shape[0])), spacing),
        shared_z_range=(0, green.shape[0] - 1),
    )

    result = MainWindow._build_process_task_result(dataset, PreprocessConfig())

    assert result.processed_dataset is dataset
    assert result.visual_dataset.green.data.shape == dataset.green.data.shape
    assert result.visual_dataset.red.data.shape == dataset.red.data.shape
    assert 0.03 <= result.threshold_green < 0.5
    assert 0.0 <= result.threshold_red <= 1.0
