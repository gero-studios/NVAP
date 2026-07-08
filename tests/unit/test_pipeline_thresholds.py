from __future__ import annotations

import numpy as np

from nvap.analysis.microglia_components import compute_component_labels
from nvap.pipeline import default_green_threshold


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


