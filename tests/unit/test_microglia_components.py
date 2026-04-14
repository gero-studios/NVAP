from __future__ import annotations

import numpy as np

from nvap.analysis import microglia_components
from nvap.analysis.microglia_components import (
    compute_component_labels,
    filter_components_by_preferred_voxel_floor,
    isolate_component,
)


def test_mask_bounds_avoids_full_nonzero_coordinate_allocation(monkeypatch) -> None:
    mask = np.zeros((5, 8, 9), dtype=bool)
    mask[1:4, 2:6, 3:8] = True

    def _fail_nonzero(*_args, **_kwargs):
        raise AssertionError("_mask_bounds should not call np.nonzero")

    monkeypatch.setattr(microglia_components.np, "nonzero", _fail_nonzero)
    bounds = microglia_components._mask_bounds(mask)

    assert bounds == (slice(1, 4), slice(2, 6), slice(3, 8))


def test_detect_soma_blobs_updates_peak_mask_in_place(monkeypatch) -> None:
    arr = np.zeros((3, 16, 16), dtype=np.float32)
    arr[1, 8, 8] = 1.0
    arr[1, 8, 9] = 0.9

    real_subtract = np.subtract
    real_greater_equal = np.greater_equal
    saw_in_place_compare = False

    def _subtract(*args, **kwargs):
        return real_subtract(*args, **kwargs)

    def _greater_equal(a, b, *args, **kwargs):
        nonlocal saw_in_place_compare
        out = kwargs.get("out")
        where = kwargs.get("where")
        if (
            isinstance(out, np.ndarray)
            and out.dtype == np.bool_
            and out.shape == np.asarray(a).shape
            and where is out
        ):
            saw_in_place_compare = True
        return real_greater_equal(a, b, *args, **kwargs)

    monkeypatch.setattr(microglia_components.np, "subtract", _subtract)
    monkeypatch.setattr(microglia_components.np, "greater_equal", _greater_equal)

    seed = microglia_components._detect_soma_blobs(
        arr,
        threshold=0.1,
        branch_sensitivity=1.0,
        spacing=(1.0, 1.0, 1.0),
    )

    assert np.any(seed)
    assert saw_in_place_compare


def test_detect_soma_blobs_avoids_scipy_maximum_position(monkeypatch) -> None:
    arr = np.zeros((4, 32, 32), dtype=np.float32)
    arr[1, 8, 8] = 0.8
    arr[2, 22, 22] = 0.7

    def _fail_maximum_position(*_args, **_kwargs):
        raise AssertionError("large-volume seed detection should not call maximum_position")

    monkeypatch.setattr(microglia_components.ndi, "maximum_position", _fail_maximum_position)

    seed = microglia_components._detect_soma_blobs(
        arr,
        threshold=0.1,
        branch_sensitivity=1.0,
        spacing=(1.0, 1.0, 1.0),
    )

    assert int(np.count_nonzero(seed)) >= 1


def test_compute_component_labels_orders_by_size_desc() -> None:
    arr = np.zeros((4, 16, 16), dtype=np.float32)
    arr[1:3, 8, 2:12] = 0.2  # large component
    arr[2, 3:5, 13] = 0.25   # small component
    labels, order, sizes = compute_component_labels(arr, threshold=0.15)

    assert labels.shape == arr.shape
    assert len(order) == 2
    assert int(sizes[int(order[0])]) >= int(sizes[int(order[1])])


def test_isolate_component_keeps_only_selected_microglia() -> None:
    arr = np.zeros((3, 12, 12), dtype=np.float32)
    arr[1, 4, 2:8] = 0.18
    arr[1, 9, 8:11] = 0.21
    labels, order, _ = compute_component_labels(arr, threshold=0.15)
    isolated = isolate_component(arr, labels, int(order[0]))

    assert isolated.shape == arr.shape
    assert float(np.count_nonzero(isolated)) > 0
    # Some voxels from the original should be removed when isolating one component.
    assert int(np.count_nonzero(isolated)) < int(np.count_nonzero(arr))


def test_component_filtering_avoids_noise_explosion() -> None:
    rng = np.random.default_rng(88)
    arr = np.zeros((8, 64, 64), dtype=np.float32)
    # Two meaningful components.
    arr[3:6, 20, 10:30] = 0.2
    arr[2:5, 42, 34:54] = 0.22
    # A lot of isolated noise voxels above threshold.
    noise_mask = rng.random(arr.shape) < 0.02
    arr[noise_mask] = np.maximum(arr[noise_mask], 0.18)

    labels, order, sizes = compute_component_labels(
        arr,
        threshold=0.15,
        min_voxels=20,
        max_components=8,
    )
    assert labels.shape == arr.shape
    assert len(order) <= 8
    assert len(order) >= 2
    assert int(sizes[int(order[0])]) >= int(sizes[int(order[-1])])


def test_component_cap_limits_result_count() -> None:
    rng = np.random.default_rng(9)
    arr = np.zeros((5, 80, 80), dtype=np.float32)
    # Many tiny disconnected components.
    points = rng.choice(arr.size, size=400, replace=False)
    arr.reshape(-1)[points] = 0.2

    _, order, _ = compute_component_labels(
        arr,
        threshold=0.15,
        min_voxels=1,
        max_components=32,
        smooth_sigma=(0.0, 0.0, 0.0),
    )
    assert len(order) <= 32


def test_giant_component_filtered_keeps_cell_sized_components() -> None:
    arr = np.zeros((6, 64, 64), dtype=np.float32)
    # Large connected blob that should not be treated as one microglia cell.
    arr[:, 8:56, 8:56] = 0.19
    # Smaller disconnected bright structure that should remain selectable.
    arr[2:5, 2, 5:25] = 0.24

    _, order, sizes = compute_component_labels(
        arr,
        threshold=0.15,
        min_voxels=20,
        max_components=8,
    )
    assert len(order) == 1
    assert int(sizes[int(order[0])]) >= 50


def test_hysteresis_preserves_connected_low_intensity_branches() -> None:
    arr = np.zeros((5, 48, 48), dtype=np.float32)
    # Bright core (soma-like)
    arr[2, 24, 24] = 0.36
    arr[2, 24, 25] = 0.32
    arr[2, 25, 24] = 0.31
    # Lower-intensity branches connected to core
    arr[2, 24, 26:40] = 0.12
    arr[2, 20:24, 24] = 0.11
    # Isolated dim speckle should not survive as its own component
    arr[1, 4, 4] = 0.11

    labels, order, sizes = compute_component_labels(
        arr,
        threshold=0.18,
        min_voxels=6,
        max_components=8,
    )
    assert len(order) == 1
    comp_id = int(order[0])
    assert int(sizes[comp_id]) >= 18
    assert int(labels[2, 24, 36]) == comp_id
    assert int(labels[1, 4, 4]) == 0


def test_branch_sensitivity_slider_affects_branch_retention() -> None:
    arr = np.zeros((4, 40, 40), dtype=np.float32)
    arr[1, 20, 20] = 0.35
    arr[1, 20, 21] = 0.32
    arr[1, 20, 22:31] = 0.11

    labels_low, order_low, _ = compute_component_labels(
        arr,
        threshold=0.18,
        min_voxels=4,
        branch_sensitivity=0.5,
    )
    labels_high, order_high, _ = compute_component_labels(
        arr,
        threshold=0.18,
        min_voxels=4,
        branch_sensitivity=1.8,
    )

    assert len(order_high) >= 1
    assert int(np.count_nonzero(labels_high)) >= int(np.count_nonzero(labels_low))


def test_component_requires_visible_above_threshold_voxels() -> None:
    arr = np.zeros((3, 24, 24), dtype=np.float32)
    arr[1, 12, 12] = 0.25
    arr[1, 12, 13:20] = 0.09  # connected but below render threshold

    _, order, _ = compute_component_labels(
        arr,
        threshold=0.1,
        min_voxels=6,
        branch_sensitivity=1.2,
    )
    assert len(order) == 0


def test_nearby_cells_not_merged_at_low_threshold() -> None:
    """Many cells with bright cores and dim branches should remain separate.

    Before the fix, binary_propagation merged nearby cells whose low-intensity
    regions touched, producing ~4 components instead of 30+.
    """
    rng = np.random.default_rng(42)
    arr = np.zeros((10, 128, 128), dtype=np.float32)

    # Plant 36 well-separated microglia-like cells with bright cores and dim
    # surrounding branches.  Cells are laid out in a 6x6 grid with spacing 20,
    # starting at (10, 10).  The branch halos may touch at low thresholds but
    # the cores are distinct.
    n_planted = 0
    for row in range(6):
        for col in range(6):
            cy, cx = 10 + row * 20, 10 + col * 20
            z0 = rng.integers(3, 7)
            # Bright core (soma)
            arr[z0, cy, cx] = 0.6 + rng.random() * 0.15
            arr[z0, cy + 1, cx] = 0.45 + rng.random() * 0.1
            arr[z0, cy, cx + 1] = 0.42 + rng.random() * 0.1
            arr[z0, cy - 1, cx] = 0.40 + rng.random() * 0.08
            arr[z0, cy, cx - 1] = 0.38 + rng.random() * 0.08
            # Branch-like extensions (dimmer)
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    iy, ix = cy + dy, cx + dx
                    if 0 <= iy < 128 and 0 <= ix < 128 and arr[z0, iy, ix] < 0.01:
                        dist = abs(dy) + abs(dx)
                        if dist <= 5:
                            arr[z0, iy, ix] = max(0.0, 0.18 - 0.025 * dist + rng.random() * 0.03)
            n_planted += 1

    _, order, _ = compute_component_labels(
        arr,
        threshold=0.12,
        min_voxels=4,
        max_components=512,
        smooth_sigma=(0.0, 0.0, 0.0),
        branch_sensitivity=1.0,
    )
    # Should find at least 30 of the 36 planted cells as separate components.
    assert len(order) >= 30, (
        f"Expected >=30 components from {n_planted} planted cells, got {len(order)}"
    )


def test_single_soma_not_split_by_patchy_bright_core() -> None:
    arr = np.zeros((5, 64, 64), dtype=np.float32)
    z = 2

    # One soma-like blob with a moderate-intensity body and two bright islands.
    arr[z, 28:37, 28:37] = 0.34
    arr[z, 30:32, 30:32] = 0.64
    arr[z, 33:35, 33:35] = 0.66
    arr[z, 31:34, 31:34] = np.maximum(arr[z, 31:34, 31:34], 0.48)

    # Dim branches should stay attached to the single soma.
    arr[z, 32, 37:47] = 0.20
    arr[z, 20:28, 32] = 0.19

    labels, order, sizes = compute_component_labels(
        arr,
        threshold=0.18,
        min_voxels=20,
        max_components=8,
        smooth_sigma=(0.0, 0.0, 0.0),
    )

    assert len(order) == 1
    comp_id = int(order[0])
    assert int(sizes[comp_id]) >= 90
    assert int(labels[z, 32, 43]) == comp_id
    assert int(labels[z, 24, 32]) == comp_id


def test_single_soma_with_two_close_lobes_stays_one_component() -> None:
    arr = np.zeros((5, 72, 72), dtype=np.float32)
    z = 2

    # One soma with two bright lobes and a moderate-intensity body between
    # them. These should not be treated as two neighboring microglia.
    arr[z, 31:40, 29:42] = 0.30
    arr[z, 33:37, 30:34] = 0.68
    arr[z, 33:37, 37:41] = 0.65
    arr[z, 34:36, 34:38] = np.maximum(arr[z, 34:36, 34:38], 0.42)
    arr[z - 1, 34:37, 33:39] = 0.24
    arr[z + 1, 34:37, 33:39] = 0.24
    arr[z, 35, 42:50] = 0.16

    labels, order, sizes = compute_component_labels(
        arr,
        threshold=0.14,
        min_voxels=20,
        max_components=8,
        smooth_sigma=(0.0, 0.0, 0.0),
        branch_sensitivity=1.0,
    )

    assert len(order) == 1
    comp_id = int(order[0])
    assert int(sizes[comp_id]) >= 100
    assert int(labels[z, 35, 31]) == comp_id
    assert int(labels[z, 35, 39]) == comp_id


def test_nearby_somas_connected_by_dim_bridge_stay_separate() -> None:
    arr = np.zeros((5, 80, 80), dtype=np.float32)
    z = 2

    # Two nearby somas with broad halos and bright cores.
    left_x, right_x = 30, 40
    arr[z, 24:37, left_x - 6:left_x + 7] = 0.28
    arr[z, 27:34, left_x - 3:left_x + 4] = 0.56
    arr[z, 24:37, right_x - 6:right_x + 7] = np.maximum(
        arr[z, 24:37, right_x - 6:right_x + 7], 0.28
    )
    arr[z, 27:34, right_x - 3:right_x + 4] = np.maximum(
        arr[z, 27:34, right_x - 3:right_x + 4], 0.56
    )

    # Dim bridge should not collapse both somas into one component.
    arr[z, 30, 33:37] = np.maximum(arr[z, 30, 33:37], 0.16)

    labels, order, _ = compute_component_labels(
        arr,
        threshold=0.12,
        min_voxels=40,
        max_components=16,
        smooth_sigma=(0.0, 0.0, 0.0),
    )

    assert len(order) >= 2
    left_id = int(labels[z, 30, left_x])
    right_id = int(labels[z, 30, right_x])
    assert left_id > 0
    assert right_id > 0
    assert left_id != right_id


def test_faint_bridge_branch_gets_single_soma_owner() -> None:
    arr = np.zeros((5, 96, 96), dtype=np.float32)
    z = 2

    # Two clear somas with one faint branch-like bridge between them. The
    # left soma has a stronger local connection to the bridge.
    arr[z, 42:51, 24:33] = 0.54
    arr[z, 42:51, 64:73] = 0.42
    arr[z, 46, 33:64] = 0.15
    arr[z, 45:48, 33:40] = np.maximum(arr[z, 45:48, 33:40], 0.22)

    labels, order, _ = compute_component_labels(
        arr,
        threshold=0.12,
        min_voxels=20,
        max_components=16,
        smooth_sigma=(0.0, 0.0, 0.0),
        branch_sensitivity=1.4,
    )

    assert len(order) >= 2
    left_id = int(labels[z, 46, 28])
    right_id = int(labels[z, 46, 68])
    bridge_ids = set(np.unique(labels[z, 46, 33:64]).tolist())
    bridge_ids.discard(0)

    assert left_id > 0
    assert right_id > 0
    assert left_id != right_id
    assert len(bridge_ids) == 1


def test_preferred_voxel_floor_drops_small_components_when_large_one_exists() -> None:
    labels = np.zeros((1, 4, 5), dtype=np.int32)
    labels[0, 0, 0:2] = 1
    labels[0, 1:4, 1:5] = 2
    order = np.asarray([2, 1], dtype=np.int32)
    sizes = np.asarray([0, 4_000, 18_000], dtype=np.int64)

    filtered_labels, filtered_order, filtered_sizes = filter_components_by_preferred_voxel_floor(
        labels,
        order,
        sizes,
    )

    assert filtered_order.tolist() == [1]
    assert int(filtered_sizes[1]) == 18_000
    assert int(np.max(filtered_labels)) == 1
    assert int(filtered_labels[0, 0, 0]) == 0
    assert int(filtered_labels[0, 2, 2]) == 1


def test_preferred_voxel_floor_preserves_small_components_when_no_large_one_exists() -> None:
    labels = np.zeros((1, 3, 4), dtype=np.int32)
    labels[0, 0, 0:2] = 1
    labels[0, 1:3, 2:4] = 2
    order = np.asarray([2, 1], dtype=np.int32)
    sizes = np.asarray([0, 4_200, 9_000], dtype=np.int64)

    filtered_labels, filtered_order, filtered_sizes = filter_components_by_preferred_voxel_floor(
        labels,
        order,
        sizes,
    )

    assert np.array_equal(filtered_labels, labels)
    assert np.array_equal(filtered_order, order)
    assert np.array_equal(filtered_sizes, sizes)


def test_preferred_voxel_floor_demotes_likely_merged_giant_components() -> None:
    labels = np.zeros((1, 1, 10), dtype=np.int32)
    for component_id in range(1, 11):
        labels[0, 0, component_id - 1] = component_id
    order = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    sizes = np.asarray(
        [0, 240_000, 190_000, 31_000, 30_500, 29_800, 29_100, 28_700, 28_000, 27_600, 27_000],
        dtype=np.int64,
    )

    filtered_labels, filtered_order, filtered_sizes = filter_components_by_preferred_voxel_floor(
        labels,
        order,
        sizes,
    )

    assert filtered_order.tolist() == list(range(1, 11))
    assert int(filtered_sizes[1]) == 31_000
    assert int(filtered_sizes[2]) == 30_500
    assert int(filtered_sizes[9]) == 240_000
    assert int(filtered_sizes[10]) == 190_000
    assert int(filtered_labels[0, 0, 0]) == 9
    assert int(filtered_labels[0, 0, 2]) == 1
