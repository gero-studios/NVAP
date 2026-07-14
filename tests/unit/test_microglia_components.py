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


def test_distance_peak_centers_uses_component_local_edt(monkeypatch) -> None:
    mask = np.zeros((12, 120, 120), dtype=bool)
    mask[1:3, 5:9, 5:9] = True
    mask[9:11, 110:114, 112:116] = True

    calls: list[tuple[int, int, int]] = []
    real_edt = microglia_components.ndi.distance_transform_edt

    def _track_edt(input_mask, *args, **kwargs):
        arr = np.asarray(input_mask)
        calls.append(tuple(int(v) for v in arr.shape))
        return real_edt(input_mask, *args, **kwargs)

    monkeypatch.setattr(microglia_components.ndi, "distance_transform_edt", _track_edt)
    centers = microglia_components._distance_peak_centers_from_soma_mask(
        mask,
        np.asarray((1.0, 1.0, 1.0), dtype=np.float32),
        max_candidates=64,
    )

    assert centers.shape[0] >= 2
    assert len(calls) >= 2
    assert all(int(np.prod(shape)) < int(mask.size) for shape in calls)


def test_distance_peak_centers_downsamples_large_component(monkeypatch) -> None:
    mask = np.ones((12, 64, 64), dtype=bool)
    calls: list[tuple[int, int, int]] = []
    real_edt = microglia_components.ndi.distance_transform_edt

    monkeypatch.setattr(microglia_components, "_MAX_DISTANCE_EDT_VOXELS", 512)

    def _track_edt(input_mask, *args, **kwargs):
        arr = np.asarray(input_mask)
        calls.append(tuple(int(v) for v in arr.shape))
        return real_edt(input_mask, *args, **kwargs)

    monkeypatch.setattr(microglia_components.ndi, "distance_transform_edt", _track_edt)
    centers = microglia_components._distance_peak_centers_from_soma_mask(
        mask,
        np.asarray((1.0, 1.0, 1.0), dtype=np.float32),
        max_candidates=128,
    )

    assert centers.shape[0] >= 1
    assert len(calls) >= 1
    assert max(int(np.prod(shape)) for shape in calls) <= 512


def test_distance_peak_centers_bypasses_labeling_for_oversized_masks(monkeypatch) -> None:
    mask = np.ones((12, 64, 64), dtype=bool)
    monkeypatch.setattr(microglia_components, "_MAX_DISTANCE_COMPONENT_LABEL_VOXELS", 256)

    def _fail_label(*_args, **_kwargs):
        raise AssertionError("oversized-mask path should bypass ndi.label")

    monkeypatch.setattr(microglia_components.ndi, "label", _fail_label)

    centers = microglia_components._distance_peak_centers_from_soma_mask(
        mask,
        np.asarray((1.0, 1.0, 1.0), dtype=np.float32),
        max_candidates=64,
    )

    assert centers.shape[0] >= 1


def test_merge_close_soma_marker_positions_uses_fast_path_for_large_volume(monkeypatch) -> None:
    working = np.zeros((2, 16, 16), dtype=np.float32)
    positions: list[tuple[int, int, int]] = []
    for y in range(2, 14, 2):
        for x in range(2, 14, 2):
            working[1, y, x] = 0.6 + (0.01 * (x + y))
            positions.append((1, y, x))

    monkeypatch.setattr(microglia_components, "_LARGE_VOLUME_VOXELS", 1)

    def _fail_label(*_args, **_kwargs):
        raise AssertionError("fast-path merge should avoid ndi.label on large volumes")

    monkeypatch.setattr(microglia_components.ndi, "label", _fail_label)
    merged = microglia_components._merge_close_soma_marker_positions(
        positions,
        working,
        low_floor=0.1,
        high_t=0.3,
        branch_sensitivity=1.0,
        spacing_zyx=np.asarray((1.0, 1.0, 1.0), dtype=np.float32),
    )

    assert 1 <= len(merged) <= len(positions)


def test_segment_soma_markers_skips_edt_for_large_candidates(monkeypatch) -> None:
    working = np.zeros((3, 12, 12), dtype=np.float32)
    working[:, 2:10, 2:10] = 0.45
    finite = np.isfinite(working)
    seed = np.zeros_like(working, dtype=bool)
    seed[1, 4, 4] = True
    seed[1, 7, 7] = True

    monkeypatch.setattr(microglia_components, "_MAX_SOMA_EDT_VOXELS", 8)
    monkeypatch.setattr(microglia_components, "_MAX_SOMA_CORE_LABEL_VOXELS", 8)

    def _fail_edt(*_args, **_kwargs):
        raise AssertionError("large-candidate path should skip full-resolution EDT")

    monkeypatch.setattr(microglia_components.ndi, "distance_transform_edt", _fail_edt)

    labels = microglia_components._segment_soma_markers_from_reduced_threshold(
        seed,
        working,
        finite,
        low_floor=0.1,
        high_t=0.35,
        branch_sensitivity=1.0,
        min_keep=2,
        spacing_zyx=np.asarray((1.0, 1.0, 1.0), dtype=np.float32),
        structure=microglia_components._CUBIC_STRUCTURE,
    )

    assert labels.shape == working.shape
    assert int(np.max(labels)) >= 1


def test_segment_soma_markers_skips_watershed_for_large_volume(monkeypatch) -> None:
    working = np.zeros((3, 12, 12), dtype=np.float32)
    working[:, 3:9, 3:9] = 0.45
    finite = np.isfinite(working)
    seed = np.zeros_like(working, dtype=bool)
    seed[1, 5, 5] = True

    monkeypatch.setattr(microglia_components, "_LARGE_VOLUME_VOXELS", 1)

    def _fail_watershed(*_args, **_kwargs):
        raise AssertionError("large-volume path should bypass watershed")

    monkeypatch.setattr(microglia_components, "_watershed", _fail_watershed)

    labels = microglia_components._segment_soma_markers_from_reduced_threshold(
        seed,
        working,
        finite,
        low_floor=0.1,
        high_t=0.35,
        branch_sensitivity=1.0,
        min_keep=2,
        spacing_zyx=np.asarray((1.0, 1.0, 1.0), dtype=np.float32),
        structure=microglia_components._CUBIC_STRUCTURE,
    )

    assert labels.shape == working.shape
    assert int(np.max(labels)) >= 1


def test_compute_component_labels_skips_branch_reassignment_for_oversized_label_volume(
    monkeypatch,
) -> None:
    arr = np.zeros((4, 32, 32), dtype=np.float32)
    arr[:, 8:24, 8:24] = 0.28
    arr[2, 12:20, 12:20] = 0.55

    monkeypatch.setattr(microglia_components, "_MAX_BRANCH_REASSIGN_VOXELS", 1)

    def _fail_assign(*_args, **_kwargs):
        raise AssertionError("oversized label volume should skip branch reassignment")

    monkeypatch.setattr(
        microglia_components,
        "_assign_low_confidence_branches_to_one_owner",
        _fail_assign,
    )

    labels, order, sizes = compute_component_labels(
        arr,
        threshold=0.12,
        min_voxels=8,
        max_components=16,
        smooth_sigma=(0.0, 0.0, 0.0),
    )

    assert labels.shape == arr.shape
    assert len(order) >= 1
    assert int(sizes[int(order[0])]) > 0


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


def test_visible_raw_branches_survive_dense_smoothed_growth_mask() -> None:
    arr = np.zeros((5, 96, 96), dtype=np.float32)
    z = 2
    yy, xx = np.ogrid[:96, :96]
    soma = (yy - 68) ** 2 + (xx - 55) ** 2 <= 10**2
    for zz in (1, 2, 3):
        arr[zz, soma] = 0.98

    # These one-voxel processes are visible at the render threshold, but XY
    # smoothing pushes them below the dense-scene growth floor.
    arr[z, 68, 15:45] = 0.317
    for i in range(1, 20):
        arr[z, 68 - i, 55 + i] = 0.317

    # Unrelated dense signal raises the data-driven growth floor above the
    # render threshold. It must not make visible processes disappear.
    arr[1:4, 5:23, 5:23] = 0.64

    labels, order, _ = compute_component_labels(
        arr,
        threshold=0.316,
        min_voxels=8,
        max_components=16,
        smooth_sigma=(0.2, 0.45, 0.45),
        branch_sensitivity=1.0,
        spacing=(0.4, 0.331, 0.331),
    )

    component_id = int(labels[z, 68, 55])
    assert len(order) >= 2
    assert component_id > 0
    assert int(labels[z, 68, 15]) == component_id
    assert int(labels[z, 49, 74]) == component_id


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


def test_bright_terminal_bulbs_are_not_mistaken_for_somas() -> None:
    shape = (25, 128, 128)
    spacing = np.asarray((0.331, 0.198467, 0.198467), dtype=np.float32)
    zz, yy, xx = np.indices(shape, dtype=np.float32)
    points_zyx_um = np.stack(
        (
            (zz - 12) * spacing[0],
            (yy - 64) * spacing[1],
            (xx - 64) * spacing[2],
        ),
        axis=-1,
    )
    directions_yx = [
        np.asarray((1.0, 0.0)),
        np.asarray((-1.0, 0.0)),
        np.asarray((0.0, 1.0)),
        np.asarray((0.0, -1.0)),
        np.asarray((0.707, 0.707)),
        np.asarray((-0.707, 0.707)),
    ]

    arr = np.zeros(shape, dtype=np.float32)
    soma_radius_um = 1.8
    branch_radius_um = 0.25
    tip_radius_um = 0.75
    arr[np.linalg.norm(points_zyx_um, axis=-1) <= soma_radius_um] = 0.86

    points_yx_um = points_zyx_um[..., 1:]
    for direction_yx in directions_yx:
        start_yx_um = direction_yx * (soma_radius_um * 0.78)
        end_yx_um = direction_yx * 7.0
        segment_yx_um = end_yx_um - start_yx_um
        projection = np.clip(
            np.sum((points_yx_um - start_yx_um) * segment_yx_um, axis=-1)
            / np.sum(segment_yx_um * segment_yx_um),
            0.0,
            1.0,
        )
        closest_yx_um = start_yx_um + projection[..., None] * segment_yx_um
        branch_distance_um = np.sqrt(
            points_zyx_um[..., 0] ** 2
            + np.sum((points_yx_um - closest_yx_um) ** 2, axis=-1)
        )
        branch = branch_distance_um <= branch_radius_um
        arr[branch] = np.maximum(arr[branch], 0.56)

        tip_center_zyx_um = np.concatenate(([0.0], end_yx_um))
        tip_distance_um = np.linalg.norm(points_zyx_um - tip_center_zyx_um, axis=-1)
        tip = tip_distance_um <= tip_radius_um
        arr[tip] = np.maximum(arr[tip], 0.94)

    labels, order, sizes = compute_component_labels(
        arr,
        threshold=0.316,
        min_voxels=53,
        max_components=32,
        smooth_sigma=(0.2, 0.45, 0.45),
        branch_sensitivity=1.0,
        spacing=tuple(float(value) for value in spacing),
    )

    soma_id = int(labels[12, 64, 64])
    assert len(order) == 1
    assert soma_id > 0
    assert int(sizes[soma_id]) == int(np.count_nonzero(arr)) == 3041
    for direction_yx in directions_yx:
        tip_y = int(round(64 + direction_yx[0] * 7.0 / spacing[1]))
        tip_x = int(round(64 + direction_yx[1] * 7.0 / spacing[2]))
        assert int(labels[12, tip_y, tip_x]) == soma_id


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


def test_two_somas_in_one_high_confidence_island_can_split() -> None:
    arr = np.zeros((6, 96, 96), dtype=np.float32)
    z = 3

    # Two soma regions are connected by a moderate-intensity bridge that sits
    # above merge-floor levels, so they initially appear as one high-confidence
    # island. Separation should still produce two components.
    arr[z, 30:56, 20:46] = 0.24
    arr[z, 30:56, 42:68] = np.maximum(arr[z, 30:56, 42:68], 0.24)
    arr[z, 39:47, 28:36] = np.maximum(arr[z, 39:47, 28:36], 0.62)
    arr[z, 39:47, 52:60] = np.maximum(arr[z, 39:47, 52:60], 0.61)
    arr[z, 41:45, 42:46] = np.maximum(arr[z, 41:45, 42:46], 0.24)

    # Branch-like extensions that should remain attached after splitting.
    arr[z, 43, 12:22] = 0.16
    arr[z, 43, 66:78] = 0.16

    labels, order, _ = compute_component_labels(
        arr,
        threshold=0.12,
        min_voxels=24,
        max_components=16,
        smooth_sigma=(0.0, 0.0, 0.0),
        branch_sensitivity=1.2,
    )

    assert len(order) >= 2
    left_id = int(labels[z, 43, 32])
    right_id = int(labels[z, 43, 56])
    assert left_id > 0
    assert right_id > 0
    assert left_id != right_id


def test_dense_field_haze_does_not_swallow_soma_isolation() -> None:
    arr = np.full((4, 32, 32), 0.08, dtype=np.float32)
    z = 2

    # Two soma-like regions sit in a weak whole-field haze. The haze should
    # not become part of the isolated components or collapse both cells into
    # one full-volume label.
    arr[z, 9:23, 4:16] = 0.24
    arr[z, 9:23, 18:30] = np.maximum(arr[z, 9:23, 18:30], 0.24)
    arr[z, 13:17, 7:11] = np.maximum(arr[z, 13:17, 7:11], 0.52)
    arr[z, 13:17, 21:25] = np.maximum(arr[z, 13:17, 21:25], 0.50)

    labels, order, sizes = compute_component_labels(
        arr,
        threshold=0.18,
        min_voxels=20,
        max_components=16,
        smooth_sigma=(0.0, 0.0, 0.0),
        branch_sensitivity=1.0,
    )

    assert len(order) >= 2
    left_id = int(labels[z, 15, 9])
    right_id = int(labels[z, 15, 23])
    assert left_id > 0
    assert right_id > 0
    assert left_id != right_id
    assert int(labels[0, 0, 0]) == 0
    assert int(labels[z, 15, 17]) == 0
    assert max(int(sizes[int(order[0])]), int(sizes[int(order[1])])) < 1200


def test_somas_separate_along_z_axis_in_true_3d() -> None:
    arr = np.zeros((12, 48, 48), dtype=np.float32)
    y = 24
    x = 24

    # Two soma-like cores at the same XY but different Z.
    arr[3, y - 1:y + 2, x - 1:x + 2] = 0.62
    arr[8, y - 1:y + 2, x - 1:x + 2] = 0.64

    # A faint vertical bridge should not collapse both somas into one label.
    arr[4:8, y, x] = np.maximum(arr[4:8, y, x], 0.14)

    labels, order, _ = compute_component_labels(
        arr,
        threshold=0.12,
        min_voxels=8,
        max_components=16,
        smooth_sigma=(0.0, 0.0, 0.0),
        branch_sensitivity=1.3,
        spacing=(0.40, 0.331, 0.331),
    )

    assert len(order) >= 2
    top_id = int(labels[3, y, x])
    bottom_id = int(labels[8, y, x])
    assert top_id > 0
    assert bottom_id > 0
    assert top_id != bottom_id


def test_coarse_z_spacing_keeps_diagonal_in_plane_branches_connected() -> None:
    arr = np.zeros((3, 64, 64), dtype=np.float32)
    z = 1
    arr[z, 28:37, 28:37] = 0.75
    for i in range(1, 23):
        arr[z, 28 - i, 28 - i] = 0.36
        arr[z, 36 + i, 36 + i] = 0.36

    labels, order, sizes = compute_component_labels(
        arr,
        threshold=0.316,
        min_voxels=16,
        max_components=8,
        smooth_sigma=(0.0, 0.0, 0.0),
        branch_sensitivity=1.0,
        spacing=(2.0, 0.3, 0.3),
    )

    assert len(order) == 1
    component_id = int(labels[z, 32, 32])
    assert component_id > 0
    assert int(labels[z, 6, 6]) == component_id
    assert int(labels[z, 58, 58]) == component_id
    assert int(sizes[component_id]) == 125


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
