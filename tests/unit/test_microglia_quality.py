import numpy as np
from nvap.analysis.microglia_quality import analyzable_component_ids, filter_analyzable_components
from nvap.analysis.microglia_analysis import analyze_microglia_cells
from nvap.config.types import RenderConfig


def _scene():
    spacing = (0.4, 0.3, 0.3)
    zz, yy, xx = np.indices((25, 64, 100), dtype=np.float32)
    labels = np.zeros(zz.shape, dtype=np.int32)
    for cid, x, y in ((1, 18, 20), (3, 50, 42), (3, 80, 42)):
        sphere = ((zz - 12) * spacing[0]) ** 2 + ((yy - y) * spacing[1]) ** 2 + ((xx - x) * spacing[2]) ** 2 <= 2.1 ** 2
        labels[sphere] = cid
    labels[12, 20, 18:36] = 1  # Keep the accepted cell's thin process.
    labels[12, 8, 10:70] = 2  # Detached process with no soma.
    green = np.where(labels > 0, 0.9, 0.0).astype(np.float32)
    return green, labels, np.array([1, 2, 3]), spacing


def test_contact_repair_splits_somas_and_hides_somaless_objects():
    green, labels, order, spacing = _scene()
    filtered, ids, sizes = filter_analyzable_components(
        green, labels, order, np.bincount(labels.ravel()), threshold=0.4, spacing=spacing,
    )
    assert len(ids) == 3
    np.testing.assert_array_equal(filtered > 0, (labels == 1) | (labels == 3))
    assert sizes[1] == np.count_nonzero(labels == 1)
    assert filtered[12, 42, 50] != filtered[12, 42, 80]


def test_trimmed_or_below_threshold_soma_is_ineligible():
    green, labels, order, spacing = _scene()
    assert not analyzable_component_ids(green, labels, order, threshold=0.95, spacing=spacing).size
    assert not analyzable_component_ids(green, labels, order, threshold=0.4, spacing=spacing, trim_first_slices=18).size


def test_analysis_uses_same_single_soma_population_as_display():
    green, labels, order, spacing = _scene()
    repaired, expected, _ = filter_analyzable_components(
        green, labels, order, np.bincount(labels.ravel()), threshold=0.4,
        spacing=spacing, compact_ids=False,
    )
    analysis = analyze_microglia_cells(
        green, np.zeros_like(green), labels, order, spacing=spacing,
        render=RenderConfig(threshold_green=0.4, threshold_red=0.5),
    )
    assert [cell.component_id for cell in analysis.cells] == expected.tolist()
    assert len(expected) == 3
    assert analysis.analyzed_cell_count == len(expected)


def test_repair_keeps_detached_process_separate_from_nearest_soma():
    green, labels, order, spacing = _scene()
    labels[12, 55, 60:70] = 3
    green[12, 55, 60:70] = 0.9
    repaired, ids, sizes = filter_analyzable_components(
        green, labels, order, np.bincount(labels.ravel()), threshold=0.4, spacing=spacing,
    )
    assert not np.any(repaired[12, 55, 60:70])
    again, again_ids, _ = filter_analyzable_components(green, repaired, ids, sizes, threshold=0.4, spacing=spacing)
    np.testing.assert_array_equal(again, repaired)
    np.testing.assert_array_equal(again_ids, ids)


def test_single_soma_does_not_inherit_remote_somaless_arbor():
    green, labels, _, spacing = _scene()
    labels[labels == 3] = 0
    labels[12, 8, 10:70] = 1  # A previous shared label is not ownership evidence.
    repaired, ids, _ = filter_analyzable_components(
        green, labels, np.array([1]), np.bincount(labels.ravel()), threshold=0.4, spacing=spacing,
    )
    assert len(ids) == 1
    assert not np.any(repaired[12, 8, 10:70])
    assert repaired[12, 20, 18] == repaired[12, 20, 35] == 1


def test_faint_path_cannot_make_disconnected_visible_parts_one_component():
    from nvap.analysis.microglia_components import _recover_faint_connected_branches
    zz, yy, xx = np.indices((25, 40, 90))
    soma = (zz - 12)**2 + (yy - 20)**2 + (xx - 20)**2 <= 7**2
    green = np.where(soma, 0.9, 0.0).astype(np.float32)
    green[12, 20, 27:40] = 0.08  # A real path below the displayed cutoff.
    green[12, 20, 40:61] = 0.8
    green[12, 32, 45:61] = 0.8  # Remote bright signal has no path.
    labels = soma.astype(np.int32)
    candidate = green >= 0.4
    recovered = _recover_faint_connected_branches(
        labels, candidate, green, candidate, weak_floor=0.04, min_voxels=2,
    )
    assert recovered[12, 20, 50] == 1
    repaired, ids, sizes = filter_analyzable_components(
        green, recovered, np.array([1]), np.bincount(recovered.ravel()),
        threshold=0.4, spacing=(0.3,) * 3,
    )
    assert len(ids) == 1
    assert repaired[12, 20, 50] == 0
    assert not np.any(repaired[12, 20, 28:39])  # Don't paint the path into the cell.
    assert not np.any(repaired[12, 32, 45:61])
    again, _, _ = filter_analyzable_components(
        green, repaired, ids, sizes, threshold=0.4, spacing=(0.3,) * 3,
    )
    np.testing.assert_array_equal(again, repaired)


def test_quality_faint_paths_cannot_cross_a_different_labeled_cell():
    green, labels, _, spacing = _scene()
    labels[labels == 3] = 0
    green[12, 20, 36:45] = 0.08
    green[12, 20, 45:55] = 0.9
    labels[12, 20, 45:55] = 1
    labels[12, 20, 40] = 2  # A competitor blocks the only measured path.
    repaired, _, _ = filter_analyzable_components(
        green, labels, np.array([1]), np.bincount(labels.ravel()),
        threshold=0.4, spacing=spacing,
    )
    assert not np.any(repaired[12, 20, 45:55])


def test_two_core_lobes_within_one_thick_body_stay_one_cell():
    zz, yy, xx = np.indices((24, 48, 96), dtype=np.float32)
    body = np.zeros(zz.shape, dtype=bool)
    for x in (40, 53):
        body |= ((zz - 12) * 0.3) ** 2 + ((yy - 24) * 0.3) ** 2 + ((xx - x) * 0.3) ** 2 <= 2.1 ** 2
    labels = body.astype(np.int32)
    volume = body.astype(np.float32) * 0.9
    repaired, ids, _ = filter_analyzable_components(
        volume, labels, np.array([1]), np.bincount(labels.ravel()),
        threshold=0.4, spacing=(0.3,) * 3,
    )
    assert ids.tolist() == [1]
    np.testing.assert_array_equal(repaired > 0, body)


def test_bright_neighbor_cannot_steal_dim_soma_core():
    zz, yy, xx = np.indices((24, 48, 96), dtype=np.float32)
    volume = np.zeros(zz.shape, dtype=np.float32)
    for x, value in ((30, 0.99), (58, 0.5)):
        body = ((zz - 12) * 0.3) ** 2 + ((yy - 24) * 0.3) ** 2 + ((xx - x) * 0.3) ** 2 <= 2.1 ** 2
        volume[body] = value
    volume[12, 24, 37:52] = 0.6
    labels = (volume > 0).astype(np.int32)
    repaired, ids, sizes = filter_analyzable_components(
        volume, labels, np.array([1]), np.bincount(labels.ravel()),
        threshold=0.4, spacing=(0.3,) * 3,
    )
    assert len(ids) == 2
    assert repaired[12, 24, 30] != repaired[12, 24, 58]
    assert np.min(sizes[1:]) > 10
    np.testing.assert_array_equal(repaired > 0, labels > 0)
    assert len(analyzable_component_ids(volume, repaired, ids, threshold=0.4, spacing=(0.3,) * 3)) == 2


def test_touching_arbors_with_separate_somas_are_not_one_cell():
    zz, yy, xx = np.indices((24, 48, 96), dtype=np.float32)
    body = np.zeros(zz.shape, dtype=bool)
    for x in (28, 60):
        body |= ((zz - 12) * 0.3) ** 2 + ((yy - 24) * 0.3) ** 2 + ((xx - x) * 0.3) ** 2 <= 2.1 ** 2
    # This process is thick enough to join the old body envelopes, but there
    # are still two separate soma peaks and two cells.
    tube = ((zz - 12) * 0.3) ** 2 + ((yy - 24) * 0.3) ** 2 <= 1.0 ** 2
    body |= tube & (xx >= 28) & (xx <= 60)
    labels = body.astype(np.int32)
    volume = body.astype(np.float32) * 0.9
    repaired, ids, _ = filter_analyzable_components(
        volume, labels, np.array([1]), np.bincount(labels.ravel()),
        threshold=0.4, spacing=(0.3,) * 3,
    )
    assert len(ids) == 2
    assert repaired[12, 24, 28] != repaired[12, 24, 60]
    np.testing.assert_array_equal(repaired > 0, body)


def test_large_soma_does_not_suppress_smaller_neighbor():
    zz, yy, xx = np.indices((32, 64, 128), dtype=np.float32)
    body = np.zeros(zz.shape, dtype=bool)
    for x, radius in ((28, 3.0), (90, 1.8)):
        body |= ((zz - 16) * 0.3) ** 2 + ((yy - 32) * 0.3) ** 2 + ((xx - x) * 0.3) ** 2 <= radius ** 2
    body[16, 32, 28:91] = True
    labels = body.astype(np.int32)
    volume = body.astype(np.float32) * 0.9
    repaired, ids, _ = filter_analyzable_components(
        volume, labels, np.array([1]), np.bincount(labels.ravel()),
        threshold=0.4, spacing=(0.3,) * 3,
    )
    assert len(ids) == 2
    assert repaired[16, 32, 28] != repaired[16, 32, 90]
    np.testing.assert_array_equal(repaired > 0, body)
