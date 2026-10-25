import numpy as np
from nvap.analysis.contact_components import prepare_contact_components, classify_components
from nvap.analysis.microglia_analysis import analyze_microglia_cells, microglia_analysis_to_csv_rows
from nvap.config.types import RenderConfig


def _arbor(green, z, y=45, x=75):
    # Large planar four-arm tree: clearly branched but deliberately no 3D soma.
    green[z, y-3:y+4, x-30:x+31] = .9
    green[z, y-30:y+31, x-3:x+4] = .9


def _scene():
    green = np.zeros((17, 160, 160), dtype=np.float32)
    for z in (0, 8, 16):
        _arbor(green, z)
    return green


def test_neither_z_boundary_bypasses_soma_but_either_boundary_does_not():
    green = _scene()
    labels, order, _ = prepare_contact_components(green, threshold=.4, spacing=(.3,) * 3, require_soma=False)
    result = classify_components(green, labels, order, threshold=.4, spacing=(.3,) * 3)
    assert not result.soma_ids.size
    assert result.cell_ids.tolist() == [int(labels[8, 45, 75])]
    assert result.interior_ids.tolist() == result.cell_ids.tolist()
    assert labels[0, 45, 75] not in result.cell_ids
    assert labels[-1, 45, 75] not in result.cell_ids
    accepted, _, _ = prepare_contact_components(green, threshold=.4, spacing=(.3,) * 3)
    assert not np.any(accepted[0]) and not np.any(accepted[-1])
    assert np.all(accepted[8, 45, 45:106] > 0)


def test_trimmed_planes_are_the_z_boundaries_and_lateral_edges_do_not_disqualify():
    green = _scene()
    _arbor(green, 2)
    _arbor(green, 14)
    _arbor(green, 8, y=120, x=30)  # Touches X acquisition edge, deliberately allowed.
    labels, order, _ = prepare_contact_components(green, threshold=.4, spacing=(.3,) * 3, require_soma=False,
        trim_first_slices=2, trim_last_slices=2)
    result = classify_components(green, labels, order, threshold=.4, spacing=(.3,) * 3,
        trim_first_slices=2, trim_last_slices=2)
    assert set(result.interior_ids) == {labels[8, 45, 75], labels[8, 120, 30]}
    assert labels[2, 45, 75] not in result.cell_ids
    assert labels[-3, 45, 75] not in result.cell_ids


def test_speck_filter_and_contact_still_apply_to_the_exception():
    green = _scene()
    green[8, 10, 20] = .9
    labels, order, _ = prepare_contact_components(green, threshold=.4, spacing=(.3,) * 3, require_soma=False)
    result = classify_components(green, labels, order, threshold=.4, spacing=(.3,) * 3)
    assert labels[8, 10, 20] not in result.cell_ids
    assert not classify_components(green, labels, order, threshold=.4, spacing=(.3,) * 3,
        interior_min_voxels=806).cell_ids.size
    # Even a supplied shared ID cannot absorb a disconnected piece.
    labels[8, 10, 20] = labels[8, 45, 75]
    assert not classify_components(green, labels, [labels[8, 45, 75]], threshold=.4, spacing=(.3,) * 3).cell_ids.size


def test_inferred_candidates_are_measured_and_exported_without_fabricating_a_soma():
    green = _scene()
    labels, order, _ = prepare_contact_components(green, threshold=.4, spacing=(.3,) * 3, require_soma=False)
    result = analyze_microglia_cells(green, np.zeros_like(green), labels, order, spacing=(.3,) * 3,
        render=RenderConfig(threshold_green=.4), workers=1)
    assert result.analyzed_cell_count == 1
    cell = result.cells[0]
    assert cell.inferred_from_z_interior and not cell.soma_detected
    assert cell.soma_to_vessel_um is None
    row = microglia_analysis_to_csv_rows(result)[0]
    assert row['classification_basis'] == 'z_interior_inferred'
    assert row['soma_detected'] is False


def test_all_trimmed_signal_produces_no_inference():
    green = _scene()
    labels = np.ones_like(green, dtype=np.int32)
    result = classify_components(green, labels, [1], threshold=.4, spacing=(.3,) * 3, trim_first_slices=100)
    assert not result.cell_ids.size


def test_small_arbor_and_long_unbranched_fiber_do_not_bypass_soma():
    green = _scene()
    green[10, 140, 5:155] = .9
    green[10, 100, 10:21] = .9
    green[10, 95:106, 15] = .9
    labels, order, _ = prepare_contact_components(green, threshold=.4, spacing=(.3,) * 3, require_soma=False)
    result = classify_components(green, labels, order, threshold=.4, spacing=(.3,) * 3)
    assert result.cell_ids.tolist() == [int(labels[8, 45, 75])]
    assert labels[10, 100, 15] not in result.cell_ids
    assert labels[10, 140, 20] not in result.cell_ids


def test_large_unbranched_fiber_is_excluded_even_with_enough_volume_and_extent():
    green = np.zeros((17, 100, 160), dtype=np.float32)
    for i in range(20, 140):
        green[8, i // 2: i // 2 + 7, i] = .9
    labels, order, _ = prepare_contact_components(green, threshold=.4, spacing=(.3,) * 3, require_soma=False)
    result = classify_components(green, labels, order, threshold=.4, spacing=(.3,) * 3)
    assert not result.cell_ids.size
