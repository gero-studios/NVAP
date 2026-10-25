import numpy as np
import scipy.ndimage as ndi
from nvap.analysis.contact_components import prepare_contact_components
from nvap.analysis.microglia_analysis import analyze_microglia_cells, microglia_analysis_to_csv_rows
from nvap.config.types import RenderConfig


def _check_contacts(labels):
    for cid, bounds in enumerate(ndi.find_objects(labels), 1):
        if bounds is not None:
            assert ndi.label(labels[bounds] == cid, structure=ndi.generate_binary_structure(3, 1))[1] == 1


def test_somaless_pieces_are_preserved_without_faint_or_distance_attachment():
    green = np.zeros((5, 20, 50), dtype=np.float32)
    green[2, 8, 4:15] = .9
    green[2, 8, 15:25] = .08
    green[2, 8, 25:38] = .9
    green[2, 16, 30:40] = .9
    labels, order, sizes = prepare_contact_components(green, threshold=.4, spacing=(.3,) * 3, require_soma=False)
    assert len(order) == 3
    np.testing.assert_array_equal(labels > 0, green >= .4)
    assert labels[2, 8, 8] != labels[2, 8, 30]
    _check_contacts(labels)
    result = analyze_microglia_cells(green, np.zeros_like(green), labels, order, spacing=(.3,) * 3, render=RenderConfig(threshold_green=.4), components_prepared=True, workers=1)
    assert result.analyzed_cell_count == 3
    assert sum(cell.voxel_count for cell in result.cells) == sizes.sum()
    assert all(not cell.soma_detected for cell in result.cells)
    assert all(cell.soma_to_vessel_um is None for cell in result.cells)
    assert all(row['soma_detected'] is False for row in microglia_analysis_to_csv_rows(result))


def test_no_256_component_cap_or_hidden_size_gate():
    green = np.zeros((3, 61, 61), dtype=np.float32)
    green[1, ::3, ::3] = .9
    labels, order, _ = prepare_contact_components(green, threshold=.4, spacing=(.3,) * 3, require_soma=False)
    assert len(order) == 441
    np.testing.assert_array_equal(labels > 0, green >= .4)
    _check_contacts(labels)


def test_touching_somas_split_without_losing_contact_signal():
    zz, yy, xx = np.indices((25, 45, 95))
    green = np.zeros(zz.shape, dtype=np.float32)
    for x in (25, 65):
        body = (zz - 12)**2 + (yy - 22)**2 + (xx - x)**2 <= 7**2
        green[body] = .9
    green[12, 22, 25:66] = .7
    labels, order, _ = prepare_contact_components(green, threshold=.4, spacing=(.3,) * 3)
    assert len(order) == 2
    assert labels[12, 22, 25] != labels[12, 22, 65]
    np.testing.assert_array_equal(labels > 0, green >= .4)
    _check_contacts(labels)


def test_explicit_trim_does_not_require_remaining_objects_to_have_a_soma():
    green = np.zeros((10, 10, 20), dtype=np.float32)
    green[0, 3, 3:8] = .9
    green[5, 3, 12:18] = .9
    labels, order, _ = prepare_contact_components(green, threshold=.4, spacing=(.3,) * 3, trim_first_slices=1, require_soma=False)
    assert len(order) == 1
    assert not np.any(labels[0])
    assert np.all(labels[5, 3, 12:18] > 0)


def test_default_requires_soma_and_never_inherits_disconnected_fragments():
    zz, yy, xx = np.indices((25, 45, 95))
    green = np.zeros(zz.shape, dtype=np.float32)
    green[(zz - 12)**2 + (yy - 22)**2 + (xx - 25)**2 <= 7**2] = .9
    green[12, 22, 33:45] = .08
    green[12, 22, 45:65] = .9
    labels, order, _ = prepare_contact_components(green, threshold=.4, spacing=(.3,) * 3)
    assert len(order) == 1
    assert labels[12, 22, 25] == 1
    assert not np.any(labels[12, 22, 33:65])
    _check_contacts(labels)


def test_smaller_supported_soma_is_not_removed_beside_a_large_one():
    zz, yy, xx = np.indices((35, 45, 95))
    green = np.zeros(zz.shape, dtype=np.float32)
    for x, radius in ((25, 2.2), (65, 1.4)):
        green[((zz - 17) * .3)**2 + ((yy - 22) * .3)**2 + ((xx - x) * .3)**2 <= radius**2] = .9
    labels, order, _ = prepare_contact_components(green, threshold=.4, spacing=(.3,) * 3)
    assert len(order) == 2
    assert labels[17, 22, 25] != labels[17, 22, 65]
    assert labels[17, 22, 65] > 0
    np.testing.assert_array_equal(labels > 0, green >= .4)


def test_corner_only_contact_cannot_transfer_cell_identity_to_a_speck():
    from nvap.analysis.contact_components import soma_component_ids
    green = np.zeros((20, 30, 30), dtype=np.float32)
    green[5:15, 8:20, 8:20] = .9
    green[15, 20, 20] = .9  # Shares only a corner with the body.
    labels, order, _ = prepare_contact_components(green, threshold=.4, spacing=(.3,) * 3, require_soma=False)
    assert len(order) == 2
    assert labels[15, 20, 20] != labels[10, 12, 12]
    supported = soma_component_ids(green, labels, order, threshold=.4, spacing=(.3,) * 3)
    assert supported.tolist() == [int(labels[10, 12, 12])]
    _check_contacts(labels)


def test_thin_branch_and_planar_core_are_not_soma_anchors():
    from nvap.analysis.microglia_quality import _soma_geometry
    z, y, x = np.indices((25, 40, 120))
    tube = (((z - 12) * .3)**2 + ((y - 20) * .3)**2 <= 1.1**2) & (x > 10) & (x < 110)
    assert _soma_geometry(tube, np.array([.3] * 3))[0] == []
    # Coarse axial sampling cannot turn a single-plane knot into 3D support.
    plane = np.zeros((3, 15, 15), dtype=bool)
    plane[1, 3:12, 3:12] = True
    assert _soma_geometry(plane, np.array([2., .3, .3]))[0] == []
