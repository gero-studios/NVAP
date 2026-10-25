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


def test_eligibility_requires_one_supported_soma_and_keeps_arbor():
    green, labels, order, spacing = _scene()
    filtered, ids, sizes = filter_analyzable_components(
        green, labels, order, np.bincount(labels.ravel()), threshold=0.4, spacing=spacing,
    )
    assert ids.tolist() == [1]
    np.testing.assert_array_equal(filtered > 0, labels == 1)
    assert sizes[1] == np.count_nonzero(labels == 1)


def test_trimmed_or_below_threshold_soma_is_ineligible():
    green, labels, order, spacing = _scene()
    assert not analyzable_component_ids(green, labels, order, threshold=0.95, spacing=spacing).size
    assert not analyzable_component_ids(green, labels, order, threshold=0.4, spacing=spacing, trim_first_slices=18).size


def test_analysis_uses_same_single_soma_population_as_display():
    green, labels, order, spacing = _scene()
    expected = analyzable_component_ids(green, labels, order, threshold=0.4, spacing=spacing)
    analysis = analyze_microglia_cells(
        green, np.zeros_like(green), labels, order, spacing=spacing,
        render=RenderConfig(threshold_green=0.4, threshold_red=0.5),
    )
    assert [cell.component_id for cell in analysis.cells] == expected.tolist() == [1]
    assert analysis.analyzed_cell_count == len(expected)
