import warnings
import numpy as np
import pytest
from skimage import morphology as reference
from nvap.preprocess import morphology as fast


@pytest.mark.parametrize('operation', ['binary_dilation', 'binary_opening', 'binary_closing'])
@pytest.mark.parametrize('shape', [(1, 13), (17, 19), (4, 11, 9)])
@pytest.mark.parametrize('footprint_size', [2, 3, 5])
def test_binary_kernels_match_library_at_image_edges(operation, shape, footprint_size):
    rng = np.random.default_rng(309)
    footprint = np.ones((footprint_size,) * len(shape), dtype=bool)
    for mask in (rng.random(shape) > 0.6, np.ones(shape, dtype=bool), np.zeros(shape, dtype=bool)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            expected = getattr(reference, operation)(mask, footprint=footprint)
            actual = getattr(fast, operation)(mask, footprint=footprint)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('threshold', [1, 2, 4, 5, 8, 16])
def test_small_object_and_hole_cutoffs_match_installed_library(threshold):
    rng = np.random.default_rng(902)
    mask = rng.random((24, 25)) > 0.5
    mask[2:4, 2:4] = True
    for connectivity in (1, 2):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            objects = reference.remove_small_objects(mask, min_size=threshold, connectivity=connectivity)
            holes = reference.remove_small_holes(mask, area_threshold=threshold, connectivity=connectivity)
        np.testing.assert_array_equal(fast.remove_small_objects(mask, min_size=threshold, connectivity=connectivity), objects)
        np.testing.assert_array_equal(fast.remove_small_holes(mask, area_threshold=threshold, connectivity=connectivity), holes)
