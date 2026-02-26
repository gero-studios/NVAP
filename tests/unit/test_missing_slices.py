from __future__ import annotations

import numpy as np

from nvap.preprocess.missing_slices import interpolate_missing_slices


def test_interpolate_missing_middle_slice() -> None:
    z0 = np.zeros((4, 4), dtype=np.float32)
    z2 = np.ones((4, 4), dtype=np.float32)
    volume = np.stack([z0, z2], axis=0)
    z_indices = [1, 3]

    filled, full_z = interpolate_missing_slices(volume, z_indices)

    assert full_z == [1, 2, 3]
    assert filled.shape == (3, 4, 4)
    assert np.allclose(filled[0], 0.0)
    assert np.allclose(filled[2], 1.0)
    assert np.allclose(filled[1], 0.5)
