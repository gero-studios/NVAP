from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter


def test_directml_gaussian_3d_conv2d_path_matches_scipy_nearest() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from nvap.accelerate import _gaussian_filter_3d_via_conv2d

    arr = np.random.default_rng(42).random((7, 11, 13), dtype=np.float32)
    sigma = (1.2, 0.8, 1.5)
    tensor = torch.from_numpy(arr)

    out = _gaussian_filter_3d_via_conv2d(torch, F, tensor, sigma, torch.device("cpu"))
    expected = gaussian_filter(arr, sigma=sigma, mode="nearest")

    np.testing.assert_allclose(out.detach().numpy(), expected, rtol=2.0e-4, atol=5.0e-5)
