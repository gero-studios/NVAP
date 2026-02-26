from __future__ import annotations

import numpy as np

from nvap.config.types import PSFConfig, VoxelSpacing
from nvap.preprocess.psf import build_gaussian_psf, deconvolve_volume


def test_build_gaussian_psf_normalized() -> None:
    psf = build_gaussian_psf(VoxelSpacing(), sigma_xy_um=0.4, sigma_z_um=0.8)
    assert psf.ndim == 3
    assert psf.shape[0] % 2 == 1
    assert np.isclose(float(psf.sum()), 1.0, atol=1e-4)


def test_deconvolution_shape_and_finiteness() -> None:
    volume = np.random.default_rng(0).random((6, 10, 10), dtype=np.float32)
    out = deconvolve_volume(volume, VoxelSpacing(), PSFConfig(enabled=True, iterations=2))
    assert out.shape == volume.shape
    assert np.isfinite(out).all()
    assert float(out.min()) >= 0.0
