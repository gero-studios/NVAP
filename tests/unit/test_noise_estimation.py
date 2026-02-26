from __future__ import annotations

import numpy as np

from nvap.preprocess.denoisers import estimate_noise_sigma


def test_noise_estimation_is_positive_and_stable_across_scale() -> None:
    rng = np.random.default_rng(42)
    base = rng.normal(0.2, 0.03, size=(8, 32, 32)).astype(np.float32)
    sigma_a = estimate_noise_sigma(np.clip(base, 0.0, 1.0))
    sigma_b = estimate_noise_sigma(np.clip(base * 255.0, 0.0, 255.0) / 255.0)

    assert sigma_a > 0.0
    assert sigma_b > 0.0
    rel = abs(sigma_a - sigma_b) / max(1.0e-6, sigma_a)
    assert rel < 0.25
