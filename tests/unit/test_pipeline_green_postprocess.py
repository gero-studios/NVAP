from __future__ import annotations

import numpy as np

from nvap.config.types import ChannelVolume, DatasetVolume, PSFConfig, PreprocessConfig, VoxelSpacing
from nvap.pipeline import apply_psf_to_dataset


def _make_dataset() -> DatasetVolume:
    rng = np.random.default_rng(42)
    green = np.zeros((6, 48, 48), dtype=np.float32)
    green[2, 24, 10:38] = 0.18
    green[3, 25, 12:36] = 0.16
    green += rng.normal(0.0, 0.02, size=green.shape).astype(np.float32)
    green = np.clip(green, 0.0, 1.0)
    red = np.clip(0.1 * rng.random(green.shape, dtype=np.float32), 0.0, 1.0)
    spacing = VoxelSpacing()
    return DatasetVolume(
        green=ChannelVolume("green", green, list(range(green.shape[0])), spacing),
        red=ChannelVolume("red", red, list(range(red.shape[0])), spacing),
        shared_z_range=(0, green.shape[0] - 1),
    )


def test_apply_psf_uses_green_postprocess_when_configured() -> None:
    dataset = _make_dataset()
    psf_cfg = PSFConfig(enabled=True, iterations=1)

    out_no_post = apply_psf_to_dataset(dataset, psf_cfg, preprocess_config=None)
    out_post = apply_psf_to_dataset(
        dataset,
        psf_cfg,
        preprocess_config=PreprocessConfig(
            green_denoise_strategy="classical_branch_aware",
            green_post_deconv_strength=0.2,
        ),
    )

    green_delta = float(np.mean(np.abs(out_post.green.data - out_no_post.green.data)))
    assert green_delta > 1.0e-5
    assert np.allclose(out_post.red.data, out_no_post.red.data)
