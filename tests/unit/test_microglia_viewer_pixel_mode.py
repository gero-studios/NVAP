from __future__ import annotations

import numpy as np

from nvap.analysis.microglia_components import compute_component_labels, isolate_component
from nvap.config.types import ChannelVolume, DatasetVolume, PreprocessConfig, VoxelSpacing
from nvap.preprocess.enhancement import preprocess_dataset


def test_microglia_component_view_works_with_pixel_no_psf_mode() -> None:
    rng = np.random.default_rng(2027)
    green = np.zeros((6, 64, 64), dtype=np.float32)
    green[2, 20, 8:26] = 0.18
    green[3, 21, 10:24] = 0.16
    green[2, 42, 38:56] = 0.19
    green[3, 43, 40:54] = 0.17
    green += rng.normal(0.0, 0.02, size=green.shape).astype(np.float32)
    green = np.clip(green, 0.0, 1.0)

    red = np.clip(0.08 * rng.random(green.shape, dtype=np.float32), 0.0, 1.0)
    spacing = VoxelSpacing()
    dataset = DatasetVolume(
        green=ChannelVolume("green", green, list(range(green.shape[0])), spacing),
        red=ChannelVolume("red", red, list(range(red.shape[0])), spacing),
        shared_z_range=(0, green.shape[0] - 1),
    )

    cfg = PreprocessConfig()
    assert cfg.green_denoise_strategy == "pixel2voxel_no_psf"
    processed = preprocess_dataset(dataset, cfg)

    labels, order, _ = compute_component_labels(processed.green.data, threshold=0.1)
    assert len(order) >= 1

    isolated = isolate_component(processed.green.data, labels, int(order[0]))
    assert 0 < int(np.count_nonzero(isolated)) < int(np.count_nonzero(processed.green.data))
