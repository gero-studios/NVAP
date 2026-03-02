from __future__ import annotations

import numpy as np

from nvap.config.types import ChannelVolume, PreprocessConfig, VoxelSpacing
from nvap.preprocess.enhancement import preprocess_channel


def test_green_channel_is_passthrough_even_if_microglia_strategy_selected() -> None:
    arr = np.linspace(0.0, 1.0, num=4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
    channel = ChannelVolume("green", arr, [10, 11, 12, 13], VoxelSpacing())

    cfg = PreprocessConfig(enabled=True, green_denoise_strategy="microglia_masking")
    out = preprocess_channel(channel, cfg)

    assert out.name == "green"
    assert out.z_indices == [10, 11, 12, 13]
    assert np.allclose(out.data, arr, atol=0.0)
