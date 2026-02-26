from __future__ import annotations

from typing import Literal, Protocol

import numpy as np

from nvap.config.types import ChannelVolume, DatasetVolume


class ChannelAnalyzerPlugin(Protocol):
    plugin_id: str
    target_channel: Literal["green", "red", "both"]

    def analyze(
        self,
        channel_volume: ChannelVolume,
        mask: np.ndarray,
        dataset: DatasetVolume,
    ) -> dict:
        ...
