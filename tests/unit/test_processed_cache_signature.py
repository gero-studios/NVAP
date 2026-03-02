from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from nvap.cache.processed_cache import build_dataset_signature


def _write_stack(path: Path, values: list[int]) -> None:
    frames = [np.full((8, 8), v, dtype=np.uint8) for v in values]
    stack = np.stack(frames, axis=0)
    iio.imwrite(path, stack, plugin="tifffile", photometric="minisblack")


def test_build_dataset_signature_supports_file_sources(tmp_path: Path) -> None:
    red = tmp_path / "red.tif"
    green = tmp_path / "green.tif"
    _write_stack(red, [10, 20, 30])
    _write_stack(green, [5, 15, 25])

    sig_a = build_dataset_signature({"red": red, "green": green})
    sig_b = build_dataset_signature({"red": red, "green": green})
    assert sig_a == sig_b

    _write_stack(green, [5, 15, 40, 50])
    sig_c = build_dataset_signature({"red": red, "green": green})
    assert sig_c != sig_a
