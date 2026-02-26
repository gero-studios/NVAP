from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from nvap.config.types import VoxelSpacing
from nvap.io.stack_loader import load_dataset


def _write_rgb(path: Path, r: int, g: int, b: int) -> None:
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[..., 0] = r
    arr[..., 1] = g
    arr[..., 2] = b
    iio.imwrite(path, arr)


def _write_mixed_rg(path: Path, red: int, green: int, blue: int = 0) -> None:
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[:4, :, 0] = red
    arr[4:, :, 1] = green
    if blue > 0:
        arr[:, :, 2] = blue
    iio.imwrite(path, arr)


def test_load_dataset_sorts_z_and_extracts_channels(tmp_path: Path) -> None:
    green_dir = tmp_path / "Segmented" / "Green"
    red_dir = tmp_path / "Segmented" / "Red"
    green_dir.mkdir(parents=True)
    red_dir.mkdir(parents=True)

    _write_rgb(green_dir / "sample_z002c1.png", r=0, g=40, b=0)
    _write_rgb(green_dir / "sample_z001c1.png", r=0, g=20, b=0)
    _write_rgb(red_dir / "sample_z001c2.png", r=10, g=0, b=0)
    _write_rgb(red_dir / "sample_z002c2.png", r=50, g=0, b=0)

    dataset = load_dataset(tmp_path, spacing=VoxelSpacing())
    assert dataset.green.z_indices == [1, 2]
    assert dataset.red.z_indices == [1, 2]
    assert dataset.green.data.shape == (2, 8, 8)
    assert dataset.red.data.shape == (2, 8, 8)
    assert float(dataset.green.data[0, 0, 0]) < float(dataset.green.data[1, 0, 0])
    assert float(dataset.red.data[0, 0, 0]) < float(dataset.red.data[1, 0, 0])


def test_load_dataset_from_single_folder_combined_rgb_slices(tmp_path: Path) -> None:
    _write_mixed_rg(tmp_path / "sample_z002.png", red=60, green=20)
    _write_mixed_rg(tmp_path / "sample_z001.png", red=10, green=40)

    dataset = load_dataset(tmp_path, spacing=VoxelSpacing())

    assert dataset.green.z_indices == [1, 2]
    assert dataset.red.z_indices == [1, 2]
    assert dataset.green.data.shape == (2, 8, 8)
    assert dataset.red.data.shape == (2, 8, 8)
    # Top half has red signal only; bottom half has green signal only.
    assert float(dataset.red.data[0, 0, 0]) > 0.0
    assert float(dataset.green.data[0, 0, 0]) == 0.0
    assert float(dataset.green.data[0, 7, 0]) > 0.0
    assert float(dataset.red.data[0, 7, 0]) == 0.0


def test_combined_folder_ignores_non_red_green_images(tmp_path: Path) -> None:
    _write_mixed_rg(tmp_path / "sample_z001.png", red=25, green=45)
    _write_mixed_rg(tmp_path / "sample_z002.png", red=25, green=45, blue=10)

    dataset = load_dataset(tmp_path, spacing=VoxelSpacing())
    assert dataset.green.z_indices == [1]
    assert dataset.red.z_indices == [1]


def test_combined_folder_without_red_green_images_raises(tmp_path: Path) -> None:
    gray = np.full((8, 8), 90, dtype=np.uint8)
    iio.imwrite(tmp_path / "sample_z001.png", gray)

    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path, spacing=VoxelSpacing())
