from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from nvap.config.types import VoxelSpacing
from nvap.io.stack_loader import discover_dataset_projects, inspect_dataset_stats, load_dataset


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


def _write_gray_stack(path: Path, values: list[int]) -> None:
    frames = []
    for value in values:
        plane = np.full((8, 8), value, dtype=np.uint8)
        frames.append(plane)
    stack = np.stack(frames, axis=0)
    if path.suffix.lower() in {".tif", ".tiff"}:
        iio.imwrite(path, stack, plugin="tifffile", photometric="minisblack")
    else:
        iio.imwrite(path, stack)


def _write_rgb_stack(path: Path, red_values: list[int], green_values: list[int]) -> None:
    frames = []
    for red, green in zip(red_values, green_values):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[:4, :, 0] = red
        frame[4:, :, 1] = green
        frames.append(frame)
    stack = np.stack(frames, axis=0)
    iio.imwrite(path, stack)


def _write_hyperstack_zcyx(path: Path, red_values: list[int], green_values: list[int]) -> None:
    depth = len(red_values)
    arr = np.zeros((depth, 2, 8, 8), dtype=np.uint16)
    for z, (r_val, g_val) in enumerate(zip(red_values, green_values)):
        arr[z, 0, :, :] = int(r_val)
        arr[z, 1, :, :] = int(g_val)
    iio.imwrite(path, arr, plugin="tifffile", photometric="minisblack")


def _write_hyperstack_czyx(path: Path, red_values: list[int], green_values: list[int]) -> None:
    depth = len(red_values)
    arr = np.zeros((2, depth, 8, 8), dtype=np.uint16)
    for z, (r_val, g_val) in enumerate(zip(red_values, green_values)):
        arr[0, z, :, :] = int(r_val)
        arr[1, z, :, :] = int(g_val)
    iio.imwrite(path, arr, plugin="tifffile", photometric="minisblack")


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


def test_discover_dataset_projects_finds_multiple_child_series(tmp_path: Path) -> None:
    for name, green, red in [("sample_a", 30, 60), ("sample_b", 80, 120)]:
        child = tmp_path / name
        child.mkdir()
        _write_mixed_rg(child / "series_z001.png", red=red, green=green)
        _write_mixed_rg(child / "series_z002.png", red=red + 1, green=green + 1)

    entries = discover_dataset_projects(tmp_path)

    assert [entry.name for entry in entries] == ["sample_a", "sample_b"]
    assert all(entry.root.parent == tmp_path for entry in entries)
    assert all(set(entry.channel_dirs) == {"green", "red"} for entry in entries)


def test_load_dataset_from_wrapped_sequence_folder_ignores_exports(tmp_path: Path) -> None:
    inner = tmp_path / "7951-3_M_CL_cortex_25X"
    inner.mkdir()

    for z, green, red in [(1, 30, 60), (2, 70, 120)]:
        _write_rgb(inner / f"7951-3_M_CL_cortex_25X_z{z:03d}c1.png", r=0, g=green, b=0)
        _write_rgb(inner / f"7951-3_M_CL_cortex_25X_z{z:03d}c2.png", r=red, g=0, b=0)
        _write_mixed_rg(
            inner / f"7951-3_M_CL_cortex_25X_z{z:03d}c1+2.png",
            red=255,
            green=255,
        )
        _write_rgb(inner / f"7951-3_M_CL_cortex_25X_z{z:03d}c1_ORG.png", r=0, g=255, b=0)
        _write_rgb(inner / f"7951-3_M_CL_cortex_25X_z{z:03d}c2_ORG.png", r=255, g=0, b=0)

    dataset = load_dataset(tmp_path, spacing=VoxelSpacing())

    assert dataset.green.z_indices == [1, 2]
    assert dataset.red.z_indices == [1, 2]
    assert dataset.green.data.shape == (2, 8, 8)
    assert dataset.red.data.shape == (2, 8, 8)
    assert float(dataset.green.data[0, 0, 0]) == pytest.approx(30 / 255)
    assert float(dataset.green.data[1, 0, 0]) == pytest.approx(70 / 255)
    assert float(dataset.red.data[0, 0, 0]) == pytest.approx(60 / 255)
    assert float(dataset.red.data[1, 0, 0]) == pytest.approx(120 / 255)


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


def test_load_dataset_supports_tiff_slice_sequences(tmp_path: Path) -> None:
    green_dir = tmp_path / "Segmented" / "Green"
    red_dir = tmp_path / "Segmented" / "Red"
    green_dir.mkdir(parents=True)
    red_dir.mkdir(parents=True)

    _write_rgb(green_dir / "sample_z002c1.tif", r=0, g=80, b=0)
    _write_rgb(green_dir / "sample_z001c1.tif", r=0, g=20, b=0)
    _write_rgb(red_dir / "sample_z001c2.tif", r=40, g=0, b=0)
    _write_rgb(red_dir / "sample_z002c2.tif", r=90, g=0, b=0)

    dataset = load_dataset(tmp_path, spacing=VoxelSpacing())
    assert dataset.green.z_indices == [1, 2]
    assert dataset.red.z_indices == [1, 2]
    assert dataset.green.data.shape == (2, 8, 8)
    assert dataset.red.data.shape == (2, 8, 8)
    assert float(dataset.green.data[0, 0, 0]) < float(dataset.green.data[1, 0, 0])
    assert float(dataset.red.data[0, 0, 0]) < float(dataset.red.data[1, 0, 0])


def test_load_dataset_supports_single_tiff_stack_per_channel(tmp_path: Path) -> None:
    green_dir = tmp_path / "Segmented" / "Green"
    red_dir = tmp_path / "Segmented" / "Red"
    green_dir.mkdir(parents=True)
    red_dir.mkdir(parents=True)

    _write_gray_stack(green_dir / "green_stack.tiff", [20, 40, 60, 80])
    _write_gray_stack(red_dir / "red_stack.tiff", [10, 25, 45, 70])

    dataset = load_dataset(tmp_path, spacing=VoxelSpacing())
    assert dataset.green.z_indices == [1, 2, 3, 4]
    assert dataset.red.z_indices == [1, 2, 3, 4]
    assert dataset.green.data.shape == (4, 8, 8)
    assert dataset.red.data.shape == (4, 8, 8)
    assert float(dataset.green.data[0, 0, 0]) < float(dataset.green.data[3, 0, 0])
    assert float(dataset.red.data[0, 0, 0]) < float(dataset.red.data[3, 0, 0])


def test_load_dataset_supports_single_combined_rgb_tiff_stack(tmp_path: Path) -> None:
    _write_rgb_stack(
        tmp_path / "combined_stack.tif",
        red_values=[20, 60],
        green_values=[40, 10],
    )

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


def test_load_dataset_supports_single_hyperstack_zcyx_tiff(tmp_path: Path) -> None:
    stack_path = tmp_path / "hyper_zcyx.tif"
    _write_hyperstack_zcyx(stack_path, red_values=[100, 220], green_values=[300, 500])

    dataset = load_dataset(
        tmp_path,
        spacing=VoxelSpacing(),
        channel_overrides={"green": stack_path, "red": stack_path},
    )

    assert dataset.green.data.shape == (2, 8, 8)
    assert dataset.red.data.shape == (2, 8, 8)
    assert dataset.green.z_indices == [1, 2]
    assert dataset.red.z_indices == [1, 2]
    assert float(dataset.green.data[0, 0, 0]) > float(dataset.red.data[0, 0, 0])
    assert float(dataset.green.data[1, 0, 0]) > float(dataset.green.data[0, 0, 0])
    assert float(dataset.red.data[1, 0, 0]) > float(dataset.red.data[0, 0, 0])


def test_load_dataset_supports_single_hyperstack_czyx_tiff(tmp_path: Path) -> None:
    stack_path = tmp_path / "hyper_czyx.tif"
    _write_hyperstack_czyx(stack_path, red_values=[90, 180, 240], green_values=[200, 350, 600])

    dataset = load_dataset(
        tmp_path,
        spacing=VoxelSpacing(),
        channel_overrides={"green": stack_path, "red": stack_path},
    )

    assert dataset.green.data.shape == (3, 8, 8)
    assert dataset.red.data.shape == (3, 8, 8)
    assert dataset.green.z_indices == [1, 2, 3]
    assert dataset.red.z_indices == [1, 2, 3]
    assert float(dataset.green.data[0, 0, 0]) > float(dataset.red.data[0, 0, 0])
    assert float(dataset.green.data[2, 0, 0]) > float(dataset.green.data[0, 0, 0])
    assert float(dataset.red.data[2, 0, 0]) > float(dataset.red.data[0, 0, 0])


def test_load_dataset_supports_file_sources_in_channel_overrides(tmp_path: Path) -> None:
    red_file = tmp_path / "vasculature_stack.tif"
    green_file = tmp_path / "microglia_stack.tif"
    _write_gray_stack(red_file, [10, 30, 60])
    _write_gray_stack(green_file, [20, 40, 80])

    dataset = load_dataset(
        tmp_path,
        spacing=VoxelSpacing(),
        channel_overrides={"red": red_file, "green": green_file},
    )
    assert dataset.green.z_indices == [1, 2, 3]
    assert dataset.red.z_indices == [1, 2, 3]
    assert dataset.green.data.shape == (3, 8, 8)
    assert dataset.red.data.shape == (3, 8, 8)
    assert float(dataset.green.data[0, 0, 0]) < float(dataset.green.data[2, 0, 0])
    assert float(dataset.red.data[0, 0, 0]) < float(dataset.red.data[2, 0, 0])


def test_inspect_dataset_stats_supports_file_sources(tmp_path: Path) -> None:
    red_file = tmp_path / "red_stack.tiff"
    green_file = tmp_path / "green_stack.tiff"
    _write_gray_stack(red_file, [10, 20, 30, 50])
    _write_gray_stack(green_file, [5, 15, 25, 35])

    stats = inspect_dataset_stats(
        tmp_path,
        channel_overrides={"red": red_file, "green": green_file},
    )
    assert stats.green.slice_count == 4
    assert stats.red.slice_count == 4
    assert stats.green.missing_count == 0
    assert stats.red.missing_count == 0
    assert stats.shared_z_range == (1, 4)


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
