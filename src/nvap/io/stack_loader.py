from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from nvap.config.types import DEFAULT_SPACING, ChannelVolume, DatasetVolume, VoxelSpacing

logger = logging.getLogger(__name__)

FILE_PATTERN = re.compile(r"_z(?P<z>\d+)(?P<channel>c[12])\.png$", re.IGNORECASE)
COMBINED_FILE_PATTERN = re.compile(r"_z(?P<z>\d+)(?:c[12])?\.png$", re.IGNORECASE)
CHANNEL_DIR = {"green": "Green", "red": "Red"}
CHANNEL_ID = {"green": "c1", "red": "c2"}
CHANNEL_RGB_INDEX = {"green": 1, "red": 0}


@dataclass(frozen=True)
class ChannelStackStats:
    name: str
    slice_count: int
    z_min: int
    z_max: int
    full_slice_count: int
    missing_count: int
    width: int
    height: int

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    @property
    def raw_voxel_count(self) -> int:
        return self.slice_count * self.pixel_count

    @property
    def full_voxel_count(self) -> int:
        return self.full_slice_count * self.pixel_count


@dataclass(frozen=True)
class DatasetStackStats:
    green: ChannelStackStats
    red: ChannelStackStats
    shared_z_range: tuple[int, int]

    @property
    def total_raw_voxels(self) -> int:
        return self.green.raw_voxel_count + self.red.raw_voxel_count

    @property
    def total_full_voxels(self) -> int:
        return self.green.full_voxel_count + self.red.full_voxel_count

    @property
    def total_missing_slices(self) -> int:
        return self.green.missing_count + self.red.missing_count


def _candidate_segmented_roots(input_root: Path) -> list[Path]:
    return [
        input_root / "Segmented",
        input_root / "Input" / "Segmented",
        input_root,
    ]


def _find_channel_dir(segmented_root: Path, channel_name: str) -> Path | None:
    target = CHANNEL_DIR[channel_name].lower()
    if not segmented_root.exists():
        return None
    for child in segmented_root.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child
    return None


def _extract_and_normalize_plane(image: np.ndarray, channel_name: str) -> np.ndarray:
    if image.ndim == 2:
        plane = image
    elif image.ndim == 3:
        rgb_index = CHANNEL_RGB_INDEX[channel_name]
        if image.shape[-1] <= rgb_index:
            raise ValueError(f"Invalid channel count for {channel_name}: {image.shape}.")
        plane = image[..., rgb_index]
    else:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")

    if np.issubdtype(plane.dtype, np.integer):
        denom = float(np.iinfo(plane.dtype).max)
    else:
        max_val = float(np.nanmax(plane))
        denom = max(max_val, 1.0)

    normalized = plane.astype(np.float32) / denom
    return np.clip(normalized, 0.0, 1.0)


def _is_red_green_only_image(image: np.ndarray) -> bool:
    if image.ndim != 3 or image.shape[-1] < 3:
        return False
    rgb = image[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        eps = 1.0e-6
        red = rgb[..., 0] > eps
        green = rgb[..., 1] > eps
        blue = rgb[..., 2] > eps
    else:
        red = rgb[..., 0] != 0
        green = rgb[..., 1] != 0
        blue = rgb[..., 2] != 0
    # Allow pure red, pure green, or black background only.
    return not bool(np.any(blue | (red & green)))


def _list_combined_rg_files(channel_dir: Path) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    for file_path in channel_dir.glob("*.png"):
        match = COMBINED_FILE_PATTERN.search(file_path.name)
        if not match:
            continue
        image = iio.imread(file_path)
        if not _is_red_green_only_image(image):
            continue
        z_index = int(match.group("z"))
        pairs.append((z_index, file_path))
    pairs.sort(key=lambda item: item[0])
    return pairs


def _list_channel_files(channel_dir: Path, channel_name: str) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    expected_channel = CHANNEL_ID[channel_name]
    for file_path in channel_dir.glob("*.png"):
        match = FILE_PATTERN.search(file_path.name)
        if not match:
            continue
        z_index = int(match.group("z"))
        channel_id = match.group("channel").lower()
        if channel_id != expected_channel:
            continue
        pairs.append((z_index, file_path))
    pairs.sort(key=lambda item: item[0])
    if pairs:
        return pairs

    pairs = _list_combined_rg_files(channel_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No channel files found for '{channel_name}' in {channel_dir}."
        )
    logger.info(
        "Using combined red/green RGB slices for channel '%s' from %s (count=%d).",
        channel_name,
        channel_dir,
        len(pairs),
    )
    return pairs


def _load_channel(
    channel_name: str, channel_dir: Path, spacing: VoxelSpacing
) -> ChannelVolume:
    logger.debug("Loading channel '%s' from %s", channel_name, channel_dir)
    z_and_files = _list_channel_files(channel_dir, channel_name)
    z_indices = [z for z, _ in z_and_files]
    slices = []
    for _, file_path in z_and_files:
        img = iio.imread(file_path)
        slices.append(_extract_and_normalize_plane(img, channel_name))
    volume = np.stack(slices, axis=0).astype(np.float32, copy=False)
    logger.info(
        "Loaded channel '%s': slices=%d, shape=%s, z_range=(%d,%d)",
        channel_name,
        len(z_indices),
        volume.shape,
        min(z_indices),
        max(z_indices),
    )
    return ChannelVolume(
        name=channel_name,
        data=volume,
        z_indices=z_indices,
        spacing=spacing,
    )


def _shared_z_range(green: ChannelVolume, red: ChannelVolume) -> tuple[int, int]:
    green_min, green_max = min(green.z_indices), max(green.z_indices)
    red_min, red_max = min(red.z_indices), max(red.z_indices)
    shared_min = max(green_min, red_min)
    shared_max = min(green_max, red_max)
    if shared_min > shared_max:
        raise ValueError("Green and red channels do not share an overlapping z-range.")
    return shared_min, shared_max


def _channel_stack_stats(channel_name: str, channel_dir: Path) -> ChannelStackStats:
    z_and_files = _list_channel_files(channel_dir, channel_name)
    z_values = [z for z, _ in z_and_files]
    z_min = min(z_values)
    z_max = max(z_values)
    full_slice_count = z_max - z_min + 1
    missing_count = full_slice_count - len(z_values)

    first_image = iio.imread(z_and_files[0][1])
    if first_image.ndim == 2:
        height, width = first_image.shape
    elif first_image.ndim == 3:
        height, width = first_image.shape[0], first_image.shape[1]
    else:
        raise ValueError(f"Unsupported image dimensions for stats: {first_image.shape}")

    return ChannelStackStats(
        name=channel_name,
        slice_count=len(z_values),
        z_min=z_min,
        z_max=z_max,
        full_slice_count=full_slice_count,
        missing_count=missing_count,
        width=int(width),
        height=int(height),
    )


def load_dataset(
    input_root: str | Path,
    spacing: VoxelSpacing = DEFAULT_SPACING,
    channel_overrides: dict[str, str | Path] | None = None,
) -> DatasetVolume:
    root = Path(input_root).resolve()
    logger.info("Loading dataset from root: %s", root)
    channel_dirs = resolve_channel_dirs(root, channel_overrides=channel_overrides)

    green = _load_channel("green", channel_dirs["green"], spacing)
    red = _load_channel("red", channel_dirs["red"], spacing)
    shared = _shared_z_range(green, red)
    logger.info("Shared z range: %s", shared)
    return DatasetVolume(green=green, red=red, shared_z_range=shared)


def resolve_channel_dirs(
    input_root: str | Path,
    channel_overrides: dict[str, str | Path] | None = None,
) -> dict[str, Path]:
    root = Path(input_root).resolve()
    channel_dirs: dict[str, Path] = {}

    if channel_overrides:
        logger.info("Using manual channel overrides.")
        for channel_name in ("green", "red"):
            if channel_name not in channel_overrides:
                raise ValueError(
                    "channel_overrides must include both 'green' and 'red' keys."
                )
            channel_dir = Path(channel_overrides[channel_name]).resolve()
            if not channel_dir.exists():
                raise FileNotFoundError(f"Channel path does not exist: {channel_dir}")
            channel_dirs[channel_name] = channel_dir
        return channel_dirs

    selected_root = None
    for candidate in _candidate_segmented_roots(root):
        logger.debug("Checking segmented candidate: %s", candidate)
        green_dir = _find_channel_dir(candidate, "green")
        red_dir = _find_channel_dir(candidate, "red")
        if green_dir and red_dir:
            selected_root = candidate
            break

    if selected_root is None:
        for candidate in _candidate_segmented_roots(root):
            if not candidate.exists() or not candidate.is_dir():
                continue
            combined_files = _list_combined_rg_files(candidate)
            if not combined_files:
                continue
            channel_dirs["green"] = candidate
            channel_dirs["red"] = candidate
            logger.info(
                "Auto-detected combined red/green RGB dataset under: %s (slices=%d)",
                candidate,
                len(combined_files),
            )
            return channel_dirs
        raise FileNotFoundError(
            f"Could not auto-detect channel folders or combined red/green RGB slices under: {root}. "
            "Expected Green and Red directories in a Segmented folder, or RGB PNG slices with only red/green pixels."
        )
    green_dir = _find_channel_dir(selected_root, "green")
    red_dir = _find_channel_dir(selected_root, "red")
    if green_dir is None or red_dir is None:
        raise FileNotFoundError("Auto-detected segmented root, but channel directories are missing.")
    channel_dirs["green"] = green_dir
    channel_dirs["red"] = red_dir
    logger.info("Auto-detected segmented root: %s", selected_root)
    return channel_dirs


def inspect_dataset_stats(
    input_root: str | Path,
    channel_overrides: dict[str, str | Path] | None = None,
) -> DatasetStackStats:
    channel_dirs = resolve_channel_dirs(input_root, channel_overrides=channel_overrides)
    green = _channel_stack_stats("green", channel_dirs["green"])
    red = _channel_stack_stats("red", channel_dirs["red"])
    shared_min = max(green.z_min, red.z_min)
    shared_max = min(green.z_max, red.z_max)
    if shared_min > shared_max:
        raise ValueError("Green and red channels do not share an overlapping z-range.")
    stats = DatasetStackStats(green=green, red=red, shared_z_range=(shared_min, shared_max))
    logger.debug(
        "Dataset stats: green_slices=%d red_slices=%d missing=%d total_full_voxels=%d",
        green.slice_count,
        red.slice_count,
        stats.total_missing_slices,
        stats.total_full_voxels,
    )
    return stats
