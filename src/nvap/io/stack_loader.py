from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import logging
import os
import re
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from nvap.config.types import DEFAULT_SPACING, ChannelVolume, DatasetVolume, VoxelSpacing
from nvap.io.czi_loader import load_czi_channels, read_czi_shape
from nvap.preprocess._executor import get_executor
from nvap.runtime_optimization import configured_cpu_workers

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".tif", ".tiff", ".czi")
FILE_PATTERN = re.compile(
    r"_z(?P<z>\d+)(?P<channel>c[12])(?:\.png|\.tif|\.tiff)$",
    re.IGNORECASE,
)
COMBINED_FILE_PATTERN = re.compile(
    r"_z(?P<z>\d+)(?:c1\+2)?(?:\.png|\.tif|\.tiff)$",
    re.IGNORECASE,
)
CHANNEL_DIR = {"green": "Green", "red": "Red"}
CHANNEL_ID = {"green": "c1", "red": "c2"}
CHANNEL_RGB_INDEX = {"green": 1, "red": 0}
CHANNEL_STACK_INDEX = {"green": 1, "red": 0}
_COMBINED_RG_CACHE_MAX = 8
_combined_rg_cache = OrderedDict()
_COMBINED_RG_PRESENT_CACHE_MAX = 16
_combined_rg_present_cache = OrderedDict()


def _resolve_io_workers(item_count: int) -> int:
    count = int(max(0, item_count))
    if count < 4:
        return 1
    raw = os.environ.get("NVAP_IO_WORKERS", "").strip()
    if raw:
        try:
            requested = int(raw)
            if requested > 0:
                return max(1, min(requested, count))
        except ValueError:
            logger.warning("Invalid NVAP_IO_WORKERS=%r. Falling back to auto.", raw)
    return max(1, min(configured_cpu_workers(os.cpu_count() or 1), count))


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


@dataclass(frozen=True)
class DatasetProjectCandidate:
    """A loadable dataset found inside a larger project-set folder."""

    name: str
    root: Path
    channel_dirs: dict[str, Path]


def _candidate_segmented_roots(input_root: Path) -> list[Path]:
    candidates = [
        input_root / "Segmented",
        input_root / "Input" / "Segmented",
        input_root,
    ]
    if input_root.exists() and input_root.is_dir():
        child_dirs = [child for child in input_root.iterdir() if child.is_dir()]
        if len(child_dirs) == 1:
            wrapper = child_dirs[0]
            candidates.extend(
                [
                    wrapper / "Segmented",
                    wrapper / "Input" / "Segmented",
                    wrapper,
                ]
            )
    return candidates


def _is_supported_input_image(path: Path) -> bool:
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return False
    stem = path.stem.lower()
    if stem.endswith("_org"):
        return False
    return True


def _find_channel_dir(segmented_root: Path, channel_name: str) -> Path | None:
    target = CHANNEL_DIR[channel_name].lower()
    if not segmented_root.exists() or not segmented_root.is_dir():
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


def _iter_image_files(channel_source: Path) -> list[Path]:
    if channel_source.is_file():
        if _is_supported_input_image(channel_source):
            return [channel_source]
        return []
    if not channel_source.exists() or not channel_source.is_dir():
        return []
    files = [
        p
        for p in channel_source.iterdir()
        if p.is_file() and _is_supported_input_image(p)
    ]
    files.sort(key=lambda p: p.name.lower())
    return files


def _image_file_signature(files: list[Path]) -> tuple[int, int, str, str]:
    if not files:
        return (0, 0, "", "")
    newest_mtime_ns = 0
    for path in files:
        try:
            newest_mtime_ns = max(newest_mtime_ns, int(path.stat().st_mtime_ns))
        except OSError:
            continue
    return (
        int(len(files)),
        int(newest_mtime_ns),
        files[0].name.lower(),
        files[-1].name.lower(),
    )


def _combined_cache_key(channel_source: Path, files: list[Path]) -> tuple[str, tuple[int, int, str, str]]:
    try:
        source_key = str(channel_source.resolve())
    except OSError:
        source_key = str(channel_source)
    return (source_key, _image_file_signature(files))


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


def _is_red_green_only_stack(image: np.ndarray) -> bool:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return _is_red_green_only_image(arr)
    if arr.ndim == 4 and arr.shape[-1] >= 3:
        for idx in range(arr.shape[0]):
            if not _is_red_green_only_image(arr[idx]):
                return False
        return True
    return False


def _list_combined_rg_files(channel_source: Path) -> list[tuple[int, Path]]:
    files = _iter_image_files(channel_source)
    if not files:
        return []

    cache_key = _combined_cache_key(channel_source, files)
    cached = _combined_rg_cache.get(cache_key)
    if cached is not None:
        return list(cached)

    candidates: list[tuple[int, Path]] = []
    for file_path in files:
        match = COMBINED_FILE_PATTERN.search(file_path.name)
        if not match:
            continue
        z_index = int(match.group("z"))
        candidates.append((z_index, file_path))

    def _validate(item: tuple[int, Path]) -> tuple[int, Path] | None:
        z_index, file_path = item
        try:
            image = iio.imread(file_path)
        except Exception:
            return None
        if not _is_red_green_only_image(image):
            return None
        return z_index, file_path

    workers = _resolve_io_workers(len(candidates))
    if workers > 1:
        logger.info(
            "Validating combined red/green slices in parallel: files=%d workers=%d",
            len(candidates),
            workers,
        )
        with get_executor(workers, "nvap-io") as pool:
            validated = list(pool.map(_validate, candidates))
    else:
        validated = [_validate(item) for item in candidates]

    pairs = [item for item in validated if item is not None]
    pairs.sort(key=lambda item: item[0])

    while len(_combined_rg_cache) >= _COMBINED_RG_CACHE_MAX:
        _combined_rg_cache.popitem(last=False)
    _combined_rg_cache[cache_key] = list(pairs)
    while len(_combined_rg_present_cache) >= _COMBINED_RG_PRESENT_CACHE_MAX:
        _combined_rg_present_cache.popitem(last=False)
    _combined_rg_present_cache[cache_key] = bool(pairs)

    return pairs


def _has_combined_rg_files(channel_source: Path) -> bool:
    files = _iter_image_files(channel_source)
    if not files:
        return False
    cache_key = _combined_cache_key(channel_source, files)
    cached_pairs = _combined_rg_cache.get(cache_key)
    if cached_pairs is not None:
        return bool(cached_pairs)
    cached_present = _combined_rg_present_cache.get(cache_key)
    if cached_present is not None:
        return bool(cached_present)

    has_valid = False
    for file_path in files:
        match = COMBINED_FILE_PATTERN.search(file_path.name)
        if not match:
            continue
        try:
            image = iio.imread(file_path)
        except Exception:
            continue
        if _is_red_green_only_image(image):
            has_valid = True
            break

    while len(_combined_rg_present_cache) >= _COMBINED_RG_PRESENT_CACHE_MAX:
        _combined_rg_present_cache.popitem(last=False)
    _combined_rg_present_cache[cache_key] = has_valid
    return has_valid


def _list_channel_files(channel_source: Path, channel_name: str) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    expected_channel = CHANNEL_ID[channel_name]
    for file_path in _iter_image_files(channel_source):
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

    pairs = _list_combined_rg_files(channel_source)
    if not pairs:
        raise FileNotFoundError(
            f"No channel files found for '{channel_name}' in {channel_source}."
        )
    logger.info(
        "Using combined red/green RGB slices for channel '%s' from %s (count=%d).",
        channel_name,
        channel_source,
        len(pairs),
    )
    return pairs


def _has_exact_channel_sequence(channel_source: Path) -> bool:
    found_channels: set[str] = set()
    for file_path in _iter_image_files(channel_source):
        match = FILE_PATTERN.search(file_path.name)
        if not match:
            continue
        found_channels.add(match.group("channel").lower())
    return {"c1", "c2"}.issubset(found_channels)


def _load_volume_from_stack_file(stack_path: Path, channel_name: str) -> tuple[list[int], np.ndarray]:
    image = iio.imread(stack_path)
    arr = np.asarray(image)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        planes = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        # Either grayscale stack (z, y, x) or single RGB plane (y, x, c).
        if arr.shape[-1] in (3, 4):
            planes = arr[np.newaxis, ...]
        else:
            planes = arr
    elif arr.ndim == 4:
        if arr.shape[-1] in (3, 4):
            # RGB/RGBA stack (z, y, x, c)
            planes = arr
        elif arr.shape[2] >= 4 and arr.shape[3] >= 4:
            d0 = int(arr.shape[0])
            d1 = int(arr.shape[1])
            channel_axis: int | None = None

            # Common microscopy 2-channel stacks: prefer the dimension with size 2.
            if d0 == 2 and d1 != 2:
                channel_axis = 0
            elif d1 == 2 and d0 != 2:
                channel_axis = 1
            elif d0 in (2, 3, 4) and d1 not in (2, 3, 4):
                channel_axis = 0
            elif d1 in (2, 3, 4) and d0 not in (2, 3, 4):
                channel_axis = 1
            elif d0 in (2, 3, 4) and d1 in (2, 3, 4):
                # Ambiguous small dims: choose the smaller axis as channels.
                channel_axis = 0 if d0 <= d1 else 1

            if channel_axis is None:
                raise ValueError(f"Unsupported 4D stack layout: {arr.shape}")

            if channel_axis == 0:
                # Channel-leading stack (c, z, y, x)
                c_idx = int(min(arr.shape[0] - 1, CHANNEL_STACK_INDEX[channel_name]))
                planes = np.asarray(arr[c_idx, ...])
            else:
                # Channel-middle stack (z, c, y, x)
                c_idx = int(min(arr.shape[1] - 1, CHANNEL_STACK_INDEX[channel_name]))
                planes = np.asarray(arr[:, c_idx, ...])
        else:
            raise ValueError(f"Unsupported 4D stack layout: {arr.shape}")
    else:
        raise ValueError(f"Unsupported stack dimensions: {arr.shape}")

    slices: list[np.ndarray] = []
    for idx in range(int(planes.shape[0])):
        slices.append(_extract_and_normalize_plane(np.asarray(planes[idx]), channel_name))
    volume = np.stack(slices, axis=0).astype(np.float32, copy=False)
    z_indices = list(range(1, int(volume.shape[0]) + 1))
    return z_indices, volume


def _load_channel_from_single_stack(channel_name: str, channel_source: Path) -> ChannelVolume:
    image_files = _iter_image_files(channel_source)
    if not image_files:
        raise FileNotFoundError(f"No image files found in {channel_source}.")
    if len(image_files) != 1:
        raise FileNotFoundError(
            f"No z-indexed files found for '{channel_name}' in {channel_source}, and "
            f"cannot infer a single stack from {len(image_files)} image files."
        )

    stack_path = image_files[0]
    z_indices, volume = _load_volume_from_stack_file(stack_path, channel_name)
    logger.info(
        "Loaded channel '%s' from stack file: %s slices=%d shape=%s",
        channel_name,
        stack_path,
        len(z_indices),
        volume.shape,
    )
    return ChannelVolume(
        name=channel_name,
        data=volume,
        z_indices=z_indices,
        spacing=DEFAULT_SPACING,
    )


def _load_channel(
    channel_name: str, channel_source: Path, spacing: VoxelSpacing
) -> ChannelVolume:
    logger.debug("Loading channel '%s' from %s", channel_name, channel_source)
    try:
        z_and_files = _list_channel_files(channel_source, channel_name)
        z_indices = [z for z, _ in z_and_files]

        def _read_slice(item: tuple[int, Path]) -> tuple[int, np.ndarray]:
            z, file_path = item
            img = iio.imread(file_path)
            return z, _extract_and_normalize_plane(img, channel_name)

        workers = _resolve_io_workers(len(z_and_files))
        if workers > 1:
            logger.info(
                "Loading channel '%s' slices in parallel: files=%d workers=%d",
                channel_name,
                len(z_and_files),
                workers,
            )
            with get_executor(workers, "nvap-io") as pool:
                loaded = list(pool.map(_read_slice, z_and_files))
        else:
            loaded = [_read_slice(item) for item in z_and_files]
        slices = [plane for _, plane in loaded]
        volume = np.stack(slices, axis=0).astype(np.float32, copy=False)
    except FileNotFoundError:
        stack_channel = _load_channel_from_single_stack(channel_name, channel_source)
        return ChannelVolume(
            name=channel_name,
            data=stack_channel.data,
            z_indices=list(stack_channel.z_indices),
            spacing=spacing,
        )

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


def _load_combined_channels(
    channel_source: Path,
    spacing: VoxelSpacing,
) -> tuple[ChannelVolume, ChannelVolume] | None:
    if channel_source.is_file():
        if channel_source.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            return None
        image = iio.imread(channel_source)
        arr = np.asarray(image)
        if not _is_red_green_only_stack(arr):
            return None

        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            planes = arr[np.newaxis, ...]
        elif arr.ndim == 4 and arr.shape[-1] in (3, 4):
            planes = arr
        else:
            return None

        green_slices: list[np.ndarray] = []
        red_slices: list[np.ndarray] = []
        depth = int(planes.shape[0])
        for idx in range(depth):
            plane = np.asarray(planes[idx])
            green_slices.append(_extract_and_normalize_plane(plane, "green"))
            red_slices.append(_extract_and_normalize_plane(plane, "red"))

        z_indices = list(range(1, depth + 1))
        green_volume = np.stack(green_slices, axis=0).astype(np.float32, copy=False)
        red_volume = np.stack(red_slices, axis=0).astype(np.float32, copy=False)
        logger.info(
            "Loaded combined red/green stack once: %s slices=%d shape=%s",
            channel_source,
            depth,
            green_volume.shape,
        )
        return (
            ChannelVolume(
                name="green",
                data=green_volume,
                z_indices=z_indices,
                spacing=spacing,
            ),
            ChannelVolume(
                name="red",
                data=red_volume,
                z_indices=z_indices,
                spacing=spacing,
            ),
        )

    image_files = _iter_image_files(channel_source)
    if len(image_files) == 1:
        return _load_combined_channels(image_files[0], spacing)

    z_and_files = _list_combined_rg_files(channel_source)
    if not z_and_files:
        return None

    z_indices = [z for z, _ in z_and_files]
    if len(set(z_indices)) != len(z_indices):
        # Repeated z-indices usually indicate channel-tagged c1/c2 files.
        return None

    def _read_combined_slice(item: tuple[int, Path]) -> tuple[int, np.ndarray, np.ndarray]:
        z, file_path = item
        img = iio.imread(file_path)
        return (
            z,
            _extract_and_normalize_plane(img, "green"),
            _extract_and_normalize_plane(img, "red"),
        )

    workers = _resolve_io_workers(len(z_and_files))
    if workers > 1:
        logger.info(
            "Loading combined red/green slices in parallel: files=%d workers=%d",
            len(z_and_files),
            workers,
        )
        with get_executor(workers, "nvap-io") as pool:
            loaded = list(pool.map(_read_combined_slice, z_and_files))
    else:
        loaded = [_read_combined_slice(item) for item in z_and_files]

    green_slices = [green for _, green, _ in loaded]
    red_slices = [red for _, _, red in loaded]

    green_volume = np.stack(green_slices, axis=0).astype(np.float32, copy=False)
    red_volume = np.stack(red_slices, axis=0).astype(np.float32, copy=False)
    logger.info(
        "Loaded combined red/green slices once: source=%s slices=%d shape=%s",
        channel_source,
        len(z_indices),
        green_volume.shape,
    )
    return (
        ChannelVolume(
            name="green",
            data=green_volume,
            z_indices=z_indices,
            spacing=spacing,
        ),
        ChannelVolume(
            name="red",
            data=red_volume,
            z_indices=z_indices,
            spacing=spacing,
        ),
    )


def _shared_z_range(green: ChannelVolume, red: ChannelVolume) -> tuple[int, int]:
    green_min, green_max = min(green.z_indices), max(green.z_indices)
    red_min, red_max = min(red.z_indices), max(red.z_indices)
    shared_min = max(green_min, red_min)
    shared_max = min(green_max, red_max)
    if shared_min > shared_max:
        raise ValueError("Green and red channels do not share an overlapping z-range.")
    return shared_min, shared_max


def _channel_stack_stats(channel_name: str, channel_source: Path) -> ChannelStackStats:
    if channel_source.is_file() and channel_source.suffix.lower() == ".czi":
        channels, depth, height, width = read_czi_shape(channel_source)
        if channels < 2:
            raise ValueError(
                f"NVAP requires at least two CZI channels; {channel_source.name} contains {channels}."
            )
        return ChannelStackStats(
            name=channel_name,
            slice_count=int(depth),
            z_min=1,
            z_max=int(depth),
            full_slice_count=int(depth),
            missing_count=0,
            width=int(width),
            height=int(height),
        )
    try:
        z_and_files = _list_channel_files(channel_source, channel_name)
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
    except FileNotFoundError:
        stack_channel = _load_channel_from_single_stack(channel_name, channel_source)
        depth, height, width = stack_channel.data.shape
        return ChannelStackStats(
            name=channel_name,
            slice_count=int(depth),
            z_min=int(min(stack_channel.z_indices)),
            z_max=int(max(stack_channel.z_indices)),
            full_slice_count=int(depth),
            missing_count=0,
            width=int(width),
            height=int(height),
        )


def load_dataset(
    input_root: str | Path,
    spacing: VoxelSpacing | None = None,
    channel_overrides: dict[str, str | Path] | None = None,
) -> DatasetVolume:
    root = Path(input_root).resolve()
    logger.info("Loading dataset from root: %s", root)
    channel_dirs = resolve_channel_dirs(root, channel_overrides=channel_overrides)

    green_source = channel_dirs["green"]
    red_source = channel_dirs["red"]
    if green_source == red_source and green_source.suffix.lower() == ".czi":
        green, red, _ = load_czi_channels(green_source, spacing=spacing)
        shared = _shared_z_range(green, red)
        logger.info("Shared z range: %s", shared)
        return DatasetVolume(green=green, red=red, shared_z_range=shared)

    effective_spacing = spacing or DEFAULT_SPACING
    combined = None
    if green_source == red_source and not _has_exact_channel_sequence(green_source):
        combined = _load_combined_channels(green_source, effective_spacing)
    if combined is not None:
        green, red = combined
    else:
        green = _load_channel("green", green_source, effective_spacing)
        red = _load_channel("red", red_source, effective_spacing)
    shared = _shared_z_range(green, red)
    logger.info("Shared z range: %s", shared)
    return DatasetVolume(green=green, red=red, shared_z_range=shared)


def resolve_channel_dirs(
    input_root: str | Path,
    channel_overrides: dict[str, str | Path] | None = None,
) -> dict[str, Path]:
    root = Path(input_root).resolve()
    channel_dirs: dict[str, Path] = {}

    if root.is_file() and root.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
        channel_dirs["green"] = root
        channel_dirs["red"] = root
        logger.info("Using single stack source for both channels: %s", root)
        return channel_dirs

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
            if _has_exact_channel_sequence(candidate):
                channel_dirs["green"] = candidate
                channel_dirs["red"] = candidate
                logger.info(
                    "Auto-detected exact c1/c2 sequence dataset under: %s",
                    candidate,
                )
                return channel_dirs
            if _has_combined_rg_files(candidate):
                channel_dirs["green"] = candidate
                channel_dirs["red"] = candidate
                logger.info(
                    "Auto-detected combined red/green RGB dataset under: %s",
                    candidate,
                )
                return channel_dirs
            # Also allow a single combined RGB stack TIFF/PNG file.
            image_files = _iter_image_files(candidate)
            if len(image_files) == 1:
                if image_files[0].suffix.lower() == ".czi":
                    channel_dirs["green"] = image_files[0]
                    channel_dirs["red"] = image_files[0]
                    logger.info("Auto-detected CZI stack under: %s", candidate)
                    return channel_dirs
                try:
                    img = iio.imread(image_files[0])
                except Exception:
                    img = None
                if img is not None and _is_red_green_only_stack(np.asarray(img)):
                    channel_dirs["green"] = candidate
                    channel_dirs["red"] = candidate
                    logger.info(
                        "Auto-detected combined red/green stack under: %s (file=%s)",
                        candidate,
                        image_files[0].name,
                    )
                    return channel_dirs
        raise FileNotFoundError(
            f"Could not auto-detect channel folders or combined red/green RGB slices under: {root}. "
            "Expected Green and Red directories in a Segmented folder, RGB PNG/TIFF slices, "
            "or a single RGB PNG/TIFF stack with red/green-only pixels."
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


def discover_dataset_projects(
    input_root: str | Path,
    *,
    max_depth: int = 2,
) -> list[DatasetProjectCandidate]:
    """Find loadable dataset folders under ``input_root``.

    The normal loader opens one dataset root. A project set is a parent folder
    whose children are each normal dataset roots, e.g. ten image-series folders.
    Discovery is intentionally shallow and stops descending once a loadable
    dataset is found so nested ``Segmented``/``Green``/``Red`` folders do not
    appear as duplicate projects.
    """
    root = Path(input_root).resolve()
    if root.is_file():
        try:
            channel_dirs = resolve_channel_dirs(root)
        except Exception:
            return []
        return [
            DatasetProjectCandidate(
                name=root.stem,
                root=root,
                channel_dirs=channel_dirs,
            )
        ]
    if not root.exists() or not root.is_dir():
        return []

    found: list[DatasetProjectCandidate] = []
    seen: set[Path] = set()

    def _add_candidate(path: Path, channel_dirs: dict[str, Path]) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        found.append(
            DatasetProjectCandidate(
                name=resolved.name,
                root=resolved,
                channel_dirs=channel_dirs,
            )
        )

    def _walk(path: Path, depth: int) -> None:
        try:
            channel_dirs = resolve_channel_dirs(path)
        except Exception:
            channel_dirs = None
        if channel_dirs is not None:
            _add_candidate(path, channel_dirs)
            return
        if depth >= int(max_depth):
            return
        try:
            children = [child for child in path.iterdir() if child.is_dir()]
        except OSError:
            return
        for child in sorted(children, key=lambda p: p.name.lower()):
            _walk(child, depth + 1)

    for child in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        _walk(child, 1)

    if not found:
        try:
            channel_dirs = resolve_channel_dirs(root)
        except Exception:
            channel_dirs = None
        if channel_dirs is not None:
            _add_candidate(root, channel_dirs)

    return found
