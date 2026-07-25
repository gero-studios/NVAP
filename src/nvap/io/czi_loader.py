"""CZI image and physical-scaling support."""
from __future__ import annotations

import logging
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from nvap.config.types import ChannelVolume, VoxelSpacing

logger = logging.getLogger(__name__)

_UNIT_TO_UM = {
    "m": 1_000_000.0,
    "meter": 1_000_000.0,
    "metre": 1_000_000.0,
    "mm": 1_000.0,
    "millimeter": 1_000.0,
    "millimetre": 1_000.0,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "micrometer": 1.0,
    "micrometre": 1.0,
    "nm": 0.001,
    "nanometer": 0.001,
    "nanometre": 0.001,
}


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def parse_czi_spacing(metadata: ElementTree.Element | str | bytes) -> VoxelSpacing:
    """Extract Zeiss X/Y/Z scaling metadata and return micrometres."""
    if isinstance(metadata, ElementTree.Element):
        root = metadata
    else:
        root = ElementTree.fromstring(metadata)

    values: dict[str, float] = {}
    for element in root.iter():
        if _local_name(element.tag).lower() != "distance":
            continue
        axis = str(element.attrib.get("Id") or element.attrib.get("id") or "").upper()
        if axis not in {"X", "Y", "Z"}:
            continue
        raw_value: str | None = None
        raw_unit: str | None = None
        for child in element.iter():
            name = _local_name(child.tag).lower()
            if name == "value" and child.text:
                raw_value = child.text.strip()
            elif name in {"defaultunitformat", "unit"} and child.text:
                raw_unit = child.text.strip()
        if raw_value is None:
            continue
        value = float(raw_value)
        has_explicit_unit = any(
            _local_name(child.tag).lower() == "unit" and bool(child.text)
            for child in element.iter()
        )
        # DefaultUnitFormat is a display preference in standard CZI metadata.
        # Values near 1e-7 remain metres even when that display tag says um.
        if not has_explicit_unit and abs(value) < 1.0e-3:
            raw_unit = "m"
        # Zeiss Scaling/Items/Distance values are metres when the unit is omitted.
        unit = (raw_unit or "m").strip().lower().replace("µ", "u").replace("μ", "u")
        factor = _UNIT_TO_UM.get(unit)
        if factor is None:
            logger.warning("Unknown CZI spacing unit %r for axis %s; assuming metres.", raw_unit, axis)
            factor = 1_000_000.0
        spacing_um = value * factor
        if np.isfinite(spacing_um) and spacing_um > 0:
            values[axis] = float(spacing_um)

    missing = sorted({"X", "Y", "Z"} - set(values))
    if missing:
        raise ValueError(f"CZI metadata is missing positive spacing for axis/axes: {', '.join(missing)}")
    return VoxelSpacing(x_um=values["X"], y_um=values["Y"], z_um=values["Z"])


def _normalise_channel_text(value: str) -> str:
    return " ".join(
        str(value).strip().lower().replace("-", " ").replace("_", " ").split()
    )


def _channel_role_scores(channel: ElementTree.Element) -> tuple[float, float]:
    """Return (green, red) evidence scores from one CZI channel element."""
    parts = [str(value) for value in channel.attrib.values()]
    color: str | None = None
    excitation: float | None = None
    emission: float | None = None
    for child in channel.iter():
        name = _local_name(child.tag).lower()
        if child.text and name in {
            "name",
            "description",
            "fluor",
            "dyeid",
            "dye",
            "color",
        }:
            parts.append(child.text.strip())
        if child.text and name == "color":
            color = child.text.strip()
        elif child.text and name == "excitationwavelength":
            try:
                excitation = float(child.text)
            except ValueError:
                pass
        elif child.text and name == "emissionwavelength":
            try:
                emission = float(child.text)
            except ValueError:
                pass

    text = _normalise_channel_text(" ".join(parts))
    green = 0.0
    red = 0.0
    green_terms = ("egfp", "gfp", "fitc", "alexa 488", "af488", "green")
    red_terms = (
        "texas red",
        "texre",
        "tritc",
        "alexa 555",
        "alexa 568",
        "alexa 594",
        "red",
    )
    green += 5.0 * sum(term in text for term in green_terms)
    red += 5.0 * sum(term in text for term in red_terms)
    if excitation is not None:
        if 450.0 <= excitation <= 520.0:
            green += 2.0
        elif 530.0 <= excitation <= 650.0:
            red += 2.0
    if emission is not None:
        if 490.0 <= emission <= 550.0:
            green += 2.0
        elif 560.0 <= emission <= 700.0:
            red += 2.0
    if color:
        raw = color.lstrip("#")
        if len(raw) == 8:  # AARRGGBB
            raw = raw[2:]
        if len(raw) == 6:
            try:
                red_value = int(raw[0:2], 16)
                green_value = int(raw[2:4], 16)
            except ValueError:
                pass
            else:
                if green_value > red_value * 1.25:
                    green += 2.0
                elif red_value > green_value * 1.25:
                    red += 2.0
    return green, red


def infer_czi_channel_indices(
    metadata: ElementTree.Element | str | bytes,
    channel_count: int,
) -> tuple[int, int]:
    """Return (green_index, red_index), with legacy C1/C0 as fallback."""
    if isinstance(metadata, ElementTree.Element):
        root = metadata
    else:
        root = ElementTree.fromstring(metadata)
    count = int(channel_count)
    if count < 2:
        raise ValueError("At least two CZI channels are required.")

    candidates: list[tuple[float, int, int]] = []
    for parent in root.iter():
        if _local_name(parent.tag).lower() != "channels":
            continue
        channels = [
            child
            for child in list(parent)
            if _local_name(child.tag).lower() == "channel"
        ]
        if len(channels) < count:
            continue
        scores = [_channel_role_scores(channel) for channel in channels[:count]]
        for green_index, (green_score, _) in enumerate(scores):
            for red_index, (_, red_score) in enumerate(scores):
                if green_index != red_index:
                    candidates.append(
                        (float(green_score + red_score), green_index, red_index)
                    )

    if candidates:
        confidence, green_index, red_index = max(candidates, key=lambda item: item[0])
        if confidence >= 4.0:
            return int(green_index), int(red_index)
    logger.warning(
        "Could not identify green/red CZI channels from metadata; using legacy C1/C0 order."
    )
    return 1, 0


def _open_czi(path: Path):
    try:
        from aicspylibczi import CziFile
    except ImportError as exc:  # pragma: no cover - exercised only in incomplete installs
        raise RuntimeError(
            "CZI support requires aicspylibczi. Reinstall NVAP with its declared dependencies."
        ) from exc
    return CziFile(path)


def read_czi_spacing(path: str | Path) -> VoxelSpacing:
    czi = _open_czi(Path(path).resolve())
    return parse_czi_spacing(czi.meta)


def read_czi_shape(path: str | Path) -> tuple[int, int, int, int]:
    """Return the first CZI scene shape as (channels, z, y, x) without reading pixels."""
    czi = _open_czi(Path(path).resolve())
    shapes = czi.get_dims_shape()
    if not shapes:
        raise ValueError(f"CZI file has no image dimensions: {path}")
    first = shapes[0]

    def size(axis: str, default: int = 1) -> int:
        bounds = first.get(axis)
        if bounds is None:
            return default
        return int(bounds[1]) - int(bounds[0])

    channels, depth, height, width = size("C"), size("Z"), size("Y", 0), size("X", 0)
    if height <= 0 or width <= 0:
        raise ValueError(f"CZI file has invalid Y/X dimensions: {first}")
    return channels, depth, height, width


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume)
    if np.issubdtype(arr.dtype, np.integer):
        denom = float(np.iinfo(arr.dtype).max)
    else:
        finite_max = float(np.nanmax(arr)) if arr.size else 0.0
        denom = max(finite_max, 1.0)
    return np.clip(arr.astype(np.float32) / denom, 0.0, 1.0)


def czi_array_to_czyx(image: np.ndarray, dimensions: list[tuple[str, int]]) -> np.ndarray:
    """Reduce one CZI scene/time point and return channel-first CZYX data."""
    arr = np.asarray(image)
    axes = [str(axis).upper() for axis, _ in dimensions]
    if arr.ndim != len(axes):
        raise ValueError(f"CZI dimension metadata {axes} does not match array shape {arr.shape}.")

    # NVAP analyzes one volume. Select the first entry for scene, time, mosaic,
    # illumination, and other acquisition dimensions.
    for index in range(len(axes) - 1, -1, -1):
        if axes[index] not in {"C", "Z", "Y", "X"}:
            arr = np.take(arr, 0, axis=index)
            axes.pop(index)

    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"CZI data has no Y/X image plane (axes={axes}).")
    if "C" not in axes:
        arr = np.expand_dims(arr, axis=0)
        axes.insert(0, "C")
    if "Z" not in axes:
        insert_at = axes.index("C") + 1
        arr = np.expand_dims(arr, axis=insert_at)
        axes.insert(insert_at, "Z")
    arr = np.transpose(arr, [axes.index(axis) for axis in ("C", "Z", "Y", "X")])
    return np.asarray(arr)


def load_czi_channels(
    path: str | Path,
    spacing: VoxelSpacing | None = None,
) -> tuple[ChannelVolume, ChannelVolume, VoxelSpacing]:
    """Load the first CZI acquisition and map green/red channels from metadata."""
    source = Path(path).resolve()
    czi = _open_czi(source)
    effective_spacing = spacing or parse_czi_spacing(czi.meta)
    image, dimensions = czi.read_image()
    czyx = czi_array_to_czyx(image, list(dimensions))
    if czyx.shape[0] < 2:
        raise ValueError(
            f"NVAP requires at least two CZI channels; {source.name} contains {czyx.shape[0]}."
        )
    if czyx.shape[0] > 2:
        logger.warning(
            "CZI file %s contains %d channels; NVAP is using C0 as red and C1 as green.",
            source,
            czyx.shape[0],
        )
    green_index, red_index = infer_czi_channel_indices(czi.meta, int(czyx.shape[0]))
    logger.info(
        "CZI channel mapping for %s: green=C%d red=C%d",
        source.name,
        green_index,
        red_index,
    )
    red_data = _normalize_volume(czyx[red_index])
    green_data = _normalize_volume(czyx[green_index])
    depth = int(red_data.shape[0])
    z_indices = list(range(1, depth + 1))
    return (
        ChannelVolume("green", green_data, list(z_indices), effective_spacing),
        ChannelVolume("red", red_data, list(z_indices), effective_spacing),
        effective_spacing,
    )
