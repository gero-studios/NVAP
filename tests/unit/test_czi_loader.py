from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nvap.config.types import ChannelVolume, VoxelSpacing
from nvap.io import stack_loader
from nvap.io.czi_loader import czi_array_to_czyx, parse_czi_spacing


def test_parse_czi_spacing_converts_zeiss_meters_to_micrometers() -> None:
    metadata = """
    <ImageDocument><Metadata><Scaling><Items>
      <Distance Id="X"><Value>3.31e-7</Value><DefaultUnitFormat>m</DefaultUnitFormat></Distance>
      <Distance Id="Y"><Value>3.32e-7</Value></Distance>
      <Distance Id="Z"><Value>4.0e-7</Value><DefaultUnitFormat>m</DefaultUnitFormat></Distance>
    </Items></Scaling></Metadata></ImageDocument>
    """

    spacing = parse_czi_spacing(metadata)

    assert spacing.x_um == pytest.approx(0.331)
    assert spacing.y_um == pytest.approx(0.332)
    assert spacing.z_um == pytest.approx(0.4)


def test_parse_czi_spacing_accepts_explicit_micrometer_and_nanometer_units() -> None:
    metadata = """
    <Scaling><Items>
      <Distance Id="X"><Value>0.25</Value><Unit>um</Unit></Distance>
      <Distance Id="Y"><Value>250</Value><Unit>nm</Unit></Distance>
      <Distance Id="Z"><Value>1.5</Value><Unit>µm</Unit></Distance>
    </Items></Scaling>
    """

    assert parse_czi_spacing(metadata) == VoxelSpacing(0.25, 0.25, 1.5)


def test_parse_czi_spacing_rejects_incomplete_metadata() -> None:
    metadata = """
    <Scaling><Items>
      <Distance Id="X"><Value>2.5e-7</Value></Distance>
      <Distance Id="Y"><Value>2.5e-7</Value></Distance>
    </Items></Scaling>
    """

    with pytest.raises(ValueError, match="Z"):
        parse_czi_spacing(metadata)


def test_czi_array_to_czyx_selects_first_time_and_reorders_axes() -> None:
    image = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.uint16).reshape(2, 3, 4, 5, 6)

    result = czi_array_to_czyx(
        image,
        [("T", 2), ("Z", 3), ("C", 4), ("Y", 5), ("X", 6)],
    )

    assert result.shape == (4, 3, 5, 6)
    np.testing.assert_array_equal(result[2, 1], image[0, 1, 2])


def test_load_dataset_dispatches_single_czi_and_uses_metadata_spacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    czi_path = tmp_path / "two_channel.czi"
    czi_path.write_bytes(b"test placeholder")
    metadata_spacing = VoxelSpacing(0.2, 0.21, 0.8)

    def fake_load(path: str | Path, spacing: VoxelSpacing | None = None):
        effective = spacing or metadata_spacing
        data = np.ones((3, 5, 6), dtype=np.float32)
        return (
            ChannelVolume("green", data, [1, 2, 3], effective),
            ChannelVolume("red", data, [1, 2, 3], effective),
            effective,
        )

    monkeypatch.setattr(stack_loader, "load_czi_channels", fake_load)

    dataset = stack_loader.load_dataset(czi_path)

    assert dataset.green.spacing == metadata_spacing
    assert dataset.red.spacing == metadata_spacing
    assert dataset.shared_z_range == (1, 3)


def test_load_dataset_allows_manual_czi_spacing_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    czi_path = tmp_path / "two_channel.czi"
    czi_path.write_bytes(b"test placeholder")
    manual = VoxelSpacing(1.0, 1.1, 2.0)

    def fake_load(path: str | Path, spacing: VoxelSpacing | None = None):
        assert spacing == manual
        data = np.ones((2, 4, 4), dtype=np.float32)
        return (
            ChannelVolume("green", data, [1, 2], spacing),
            ChannelVolume("red", data, [1, 2], spacing),
            spacing,
        )

    monkeypatch.setattr(stack_loader, "load_czi_channels", fake_load)

    dataset = stack_loader.load_dataset(czi_path, spacing=manual)

    assert dataset.green.spacing == manual

