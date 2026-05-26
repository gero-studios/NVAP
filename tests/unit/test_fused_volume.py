from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

from nvap.config.types import ChannelVolume, DatasetVolume, VoxelSpacing
from nvap.preprocess.fused_volume import (
    DEFAULT_FUSED_PARAMS,
    FusedVolumeParams,
    bridge_component_gaps,
    build_fused_dataset,
    resample_channel_isotropic,
    resolve_isotropic_target_um,
    smooth_shape_preserving_thin,
)

_FULL = ndi.generate_binary_structure(3, 3)


def _component_count(mask: np.ndarray) -> int:
    _, count = ndi.label(np.asarray(mask, dtype=bool), structure=_FULL)
    return int(count)


# ---------------------------------------------------------------------------
# Isotropic resampling
# ---------------------------------------------------------------------------

def test_resolve_target_auto_anchors_on_finest_spacing() -> None:
    spacing = VoxelSpacing(x_um=0.331, y_um=0.331, z_um=0.4)
    target = resolve_isotropic_target_um(spacing, (40, 64, 64), DEFAULT_FUSED_PARAMS)
    assert abs(target - 0.331) < 1e-6


def test_resolve_target_steps_down_for_huge_stacks() -> None:
    spacing = VoxelSpacing(x_um=0.331, y_um=0.331, z_um=0.4)
    target = resolve_isotropic_target_um(spacing, (500, 2000, 2000), DEFAULT_FUSED_PARAMS)
    assert target > 0.331  # stepped coarser to respect the voxel cap


def test_resolve_target_honours_explicit_override() -> None:
    spacing = VoxelSpacing(x_um=0.331, y_um=0.331, z_um=0.4)
    params = FusedVolumeParams(isotropic_target_um=0.5)
    assert resolve_isotropic_target_um(spacing, (40, 64, 64), params) == 0.5


def test_resample_channel_isotropic_makes_voxels_cubic() -> None:
    channel = ChannelVolume(
        name="green",
        data=np.zeros((10, 16, 16), dtype=np.float32),
        z_indices=list(range(1, 11)),
        spacing=VoxelSpacing(x_um=0.331, y_um=0.331, z_um=0.4),
    )
    out = resample_channel_isotropic(channel, 0.331, order=3)
    assert out.spacing.x_um == out.spacing.y_um == out.spacing.z_um == 0.331
    # z is upsampled (0.4 -> 0.331), xy unchanged.
    assert out.data.shape[0] > 10
    assert out.data.shape[1:] == (16, 16)
    assert len(out.z_indices) == out.data.shape[0]


def test_resample_clamps_cubic_spline_overshoot() -> None:
    rng = np.random.default_rng(0)
    data = rng.random((8, 24, 24), dtype=np.float32)
    channel = ChannelVolume(
        name="green",
        data=data,
        z_indices=list(range(1, 9)),
        spacing=VoxelSpacing(x_um=0.331, y_um=0.331, z_um=0.4),
    )
    out = resample_channel_isotropic(channel, 0.331, order=3)
    assert float(out.data.min()) >= 0.0
    assert float(out.data.max()) <= float(data.max()) + 1e-5


# ---------------------------------------------------------------------------
# Gap bridging
# ---------------------------------------------------------------------------

def _two_cubes(gap_voxels: int, size: int = 5) -> np.ndarray:
    vol = np.zeros((size + 4, size + 4, 60), dtype=bool)
    y0 = 2
    vol[2:2 + size, y0:y0 + size, 5:5 + size] = True
    x1 = 5 + size + gap_voxels
    vol[2:2 + size, y0:y0 + size, x1:x1 + size] = True
    return vol


def test_bridge_connects_close_fragments() -> None:
    mask = _two_cubes(gap_voxels=3)
    assert _component_count(mask) == 2
    bridged, added = bridge_component_gaps(
        mask, (0.331, 0.331, 0.331), max_gap_um=4.5, tube_radius_um=0.35
    )
    assert added == 1
    assert _component_count(bridged) == 1


def test_bridge_leaves_distant_fragments_apart() -> None:
    mask = _two_cubes(gap_voxels=40)
    bridged, added = bridge_component_gaps(
        mask, (0.331, 0.331, 0.331), max_gap_um=4.5, tube_radius_um=0.35
    )
    assert added == 0
    assert _component_count(bridged) == 2


def test_bridge_noop_on_single_component() -> None:
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    bridged, added = bridge_component_gaps(
        mask, (0.331, 0.331, 0.331), max_gap_um=4.5, tube_radius_um=0.35
    )
    assert added == 0
    assert np.array_equal(bridged, mask)


# ---------------------------------------------------------------------------
# Thin-aware smoothing
# ---------------------------------------------------------------------------

def test_smoothing_preserves_thin_process_connectivity() -> None:
    mask = np.zeros((24, 24, 24), dtype=bool)
    # Thick blob.
    mask[8:16, 8:16, 4:10] = True
    # One-voxel-thick process trailing off the blob.
    mask[11, 11, 10:21] = True
    assert _component_count(mask) == 1

    result, _occupancy = smooth_shape_preserving_thin(
        mask, (0.331, 0.331, 0.331), sigma_um=0.33, thin_thickness_um=0.7
    )
    assert _component_count(result) == 1, "thin process must not be smoothed off"
    # The far tip of the thin process survives.
    assert bool(result[11, 11, 20])


def test_smoothing_occupancy_recovers_mask_at_half() -> None:
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[5:15, 5:15, 5:15] = True
    result, occupancy = smooth_shape_preserving_thin(
        mask, (0.331, 0.331, 0.331), sigma_um=0.33, thin_thickness_um=0.7
    )
    assert occupancy.min() >= 0.0 and occupancy.max() <= 1.0
    assert np.array_equal(occupancy >= 0.5, result)


def test_smoothing_removes_staircase_nub_on_thick_block() -> None:
    mask = np.zeros((24, 24, 24), dtype=bool)
    mask[6:18, 6:18, 6:18] = True
    # A lone protruding voxel (staircase artifact) on a thick face.
    mask[12, 12, 18] = True
    result, _ = smooth_shape_preserving_thin(
        mask, (0.331, 0.331, 0.331), sigma_um=0.33, thin_thickness_um=0.7
    )
    assert not bool(result[12, 12, 18]), "isolated nub on a thick block should smooth away"
    assert _component_count(result) == 1


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def _dataset_with_blob() -> DatasetVolume:
    green = np.zeros((10, 32, 32), dtype=np.float32)
    green[3:7, 10:22, 10:22] = 1.0
    red = np.zeros((10, 32, 32), dtype=np.float32)
    red[3:7, 4:8, 4:28] = 1.0
    spacing = VoxelSpacing(x_um=0.331, y_um=0.331, z_um=0.4)
    return DatasetVolume(
        green=ChannelVolume("green", green, list(range(1, 11)), spacing),
        red=ChannelVolume("red", red, list(range(1, 11)), spacing),
        shared_z_range=(1, 10),
    )


def test_build_fused_dataset_coregisters_green_and_red() -> None:
    result = build_fused_dataset(_dataset_with_blob())
    fused = result.dataset
    # Both channels land on the same isotropic grid.
    assert fused.green.data.shape == fused.red.data.shape
    assert fused.green.spacing.x_um == fused.green.spacing.z_um
    assert fused.red.spacing.z_um == fused.green.spacing.z_um
    # Green is an occupancy field in [0, 1] with a recoverable shape.
    assert fused.green.data.min() >= 0.0 and fused.green.data.max() <= 1.0
    assert result.green_mask.shape == fused.green.data.shape
    assert np.array_equal(fused.green.data >= 0.5, result.green_mask)
    assert bool(result.green_mask.any())


def test_build_fused_dataset_disabled_passthrough() -> None:
    dataset = _dataset_with_blob()
    result = build_fused_dataset(dataset, FusedVolumeParams(enabled=False))
    assert result.dataset is dataset
    assert result.bridges_added == 0
