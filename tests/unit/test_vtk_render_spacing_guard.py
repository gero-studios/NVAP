from __future__ import annotations

import numpy as np

from nvap.analysis.microglia_analysis import _offset_shift_zyx
from nvap.config.types import RenderConfig, VoxelSpacing
from nvap.render.vtk_scene import (
    _cubic_render_spacing,
    _downsample_volume_for_render,
    _recommended_sample_distance,
    _render_downsample_factors,
    _snap_offset_to_voxel,
)


def test_cubic_render_spacing_skips_xy_upsample_when_z_is_finer() -> None:
    spacing = VoxelSpacing(x_um=0.331, y_um=0.331, z_um=0.0993)
    out = _cubic_render_spacing((168, 1024, 1024), spacing)
    assert np.isclose(out.x_um, spacing.x_um)
    assert np.isclose(out.y_um, spacing.y_um)
    assert np.isclose(out.z_um, spacing.z_um)


def test_cubic_render_spacing_applies_budget_guard_on_large_volumes() -> None:
    spacing = VoxelSpacing(x_um=0.331, y_um=0.331, z_um=0.8)
    out = _cubic_render_spacing((278, 1024, 1024), spacing)
    assert np.isclose(out.x_um, spacing.x_um)
    assert np.isclose(out.y_um, spacing.y_um)
    assert np.isclose(out.z_um, spacing.z_um)


def test_cubic_render_spacing_still_upsamples_z_when_safe() -> None:
    spacing = VoxelSpacing(x_um=0.331, y_um=0.331, z_um=0.55)
    out = _cubic_render_spacing((64, 256, 256), spacing)
    assert np.isclose(out.x_um, spacing.x_um)
    assert np.isclose(out.y_um, spacing.y_um)
    assert out.z_um < spacing.z_um


def test_recommended_sample_distance_tightens_for_anisotropic_spacing() -> None:
    iso = VoxelSpacing(x_um=0.25, y_um=0.25, z_um=0.25)
    aniso = VoxelSpacing(x_um=0.25, y_um=0.25, z_um=0.80)

    iso_step = _recommended_sample_distance(iso)
    aniso_step = _recommended_sample_distance(aniso)

    assert aniso_step < iso_step
    assert aniso_step > 0.0


def test_recommended_sample_distance_label_mode_is_tighter() -> None:
    spacing = VoxelSpacing(x_um=0.331, y_um=0.331, z_um=0.55)
    default_step = _recommended_sample_distance(spacing, label_mode=False)
    label_step = _recommended_sample_distance(spacing, label_mode=True)

    assert label_step < default_step
    assert label_step > 0.0


def test_render_downsample_factors_reduce_large_xy_volume() -> None:
    factors = _render_downsample_factors((168, 1024, 1024), max_voxels=72_000_000)

    assert factors == (1, 2, 2)


def test_downsample_volume_for_render_uses_max_pool_and_adjusts_spacing() -> None:
    volume = np.zeros((2, 4, 4), dtype=np.float32)
    volume[0, 1, 1] = 2.5
    volume[1, 3, 3] = 4.0
    spacing = VoxelSpacing(x_um=0.5, y_um=0.25, z_um=1.0)

    reduced, reduced_spacing, factors = _downsample_volume_for_render(
        volume,
        spacing,
        max_voxels=8,
    )

    assert factors == (1, 2, 2)
    assert reduced.shape == (2, 2, 2)
    assert np.isclose(reduced[0, 0, 0], 2.5)
    assert np.isclose(reduced[1, 1, 1], 4.0)
    assert np.isclose(reduced_spacing.x_um, 1.0)
    assert np.isclose(reduced_spacing.y_um, 0.5)
    assert np.isclose(reduced_spacing.z_um, 1.0)


def test_downsample_volume_for_render_label_mode_uses_nearest_stride() -> None:
    volume = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)
    spacing = VoxelSpacing(x_um=0.5, y_um=0.25, z_um=1.0)

    reduced, _spacing, factors = _downsample_volume_for_render(
        volume,
        spacing,
        label_mode=True,
        max_voxels=8,
    )

    assert factors == (1, 2, 2)
    np.testing.assert_array_equal(reduced, volume[:, ::2, ::2])


def test_snap_offset_to_voxel_rounds_to_nearest_voxel_multiple() -> None:
    # A sub-voxel offset must collapse onto the nearest whole-voxel position,
    # not render as a smooth continuous translation.
    assert np.isclose(_snap_offset_to_voxel(0.6, 0.5), 0.5)
    assert np.isclose(_snap_offset_to_voxel(0.8, 0.5), 1.0)
    assert np.isclose(_snap_offset_to_voxel(-0.6, 0.5), -0.5)
    # Exact multiples are unaffected.
    assert np.isclose(_snap_offset_to_voxel(1.5, 0.5), 1.5)
    # Zero spacing must not divide by zero.
    assert np.isfinite(_snap_offset_to_voxel(1.0, 0.0))


def test_render_offset_snapping_matches_analysis_voxel_shift() -> None:
    # The rendered green-actor translation must land on exactly the same
    # physical position as the whole-voxel shift the analysis pipeline applies
    # (microglia_analysis._offset_shift_zyx), so a sub-voxel offset dialed in on
    # the UI cannot make the visual overlap disagree with the reported metric.
    spacing = VoxelSpacing(x_um=0.33, y_um=0.33, z_um=0.9)
    render = RenderConfig(offset_x_um=0.5, offset_y_um=-0.2, offset_z_um=1.1)

    dz, dy, dx = _offset_shift_zyx(render, np.asarray([spacing.z_um, spacing.y_um, spacing.x_um]))
    analysis_shift_um = (dx * spacing.x_um, dy * spacing.y_um, dz * spacing.z_um)

    render_shift_um = (
        _snap_offset_to_voxel(render.offset_x_um, spacing.x_um),
        _snap_offset_to_voxel(render.offset_y_um, spacing.y_um),
        _snap_offset_to_voxel(render.offset_z_um, spacing.z_um),
    )

    assert np.allclose(render_shift_um, analysis_shift_um)
