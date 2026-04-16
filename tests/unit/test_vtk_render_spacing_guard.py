from __future__ import annotations

import numpy as np

from nvap.config.types import VoxelSpacing
from nvap.render.vtk_scene import _cubic_render_spacing, _recommended_sample_distance


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
