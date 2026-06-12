"""Validate vascular morphometry against synthetic phantoms of known geometry."""

from __future__ import annotations

import numpy as np
import pytest

from nvap.analysis.vascular_analysis import (
    analyze_vasculature,
    vascular_analysis_to_csv_rows,
)
from nvap.config.types import RenderConfig, VoxelSpacing


def _make_cylinder(
    *,
    shape: tuple[int, int, int],
    radius_vox: float,
    axis: int = 2,
) -> np.ndarray:
    """Solid cylinder of the given voxel radius running along ``axis``."""
    zz, yy, xx = np.indices(shape)
    coords = {0: (yy, xx), 1: (zz, xx), 2: (zz, yy)}[axis]
    a0 = shape[[0, 1, 2][[ax for ax in (0, 1, 2) if ax != axis][0]]] / 2.0
    a1 = shape[[ax for ax in (0, 1, 2) if ax != axis][1]] / 2.0
    c0, c1 = coords
    rr = np.sqrt((c0 - a0) ** 2 + (c1 - a1) ** 2)
    return (rr <= radius_vox).astype(np.float32)


def test_cylinder_radius_and_length_recovered():
    # Isotropic 1 um spacing, radius 5 vox along x over a 60-voxel run.
    shape = (24, 24, 60)
    radius_vox = 5.0
    vol = _make_cylinder(shape=shape, radius_vox=radius_vox, axis=2)
    spacing = VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0)

    result = analyze_vasculature(vol, threshold=0.5, spacing=spacing)

    # Mean medial-axis radius should be within ~1 voxel of the true radius.
    assert abs(result.mean_radius_um - radius_vox) <= 1.5
    assert result.mean_diameter_um == pytest.approx(2.0 * result.mean_radius_um)

    # Centreline length ~ the cylinder run length (skeleton may stop a little
    # short of the flat end caps); allow generous tolerance.
    assert 40.0 <= result.total_length_um <= 65.0

    # A single straight tube: one segment, no junctions, ~straight tortuosity.
    assert result.junction_count == 0
    assert result.component_count == 1
    assert result.mean_tortuosity == pytest.approx(1.0, abs=0.15)

    # Volume fraction matches the analytic cylinder volume fraction.
    expected_frac = float(vol.sum()) / float(vol.size)
    assert result.vessel_volume_fraction == pytest.approx(expected_frac, rel=1e-6)


def test_anisotropic_spacing_scales_radius():
    shape = (24, 24, 60)
    vol = _make_cylinder(shape=shape, radius_vox=5.0, axis=2)
    iso = analyze_vasculature(vol, threshold=0.5, spacing=VoxelSpacing(1.0, 1.0, 1.0))
    fine = analyze_vasculature(vol, threshold=0.5, spacing=VoxelSpacing(0.5, 0.5, 0.5))
    # Halving in-plane spacing halves the physical radius.
    assert fine.mean_radius_um == pytest.approx(iso.mean_radius_um * 0.5, rel=0.1)


def test_branching_phantom_reports_junction():
    # A Y-shaped vessel: trunk + two diverging arms -> exactly one junction.
    vol = np.zeros((1, 40, 40), dtype=np.float32)
    vol[0, 20, 5:20] = 1.0  # trunk along x
    for i in range(15):
        vol[0, 20 - i, 20 + i] = 1.0  # upper arm
        vol[0, 20 + i, 20 + i] = 1.0  # lower arm
    # Thicken so skeletonisation is stable.
    from scipy.ndimage import binary_dilation

    vol[0] = binary_dilation(vol[0] > 0, iterations=1).astype(np.float32)

    result = analyze_vasculature(vol, threshold=0.5, spacing=VoxelSpacing(1.0, 1.0, 1.0))
    assert result.junction_count >= 1
    assert result.segment_count >= 3
    assert result.endpoint_count >= 3


def test_empty_volume_is_safe():
    vol = np.zeros((10, 10, 10), dtype=np.float32)
    result = analyze_vasculature(vol, threshold=0.5, spacing=VoxelSpacing(1.0, 1.0, 1.0))
    assert result.vessel_voxel_count == 0
    assert result.total_length_um == 0.0
    assert result.tissue_volume_um3 > 0.0
    rows = vascular_analysis_to_csv_rows(result)
    assert any(r["metric"] == "vessel_volume_fraction" for r in rows)


def test_render_trim_is_honoured():
    # Cylinder along z so trimming the z-ends actually removes vessel voxels.
    shape = (60, 24, 24)
    vol = _make_cylinder(shape=shape, radius_vox=4.0, axis=0)
    render = RenderConfig(trim_first_slices=5, trim_last_slices=5)
    full = analyze_vasculature(vol, threshold=0.5, spacing=VoxelSpacing(1.0, 1.0, 1.0))
    trimmed = analyze_vasculature(
        vol, threshold=0.5, spacing=VoxelSpacing(1.0, 1.0, 1.0), render=render
    )
    # Trimming removes vessel voxels in the cut slices.
    assert trimmed.vessel_voxel_count < full.vessel_voxel_count
