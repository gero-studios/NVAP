"""Validate vascular morphometry against synthetic phantoms of known geometry."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage as ndi

from nvap.analysis.vascular_analysis import (
    _anatomical_radius_is_reliable,
    _prune_terminal_spurs,
    _skeleton_topology,
    analyze_vasculature,
    build_vascular_masks,
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


def _make_hollow_cylinder(
    *,
    shape: tuple[int, int, int],
    outer_radius_vox: float,
    inner_radius_vox: float,
    axis: int = 2,
) -> np.ndarray:
    zz, yy, xx = np.indices(shape)
    coords = {0: (yy, xx), 1: (zz, xx), 2: (zz, yy)}[axis]
    other_axes = [ax for ax in (0, 1, 2) if ax != axis]
    center0 = shape[other_axes[0]] / 2.0
    center1 = shape[other_axes[1]] / 2.0
    c0, c1 = coords
    rr = np.sqrt((c0 - center0) ** 2 + (c1 - center1) ** 2)
    wall = (rr <= outer_radius_vox) & (rr >= inner_radius_vox)
    return wall.astype(np.float32)


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
    assert result.radius_p10_um <= result.mean_radius_um + 1.0e-12
    assert result.mean_radius_um <= result.radius_p90_um + 1.0e-12
    assert result.volume_length_equivalent_radius_um > 0.0
    assert result.volume_length_equivalent_diameter_um == pytest.approx(
        2.0 * result.volume_length_equivalent_radius_um
    )
    assert result.radius_ridge_search_um > 0.0

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


def test_hollow_cylinder_uses_reconstructed_solid_mask_for_anatomical_metrics():
    shape = (32, 32, 60)
    vol = _make_hollow_cylinder(
        shape=shape,
        outer_radius_vox=6.0,
        inner_radius_vox=3.0,
        axis=2,
    )
    spacing = VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0)

    masks = build_vascular_masks(vol, threshold=0.5, spacing=spacing)
    result = analyze_vasculature(vol, threshold=0.5, spacing=spacing)

    assert masks.wall_mask[shape[0] // 2, shape[1] // 2, shape[2] // 2] == 0
    assert masks.solid_mask[shape[0] // 2, shape[1] // 2, shape[2] // 2] == 1
    assert result.wall_voxel_count < result.solid_vessel_voxel_count
    assert result.mean_radius_um == pytest.approx(6.0, abs=1.6)
    assert 40.0 <= result.total_length_um <= 65.0
    assert result.segment_count <= 3


def test_small_hollow_wall_reports_radius_after_default_tube_fill():
    shape = (32, 32, 60)
    vol = _make_hollow_cylinder(
        shape=shape,
        outer_radius_vox=6.0,
        inner_radius_vox=3.0,
        axis=2,
    )
    spacing = VoxelSpacing(x_um=0.2, y_um=0.2, z_um=0.2)

    result = analyze_vasculature(vol, threshold=0.5, spacing=spacing)
    rows = vascular_analysis_to_csv_rows(result)

    assert result.anatomical_radius_reliable is True
    assert result.reconstructed_lumen_fill_fraction >= 0.20
    assert result.mean_radius_um > 0.0
    assert result.mean_diameter_um == pytest.approx(2.0 * result.mean_radius_um)
    assert result.mean_reconstructed_mask_radius_um > 0.0
    assert not any(row["metric"] == "vascular_radius_interpretation" for row in rows)


def test_underfilled_hollow_wall_withholds_anatomical_radius():
    shape = (32, 32, 60)
    vol = _make_hollow_cylinder(
        shape=shape,
        outer_radius_vox=6.0,
        inner_radius_vox=3.0,
        axis=2,
    )
    spacing = VoxelSpacing(x_um=0.2, y_um=0.2, z_um=0.2)

    result = analyze_vasculature(
        vol,
        threshold=0.5,
        spacing=spacing,
        fill_cavities=False,
    )
    rows = vascular_analysis_to_csv_rows(result)

    assert result.anatomical_radius_reliable is False
    assert np.isnan(result.mean_radius_um)
    assert np.isnan(result.mean_diameter_um)
    assert any(row["metric"] == "vascular_radius_interpretation" for row in rows)


def test_low_fill_radius_passes_when_independent_estimators_agree():
    solid = np.zeros((24, 24, 40), dtype=bool)
    solid[9:15, 9:15, :] = True

    assert _anatomical_radius_is_reliable(
        np.asarray([1.45, 1.50, 1.55]),
        lumen_fill_fraction=0.188,
        reconstructed_solid=True,
        equivalent_radius_um=1.59,
        solid_mask=solid,
    )


def test_sheet_like_reconstruction_withholds_radius_even_with_high_fill():
    sheet = np.zeros((20, 32, 32), dtype=bool)
    sheet[:8] = True

    assert not _anatomical_radius_is_reliable(
        np.asarray([1.5, 1.8, 2.0]),
        lumen_fill_fraction=0.80,
        reconstructed_solid=True,
        equivalent_radius_um=1.6,
        solid_mask=sheet,
    )


def test_open_ended_hollow_tube_touching_z_boundary_is_filled_slicewise():
    shape = (24, 32, 32)
    vol = _make_hollow_cylinder(
        shape=shape,
        outer_radius_vox=6.0,
        inner_radius_vox=3.0,
        axis=0,
    )

    masks = build_vascular_masks(
        vol,
        threshold=0.5,
        spacing=VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0),
    )

    assert np.all(masks.wall_mask[:, shape[1] // 2, shape[2] // 2] == 0)
    assert np.all(masks.solid_mask[:, shape[1] // 2, shape[2] // 2] == 1)


def test_gapped_hollow_tube_wall_is_sealed_and_filled():
    shape = (32, 32, 60)
    vol = _make_hollow_cylinder(
        shape=shape,
        outer_radius_vox=7.0,
        inner_radius_vox=4.0,
        axis=2,
    )
    # Cut a consistent radial slit through the wall. A pure fill-holes pass will
    # not fill this because the lumen is connected to exterior background.
    vol[shape[0] // 2, shape[1] // 2 + 4 :, :] = 0.0

    masks = build_vascular_masks(
        vol,
        threshold=0.5,
        spacing=VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0),
    )
    result = analyze_vasculature(
        vol,
        threshold=0.5,
        spacing=VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0),
    )

    assert masks.wall_mask[shape[0] // 2, shape[1] // 2, shape[2] // 2] == 0
    assert masks.solid_mask[shape[0] // 2, shape[1] // 2, shape[2] // 2] == 1
    assert result.anatomical_radius_reliable is True
    assert result.mean_radius_um > 0.0


def test_default_estimated_fill_recovers_more_of_a_broad_wall_gap():
    shape = (60, 32, 32)
    vol = _make_hollow_cylinder(
        shape=shape,
        outer_radius_vox=7.0,
        inner_radius_vox=4.0,
        axis=0,
    )
    # Three-voxel acquisition dropout through one side of the fluorescent wall.
    # The 2.0 um default should reconstruct more of the known tube interior than
    # the former 1.5 um tolerance without changing the tube's component count.
    vol[:, 15:18, 21:] = 0.0
    spacing = VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0)

    former = build_vascular_masks(
        vol,
        threshold=0.5,
        spacing=spacing,
        closing_radius_um=1.5,
    )
    improved = build_vascular_masks(vol, threshold=0.5, spacing=spacing)

    expected_solid = _make_cylinder(
        shape=shape,
        radius_vox=7.0,
        axis=0,
    ).astype(bool)
    former_missing = int(np.count_nonzero(expected_solid & ~former.solid_mask))
    improved_missing = int(np.count_nonzero(expected_solid & ~improved.solid_mask))
    _, former_components = ndi.label(former.solid_mask)
    _, improved_components = ndi.label(improved.solid_mask)

    assert improved_missing < former_missing
    assert improved.solid_mask[shape[0] // 2, shape[1] // 2, shape[2] // 2]
    assert improved_components == former_components == 1


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


def test_short_terminal_spur_is_pruned_from_topology():
    skeleton = np.zeros((1, 20, 30), dtype=bool)
    skeleton[0, 10, 3:26] = True
    skeleton[0, 9, 14] = True
    spacing = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)

    unpruned = _skeleton_topology(skeleton, spacing)
    pruned_skeleton = _prune_terminal_spurs(skeleton, spacing, min_length_um=2.0)
    pruned = _skeleton_topology(pruned_skeleton, spacing)

    assert pruned.segment_count < unpruned.segment_count
    assert pruned.total_length_um < unpruned.total_length_um


def test_stacked_centerline_columns_report_decussation_candidates():
    vol = np.zeros((7, 15, 15), dtype=np.float32)
    vol[1, 7, 3:12] = 1.0
    vol[5, 3:12, 7] = 1.0

    result = analyze_vasculature(vol, threshold=0.5, spacing=VoxelSpacing(1.0, 1.0, 1.0))

    assert result.decussation_candidate_count >= 1
    assert result.mean_decussation_z_separation_um >= 4.0
    rows = vascular_analysis_to_csv_rows(result)
    assert any(row["metric"] == "decussation_candidate_count" for row in rows)


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


def test_trim_excludes_trimmed_slices_from_tissue_volume():
    # Cylinder along z is uniform in z, so trimming removes vessel *and* tissue in
    # equal proportion. Tissue volume must count only the retained slices; if the
    # trimmed slices leaked into the denominator the fraction would drop.
    shape = (60, 24, 24)
    voxel_volume = 1.0
    vol = _make_cylinder(shape=shape, radius_vox=4.0, axis=0)
    render = RenderConfig(trim_first_slices=5, trim_last_slices=5)
    spacing = VoxelSpacing(1.0, 1.0, 1.0)

    full = analyze_vasculature(vol, threshold=0.5, spacing=spacing)
    trimmed = analyze_vasculature(vol, threshold=0.5, spacing=spacing, render=render)

    retained_slices = shape[0] - 10
    expected_tissue = retained_slices * shape[1] * shape[2] * voxel_volume
    assert trimmed.tissue_volume_um3 == pytest.approx(expected_tissue)
    # A z-uniform vessel keeps the same volume fraction after trimming z-ends;
    # this only holds when the trimmed slices leave the denominator too.
    assert trimmed.vessel_volume_fraction == pytest.approx(
        full.vessel_volume_fraction, rel=1e-6
    )
