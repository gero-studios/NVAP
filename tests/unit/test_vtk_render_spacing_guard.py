from __future__ import annotations

import numpy as np
from dataclasses import replace
from vtkmodules.util.numpy_support import vtk_to_numpy

from nvap.analysis.microglia_analysis import _offset_shift_zyx
from nvap.config.types import RenderConfig, VoxelSpacing
from nvap.render.vtk_scene import (
    VTKScene,
    _cubic_render_spacing,
    _downsample_volume_for_render,
    _recommended_sample_distance,
    _render_downsample_factors,
    _snap_offset_to_voxel,
)


def _color_at(transfer_function, value):
    color = [0.0, 0.0, 0.0]
    transfer_function.GetColor(value, color)
    return color


def test_segmented_components_have_distinct_surface_and_volume_tints(qtbot, monkeypatch) -> None:
    scene = VTKScene()
    qtbot.addWidget(scene.widget())
    monkeypatch.setattr(scene, "render", lambda: None)
    labels = np.zeros((6, 8, 10), dtype=np.uint8)
    # Touching cells must retain a boundary and separate cell scalars.
    labels[1:5, 1:7, 1:5] = 1
    labels[1:5, 1:7, 5:9] = 2
    config = RenderConfig(show_iso_green=True, opacity_green=0.6)
    scene.set_channel_component_coloring("green", True, label_count=2)
    scene.set_channel_data("green", labels, VoxelSpacing(x_um=1, y_um=1, z_um=1))
    scene.apply_render_config(config)

    actor = scene._actors["green"]
    mapper = actor.iso_actor.GetMapper()
    mapper.Update()
    mesh = mapper.GetInput()
    assert actor.marching.IsA("vtkDiscreteMarchingCubes")
    assert mapper.GetScalarVisibility()
    cell_labels = vtk_to_numpy(mesh.GetCellData().GetScalars())
    np.testing.assert_array_equal(np.unique(cell_labels), [1, 2])
    # Each side of the shared boundary at x=4.5 belongs to a different label.
    for component_id in (1, 2):
        boundary_cells = [
            mesh.GetCell(int(i)) for i in np.flatnonzero(cell_labels == component_id)
            if np.allclose([mesh.GetCell(int(i)).GetPoints().GetPoint(j)[0] for j in range(3)], 4.5)
        ]
        assert boundary_cells
    surface_colors = mapper.GetLookupTable()
    colors = [_color_at(surface_colors, i) for i in (1, 2)]
    assert not np.allclose(*colors)
    for i, color in enumerate(colors, start=1):
        np.testing.assert_allclose(_color_at(actor.volume_property.GetRGBTransferFunction(), i), color)
    assert actor.iso_actor.GetVisibility()
    assert not actor.volume_actor.GetVisibility()
    assert np.isclose(actor.iso_actor.GetProperty().GetOpacity(), 0.85)

    scene.apply_render_config(replace(config, show_iso_green=False))
    assert actor.volume_actor.GetVisibility()
    assert not actor.iso_actor.GetVisibility()
    assert actor.volume_property.GetInterpolationTypeAsString() == "Nearest Neighbor"
    assert np.isclose(actor.volume_property.GetScalarOpacity().GetValue(1),
                      actor.volume_property.GetScalarOpacity().GetValue(2))
    # Reusing the same actor when selecting one cell preserves that cell's tint.
    scene.set_channel_data("green", np.where(labels == 2, labels, 0), VoxelSpacing(x_um=1, y_um=1, z_um=1))
    np.testing.assert_allclose(_color_at(actor.volume_property.GetRGBTransferFunction(), 2), colors[1])

    # Returning to intensities must rebuild the continuous surface pipeline.
    scene.set_channel_component_coloring("green", False)
    scene.set_channel_data("green", (labels > 0).astype(np.float32), VoxelSpacing(x_um=1, y_um=1, z_um=1))
    scene.apply_render_config(config)
    assert not actor.marching.IsA("vtkDiscreteMarchingCubes")
    assert not actor.iso_actor.GetMapper().GetScalarVisibility()


def test_red_surface_stays_on_native_grid_on_upload_and_fast_update(qtbot, monkeypatch) -> None:
    scene = VTKScene()
    qtbot.addWidget(scene.widget())
    monkeypatch.setattr(scene, "render", lambda: None)
    spacing = VoxelSpacing(x_um=0.131531, y_um=0.131531, z_um=0.4)
    volume = np.zeros((5, 16, 16), dtype=np.float32)
    volume[:, 5:10, 5:10] = 0.9

    for data in (volume, volume * 0.9):
        scene.set_channel_data("red", data, spacing)
        actor = scene._actors["red"]
        assert actor.resample is None
        assert actor.image.GetDimensions() == (16, 16, 5)
        assert scene._spacing["red"] == spacing
        assert scene._surface_input(actor) is actor.image


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


def test_component_render_preserves_exact_native_label_grid() -> None:
    volume = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)
    spacing = VoxelSpacing(x_um=0.5, y_um=0.25, z_um=1.0)

    reduced, _spacing, factors = _downsample_volume_for_render(
        volume,
        spacing,
        label_mode=True,
        max_voxels=8,
    )

    assert factors == (1, 1, 1)
    assert reduced is volume
    assert _spacing == spacing
    np.testing.assert_array_equal(reduced, volume)


def test_label_downsampling_keeps_thin_branches_between_sample_positions() -> None:
    labels = np.zeros((2, 4, 4), dtype=np.uint16)
    labels[:, 1, :] = 2
    labels[:, 3, :] = 256

    reduced, _spacing, _factors = _downsample_volume_for_render(
        labels, VoxelSpacing(), label_mode=True, max_voxels=8,
    )

    np.testing.assert_array_equal(reduced, labels)
    np.testing.assert_array_equal(np.unique(reduced), [0, 2, 256])


def test_unassigned_microglia_signal_stays_green_in_both_render_modes(qtbot, monkeypatch) -> None:
    scene = VTKScene()
    qtbot.addWidget(scene.widget())
    monkeypatch.setattr(scene, "render", lambda: None)
    labels = np.zeros((6, 8, 10), dtype=np.uint16)
    labels[1:5, 1:7, 1:4] = 1
    labels[1:5, 1:7, 6:9] = 2  # Display-only unassigned signal, not a detected cell.
    scene.set_channel_component_coloring("green", True, label_count=1, unassigned_label=2)
    scene.set_channel_data("green", labels, VoxelSpacing(x_um=1, y_um=1, z_um=1))
    config = RenderConfig(show_iso_green=True)
    scene.apply_render_config(config)
    actor = scene._actors["green"]
    mapper = actor.iso_actor.GetMapper()
    mapper.Update()
    np.testing.assert_array_equal(np.unique(vtk_to_numpy(mapper.GetInput().GetCellData().GetScalars())), [1, 2])
    np.testing.assert_allclose(_color_at(mapper.GetLookupTable(), 2), [0.1, 1.0, 0.2])
    assigned_color = _color_at(mapper.GetLookupTable(), 1)

    scene.apply_render_config(replace(config, show_iso_green=False))
    np.testing.assert_allclose(_color_at(actor.volume_property.GetRGBTransferFunction(), 2), [0.1, 1.0, 0.2])
    np.testing.assert_allclose(_color_at(actor.volume_property.GetRGBTransferFunction(), 1), assigned_color)
    assert actor.volume_actor.GetVisibility()
    scene.set_channel_component_coloring("green", False)
    assert "green" not in scene._component_unassigned_labels


def test_gray_fragments_share_context_color_without_sharing_component_ids(qtbot, monkeypatch):
    scene = VTKScene()
    qtbot.addWidget(scene.widget())
    monkeypatch.setattr(scene, 'render', lambda: None)
    labels = np.zeros((8, 10, 20), dtype=np.uint16)
    labels[2:6, 2:7, 2:6] = 1
    labels[3:5, 3:6, 10:12] = 2
    labels[3:5, 3:6, 16:18] = 3
    scene.set_channel_component_coloring('green', True, label_count=3, gray_labels=[2, 3])
    scene.set_channel_data('green', labels, VoxelSpacing())
    scene.apply_render_config(RenderConfig(show_iso_green=True))
    actor = scene._actors['green']
    mapper = actor.iso_actor.GetMapper()
    mapper.Update()
    np.testing.assert_array_equal(np.unique(vtk_to_numpy(mapper.GetInput().GetCellData().GetScalars())), [1, 2, 3])
    for cid in [2, 3]:
        np.testing.assert_allclose(_color_at(mapper.GetLookupTable(), cid), [.48, .50, .54])
    assert not np.allclose(_color_at(mapper.GetLookupTable(), 1), [.48, .50, .54])
    image_time = actor.image.GetMTime()
    contours_time = actor.marching.GetMTime()
    scene.set_gray_branches_visible(False)
    assert mapper.GetLookupTable().GetOpacity(2) == 0.0
    assert mapper.GetLookupTable().GetOpacity(1) == 1.0
    assert actor.image.GetMTime() == image_time
    assert actor.marching.GetMTime() == contours_time
    scene.set_gray_branches_visible(True)
    assert mapper.GetLookupTable().GetOpacity(2) == 1.0
    scene.apply_render_config(replace(RenderConfig(), show_iso_green=False))
    np.testing.assert_allclose(_color_at(actor.volume_property.GetRGBTransferFunction(), 3), [.48, .50, .54])
    scene.set_gray_branches_visible(False)
    assert actor.volume_property.GetScalarOpacity().GetValue(3) == 0.0
    assert actor.volume_property.GetScalarOpacity().GetValue(1) > 0.0


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
