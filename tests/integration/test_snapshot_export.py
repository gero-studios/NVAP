from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from nvap.config.types import VoxelSpacing


@pytest.mark.integration
def test_cubic_render_spacing_refines_coarse_axis() -> None:
    pytest.importorskip("vtkmodules")

    from nvap.render.vtk_scene import _cubic_render_spacing

    render_spacing = _cubic_render_spacing(
        (8, 16, 16),
        VoxelSpacing(x_um=0.33, y_um=0.33, z_um=0.66),
    )

    assert render_spacing.z_um < 0.66
    assert render_spacing.x_um == pytest.approx(0.33)
    assert render_spacing.y_um == pytest.approx(0.33)


@pytest.mark.integration
def test_cubic_render_spacing_keeps_isotropic_volume_unchanged() -> None:
    pytest.importorskip("vtkmodules")

    from nvap.render.vtk_scene import _cubic_render_spacing

    render_spacing = _cubic_render_spacing(
        (8, 16, 16),
        VoxelSpacing(x_um=0.33, y_um=0.33, z_um=0.33),
    )

    assert render_spacing == VoxelSpacing(x_um=0.33, y_um=0.33, z_um=0.33)


@pytest.mark.integration
def test_cubic_render_spacing_falls_back_for_nonpositive_spacing() -> None:
    pytest.importorskip("vtkmodules")

    from nvap.render.vtk_scene import _cubic_render_spacing

    spacing = VoxelSpacing(x_um=0.33, y_um=0.33, z_um=0.0)
    render_spacing = _cubic_render_spacing((8, 16, 16), spacing)

    assert render_spacing == spacing


@pytest.mark.integration
def test_cubic_render_spacing_falls_back_for_nonfinite_spacing() -> None:
    pytest.importorskip("vtkmodules")

    from nvap.render.vtk_scene import _cubic_render_spacing

    spacing = VoxelSpacing(x_um=0.33, y_um=float("nan"), z_um=0.66)
    render_spacing = _cubic_render_spacing((8, 16, 16), spacing)

    assert np.isnan(render_spacing.y_um)
    assert render_spacing.x_um == pytest.approx(0.33)
    assert render_spacing.z_um == pytest.approx(0.66)


@pytest.mark.integration
def test_scene_uses_vtk_cubic_resample_for_anisotropic_volume() -> None:
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("vtkmodules")

    from nvap.render.vtk_scene import VTKScene

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scene = VTKScene()
    volume = np.zeros((6, 16, 16), dtype=np.float32)
    volume[:, 8, 8] = 1.0
    scene.set_channel_data("green", volume, VoxelSpacing(x_um=0.33, y_um=0.33, z_um=0.66))

    actor = scene._actors["green"]
    assert actor.resample is not None
    assert actor.resample.GetInterpolationModeAsString().lower() == "linear"
    assert scene._spacing["green"].z_um < 0.66

    scene.widget().close()
    app.processEvents()


@pytest.mark.integration
def test_snapshot_export(tmp_path: Path) -> None:
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("vtkmodules")

    from nvap.render.vtk_scene import VTKScene

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scene = VTKScene()
    volume = np.zeros((8, 32, 32), dtype=np.float32)
    volume[:, 16, 16] = 1.0
    scene.set_channel_data("green", volume, VoxelSpacing())
    out = scene.capture_snapshot(tmp_path / "scene.png")
    scene.widget().close()
    app.processEvents()

    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.integration
def test_scene_builds_surface_pipeline_only_when_surface_mode_is_enabled() -> None:
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("vtkmodules")

    from nvap.config.types import RenderConfig
    from nvap.render.vtk_scene import VTKScene

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scene = VTKScene()
    scene.apply_render_config(RenderConfig(show_iso_green=False, show_iso_red=False))
    volume = np.zeros((8, 24, 24), dtype=np.float32)
    volume[:, 12, 12] = 1.0

    scene.set_channel_data("green", volume, VoxelSpacing())

    actor = scene._actors["green"]
    assert actor.marching is None
    assert actor.volume_actor.GetVisibility() == 1
    assert actor.iso_actor.GetVisibility() == 0

    scene.apply_render_config(replace(scene._current, show_iso_green=True))

    assert actor.marching is not None
    assert actor.volume_actor.GetVisibility() == 0
    assert actor.iso_actor.GetVisibility() == 1

    scene.widget().close()
    app.processEvents()


@pytest.mark.integration
def test_scene_falls_back_to_volume_render_when_surface_pipeline_is_blocked(monkeypatch) -> None:
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("vtkmodules")

    from nvap.render import vtk_scene as vtk_scene_module
    from nvap.render.vtk_scene import VTKScene

    monkeypatch.setattr(vtk_scene_module, "_SURFACE_PIPELINE_MAX_VOXELS", 32)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scene = VTKScene()
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    volume[1:3, 1:3, 1:3] = 1.0

    scene.set_channel_data("green", volume, VoxelSpacing())

    actor = scene._actors["green"]
    assert actor.marching is None
    assert actor.volume_actor.GetVisibility() == 1
    assert actor.iso_actor.GetVisibility() == 0

    scene.widget().close()
    app.processEvents()


@pytest.mark.integration
def test_scene_enables_volume_shading_for_depth_cues() -> None:
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("vtkmodules")

    from nvap.render.vtk_scene import VTKScene

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scene = VTKScene()
    volume = np.zeros((6, 16, 16), dtype=np.float32)
    volume[:, 8, 8] = 1.0
    scene.set_channel_data("green", volume, VoxelSpacing())

    assert scene._actors["green"].volume_property.GetShade() == 1

    scene.widget().close()
    app.processEvents()


@pytest.mark.integration
def test_numpy_to_vtk_scalar_order_preserves_zyx_layout() -> None:
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("vtkmodules")

    from vtkmodules.util.numpy_support import vtk_to_numpy

    from nvap.render.vtk_scene import VTKScene

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scene = VTKScene()
    volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    image = scene._numpy_to_vtk_image(volume, VoxelSpacing())
    scalars = vtk_to_numpy(image.GetPointData().GetScalars())

    assert tuple(image.GetDimensions()) == (4, 3, 2)
    assert scalars[0] == volume[0, 0, 0]
    assert scalars[1] == volume[0, 0, 1]
    assert scalars[4] == volume[0, 1, 0]
    assert scalars[12] == volume[1, 0, 0]

    scene.widget().close()
    app.processEvents()
