from __future__ import annotations

from pathlib import Path

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
    assert actor.resample.GetInterpolationModeAsString().lower() == "cubic"
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
