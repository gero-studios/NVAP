from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nvap.config.types import VoxelSpacing


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
