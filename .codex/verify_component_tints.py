"""Render existing cortex labels through the component display pipeline."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication
from vtkmodules.util.numpy_support import vtk_to_numpy

from nvap.config.types import RenderConfig, VoxelSpacing
from nvap.render.vtk_scene import VTKScene

root = Path(__file__).resolve().parents[1]
source = Path('C:/Users/giaco/Downloads/NVAP_Output/cortex_7015-4_tuned')
meta = json.loads((source / 'provenance.json').read_text())
labels = np.load(root / '.codex/current_segmentation_audit/current_labels.npy', mmap_mode='r')
labels = np.ascontiguousarray(labels[:, ::2, ::2], dtype=np.uint8)
labels[:10] = labels[-10:] = 0
spacing = VoxelSpacing(**meta['spacing'])
spacing = replace(spacing, x_um=spacing.x_um * 2, y_um=spacing.y_um * 2)
app = QApplication([])
scene = VTKScene()
scene.widget().resize(1000, 760)
scene._render_window.SetOffScreenRendering(1)
scene.set_channel_component_coloring('green', True, label_count=int(labels.max()))
scene.set_channel_data('green', labels, spacing)
config = RenderConfig(show_red=False, show_iso_green=True, opacity_green=1.0)
scene.apply_render_config(config)
camera = scene._renderer.GetActiveCamera()
camera.Azimuth(15)
camera.Elevation(40)
scene._renderer.ResetCamera()
scene.capture_snapshot(root / '.codex/component-tints-surface.png', scale=1)
mesh = scene._actors['green'].iso_actor.GetMapper().GetInput()
print('Surface component IDs:', np.unique(vtk_to_numpy(mesh.GetCellData().GetScalars())).tolist(), flush=True)
scene.apply_render_config(replace(config, show_iso_green=False))
scene.capture_snapshot(root / '.codex/component-tints-volume.png', scale=1)
scene.widget().close()
print('Saved surface and volume previews.', flush=True)
