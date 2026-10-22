"""Verify the real cortex All view keeps the complete threshold mask."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication

from nvap.config.types import ChannelVolume, DatasetVolume, VoxelSpacing
from nvap.render.vtk_scene import _block_reduce_max
from nvap.ui.main_window import MainWindow

root = Path(__file__).resolve().parents[1]
source = Path('C:/Users/giaco/Downloads/NVAP_Output/cortex_7015-4_tuned')
meta = json.loads((source / 'provenance.json').read_text())
green = np.load(source / 'green_clean.npy', mmap_mode='r')
red = np.load(source / 'red_clean.npy', mmap_mode='r')
labels = np.load(root / '.codex/current_segmentation_audit/current_labels.npy', mmap_mode='r')
order = np.arange(1, int(labels.max()) + 1, dtype=np.int32)
sizes = np.bincount(labels.ravel())
spacing = VoxelSpacing(**meta['spacing'])
z = list(range(green.shape[0]))
dataset = DatasetVolume(ChannelVolume('green', green, z, spacing), ChannelVolume('red', red, z, spacing), (0, len(z)-1))
app = QApplication([])
window = MainWindow()
window.processed_dataset = window.visual_dataset = dataset
window.controls.microglia_isolate.blockSignals(True)
window.controls.microglia_isolate.setChecked(True)
window.controls.microglia_isolate.blockSignals(False)
window.current_render = replace(window.current_render, threshold_green=meta['threshold_green'], threshold_red=meta['threshold_red'])
mask = green >= meta['threshold_green']
report = {
    'source_threshold_voxels': int(mask.sum()),
    'previously_hidden_voxels': int(np.count_nonzero(mask & (labels == 0))),
    'detected_cells': len(order),
}
for sparse in (False, True):
    window._microglia_label_cache_max_bytes = 0 if sparse else 256 * 1024 * 1024
    window._cache_microglia_components(
        labels, order, sizes, threshold=meta['threshold_green'],
        branch_sensitivity=window.controls.current_microglia_branch_sensitivity(),
        shape=green.shape, selected_index=0,
    )
    view = window._current_green_volume_for_view()
    np.testing.assert_array_equal(view > 0, mask)
    np.testing.assert_array_equal(window._green_component_order, order)
    expected_trim = window._apply_render_trim(mask, 10, 10)
    view = window._apply_render_trim(view, 10, 10)
    np.testing.assert_array_equal(view > 0, expected_trim)
    report['sparse' if sparse else 'dense'] = {
        'missing_voxels_after_fix': int(np.count_nonzero(expected_trim & (view == 0))),
        'visible_voxels_after_trim': int(np.count_nonzero(view)),
    }
    print(json.dumps(report), flush=True)

# Use the same foreground-preserving reduction as large render uploads.
preview = _block_reduce_max(view, (1, 2, 2))
red_preview = _block_reduce_max(window._apply_render_trim(red, 10, 10), (1, 2, 2))
preview_spacing = replace(spacing, x_um=spacing.x_um * 2, y_um=spacing.y_um * 2)
scene = window.scene
scene._render_window.SetOffScreenRendering(1)
scene._render_window.SetSize(1100, 900)
scene.set_channel_component_coloring('green', True, label_count=len(order), unassigned_label=len(order)+1)
scene.set_channel_data('green', preview, preview_spacing)
scene.set_channel_data('red', red_preview, preview_spacing)
scene.apply_render_config(window.current_render)
camera = scene._renderer.GetActiveCamera()
camera.Azimuth(15)
camera.Elevation(40)
scene._renderer.ResetCamera()
scene.capture_snapshot(root / '.codex/microglia-full-coverage-fixed.png', scale=1)
scene.apply_render_config(replace(window.current_render, show_iso_green=False))
scene.capture_snapshot(root / '.codex/microglia-full-coverage-volume.png', scale=1)
(root / '.codex/microglia-full-coverage-validation.json').write_text(json.dumps(report, indent=2))
window.close()
print('Full native-grid coverage and both renderer previews verified.', flush=True)
