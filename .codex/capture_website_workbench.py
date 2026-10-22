import json
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from nvap.config.types import ChannelVolume,DatasetVolume,VoxelSpacing
from nvap.analysis.metrics import compute_metrics
from nvap.ui.main_window import MainWindow
from nvap.ui.theme import DARK_THEME_QSS
root=Path.cwd();data=Path('C:/Users/giaco/Downloads/NVAP_Output/cortex_7015-4_tuned')
app=QApplication([]);app.setStyleSheet(DARK_THEME_QSS)
w=MainWindow();w.resize(1600,960);w.show();w._nav_to(1);QTest.qWait(300)
meta=json.loads((data/'provenance.json').read_text());spacing=VoxelSpacing(**meta['spacing'])
green=np.load(data/'green_clean.npy');red=np.load(data/'red_clean.npy');print('Volume:',green.shape,flush=True)
z=list(range(green.shape[0]));ds=DatasetVolume(ChannelVolume('green',green,z,spacing),ChannelVolume('red',red,z,spacing),(0,len(z)-1))
w.spacing=spacing;w.processed_dataset=ds;w.visual_dataset=ds
c=w.controls;c.blockSignals(True);c.set_threshold_defaults(meta['threshold_green'],meta['threshold_red']);c.display_z_scale.setValue(1.0);c.set_voxel_spacing(spacing,source='CZI metadata',emit=False);c.blockSignals(False)
w.current_render=c.current_render_config();w._push_scene_channels();w.scene.apply_render_config(w.current_render);w.scene.reset_camera()
c.set_microglia_workflow_enabled(True)
w._metrics_panel.update_from_metrics(compute_metrics(ds,w.current_render))
w.statusBar().showMessage('Example cortex volume | green: microglia | red: vasculature')
QTest.qWait(900)
# VTK has a native child window, so capture its real framebuffer separately.
from PySide6.QtGui import QPainter,QImage
from PySide6.QtCore import QPoint
w.scene.render()
w.scene.capture_snapshot(root/'.codex/workbench-viewport.png',scale=1)
shot=w.grab();painter=QPainter(shot);viewport=w.scene.widget()
position=viewport.mapTo(w,QPoint(0,0))
painter.drawImage(position.x(),position.y(),QImage(str(root/'.codex/workbench-viewport.png')).scaled(viewport.size()))
painter.end();shot.save(str(root/'website/assets/nvap-workbench.png'));print('Screenshot saved',flush=True)
w.close();app.processEvents()
