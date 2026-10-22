import sys,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from nvap.config.types import ChannelVolume,DatasetVolume,VoxelSpacing,PreprocessConfig
from nvap.ui.main_window import MainWindow
from nvap.ui.theme import DARK_THEME_QSS
p=Path(r'C:\Users\giaco\Downloads\NVAP_Output\cortex_7015-4_tuned')
app=QApplication([]);app.setStyleSheet(DARK_THEME_QSS)
w=MainWindow();w.setWindowTitle('NVAP — 7015-4 cortex — tuned thresholds');w.show()
def load():
    meta=json.loads((p/'provenance.json').read_text());s=VoxelSpacing(**meta['spacing'])
    g=np.load(p/'green_clean.npy');r=np.load(p/'red_clean.npy');z=list(range(1,g.shape[0]+1))
    ds=DatasetVolume(ChannelVolume('green',g,z,s),ChannelVolume('red',r,z,s),(1,len(z)))
    w.dataset_root=p;w.spacing=s;w.preprocess_config=PreprocessConfig(resample_for_mesh=False)
    w.raw_dataset=ds;w.synced_dataset=ds;w.processed_dataset=ds;w.visual_dataset=ds
    c=w.controls;c.blockSignals(True);c.set_threshold_defaults(meta['threshold_green'],meta['threshold_red']);c.display_z_scale.setValue(1.0)
    c.set_voxel_spacing(s,source='CZI metadata',emit=False);c.auto_wipe_on_threshold_edit.setChecked(False);c.blockSignals(False)
    w.current_render=c.current_render_config();w._current_microglia_enhancement_method='microscopy_clean';w._current_speck_wipe_applied=True
    w._push_scene_channels();w.scene.apply_render_config(w.current_render);w._nav_to(1);w._refresh_section_state()
    w.statusBar().showMessage('Tuned cortex: green 0.385 / red 0.500. Analysis exports saved in this dataset folder.')
    w.raise_();w.activateWindow()
QTimer.singleShot(200,load)
sys.exit(app.exec())
