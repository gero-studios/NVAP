import os
os.environ['OMP_NUM_THREADS']='1'
import json,logging,hashlib,time
from pathlib import Path
import numpy as np
from nvap.config.types import PreprocessConfig
from nvap.ui.main_window import MainWindow
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
old=Path('C:/Users/giaco/Downloads/NVAP_Output/cortex_7015-4_tuned');final=old.parent/'7015-4_cortex_lab_segmented_only';out=Path('.codex/current_segmentation_audit');out.mkdir(exist_ok=True)
meta=json.loads((old/'provenance.json').read_text());g=np.load(old/'green_clean.npy');spacing=tuple(meta['spacing'][k] for k in ['z_um','y_um','x_um']);base=max(64,PreprocessConfig().green_speckle_min_voxels*4)
start=time.time();labels,order,sizes=MainWindow._compute_microglia_components_from_params(None,g,threshold=meta['threshold_green'],branch_sense=1.,base_min_voxels=base,spacing=spacing)
visible=labels.copy();visible[g<meta['threshold_green']]=0
saved=np.load(final/'analysis_masks.npz')['microglia_cell_labels']
report={'method':'Current MainWindow._compute_microglia_components_from_params, exactly the app segmentation path','input':'Saved cleaned cortex volume, unchanged thresholds and native spacing','seconds':time.time()-start,'cell_count':len(order),'exact_match_to_october_1_export':bool(np.array_equal(visible,saved)),'different_voxels':int(np.count_nonzero(visible!=saved)),'base_min_voxels':base,'threshold':meta['threshold_green'],'branch_sensitivity':1.,'parts':[]}
for cid in order:
 coords=np.nonzero(visible==cid)
 if not len(coords[0]):continue
 report['parts'].append({'id':int(cid),'visible_voxels':len(coords[0]),'extent_zyx_um':[(int(a.max())-int(a.min())+1)*spacing[j] for j,a in enumerate(coords)]})
np.save(out/'current_labels.npy',visible)
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
