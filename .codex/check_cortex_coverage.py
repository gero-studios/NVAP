import sys,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'));sys.path.insert(0,str(root))
import numpy as np
from PIL import Image,ImageDraw
from scripts.export_final_ppa_saline import _render_viewport
from nvap.config.types import VoxelSpacing
p=Path(r'C:\Users\giaco\Downloads\NVAP_Output\cortex_7015-4_tuned')
meta=json.loads((p/'provenance.json').read_text());g=np.load(p/'green_clean.npy');m=np.load(p/'analysis_masks.npz');labels=m['microglia_labels'];mask=g>=meta['threshold_green'];missing=mask&(labels==0)
from scipy import ndimage as ndi
cc,n=ndi.label(missing);sizes=np.bincount(cc.ravel());order=np.argsort(sizes[1:])[::-1]+1
result=dict(thresholded_green_voxels=int(mask.sum()),assigned_above_threshold_voxels=int((mask&(labels>0)).sum()),unassigned_above_threshold_voxels=int(missing.sum()),unassigned_fraction=float(missing.sum()/mask.sum()),largest_unassigned_components_voxels=[int(sizes[i]) for i in order[:20]],preview_policy='full cleaned green threshold mask; cell labels are a separate subset',per_cell_analysis_status='partial coverage; 11 assigned cells only')
(p/'segmentation_coverage.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
rgb=np.zeros((1024,1024,3),np.uint8);rgb[...,1]=((mask&(labels>0)).max(axis=0))*255;mis=missing.max(axis=0);rgb[...,0]=mis*255;rgb[...,2]=mis*255
Image.fromarray(rgb).save(p/'cell_assignment_coverage.png')
_render_viewport(p/'full_thresholded_3d.png',solid_vessel=m['vascular_solid_mask'],vessel_spacing=VoxelSpacing(**meta['spacing']),microglia_labels=mask,microglia_spacing=VoxelSpacing(**meta['spacing']))
