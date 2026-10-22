import json,zipfile,shutil
from pathlib import Path
import numpy as np
p=Path(r'C:\Users\giaco\Downloads\NVAP_Output\cortex_7015-4_tuned')
coverage=json.loads((p/'segmentation_coverage.json').read_text())
shutil.copy2(p/'tuned_3d.png',p/'assigned_cells_only_3d.png')
shutil.copy2(p/'full_thresholded_3d.png',p/'tuned_3d.png')
g=np.load(p/'green_clean.npy',mmap_mode='r');meta=json.loads((p/'provenance.json').read_text());m=np.load(p/'analysis_masks.npz')
np.savez_compressed(p/'full_green_coverage_masks.npz',microglia_threshold_mask=(g>=meta['threshold_green']).astype(np.uint8),unassigned_green_mask=((g>=meta['threshold_green'])&(m['microglia_labels']==0)).astype(np.uint8))
for name in ['provenance.json','summary.json','export_validation.json']:
    obj=json.loads((p/name).read_text());obj['per_cell_population_complete']=False;obj['unassigned_green_fraction']=coverage['unassigned_fraction'];obj['preview_uses_full_thresholded_green']=True
    (p/name).write_text(json.dumps(obj,indent=2))
with (p/'README.txt').open('a',encoding='utf-8') as f:
    f.write('\nCORRECTION: The original 3D preview displayed only assigned cell labels and omitted 31.82% of above-threshold green signal. tuned_3d.png now displays the FULL thresholded green mask. assigned_cells_only_3d.png preserves the earlier subset view. Basic green-volume and vascular CSVs cover their full selected masks. Per-cell and neurovascular CSVs describe only 11 assigned cells and MUST NOT be treated as complete population measurements. Green/magenta in cell_assignment_coverage.png indicates assigned/unassigned thresholded signal. Full green and unassigned masks are saved separately; missing fragments are not relabeled as artificial cells.\n')
with zipfile.ZipFile(p.parent/'7015-4_cortex_tuned_results.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for file in p.iterdir():
        if file.suffix in {'.csv','.json','.png','.npz','.txt'}:archive.write(file,file.name)
print('Corrected preview, coverage masks, completeness flags, README, and ZIP.')
