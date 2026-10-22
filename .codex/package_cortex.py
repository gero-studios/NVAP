import csv,json,zipfile
from pathlib import Path
import numpy as np
p=Path(r'C:\Users\giaco\Downloads\NVAP_Output\cortex_7015-4_tuned')
meta=json.loads((p/'provenance.json').read_text());summary=json.loads((p/'summary.json').read_text())
with (p/'metrics.csv').open(newline='') as f: basic=list(csv.DictReader(f))
with (p/'metrics_vascular.csv').open(newline='') as f: vascular={r['metric']:r['value'] for r in csv.DictReader(f)}
with (p/'metrics_microglia.csv').open(newline='') as f: cells=list(csv.DictReader(f))
with (p/'metrics_neurovascular.csv').open(newline='') as f: neuro={r['metric']:r['value'] for r in csv.DictReader(f)}
m=np.load(p/'analysis_masks.npz');g=np.load(p/'green_clean.npy',mmap_mode='r');r=np.load(p/'red_clean.npy',mmap_mode='r')
vv=np.prod(list(meta['spacing'].values()))
assert len(cells)==summary['cells']
assert int(vascular['wall_voxel_count'])==int(m['vascular_wall_mask'].sum())==int(basic[1]['voxel_count'])
assert int(vascular['solid_vessel_voxel_count'])==int(m['vascular_solid_mask'].sum())
assert np.unique(m['microglia_labels']).size-1==len(cells)
for row in basic[:2]: assert np.isclose(float(row['volume_um3']),int(row['voxel_count'])*vv)
assert int(basic[0]['voxel_count'])==int((g>=meta['threshold_green']).sum())
assert int(basic[1]['voxel_count'])==int((r>=meta['threshold_red']).sum())
assert int(basic[0]['overlap_voxel_count'])==int(((g>=meta['threshold_green'])&(r>=meta['threshold_red'])).sum())
qc=dict(verified=True,checks=['CSV volumes match voxel spacing','mask counts match vascular CSV','per-cell row count matches labels','basic masks match selected thresholds','channel overlap independently recomputed'],cells=len(cells),anatomical_radius_reliable=vascular.get('anatomical_radius_reliable'))
(p/'export_validation.json').write_text(json.dumps(qc,indent=2))
readme=f'''7015-4_M_CL_cortex_63X (1).czi — tuned NVAP analysis

Thresholds (normalized processed intensity): green {meta['threshold_green']:.6f}; red {meta['threshold_red']:.6f}.
Microglia C0, vasculature C1, inferred by NVAP from CZI metadata.
Native grid: {meta['shape']}; spacing x/y/z: {meta['spacing']} micrometers.
No slice trimming, channel offset, or measurement resampling. GUI depth scale 1.0.
Green enhancement: microscopy_clean. Green isolated speck cutoff 128 voxels; compact red blob cutoff 2048 voxels.
Thresholds chosen from automatic estimates and visual candidate/slice review. Red automatic 0.351 retained too much background; 0.500 chosen. Green branch-aware automatic 0.384766 retained for branch detail.

metrics.csv: thresholded fluorescence volumes and overlap. In the overlap row, use overlap_voxel_count / overlap_volume_um3; the general voxel_count / volume_um3 fields are placeholders in NVAP's export schema.
metrics_vascular.csv: wall staining and reconstructed solid vessel measurements, separately named.
metrics_microglia.csv: one row per algorithmically separated cell with morphology and coordinates.
metrics_neurovascular.csv: population vessel-distance and contact measurements.
analysis_masks.npz: native-grid wall/solid masks and microglia labels (z,y,x).
tuned_3d.png: solid reconstructed vessels (red) and separated microglia (green), with physical spacing.
slice_quality_check.png: fluorescence and selected masks at three representative depths.
provenance.json / export_validation.json: settings and consistency checks.

Detected cells: {summary['cells']}. These are segmentation-derived estimates, not manually confirmed cell identities. Dense adjacent branches and cells cut by field boundaries can affect segmentation. Genuine red/green overlap is retained, not erased to make channels appear separate. Threshold review does not establish biological ground truth.
'''
(p/'README.txt').write_text(readme,encoding='utf-8')
zip_path=p.parent/'7015-4_cortex_tuned_results.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as archive:
    for f in p.iterdir():
        if f.suffix in {'.csv','.json','.png','.npz','.txt'}: archive.write(f,f.name)
print(json.dumps(dict(summary=summary,validation=qc,zip=str(zip_path)),indent=2))
