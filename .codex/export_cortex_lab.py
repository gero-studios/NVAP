import sys,json,logging,hashlib,csv,zipfile,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'));sys.path.insert(0,str(root))
from nvap.config.types import ChannelVolume,DatasetVolume,RenderConfig,VoxelSpacing
from nvap.analysis.metrics import compute_metrics,metrics_to_csv_rows
from nvap.analysis.microglia_analysis import analyze_microglia_cells,microglia_analysis_to_csv_rows
from nvap.analysis.neurovascular import summarize_neurovascular_association,neurovascular_association_to_csv_rows
from nvap.export.exporters import export_metrics_csv
from scripts.export_final_ppa_saline import _render_viewport,_normalise_rows
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
src=Path(r'C:\Users\giaco\Downloads\NVAP_Output\cortex_7015-4_tuned')
out=src.parent/'7015-4_cortex_lab_segmented_only';out.mkdir(exist_ok=True)
meta=json.loads((src/'provenance.json').read_text());s=VoxelSpacing(**meta['spacing']);g=np.load(src/'green_clean.npy');r=np.load(src/'red_clean.npy');saved=np.load(src/'analysis_masks.npz')
labels=saved['microglia_labels'].copy()
# The analysis population is exactly the assigned signal visible at the selected threshold.
visible=(labels>0)&(g>=meta['threshold_green']);labels[~visible]=0
green=np.where(visible,g,0).astype(np.float32)
order=np.unique(labels);order=order[order>0].astype(np.int32)
z=list(range(1,g.shape[0]+1));ds=DatasetVolume(ChannelVolume('green',green,z,s),ChannelVolume('red',r,z,s),(1,len(z)))
render=RenderConfig(threshold_green=meta['threshold_green'],threshold_red=meta['threshold_red'],display_z_scale=1.)
basic=metrics_to_csv_rows(compute_metrics(ds,render))
# Give the overlap row its actual count/volume as well as the named overlap fields.
basic[-1]['voxel_count']=basic[-1]['overlap_voxel_count'];basic[-1]['volume_um3']=basic[-1]['overlap_volume_um3']
export_metrics_csv(basic,out/'metrics.csv')
shutil.copy2(src/'metrics_vascular.csv',out/'metrics_vascular.csv')
logging.info('Recomputing morphology and vessel association for %d visible segmented cells',len(order))
micro=analyze_microglia_cells(green,r,labels,order,spacing=s,render=render,branch_sensitivity=1.)
cell_rows=microglia_analysis_to_csv_rows(micro);neuro_rows=neurovascular_association_to_csv_rows(summarize_neurovascular_association(micro))
export_metrics_csv(cell_rows,out/'metrics_microglia.csv');export_metrics_csv(neuro_rows,out/'metrics_neurovascular.csv')
wall=saved['vascular_wall_mask'];solid=saved['vascular_solid_mask']
np.savez_compressed(out/'analysis_masks.npz',microglia_visible_mask=visible.astype(np.uint8),microglia_cell_labels=labels.astype(np.uint16),vascular_wall_mask=wall,vascular_solid_mask=solid)
_render_viewport(out/'segmented_microglia_3d.png',solid_vessel=solid,vessel_spacing=s,microglia_labels=labels,microglia_spacing=s)
with (out/'metrics_vascular.csv').open(newline='') as f:vascular_rows=list(csv.DictReader(f))
combined=[]
for family,rows,key in [('basic',basic,None),('vascular',vascular_rows,None),('microglia_cell',cell_rows,'component_id'),('neurovascular',neuro_rows,None)]:
    combined.extend(_normalise_rows(sample='7015-4_cortex',source=Path(meta['source']),family=family,rows=rows,component_key=key))
export_metrics_csv(combined,out/'all_metrics_long_format.csv',columns=('sample','source_file','metric_family','component_id','metric','value','unit'))
provenance={k:v for k,v in meta.items() if k not in ['per_cell_population_complete','unassigned_green_fraction','preview_uses_full_thresholded_green']}
provenance.update(exported_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(meta['source']).read_bytes()).hexdigest(),analysis_population='retained segmented cell labels intersected with green intensity >= selected threshold',exclusion_rule='All label-zero green signal and all subthreshold green voxels excluded from every green and cell-dependent measurement, as requested by the user.',excluded_signal_identity='Excluded by selected analysis rule; biological identity not independently verified.',cell_count=len(cell_rows),preview_matches_analysis_mask=True,cell_morphology_recomputed_on_visible_mask=True)
(out/'provenance.json').write_text(json.dumps(provenance,indent=2))
vv=s.voxel_volume_um3
counts={int(cid):int((labels==cid).sum()) for cid in order}
assert int(basic[0]['voxel_count'])==int(visible.sum())==sum(int(row['voxel_count']) for row in cell_rows)
for row in cell_rows:assert int(row['voxel_count'])==counts[int(row['component_id'])]
assert len(cell_rows)==len(order)
assert not np.any((labels>0)&~visible)
assert int(basic[0]['overlap_voxel_count'])==int((visible&(r>=meta['threshold_red'])).sum())
assert np.isclose(float(basic[0]['volume_um3']),visible.sum()*vv)
nv={row['metric']:row['value'] for row in neuro_rows};assert int(nv['cell_count'])==len(cell_rows)
summary=dict(cell_count=len(cell_rows),microglia_voxel_count=int(visible.sum()),microglia_volume_um3=float(visible.sum()*vv),overlap_voxel_count=int(basic[0]['overlap_voxel_count']),overlap_volume_um3=float(basic[0]['overlap_volume_um3']),threshold_green=meta['threshold_green'],threshold_red=meta['threshold_red'],mean_process_length_um=micro.mean_process_length_um,mean_soma_volume_um3=micro.mean_soma_volume_um3,mean_branch_count=micro.mean_branch_count,mean_tip_count=micro.mean_tip_count)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
(out/'validation.json').write_text(json.dumps(dict(passed=True,checks=['All counted green voxels belong to retained cell labels and pass the display threshold','Sum of per-cell voxel counts equals the basic green count','Every cell CSV voxel count matches its exported label mask','Overlap independently recomputed from the same visible green mask','Physical volumes match native voxel spacing','Neurovascular population count matches cell table','3D preview uses exactly the measured green mask']),indent=2))
readme=f'''NVAP LAB EXPORT — 7015-4 cortex — visible segmented microglia only

Source: {Path(meta['source']).name}
Population: {len(cell_rows)} retained segmented cells. Only green voxels belonging to these cells AND at/above threshold {meta['threshold_green']:.6f} are counted. All other green signal is excluded, per the requested analysis definition. This exclusion is an analysis choice, not independent confirmation that all excluded material is debris.
Red threshold: {meta['threshold_red']:.6f}. Green enhancement: microscopy_clean. Native grid z/y/x: {meta['shape']}. Voxel spacing x/y/z (um): {s.x_um}, {s.y_um}, {s.z_um}. No trim, offsets or measurement resampling.

metrics.csv: visible segmented green and thresholded red counts/volumes and overlap. Connected-component count measures mask connectivity; it is distinct from the 11 retained cell identities.
metrics_microglia.csv: morphology, coordinates and vessel distances recalculated from the visible segmented cell mask. Disconnected visible branches retain their original cell identity.
metrics_neurovascular.csv: association statistics for the same retained cell population; distances use the fluorescent vessel WALL mask.
metrics_vascular.csv: unchanged vascular analysis, separately reporting fluorescent walls and reconstructed filled vessels.
all_metrics_long_format.csv: all four measurement families combined for importing into analysis software.
analysis_masks.npz: native-grid arrays in z,y,x order, including microglia_visible_mask, microglia_cell_labels, vascular_wall_mask, vascular_solid_mask. Cell IDs match component_id in the per-cell CSV.
segmented_microglia_3d.png: the EXACT visible microglia mask measured in this package (green) and reconstructed solid vessels (red), with physical spacing.
provenance.json: processing settings, analysis population and source SHA-256.
summary.json / validation.json: summary and consistency checks.
manifest.json: file sizes and SHA-256 digests.

This package replaces the earlier exports for the requested segmented-only analysis. No whole-green or unassigned-signal population metrics are mixed into it. Cell identities and morphology are algorithmic estimates; field-edge cells can be truncated. True channel overlap is retained.
'''
(out/'README.txt').write_text(readme,encoding='utf-8')
manifest=[]
for f in sorted(out.iterdir()):
    if f.name!='manifest.json':manifest.append(dict(file=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
archive_path=out.parent/'7015-4_cortex_lab_segmented_only.zip'
with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
    for f in sorted(out.iterdir()):archive.write(f,f.name)
with zipfile.ZipFile(archive_path) as archive:assert archive.testzip() is None
logging.info('COMPLETE %s %s',archive_path,summary)
