import os, sys, json, logging
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
from nvap.config.types import DatasetVolume, PreprocessConfig, RenderConfig, ChannelVolume, VoxelSpacing
from nvap.io.czi_loader import load_czi_channels
from nvap.pipeline import automatic_thresholds
from nvap.preprocess.enhancement import preprocess_dataset, enhance_microglia_background, wipe_small_specks, wipe_vasculature_blobs
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
out=Path(r'C:\Users\giaco\Downloads\NVAP_Output\cortex_7015-4_tuned')
out.mkdir(parents=True,exist_ok=True)
source=Path(r'C:\Users\giaco\Downloads\7015-4_M_CL_cortex_63X (1).czi')
if sys.argv[1]=='prepare':
    g,r,s=load_czi_channels(source)
    ds=DatasetVolume(g,r,(1,g.data.shape[0]))
    cfg=PreprocessConfig(cpu_worker_threads=12,resample_for_mesh=False)
    ds=preprocess_dataset(ds,cfg)
    green=enhance_microglia_background(ds.green.data,cfg,method='microscopy_clean')
    ds.green.data=green
    tg,tr=automatic_thresholds(ds)
    np.save(out/'green.npy',green); np.save(out/'red.npy',ds.red.data)
    meta=dict(source=str(source),spacing=dict(x_um=s.x_um,y_um=s.y_um,z_um=s.z_um),shape=list(green.shape),automatic_green=tg,automatic_red=tr)
    (out/'preparation.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)
    for name,data,base in [('green',green,tg),('red',ds.red.data,tr)]:
        candidates=sorted(set([round(float(base),4),0.25,0.30,0.35,0.40] if name=='green' else [round(float(base),4),0.40,0.45,0.50,0.55]))
        panel=Image.new('RGB',(512*len(candidates),560),'#16191e'); draw=ImageDraw.Draw(panel)
        rows=[]
        for i,t in enumerate(candidates):
            mask=data>=t
            mip=mask.max(axis=0)
            rgb=np.zeros((*mip.shape,3),dtype=np.uint8); rgb[...,1 if name=='green' else 0]=mip*255
            panel.paste(Image.fromarray(rgb).resize((512,512)),(512*i,40))
            draw.text((512*i+10,10),f'{name} threshold {t:.4f} occupancy {mask.mean():.2%}',fill='white')
            rows.append(dict(threshold=t,voxels=int(mask.sum()),fraction=float(mask.mean())))
        panel.save(out/f'{name}_threshold_comparison.png')
        (out/f'{name}_threshold_candidates.json').write_text(json.dumps(rows,indent=2))
else:
    from nvap.analysis.metrics import compute_metrics,metrics_to_csv_rows
    from nvap.analysis.vascular_analysis import analyze_vasculature,vascular_analysis_to_csv_rows,build_vascular_masks
    from nvap.analysis.microglia_components import compute_component_labels,filter_components_by_preferred_voxel_floor
    from nvap.analysis.microglia_analysis import analyze_microglia_cells,microglia_analysis_to_csv_rows
    from nvap.analysis.neurovascular import summarize_neurovascular_association,neurovascular_association_to_csv_rows
    from nvap.export.exporters import export_metrics_csv
    from scripts.export_final_ppa_saline import _render_viewport
    meta=json.loads((out/'preparation.json').read_text()); tg=float(sys.argv[2]); tr=float(sys.argv[3]); s=VoxelSpacing(**meta['spacing'])
    green=wipe_small_specks(np.load(out/'green.npy'),threshold=tg,min_voxels=128)
    red=wipe_vasculature_blobs(np.load(out/'red.npy'),threshold=tr,max_voxels=2048)
    z=list(range(1,green.shape[0]+1)); ds=DatasetVolume(ChannelVolume('green',green,z,s),ChannelVolume('red',red,z,s),(1,len(z)))
    render=RenderConfig(threshold_green=tg,threshold_red=tr,display_z_scale=1.0)
    meta.update(threshold_green=tg,threshold_red=tr,speck_max_voxels=128,vascular_blob_max_voxels=2048,enhancement='microscopy_clean',analysis_grid='native CZI voxels; no display resampling',method='branch-aware automatic threshold and visually reviewed candidate projections',overlap_policy='retain true channel overlap')
    (out/'provenance.json').write_text(json.dumps(meta,indent=2))
    export_metrics_csv(metrics_to_csv_rows(compute_metrics(ds,render)),out/'metrics.csv')
    logging.info('Computing vascular metrics')
    vascular=analyze_vasculature(red,threshold=tr,spacing=s,render=render)
    export_metrics_csv(vascular_analysis_to_csv_rows(vascular),out/'metrics_vascular.csv')
    logging.info('Separating microglia')
    labels,order,sizes=compute_component_labels(green,threshold=tg,min_voxels=53,max_components=512,smooth_sigma=(0.2,0.45,0.45),branch_sensitivity=1.0,spacing=(s.z_um,s.y_um,s.x_um))
    labels,order,sizes=filter_components_by_preferred_voxel_floor(labels,order,sizes)
    logging.info('Computing per-cell metrics for %d cells',len(order))
    micro=analyze_microglia_cells(green,red,labels,order,spacing=s,render=render,branch_sensitivity=1.0)
    export_metrics_csv(microglia_analysis_to_csv_rows(micro),out/'metrics_microglia.csv')
    export_metrics_csv(neurovascular_association_to_csv_rows(summarize_neurovascular_association(micro)),out/'metrics_neurovascular.csv')
    masks=build_vascular_masks(red,threshold=tr,spacing=s,render=render)
    np.savez_compressed(out/'analysis_masks.npz',vascular_wall_mask=masks.wall_mask.astype(np.uint8),vascular_solid_mask=masks.solid_mask.astype(np.uint8),microglia_labels=labels.astype(np.uint16))
    _render_viewport(out/'tuned_3d.png',solid_vessel=masks.solid_mask,vessel_spacing=s,microglia_labels=labels,microglia_spacing=s)
    summary=dict(cells=micro.analyzed_cell_count,vascular_components=vascular.component_count,vessel_volume_fraction=vascular.vessel_volume_fraction,mean_diameter_um=vascular.mean_diameter_um,threshold_green=tg,threshold_red=tr)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    logging.info('COMPLETE %s',summary)
