from pathlib import Path
import json,struct
import numpy as np
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D,vtkQuadricDecimation,vtkPolyDataNormals
from vtkmodules.util.numpy_support import numpy_to_vtk,vtk_to_numpy
src=Path('C:/Users/giaco/Downloads/NVAP_Output/7015-4_cortex_lab_segmented_only');out=Path('website/assets')
meta=json.loads((src/'provenance.json').read_text());m=np.load(src/'analysis_masks.npz');labels=m['microglia_cell_labels'];vessel=m['vascular_solid_mask'];spacing=np.array([meta['spacing'][k] for k in ['x_um','y_um','z_um']]);vertices=[];indices=[];offset=0;counts=[]
ids,voxel_counts=np.unique(labels,return_counts=True);selected=9  # Smaller component selected for the website walkthrough.
for cid in [0]+[int(i) for i in ids if i]:
    mask=vessel if cid==0 else labels==cid
    occupied=np.nonzero(mask);lo=np.array([a.min() for a in occupied]);hi=np.array([a.max()+1 for a in occupied]);crop=np.pad(mask[tuple(slice(a,b) for a,b in zip(lo,hi))].astype(np.uint8),1)
    image=vtkImageData();image.SetDimensions(*crop.shape[::-1]);image.SetSpacing(*spacing);image.SetOrigin(*((lo[::-1]-1)*spacing));image.GetPointData().SetScalars(numpy_to_vtk(crop.ravel(),deep=True))
    marching=vtkFlyingEdges3D();marching.SetInputData(image);marching.SetValue(0,.5);marching.Update();poly=marching.GetOutput();target=65000 if cid==0 else 9500
    dec=vtkQuadricDecimation();dec.SetInputData(poly);dec.SetTargetReduction(max(0,1-target/max(1,poly.GetNumberOfCells())));dec.VolumePreservationOn();dec.Update()
    normals=vtkPolyDataNormals();normals.SetInputConnection(dec.GetOutputPort());normals.SplittingOff();normals.ConsistencyOn();normals.Update();poly=normals.GetOutput()
    pos=vtk_to_numpy(poly.GetPoints().GetData());norm=vtk_to_numpy(poly.GetPointData().GetNormals());faces=vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1,4)[:,1:]
    vertices.append(np.column_stack([pos,norm,np.full(len(pos),cid)]).astype('<f4'));indices.append((faces.ravel()+offset).astype('<u4'));offset+=len(pos);counts.append({'id':cid,'triangles':len(faces),'voxel_count':int(np.count_nonzero(mask))});print(cid,len(faces),flush=True)
v=np.concatenate(vertices);i=np.concatenate(indices);bounds=[v[:,:3].min(axis=0),v[:,:3].max(axis=0)];center=(bounds[0]+bounds[1])/2;scale=float(np.max(bounds[1]-bounds[0]));v[:,:3]=(v[:,:3]-center)*2/scale
with (out/'analyzed-cortex.bin').open('wb') as f:f.write(struct.pack('<II',len(v),len(i)));f.write(v.tobytes());f.write(i.tobytes())
public={'sample':'7015-4 cortex','cell_count':meta['cell_count'],'selected_cell':selected,'spacing_um':meta['spacing'],'source_shape_zyx':meta['shape'],'source_sha256':meta['source_sha256'],'analysis_population':meta['analysis_population'],'browser_mesh':'Surface extracted on native grid; reduced triangles for display only. No measurements recomputed.','parts':counts,'vertex_count':len(v),'index_count':len(i)}
(out/'analyzed-cortex.json').write_text(json.dumps(public,indent=2));print('Bytes',(out/'analyzed-cortex.bin').stat().st_size,flush=True)
