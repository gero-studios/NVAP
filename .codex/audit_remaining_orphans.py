import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
p=Path('.codex/cell-assignment-fixed')
a=np.load(p/'watershed_labels.npy',mmap_mode='r'); c=np.load(p/'candidate_mask.npy',mmap_mode='r')
g=np.load('C:/Users/giaco/Downloads/NVAP_Output/cortex_7015-4_tuned/green_clean.npy',mmap_mode='r')[:len(a)]
w=ndi.gaussian_filter(g,sigma=(.2,.45,.45))
o,n=ndi.label(c&(a==0),structure=np.ones((3,3,3)))
s=np.bincount(o.ravel()); objects=ndi.find_objects(o); parts=[]
for cid in np.argsort(s[1:])[::-1][:35]+1:
 b=objects[cid-1]; m=o[b]==cid; body=m&(w[b]>=.384765625)
 d=ndi.distance_transform_edt(np.pad(body,1),sampling=(.4,.1315305679563492,.1315305679563492))[1:-1,1:-1,1:-1]
 pos=np.unravel_index(d.argmax(),d.shape); center=[int(pos[k]+b[k].start) for k in range(3)]
 part=dict(id=int(cid),size=int(s[cid]),visible=int(np.count_nonzero(m&(g[b]>=.384765625))),radius=float(d.max()),center=center,bounds=[[v.start,v.stop] for v in b])
 parts.append(part)
print(json.dumps(parts,indent=2)); (p/'remaining_orphans.json').write_text(json.dumps(parts,indent=2));np.save(p/'remaining_orphans.npy',o.astype(np.uint16))
