import json,numpy as np
from scipy import ndimage as ndi
p='.codex/cell-assignment-fixed/'
o=np.load(p+'remaining_orphans.npy',mmap_mode='r');g=np.load('C:/Users/giaco/Downloads/NVAP_Output/cortex_7015-4_tuned/green_clean.npy',mmap_mode='r')[:len(o)];w=ndi.gaussian_filter(g,sigma=(.2,.45,.45))
for part in json.load(open(p+'remaining_orphans.json'))[:15]:
 b=tuple(slice(*r) for r in part['bounds']);m=o[b]==part['id']; a=w[b]
 stats={}
 for t in [.384765625,.3,.25,.210753]:
  d=ndi.distance_transform_edt(np.pad(m&(a>=t),1),sampling=(.4,.1315305679563492,.1315305679563492))[1:-1,1:-1,1:-1]
  stats[str(t)]=round(float(d.max()),3)
 print(part['id'],part['visible'],part['center'],stats,flush=True)
