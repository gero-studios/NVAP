from pathlib import Path
import numpy as np
from PIL import Image
p=Path('C:/Users/giaco/Downloads/NVAP_Output/cortex_7015-4_tuned');out=Path('website/assets')
g=np.load(p/'green_clean.npy',mmap_mode='r');r=np.load(p/'red_clean.npy',mmap_mode='r');f=np.load(p/'analysis_masks.npz');labels=f['microglia_labels'];vessel=f['vascular_solid_mask'].max(axis=0)>0
bg=np.array([11,16,17],dtype=float)
def save(name,rgb):Image.fromarray(np.clip(rgb,0,255).astype('uint8')).save(out/name,optimize=True)
gp=np.max(g,axis=0);rp=np.max(r,axis=0)
raw=bg+gp[...,None]*np.array([87,178,120])+rp[...,None]*np.array([190,64,71]);save('segmentation-channels.png',raw)
ids,counts=np.unique(labels,return_counts=True);pairs=[(int(n),int(c)) for n,c in zip(ids,counts) if n>0];selected=max(pairs,key=lambda item:item[1])[0]
projected=labels.max(axis=0);rgb=np.broadcast_to(bg,(*projected.shape,3)).copy();rgb[vessel]=[133,77,83]
palette=[[160,200,163],[163,189,213],[201,184,142],[174,163,204],[132,195,186],[197,157,170]]
for i,(cid,count) in enumerate(pairs):rgb[projected==cid]=palette[i%len(palette)]
save('segmentation-cells.png',rgb)
isolated=(labels==selected).max(axis=0);rgb=np.broadcast_to(bg,(*projected.shape,3)).copy();rgb[vessel]=[71,43,48];rgb[isolated]=[185,217,178];save('segmentation-isolated.png',rgb)
print('Saved three projections from the same cortex volume; selected cell',selected,'of',len(pairs))
