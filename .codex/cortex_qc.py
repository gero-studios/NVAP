import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from PIL import Image,ImageDraw
from nvap.preprocess.enhancement import wipe_small_specks,wipe_vasculature_blobs
p=Path(r'C:\Users\giaco\Downloads\NVAP_Output\cortex_7015-4_tuned')
g=np.load(p/'green.npy'); r=np.load(p/'red.npy')
g=wipe_small_specks(g,threshold=0.384765625,min_voxels=128)
r=wipe_vasculature_blobs(r,threshold=.5,max_voxels=2048)
np.save(p/'green_clean.npy',g);np.save(p/'red_clean.npy',r)
canvas=Image.new('RGB',(1536,1100),'#16191e');d=ImageDraw.Draw(canvas)
for j,z in enumerate([16,32,48]):
    original=np.zeros((1024,1024,3),np.uint8)
    original[...,0]=np.clip(r[z]*255,0,255).astype(np.uint8);original[...,1]=np.clip(g[z]*255,0,255).astype(np.uint8)
    mask=np.zeros_like(original);mask[...,0]=(r[z]>=.5)*255;mask[...,1]=(g[z]>=.384765625)*255
    canvas.paste(Image.fromarray(original).resize((512,512)),(512*j,35))
    canvas.paste(Image.fromarray(mask).resize((512,512)),(512*j,580))
    d.text((512*j+10,10),f'Cleaned fluorescence, slice {z+1}',fill='white')
    d.text((512*j+10,555),'Selected masks: red 0.500, green 0.385',fill='white')
canvas.save(p/'slice_quality_check.png')
