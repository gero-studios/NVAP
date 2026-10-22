import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

root = Path(__file__).resolve().parents[1]
p = root / '.codex/cell-assignment-diagnosis'
raw = np.load(p/'watershed_labels.npy', mmap_mode='r')
candidate = np.load(p/'candidate_mask.npy', mmap_mode='r')
g = np.load('C:/Users/giaco/Downloads/NVAP_Output/cortex_7015-4_tuned/green_clean.npy', mmap_mode='r')[:raw.shape[0]]
spacing = np.array([0.4,0.1315305679563492,0.1315305679563492])
filled = ndi.binary_propagation(raw > 0, structure=np.ones((3,3,3)), mask=candidate)
orphans, _ = ndi.label(candidate & ~filled, structure=np.ones((3,3,3)))
sizes = np.bincount(orphans.ravel()); sizes[0]=0
objects = ndi.find_objects(orphans)
coords = np.argwhere(raw > 0)
tree = cKDTree(coords * spacing)
owners = raw[tuple(coords.T)]
parts = []
for cid in np.argsort(sizes)[::-1][:80]:
    if sizes[cid] < 53:continue
    sl = objects[cid-1]
    local = orphans[sl] == cid
    offset = np.array([s.start for s in sl])
    pts = np.argwhere(local) + offset
    distances, indices = tree.query(pts * spacing, workers=8)
    near = distances <= 0.75
    owner_ids = owners[indices[near]]
    ids, counts = np.unique(owner_ids,return_counts=True)
    part = {'voxels':int(sizes[cid]), 'visible_voxels':int(np.count_nonzero((g[sl]>=0.384765625)&local)),
            'max_radius_um':float(ndi.distance_transform_edt(np.pad(local,1),sampling=spacing).max()),
            'nearest_gap_um':float(distances.min()),'nearby_owners':{int(i):int(n) for i,n in zip(ids,counts)}}
    parts.append(part)
print(json.dumps(parts,indent=2),flush=True)
(p/'orphan_fragments.json').write_text(json.dumps(parts,indent=2))
