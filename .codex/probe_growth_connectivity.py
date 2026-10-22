import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from nvap.analysis.microglia_components import _CUBIC_STRUCTURE, _CC_STRUCTURE

root = Path(__file__).resolve().parents[1]
p = root / '.codex/cell-assignment-diagnosis'
report = json.loads((p/'report.json').read_text())
raw = np.load(p/'watershed_labels.npy', mmap_mode='r')
candidate = np.load(p/'candidate_mask.npy', mmap_mode='r')
g = np.load('C:/Users/giaco/Downloads/NVAP_Output/cortex_7015-4_tuned/green_clean.npy', mmap_mode='r')[:raw.shape[0]]
visible = g >= 0.384765625
for name, structure in [('18', _CC_STRUCTURE), ('26', _CUBIC_STRUCTURE)]:
    filled = ndi.binary_propagation(raw > 0, structure=structure, mask=candidate)
    assigned = int(np.count_nonzero(filled & visible))
    print({'connectivity':name, 'visible_assigned':assigned, 'fraction':assigned/report['visible_voxels']}, flush=True)
    if name == '26':
        missing = visible & ~filled
        labels, n = ndi.label(missing, structure=structure)
        sizes = np.bincount(labels.ravel()); sizes[0]=0
        print({'unreachable_foreground_components':n,'largest_missing':sorted(sizes.tolist(),reverse=True)[:25]},flush=True)
        np.save(p/'unreachable26.npy', missing)
