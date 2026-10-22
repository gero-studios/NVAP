import inspect
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

from nvap.analysis.microglia_components import compute_component_labels, filter_components_by_preferred_voxel_floor

root = Path(__file__).resolve().parents[1]
data = Path('C:/Users/giaco/Downloads/NVAP_Output/cortex_7015-4_tuned')
out = root / '.codex/cell-assignment-fixed'
out.mkdir(exist_ok=True)
meta = json.loads((data / 'provenance.json').read_text())
green = np.load(data / 'green_clean.npy', mmap_mode='r')
threshold = meta['threshold_green']
spacing = tuple(meta['spacing'][axis] for axis in ('z_um', 'y_um', 'x_um'))
visible_count = int(np.count_nonzero(green >= threshold))
report = {'visible_voxels': visible_count}
code = compute_component_labels.__code__
source_lines, start = inspect.getsourcelines(compute_component_labels)
raw_line = next(start + i for i, line in enumerate(source_lines) if 'raw_sizes = np.bincount' in line)

def trace(frame, event, arg):
    if frame.f_code is not code:
        return None
    values = frame.f_locals
    if event == 'line' and frame.f_lineno == raw_line:
        raw = values['raw_labels']
        visible = values['visible_roi']
        report['watershed_visible_assigned'] = int(np.count_nonzero(raw[visible]))
        report['seed_count'] = int(np.max(values['seed_labels']))
        report['low_t'] = values['low_t']
        report['high_t'] = values['high_t']
        np.save(out / 'watershed_labels.npy', raw.astype(np.uint16))
        np.save(out / 'candidate_mask.npy', values['candidate'])
        report['active_bounds'] = [[s.start, s.stop] for s in values['active_bounds']]
    if event == 'return' and 'keep_sorted' in values:
        report['quality'] = [{
            'id': i, 'voxels': int(values['raw_sizes'][i]),
            'visible_voxels': int(values['visible_counts'][i]),
            'core_voxels': int(values['core_counts'][i]),
            'kept': i in values['keep_sorted'],
        } for i in range(1, len(values['raw_sizes']))]
        report['limits'] = {key: float(values[key]) for key in ('min_keep', 'max_component_voxels', 'min_visible_ratio')}
    return trace

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
begin = time.perf_counter()
sys.settrace(trace)
labels, order, sizes = compute_component_labels(
    green, threshold=threshold, min_voxels=53, max_components=256,
    smooth_sigma=(0.2, 0.45, 0.45), branch_sensitivity=1.0, spacing=spacing,
)
sys.settrace(None)
report['quality_cells'] = len(order)
report['quality_visible_assigned'] = int(np.count_nonzero(labels[green >= threshold]))
np.save(out / 'quality_labels.npy', labels.astype(np.uint16))
labels, order, sizes = filter_components_by_preferred_voxel_floor(labels, order, sizes)
report['final_cells'] = len(order)
report['final_visible_assigned'] = int(np.count_nonzero(labels[green >= threshold]))
report['seconds'] = time.perf_counter() - begin
(out / 'report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)

