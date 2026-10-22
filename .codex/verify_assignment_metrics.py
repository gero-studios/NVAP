import json
import logging
from pathlib import Path
import time

import numpy as np
from nvap.analysis.microglia_analysis import analyze_microglia_cells
from nvap.config.types import RenderConfig, VoxelSpacing

root = Path(__file__).resolve().parents[1]
source = Path('C:/Users/giaco/Downloads/NVAP_Output/cortex_7015-4_tuned')
meta = json.loads((source/'provenance.json').read_text())
green = np.load(source/'green_clean.npy', mmap_mode='r')
red = np.load(source/'red_clean.npy', mmap_mode='r')
labels = np.load(root/'.codex/cell-assignment-fixed/quality_labels.npy', mmap_mode='r')
order = np.arange(1, int(labels.max())+1, dtype=np.int32)
config = RenderConfig(threshold_green=meta['threshold_green'], threshold_red=meta['threshold_red'])
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
begin = time.perf_counter()
result = analyze_microglia_cells(green, red, labels, order, spacing=VoxelSpacing(**meta['spacing']), render=config)
report = {
    'analyzed_cells': result.analyzed_cell_count,
    'seconds': time.perf_counter()-begin,
    'cells': [{'id':c.component_id,'voxels':c.voxel_count,'soma_voxels':c.soma_voxel_count,
               'process_length_um':c.total_process_length_um,'branches':c.branch_count} for c in result.cells],
}
assert result.analyzed_cell_count == 13, report
assert all(c.voxel_count > 0 and c.soma_voxel_count > 0 for c in result.cells)
assert {c.component_id for c in result.cells} == set(order.tolist())
(root/'.codex/assignment-metrics-validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
