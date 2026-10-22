"""Read-only reproduction of the current red cleanup and display path."""
from pathlib import Path
import json
import numpy as np
from scipy import ndimage as ndi
from nvap.preprocess.enhancement import wipe_vasculature_blobs

structure = ndi.generate_binary_structure(3, 3)

def counts(arr, threshold):
    labels, n = ndi.label(arr >= threshold, structure)
    sizes = np.bincount(labels.ravel())[1:]
    return {
        "components": int(n),
        "components_le_64": int(np.count_nonzero(sizes <= 64)),
        "components_le_8000": int(np.count_nonzero(sizes <= 8000)),
        "largest_sizes": sorted(sizes.tolist(), reverse=True)[:20],
    }

p = Path.home() / 'Downloads/.nvap_cache/processed_439f2e5978a8efd46dd48771.npz'
with np.load(p) as f:
    red = f['red_data']
report = {"source": str(p), "native_shape": list(red.shape), "stages": {}}
for t, limit in [(0.35123395919799805, 2048), (0.35123395919799805, 8000), (0.4, 8000)]:
    red = wipe_vasculature_blobs(red, threshold=t, max_voxels=limit, connectivity=3)
    report['stages'][f'after_wipe_{t}_{limit}'] = counts(red, t)
    print(json.dumps(report['stages']), flush=True)

for t in (0.4,):
    report['native_final'] = counts(red, t)
    nearest_visual = ndi.zoom(red, (0.4 / 0.331, 1., 1.), order=0, mode='nearest', prefilter=False)
    report['nearest_z_resample'] = counts(nearest_visual, t)
    nearest_visual[:10] = 0
    nearest_visual[-10:] = 0
    z, y, x = nearest_visual.shape
    nearest_shown = nearest_visual.reshape(z, y//2, 2, x//2, 2).max(axis=(2, 4))
    report['nearest_display_trim_10'] = counts(nearest_shown, t)
    print(json.dumps({k: v for k,v in report.items() if k.startswith('nearest')}), flush=True)
    del nearest_visual, nearest_shown
    visual = ndi.zoom(red, (0.4 / 0.331, 1., 1.), order=1, mode='nearest', prefilter=False)
    report['after_z_resample'] = counts(visual, t)
    for trim in (0, 10):
        shown = visual.copy()
        if trim:
            shown[:trim] = 0
            shown[-trim:] = 0
        z, y, x = shown.shape
        shown = shown.reshape(z, y//2, 2, x//2, 2).max(axis=(2,4))
        report[f'display_trim_{trim}'] = counts(shown, t)
        report[f'display_trim_{trim}_size_only_after_cleanup'] = counts(
            wipe_vasculature_blobs(shown, threshold=t, max_voxels=8000, max_aspect_ratio=float('inf')), t
        )
out = Path(__file__).with_suffix('.json')
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
