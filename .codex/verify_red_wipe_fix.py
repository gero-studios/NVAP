"""Validate the fix against the active dataset cache without changing it."""
from pathlib import Path
import json
import numpy as np
from scipy import ndimage as ndi
from nvap.config.types import VoxelSpacing
from nvap.preprocess.enhancement import wipe_vasculature_blobs
from nvap.render.vtk_scene import _downsample_volume_for_render
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkMarchingCubes, vtkPolyDataConnectivityFilter
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer, vtkRenderWindow, vtkWindowToImageFilter
from vtkmodules.vtkIOImage import vtkPNGWriter

threshold = 0.4
structure = ndi.generate_binary_structure(3, 1)
source = Path.home() / 'Downloads/.nvap_cache/processed_439f2e5978a8efd46dd48771.npz'
with np.load(source) as cached:
    red = cached['red_data']
for t, limit in [(0.35123395919799805, 2048), (0.35123395919799805, 8000), (0.4, 8000)]:
    red = wipe_vasculature_blobs(red, threshold=t, max_voxels=limit, connectivity=3)

labels, _ = ndi.label(red[10:-10] >= threshold, structure)
sizes = np.bincount(labels.ravel())
large_ids = np.flatnonzero(sizes > 8000)
large_ids = large_ids[large_ids != 0]
large_mask = np.isin(labels, large_ids)
before = red[10:-10][large_mask].copy()
fixed = wipe_vasculature_blobs(
    red, threshold=threshold, max_voxels=8000,
    trim_first_slices=10, trim_last_slices=10,
)
unchanged = np.array_equal(before, fixed[10:-10][large_mask])
fixed[:10] = 0
fixed[-10:] = 0
shown, spacing, factors = _downsample_volume_for_render(
    fixed, VoxelSpacing(x_um=0.131531, y_um=0.131531, z_um=0.4*0.7),
)
_, n = ndi.label(shown >= threshold, structure)
display_labels, _ = ndi.label(shown >= threshold, structure)
display_sizes = np.bincount(display_labels.ravel())[1:]
report = {
    'source': str(source), 'threshold': threshold, 'native_trim_each_end': 10,
    'retained_large_vessel_components': len(large_ids),
    'large_vessel_voxels_bitwise_unchanged': unchanged,
    'fixed_display_shape': list(shown.shape), 'display_downsample_factors': list(factors),
    'fixed_display_components': int(n),
    'fixed_display_components_le_64': int(np.count_nonzero(display_sizes <= 64)),
    'retained_component_sizes': sorted(display_sizes.tolist(), reverse=True),
}
image = vtkImageData()
image.SetDimensions(shown.shape[2], shown.shape[1], shown.shape[0])
image.SetSpacing(spacing.x_um, spacing.y_um, spacing.z_um)
image.GetPointData().SetScalars(numpy_to_vtk(shown.ravel(), deep=False))
marching = vtkMarchingCubes()
marching.SetInputData(image)
marching.SetValue(0, threshold)
marching.ComputeNormalsOff()
marching.Update()
regions = vtkPolyDataConnectivityFilter()
regions.SetInputConnection(marching.GetOutputPort())
regions.SetExtractionModeToAllRegions()
regions.Update()
surface_sizes = vtk_to_numpy(regions.GetRegionSizes())
report['surface_components'] = int(regions.GetNumberOfExtractedRegions())
report['surface_components_le_64_triangles'] = int(np.count_nonzero(surface_sizes <= 64))
report['largest_surface_component_triangles'] = sorted(surface_sizes.tolist(), reverse=True)[:20]
mapper = vtkPolyDataMapper()
mapper.SetInputConnection(marching.GetOutputPort())
mapper.ScalarVisibilityOff()
actor = vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(1.0, 0.14, 0.14)
actor.GetProperty().SetAmbient(0.15)
actor.GetProperty().SetDiffuse(0.82)
renderer = vtkRenderer()
renderer.SetBackground(0.025, 0.030, 0.042)
renderer.AddActor(actor)
window = vtkRenderWindow()
window.SetOffScreenRendering(1)
window.SetSize(700, 700)
window.AddRenderer(renderer)
renderer.ResetCamera()
window.Render()
capture = vtkWindowToImageFilter()
capture.SetInput(window)
capture.ReadFrontBufferOff()
capture.Update()
writer = vtkPNGWriter()
writer.SetFileName(str(Path(__file__).with_suffix('.png')))
writer.SetInputConnection(capture.GetOutputPort())
writer.Write()
window.Finalize()
assert unchanged, report
assert report['fixed_display_components_le_64'] == 0, report
out = Path(__file__).with_suffix('.json')
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
