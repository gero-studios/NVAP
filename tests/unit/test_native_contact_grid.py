import numpy as np
import scipy.ndimage as ndi
from nvap.config.types import ChannelVolume, DatasetVolume, VoxelSpacing, PreprocessConfig
from nvap.pipeline import prepare_dataset_for_view
from nvap.analysis.contact_components import prepare_contact_components


def _fragile_diagonal():
    green = np.zeros((10, 16, 16), dtype=np.float32)
    green[:, 8, 3] = .5
    green[:, 8, 12] = .5
    green[2, 8, 3:13] = .5  # Face-connected bridge confined to one native plane.
    spacing = VoxelSpacing(x_um=.131531, y_um=.131531, z_um=.4)
    z = list(range(10))
    return DatasetVolume(ChannelVolume('green', green, z, spacing),
                         ChannelVolume('red', np.zeros_like(green), z, spacing), (0, 9))


def test_view_uses_native_grid_even_when_export_mesh_resampling_is_enabled():
    data = _fragile_diagonal()
    view = prepare_dataset_for_view(data, PreprocessConfig(resample_for_mesh=True, mesh_target_z_um=.331))
    assert view is data
    assert view.green.data is data.green.data
    assert view.red.data is data.red.data
    assert view.green.spacing == data.green.spacing
    # This measured path really does break if fluorescence is interpolated
    # first. Native contact must stay connected at exactly the same threshold.
    interpolated = ndi.zoom(data.green.data, (.4 / .331, 1, 1), order=1, mode='nearest', prefilter=False)
    assert ndi.label(interpolated >= .4, np.ones((3, 3, 3)))[1] > 1
    labels, order, _ = prepare_contact_components(view.green.data, threshold=.4,
        spacing=(.4, .131531, .131531), require_soma=False)
    assert len(order) == 1
    np.testing.assert_array_equal(labels > 0, data.green.data >= .4)


def test_native_view_preserves_connectivity_for_all_export_z_targets():
    data = _fragile_diagonal()
    for target in [.1, .331, .8, 2.0]:
        view = prepare_dataset_for_view(data, PreprocessConfig(mesh_target_z_um=target))
        assert view.green.data.shape == data.green.data.shape
        assert ndi.label(view.green.data >= .4, np.ones((3, 3, 3)))[1] == 1
