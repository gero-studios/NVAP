import builtins
import inspect
import warnings
import numpy as np
import pytest
from skimage.restoration import rolling_ball
from nvap.preprocess.enhancement import _rolling_ball_background_fast
from nvap.preprocess.missing_slices import interpolate_missing_slices
from nvap import accelerate


@pytest.mark.parametrize('radius', [5, 18, 32])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('span', [0.0, 0.1, 1.0, 20.0])
def test_pruned_rolling_ball_is_exact_at_borders_and_across_ranges(radius, dtype, span):
    rng = np.random.default_rng(455)
    image = (rng.random((17, 23)) * span - 0.2).astype(dtype)
    key = 'workers' if 'workers' in inspect.signature(rolling_ball).parameters else 'num_threads'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        expected = rolling_ball(image, radius=radius, **{key: 1})
    actual = _rolling_ball_background_fast(image, radius=radius, workers=1)
    np.testing.assert_array_equal(actual, expected)


def test_complete_stack_reuses_storage_without_allowing_source_mutation():
    original = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
    filled, indices = interpolate_missing_slices(original, [1, 2, 3])
    assert indices == [1, 2, 3]
    assert np.shares_memory(filled, original)
    assert original.flags.writeable
    with pytest.raises(ValueError):
        filled[0, 0, 0] = 100
    assert original[0, 0, 0] == 0


def test_cpu_filters_do_not_initialize_torch(monkeypatch):
    accelerate._torch_runtime.cache_clear()
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == 'torch' or name.startswith('torch.'):
            raise AssertionError('CPU mode should not import Torch')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    assert accelerate.detect_torch_gpu_backend('cpu') is None
    assert accelerate.torch_gaussian_filter(np.ones((3, 4), dtype=np.float32), (1, 1), preferred_backend='cpu') is None
    accelerate._torch_runtime.cache_clear()


def _four_cells():
    from nvap.config.types import VoxelSpacing, RenderConfig
    green = np.zeros((13, 32, 96), dtype=np.float32)
    labels = np.zeros(green.shape, dtype=np.int32)
    for cid, x in enumerate((10, 32, 54, 76), start=1):
        labels[4:9, 13:18, x - 2:x + 3] = cid
        labels[6, 15, x + 2:x + 9] = cid
    green[labels > 0] = 0.9
    return green, np.zeros_like(green), labels, np.array([4, 1, 3, 2]), VoxelSpacing(x_um=0.8, y_um=0.8, z_um=0.8), RenderConfig(threshold_green=0.4, threshold_red=0.4, trim_first_slices=0, trim_last_slices=0)


def test_parallel_analysis_preserves_order_and_every_metric(monkeypatch):
    from nvap.analysis import microglia_analysis as module
    green, red, labels, order, spacing, render = _four_cells()
    monkeypatch.setattr(module, 'available_memory_bytes', lambda: 512 * 1024**2)
    serial = module.analyze_microglia_cells(green, red, labels, order, spacing=spacing, render=render, components_prepared=True, workers=1)
    parallel = module.analyze_microglia_cells(green, red, labels, order, spacing=spacing, render=render, components_prepared=True, workers=2)
    assert [cell.component_id for cell in parallel.cells] == order.tolist()
    assert module.microglia_analysis_to_csv_rows(serial) == module.microglia_analysis_to_csv_rows(parallel)


def test_memory_constrained_analysis_runs_cells_serially(monkeypatch):
    import threading
    from nvap.analysis import microglia_analysis as module
    green, red, labels, order, spacing, render = _four_cells()
    original = module._describe_component_shape
    seen = []

    def record(*args, **kwargs):
        seen.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'available_memory_bytes', lambda: 1)
    monkeypatch.setattr(module, '_describe_component_shape', record)
    module.analyze_microglia_cells(green, red, labels, order, spacing=spacing, render=render, components_prepared=True, workers=2)
    assert seen and set(seen) == {threading.get_ident()}


def test_prepared_analysis_reuses_validation_but_raw_callers_still_validate(monkeypatch):
    from nvap.analysis import microglia_analysis as module
    green, red, labels, order, spacing, render = _four_cells()
    expected = module.analyze_microglia_cells(green, red, labels, order, spacing=spacing, render=render, workers=1)

    def unexpected(*args, **kwargs):
        raise AssertionError('raw labels need validation')

    monkeypatch.setattr(module, 'filter_analyzable_components', unexpected)
    reused = module.analyze_microglia_cells(green, red, labels, order, spacing=spacing, render=render, components_prepared=True, workers=1)
    assert module.microglia_analysis_to_csv_rows(reused) == module.microglia_analysis_to_csv_rows(expected)
    with pytest.raises(AssertionError, match='raw labels need validation'):
        module.analyze_microglia_cells(green, red, labels, order, spacing=spacing, render=render)


def test_supported_somas_do_not_require_redundant_blob_detection(monkeypatch):
    from nvap.analysis import microglia_components as module
    zz, yy, xx = np.indices((25, 40, 40))
    volume = np.where((zz - 12)**2 + (yy - 20)**2 + (xx - 20)**2 <= 5**2, 0.9, 0.0).astype(np.float32)

    def unexpected(*args, **kwargs):
        raise AssertionError('supported soma should supply its own markers')

    monkeypatch.setattr(module, '_detect_soma_blobs', unexpected)
    labels, ids, _ = module.compute_component_labels(volume, threshold=0.4, min_voxels=20, spacing=(0.4,) * 3)
    assert len(ids) == 1
    assert labels[12, 20, 20] > 0


def test_thin_signal_retains_original_blob_detection_fallback(monkeypatch):
    from nvap.analysis import microglia_components as module
    volume = np.zeros((5, 12, 20), dtype=np.float32)
    volume[2, 6, 3:15] = 0.9
    called = []

    def fallback(*args, **kwargs):
        called.append(True)
        seed = np.zeros(volume.shape, dtype=bool)
        seed[2, 6, 8] = True
        return seed

    monkeypatch.setattr(module, '_detect_soma_blobs', fallback)
    labels, ids, _ = module.compute_component_labels(volume, threshold=0.4, min_voxels=2, spacing=(0.2,) * 3)
    assert called == [True]
    assert len(ids) == 1
    assert np.all(labels[2, 6, 3:15] > 0)


def test_native_memory_detection_works_without_psutil(monkeypatch):
    import sys
    from nvap import runtime_optimization as module
    monkeypatch.setitem(sys.modules, 'psutil', None)
    assert module._total_memory_gb() > 0
    assert module.available_memory_bytes() > 0


def test_empty_analysis_skips_vessel_processing(monkeypatch):
    from nvap.analysis import microglia_analysis as module
    green, red, labels, _, spacing, render = _four_cells()

    def unexpected(*args, **kwargs):
        raise AssertionError('empty analysis should not process vessels')

    monkeypatch.setattr(module, 'build_vascular_masks', unexpected)
    result = module.analyze_microglia_cells(green, red, labels, [], spacing=spacing, render=render)
    assert result.analyzed_cell_count == 0


def test_compact_cached_labels_preserve_all_analysis_metrics():
    from nvap.analysis import microglia_analysis as module
    green, red, labels, order, spacing, render = _four_cells()
    original = module.analyze_microglia_cells(green, red, labels, order, spacing=spacing, render=render, components_prepared=True, workers=1)
    compact = module.analyze_microglia_cells(green, red, labels.astype(np.uint8), order, spacing=spacing, render=render, components_prepared=True, workers=1)
    assert module.microglia_analysis_to_csv_rows(compact) == module.microglia_analysis_to_csv_rows(original)
