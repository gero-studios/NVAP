import numpy as np
import pytest
from nvap.config.types import SomaCriteria, RenderConfig, VoxelSpacing, PSFConfig, PreprocessConfig
from nvap.analysis.contact_components import prepare_contact_components, classify_components
from nvap.analysis.microglia_analysis import analyze_microglia_cells
from nvap.ui.control_panel import ControlPanel
from nvap.ui.services.project_files import save_project_state, load_project_state


def _body():
    z, y, x = np.indices((25, 35, 35))
    return (((z-12)*.2)**2 + ((y-17)*.2)**2 + ((x-17)*.2)**2 <= 1.7**2).astype(np.float32)


def test_radius_and_core_volume_control_soma_support_and_raw_analysis():
    green = _body()
    labels, order, _ = prepare_contact_components(green, threshold=.4, spacing=(.2,)*3, require_soma=False)
    strict = SomaCriteria(required=True)
    assert len(classify_components(green, labels, order, threshold=.4, spacing=(.2,)*3, soma_criteria=strict).soma_ids) == 1
    for settings in (SomaCriteria(True, 2.5, .1), SomaCriteria(True, 1.0, 100.0)):
        classified = classify_components(green, labels, order, threshold=.4, spacing=(.2,)*3, soma_criteria=settings)
        assert not classified.cell_ids.size
        result = analyze_microglia_cells(green, np.zeros_like(green), labels, order, spacing=(.2,)*3,
            render=RenderConfig(threshold_green=.4, soma_required=True, soma_min_radius_um=settings.min_radius_um,
                                soma_min_core_volume_um3=settings.min_core_volume_um3), workers=1)
        assert result.analyzed_cell_count == 0


def test_required_soma_disables_large_interior_arbor_exception():
    green = np.zeros((17, 100, 100), np.float32)
    green[8, 47:54, 20:81] = .9
    green[8, 20:81, 47:54] = .9
    labels, order, _ = prepare_contact_components(green, threshold=.4, spacing=(.3,)*3, require_soma=False)
    assert classify_components(green, labels, order, threshold=.4, spacing=(.3,)*3).interior_ids.size == 1
    assert not classify_components(green, labels, order, threshold=.4, spacing=(.3,)*3,
                                   soma_criteria=SomaCriteria(required=True)).cell_ids.size
    kept, _, _ = prepare_contact_components(green, threshold=.4, spacing=(.3,)*3,
                                            soma_criteria=SomaCriteria(required=True))
    assert not np.any(kept)


def test_soma_controls_roundtrip_and_emit_applied_settings(qtbot, tmp_path):
    panel = ControlPanel()
    qtbot.addWidget(panel)
    settings = SomaCriteria(True, 2.2, 4.5)
    panel.set_soma_criteria(settings)
    assert panel.current_render_config().soma_criteria == settings
    with qtbot.waitSignal(panel.render_config_changed) as signal:
        panel._emit_render_config()
    assert signal.args[0].soma_criteria == settings
    save_project_state(tmp_path, channel_sources={'green':tmp_path/'green', 'red':tmp_path/'red'},
        dataset_signature='test', load_mode='folder', spacing=VoxelSpacing(), psf_config=PSFConfig(),
        preprocess_config=PreprocessConfig(), soma_criteria=settings)
    state = load_project_state(tmp_path)
    restored = SomaCriteria(**state['soma_criteria'])
    panel.set_soma_criteria(SomaCriteria())
    panel.set_soma_criteria(restored)
    assert panel.current_render_config().soma_criteria == settings


@pytest.mark.parametrize('radius,volume', [(0,.1), (-1,.1), (float('nan'),.1), (1,0), (1,float('inf'))])
def test_invalid_soma_sizes_are_rejected(radius, volume):
    with pytest.raises(ValueError):
        SomaCriteria(min_radius_um=radius, min_core_volume_um3=volume)


def test_soma_settings_invalidate_analysis_cache_and_apply_to_segmentation(qtbot, monkeypatch):
    from nvap.ui.main_window import MainWindow
    from nvap.config.types import DatasetVolume, ChannelVolume
    from dataclasses import replace
    window = MainWindow()
    qtbot.addWidget(window)
    green = _body()
    spacing = VoxelSpacing(x_um=.2, y_um=.2, z_um=.2)
    z = list(range(len(green)))
    window.visual_dataset = DatasetVolume(ChannelVolume('green', green, z, spacing),
        ChannelVolume('red', np.zeros_like(green), z, spacing), (0, len(z)-1))
    before = window._microglia_analysis_cache_key_for_render(window.current_render)
    changed = replace(window.current_render, soma_required=True, soma_min_radius_um=2.5)
    assert before != window._microglia_analysis_cache_key_for_render(changed)
    window.current_render = changed
    labels, order, _ = window._compute_microglia_components(green, .4)
    classified = window._classify_component_cells(green, labels, order, .4)
    assert not classified.cell_ids.size
