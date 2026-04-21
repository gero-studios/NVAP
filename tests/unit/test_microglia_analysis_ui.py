from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QSplitter

from nvap.analysis.microglia_vessel_report import MicrogliaCellReport, MicrogliaCellReportRow
from nvap.config.types import ChannelVolume, DatasetVolume, PreprocessConfig, PSFConfig, VoxelSpacing
from nvap.ui.main_window import MainWindow
from nvap.ui.control_panel import ControlPanel


def _sample_report() -> MicrogliaCellReport:
    return MicrogliaCellReport(
        rows=[
            MicrogliaCellReportRow(
                cell_index=1,
                component_id=1,
                segmentation_engine_used="internal",
                voxel_count=120,
                volume_um3=17.5,
                branch_endpoint_count=4,
                branch_junction_count=2,
                distance_to_vasculature_um=5.25,
                microglia_closest_x_um=1.0,
                microglia_closest_y_um=2.0,
                microglia_closest_z_um=3.0,
                vessel_closest_x_um=2.0,
                vessel_closest_y_um=2.5,
                vessel_closest_z_um=3.5,
                threshold_green_used=0.18,
                threshold_red_used=0.22,
            )
        ],
        segmentation_engine_used="internal",
        threshold_green_used=0.18,
        threshold_red_used=0.22,
    )


def _sample_dataset() -> DatasetVolume:
    spacing = VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0)
    green = np.zeros((3, 8, 8), dtype=np.float32)
    red = np.zeros_like(green)
    green[1, 2:6, 2:6] = 1.0
    red[1, 3:5, 5:7] = 1.0
    return DatasetVolume(
        green=ChannelVolume("green", green, [0, 1, 2], spacing),
        red=ChannelVolume("red", red, [0, 1, 2], spacing),
        shared_z_range=(0, 2),
    )


def test_run_microglia_analysis_button_emits_signal(qtbot) -> None:
    panel = ControlPanel()
    qtbot.addWidget(panel)
    with qtbot.waitSignal(panel.run_microglia_analysis_requested):
        qtbot.mouseClick(panel.run_microglia_analysis_btn, Qt.LeftButton)


def test_render_trim_defaults_and_updates(qtbot) -> None:
    panel = ControlPanel()
    qtbot.addWidget(panel)
    cfg = panel.current_render_config()
    assert cfg.trim_first_slices == 20
    assert cfg.trim_last_slices == 20

    panel.trim_first_slices.setValue(7)
    panel.trim_last_slices.setValue(11)
    updated = panel.current_render_config()
    assert updated.trim_first_slices == 7
    assert updated.trim_last_slices == 11


def test_main_window_render_trim_zeroes_requested_slices() -> None:
    volume = np.ones((5, 2, 2), dtype=np.float32)
    trimmed = MainWindow._apply_render_trim(volume, 1, 2)
    assert np.all(trimmed[0] == 0.0)
    assert np.all(trimmed[1:3] == 1.0)
    assert np.all(trimmed[3:] == 0.0)


def test_reprocess_button_emits_signal(qtbot) -> None:
    panel = ControlPanel()
    qtbot.addWidget(panel)
    panel.show_advanced.setChecked(True)
    with qtbot.waitSignal(panel.apply_psf_requested):
        qtbot.mouseClick(panel.reprocess_btn, Qt.LeftButton)


def test_microglia_analysis_threshold_mode_defaults_to_adaptive(qtbot) -> None:
    panel = ControlPanel()
    qtbot.addWidget(panel)
    assert panel.current_microglia_analysis_threshold_mode() == "adaptive"


def test_microglia_analysis_table_populates_and_export_enables(qtbot) -> None:
    panel = ControlPanel()
    qtbot.addWidget(panel)
    panel.set_microglia_analysis_report(_sample_report())
    assert panel.microglia_analysis_table.rowCount() == 1
    assert panel.export_microglia_analysis_btn.isEnabled()


def test_microglia_analysis_table_clear_resets_export_state(qtbot) -> None:
    panel = ControlPanel()
    qtbot.addWidget(panel)
    panel.set_microglia_analysis_report(_sample_report())
    panel.clear_microglia_analysis_table()
    assert panel.microglia_analysis_table.rowCount() == 0
    assert not panel.export_microglia_analysis_btn.isEnabled()


def test_microglia_analysis_debug_overlay_toggles_emit_state(qtbot) -> None:
    panel = ControlPanel()
    qtbot.addWidget(panel)
    assert panel.current_microglia_debug_overlay_state()["soma"]
    with qtbot.waitSignal(panel.microglia_analysis_overlay_changed):
        panel.debug_overlay_soma.setChecked(False)
    assert not panel.current_microglia_debug_overlay_state()["soma"]


def test_main_window_wraps_control_panel_in_scroll_area(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    splitter = window.centralWidget()
    assert isinstance(splitter, QSplitter)
    assert splitter.count() == 2

    scroll_area = splitter.widget(0)
    assert isinstance(scroll_area, QScrollArea)
    assert scroll_area.widget() is window.controls
    assert scroll_area.widgetResizable()


def test_main_window_cache_hit_skips_preprocess(qtbot, monkeypatch) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    dataset = _sample_dataset()
    cached = _sample_dataset()

    monkeypatch.setattr("nvap.ui.main_window.load_processed_dataset", lambda *_args, **_kwargs: cached)

    def _unexpected_preprocess(*_args, **_kwargs):
        raise AssertionError("preprocess_dataset should not run on cache hits")

    monkeypatch.setattr("nvap.ui.main_window.preprocess_dataset", _unexpected_preprocess)

    result = window._get_processed_dataset_with_cache(
        dataset,
        PSFConfig(enabled=False, iterations=0),
        PreprocessConfig(),
        dataset_signature="sig",
        cancel_event=None,
    )

    assert result is cached


def test_main_window_opacity_change_skips_metrics_refresh(qtbot, monkeypatch) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.processed_dataset = _sample_dataset()

    refresh_calls: list[str] = []
    monkeypatch.setattr(window.scene, "apply_render_config", lambda _cfg: None)
    monkeypatch.setattr(window, "_refresh_metrics", lambda: refresh_calls.append("metrics"))

    window._on_render_config_changed(replace(window.current_render, opacity_green=0.8))

    assert refresh_calls == []


def test_main_window_trim_change_reuploads_scene_channels(qtbot, monkeypatch) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    dataset = _sample_dataset()
    window.processed_dataset = dataset
    window.visual_dataset = dataset

    push_calls: list[bool] = []
    monkeypatch.setattr(window.scene, "apply_render_config", lambda _cfg: None)
    monkeypatch.setattr(window, "_refresh_metrics", lambda: None)
    monkeypatch.setattr(window, "_push_scene_channels", lambda green_only=False: push_calls.append(bool(green_only)))

    window._on_render_config_changed(replace(window.current_render, trim_first_slices=1))

    assert push_calls == [False]
