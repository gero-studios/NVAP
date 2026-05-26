from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea

from nvap.config.types import ChannelVolume, DatasetVolume, PreprocessConfig, PSFConfig, VoxelSpacing
from nvap.ui.main_window import MainWindow
from nvap.ui.control_panel import ControlPanel
from nvap.ui.services.project_files import load_project_state, save_project_state


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


def test_render_trim_defaults_and_updates(qtbot) -> None:
    panel = ControlPanel()
    qtbot.addWidget(panel)
    cfg = panel.current_render_config()
    assert cfg.display_z_scale == 0.5
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


def test_main_window_wraps_control_panel_in_scroll_area(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    # controls is always wrapped in a resizable scroll area (sidebar shell
    # means centralWidget is no longer the splitter directly)
    assert isinstance(window.controls_scroll, QScrollArea)
    assert window.controls_scroll.widget() is window.controls
    assert window.controls_scroll.widgetResizable()


def test_main_window_cache_hit_skips_preprocess(qtbot, monkeypatch) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    dataset = _sample_dataset()
    cached = _sample_dataset()
    window.dataset_root = Path.cwd()

    cache_base_dirs: list[object] = []

    def _load_cache(*_args, **kwargs):
        cache_base_dirs.append(kwargs.get("base_dir"))
        return cached

    monkeypatch.setattr("nvap.ui.main_window.load_processed_dataset", _load_cache)

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
    assert cache_base_dirs == [Path.cwd()]


def test_main_window_cache_save_uses_project_root(qtbot, monkeypatch, tmp_path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    dataset = _sample_dataset()
    window.dataset_root = tmp_path

    monkeypatch.setattr("nvap.ui.main_window.load_processed_dataset", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("nvap.ui.main_window.preprocess_dataset", lambda data, _cfg: data)
    monkeypatch.setattr("nvap.ui.main_window.apply_psf_to_dataset", lambda data, *_args, **_kwargs: data)
    saved_base_dirs: list[object] = []

    def _save_cache(_key, _dataset, **kwargs):
        saved_base_dirs.append(kwargs.get("base_dir"))
        return tmp_path / ".nvap_cache" / "processed_test.npz"

    monkeypatch.setattr("nvap.ui.main_window.save_processed_dataset", _save_cache)

    result = window._get_processed_dataset_with_cache(
        dataset,
        PSFConfig(enabled=False, iterations=0),
        PreprocessConfig(),
        dataset_signature="sig",
        cancel_event=None,
    )

    assert result is dataset
    assert saved_base_dirs == [tmp_path]


def test_recent_project_opens_saved_sources_without_prompt(qtbot, monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "project"
    green = project_root / "Segmented" / "Green"
    red = project_root / "Segmented" / "Red"
    green.mkdir(parents=True)
    red.mkdir(parents=True)
    save_project_state(
        project_root,
        channel_sources={"green": green, "red": red},
        dataset_signature="abc",
        load_mode="manual",
        spacing=VoxelSpacing(),
        psf_config=PSFConfig(enabled=False, iterations=0),
        preprocess_config=PreprocessConfig(),
        cache_key="cache123",
    )

    window = MainWindow()
    qtbot.addWidget(window)
    started: list[tuple[Path, dict[str, object]]] = []

    def _start(root, **kwargs):
        started.append((root, kwargs))

    monkeypatch.setattr(window, "_start_dataset_load", _start)
    monkeypatch.setattr(window, "_prompt_load_source_mode", lambda: (_ for _ in ()).throw(AssertionError("prompted")))

    window._open_recent_project(str(project_root))

    assert started
    assert started[0][0] == project_root.resolve()
    assert started[0][1]["load_mode"] == "manual"
    assert started[0][1]["channel_overrides"] == {
        "green": str(green.resolve()),
        "red": str(red.resolve()),
    }
    assert load_project_state(project_root) is not None


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
