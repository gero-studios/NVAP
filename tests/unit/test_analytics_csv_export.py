from __future__ import annotations

import csv
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nvap.analysis.metrics import compute_metrics
from nvap.config.types import ChannelVolume, DatasetVolume, VoxelSpacing
from nvap.export.exporters import export_metrics_csv
from nvap.ui.main_window import MainWindow


@pytest.fixture
def export_window(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    spacing = VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0)
    green = np.zeros((5, 24, 24), dtype=np.float32)
    green[1:4, 5:15, 5:15] = 1.0
    red = np.zeros_like(green)
    red[:, 11:15, 18:21] = 1.0
    dataset = DatasetVolume(
        ChannelVolume("green", green, list(range(5)), spacing),
        ChannelVolume("red", red, list(range(5)), spacing),
        (0, 4),
    )
    window.processed_dataset = dataset
    window.visual_dataset = dataset
    window.current_render = replace(window.current_render, trim_first_slices=0, trim_last_slices=0)
    return window


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("stale_summary", [False, True])
def test_csv_export_computes_current_results_off_ui_thread(
    export_window, qtbot, monkeypatch, tmp_path, stale_summary,
):
    window = export_window
    if stale_summary:
        window.latest_metrics = compute_metrics(
            window.processed_dataset, replace(window.current_render, threshold_green=1.1),
        )
    destination = tmp_path / "analysis.csv"
    errors = []
    calls = []
    ui_thread = threading.get_ident()

    def record_compute(dataset, render):
        calls.append(threading.get_ident())
        return compute_metrics(dataset, render)

    monkeypatch.setattr("nvap.ui.main_window.compute_metrics", record_compute)
    monkeypatch.setattr("nvap.ui.main_window.QFileDialog.getSaveFileName", lambda *a: (str(destination), ""))
    monkeypatch.setattr(window, "_show_error", lambda *args: errors.append(args))

    window._on_export_metrics_requested()
    qtbot.waitUntil(lambda: window._active_thread is None, timeout=30000)

    assert not errors
    assert calls and all(thread != ui_thread for thread in calls)
    rows = {row["channel"]: row for row in read_rows(destination)}
    assert int(rows["green"]["voxel_count"]) == 300
    assert int(rows["red"]["voxel_count"]) == 60
    assert read_rows(tmp_path / "analysis_vascular.csv")
    cells = read_rows(tmp_path / "analysis_microglia.csv")
    assert cells and int(cells[0]["component_id"]) == 1
    assert read_rows(tmp_path / "analysis_neurovascular.csv")
    provenance = {row["setting"]: row["value"] for row in read_rows(tmp_path / "analysis_provenance.csv")}
    assert provenance["microglia_cell_count"] == "1"
    with np.load(tmp_path / "analysis_vascular_masks.npz") as masks:
        assert masks["vascular_wall_mask"].shape == (5, 24, 24)


def test_analytics_page_exports_single_dataset_without_project_set(export_window, monkeypatch, tmp_path):
    window = export_window
    tasks = {}
    monkeypatch.setattr(window, "_start_background_task", lambda **kw: tasks.update(kw))
    monkeypatch.setattr("nvap.ui.main_window.QFileDialog.getSaveFileName", lambda *a: (str(tmp_path / "single.csv"), ""))

    window._refresh_section_state()
    assert window._analytics_export_btn.isEnabled()
    assert not window._project_analytics_btn.isEnabled()
    window._analytics_export_btn.click()

    assert callable(tasks["fn"])
    assert tasks["title"] == "Export Analytics CSV"


def test_export_uses_captured_settings_even_if_ui_changes(export_window, monkeypatch, tmp_path):
    window = export_window
    tasks = {}
    destination = tmp_path / "snapshot.csv"
    original_threshold = window.current_render.threshold_green
    monkeypatch.setattr(window, "_start_background_task", lambda **kw: tasks.update(kw))
    monkeypatch.setattr("nvap.ui.main_window.QFileDialog.getSaveFileName", lambda *a: (str(destination), ""))

    window._on_export_metrics_requested()
    window.current_render = replace(window.current_render, threshold_green=1.1)

    def fail_ui_read():
        raise AssertionError("Worker read live UI state")

    monkeypatch.setattr(window, "_provenance_rows", fail_ui_read)
    monkeypatch.setattr(window.controls, "current_microglia_branch_sensitivity", fail_ui_read)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(tasks["fn"]).result(timeout=20)

    assert int(read_rows(destination)[0]["voxel_count"]) == 300
    provenance = {row["setting"]: row["value"] for row in read_rows(tmp_path / "snapshot_provenance.csv")}
    assert float(provenance["threshold_green"]) == original_threshold


@pytest.mark.parametrize("fail_companion", [False, True])
def test_export_failure_is_reported_instead_of_claiming_success(
    export_window, qtbot, monkeypatch, tmp_path, fail_companion,
):
    window = export_window
    errors = []
    monkeypatch.setattr("nvap.ui.main_window.QFileDialog.getSaveFileName", lambda *a: (str(tmp_path / "locked.csv"), ""))
    monkeypatch.setattr(window, "_show_error", lambda *args: errors.append(args))

    def fail_write(*args, **kwargs):
        if not fail_companion or Path(args[1]).name.endswith("_microglia.csv"):
            raise PermissionError("File is open in Excel")
        return export_metrics_csv(*args, **kwargs)

    monkeypatch.setattr("nvap.ui.main_window.export_metrics_csv", fail_write)
    window._on_export_metrics_requested()
    qtbot.waitUntil(lambda: window._active_thread is None, timeout=10000)

    assert errors and errors[0][0] == "Analytics export failed"
    assert "File is open in Excel" in errors[0][1]
    assert (tmp_path / "locked.csv").exists() == fail_companion
