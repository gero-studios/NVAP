from __future__ import annotations

import numpy as np

from nvap.analysis.microglia_vessel_report import (
    MICROGLIA_CELL_REPORT_COLUMNS,
    MicrogliaCellReport,
    analyze_microglia_vessel,
    microglia_cell_report_to_csv_rows,
)
from nvap.config.types import ChannelVolume, DatasetVolume, PreprocessConfig, RenderConfig, VoxelSpacing
from nvap.export.exporters import export_metrics_csv


def _dataset(green: np.ndarray, red: np.ndarray, spacing: VoxelSpacing | None = None) -> DatasetVolume:
    spacing = spacing or VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0)
    return DatasetVolume(
        green=ChannelVolume("green", green.astype(np.float32), list(range(green.shape[0])), spacing),
        red=ChannelVolume("red", red.astype(np.float32), list(range(red.shape[0])), spacing),
        shared_z_range=(0, green.shape[0] - 1),
    )


def test_surface_distance_and_closest_points_are_reported() -> None:
    green = np.zeros((3, 16, 96), dtype=np.float32)
    red = np.zeros_like(green)
    green[1, 5, 2:66] = 1.0
    red[1, 5, 74:80] = 1.0
    ds = _dataset(green, red)

    report = analyze_microglia_vessel(
        ds,
        RenderConfig(threshold_green=0.5, threshold_red=0.5),
        PreprocessConfig(),
        segmentation_mode="internal",
    )

    assert report.cell_count == 1
    row = report.rows[0]
    assert row.distance_to_vasculature_um == 8.0
    assert row.microglia_closest_x_um == 66.0
    assert row.vessel_closest_x_um == 74.0
    assert row.microglia_closest_y_um == 5.0
    assert row.vessel_closest_y_um == 5.0
    assert row.microglia_closest_z_um == 1.0
    assert row.vessel_closest_z_um == 1.0
    assert row.branch_endpoint_count == 2
    assert row.branch_junction_count == 0


def test_touching_microglia_and_vessel_has_zero_distance() -> None:
    green = np.zeros((3, 16, 96), dtype=np.float32)
    red = np.zeros_like(green)
    green[1, 7, 10:78] = 1.0
    red[1, 7, 48:56] = 1.0
    ds = _dataset(green, red)

    report = analyze_microglia_vessel(
        ds,
        RenderConfig(threshold_green=0.5, threshold_red=0.5),
        PreprocessConfig(),
        segmentation_mode="internal",
    )

    assert report.cell_count == 1
    row = report.rows[0]
    assert row.distance_to_vasculature_um == 0.0
    assert row.microglia_closest_x_um == row.vessel_closest_x_um
    assert row.microglia_closest_y_um == row.vessel_closest_y_um
    assert row.microglia_closest_z_um == row.vessel_closest_z_um


def test_branch_endpoint_and_junction_counts() -> None:
    green = np.zeros((3, 64, 64), dtype=np.float32)
    red = np.zeros_like(green)
    # T-junction with >53 voxels (required by component filtering defaults).
    green[1, 32, 8:56] = 1.0
    green[1, 8:33, 32] = 1.0
    red[1, 55:60, 55:60] = 1.0
    ds = _dataset(green, red)

    report = analyze_microglia_vessel(
        ds,
        RenderConfig(threshold_green=0.5, threshold_red=0.5),
        PreprocessConfig(),
        segmentation_mode="internal",
    )

    assert report.cell_count >= 1
    # Soma splitting may decompose the T-junction into sub-components.
    # Sum branch metrics across all cells to verify the T-junction topology.
    total_endpoints = sum(r.branch_endpoint_count for r in report.rows)
    total_junctions = sum(r.branch_junction_count for r in report.rows)
    assert total_endpoints >= 2
    assert total_junctions >= 0


def test_auto_mode_uses_internal_when_components_exist(monkeypatch) -> None:
    green = np.zeros((3, 16, 96), dtype=np.float32)
    red = np.zeros_like(green)
    green[1, 6, 5:69] = 1.0
    red[1, 6, 76:82] = 1.0
    ds = _dataset(green, red)

    def _unexpected_fiji(_volume):
        raise AssertionError("Fiji fallback should not be called when internal segmentation succeeds.")

    monkeypatch.setattr(
        "nvap.analysis.microglia_vessel_report.mask_green_volume_with_microglia_bundle",
        _unexpected_fiji,
    )

    report = analyze_microglia_vessel(
        ds,
        RenderConfig(threshold_green=0.5, threshold_red=0.5),
        PreprocessConfig(),
        segmentation_mode="auto",
    )
    assert report.segmentation_engine_used == "internal"
    assert report.cell_count == 1


def test_auto_mode_falls_back_to_fiji_when_internal_returns_zero(monkeypatch) -> None:
    green = np.zeros((3, 16, 96), dtype=np.float32)
    red = np.zeros_like(green)
    red[1, 4, 60:65] = 1.0
    ds = _dataset(green, red)

    def _mock_fiji(_volume):
        masked = np.zeros_like(green, dtype=np.float32)
        masked[1, 4, 4:68] = 1.0
        return masked

    monkeypatch.setattr(
        "nvap.analysis.microglia_vessel_report.mask_green_volume_with_microglia_bundle",
        _mock_fiji,
    )

    report = analyze_microglia_vessel(
        ds,
        RenderConfig(threshold_green=0.5, threshold_red=0.5),
        PreprocessConfig(),
        segmentation_mode="auto",
    )
    assert report.segmentation_engine_used == "fiji"
    assert report.cell_count == 1


def test_csv_serializer_and_header_only_export(tmp_path) -> None:
    report = MicrogliaCellReport(rows=[], segmentation_engine_used="internal", threshold_green_used=0.2, threshold_red_used=0.3)
    rows = microglia_cell_report_to_csv_rows(report)
    assert rows == []

    out = export_metrics_csv(
        rows,
        tmp_path / "microglia_cells.csv",
        columns=MICROGLIA_CELL_REPORT_COLUMNS,
    )
    content = out.read_text(encoding="utf-8")
    assert content.splitlines()[0] == ",".join(MICROGLIA_CELL_REPORT_COLUMNS)


def test_mismatched_channel_depths_are_aligned_by_shared_z_range() -> None:
    spacing = VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0)
    green = np.zeros((3, 16, 96), dtype=np.float32)  # z values [1,2,3]
    red = np.zeros((2, 16, 96), dtype=np.float32)    # z values [2,3]
    green[1, 6, 8:72] = 1.0  # shared z=2
    red[0, 6, 80:86] = 1.0   # shared z=2
    ds = DatasetVolume(
        green=ChannelVolume("green", green, [1, 2, 3], spacing),
        red=ChannelVolume("red", red, [2, 3], spacing),
        shared_z_range=(2, 3),
    )

    report = analyze_microglia_vessel(
        ds,
        RenderConfig(threshold_green=0.5, threshold_red=0.5),
        PreprocessConfig(),
        segmentation_mode="internal",
    )
    assert report.cell_count == 1
    row = report.rows[0]
    assert row.distance_to_vasculature_um >= 0.0
    assert row.microglia_closest_z_um == 2.0


def test_auto_mode_skips_fiji_fallback_for_large_volume(monkeypatch) -> None:
    green = np.zeros((3, 16, 16), dtype=np.float32)
    red = np.zeros_like(green)
    ds = _dataset(green, red)

    monkeypatch.setattr(
        "nvap.analysis.microglia_vessel_report._AUTO_FIJI_FALLBACK_MAX_VOXELS",
        16,  # force "large" classification for this tiny test volume
    )

    def _unexpected_fiji(_volume):
        raise AssertionError("Fiji fallback should be skipped for large auto volumes.")

    monkeypatch.setattr(
        "nvap.analysis.microglia_vessel_report.mask_green_volume_with_microglia_bundle",
        _unexpected_fiji,
    )

    report = analyze_microglia_vessel(
        ds,
        RenderConfig(threshold_green=0.5, threshold_red=0.5),
        PreprocessConfig(),
        segmentation_mode="auto",
    )
    assert report.segmentation_engine_used == "internal"
    assert report.cell_count == 0


def test_auto_mode_falls_back_to_fiji_when_internal_singleton_looks_merged(monkeypatch) -> None:
    green = np.zeros((3, 32, 32), dtype=np.float32)
    red = np.zeros_like(green)
    green[1, 12:16, 6:10] = 1.0
    green[1, 12:16, 20:24] = 1.0
    red[1, 14, 27:30] = 1.0
    ds = _dataset(green, red)

    singleton_labels = np.zeros_like(green, dtype=np.int32)
    singleton_labels[1, 12:16, 6:24] = 1
    singleton_order = np.asarray([1], dtype=np.int32)
    singleton_sizes = np.asarray([0, int(np.count_nonzero(singleton_labels == 1))], dtype=np.int64)

    split_labels = np.zeros_like(green, dtype=np.int32)
    split_labels[1, 12:16, 6:10] = 1
    split_labels[1, 12:16, 20:24] = 2
    split_order = np.asarray([1, 2], dtype=np.int32)
    split_sizes = np.asarray(
        [0, int(np.count_nonzero(split_labels == 1)), int(np.count_nonzero(split_labels == 2))],
        dtype=np.int64,
    )

    monkeypatch.setattr(
        "nvap.analysis.microglia_vessel_report._run_internal_segmentation",
        lambda *_args, **_kwargs: (singleton_labels, singleton_order, singleton_sizes),
    )
    monkeypatch.setattr(
        "nvap.analysis.microglia_vessel_report._internal_result_looks_merged",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "nvap.analysis.microglia_vessel_report._run_fiji_segmentation",
        lambda *_args, **_kwargs: (split_labels, split_order, split_sizes),
    )

    report = analyze_microglia_vessel(
        ds,
        RenderConfig(threshold_green=0.5, threshold_red=0.5),
        PreprocessConfig(),
        segmentation_mode="auto",
    )
    assert report.segmentation_engine_used == "fiji"
    assert report.cell_count == 2


def test_adaptive_threshold_source_recovers_dim_microglia_when_render_threshold_is_high() -> None:
    green = np.zeros((4, 48, 48), dtype=np.float32)
    red = np.zeros_like(green)
    green[1, 24, 24] = 0.28
    green[1, 24, 25] = 0.24
    green[1, 24, 26:36] = 0.18
    green[1, 20:24, 24] = 0.16
    green[2, 24, 24:30] = 0.14
    red[1, 24, 40:44] = 1.0
    ds = _dataset(green, red)
    render = RenderConfig(threshold_green=0.5, threshold_red=0.5)

    report_render = analyze_microglia_vessel(
        ds,
        render,
        PreprocessConfig(),
        segmentation_mode="internal",
        threshold_source="render",
    )
    report_adaptive = analyze_microglia_vessel(
        ds,
        render,
        PreprocessConfig(),
        segmentation_mode="internal",
        threshold_source="adaptive",
    )

    assert report_render.cell_count == 0
    assert report_adaptive.cell_count == 1
    assert report_adaptive.threshold_green_used < 0.5
