from __future__ import annotations

from PySide6.QtCore import Qt

from nvap.analysis.microglia_vessel_report import MicrogliaCellReport, MicrogliaCellReportRow
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
