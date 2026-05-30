from __future__ import annotations

import pytest
import numpy as np

from nvap.analysis.microglia_analysis import (
    analyze_microglia_cells,
    build_microglia_cell_debug,
    microglia_analysis_to_csv_rows,
)
from nvap.config.types import RenderConfig, VoxelSpacing


def test_microglia_analysis_counts_branches_and_vessel_distances() -> None:
    green = np.zeros((9, 21, 21), dtype=np.float32)
    green[3:6, 8:13, 8:13] = 1.0
    green[4, 10, 13:18] = 1.0
    green[4, 10, 3:8] = 1.0

    labels = np.zeros_like(green, dtype=np.int32)
    labels[green > 0.5] = 1

    red = np.zeros_like(green)
    red[4, 10, 18] = 1.0

    result = analyze_microglia_cells(
        green,
        red,
        labels,
        np.asarray([1], dtype=np.int32),
        spacing=VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0),
        render=RenderConfig(
            threshold_green=0.5,
            threshold_red=0.5,
            trim_first_slices=0,
            trim_last_slices=0,
        ),
    )

    assert result.analyzed_cell_count == 1
    assert result.mean_branch_count == pytest.approx(2.0)
    assert result.min_cell_to_vessel_um == pytest.approx(1.0)

    cell = result.cells[0]
    assert cell.component_id == 1
    assert cell.branch_count == 2
    assert cell.tip_count == 2
    assert cell.soma_voxel_count > 0
    assert cell.soma_voxel_count < cell.voxel_count
    assert cell.nearest_tip_to_vessel_um == pytest.approx(1.0)
    assert cell.nearest_cell_to_vessel_um == pytest.approx(1.0)


def test_microglia_analysis_reports_branch_points_length_and_sholl() -> None:
    # Soma blob with a long trunk that forks into two arms on different axes
    # (a single 3D junction), so branch_count (segments) exceeds tip_count.
    green = np.zeros((13, 30, 30), dtype=np.float32)
    green[5:9, 12:17, 4:9] = 1.0  # soma blob
    green[6, 14, 9:20] = 1.0  # long trunk along +x
    green[6, 15:21, 19] = 1.0  # arm A along +y
    green[7:12, 14, 19] = 1.0  # arm B along +z

    labels = np.zeros_like(green, dtype=np.int32)
    labels[green > 0.5] = 1
    red = np.zeros_like(green)

    result = analyze_microglia_cells(
        green,
        red,
        labels,
        np.asarray([1], dtype=np.int32),
        spacing=VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0),
        render=RenderConfig(
            threshold_green=0.5,
            threshold_red=0.5,
            trim_first_slices=0,
            trim_last_slices=0,
        ),
    )

    assert result.analyzed_cell_count == 1
    cell = result.cells[0]
    assert cell.branch_point_count >= 1
    assert cell.tip_count >= 2
    # The forking topology gives more branch segments than terminal tips, so the
    # two metrics are genuinely distinct (no longer a copy of one another).
    assert cell.branch_count > cell.tip_count
    assert cell.total_process_length_um > 0.0
    assert cell.mean_branch_length_um > 0.0
    assert cell.sholl_max_intersections >= 1
    assert cell.sholl_enclosing_radius_um > 0.0
    assert result.mean_process_length_um == pytest.approx(cell.total_process_length_um)


def test_microglia_soma_to_vessel_distinct_from_contacting_process() -> None:
    # A soma far from the vessel with a long process that reaches it: the
    # cell-to-vessel distance collapses toward the contact point, but the
    # soma-to-vessel distance must remain large and meaningful.
    green = np.zeros((7, 13, 18), dtype=np.float32)
    green[2:5, 4:9, 2:7] = 1.0  # soma blob (low x)
    green[3, 6, 7:14] = 1.0  # long process reaching toward the vessel

    labels = np.zeros_like(green, dtype=np.int32)
    labels[green > 0.5] = 1

    red = np.zeros_like(green)
    red[3, 6, 14] = 1.0  # vessel just past the process tip

    result = analyze_microglia_cells(
        green,
        red,
        labels,
        np.asarray([1], dtype=np.int32),
        spacing=VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0),
        render=RenderConfig(
            threshold_green=0.5,
            threshold_red=0.5,
            trim_first_slices=0,
            trim_last_slices=0,
        ),
    )

    cell = result.cells[0]
    assert cell.nearest_cell_to_vessel_um == pytest.approx(1.0)
    assert cell.soma_to_vessel_um is not None
    assert cell.soma_to_vessel_um > cell.nearest_cell_to_vessel_um + 3.0
    assert result.min_soma_to_vessel_um == pytest.approx(cell.soma_to_vessel_um)


def test_microglia_analysis_applies_offsets_to_vessel_distances() -> None:
    green = np.zeros((5, 7, 9), dtype=np.float32)
    green[2, 3, 2] = 1.0
    labels = np.zeros_like(green, dtype=np.int32)
    labels[2, 3, 2] = 1

    red = np.zeros_like(green)
    red[2, 3, 6] = 1.0

    result = analyze_microglia_cells(
        green,
        red,
        labels,
        np.asarray([1], dtype=np.int32),
        spacing=VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0),
        render=RenderConfig(
            threshold_green=0.5,
            threshold_red=0.5,
            offset_x_um=3.0,
            trim_first_slices=0,
            trim_last_slices=0,
        ),
    )

    assert result.analyzed_cell_count == 1
    assert result.cells[0].nearest_cell_to_vessel_um == pytest.approx(1.0)
    assert result.cells[0].nearest_tip_to_vessel_um is None


def test_microglia_analysis_respects_trimmed_view() -> None:
    green = np.zeros((4, 6, 6), dtype=np.float32)
    green[0, 2:4, 2:4] = 1.0
    labels = np.zeros_like(green, dtype=np.int32)
    labels[0, 2:4, 2:4] = 1
    red = np.zeros_like(green)
    red[0, 2, 5] = 1.0

    result = analyze_microglia_cells(
        green,
        red,
        labels,
        np.asarray([1], dtype=np.int32),
        spacing=VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0),
        render=RenderConfig(
            threshold_green=0.5,
            threshold_red=0.5,
            trim_first_slices=1,
        ),
    )

    assert result.analyzed_cell_count == 0
    assert result.cells == []


def test_microglia_analysis_csv_rows_follow_cell_metrics() -> None:
    green = np.zeros((3, 6, 6), dtype=np.float32)
    green[1, 2:5, 2:5] = 1.0
    labels = np.zeros_like(green, dtype=np.int32)
    labels[green > 0.5] = 1
    red = np.zeros_like(green)

    result = analyze_microglia_cells(
        green,
        red,
        labels,
        np.asarray([1], dtype=np.int32),
        spacing=VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0),
        render=RenderConfig(
            threshold_green=0.5,
            threshold_red=0.5,
            trim_first_slices=0,
            trim_last_slices=0,
        ),
    )

    rows = microglia_analysis_to_csv_rows(result)

    assert len(rows) == 1
    assert rows[0]["component_id"] == 1
    assert rows[0]["voxel_count"] == result.cells[0].voxel_count


def test_microglia_cell_debug_returns_soma_tips_and_vessel_segments() -> None:
    green = np.zeros((9, 21, 21), dtype=np.float32)
    green[3:6, 8:13, 8:13] = 1.0
    green[4, 10, 13:18] = 1.0
    green[4, 10, 3:8] = 1.0
    labels = np.zeros_like(green, dtype=np.int32)
    labels[green > 0.5] = 1
    red = np.zeros_like(green)
    red[4, 10, 18] = 1.0

    debug = build_microglia_cell_debug(
        green,
        red,
        labels,
        1,
        spacing=VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0),
        render=RenderConfig(
            threshold_green=0.5,
            threshold_red=0.5,
            trim_first_slices=0,
            trim_last_slices=0,
        ),
        known_tip_distance_um=1.0,
        known_cell_distance_um=1.0,
    )

    assert debug is not None
    assert debug.component_id == 1
    assert debug.voxel_sample_coords_zyx.shape[0] > 0
    assert debug.branch_sample_coords_zyx.shape[0] > 0
    assert debug.soma_sample_coords_zyx.shape[0] > 0
    assert debug.tip_coords_zyx.shape[0] == 2
    assert debug.nearest_tip_segment_zyx is not None
    assert debug.nearest_cell_segment_zyx is not None


def test_microglia_cell_debug_uses_render_offset_for_distance_segments() -> None:
    green = np.zeros((5, 7, 12), dtype=np.float32)
    green[2, 3, 0:11] = 1.0
    labels = np.zeros_like(green, dtype=np.int32)
    labels[green > 0.5] = 1

    red = np.zeros_like(green)
    red[2, 3, 4] = 1.0

    render = RenderConfig(
        threshold_green=0.5,
        threshold_red=0.5,
        offset_x_um=-10.0,
        trim_first_slices=0,
        trim_last_slices=0,
    )
    spacing = VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0)

    analysis = analyze_microglia_cells(
        green,
        red,
        labels,
        np.asarray([1], dtype=np.int32),
        spacing=spacing,
        render=render,
    )

    debug = build_microglia_cell_debug(
        green,
        red,
        labels,
        1,
        spacing=spacing,
        render=render,
        known_cell_distance_um=analysis.cells[0].nearest_cell_to_vessel_um,
    )

    assert debug is not None
    assert debug.nearest_cell_segment_zyx is not None
    assert int(debug.nearest_cell_segment_zyx[0, 2]) == 10

    source = np.asarray(debug.nearest_cell_segment_zyx[0], dtype=np.float32)
    target = np.asarray(debug.nearest_cell_segment_zyx[1], dtype=np.float32)
    source_world = np.asarray(
        [
            (source[2] + 0.5) + float(render.offset_x_um),
            source[1] + 0.5,
            source[0] + 0.5,
        ],
        dtype=np.float32,
    )
    target_world = np.asarray(
        [target[2] + 0.5, target[1] + 0.5, target[0] + 0.5],
        dtype=np.float32,
    )
    world_distance = float(np.linalg.norm(target_world - source_world))

    assert analysis.cells[0].nearest_cell_to_vessel_um == pytest.approx(4.0)
    assert world_distance == pytest.approx(analysis.cells[0].nearest_cell_to_vessel_um)