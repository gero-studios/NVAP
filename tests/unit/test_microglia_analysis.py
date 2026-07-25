from __future__ import annotations

import pytest
import numpy as np

import scipy.ndimage as ndi

from nvap.analysis.microglia_analysis import (
    _HAS_GEODESIC_TIPS,
    _distance_to_vessel,
    _gate_and_cluster_tips,
    _min_distance_at_coords,
    _min_positive_distance_at_coords,
    _nearest_segment_between_masks,
    _spacing_zyx,
    _voxel_tip_coords_geodesic,
    analyze_microglia_cells,
    build_microglia_cell_debug,
    microglia_analysis_to_csv_rows,
)
from nvap.config.types import RenderConfig, VoxelSpacing


def test_gate_and_cluster_tips_merges_fans_and_drops_body_endpoints() -> None:
    spacing = _spacing_zyx((1.0, 1.0, 1.0))
    dist = np.full((5, 30, 30), 0.5, dtype=np.float32)
    soma = np.zeros((5, 30, 30), dtype=bool)
    soma[1:4, 13:18, 13:18] = True
    dist[soma] = 4.0  # soma radius 4 -> tip thickness limit 1.8 um
    dist[2, 15, 15] = 3.0  # a thick endpoint sitting inside the cell body

    tips = np.array(
        [
            [2, 5, 5],  # \
            [2, 5, 6],  #  } one lamellar fan: should collapse to a single tip
            [2, 6, 5],  # /
            [2, 6, 6],  # /
            [2, 5, 25],  # a separate distal terminal (kept)
            [2, 15, 15],  # buried in thick body (gated out by thickness)
        ],
        dtype=np.int32,
    )

    out = _gate_and_cluster_tips(
        tips,
        dist_local=dist,
        soma_local=soma,
        spacing_zyx=spacing,
        branch_sensitivity=1.0,
    )

    assert out.shape[0] == 2
    assert not any(int(r[1]) == 15 and int(r[2]) == 15 for r in out.tolist())


def test_vessel_contact_distance_stays_zero_and_gap_is_separate() -> None:
    vessel = np.zeros((1, 5, 5), dtype=bool)
    vessel[0, 2, 2] = True
    dist = _distance_to_vessel(vessel, _spacing_zyx((1.0, 1.0, 0.198467)))
    coords = np.asarray([[0, 2, 2], [0, 2, 3]], dtype=np.int64)

    contact = _min_distance_at_coords(
        dist,
        coords,
        np.zeros((1, 3), dtype=np.int64),
        vessel.shape,
    )
    gap = _min_positive_distance_at_coords(
        dist,
        coords,
        np.zeros((1, 3), dtype=np.int64),
        vessel.shape,
    )

    assert contact == pytest.approx(0.0)
    assert gap == pytest.approx(0.198467)


def test_gate_and_cluster_tips_drops_endpoints_below_visibility_floor() -> None:
    spacing = _spacing_zyx((1.0, 1.0, 1.0))
    dist = np.full((3, 20, 20), 0.5, dtype=np.float32)
    soma = np.zeros((3, 20, 20), dtype=bool)
    soma[1, 1, 1] = True
    intensity = np.zeros((3, 20, 20), dtype=np.float32)
    intensity[1, 10, 5] = 0.9  # bright, clearly visible -> kept
    intensity[1, 10, 15] = 0.2  # faint halo below the visibility floor -> dropped
    tips = np.array([[1, 10, 5], [1, 10, 15]], dtype=np.int32)

    out = _gate_and_cluster_tips(
        tips,
        dist_local=dist,
        soma_local=soma,
        spacing_zyx=spacing,
        branch_sensitivity=1.0,
        intensity_local=intensity,
        intensity_floor=0.55,
    )

    assert out.shape[0] == 1
    assert int(out[0, 2]) == 5


def test_gate_and_cluster_tips_requires_component_support() -> None:
    spacing = _spacing_zyx((1.0, 1.0, 1.0))
    dist = np.full((3, 20, 20), 0.5, dtype=np.float32)
    soma = np.zeros((3, 20, 20), dtype=bool)
    soma[1, 1, 1] = True
    support = np.zeros((3, 20, 20), dtype=bool)
    support[1, 10, 5] = True

    tips = np.array(
        [
            [1, 10, 6],  # tiny graph miss: snaps back onto the real component voxel
            [1, 10, 15],  # unsupported floater: too far from any component voxel
        ],
        dtype=np.int32,
    )

    out = _gate_and_cluster_tips(
        tips,
        dist_local=dist,
        soma_local=soma,
        spacing_zyx=spacing,
        branch_sensitivity=1.0,
        support_mask=support,
    )

    assert out.tolist() == [[1, 10, 5]]
    assert support[tuple(out[0])]


def test_gate_and_cluster_tips_radius_scales_with_sensitivity() -> None:
    spacing = _spacing_zyx((1.0, 1.0, 1.0))
    dist = np.full((3, 20, 20), 0.5, dtype=np.float32)
    soma = np.zeros((3, 20, 20), dtype=bool)
    soma[1, 1, 1] = True  # negligible soma -> 1.5 um thickness floor keeps thin tips
    tips = np.array([[1, 10, 5], [1, 10, 8]], dtype=np.int32)  # 3 um apart

    low = _gate_and_cluster_tips(
        tips, dist_local=dist, soma_local=soma, spacing_zyx=spacing, branch_sensitivity=0.4
    )
    high = _gate_and_cluster_tips(
        tips, dist_local=dist, soma_local=soma, spacing_zyx=spacing, branch_sensitivity=2.0
    )

    assert low.shape[0] == 1  # large merge radius collapses the pair
    assert high.shape[0] == 2  # small merge radius keeps them distinct


@pytest.mark.skipif(not _HAS_GEODESIC_TIPS, reason="skimage.graph unavailable")
def test_voxel_tip_coords_geodesic_finds_one_tip_per_process() -> None:
    spacing = _spacing_zyx((1.0, 1.0, 1.0))
    # Soma blob with two thin processes reaching out along +x and -x; the
    # geodesic field from the soma peaks once at each distal end.
    mask = np.zeros((5, 21, 21), dtype=bool)
    mask[2, 8:13, 8:13] = True   # soma blob
    mask[2, 10, 13:19] = True    # process +x, distal end at x=18
    mask[2, 10, 2:8] = True      # process -x, distal end at x=2
    soma = np.zeros_like(mask)
    soma[2, 9:12, 9:12] = True   # soma core (subset of the blob)
    dist = ndi.distance_transform_edt(mask, sampling=(1.0, 1.0, 1.0))

    tips = _voxel_tip_coords_geodesic(
        mask,
        soma_local=soma,
        dist_local=dist,
        spacing_zyx=spacing,
        min_process_length_um=3.0,
        branch_sensitivity=1.0,
    )

    assert tips.shape[0] == 2
    assert sorted(int(t[2]) for t in tips) == [2, 18]
    assert all(mask[tuple(t)] for t in tips)  # tips always land on the voxel cloud


@pytest.mark.skipif(not _HAS_GEODESIC_TIPS, reason="skimage.graph unavailable")
def test_voxel_tip_coords_geodesic_length_threshold_gates_short_processes() -> None:
    spacing = _spacing_zyx((1.0, 1.0, 1.0))
    # A soma with one short process (~3 um geodesic from the soma core). The
    # minimum-process-length knob decides whether it survives.
    mask = np.zeros((5, 15, 15), dtype=bool)
    mask[2, 5:10, 5:10] = True
    mask[2, 7, 10:12] = True  # short stub, distal end at x=11
    soma = np.zeros_like(mask)
    soma[2, 6:9, 6:9] = True
    dist = ndi.distance_transform_edt(mask, sampling=(1.0, 1.0, 1.0))

    def _tips(min_len: float) -> np.ndarray:
        return _voxel_tip_coords_geodesic(
            mask,
            soma_local=soma,
            dist_local=dist,
            spacing_zyx=spacing,
            min_process_length_um=min_len,
            branch_sensitivity=1.0,
        )

    assert _tips(7.0).shape[0] == 0  # threshold well above the stub length -> dropped
    kept = _tips(1.0)
    assert kept.shape[0] == 1  # threshold below the stub length -> kept
    assert int(kept[0, 2]) == 11  # at the distal end


@pytest.mark.skipif(not _HAS_GEODESIC_TIPS, reason="skimage.graph unavailable")
def test_voxel_tip_coords_geodesic_keeps_short_visible_terminal_by_default() -> None:
    spacing = _spacing_zyx((1.0, 1.0, 1.0))
    mask = np.zeros((5, 15, 15), dtype=bool)
    mask[2, 5:10, 5:10] = True
    mask[2, 7, 10:12] = True
    soma = np.zeros_like(mask)
    soma[2, 6:9, 6:9] = True
    dist = ndi.distance_transform_edt(mask, sampling=(1.0, 1.0, 1.0))

    tips = _voxel_tip_coords_geodesic(
        mask,
        soma_local=soma,
        dist_local=dist,
        spacing_zyx=spacing,
        min_process_length_um=3.0,
        branch_sensitivity=1.0,
    )

    assert tips.shape[0] == 1
    assert tuple(int(v) for v in tips[0]) == (2, 7, 11)


def test_nearest_segment_between_masks_prefers_positive_distance_pair() -> None:
    spacing = _spacing_zyx((1.0, 1.0, 1.0))
    sample = np.zeros((1, 1, 6), dtype=bool)
    sample[0, 0, 2] = True
    sample[0, 0, 5] = True
    vessel = np.zeros_like(sample)
    vessel[0, 0, 0] = True
    vessel[0, 0, 5] = True

    segment = _nearest_segment_between_masks(
        sample,
        vessel,
        spacing_zyx=spacing,
        known_distance_um=2.0,
    )

    assert segment is not None
    assert tuple(int(v) for v in segment[0]) == (0, 0, 2)
    assert tuple(int(v) for v in segment[1]) == (0, 0, 0)


def test_microglia_analysis_counts_branches_and_vessel_distances() -> None:
    green = np.zeros((9, 21, 21), dtype=np.float32)
    green[3:6, 8:13, 8:13] = 1.0
    green[4, 10, 13:18] = 1.0
    green[4, 10, 3:8] = 1.0

    labels = np.zeros_like(green, dtype=np.int32)
    labels[green > 0.5] = 1

    red = np.zeros_like(green)
    red[4, 10, 2] = 1.0
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
    assert cell.soma_centroid_to_vessel_um is not None
    assert cell.soma_centroid_to_vessel_um > cell.nearest_cell_to_vessel_um
    assert cell.soma_equivalent_diameter_um > 0.0
    assert 0.0 < cell.soma_roundness <= 1.0
    assert cell.mean_branch_tortuosity >= 1.0
    assert cell.tip_near_vessel_component_count == 2
    assert cell.tips_near_multiple_vessels is True
    assert result.cells_with_tips_near_multiple_vessels == 1


def test_microglia_analysis_counts_component_owned_low_threshold_voxels() -> None:
    green = np.zeros((5, 24, 24), dtype=np.float32)
    green[2, 11:14, 11:14] = 0.8
    green[2, 12, 14:21] = 0.16

    labels = np.zeros_like(green, dtype=np.int32)
    labels[green > 0.0] = 1
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
    assert result.cells[0].voxel_count == int(np.count_nonzero(labels))


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
    assert "soma_centroid_to_vessel_um" in rows[0]
    assert "soma_roundness" in rows[0]
    assert "mean_branch_tortuosity" in rows[0]
    assert "tip_near_vessel_component_count" in rows[0]
    assert rows[0]["coordinate_reference"]
    assert rows[0]["cell_centroid_z_vox"] == pytest.approx(1.0)
    assert rows[0]["cell_centroid_y_vox"] == pytest.approx(3.0)
    assert rows[0]["cell_centroid_x_vox"] == pytest.approx(3.0)
    assert rows[0]["cell_centroid_x_um"] == pytest.approx(3.0)
    assert "tip_coordinates_zyx_vox_json" in rows[0]
    assert "tip_distances_to_vessel_um_json" in rows[0]
    assert "nearest_soma_vessel_point_x_um" in rows[0]


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
        known_soma_distance_um=1.0,
        known_cell_distance_um=1.0,
    )

    assert debug is not None
    assert debug.component_id == 1
    assert debug.voxel_sample_coords_zyx.shape[0] > 0
    assert debug.branch_sample_coords_zyx.shape[0] > 0
    assert debug.soma_sample_coords_zyx.shape[0] > 0
    assert debug.tip_coords_zyx.shape[0] == 2
    assert debug.nearest_tip_segment_zyx is not None
    assert debug.nearest_soma_segment_zyx is not None
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
