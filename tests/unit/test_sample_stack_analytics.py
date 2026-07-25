from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nvap.analysis.metrics import compute_metrics
from nvap.analysis.microglia_analysis import analyze_microglia_cells
from nvap.analysis.neurovascular import summarize_neurovascular_association
from nvap.analysis.vascular_analysis import analyze_vasculature
from nvap.config.types import DEFAULT_SPACING, RenderConfig
from nvap.io.stack_loader import inspect_dataset_stats, load_dataset
from scripts.generate_sample_stack import expected_summary, write_sample_stack


def test_generated_sample_stack_analytics_match_expected(tmp_path: Path) -> None:
    sample_root = write_sample_stack(tmp_path / "nvap_synthetic_sample")
    expected = expected_summary()

    stats = inspect_dataset_stats(sample_root)
    assert stats.green.slice_count == expected["shape_zyx"]["z"]
    assert stats.red.slice_count == expected["shape_zyx"]["z"]
    assert stats.green.missing_count == 0
    assert stats.red.missing_count == 0

    dataset = load_dataset(sample_root, spacing=DEFAULT_SPACING)
    render = RenderConfig(
        threshold_green=0.5,
        threshold_red=0.5,
        trim_first_slices=0,
        trim_last_slices=0,
    )

    metrics = compute_metrics(dataset, render)
    by_channel = {item.channel: item for item in metrics.channel_results}

    assert by_channel["green"].voxel_count == expected["green_voxel_count"]
    assert by_channel["green"].component_count == expected["green_component_count"]
    assert by_channel["green"].largest_component_voxels == expected["green_voxel_count"]
    assert by_channel["green"].volume_um3 == pytest.approx(expected["green_volume_um3"])

    assert by_channel["red"].voxel_count == expected["red_voxel_count"]
    assert by_channel["red"].component_count == expected["red_component_count"]
    assert by_channel["red"].largest_component_voxels == 3
    assert by_channel["red"].volume_um3 == pytest.approx(expected["red_volume_um3"])

    assert metrics.overlap_voxel_count == expected["overlap_voxel_count"]
    assert metrics.overlap_volume_um3 == pytest.approx(0.0)

    labels = np.zeros(dataset.green.data.shape, dtype=np.int32)
    labels[dataset.green.data >= render.threshold_green] = 1
    microglia = analyze_microglia_cells(
        dataset.green.data,
        dataset.red.data,
        labels,
        np.asarray([1], dtype=np.int32),
        spacing=DEFAULT_SPACING,
        render=render,
    )

    assert microglia.analyzed_cell_count == expected["expected_microglia_count"]
    assert microglia.mean_branch_count == pytest.approx(expected["expected_branch_count"])
    assert microglia.mean_tip_count == pytest.approx(expected["expected_tip_count"])
    assert microglia.min_cell_to_vessel_um == pytest.approx(
        expected["expected_cell_to_vessel_um"], abs=1.0e-6
    )

    cell = microglia.cells[0]
    assert cell.voxel_count == expected["green_voxel_count"]
    assert cell.branch_count == expected["expected_branch_count"]
    assert cell.tip_count == expected["expected_tip_count"]
    assert cell.nearest_tip_to_vessel_um == pytest.approx(
        expected["expected_tip_to_vessel_um"], abs=1.0e-6
    )
    assert cell.soma_to_vessel_um is not None
    assert cell.soma_to_vessel_um > cell.nearest_cell_to_vessel_um
    assert cell.tip_near_vessel_component_count == 2
    assert cell.tips_near_multiple_vessels is True

    association = summarize_neurovascular_association(microglia)
    assert association.cell_count == 1
    assert association.cells_with_vessel == 1
    assert association.perivascular_fraction_by_radius[5.0] == pytest.approx(1.0)
    assert association.tip_leading_fraction == pytest.approx(1.0)

    vascular = analyze_vasculature(
        dataset.red.data,
        threshold=render.threshold_red,
        spacing=DEFAULT_SPACING,
        render=render,
    )
    assert vascular.red_positive_voxel_count == expected["red_voxel_count"]
    assert vascular.red_positive_volume_um3 == pytest.approx(expected["red_volume_um3"])
    assert vascular.red_positive_volume_fraction == pytest.approx(
        expected["red_voxel_count"] / float(np.prod(dataset.red.data.shape))
    )
    # The synthetic red channel contains two tiny wall markers. They
    # remain valid raw red-positive counts, but are rejected from anatomical
    # solid-vessel morphometry as specks.
    assert vascular.vessel_voxel_count == 0
    assert vascular.component_count == 0
    assert vascular.vessel_volume_um3 == pytest.approx(0.0)
