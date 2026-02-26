from __future__ import annotations

from pathlib import Path

import pytest

from nvap.analysis.metrics import compute_metrics
from nvap.config.types import DEFAULT_SPACING, PSFConfig, RenderConfig
from nvap.export.exporters import export_metrics_csv
from nvap.io.stack_loader import load_dataset
from nvap.pipeline import apply_psf_to_dataset, default_threshold, fill_and_sync_dataset


@pytest.mark.integration
def test_pipeline_on_repo_dataset() -> None:
    root = Path("Input")
    if not root.exists():
        pytest.skip("Input directory not present.")

    dataset = load_dataset(root, spacing=DEFAULT_SPACING)
    dataset = fill_and_sync_dataset(dataset)
    processed = apply_psf_to_dataset(dataset, config=DEFAULT_PSFCFG)

    render = RenderConfig(
        threshold_green=default_threshold(processed.green.data),
        threshold_red=default_threshold(processed.red.data),
    )
    metrics = compute_metrics(processed, render)
    assert len(metrics.channel_results) == 2
    assert metrics.overlap_voxel_count >= 0


@pytest.mark.integration
def test_metrics_export_csv(tmp_path: Path) -> None:
    root = Path("Input")
    if not root.exists():
        pytest.skip("Input directory not present.")
    dataset = load_dataset(root, spacing=DEFAULT_SPACING)
    dataset = fill_and_sync_dataset(dataset)
    processed = apply_psf_to_dataset(dataset, config=DEFAULT_PSFCFG)
    render = RenderConfig(
        threshold_green=default_threshold(processed.green.data),
        threshold_red=default_threshold(processed.red.data),
    )
    metrics = compute_metrics(processed, render)
    rows = [
        {
            "channel": item.channel,
            "voxel_count": item.voxel_count,
            "volume_um3": item.volume_um3,
            "component_count": item.component_count,
            "largest_component_voxels": item.largest_component_voxels,
            "overlap_voxel_count": item.overlap_voxel_count,
            "overlap_volume_um3": item.overlap_volume_um3,
        }
        for item in metrics.channel_results
    ]
    rows.append(
        {
            "channel": "overlap",
            "voxel_count": 0,
            "volume_um3": 0.0,
            "component_count": 0,
            "largest_component_voxels": 0,
            "overlap_voxel_count": metrics.overlap_voxel_count,
            "overlap_volume_um3": metrics.overlap_volume_um3,
        }
    )
    out = export_metrics_csv(rows, tmp_path / "metrics.csv")
    assert out.exists()
    assert out.stat().st_size > 0


DEFAULT_PSFCFG = PSFConfig(
    enabled=True,
    iterations=1,
)
