from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from nvap.analysis.microglia_vessel_report import MICROGLIA_CELL_REPORT_COLUMNS
from nvap.app import run_microglia_analysis


@pytest.mark.integration
def test_run_microglia_analysis_cli_exports_csv(tmp_path: Path) -> None:
    root = tmp_path / "Input" / "Segmented"
    green_dir = root / "Green"
    red_dir = root / "Red"
    green_dir.mkdir(parents=True)
    red_dir.mkdir(parents=True)

    for z in range(3):
        green = np.zeros((64, 64), dtype=np.uint8)
        red = np.zeros((64, 64), dtype=np.uint8)
        if z == 1:
            green[20, 8:48] = 255
            red[20, 54:60] = 255
        iio.imwrite(green_dir / f"sample_z{z:03d}c1.png", green)
        iio.imwrite(red_dir / f"sample_z{z:03d}c2.png", red)

    out_csv = tmp_path / "analysis.csv"
    code = run_microglia_analysis(root, out_csv, segmentation_mode="internal")
    assert code == 0
    assert out_csv.exists()
    header = out_csv.read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(MICROGLIA_CELL_REPORT_COLUMNS)
