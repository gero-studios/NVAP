"""Validate neurovascular association aggregation."""

from __future__ import annotations

from nvap.analysis.microglia_analysis import MicrogliaAnalysisResult, MicrogliaCellAnalysis
from nvap.analysis.neurovascular import (
    neurovascular_association_to_csv_rows,
    summarize_neurovascular_association,
)


def _cell(cid: int, cell_d: float | None, soma_d: float | None, tip_d: float | None):
    return MicrogliaCellAnalysis(
        component_id=cid,
        voxel_count=100,
        volume_um3=10.0,
        soma_voxel_count=20,
        soma_volume_um3=2.0,
        branch_count=5,
        tip_count=6,
        branch_point_count=3,
        total_process_length_um=40.0,
        mean_branch_length_um=8.0,
        sholl_max_intersections=4,
        sholl_critical_radius_um=12.0,
        sholl_enclosing_radius_um=30.0,
        nearest_tip_to_vessel_um=tip_d,
        nearest_cell_to_vessel_um=cell_d,
        soma_to_vessel_um=soma_d,
        soma_centroid_to_vessel_um=None if soma_d is None else soma_d + 1.0,
    )


def test_perivascular_fractions_and_tip_leading():
    cells = [
        _cell(1, 2.0, 6.0, 1.0),   # within 5um, tip leads
        _cell(2, 8.0, 9.0, 12.0),  # within 10um, soma leads
        _cell(3, 30.0, 32.0, 28.0),  # within 50um, tip leads
        _cell(4, None, None, None),  # no vessel measured -> excluded
    ]
    analysis = MicrogliaAnalysisResult(cells=cells, analyzed_cell_count=len(cells))

    assoc = summarize_neurovascular_association(analysis)

    assert assoc.cell_count == 4
    assert assoc.cells_with_vessel == 3
    # 1 of 3 within 5um; 2 of 3 within 10um; 3 of 3 within 50um.
    fr = assoc.perivascular_fraction_by_radius
    assert fr[5.0] == 1 / 3
    assert fr[10.0] == 2 / 3
    assert fr[50.0] == 1.0
    assert assoc.min_cell_to_vessel_um == 2.0
    assert assoc.mean_soma_centroid_to_vessel_um > assoc.mean_soma_to_vessel_um
    # Tip-leading compares tip vs soma: cells 1 (1<=6) and 3 (28<=32) lead,
    # cell 2 (12>9) does not -> 2/3.
    assert abs(assoc.tip_leading_fraction - 2 / 3) < 1e-9


def test_tip_leading_uses_soma_not_whole_cell_minimum():
    # nearest_cell_to_vessel_um is the minimum over ALL cell voxels (the tips are
    # a subset of them), so comparing the tip against it would collapse this
    # fraction to ~0 by construction. The reference must be the soma: here the tip
    # (5) sits between the whole-cell minimum (1) and the soma (10), so it leads
    # the soma even though it is farther than the cell minimum.
    cells = [_cell(1, cell_d=1.0, soma_d=10.0, tip_d=5.0)]
    analysis = MicrogliaAnalysisResult(cells=cells, analyzed_cell_count=len(cells))

    assoc = summarize_neurovascular_association(analysis)

    assert assoc.tip_leading_fraction == 1.0


def test_empty_analysis_is_safe():
    assoc = summarize_neurovascular_association(MicrogliaAnalysisResult())
    assert assoc.cell_count == 0
    assert assoc.cells_with_vessel == 0
    rows = neurovascular_association_to_csv_rows(assoc)
    assert any(r["metric"] == "mean_cell_to_vessel_um" for r in rows)
