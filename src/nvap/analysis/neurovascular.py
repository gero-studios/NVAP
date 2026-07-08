"""Population-level neurovascular association patterns.

The per-cell microglia analysis already measures each cell's distance to the
nearest vessel. This module aggregates those distances into the spatial
*patterns* that are the actual scientific output of NVAP: how strongly the
microglia population associates with the vasculature, what fraction sits in the
perivascular niche, and how soma vs. process-tip contact differs.

Distances are micrometres. Fractions are in ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging

import numpy as np

from nvap.analysis.microglia_analysis import MicrogliaAnalysisResult

logger = logging.getLogger(__name__)

# Perivascular-niche radii (um) at which we report the cumulative fraction of
# microglia in contact. 10 um is a common operational definition of a
# perivascular/juxtavascular microglion; the wider bands describe the falloff.
_DEFAULT_PERIVASCULAR_RADII_UM = (5.0, 10.0, 20.0, 50.0)


@dataclass(frozen=True)
class NeurovascularAssociation:
    cell_count: int
    cells_with_vessel: int

    mean_cell_to_vessel_um: float
    median_cell_to_vessel_um: float
    min_cell_to_vessel_um: float
    mean_soma_to_vessel_um: float
    median_soma_to_vessel_um: float
    mean_soma_centroid_to_vessel_um: float
    median_soma_centroid_to_vessel_um: float
    mean_tip_to_vessel_um: float
    median_tip_to_vessel_um: float

    # Cumulative perivascular fractions keyed by radius (um) -> fraction in [0,1].
    perivascular_fraction_by_radius: dict[float, float] = field(default_factory=dict)

    # Fraction of cells whose nearest *tip* reaches a vessel sooner than the
    # cell body would, i.e. processes actively extend toward vasculature.
    tip_leading_fraction: float = 0.0


def _finite(values: list[float | None]) -> np.ndarray:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(float(v))], dtype=np.float64)
    return arr


def summarize_neurovascular_association(
    analysis: MicrogliaAnalysisResult,
    *,
    perivascular_radii_um: tuple[float, ...] = _DEFAULT_PERIVASCULAR_RADII_UM,
) -> NeurovascularAssociation:
    """Aggregate per-cell vessel distances into population association patterns."""
    cells = list(analysis.cells)
    cell_d = _finite([c.nearest_cell_to_vessel_um for c in cells])
    soma_d = _finite([c.soma_to_vessel_um for c in cells])
    soma_center_d = _finite([c.soma_centroid_to_vessel_um for c in cells])
    tip_d = _finite([c.nearest_tip_to_vessel_um for c in cells])

    cells_with_vessel = int(cell_d.size)

    perivascular: dict[float, float] = {}
    if cells_with_vessel > 0:
        for radius in perivascular_radii_um:
            frac = float(np.count_nonzero(cell_d <= float(radius))) / float(cells_with_vessel)
            perivascular[float(radius)] = frac

    # Tip-leading: among cells with both measurements, how often does a process
    # tip sit closer to a vessel than the cell body (processes reaching out)?
    # The reference is the SOMA, not the whole-cell minimum: the tips are a
    # subset of the cell voxels, so nearest_cell_to_vessel <= nearest_tip always
    # and comparing against it would make this fraction collapse to ~0 by
    # construction. Comparing tip-to-vessel against soma-to-vessel is what
    # actually measures processes extending ahead of the cell body.
    tip_leading = 0.0
    paired = [
        (c.nearest_tip_to_vessel_um, c.soma_to_vessel_um)
        for c in cells
        if c.nearest_tip_to_vessel_um is not None
        and c.soma_to_vessel_um is not None
        and np.isfinite(float(c.nearest_tip_to_vessel_um))
        and np.isfinite(float(c.soma_to_vessel_um))
    ]
    if paired:
        leads = sum(1 for tip, soma in paired if float(tip) <= float(soma) + 1.0e-6)
        tip_leading = float(leads) / float(len(paired))

    result = NeurovascularAssociation(
        cell_count=int(len(cells)),
        cells_with_vessel=cells_with_vessel,
        mean_cell_to_vessel_um=float(np.mean(cell_d)) if cell_d.size else 0.0,
        median_cell_to_vessel_um=float(np.median(cell_d)) if cell_d.size else 0.0,
        min_cell_to_vessel_um=float(np.min(cell_d)) if cell_d.size else 0.0,
        mean_soma_to_vessel_um=float(np.mean(soma_d)) if soma_d.size else 0.0,
        median_soma_to_vessel_um=float(np.median(soma_d)) if soma_d.size else 0.0,
        mean_soma_centroid_to_vessel_um=(
            float(np.mean(soma_center_d)) if soma_center_d.size else 0.0
        ),
        median_soma_centroid_to_vessel_um=(
            float(np.median(soma_center_d)) if soma_center_d.size else 0.0
        ),
        mean_tip_to_vessel_um=float(np.mean(tip_d)) if tip_d.size else 0.0,
        median_tip_to_vessel_um=float(np.median(tip_d)) if tip_d.size else 0.0,
        perivascular_fraction_by_radius=perivascular,
        tip_leading_fraction=tip_leading,
    )
    logger.info(
        "Neurovascular association: cells=%d with_vessel=%d mean_cell=%.2fum "
        "perivascular=%s tip_leading=%.2f",
        result.cell_count,
        result.cells_with_vessel,
        result.mean_cell_to_vessel_um,
        {k: round(v, 3) for k, v in perivascular.items()},
        result.tip_leading_fraction,
    )
    return result


def neurovascular_association_to_csv_rows(
    result: NeurovascularAssociation,
) -> list[dict[str, float | int | str]]:
    data = asdict(result)
    perivascular = data.pop("perivascular_fraction_by_radius", {})
    rows: list[dict[str, float | int | str]] = [
        {"metric": key, "value": value} for key, value in data.items()
    ]
    for radius, frac in sorted(perivascular.items()):
        rows.append({"metric": f"perivascular_fraction_within_{radius:g}um", "value": float(frac)})
    return rows
