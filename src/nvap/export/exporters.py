from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def export_metrics_csv(
    rows: Iterable[dict],
    output_path: str | Path,
    *,
    columns: Sequence[str] | None = None,
) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    df = pd.DataFrame(row_list, columns=list(columns) if columns is not None else None)
    df.to_csv(path, index=False)
    logger.info("Metrics CSV written: %s (rows=%d)", path, len(df))
    return path
