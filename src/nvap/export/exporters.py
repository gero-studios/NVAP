from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable, Sequence

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
    if columns is not None:
        fieldnames = list(columns)
    else:
        fieldnames = []
        seen: set[str] = set()
        for row in row_list:
            for key in row.keys():
                name = str(key)
                if name not in seen:
                    seen.add(name)
                    fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row_list)
    logger.info("Metrics CSV written: %s (rows=%d)", path, len(row_list))
    return path
