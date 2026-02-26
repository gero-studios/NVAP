from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def export_metrics_csv(rows: Iterable[dict], output_path: str | Path) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    df.to_csv(path, index=False)
    logger.info("Metrics CSV written: %s (rows=%d)", path, len(df))
    return path
