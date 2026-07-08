from __future__ import annotations

import logging
import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np

logger = logging.getLogger(__name__)


def _resolve_bundle_root() -> Path:
    configured = os.environ.get("NVAP_MICROGLIA_BUNDLE_DIR", "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))

    here = Path(__file__).resolve()
    # .../src/nvap/preprocess/microglia_masking.py -> repo root at parents[3]
    if len(here.parents) >= 4:
        candidates.append(here.parents[3] / "MicrogliaMaskingIsolated")
    candidates.append(Path.cwd() / "MicrogliaMaskingIsolated")

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Microglia masking bundle not found. "
        "Set NVAP_MICROGLIA_BUNDLE_DIR or place MicrogliaMaskingIsolated at repo/cwd. "
        f"Searched: {searched}"
    )


def _normalize_volume(
    volume: np.ndarray,
    *,
    expected_depth: int,
    expected_hw: tuple[int, int],
) -> np.ndarray:
    arr = np.asarray(volume)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        if arr.shape[0] == expected_depth:
            pass
        elif arr.shape[-1] == expected_depth:
            arr = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"Unsupported masked stack dimensions: {arr.shape}")

    if arr.ndim != 3:
        raise ValueError(f"Masked stack must be 3D after normalization, got {arr.shape}")
    if arr.shape[0] != expected_depth:
        raise ValueError(
            f"Masked depth mismatch: expected={expected_depth}, got={arr.shape[0]}"
        )
    if tuple(arr.shape[1:]) != tuple(expected_hw):
        raise ValueError(
            f"Masked XY shape mismatch: expected={expected_hw}, got={tuple(arr.shape[1:])}"
        )

    data = arr.astype(np.float32, copy=False)
    if np.issubdtype(arr.dtype, np.integer):
        denom = float(np.iinfo(arr.dtype).max)
    else:
        denom = float(np.nanmax(data))
        if not np.isfinite(denom) or denom <= 0.0:
            denom = 1.0
    return np.clip(data / denom, 0.0, 1.0).astype(np.float32, copy=False)


def _write_input_stack(input_dir: Path, volume: np.ndarray) -> None:
    arr = np.clip(np.asarray(volume, dtype=np.float32), 0.0, 1.0)
    for z in range(arr.shape[0]):
        plane = np.rint(arr[z] * 255.0).astype(np.uint8, copy=False)
        iio.imwrite(input_dir / f"slice_z{z:04d}.png", plane)


def _resolve_masked_output(output_dir: Path) -> Path:
    status_path = output_dir / "pipeline_status.txt"
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("segMaskedPath="):
                candidate = Path(line.split("=", 1)[1].strip())
                if candidate.exists():
                    return candidate

    for name in (
        "for_microglia_processed_masked.tif",
        "for_microglia_processed_masked.tiff",
        "bg_subtracted_masked.tif",
        "bg_subtracted_masked.tiff",
    ):
        candidate = output_dir / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No masked TIFF output found in {output_dir}")


