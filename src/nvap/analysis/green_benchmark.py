from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from skimage.filters import meijering
from skimage.metrics import structural_similarity

from nvap.config.types import DEFAULT_SPACING, PreprocessConfig
from nvap.io.stack_loader import load_dataset
from nvap.pipeline import fill_and_sync_dataset
from nvap.preprocess.enhancement import preprocess_channel


@dataclass
class BenchmarkResult:
    name: str
    runtime_sec: float
    psnr: float
    ssim: float
    background_std: float
    branch_continuity_score: float
    small_speckle_count: int


def _pseudo_reference(raw: np.ndarray) -> np.ndarray:
    # Pseudo-reference for relative comparisons when clean ground truth is unavailable.
    return ndi.gaussian_filter(np.asarray(raw, dtype=np.float32), sigma=(0.15, 0.65, 0.65), mode="nearest")


def _psnr(pred: np.ndarray, ref: np.ndarray, max_val: float = 1.0) -> float:
    mse = float(np.mean((np.asarray(pred, dtype=np.float32) - np.asarray(ref, dtype=np.float32)) ** 2))
    if mse <= 1.0e-12:
        return 99.0
    return float(10.0 * np.log10((max_val * max_val) / mse))


def _ssim_3d(pred: np.ndarray, ref: np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=np.float32)
    ref_arr = np.asarray(ref, dtype=np.float32)
    scores: list[float] = []
    for z in range(pred_arr.shape[0]):
        scores.append(
            float(
                structural_similarity(
                    ref_arr[z],
                    pred_arr[z],
                    data_range=1.0,
                )
            )
        )
    return float(np.mean(scores)) if scores else 0.0


def _branch_continuity_score(volume: np.ndarray) -> float:
    arr = np.asarray(volume, dtype=np.float32)
    response = np.asarray(
        meijering(arr, sigmas=[0.8, 1.2, 1.8], black_ridges=False, mode="reflect"),
        dtype=np.float32,
    )
    if float(np.max(response)) <= 0.0:
        return 0.0
    mask = response >= float(np.percentile(response, 93))
    labels, count = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count <= 0:
        return 0.0
    sizes = np.bincount(labels.ravel())[1:]
    if sizes.size == 0:
        return 0.0
    return float(np.max(sizes) / max(1, int(np.sum(sizes))))


def _small_speckle_count(volume: np.ndarray) -> int:
    arr = np.asarray(volume, dtype=np.float32)
    mask = arr >= float(max(0.08, np.percentile(arr, 85)))
    labels, count = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count <= 0:
        return 0
    sizes = np.bincount(labels.ravel())
    small = np.where((sizes > 0) & (sizes < 10))[0]
    small = small[small != 0]
    return int(len(small))


def _background_std(volume: np.ndarray, raw: np.ndarray) -> float:
    arr = np.asarray(volume, dtype=np.float32)
    raw_arr = np.asarray(raw, dtype=np.float32)
    background_mask = raw_arr <= float(np.percentile(raw_arr, 20))
    if not np.any(background_mask):
        return float(np.std(arr))
    return float(np.std(arr[background_mask]))


def _evaluate_result(name: str, output: np.ndarray, raw: np.ndarray, reference: np.ndarray, runtime_sec: float) -> BenchmarkResult:
    return BenchmarkResult(
        name=name,
        runtime_sec=float(runtime_sec),
        psnr=_psnr(output, reference),
        ssim=_ssim_3d(output, reference),
        background_std=_background_std(output, raw),
        branch_continuity_score=_branch_continuity_score(output),
        small_speckle_count=_small_speckle_count(output),
    )


def run_green_denoise_benchmark(
    input_path: str | Path,
    output_path: str | Path,
    preprocess_config: PreprocessConfig,
) -> Path:
    source = Path(input_path).resolve()
    output_file = Path(output_path).resolve()

    dataset = load_dataset(source, spacing=DEFAULT_SPACING)
    dataset = fill_and_sync_dataset(dataset)
    green = dataset.green
    raw = np.asarray(green.data, dtype=np.float32)
    reference = _pseudo_reference(raw)

    baseline_cfg = replace(
        preprocess_config,
        green_denoise_strategy="legacy_anisotropic",
        preserve_branches=True,
    )
    start = time.perf_counter()
    baseline_out = preprocess_channel(green, baseline_cfg).data
    baseline_runtime = time.perf_counter() - start

    start = time.perf_counter()
    candidate_out = preprocess_channel(green, preprocess_config).data
    candidate_runtime = time.perf_counter() - start

    baseline_result = _evaluate_result(
        name="baseline_legacy_anisotropic",
        output=baseline_out,
        raw=raw,
        reference=reference,
        runtime_sec=baseline_runtime,
    )
    candidate_result = _evaluate_result(
        name=f"candidate_{preprocess_config.green_denoise_strategy}",
        output=candidate_out,
        raw=raw,
        reference=reference,
        runtime_sec=candidate_runtime,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(source),
        "baseline": asdict(baseline_result),
        "candidate": asdict(candidate_result),
        "comparison": {
            "background_std_reduction_pct": (
                100.0
                * (
                    (baseline_result.background_std - candidate_result.background_std)
                    / max(1.0e-8, baseline_result.background_std)
                )
            ),
            "branch_continuity_gain_pct": (
                100.0
                * (
                    (candidate_result.branch_continuity_score - baseline_result.branch_continuity_score)
                    / max(1.0e-8, baseline_result.branch_continuity_score)
                )
            ),
            "speckle_reduction_pct": (
                100.0
                * (
                    (baseline_result.small_speckle_count - candidate_result.small_speckle_count)
                    / max(1, baseline_result.small_speckle_count)
                )
            ),
        },
        "config": asdict(preprocess_config),
    }
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_file
