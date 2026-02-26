from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import logging
import os
import threading
import time
from typing import Callable

import numpy as np
from skimage.filters import threshold_otsu

from nvap.config.types import ChannelVolume, DatasetVolume, PSFConfig, PreprocessConfig
from nvap.preprocess.enhancement import (
    postprocess_green_after_deconvolution,
    preprocess_dataset,
    suggest_green_threshold,
)
from nvap.preprocess.missing_slices import fill_channel_missing_slices
from nvap.preprocess.psf import deconvolve_volume
from nvap.preprocess.resample import prepare_mesh_dataset

logger = logging.getLogger(__name__)


def _resolve_psf_channel_workers() -> int:
    raw = os.environ.get("NVAP_PSF_CHANNEL_WORKERS", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            logger.warning("Invalid NVAP_PSF_CHANNEL_WORKERS=%r. Falling back to auto.", raw)
    cpus = os.cpu_count() or 1
    # Two channels at most; keep default conservative.
    return 2 if cpus >= 2 else 1


def _shared_range(green: ChannelVolume, red: ChannelVolume) -> tuple[int, int]:
    start = max(min(green.z_indices), min(red.z_indices))
    end = min(max(green.z_indices), max(red.z_indices))
    if start > end:
        raise ValueError("No overlapping z range between channels.")
    return start, end


def fill_and_sync_dataset(dataset: DatasetVolume) -> DatasetVolume:
    logger.info("Filling missing slices and syncing channel z-ranges.")
    green = fill_channel_missing_slices(dataset.green)
    red = fill_channel_missing_slices(dataset.red)
    shared = _shared_range(green, red)
    logger.info(
        "Synced dataset: green_z=%d red_z=%d shared_range=%s",
        len(green.z_indices),
        len(red.z_indices),
        shared,
    )
    return DatasetVolume(green=green, red=red, shared_z_range=shared)


def apply_psf_to_dataset(
    dataset: DatasetVolume,
    config: PSFConfig,
    preprocess_config: PreprocessConfig | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> DatasetVolume:
    logger.info("Applying PSF pipeline to both channels.")
    t0 = time.perf_counter()
    channel_workers = _resolve_psf_channel_workers()
    effective_config = config
    if (
        preprocess_config is not None
        and preprocess_config.green_denoise_strategy == "pixel2voxel_no_psf"
        and (config.enabled or config.iterations > 0)
    ):
        logger.info("PSF bypass enabled by preprocess strategy=pixel2voxel_no_psf.")
        effective_config = replace(config, enabled=False, iterations=0)

    progress_seen: dict[str, int] = {"green": 0, "red": 0}
    progress_lock = threading.Lock()

    def _progress(channel: str, current: int, total: int) -> None:
        if progress_callback is not None:
            progress_callback(channel, current, total)
        if total <= 0:
            return
        percent = int((100 * current) / total)
        with progress_lock:
            last = progress_seen.get(channel, -1)
            emit = percent >= 100 or percent >= (last + 10)
            if emit:
                progress_seen[channel] = percent
        # Info logs every 10% and on completion.
        if emit:
            logger.info(
                "PSF progress channel=%s %d/%d (%d%%)",
                channel,
                current,
                total,
                percent,
            )

    def _run_channel(channel_name: str, channel: ChannelVolume) -> tuple[str, np.ndarray, float]:
        t = time.perf_counter()
        logger.info("PSF channel start: %s shape=%s", channel_name, channel.data.shape)
        data = deconvolve_volume(
            channel.data,
            channel.spacing,
            effective_config,
            cancel_event=cancel_event,
            progress_callback=(lambda current, total: _progress(channel_name, current, total)),
        )
        return channel_name, data, (time.perf_counter() - t)

    if channel_workers <= 1:
        logger.info("PSF channel execution: sequential workers=%d", channel_workers)
        green_name, green_data, tg = _run_channel("green", dataset.green)
        red_name, red_data, tr = _run_channel("red", dataset.red)
        logger.info("PSF channel complete: %s dt=%.2fs", green_name, tg)
        logger.info("PSF channel complete: %s dt=%.2fs", red_name, tr)
    else:
        logger.info("PSF channel execution: parallel workers=%d", channel_workers)
        outputs: dict[str, np.ndarray] = {}
        with ThreadPoolExecutor(max_workers=channel_workers, thread_name_prefix="nvap-psf") as pool:
            futures = [
                pool.submit(_run_channel, "green", dataset.green),
                pool.submit(_run_channel, "red", dataset.red),
            ]
            for future in as_completed(futures):
                name, data, dt = future.result()
                outputs[name] = data
                logger.info("PSF channel complete: %s dt=%.2fs", name, dt)
        green_data = outputs["green"]
        red_data = outputs["red"]

    green = ChannelVolume(
        name="green",
        data=green_data,
        z_indices=list(dataset.green.z_indices),
        spacing=dataset.green.spacing,
    )
    red = ChannelVolume(
        name="red",
        data=red_data,
        z_indices=list(dataset.red.z_indices),
        spacing=dataset.red.spacing,
    )
    out = DatasetVolume(green=green, red=red, shared_z_range=dataset.shared_z_range)
    if (
        preprocess_config is not None
        and bool(effective_config.enabled)
        and int(effective_config.iterations) > 0
    ):
        out = postprocess_green_after_deconvolution(out, preprocess_config)
    logger.info("PSF pipeline complete total_dt=%.2fs", time.perf_counter() - t0)
    return out


def preprocess_for_deconvolution(
    dataset: DatasetVolume,
    preprocess_config: PreprocessConfig,
) -> DatasetVolume:
    """Apply preprocessing (denoising, branch preservation) before deconvolution."""
    if not preprocess_config.enabled:
        logger.info("Preprocessing disabled: using raw synchronized channels.")
        return dataset
    logger.info("Preprocessing for deconvolution: strategy=%s", preprocess_config.green_denoise_strategy)
    return preprocess_dataset(dataset, preprocess_config)


def prepare_dataset_for_mesh(
    dataset: DatasetVolume,
    preprocess_config: PreprocessConfig,
) -> DatasetVolume:
    return prepare_mesh_dataset(dataset, preprocess_config)


def default_threshold(volume: np.ndarray, fallback: float = 0.15) -> float:
    arr = np.asarray(volume, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or float(finite.max()) <= 0.0:
        return fallback
    try:
        value = float(threshold_otsu(finite))
    except ValueError:
        return fallback
    result = float(np.clip(value, 0.0, 1.0))
    logger.debug("Computed Otsu threshold=%.5f fallback=%.5f", result, fallback)
    return result


def default_green_threshold(volume: np.ndarray, fallback: float = 0.15) -> float:
    result = suggest_green_threshold(volume, fallback=fallback)
    logger.debug("Computed branch-aware green threshold=%.5f fallback=%.5f", result, fallback)
    return result
