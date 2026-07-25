from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import time
from typing import Callable

import numpy as np
import scipy.ndimage as ndi
from skimage import exposure
from skimage.filters import apply_hysteresis_threshold, frangi, meijering, sato, threshold_otsu
from skimage.morphology import binary_closing, binary_dilation, binary_opening, remove_small_holes, remove_small_objects, disk, white_tophat
from skimage.restoration import denoise_bilateral, denoise_nl_means, denoise_wavelet, rolling_ball

from nvap.accelerate import torch_gaussian_filter
from nvap.config.types import ChannelVolume, DatasetVolume, PreprocessConfig
from nvap.preprocess._executor import get_executor
from nvap.preprocess.denoisers import (
    denoise_wavelet_3d,
    estimate_noise_sigma,
    run_green_denoiser,
)
from nvap.runtime_optimization import configured_cpu_workers

logger = logging.getLogger(__name__)
_BRANCH_MAP_CHUNK_THRESHOLD = 64 * 1024 * 1024
_FAST_BG_VOXEL_THRESHOLD = 64 * 1024 * 1024
_GPU_FILTER_MIN_VOXELS = 4 * 1024 * 1024


def _gaussian_filter_fast(
    volume: np.ndarray,
    sigma: tuple[float, ...] | float,
    *,
    mode: str = "nearest",
) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    if isinstance(sigma, tuple):
        sigma_tuple = tuple(float(v) for v in sigma)
    else:
        sigma_tuple = tuple(float(sigma) for _ in range(arr.ndim))
    if mode == "nearest" and arr.size >= _GPU_FILTER_MIN_VOXELS:
        accelerated = torch_gaussian_filter(arr, sigma=sigma_tuple)
        if accelerated is not None:
            return np.asarray(accelerated, dtype=np.float32)
    return np.asarray(ndi.gaussian_filter(arr, sigma=sigma, mode=mode), dtype=np.float32)


def _resolve_worker_threads(config: PreprocessConfig) -> int:
    requested = int(config.cpu_worker_threads)
    if requested > 0:
        return max(1, requested)
    return max(1, min(32, configured_cpu_workers(os.cpu_count() or 1)))


def _fmt_stage_stats(arr: np.ndarray) -> str:
    finite = np.asarray(arr, dtype=np.float32)
    if finite.size == 0:
        return "empty"
    return (
        f"shape={finite.shape} min={float(np.min(finite)):.4f} "
        f"mean={float(np.mean(finite)):.4f} "
        f"std={float(np.std(finite)):.4f} "
        f"max={float(np.max(finite)):.4f}"
    )


def _normalize_branch_map(branch_map: np.ndarray) -> np.ndarray:
    arr = np.asarray(branch_map, dtype=np.float32)
    if arr.size == 0:
        return np.asarray(arr, dtype=np.float32)
    max_val = float(np.nanmax(arr))
    if not np.isfinite(max_val) or max_val <= 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    sample = arr
    if arr.size >= 32 * 1024 * 1024 and arr.ndim == 3:
        sample = arr[::2, ::4, ::4]
    lo = float(np.percentile(sample, 60.0))
    hi = float(np.percentile(sample, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= (lo + 1.0e-6):
        norm = arr / max_val
    else:
        norm = (arr - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    norm = _gaussian_filter_fast(norm, sigma=(0.0, 0.75, 0.75), mode="nearest")
    return np.clip(norm, 0.0, 1.0).astype(np.float32, copy=False)


def _tubeness_slicewise(
    volume: np.ndarray,
    sigmas: list[float],
    workers: int,
    progress_tag: str,
) -> np.ndarray:
    """Compute tubeness/vesselness filter slice-by-slice.
    
    Uses Sato tubeness (better for thin microglia branches than Meijering)
    with Meijering as fallback for robustness.
    """
    arr = np.asarray(volume, dtype=np.float32)
    depth = int(arr.shape[0])
    out = np.zeros_like(arr, dtype=np.float32)
    if depth <= 0:
        return out
    progress_every = max(1, depth // 8)

    def _run_slice(z: int) -> tuple[int, np.ndarray]:
        plane = np.asarray(arr[z], dtype=np.float32)
        # Sato tubeness is better for thin filaments (microglia branches)
        try:
            response = sato(
                plane,
                sigmas=sigmas,
                black_ridges=False,
                mode="reflect",
            )
        except Exception:
            # Fallback to meijering if sato fails
            response = meijering(
                plane,
                sigmas=sigmas,
                black_ridges=False,
                mode="reflect",
            )
        return z, np.asarray(response, dtype=np.float32)

    if workers > 1 and depth > 1:
        logger.info("%s: parallel tubeness workers=%d slices=%d", progress_tag, workers, depth)
        completed = 0
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nvap-branch") as pool:
            futures = [pool.submit(_run_slice, z) for z in range(depth)]
            for future in as_completed(futures):
                try:
                    z, response = future.result()
                    out[z] = response
                except Exception as exc:
                    logger.debug("%s slice failed: %s", progress_tag, exc)
                completed += 1
                if completed == 1 or completed == depth or (completed % progress_every) == 0:
                    logger.info("%s progress: slices=%d/%d", progress_tag, completed, depth)
        return out

    for z in range(depth):
        try:
            _, response = _run_slice(z)
            out[z] = response
        except Exception as exc:
            logger.debug("%s slice failed z=%d: %s", progress_tag, z, exc)
        done = z + 1
        if done == 1 or done == depth or (done % progress_every) == 0:
            logger.info("%s progress: slices=%d/%d", progress_tag, done, depth)
    return out


def stage_illumination_correction(volume: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    if config.flatfield_sigma_xy <= 0:
        return np.asarray(volume, dtype=np.float32).copy()
    arr = np.asarray(volume, dtype=np.float32)
    sigma_xy = float(config.flatfield_sigma_xy)
    # Fast approximation for large volumes: estimate background on downsampled XY and upsample back.
    if arr.size >= _FAST_BG_VOXEL_THRESHOLD and sigma_xy >= 12.0:
        ds = int(np.clip(np.floor(sigma_xy / 12.0), 2, 4))
        logger.info(
            "Illumination correction: fast downsampled background path (downsample=%dx).",
            ds,
        )
        reduced = arr[:, ::ds, ::ds]
        bg_small = _gaussian_filter_fast(
            reduced,
            sigma=(0.0, sigma_xy / ds, sigma_xy / ds),
            mode="nearest",
        )
        zoom_y = arr.shape[1] / bg_small.shape[1]
        zoom_x = arr.shape[2] / bg_small.shape[2]
        background = ndi.zoom(bg_small, zoom=(1.0, zoom_y, zoom_x), order=1)
        # Guard against zoom rounding producing off-by-one shape mismatch
        if background.shape[1] < arr.shape[1] or background.shape[2] < arr.shape[2]:
            padded = np.zeros_like(arr, dtype=np.float32)
            sy = min(background.shape[1], arr.shape[1])
            sx = min(background.shape[2], arr.shape[2])
            padded[:, :sy, :sx] = background[:, :sy, :sx]
            background = padded
        else:
            background = background[:, : arr.shape[1], : arr.shape[2]]
    else:
        background = _gaussian_filter_fast(
            arr,
            sigma=(0.0, sigma_xy, sigma_xy),
            mode="nearest",
        )
    corrected = arr - background
    corrected -= float(np.min(corrected))
    return corrected.astype(np.float32, copy=False)


def stage_intensity_stabilization(volume: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    out = np.empty_like(volume, dtype=np.float32)
    low_pct = float(config.contrast_low_pct)
    high_pct = float(config.contrast_high_pct)
    workers = _resolve_worker_threads(config)
    depth = int(volume.shape[0])

    def _normalize_plane(z: int) -> tuple[int, np.ndarray]:
        plane = np.asarray(volume[z], dtype=np.float32)
        low = float(np.percentile(plane, low_pct))
        high = float(np.percentile(plane, high_pct))
        if high <= low:
            return z, np.zeros_like(plane, dtype=np.float32)
        norm = np.clip((plane - low) / (high - low), 0.0, 1.0).astype(np.float32, copy=False)
        return z, norm

    if workers > 1 and depth > 1:
        logger.info(
            "Intensity stabilization: parallel slice normalization threads=%d slices=%d",
            workers,
            depth,
        )
        with get_executor(workers, "nvap-pre") as pool:
            for z, normalized in pool.map(_normalize_plane, range(depth)):
                out[z] = normalized
    else:
        for z in range(depth):
            _, normalized = _normalize_plane(z)
            out[z] = normalized
    return out


def stage_noise_model_estimation(volume: np.ndarray, config: PreprocessConfig) -> dict[str, float | str]:
    sigma = estimate_noise_sigma(volume)
    if config.green_noise_model == "auto":
        noise_model = "poisson_gaussian" if float(np.max(volume)) > 0.2 else "gaussian"
    else:
        noise_model = config.green_noise_model
    return {"sigma": float(sigma), "noise_model": str(noise_model)}


def stage_branch_map_estimation(
    volume: np.ndarray,
    config: PreprocessConfig,
    channel_name: str,
) -> np.ndarray:
    if channel_name != "green" or not config.preserve_branches:
        return np.zeros_like(volume, dtype=np.float32)
    arr = np.asarray(volume, dtype=np.float32)

    workers = _resolve_worker_threads(config)
    # Large stacks use a fast slice-wise path (optionally downsampled in XY).
    if arr.size >= _BRANCH_MAP_CHUNK_THRESHOLD and bool(config.green_chunked_processing):
        slice_pixels = int(arr.shape[1] * arr.shape[2]) if arr.ndim == 3 else 0
        downsample_xy = 1
        if slice_pixels >= (1024 * 1024):
            downsample_xy = 2
        if slice_pixels >= (2048 * 2048):
            downsample_xy = 3
        sigmas = [0.8, 1.2] if downsample_xy > 1 else [0.8, 1.2, 1.8]
        logger.info(
            "Branch map: accelerated mode enabled (voxels=%d downsample_xy=%d workers=%d sigmas=%s).",
            int(arr.size),
            downsample_xy,
            workers,
            sigmas,
        )
        src = arr[:, ::downsample_xy, ::downsample_xy] if downsample_xy > 1 else arr
        branch_small = _tubeness_slicewise(
            src,
            sigmas=sigmas,
            workers=workers,
            progress_tag="Branch map",
        )
        if downsample_xy > 1:
            zoom_y = arr.shape[1] / branch_small.shape[1]
            zoom_x = arr.shape[2] / branch_small.shape[2]
            branch = ndi.zoom(branch_small, zoom=(1.0, zoom_y, zoom_x), order=1)
            # Guard against zoom rounding producing off-by-one shape mismatch
            if branch.shape[1] < arr.shape[1] or branch.shape[2] < arr.shape[2]:
                padded = np.zeros_like(arr, dtype=np.float32)
                sy = min(branch.shape[1], arr.shape[1])
                sx = min(branch.shape[2], arr.shape[2])
                padded[:, :sy, :sx] = branch[:, :sy, :sx]
                branch = padded
            else:
                branch = branch[:, : arr.shape[1], : arr.shape[2]]
        else:
            branch = branch_small
        return _normalize_branch_map(branch)

    try:
        # Sato tubeness: better sensitivity to thin microglia processes
        response = sato(
            arr,
            sigmas=[0.8, 1.2, 1.8],
            black_ridges=False,
            mode="reflect",
        )
        branch = np.asarray(response, dtype=np.float32)
    except Exception:
        try:
            response = meijering(
                arr,
                sigmas=[0.8, 1.2, 1.8],
                black_ridges=False,
                mode="reflect",
            )
            branch = np.asarray(response, dtype=np.float32)
        except Exception as exc:
            logger.debug("Branch map estimation failed, using zeros: %s", exc)
            return np.zeros_like(volume, dtype=np.float32)
    return _normalize_branch_map(branch)


def _denoise_volume_default(
    volume: np.ndarray,
    config: PreprocessConfig,
    denoise_strength: float,
) -> np.ndarray:
    method = config.denoise_method
    if method == "none":
        return np.asarray(volume, dtype=np.float32).copy()

    if method == "anisotropic":
        sigma_xy = float(max(denoise_strength * 24.0, 0.15))
        sigma_z = sigma_xy * 0.55
        return ndi.gaussian_filter(
            np.asarray(volume, dtype=np.float32),
            sigma=(sigma_z, sigma_xy, sigma_xy),
            mode="nearest",
        ).astype(np.float32, copy=False)

    if method == "wavelet":
        sigma = estimate_noise_sigma(np.asarray(volume, dtype=np.float32))
        return denoise_wavelet_3d(
            np.asarray(volume, dtype=np.float32),
            sigma=sigma,
            config=config,
            strength=denoise_strength * 40.0,
        )

    if method == "non_local_means":
        patch_kw = dict(patch_size=3, patch_distance=4, fast_mode=True, channel_axis=None)
        h = float(max(denoise_strength * 0.9, 0.004))
        out = denoise_nl_means(
            np.asarray(volume, dtype=np.float32),
            h=h,
            sigma=0.0,
            preserve_range=True,
            **patch_kw,
        )
        return np.asarray(out, dtype=np.float32)

    if method == "bilateral":
        out = np.empty_like(volume, dtype=np.float32)
        sigma_color = float(max(denoise_strength * 1.4, 0.01))
        sigma_spatial = 2.0
        for z in range(volume.shape[0]):
            out[z] = np.asarray(
                denoise_bilateral(
                    np.asarray(volume[z], dtype=np.float32),
                    sigma_color=sigma_color,
                    sigma_spatial=sigma_spatial,
                    channel_axis=None,
                ),
                dtype=np.float32,
            )
        return out

    raise ValueError(f"Unsupported denoise method: {method}")


def stage_denoise_main(
    volume: np.ndarray,
    channel_name: str,
    config: PreprocessConfig,
    denoise_strength: float,
    branch_map: np.ndarray,
) -> tuple[np.ndarray, str]:
    if channel_name == "green":
        denoised, backend = run_green_denoiser(
            np.asarray(volume, dtype=np.float32),
            np.asarray(branch_map, dtype=np.float32),
            config=config,
            denoise_strength=denoise_strength,
        )
        return denoised, backend
    out = _denoise_volume_default(np.asarray(volume, dtype=np.float32), config, denoise_strength)
    return out.astype(np.float32, copy=False), config.denoise_method


def _attenuate_small_isolated_components(
    volume: np.ndarray,
    threshold: float,
    min_voxels: int,
    attenuation: float,
    exempt_mask: np.ndarray | None = None,
) -> np.ndarray:
    min_voxels = max(2, int(min_voxels))
    attenuation = float(np.clip(attenuation, 0.0, 1.0))
    threshold = float(np.clip(threshold, 0.0, 1.0))
    mask = np.asarray(volume, dtype=np.float32) >= threshold
    if not np.any(mask):
        return np.asarray(volume, dtype=np.float32).copy()

    labels, count = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count <= 0:
        return np.asarray(volume, dtype=np.float32).copy()

    sizes = np.bincount(labels.ravel())
    small_labels = np.where((sizes > 0) & (sizes < min_voxels))[0]
    small_labels = small_labels[small_labels != 0]
    if small_labels.size == 0:
        return np.asarray(volume, dtype=np.float32).copy()

    small_mask = np.isin(labels, small_labels)
    if exempt_mask is not None:
        small_mask &= ~np.asarray(exempt_mask, dtype=bool)
    out = np.asarray(volume, dtype=np.float32).copy()
    out[small_mask] *= attenuation
    return out


def _apply_green_branch_preservation(
    reference: np.ndarray,
    denoised: np.ndarray,
    branch_map: np.ndarray,
    config: PreprocessConfig,
    *,
    mode: str,
) -> np.ndarray:
    """Preserve branch-like structures using branch-map-guided intensity floors."""
    ref = np.asarray(reference, dtype=np.float32)
    out = np.asarray(denoised, dtype=np.float32).copy()
    branch = np.clip(np.asarray(branch_map, dtype=np.float32), 0.0, 1.0)
    if ref.shape != out.shape or ref.shape != branch.shape:
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    branch_protect = float(np.clip(config.green_branch_protection, 0.0, 1.0))
    floor_scale = (0.44 + (0.28 * branch_protect))
    if mode == "post_speckle":
        floor_scale *= 0.94
    floor = np.clip(floor_scale * branch, 0.12, 0.9)
    branch_floor = ref * floor
    out = np.maximum(out, branch_floor)

    # Keep connected thin branch segments from being attenuated too aggressively.
    branch_mask = branch >= max(0.52, branch_protect * 0.78)
    if np.any(branch_mask):
        labels, count = ndi.label(branch_mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
        if count > 0:
            sizes = np.bincount(labels.ravel())
            min_keep = max(4, int(config.green_speckle_min_voxels) // 2)
            large_labels = np.where(sizes >= min_keep)[0]
            large_labels = large_labels[large_labels != 0]
            if large_labels.size > 0:
                keep_mask = np.isin(labels, large_labels)
                boost_scale = 0.68 + (0.20 * branch_protect)
                if mode == "post_speckle":
                    boost_scale *= 0.96
                boosted = ref * boost_scale
                out[keep_mask] = np.maximum(out[keep_mask], boosted[keep_mask])

    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _compute_branch_exempt_mask(
    arr: np.ndarray,
    branch: np.ndarray,
    config: PreprocessConfig,
) -> np.ndarray:
    local_mean = ndi.uniform_filter(arr, size=(1, 5, 5), mode="nearest")
    local_contrast = np.maximum(arr - local_mean, 0.0)
    contrast_gate = local_contrast >= np.quantile(local_contrast, 0.45)
    intensity_gate = arr >= float(np.quantile(arr, 0.60))
    branch_gate = branch >= max(0.45, float(config.green_branch_protection) * 0.58)
    candidate = branch_gate & (contrast_gate | intensity_gate)
    candidate = ndi.maximum_filter(candidate.astype(np.uint8), size=(1, 3, 3), mode="nearest") > 0
    labels, count = ndi.label(candidate, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count <= 0:
        return np.zeros_like(candidate)
    sizes = np.bincount(labels.ravel())
    keep_min = max(4, int(config.green_speckle_min_voxels) // 2)
    keep = np.where(sizes >= keep_min)[0]
    keep = keep[keep != 0]
    return np.isin(labels, keep) if keep.size > 0 else np.zeros_like(candidate)


def stage_speckle_control(
    volume: np.ndarray,
    branch_map: np.ndarray,
    config: PreprocessConfig,
    channel_name: str,
) -> np.ndarray:
    if channel_name != "green":
        return np.asarray(volume, dtype=np.float32)
    arr = np.asarray(volume, dtype=np.float32)
    branch = np.clip(np.asarray(branch_map, dtype=np.float32), 0.0, 1.0)

    speckle_threshold = float(max(0.08, np.quantile(arr, 0.82) * 0.75))
    branch_exempt = _compute_branch_exempt_mask(arr, branch, config)

    out = _attenuate_small_isolated_components(
        arr,
        threshold=speckle_threshold,
        min_voxels=int(config.green_speckle_min_voxels),
        attenuation=float(config.green_speckle_attenuation),
        exempt_mask=branch_exempt,
    )
    # Secondary peak clamp for isolated bright spikes that survive component attenuation.
    local_mean = ndi.uniform_filter(out, size=(1, 5, 5), mode="nearest")
    peak_strength = np.maximum(arr - local_mean, 0.0)
    peak_cut = float(np.quantile(peak_strength, 0.97))
    peak_mask = (
        (arr >= ndi.maximum_filter(arr, size=(1, 3, 3), mode="nearest"))
        & (peak_strength >= peak_cut)
        & (~branch_exempt)
    )
    if np.any(peak_mask):
        clamp = float(np.quantile(out, 0.70) * 0.6)
        out[peak_mask] = np.minimum(out[peak_mask], clamp)
    out = _apply_green_branch_preservation(arr, out, branch, config, mode="post_speckle")
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)



def stage_restore_branches_near_bright_pixels(
    reference: np.ndarray,
    denoised: np.ndarray,
) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float32)
    out = np.asarray(denoised, dtype=np.float32).copy()
    if ref.size == 0:
        return out
    bright_cut = float(np.quantile(ref, 0.985))
    if not np.isfinite(bright_cut) or bright_cut <= 0.0:
        return out
    bright = ref >= bright_cut
    if not np.any(bright):
        return out
    near_bright = ndi.maximum_filter(bright.astype(np.uint8), size=(1, 9, 9), mode="nearest") > 0
    detail = np.maximum(ref - ndi.gaussian_filter(ref, sigma=(0.0, 0.9, 0.9), mode="nearest"), 0.0)
    if np.any(near_bright):
        detail_cut = float(np.quantile(detail[near_bright], 0.70))
    else:
        detail_cut = float(np.quantile(detail, 0.92))
    if not np.isfinite(detail_cut):
        return out
    branch_like = near_bright & (detail >= detail_cut)
    if np.any(branch_like):
        out[branch_like] = np.maximum(out[branch_like], ref[branch_like] * 0.93)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _normalize_positive_detail(detail: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(detail, dtype=np.float32), 0.0)
    if arr.size == 0:
        return arr
    sample = arr
    if arr.size >= 32 * 1024 * 1024 and arr.ndim == 3:
        sample = arr[::2, ::4, ::4]
    hi = float(np.percentile(sample, 99.7))
    if not np.isfinite(hi) or hi <= 1.0e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / hi, 0.0, 1.0).astype(np.float32, copy=False)


def _normalize_positive_percentile(detail: np.ndarray, percentile: float) -> np.ndarray:
    arr = np.maximum(np.asarray(detail, dtype=np.float32), 0.0)
    if arr.size == 0:
        return arr
    sample = arr
    if arr.size >= 32 * 1024 * 1024 and arr.ndim == 3:
        sample = arr[::2, ::4, ::4]
    hi = float(np.percentile(sample, float(percentile)))
    if not np.isfinite(hi) or hi <= 1.0e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / hi, 0.0, 1.0).astype(np.float32, copy=False)


def _white_tophat_slicewise(volume: np.ndarray, radius: int) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    radius = int(max(3, radius))
    footprint = disk(radius)
    out = np.empty_like(arr, dtype=np.float32)
    for z in range(int(arr.shape[0])):
        out[z] = np.asarray(
            white_tophat(arr[z], footprint=footprint),
            dtype=np.float32,
        )
    return out


def _clahe_slicewise(volume: np.ndarray, clip_limit: float = 0.012) -> np.ndarray:
    arr = np.clip(np.asarray(volume, dtype=np.float32), 0.0, 1.0)
    out = np.empty_like(arr, dtype=np.float32)
    for z in range(int(arr.shape[0])):
        plane = np.asarray(arr[z], dtype=np.float32)
        if float(np.max(plane)) <= 1.0e-6:
            out[z] = plane
            continue
        tile_y = max(16, int(round(plane.shape[0] / 8)))
        tile_x = max(16, int(round(plane.shape[1] / 8)))
        out[z] = np.asarray(
            exposure.equalize_adapthist(
                plane,
                kernel_size=(tile_y, tile_x),
                clip_limit=float(clip_limit),
            ),
            dtype=np.float32,
        )
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _branch_soma_preserve_2d(
    raw: np.ndarray,
    processed: np.ndarray,
    *,
    branch_strength: float = 0.45,
    soma_strength: float = 0.90,
) -> np.ndarray:
    """Lift branch ridges and soma cores after background/noise suppression."""
    arr = np.clip(np.asarray(raw, dtype=np.float32), 0.0, 1.0)
    out = np.clip(np.asarray(processed, dtype=np.float32), 0.0, 1.0).copy()
    if arr.shape != out.shape or arr.ndim != 2 or arr.size == 0:
        return out

    raw_detail = _normalize_positive_detail(
        np.maximum(arr - ndi.gaussian_filter(arr, sigma=2.5, mode="nearest"), 0.0)
    )
    try:
        branch_response = sato(
            _normalize_positive_detail(arr),
            sigmas=[0.7, 1.1, 1.6],
            black_ridges=False,
            mode="reflect",
        )
        branch = _normalize_positive_detail(branch_response)
    except Exception:
        branch = raw_detail
    branch = ndi.gaussian_filter(branch, sigma=0.45, mode="nearest")
    branch_mask = branch > 0.18
    if np.any(branch_mask):
        out[branch_mask] = np.maximum(
            out[branch_mask],
            raw_detail[branch_mask] * float(branch_strength),
        )

    soma_cut = float(np.percentile(arr, 99.2))
    if np.isfinite(soma_cut) and soma_cut > 0.0:
        soma_mask = arr >= soma_cut
        soma_mask = ndi.binary_dilation(
            soma_mask,
            structure=np.ones((7, 7), dtype=bool),
            iterations=1,
        )
        raw_norm = _normalize_positive_detail(arr)
        out[soma_mask] = np.maximum(
            out[soma_mask],
            raw_norm[soma_mask] * float(soma_strength),
        )
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _despeckle_slicewise(
    volume: np.ndarray,
    *,
    threshold_pct: float = 99.2,
    min_size: int = 4,
    attenuation: float = 0.20,
) -> np.ndarray:
    arr = np.clip(np.asarray(volume, dtype=np.float32), 0.0, 1.0)
    out = arr.copy()
    min_size = max(2, int(min_size))
    for z in range(int(arr.shape[0])):
        plane = arr[z]
        threshold = float(np.percentile(plane, threshold_pct))
        mask = plane >= threshold
        labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        if count <= 0:
            continue
        sizes = np.bincount(labels.ravel())
        small_labels = np.where((sizes > 0) & (sizes < min_size))[0]
        small_labels = small_labels[small_labels != 0]
        if small_labels.size > 0:
            out[z][np.isin(labels, small_labels)] *= float(attenuation)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _wavelet_denoise_slicewise_light(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    out = np.empty_like(arr, dtype=np.float32)
    for z in range(int(arr.shape[0])):
        try:
            denoised = denoise_wavelet(
                np.clip(arr[z], 0.0, 1.0),
                sigma=0.018,
                method="BayesShrink",
                mode="soft",
                rescale_sigma=True,
                channel_axis=None,
            )
            out[z] = np.asarray(denoised, dtype=np.float32)
        except Exception:
            out[z] = ndi.gaussian_filter(arr[z], sigma=0.25, mode="nearest")
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _tuned_microglia_preserve_background(volume: np.ndarray) -> np.ndarray:
    """Tuned background suppression that keeps microglia branches and somas."""
    arr = np.clip(np.asarray(volume, dtype=np.float32), 0.0, 1.0)
    background = ndi.gaussian_filter(arr, sigma=(0.0, 24.0, 24.0), mode="nearest")
    detail = _normalize_positive_detail(np.maximum(arr - (0.94 * background), 0.0))
    contrast = _clahe_slicewise(detail, clip_limit=0.004)
    enhanced = np.clip((0.72 * detail) + (0.28 * contrast), 0.0, 1.0)

    preserved = np.empty_like(enhanced, dtype=np.float32)
    for z in range(int(arr.shape[0])):
        preserved[z] = _branch_soma_preserve_2d(
            arr[z],
            enhanced[z],
            branch_strength=0.45,
            soma_strength=0.92,
        )

    cleaned = _wavelet_denoise_slicewise_light(preserved)
    cleaned = _despeckle_slicewise(
        cleaned,
        threshold_pct=99.2,
        min_size=4,
        attenuation=0.20,
    )

    for z in range(int(arr.shape[0])):
        cleaned[z] = _branch_soma_preserve_2d(
            arr[z],
            cleaned[z],
            branch_strength=0.41,
            soma_strength=0.90,
        )
    return np.clip(cleaned, 0.0, 1.0).astype(np.float32, copy=False)


def _keep_process_like_components(
    mask: np.ndarray,
    vesselness: np.ndarray,
    detail: np.ndarray,
) -> np.ndarray:
    """Keep branch-like connected components and reject dot-like speckles."""
    labels, count = ndi.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3), dtype=bool))
    if count <= 0:
        return np.zeros_like(mask, dtype=bool)

    keep = np.zeros(int(count) + 1, dtype=bool)
    for label, bbox in enumerate(ndi.find_objects(labels), start=1):
        if bbox is None:
            continue
        component = labels[bbox] == label
        area = int(np.count_nonzero(component))
        if area <= 0:
            continue

        height, width = component.shape
        longest = max(height, width)
        shortest = max(1, min(height, width))
        aspect = float(longest) / float(shortest)
        extent = float(area) / float(max(1, height * width))
        vessel_mean = float(np.mean(vesselness[bbox][component]))
        detail_mean = float(np.mean(detail[bbox][component]))

        broad_process = area >= 44 and vessel_mean >= 0.12 and detail_mean >= 0.035
        elongated_process = (
            area >= 7
            and longest >= 6
            and aspect >= 1.8
            and vessel_mean >= 0.10
            and detail_mean >= 0.030
        )
        compact_but_supported = (
            area >= 20
            and longest >= 6
            and extent <= 0.78
            and vessel_mean >= 0.24
            and detail_mean >= 0.055
        )
        keep[label] = broad_process or elongated_process or compact_but_supported

    if not np.any(keep):
        return np.zeros_like(mask, dtype=bool)
    return keep[labels]


# Opening radius (px) used to extract the compact soma body. A morphological
# opening removes bright structures thinner than ~2x this radius (processes,
# speckles) while preserving the rounder cell body.
_SOMA_OPEN_RADIUS = 2


def _microscopy_clean_background(
    volume: np.ndarray,
    workers: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Library-first microscopy cleanup for microglia soma and processes.

    Each z-slice is processed independently (rolling-ball background subtraction
    + multi-scale vesselness), so the loop is parallelized across CPU workers.
    These are CPU skimage/scipy ops with no GPU/DirectML path; threads give real
    speedup because the heavy kernels release the GIL.
    """
    arr = np.clip(np.asarray(volume, dtype=np.float32), 0.0, 1.0)
    if arr.size == 0:
        return arr.copy()

    out = np.zeros_like(arr, dtype=np.float32)
    raw_norm = _normalize_positive_detail(arr)
    closing_fp = disk(1)
    soma_fp = disk(2)
    depth = int(arr.shape[0])

    def _run_slice(z: int):
        plane = np.asarray(arr[z], dtype=np.float32)
        if plane.size == 0:
            return z, None
        plane_smooth = ndi.gaussian_filter(plane, sigma=0.65, mode="nearest")

        try:
            background = rolling_ball(plane_smooth, radius=18)
            detail = np.maximum(plane_smooth - np.asarray(background, dtype=np.float32), 0.0)
        except Exception:
            background = ndi.gaussian_filter(plane_smooth, sigma=18.0, mode="nearest")
            detail = np.maximum(plane_smooth - (0.95 * background), 0.0)

        detail = _normalize_positive_percentile(detail, 99.8)
        local_detail = _normalize_positive_detail(
            np.maximum(
                plane_smooth - ndi.gaussian_filter(plane_smooth, sigma=2.0, mode="nearest"),
                0.0,
            )
        )
        vessel_input = _normalize_positive_percentile((0.62 * detail) + (0.38 * local_detail), 99.7)
        vessel_input = ndi.gaussian_filter(vessel_input, sigma=0.45, mode="nearest")
        try:
            branch_response = sato(
                vessel_input,
                sigmas=[0.7, 1.1, 1.6, 2.2],
                black_ridges=False,
                mode="reflect",
            )
        except Exception:
            branch_response = frangi(
                vessel_input,
                sigmas=[0.7, 1.1, 1.6, 2.2],
                black_ridges=False,
                mode="reflect",
            )
        branch = _normalize_positive_percentile(branch_response, 99.5)
        branch = ndi.gaussian_filter(branch, sigma=0.35, mode="nearest")

        branch_cut = max(0.10, float(np.percentile(branch, 96.5)))
        detail_cut = max(0.035, float(np.percentile(detail, 90.0)))
        local_cut = max(0.055, float(np.percentile(local_detail, 88.0)))
        process_mask = (branch >= branch_cut) & (detail >= detail_cut)
        process_mask |= (branch >= 0.20) & (local_detail >= local_cut) & (detail >= 0.025)
        process_mask |= (local_detail >= 0.34) & (branch >= 0.12) & (detail >= 0.08)
        process_mask = binary_closing(process_mask, footprint=closing_fp)
        process_mask = _keep_process_like_components(process_mask, branch, detail)

        # Bridge faint gaps along a process: hysteresis grows the strong ridge
        # seeds through weaker but still ridge-like pixels, then we keep only the
        # weak pixels that connect back to a confirmed process. A process whose
        # middle dims toward background stays continuous, while isolated blobs
        # (speckles) have no ridge to grow along and are never connected in.
        if np.any(process_mask):
            branch_hi = max(branch_cut, float(np.percentile(branch, 97.5)))
            branch_lo = max(0.04, float(np.percentile(branch, 80.0)))
            if branch_hi > branch_lo:
                ridge = apply_hysteresis_threshold(branch, branch_lo, branch_hi)
                ridge = ridge & ((detail >= 0.012) | (local_detail >= 0.030))
                bridged = ndi.binary_propagation(process_mask, mask=(process_mask | ridge))
                process_mask = _keep_process_like_components(bridged, branch, detail)

        loose_process = (branch >= 0.075) & (detail >= 0.025) & (local_detail >= 0.045)
        loose_process = binary_closing(loose_process, footprint=closing_fp)
        loose_process = _keep_process_like_components(loose_process, branch, detail)
        process_context = loose_process | (
            binary_dilation(process_mask, footprint=disk(2))
            & ((branch >= 0.060) | (local_detail >= 0.085))
            & (detail >= 0.020)
        )

        # Soma = compact bright *body*, not merely the brightest pixels. A
        # morphological opening strips thin bright streaks (processes/speckles)
        # that the old top-percentile rule mislabeled as soma and rounds the
        # boundary; a fallback keeps the body if opening would erase it entirely.
        soma_cut = float(np.percentile(plane, 98.7))
        if np.isfinite(soma_cut) and soma_cut > 0.0:
            soma_body = plane >= soma_cut
            soma_body = binary_closing(soma_body, footprint=soma_fp)
            soma_body = remove_small_holes(soma_body, area_threshold=16)
            soma_body = remove_small_objects(soma_body, min_size=18)
            soma_opened = binary_opening(soma_body, footprint=disk(_SOMA_OPEN_RADIUS))
            soma_mask = soma_opened if np.any(soma_opened) else soma_body
            soma_mask = binary_dilation(soma_mask, footprint=soma_fp)
        else:
            soma_mask = np.zeros_like(plane, dtype=bool)

        # Keep soma and branch responses disjoint: pixels claimed by the soma body
        # should not also be lifted as "process" ridges (which produced a halo of
        # false branches around the soma rim).
        if np.any(soma_mask):
            process_mask = process_mask & (~soma_mask)
            process_context = process_context & (~soma_mask)

        support = process_mask | soma_mask
        support = remove_small_objects(support, min_size=8)

        plane_out = np.zeros_like(plane, dtype=np.float32)
        process_signal = np.maximum.reduce(
            (
                detail * 0.92,
                branch * 1.14,
                local_detail * 0.90,
            )
        )
        process_alpha = np.clip(
            ndi.gaussian_filter(process_mask.astype(np.float32), sigma=1.1, mode="nearest"),
            0.0,
            1.0,
        )
        context_alpha = np.maximum(process_alpha, np.where(loose_process, 0.56, 0.0))
        context_alpha = np.maximum(context_alpha, np.where(process_context, 0.34, 0.0))
        plane_out[process_context] = process_signal[process_context] * context_alpha[process_context]
        plane_out[process_mask] = np.maximum(plane_out[process_mask], process_signal[process_mask])

        soma_signal = np.maximum(raw_norm[z] * 1.25, detail * 1.08)
        plane_out[soma_mask] = np.maximum(plane_out[soma_mask], soma_signal[soma_mask])

        halo = binary_dilation(support, footprint=disk(2)) & (~support)
        halo_keep = halo & (branch > 0.34) & (detail > 0.16)
        plane_out[halo_keep] = np.maximum(plane_out[halo_keep], process_signal[halo_keep] * 0.08)

        visible = plane_out >= 0.025
        visible = remove_small_objects(visible, min_size=32)
        plane_out[~visible] = 0.0
        plane_out = ndi.gaussian_filter(plane_out, sigma=0.35, mode="nearest")
        plane_out[~binary_dilation(visible, footprint=closing_fp)] = 0.0
        plane_out[plane_out < 0.018] = 0.0
        return z, np.clip(plane_out, 0.0, 1.0)

    if workers > 1 and depth > 1:
        logger.info("microscopy_clean: parallel workers=%d slices=%d", workers, depth)
        progress_every = max(1, depth // 8)
        completed = 0
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nvap-mclean") as pool:
            for future in as_completed([pool.submit(_run_slice, z) for z in range(depth)]):
                try:
                    z, plane_out = future.result()
                    if plane_out is not None:
                        out[z] = plane_out
                except Exception as exc:  # pragma: no cover - defensive per-slice path
                    logger.debug("microscopy_clean slice failed: %s", exc)
                completed += 1
                if completed == 1 or completed == depth or (completed % progress_every) == 0:
                    logger.info("microscopy_clean progress: slices=%d/%d", completed, depth)
                    if progress_callback is not None:
                        progress_callback(completed, depth)
    else:
        for z in range(depth):
            _, plane_out = _run_slice(z)
            if plane_out is not None:
                out[z] = plane_out
            if progress_callback is not None:
                progress_callback(z + 1, depth)

    return out.astype(np.float32, copy=False)


def _reconnect_and_denoise_microglia(
    volume: np.ndarray,
    *,
    grain_open_radius: int = 1,
    bridge_radius_px: int = 2,
    far_voxels: float = 12.0,
    min_object_voxels: int = 16,
    cell_min_voxels: int = 256,
) -> np.ndarray:
    """Strip tiny loose noise grains, then reinforce/denoise microglia.

    Run after the slice-wise microscopy cleanup:

    0. De-speckle - a per-slice grayscale opening removes bright specks thinner
       than ``grain_open_radius`` (the diffuse background "grain" the cleanup
       leaves behind) while keeping soma bodies and branches, which are wider
       than the element. This is the main grain remover.
    1. Reinforce - a small per-slice grayscale closing bridges faint in-plane
       gaps so segments of the same process/cell reconnect. Genuine branches
       rejoin their soma and become part of one large 3D component (an
       "anchor"); isolated bits stay separate.
    2. Denoise - label the bridged volume in 3D. Substantial components
       (>= ``cell_min_voxels``) are real cells and always kept (a field of many
       microglia is preserved). A *small, separate* component is only kept when
       it is big enough (>= ``min_object_voxels``) AND hugging a cell (within
       ``far_voxels``); everything else - tiny specks and far-flung debris - is
       dropped.
    """
    arr = np.clip(np.asarray(volume, dtype=np.float32), 0.0, 1.0)
    if arr.ndim != 3 or arr.size == 0:
        return arr.copy()

    # 0) De-speckle: grayscale opening kills fine grain (bright specks thinner
    # than the structuring element) without eroding wider soma/branch signal.
    if grain_open_radius and grain_open_radius > 0:
        open_fp = disk(int(grain_open_radius))
        opened = np.empty_like(arr)
        for z in range(int(arr.shape[0])):
            opened[z] = ndi.grey_opening(arr[z], footprint=open_fp, mode="nearest")
        arr = opened

    floor = 0.03
    mask = arr > floor
    if not np.any(mask):
        return arr.copy()

    # 1) Reinforce connections (per-slice grayscale closing bridges small gaps).
    footprint = disk(int(max(1, bridge_radius_px)))
    reinforced = np.empty_like(arr)
    for z in range(int(arr.shape[0])):
        reinforced[z] = ndi.grey_closing(arr[z], footprint=footprint, mode="nearest")
    bridged_mask = reinforced > floor

    # 2) Keep every cell-sized mass plus small bits near one; drop far debris.
    labels, n = ndi.label(bridged_mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n <= 1:
        kept = bridged_mask
    else:
        comp_ids = np.arange(1, n + 1)
        sizes = np.bincount(labels.ravel(), minlength=n + 1)[1:]
        substantial = sizes >= int(cell_min_voxels)
        # Anchors = real cell bodies. If none reach the cell threshold (faint or
        # heavily fragmented field), fall back to the single largest component so
        # we never wipe out all signal.
        if np.any(substantial):
            anchor_ids = comp_ids[substantial]
        else:
            anchor_ids = comp_ids[[int(np.argmax(sizes))]]
        anchor_mask = np.isin(labels, anchor_ids)
        dist_to_anchor = ndi.distance_transform_edt(~anchor_mask)
        min_dist = np.asarray(ndi.minimum(dist_to_anchor, labels, comp_ids), dtype=np.float64)
        is_anchor = np.isin(comp_ids, anchor_ids)
        keep_comp = is_anchor | (
            (sizes >= int(min_object_voxels)) & (min_dist <= float(far_voxels))
        )
        lut = np.zeros(n + 1, dtype=bool)
        lut[comp_ids[keep_comp]] = True
        kept = lut[labels]

    # 3) Keep original signal where retained, fill bridged gaps, drop the rest.
    out = np.where(kept & mask, arr, 0.0).astype(np.float32, copy=False)
    bridge_fill = kept & bridged_mask & (~mask)
    if np.any(bridge_fill):
        out[bridge_fill] = reinforced[bridge_fill]
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


MICROGLIA_ENHANCEMENT_METHODS: dict[str, str] = {
    "microglia_preserve": "Microglia-preserving combined",
    "microscopy_clean": "Microscopy clean soma/branch enhancement",
    "imagej_rolling_ball": "ImageJ/Fiji rolling ball background subtraction",
    "basic": "BaSiC-style retrospective shading correction",
    "cidre": "CIDRE-style illumination correction",
    "white_tophat": "scikit-image white top-hat",
    "clahe": "scikit-image CLAHE / adaptive histogram equalization",
}


def _rolling_ball_slicewise(volume: np.ndarray, radius: int) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    out = np.empty_like(arr, dtype=np.float32)
    radius = int(max(3, radius))
    for z in range(int(arr.shape[0])):
        background = rolling_ball(arr[z], radius=radius)
        out[z] = np.maximum(arr[z] - np.asarray(background, dtype=np.float32), 0.0)
    return out


def _imagej_rolling_ball_slicewise(
    volume: np.ndarray,
    *,
    radius: int = 5,
    multiplier: float = 1.5,
) -> np.ndarray:
    """ImageJ-style Subtract Background followed by Multiply.

    Matches the requested macro sequence:
    run("Subtract Background...", "rolling=5 sliding");
    run("Multiply...", "value=1.5");
    """
    subtracted = _rolling_ball_slicewise(volume, radius=int(radius))
    return np.clip(
        subtracted * float(multiplier),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)


def _restore_soma_interiors_after_imagej_rolling_ball(
    raw: np.ndarray,
    imagej_enhanced: np.ndarray,
) -> np.ndarray:
    """Fill soma interiors that strict rolling-ball subtraction turns into rings."""
    arr = np.clip(np.asarray(raw, dtype=np.float32), 0.0, 1.0)
    out = np.clip(np.asarray(imagej_enhanced, dtype=np.float32), 0.0, 1.0).copy()
    if arr.shape != out.shape or arr.ndim != 3 or arr.size == 0:
        return out

    raw_norm = _normalize_positive_detail(arr)
    structure = np.ones((3, 3), dtype=np.uint8)
    for z in range(int(arr.shape[0])):
        raw_plane = arr[z]
        out_plane = out[z]
        if raw_plane.size == 0:
            continue

        raw_cut = float(np.percentile(raw_plane, 98.8))
        imagej_cut = float(np.percentile(out_plane, 99.0))
        raw_seed = raw_plane >= raw_cut if np.isfinite(raw_cut) and raw_cut > 0.0 else np.zeros_like(raw_plane, dtype=bool)
        ring_seed = (
            out_plane >= max(imagej_cut, 0.04)
            if np.isfinite(imagej_cut) and imagej_cut > 0.0
            else np.zeros_like(out_plane, dtype=bool)
        )
        candidate = raw_seed | ring_seed
        if not np.any(candidate):
            continue

        candidate = ndi.binary_dilation(candidate, structure=structure, iterations=1)
        candidate = ndi.binary_closing(candidate, structure=np.ones((5, 5), dtype=bool), iterations=1)
        candidate = ndi.binary_fill_holes(candidate)

        labels, count = ndi.label(candidate, structure=structure)
        if count <= 0:
            continue

        sizes = np.bincount(labels.ravel(), minlength=int(count) + 1)
        dist = ndi.distance_transform_edt(candidate)
        max_dists = np.asarray(
            ndi.maximum(dist, labels=labels, index=np.arange(1, int(count) + 1)),
            dtype=np.float32,
        )
        keep_labels = np.flatnonzero((sizes[1:] >= 16) & (max_dists >= 2.0)) + 1
        restore_mask = np.isin(labels, keep_labels) if keep_labels.size > 0 else np.zeros_like(candidate, dtype=bool)

        if np.any(restore_mask):
            soma_floor = raw_norm[z] * 0.92
            out[z][restore_mask] = np.maximum(out_plane[restore_mask], soma_floor[restore_mask])

    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _refine_imagej_rolling_ball_microglia(
    raw: np.ndarray,
    imagej_enhanced: np.ndarray,
) -> np.ndarray:
    arr = np.clip(np.asarray(raw, dtype=np.float32), 0.0, 1.0)
    out = _restore_soma_interiors_after_imagej_rolling_ball(arr, imagej_enhanced)
    if arr.shape != out.shape or arr.ndim != 3 or arr.size == 0:
        return out

    for z in range(int(arr.shape[0])):
        raw_plane = arr[z]
        out_plane = out[z]
        raw_detail = _normalize_positive_detail(
            np.maximum(
                raw_plane - ndi.gaussian_filter(raw_plane, sigma=2.5, mode="nearest"),
                0.0,
            )
        )
        try:
            branch_response = sato(
                _normalize_positive_detail(raw_plane),
                sigmas=[0.7, 1.1, 1.6],
                black_ridges=False,
                mode="reflect",
            )
            branch = _normalize_positive_detail(branch_response)
        except Exception:
            branch = raw_detail
        branch = ndi.gaussian_filter(branch, sigma=0.45, mode="nearest")

        support_cut = float(np.percentile(out_plane, 88.0))
        if not np.isfinite(support_cut):
            support_cut = 0.02
        support = ndi.maximum_filter(out_plane, size=11, mode="nearest") > max(support_cut, 0.018)
        branch_mask = (branch > 0.18) & support & (raw_detail > 0.10)
        branch_labels, branch_count = ndi.label(branch_mask, structure=np.ones((3, 3), dtype=np.uint8))
        branch_keep = np.zeros_like(branch_mask, dtype=bool)
        if branch_count > 0:
            branch_sizes = np.bincount(branch_labels.ravel(), minlength=int(branch_count) + 1)
            keep_branch_labels = np.flatnonzero(branch_sizes >= 8)
            keep_branch_labels = keep_branch_labels[keep_branch_labels != 0]
            if keep_branch_labels.size > 0:
                branch_keep = np.isin(branch_labels, keep_branch_labels)
        if np.any(branch_keep):
            out[z][branch_keep] = np.maximum(
                out_plane[branch_keep],
                raw_detail[branch_keep] * 0.62,
            )

        raw_cut = float(np.percentile(raw_plane, 98.8))
        raw_bright = (
            raw_plane >= raw_cut
            if np.isfinite(raw_cut) and raw_cut > 0.0
            else np.zeros_like(raw_plane, dtype=bool)
        )
        raw_labels, raw_count = ndi.label(raw_bright, structure=np.ones((3, 3), dtype=np.uint8))
        soma_like = np.zeros_like(raw_bright, dtype=bool)
        if raw_count > 0:
            raw_sizes = np.bincount(raw_labels.ravel(), minlength=int(raw_count) + 1)
            raw_dist = ndi.distance_transform_edt(raw_bright)
            raw_max_dists = np.asarray(
                ndi.maximum(raw_dist, labels=raw_labels, index=np.arange(1, int(raw_count) + 1)),
                dtype=np.float32,
            )
            soma_labels = np.flatnonzero((raw_sizes[1:] >= 16) & (raw_max_dists >= 2.0)) + 1
            if soma_labels.size > 0:
                soma_like = np.isin(raw_labels, soma_labels)
        protect = ndi.binary_dilation(
            branch_keep | soma_like,
            structure=np.ones((5, 5), dtype=bool),
            iterations=1,
        )
        tiny_branch_like = branch_mask & (~branch_keep) & (raw_detail > 0.70) & (~soma_like)
        if np.any(tiny_branch_like):
            out[z][tiny_branch_like] *= 0.12
        unsupported_islands = (out[z] >= 0.04) & (~protect)
        island_labels, island_count = ndi.label(
            unsupported_islands,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        if island_count > 0:
            island_sizes = np.bincount(island_labels.ravel())
            small_islands = np.flatnonzero((island_sizes > 0) & (island_sizes < 20))
            small_islands = small_islands[small_islands != 0]
            if small_islands.size > 0:
                out[z][np.isin(island_labels, small_islands)] *= 0.12
        bright_cut = max(0.08, float(np.percentile(out[z], 99.0)))
        bright = (out[z] >= bright_cut) & (~protect)
        labels, count = ndi.label(bright, structure=np.ones((3, 3), dtype=np.uint8))
        if count > 0:
            sizes = np.bincount(labels.ravel())
            small = np.flatnonzero((sizes > 0) & (sizes < 8))
            small = small[small != 0]
            if small.size > 0:
                out[z][np.isin(labels, small)] *= 0.12

    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _basic_style_correction(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    dark = np.percentile(arr, 2.0, axis=0).astype(np.float32, copy=False)
    signal = np.maximum(arr - dark[np.newaxis, ...], 0.0)
    flat = np.percentile(signal, 70.0, axis=0).astype(np.float32, copy=False)
    flat = ndi.gaussian_filter(flat, sigma=24.0, mode="nearest")
    flat_mean = float(np.mean(flat[flat > 0])) if np.any(flat > 0) else 1.0
    corrected = signal / np.maximum(flat[np.newaxis, ...] / max(flat_mean, 1.0e-6), 0.15)
    return _normalize_positive_detail(corrected)


def _cidre_style_correction(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    dark = np.percentile(arr, 1.0, axis=0).astype(np.float32, copy=False)
    bright = np.percentile(arr, 92.0, axis=0).astype(np.float32, copy=False)
    shading = ndi.gaussian_filter(np.maximum(bright - dark, 0.0), sigma=32.0, mode="nearest")
    shading_mean = float(np.mean(shading[shading > 0])) if np.any(shading > 0) else 1.0
    corrected = np.maximum(arr - dark[np.newaxis, ...], 0.0)
    corrected = corrected / np.maximum(shading[np.newaxis, ...] / max(shading_mean, 1.0e-6), 0.15)
    return _normalize_positive_detail(corrected)


def _enhance_microglia_core(
    arr: np.ndarray,
    cfg: PreprocessConfig,
    method: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Remove fluorescence background while preserving microglia somas and branches."""
    sigma_xy = float(max(18.0, cfg.flatfield_sigma_xy))
    radius = int(np.clip(round(sigma_xy * 0.60), 8, 32))
    if method == "imagej_rolling_ball":
        enhanced = _imagej_rolling_ball_slicewise(arr, radius=5, multiplier=1.5)
        return _refine_imagej_rolling_ball_microglia(arr, enhanced)
    elif method == "microscopy_clean":
        cleaned = _microscopy_clean_background(
            arr, workers=_resolve_worker_threads(cfg), progress_callback=progress_callback
        )
        # Reinforce fragmented processes, then remove far-flung isolated debris.
        return _reconnect_and_denoise_microglia(cleaned)
    elif method == "basic":
        enhanced = _basic_style_correction(arr)
    elif method == "cidre":
        enhanced = _cidre_style_correction(arr)
    elif method == "white_tophat":
        enhanced = _normalize_positive_detail(_white_tophat_slicewise(arr, radius=radius))
    elif method == "clahe":
        background = ndi.gaussian_filter(arr, sigma=(0.0, sigma_xy, sigma_xy), mode="nearest")
        detail = _normalize_positive_detail(arr - (0.88 * background))
        if float(np.max(detail)) <= 1.0e-6:
            return arr.copy()
        contrast = _clahe_slicewise(detail, clip_limit=0.01)
        enhanced = np.maximum(detail, contrast * 0.75).astype(np.float32, copy=False)
    elif method == "microglia_preserve":
        enhanced = _tuned_microglia_preserve_background(arr)
        return enhanced
    else:
        raise ValueError(f"Unknown microglia enhancement method: {method}")

    if float(np.max(enhanced)) <= 1.0e-6:
        return arr.copy()

    branch_map = stage_branch_map_estimation(arr, cfg, "green")
    branch_protected = _apply_green_branch_preservation(
        arr,
        enhanced,
        branch_map,
        cfg,
        mode="post_background",
    )

    soma_cut = float(np.percentile(arr, 98.8))
    soma_mask = arr >= soma_cut if np.isfinite(soma_cut) and soma_cut > 0.0 else np.zeros_like(arr, dtype=bool)
    if np.any(soma_mask):
        soma_mask = ndi.maximum_filter(soma_mask.astype(np.uint8), size=(1, 5, 5), mode="nearest") > 0
        branch_protected[soma_mask] = np.maximum(branch_protected[soma_mask], arr[soma_mask] * 0.95)

    restored = stage_restore_branches_near_bright_pixels(arr, branch_protected)
    restored = stage_speckle_control(restored, branch_map, cfg, "green")
    if np.any(soma_mask):
        soma_floor = max(float(np.percentile(restored, 99.0)) * 0.94, 0.84)
        restored[soma_mask] = np.maximum(
            restored[soma_mask],
            np.maximum(arr[soma_mask] * 1.35, soma_floor),
        )
    return np.clip(restored, 0.0, 1.0).astype(np.float32, copy=False)


def enhance_microglia_background(
    volume: np.ndarray,
    config: PreprocessConfig | None = None,
    method: str = "microglia_preserve",
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    cfg = config or PreprocessConfig()
    arr = np.clip(np.asarray(volume, dtype=np.float32), 0.0, 1.0)
    if arr.ndim != 3:
        raise ValueError(f"Microglia volume must be 3D (z, y, x), got {arr.shape}")
    if arr.size == 0:
        return arr.copy()
    return _enhance_microglia_core(arr, cfg, method, progress_callback=progress_callback)


def enhance_vasculature_background(
    volume: np.ndarray,
    config: PreprocessConfig | None = None,
) -> np.ndarray:
    """Suppress red-channel background while retaining vessel-like detail.

    This deliberately avoids the soma/branch restoration used for microglia.
    A broad, per-slice background estimate removes uneven red fluorescence and
    a light tubeness-weighted blend keeps elongated vessel signal prominent
    without amplifying compact debris.
    """
    cfg = config or PreprocessConfig()
    arr = np.clip(np.asarray(volume, dtype=np.float32), 0.0, 1.0)
    if arr.ndim != 3:
        raise ValueError(f"Vasculature volume must be 3D (z, y, x), got {arr.shape}")
    if arr.size == 0:
        return arr.copy()

    sigma_xy = float(max(12.0, cfg.flatfield_sigma_xy))
    background = _gaussian_filter_fast(arr, sigma=(0.0, sigma_xy, sigma_xy), mode="nearest")
    detail = _normalize_positive_detail(np.maximum(arr - (0.90 * background), 0.0))
    vesselness = _tubeness_slicewise(
        detail,
        sigmas=[1.0, 1.8, 3.0, 4.5],
        workers=_resolve_worker_threads(cfg),
        progress_tag="vasculature enhancement",
    )
    vesselness = _normalize_branch_map(vesselness)
    # Keep broad, bright vessel trunks while making non-tubular background and
    # compact debris much less prominent. The dedicated blob wipe can then
    # remove those residual components deterministically.
    enhanced = detail * (0.30 + (0.70 * vesselness))
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32, copy=False)


def wipe_small_specks(
    volume: np.ndarray,
    *,
    threshold: float,
    min_voxels: int,
    connectivity: int = 3,
) -> np.ndarray:
    """Remove small isolated specks from a 3D channel volume.

    Voxels at or above ``threshold`` are grouped into connected components
    (26-connectivity by default). Any component with fewer than ``min_voxels``
    voxels is treated as a speck and zeroed out, while larger structures
    (vessels, microglia somas and branches) are left untouched. This backs the
    "Wipe Specks" action and is applied independently to the green (microglia)
    and red (vasculature) channels.
    """
    arr = np.asarray(volume, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Speck-wipe volume must be 3D (z, y, x), got {arr.shape}")
    if arr.size == 0:
        return arr.copy()

    t = float(np.clip(threshold, 0.0, 1.0))
    min_voxels = max(1, int(min_voxels))
    mask = arr >= t
    if not np.any(mask):
        return arr.copy()

    structure = ndi.generate_binary_structure(3, int(np.clip(connectivity, 1, 3)))
    labels, count = ndi.label(mask, structure=structure)
    if count <= 0:
        return arr.copy()

    sizes = np.bincount(labels.ravel())
    small_labels = np.where((sizes > 0) & (sizes < min_voxels))[0]
    small_labels = small_labels[small_labels != 0]
    if small_labels.size == 0:
        return arr.copy()

    speck_mask = np.isin(labels, small_labels)
    out = arr.copy()
    out[speck_mask] = 0.0
    logger.info(
        "Wiped %d specks (<%d voxels) of %d components at threshold %.3f (%d voxels cleared)",
        int(small_labels.size),
        min_voxels,
        int(count),
        t,
        int(np.count_nonzero(speck_mask)),
    )
    return out


def wipe_vasculature_blobs(
    volume: np.ndarray,
    *,
    threshold: float,
    max_voxels: int,
    max_aspect_ratio: float = 4.0,
    min_solid_voxels: int = 64,
    connectivity: int = 3,
) -> np.ndarray:
    """Remove isolated red-channel debris while preserving the vessel network.

    Compact connected components no larger than ``max_voxels`` are removed.
    Elongated components with an aspect ratio above ``max_aspect_ratio`` are
    kept as likely disconnected vessel fragments, matching the UI wording.
    Passing ``float("inf")`` restores pure size-only wiping.
    Because that ratio is meaningless for a handful of voxels (a 3-voxel diagonal
    reads as highly "elongated"), the guard never protects components at or below
    ``min_solid_voxels`` — those are always removed as noise.
    """
    arr = np.asarray(volume, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Vasculature blob-wipe volume must be 3D (z, y, x), got {arr.shape}")
    if arr.size == 0:
        return arr.copy()

    t = float(np.clip(threshold, 0.0, 1.0))
    limit = max(1, int(max_voxels))
    solid_floor = max(0, int(min_solid_voxels))
    # Finite ratio => keep elongated fragments; inf (default) => pure size wipe.
    size_only = not np.isfinite(float(max_aspect_ratio))
    mask = arr >= t
    if not np.any(mask):
        return arr.copy()

    structure = ndi.generate_binary_structure(3, int(np.clip(connectivity, 1, 3)))
    labels, count = ndi.label(mask, structure=structure)
    if count <= 0:
        return arr.copy()

    def _aspect_ratio(component: np.ndarray) -> float:
        coords = np.argwhere(component).astype(np.float32, copy=False)
        if coords.shape[0] < 3:
            return 1.0
        centered = coords - np.mean(coords, axis=0, keepdims=True)
        eigenvalues = np.linalg.eigvalsh(centered.T @ centered)
        eigenvalues.sort()
        return float(np.sqrt(eigenvalues[-1] / max(float(eigenvalues[-2]), 0.25)))

    # Size per component in one pass so large structures (vessels) are rejected
    # by the cheap size test before the eigen solve or any per-voxel slicing.
    sizes = np.bincount(labels.ravel())
    remove_labels: list[int] = []
    objects = ndi.find_objects(labels)
    for label_id, component_slice in enumerate(objects, start=1):
        if component_slice is None:
            continue
        size = int(sizes[label_id])
        if not (0 < size <= limit):
            continue
        if size_only or size <= solid_floor:
            # Size-only wipe, or too small for shape to mean anything: noise.
            remove_labels.append(label_id)
            continue
        component = labels[component_slice] == label_id
        if _aspect_ratio(component) <= float(max_aspect_ratio):
            remove_labels.append(label_id)

    if not remove_labels:
        return arr.copy()

    blob_mask = np.isin(labels, np.asarray(remove_labels, dtype=labels.dtype))

    # The strict cutoff separates blobs from vessels. Once selected, remove a
    # compact, dimmer halo too, otherwise a high-intensity blob leaves a visible
    # low-intensity shell in the mesh. A halo that connects to an elongated
    # vessel is retained rather than risking a vessel break.
    support_threshold = max(0.01, t * 0.5)
    support_labels, support_count = ndi.label(arr >= support_threshold, structure=structure)
    if support_count > 0:
        # Collect every support component that a removed blob sits inside in a
        # single pass. The previous per-blob `labels == label_id` mask was
        # O(n_blobs x voxels) and stalled on noisy channels with thousands of
        # blobs; this is O(voxels) plus a bounded per-component check.
        touched = np.unique(support_labels[blob_mask])
        touched = touched[touched != 0]
        if touched.size:
            support_sizes = np.bincount(support_labels.ravel())
            support_objects = ndi.find_objects(support_labels)
            support_remove: list[int] = []
            for support_id in touched.tolist():
                support_size = int(support_sizes[support_id])
                if not (0 < support_size <= limit):
                    continue
                if size_only or support_size <= solid_floor:
                    support_remove.append(support_id)
                    continue
                component_slice = support_objects[support_id - 1]
                if component_slice is None:
                    continue
                component = support_labels[component_slice] == support_id
                if _aspect_ratio(component) <= float(max_aspect_ratio):
                    support_remove.append(support_id)
            if support_remove:
                blob_mask |= np.isin(
                    support_labels,
                    np.asarray(support_remove, dtype=support_labels.dtype),
                )
    out = arr.copy()
    out[blob_mask] = 0.0
    logger.info(
        "Wiped %d compact vascular blobs (<=%d voxels, aspect<=%.2f) of %d components at threshold %.3f (%d voxels cleared)",
        len(remove_labels),
        limit,
        float(max_aspect_ratio),
        int(count),
        t,
        int(np.count_nonzero(blob_mask)),
    )
    return out


def preprocess_channel(channel: ChannelVolume, config: PreprocessConfig) -> ChannelVolume:
    if not config.enabled:
        return ChannelVolume(
            name=channel.name,
            data=np.asarray(channel.data, dtype=np.float32),
            z_indices=list(channel.z_indices),
            spacing=channel.spacing,
        )

    green_passthrough = channel.name == "green" and (
        config.green_denoise_strategy == "microglia_masking"
        or (
            config.green_denoise_strategy == "pixel2voxel_no_psf"
            and config.denoise_method == "wavelet"
        )
    )
    if green_passthrough:
        logger.info(
            "Preprocess[green] passthrough strategy=%s; returning input unchanged.",
            config.green_denoise_strategy,
        )
        return ChannelVolume(
            name=channel.name,
            data=np.asarray(channel.data, dtype=np.float32),
            z_indices=list(channel.z_indices),
            spacing=channel.spacing,
        )

    denoise_strength = float(config.denoise_strength)
    if channel.name == "green":
        denoise_strength *= float(max(0.1, config.green_denoise_multiplier))

    logger.info(
        "Preprocessing channel '%s' strategy=%s green_strategy=%s strength=%.5f worker_threads=%d",
        channel.name,
        config.denoise_method,
        config.green_denoise_strategy,
        denoise_strength,
        _resolve_worker_threads(config),
    )
    t0 = time.perf_counter()

    t = time.perf_counter()
    logger.info("Preprocess[%s] stage=illumination start", channel.name)
    working = stage_illumination_correction(channel.data, config)
    logger.info(
        "Preprocess[%s] stage=illumination dt=%.2fs %s",
        channel.name,
        time.perf_counter() - t,
        _fmt_stage_stats(working),
    )

    t = time.perf_counter()
    stabilized = stage_intensity_stabilization(working, config)
    logger.info(
        "Preprocess[%s] stage=intensity_stabilization dt=%.2fs %s",
        channel.name,
        time.perf_counter() - t,
        _fmt_stage_stats(stabilized),
    )

    t = time.perf_counter()
    noise = stage_noise_model_estimation(stabilized, config)
    logger.info(
        "Preprocess[%s] stage=noise_model dt=%.2fs sigma=%.5f model=%s",
        channel.name,
        time.perf_counter() - t,
        float(noise["sigma"]),
        str(noise["noise_model"]),
    )

    t = time.perf_counter()
    logger.info("Preprocess[%s] stage=branch_map start", channel.name)
    branch_map = stage_branch_map_estimation(stabilized, config, channel.name)
    logger.info(
        "Preprocess[%s] stage=branch_map dt=%.2fs nonzero=%.2f%%",
        channel.name,
        time.perf_counter() - t,
        100.0 * float(np.count_nonzero(branch_map)) / max(1, int(branch_map.size)),
    )

    t = time.perf_counter()
    logger.info("Preprocess[%s] stage=denoise_main start", channel.name)
    denoised, backend = stage_denoise_main(
        stabilized,
        channel.name,
        config=config,
        denoise_strength=denoise_strength,
        branch_map=branch_map,
    )
    logger.info(
        "Preprocess[%s] stage=denoise_main dt=%.2fs backend=%s sigma=%.5f noise_model=%s",
        channel.name,
        time.perf_counter() - t,
        backend,
        float(noise["sigma"]),
        str(noise["noise_model"]),
    )

    t = time.perf_counter()
    denoised = stage_speckle_control(denoised, branch_map, config, channel.name)
    logger.info(
        "Preprocess[%s] stage=speckle_control dt=%.2fs %s",
        channel.name,
        time.perf_counter() - t,
        _fmt_stage_stats(denoised),
    )
    logger.info(
        "Preprocess[%s] complete dt=%.2fs",
        channel.name,
        time.perf_counter() - t0,
    )
    return ChannelVolume(
        name=channel.name,
        data=np.asarray(denoised, dtype=np.float32),
        z_indices=list(channel.z_indices),
        spacing=channel.spacing,
    )


def preprocess_dataset(dataset: DatasetVolume, config: PreprocessConfig) -> DatasetVolume:
    t0 = time.perf_counter()
    green = preprocess_channel(dataset.green, config)
    red = preprocess_channel(dataset.red, config)
    logger.info("Preprocess dataset complete dt=%.2fs", time.perf_counter() - t0)
    return DatasetVolume(green=green, red=red, shared_z_range=dataset.shared_z_range)


def postprocess_green_after_deconvolution(
    dataset: DatasetVolume,
    config: PreprocessConfig,
) -> DatasetVolume:
    logger.info("Post-deconvolution green cleanup disabled; using green input unchanged.")
    return dataset


def suggest_green_threshold(volume: np.ndarray, fallback: float = 0.15) -> float:
    arr = np.asarray(volume, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(fallback)
    try:
        otsu = float(threshold_otsu(finite))
    except ValueError:
        otsu = float(fallback)
    branch_map = stage_branch_map_estimation(arr, PreprocessConfig(), "green")
    branch_vals = arr[branch_map > 0.35]
    if branch_vals.size == 0:
        return float(np.clip(otsu, 0.0, 1.0))
    floor = float(np.percentile(branch_vals, 35))
    result = min(otsu, floor * 1.2)
    # Keep the adaptive floor slightly above the noise regime so nearby dim
    # cells are less likely to merge into a single component.
    return float(np.clip(max(0.04, result), 0.0, 1.0))
