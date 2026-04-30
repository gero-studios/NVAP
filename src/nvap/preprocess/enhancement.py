from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import time

import numpy as np
import scipy.ndimage as ndi
from skimage import exposure
from skimage.filters import frangi, meijering, sato, threshold_otsu
from skimage.morphology import disk, white_tophat
from skimage.restoration import denoise_bilateral, denoise_nl_means, denoise_wavelet, rolling_ball

from nvap.config.types import ChannelVolume, DatasetVolume, PreprocessConfig
from nvap.preprocess.denoisers import (
    denoise_wavelet_3d,
    denoise_wavelet_slicewise,
    estimate_noise_sigma,
    run_green_denoiser,
)

logger = logging.getLogger(__name__)
_BRANCH_MAP_CHUNK_THRESHOLD = 64 * 1024 * 1024
_FAST_BG_VOXEL_THRESHOLD = 64 * 1024 * 1024


def _resolve_worker_threads(config: PreprocessConfig) -> int:
    requested = int(config.cpu_worker_threads)
    if requested > 0:
        return max(1, requested)
    cpus = os.cpu_count() or 1
    return max(1, min(8, cpus))


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
    norm = ndi.gaussian_filter(norm, sigma=(0.0, 0.75, 0.75), mode="nearest")
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
        bg_small = ndi.gaussian_filter(
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
        background = ndi.gaussian_filter(
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
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nvap-pre") as pool:
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


def stage_legacy_light_speckle_control(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    if arr.size == 0:
        return arr
    local_mean = ndi.uniform_filter(arr, size=(1, 3, 3), mode="nearest")
    delta = np.maximum(arr - local_mean, 0.0)
    cut = float(np.quantile(delta, 0.996))
    if not np.isfinite(cut) or cut <= 0.0:
        return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
    spike_mask = delta >= cut
    if not np.any(spike_mask):
        return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
    out = arr.copy()
    out[spike_mask] = local_mean[spike_mask] + (0.25 * delta[spike_mask])
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


MICROGLIA_ENHANCEMENT_METHODS: dict[str, str] = {
    "microglia_preserve": "Microglia-preserving combined",
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
) -> np.ndarray:
    """Remove fluorescence background while preserving microglia somas and branches."""
    sigma_xy = float(max(18.0, cfg.flatfield_sigma_xy))
    radius = int(np.clip(round(sigma_xy * 0.60), 8, 32))
    if method == "imagej_rolling_ball":
        enhanced = _imagej_rolling_ball_slicewise(arr, radius=5, multiplier=1.5)
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
        background = ndi.gaussian_filter(arr, sigma=(0.0, sigma_xy, sigma_xy), mode="nearest")
        bg_detail = _normalize_positive_detail(arr - (0.88 * background))
        if float(np.max(bg_detail)) <= 1.0e-6:
            return arr.copy()
        top_hat = _normalize_positive_detail(_white_tophat_slicewise(arr, radius=radius))
        contrast_src = np.maximum(bg_detail, top_hat * 0.85)
        contrast = _clahe_slicewise(contrast_src, clip_limit=0.012)
        enhanced = np.clip(
            (0.62 * bg_detail) + (0.25 * top_hat) + (0.13 * contrast),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
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
) -> np.ndarray:
    cfg = config or PreprocessConfig()
    arr = np.clip(np.asarray(volume, dtype=np.float32), 0.0, 1.0)
    if arr.ndim != 3:
        raise ValueError(f"Microglia volume must be 3D (z, y, x), got {arr.shape}")
    if arr.size == 0:
        return arr.copy()
    return _enhance_microglia_core(arr, cfg, method)


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
