"""Denoising backends for NVAP — optimized for microglia branch preservation.

Key strategies:
- Wavelet (BayesShrink): fast, edge-preserving, default for green channel
- Classical branch-aware: wavelet + branch map blending
- BM4D / Noise2Void: optional high-quality backends
- Legacy anisotropic: Gaussian blur fallback (not recommended)
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
from pathlib import Path
import time
from typing import Callable

import numpy as np
import scipy.ndimage as ndi
from skimage.restoration import denoise_nl_means, denoise_wavelet

from nvap.config.types import PreprocessConfig

logger = logging.getLogger(__name__)
_CHUNK_NLM_VOXEL_THRESHOLD = 64 * 1024 * 1024
_FAST_CLASSICAL_VOXEL_THRESHOLD = 96 * 1024 * 1024


def _resolve_worker_threads(config: PreprocessConfig) -> int:
    requested = int(config.cpu_worker_threads)
    if requested > 0:
        return max(1, requested)
    cpus = os.cpu_count() or 1
    return max(1, min(8, cpus))


def _detect_gpu_backend() -> str:
    """Detect available GPU backend: 'rocm', 'cuda', or 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0).lower()
            if any(kw in device_name for kw in ("amd", "radeon", "gfx")):
                return "rocm"
            return "cuda"
        if hasattr(torch.version, "hip"):
            return "rocm"
    except ImportError:
        pass
    return "cpu"


class BackendUnavailableError(RuntimeError):
    """Raised when an optional denoising backend cannot run."""


def estimate_noise_sigma(volume: np.ndarray) -> float:
    """Robust sigma estimate via MAD on high-pass residuals."""
    arr = np.asarray(volume, dtype=np.float32)
    if arr.size >= 32 * 1024 * 1024 and arr.ndim == 3:
        z = arr.shape[0]
        target_slices = min(12, z)
        step = max(1, z // target_slices)
        sample = arr[::step][:target_slices]
        sample = sample[:, ::2, ::2]
        arr = sample
    smooth = ndi.gaussian_filter(arr, sigma=(0.0, 1.2, 1.2), mode="nearest")
    residual = arr - smooth
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    sigma = mad / 0.6745 if mad > 0 else 0.0
    return float(max(sigma, 1.0e-4))


def anscombe_forward(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    return 2.0 * np.sqrt(np.clip(arr, 0.0, None) + 3.0 / 8.0)


def anscombe_inverse(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    return np.clip((arr * 0.5) ** 2 - 3.0 / 8.0, 0.0, None)


# ---------------------------------------------------------------------------
# Wavelet denoising — fast, edge & branch preserving
# ---------------------------------------------------------------------------

def denoise_wavelet_3d(
    volume: np.ndarray,
    sigma: float,
    config: PreprocessConfig,
    strength: float = 1.0,
) -> np.ndarray:
    """3D wavelet denoising with BayesShrink — preserves edges and fine branches."""
    arr = np.asarray(volume, dtype=np.float32)
    wavelet = config.wavelet_name or "db4"
    level = config.wavelet_level if config.wavelet_level > 0 else None
    effective_sigma = float(sigma * np.clip(0.6 + strength * 0.6, 0.4, 1.5))

    denoised = denoise_wavelet(
        arr,
        sigma=effective_sigma,
        wavelet=wavelet,
        mode="soft",
        wavelet_levels=level,
        method="BayesShrink",
        rescale_sigma=True,
        channel_axis=None,
    )
    return np.asarray(denoised, dtype=np.float32)


def denoise_wavelet_slicewise(
    volume: np.ndarray,
    sigma: float,
    config: PreprocessConfig,
    strength: float = 1.0,
    workers: int = 1,
) -> np.ndarray:
    """Slice-by-slice wavelet denoising for large volumes."""
    arr = np.asarray(volume, dtype=np.float32)
    wavelet = config.wavelet_name or "db4"
    effective_sigma = float(sigma * np.clip(0.6 + strength * 0.6, 0.4, 1.5))

    out = np.empty_like(arr, dtype=np.float32)

    def _denoise_slice(z: int) -> tuple[int, np.ndarray]:
        result = denoise_wavelet(
            arr[z],
            sigma=effective_sigma,
            wavelet=wavelet,
            mode="soft",
            method="BayesShrink",
            rescale_sigma=True,
            channel_axis=None,
        )
        return z, np.asarray(result, dtype=np.float32)

    depth = arr.shape[0]
    if workers > 1 and depth > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nvap-wav") as pool:
            for z, result in pool.map(_denoise_slice, range(depth)):
                out[z] = result
    else:
        for z in range(depth):
            _, result = _denoise_slice(z)
            out[z] = result

    # Light Z-axis smoothing for inter-slice continuity
    out = ndi.gaussian_filter(out, sigma=(0.3, 0.0, 0.0), mode="nearest")
    return out


# ---------------------------------------------------------------------------
# Legacy / baseline
# ---------------------------------------------------------------------------

def denoise_legacy_anisotropic(volume: np.ndarray, denoise_strength: float) -> np.ndarray:
    sigma_xy = float(max(denoise_strength * 24.0, 0.15))
    sigma_z = sigma_xy * 0.55
    out = ndi.gaussian_filter(
        np.asarray(volume, dtype=np.float32),
        sigma=(sigma_z, sigma_xy, sigma_xy),
        mode="nearest",
    )
    return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Chunked processing
# ---------------------------------------------------------------------------

def _chunk_slices(depth: int, chunk_depth: int, overlap: int) -> list[tuple[int, int, int, int]]:
    if depth <= chunk_depth or chunk_depth <= 0:
        return [(0, depth, 0, depth)]
    segments: list[tuple[int, int, int, int]] = []
    step = max(1, chunk_depth - (2 * max(0, overlap)))
    start = 0
    while start < depth:
        end = min(depth, start + chunk_depth)
        keep_start = start + (0 if start == 0 else overlap)
        keep_end = end - (0 if end == depth else overlap)
        segments.append((start, end, keep_start, keep_end))
        if end >= depth:
            break
        start += step
    return segments


def run_in_chunks(
    volume: np.ndarray,
    fn: Callable[[np.ndarray], np.ndarray],
    chunk_depth: int,
    overlap: int,
    workers: int = 1,
) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    depth = int(arr.shape[0])
    pieces = _chunk_slices(depth, int(chunk_depth), int(overlap))
    if len(pieces) == 1:
        return np.asarray(fn(arr), dtype=np.float32)

    logger.info(
        "Chunked denoise: depth=%d chunk_depth=%d overlap=%d chunks=%d",
        depth, int(chunk_depth), int(overlap), len(pieces),
    )
    out = np.zeros_like(arr, dtype=np.float32)
    weights = np.zeros_like(arr, dtype=np.float32)

    if workers > 1 and len(pieces) > 1:
        def _run_piece(piece):
            start, end, _, _ = piece
            t0 = time.perf_counter()
            den = np.asarray(fn(arr[start:end]), dtype=np.float32)
            return piece, den, time.perf_counter() - t0

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nvap-denoise") as pool:
            futures = [pool.submit(_run_piece, piece) for piece in pieces]
            done = 0
            for future in as_completed(futures):
                piece, denoised, dt = future.result()
                start, end, keep_start, keep_end = piece
                out[keep_start:keep_end] += denoised[keep_start - start:keep_end - start]
                weights[keep_start:keep_end] += 1.0
                done += 1
                logger.info("Chunked denoise %d/%d z=[%d,%d) dt=%.2fs", done, len(pieces), start, end, dt)
    else:
        for idx, (start, end, keep_start, keep_end) in enumerate(pieces, start=1):
            t0 = time.perf_counter()
            denoised = np.asarray(fn(arr[start:end]), dtype=np.float32)
            out[keep_start:keep_end] += denoised[keep_start - start:keep_end - start]
            weights[keep_start:keep_end] += 1.0
            logger.info("Chunked denoise %d/%d z=[%d,%d) dt=%.2fs", idx, len(pieces), start, end, time.perf_counter() - t0)

    weights = np.maximum(weights, 1.0)
    return out / weights


# ---------------------------------------------------------------------------
# Branch-aware denoising — core green-channel strategy
# ---------------------------------------------------------------------------

def denoise_classical_branch_aware(
    volume: np.ndarray,
    branch_map: np.ndarray,
    config: PreprocessConfig,
    denoise_strength: float,
) -> np.ndarray:
    """Branch-aware denoising using wavelet + adaptive blending.

    Wavelet denoising preserves edges far better than NLM or Gaussian.
    Branch map guides spatial adaptation: mild on branches, strong on BG.
    """
    arr = np.asarray(volume, dtype=np.float32)
    branch = np.clip(np.asarray(branch_map, dtype=np.float32), 0.0, 1.0)

    noise_sigma = estimate_noise_sigma(arr)
    working = arr

    if config.green_apply_vst and config.green_noise_model != "gaussian":
        working = anscombe_forward(working)
        noise_sigma = estimate_noise_sigma(working)

    t0 = time.perf_counter()
    workers = _resolve_worker_threads(config)
    use_fast = arr.size >= _FAST_CLASSICAL_VOXEL_THRESHOLD or config.green_denoise_strategy == "hybrid_auto"

    if use_fast:
        strong = denoise_wavelet_slicewise(working, noise_sigma, config, strength=1.0 + denoise_strength, workers=workers)
        mild = denoise_wavelet_slicewise(working, noise_sigma, config, strength=0.3, workers=workers)
    else:
        strong = denoise_wavelet_3d(working, noise_sigma, config, strength=1.0 + denoise_strength)
        mild = denoise_wavelet_3d(working, noise_sigma, config, strength=0.3)

    blended = (branch * mild) + ((1.0 - branch) * strong)

    if config.green_apply_vst and config.green_noise_model != "gaussian":
        blended = anscombe_inverse(blended)

    # Very light post-smoothing for residual wavelet artifacts
    post = ndi.gaussian_filter(
        np.asarray(blended, dtype=np.float32),
        sigma=(0.08, 0.15, 0.15),
        mode="nearest",
    )

    logger.info(
        "Branch-aware denoise complete dt=%.2fs fast=%s sigma=%.4f",
        time.perf_counter() - t0, use_fast, noise_sigma,
    )
    return np.clip(post, 0.0, 1.0).astype(np.float32, copy=False)


def denoise_pixel2voxel_no_psf(
    volume: np.ndarray,
    branch_map: np.ndarray,
    config: PreprocessConfig,
    denoise_strength: float,
) -> np.ndarray:
    """Pixel-to-voxel denoising model for no-PSF workflows.

    Combines:
    - slice-wise denoising (preserves thin branch details),
    - volumetric denoising (suppresses background noise), and
    - a branch/confidence-guided blend with light axial smoothing.
    """
    arr = np.asarray(volume, dtype=np.float32)
    branch = np.clip(np.asarray(branch_map, dtype=np.float32), 0.0, 1.0)
    if branch.shape != arr.shape:
        branch = np.zeros_like(arr, dtype=np.float32)

    noise_sigma = estimate_noise_sigma(arr)
    workers = _resolve_worker_threads(config)
    t0 = time.perf_counter()

    slice_strength = float(np.clip(0.55 + (denoise_strength * 0.65), 0.45, 1.35))
    voxel_strength = float(np.clip(0.85 + denoise_strength, 0.65, 1.6))
    slicewise = denoise_wavelet_slicewise(arr, noise_sigma, config, strength=slice_strength, workers=workers)
    volumetric = denoise_wavelet_3d(arr, noise_sigma, config, strength=voxel_strength)
    axial = ndi.gaussian_filter(volumetric, sigma=(0.65, 0.0, 0.0), mode="nearest")

    # Estimate how coherent signal is across neighboring slices.
    z_delta = np.abs(np.diff(arr, axis=0, prepend=arr[:1]))
    z_consistency = np.exp(-z_delta / float(max(noise_sigma * 2.5, 1.0e-4)))
    z_consistency = ndi.gaussian_filter(z_consistency, sigma=(0.45, 0.7, 0.7), mode="nearest")

    # Protect branch-like structures with higher slice-wise contribution.
    slice_weight = np.clip(0.2 + (0.65 * branch) + (0.25 * (1.0 - z_consistency)), 0.2, 0.95)
    fused = (slice_weight * slicewise) + ((1.0 - slice_weight) * axial)

    # Background-only extra suppression to reduce speckle without thinning branches.
    bg_weight = 1.0 - branch
    bg_suppressed = ndi.median_filter(fused, size=(1, 3, 3), mode="nearest")
    fused = (branch * fused) + (bg_weight * ((0.74 * fused) + (0.26 * bg_suppressed)))
    fused = np.clip(fused, 0.0, 1.0).astype(np.float32, copy=False)

    logger.info(
        "Pixel2Voxel denoise complete dt=%.2fs sigma=%.4f workers=%d",
        time.perf_counter() - t0,
        noise_sigma,
        workers,
    )
    return fused


# ---------------------------------------------------------------------------
# Optional high-quality backends
# ---------------------------------------------------------------------------

def _denoise_bm4d_impl(volume: np.ndarray, sigma: float) -> np.ndarray:
    try:
        import bm4d  # type: ignore
    except ImportError as exc:
        raise BackendUnavailableError("bm4d package not installed.") from exc

    arr = np.asarray(volume, dtype=np.float32)
    try:
        denoised = bm4d.bm4d(arr, sigma_psd=float(max(sigma, 1.0e-4)))
    except Exception as exc:
        raise BackendUnavailableError(f"bm4d backend failed: {exc}") from exc
    return np.asarray(denoised, dtype=np.float32)


def _denoise_noise2void_torch(volume: np.ndarray, model_path: str) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError("torch is not installed for noise2void backend.") from exc

    model_file = Path(model_path).expanduser().resolve()
    if not model_file.exists():
        raise BackendUnavailableError(f"noise2void model not found: {model_file}")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.jit.load(str(model_file), map_location=device)
        model.eval()
    except Exception as exc:
        raise BackendUnavailableError(f"Failed to load noise2void model: {exc}") from exc

    arr = np.asarray(volume, dtype=np.float32)
    gpu_backend = _detect_gpu_backend()
    logger.info("Noise2Void: device=%s gpu_backend=%s", device, gpu_backend)

    with torch.inference_mode():
        tensor = torch.from_numpy(arr[None, None]).to(device=device, dtype=torch.float32)
        output = model(tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        out = output.detach().to("cpu").numpy()[0, 0]
    return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def run_green_denoiser(
    volume: np.ndarray,
    branch_map: np.ndarray,
    config: PreprocessConfig,
    denoise_strength: float,
) -> tuple[np.ndarray, str]:
    """Route green channel through the configured denoising strategy."""
    arr = np.asarray(volume, dtype=np.float32)
    strategy = config.green_denoise_strategy
    logger.info(
        "Green denoiser: strategy=%s shape=%s chunked=%s",
        strategy, arr.shape, config.green_chunked_processing,
    )

    def run_classical(v: np.ndarray) -> np.ndarray:
        return denoise_classical_branch_aware(
            v,
            branch_map=np.clip(branch_map[:v.shape[0]], 0.0, 1.0),
            config=config,
            denoise_strength=denoise_strength,
        )

    if strategy == "legacy_anisotropic":
        logger.info("Green denoiser backend=legacy_anisotropic")
        return denoise_legacy_anisotropic(arr, denoise_strength=denoise_strength), "legacy_anisotropic"

    if strategy == "classical_branch_aware":
        logger.info("Green denoiser backend=classical_branch_aware")
        return run_classical(arr), "classical_branch_aware"

    if strategy == "pixel2voxel_no_psf":
        denoised = denoise_pixel2voxel_no_psf(
            arr,
            branch_map=branch_map,
            config=config,
            denoise_strength=denoise_strength,
        )
        logger.info("Green denoiser backend=pixel2voxel_no_psf")
        return denoised, "pixel2voxel_no_psf"

    if strategy == "noise2void":
        try:
            if not config.green_noise2void_model_path.strip():
                raise BackendUnavailableError("noise2void model path not configured.")
            denoised = _denoise_noise2void_torch(arr, config.green_noise2void_model_path)
            logger.info("Green denoiser backend=noise2void")
            return np.clip(denoised, 0.0, 1.0).astype(np.float32, copy=False), "noise2void"
        except BackendUnavailableError as exc:
            logger.warning("%s Falling back to classical_branch_aware.", exc)
            return run_classical(arr), "classical_branch_aware(fallback)"

    if strategy == "bm4d":
        try:
            sigma = estimate_noise_sigma(arr)
            if config.green_chunked_processing:
                denoised = run_in_chunks(
                    arr,
                    lambda chunk: _denoise_bm4d_impl(chunk, sigma=sigma),
                    chunk_depth=config.green_chunk_depth,
                    overlap=config.green_chunk_overlap,
                )
            else:
                denoised = _denoise_bm4d_impl(arr, sigma=sigma)
            logger.info("Green denoiser backend=bm4d")
            return np.clip(denoised, 0.0, 1.0).astype(np.float32, copy=False), "bm4d"
        except BackendUnavailableError as exc:
            logger.warning("%s Falling back to classical_branch_aware.", exc)
            return run_classical(arr), "classical_branch_aware(fallback)"

    # hybrid_auto
    try:
        try:
            import torch
        except ImportError:
            torch = None  # type: ignore

        if (
            torch is not None
            and bool(torch.cuda.is_available())
            and config.green_noise2void_model_path.strip()
        ):
            denoised = _denoise_noise2void_torch(arr, config.green_noise2void_model_path)
            logger.info("Green denoiser backend=noise2void (hybrid_auto)")
            return np.clip(denoised, 0.0, 1.0).astype(np.float32, copy=False), "noise2void"

        manageable_voxels = 512 * 512 * 160
        if arr.size <= manageable_voxels:
            sigma = estimate_noise_sigma(arr)
            denoised = _denoise_bm4d_impl(arr, sigma=sigma)
            logger.info("Green denoiser backend=bm4d (hybrid_auto)")
            return np.clip(denoised, 0.0, 1.0).astype(np.float32, copy=False), "bm4d"
    except BackendUnavailableError as exc:
        logger.info("Hybrid backend unavailable: %s", exc)

    logger.info("Green denoiser backend=classical_branch_aware (hybrid_auto fallback)")
    return run_classical(arr), "classical_branch_aware"
