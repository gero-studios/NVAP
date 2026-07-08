from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
import time
from typing import Callable

import imageio.v3 as iio
import numpy as np
import scipy.ndimage as ndi
from scipy import fft as sp_fft
from scipy.signal import fftconvolve

from nvap.config.types import PSFConfig, VoxelSpacing
from nvap.runtime_optimization import configured_cpu_workers

logger = logging.getLogger(__name__)


class OperationCanceledError(RuntimeError):
    """Raised when a long-running operation is canceled by the user."""


def _resolve_fft_workers() -> int:
    raw = os.environ.get("NVAP_PSF_FFT_WORKERS", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            logger.warning("Invalid NVAP_PSF_FFT_WORKERS=%r. Falling back to auto.", raw)
    cpus = configured_cpu_workers(os.cpu_count() or 1)
    # Leave headroom for channel-level parallelism.
    return max(1, min(8, cpus // 2 if cpus >= 4 else cpus))


def build_gaussian_psf(
    spacing: VoxelSpacing,
    sigma_xy_um: float,
    sigma_z_um: float,
    truncate: float = 3.0,
) -> np.ndarray:
    sigma_x = max(sigma_xy_um / spacing.x_um, 1e-6)
    sigma_y = max(sigma_xy_um / spacing.y_um, 1e-6)
    sigma_z = max(sigma_z_um / spacing.z_um, 1e-6)

    rx = max(1, int(np.ceil(truncate * sigma_x)))
    ry = max(1, int(np.ceil(truncate * sigma_y)))
    rz = max(1, int(np.ceil(truncate * sigma_z)))

    z, y, x = np.mgrid[-rz : rz + 1, -ry : ry + 1, -rx : rx + 1]
    exponent = -0.5 * ((x / sigma_x) ** 2 + (y / sigma_y) ** 2 + (z / sigma_z) ** 2)
    psf = np.exp(exponent).astype(np.float32)
    psf_sum = float(psf.sum())
    if psf_sum <= 0:
        raise ValueError("Invalid PSF kernel generated.")
    psf /= psf_sum
    logger.debug(
        "Built Gaussian PSF: shape=%s sigma_xy_um=%.4f sigma_z_um=%.4f",
        psf.shape,
        sigma_xy_um,
        sigma_z_um,
    )
    return psf


def _load_measured_psf_from_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            if "psf" in data:
                arr = data["psf"]
            else:
                first_key = list(data.keys())[0]
                arr = data[first_key]
    else:
        arr = iio.imread(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Measured PSF must be 3D, got shape={arr.shape}")
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("Measured PSF contains no positive signal.")
    arr /= total
    return arr


def _resolve_psf_kernel(config: PSFConfig, spacing: VoxelSpacing) -> np.ndarray:
    if config.use_measured_psf and config.measured_psf_path.strip():
        psf_path = Path(config.measured_psf_path).expanduser().resolve()
        if psf_path.exists():
            logger.info("Using measured PSF: %s", psf_path)
            return _load_measured_psf_from_file(psf_path)
        logger.warning("Measured PSF path not found: %s. Falling back to Gaussian PSF.", psf_path)

    return build_gaussian_psf(
        spacing=spacing,
        sigma_xy_um=config.sigma_xy_um,
        sigma_z_um=config.sigma_z_um,
    )


def _tv_regularize(volume: np.ndarray, weight: float, spacing: VoxelSpacing) -> np.ndarray:
    """Total Variation regularization step — preserves edges better than Gaussian."""
    arr = np.asarray(volume, dtype=np.float32)
    # Anisotropic TV via gradient magnitude weighting
    gz = np.diff(arr, axis=0, prepend=arr[:1]) / max(spacing.z_um, 0.01)
    gy = np.diff(arr, axis=1, prepend=arr[:, :1]) / max(spacing.y_um, 0.01)
    gx = np.diff(arr, axis=2, prepend=arr[:, :, :1]) / max(spacing.x_um, 0.01)

    grad_mag = np.sqrt(gz**2 + gy**2 + gx**2 + 1e-8)
    # Normalized gradient (edge-stopping)
    div_z = np.diff(gz / grad_mag, axis=0, append=gz[-1:] / grad_mag[-1:])
    div_y = np.diff(gy / grad_mag, axis=1, append=gy[:, -1:] / grad_mag[:, -1:])
    div_x = np.diff(gx / grad_mag, axis=2, append=gx[:, :, -1:] / grad_mag[:, :, -1:])

    divergence = div_z + div_y + div_x
    result = arr + weight * divergence
    return np.maximum(result, 0.0).astype(np.float32, copy=False)


def deconvolve_volume(
    volume: np.ndarray,
    spacing: VoxelSpacing,
    config: PSFConfig,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError("volume must be 3D.")

    if not config.enabled or config.iterations <= 0:
        logger.info(
            "PSF deconvolution skipped (enabled=%s iterations=%d).",
            config.enabled,
            config.iterations,
        )
        return volume.astype(np.float32, copy=True)

    logger.info(
        "Running Richardson-Lucy: iterations=%d sigma_xy_um=%.4f sigma_z_um=%.4f "
        "volume_shape=%s fft_workers=%d tv_reg=%s tv_weight=%.4f",
        config.iterations,
        config.sigma_xy_um,
        config.sigma_z_um,
        volume.shape,
        _resolve_fft_workers(),
        getattr(config, 'tv_regularization', False),
        getattr(config, 'tv_weight', 0.015),
    )
    start_time = time.perf_counter()
    psf = _resolve_psf_kernel(config, spacing)

    image = volume.astype(np.float32, copy=False)
    estimate = np.maximum(image.copy(), 1e-6)
    psf_mirror = psf[::-1, ::-1, ::-1]
    eps = np.float32(1e-7)
    total = int(config.iterations)
    reg = float(max(config.regularization_lambda, 0.0))
    use_tv = getattr(config, 'tv_regularization', False)
    tv_weight = float(getattr(config, 'tv_weight', 0.015))
    fft_workers = _resolve_fft_workers()

    for idx in range(total):
        if cancel_event is not None and cancel_event.is_set():
            logger.info("PSF deconvolution canceled at iteration %d/%d.", idx, total)
            raise OperationCanceledError("PSF deconvolution canceled by user.")

        with sp_fft.set_workers(fft_workers):
            conv = fftconvolve(estimate, psf, mode="same")
            relative_blur = image / np.maximum(conv, eps)
            estimate *= fftconvolve(relative_blur, psf_mirror, mode="same")

        # Regularization: TV or Gaussian smoothing
        if use_tv and tv_weight > 0:
            estimate = _tv_regularize(estimate, tv_weight, spacing)
        elif reg > 0.0:
            smooth = ndi.gaussian_filter(estimate, sigma=(0.3, 0.35, 0.35), mode="nearest")
            estimate = ((1.0 - reg) * estimate) + (reg * smooth)

        np.maximum(estimate, 0.0, out=estimate)

        if progress_callback is not None:
            progress_callback(idx + 1, total)
        elif ((idx + 1) % max(1, total // 5) == 0) or ((idx + 1) == total):
            logger.info("RL iteration progress %d/%d", idx + 1, total)

    deconv = estimate.astype(np.float32, copy=False)
    np.maximum(deconv, 0.0, out=deconv)
    logger.info("PSF deconvolution complete dt=%.2fs shape=%s", time.perf_counter() - start_time, deconv.shape)
    return deconv
