from __future__ import annotations

from functools import lru_cache
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _torch_runtime(preferred_backend: str = "auto"):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return None

    preferred = str(preferred_backend or "auto").strip().lower()
    if preferred not in {"auto", "cpu", "cuda", "rocm"}:
        preferred = "auto"
    if preferred == "cpu":
        return None
    if not torch.cuda.is_available():
        return None

    backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
    if preferred != "auto" and preferred != backend:
        return None
    return torch, F, torch.device("cuda"), backend


def detect_torch_gpu_backend(preferred_backend: str = "auto") -> str | None:
    runtime = _torch_runtime(preferred_backend)
    if runtime is None:
        return None
    return str(runtime[3])


def torch_gaussian_filter(
    volume: np.ndarray,
    sigma: tuple[float, ...],
    preferred_backend: str = "auto",
) -> np.ndarray | None:
    runtime = _torch_runtime(preferred_backend)
    if runtime is None:
        return None
    arr = np.asarray(volume, dtype=np.float32)
    if arr.ndim not in {2, 3}:
        return None

    torch, F, device, _backend = runtime
    sigma_vals = tuple(float(max(s, 0.0)) for s in sigma)
    try:
        if arr.ndim == 2:
            tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32)[None, None]
            for axis, sigma_val in enumerate(sigma_vals):
                if sigma_val <= 1.0e-6:
                    continue
                kernel = _gaussian_kernel1d(torch, sigma_val, device, tensor.dtype)
                radius = kernel.numel() // 2
                if axis == 0:
                    weight = kernel.view(1, 1, -1, 1)
                    pad = (0, 0, radius, radius)
                else:
                    weight = kernel.view(1, 1, 1, -1)
                    pad = (radius, radius, 0, 0)
                tensor = F.conv2d(F.pad(tensor, pad, mode="replicate"), weight)
            return np.asarray(tensor[0, 0].detach().to("cpu").numpy(), dtype=np.float32)

        tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32)[None, None]
        for axis, sigma_val in enumerate(sigma_vals):
            if sigma_val <= 1.0e-6:
                continue
            kernel = _gaussian_kernel1d(torch, sigma_val, device, tensor.dtype)
            radius = kernel.numel() // 2
            if axis == 0:
                weight = kernel.view(1, 1, -1, 1, 1)
                pad = (0, 0, 0, 0, radius, radius)
            elif axis == 1:
                weight = kernel.view(1, 1, 1, -1, 1)
                pad = (0, 0, radius, radius, 0, 0)
            else:
                weight = kernel.view(1, 1, 1, 1, -1)
                pad = (radius, radius, 0, 0, 0, 0)
            tensor = F.conv3d(F.pad(tensor, pad, mode="replicate"), weight)
        return np.asarray(tensor[0, 0].detach().to("cpu").numpy(), dtype=np.float32)
    except Exception as exc:
        logger.debug("Torch gaussian filter fallback to CPU: %s", exc)
        return None


def torch_uniform_filter(
    volume: np.ndarray,
    size: tuple[int, ...],
    preferred_backend: str = "auto",
) -> np.ndarray | None:
    runtime = _torch_runtime(preferred_backend)
    if runtime is None:
        return None
    arr = np.asarray(volume, dtype=np.float32)
    if arr.ndim not in {2, 3}:
        return None

    torch, F, device, _backend = runtime
    kernel_size = tuple(max(1, int(v)) for v in size)
    try:
        if arr.ndim == 2:
            tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32)[None, None]
            pad = (
                kernel_size[1] // 2,
                kernel_size[1] // 2,
                kernel_size[0] // 2,
                kernel_size[0] // 2,
            )
            out = F.avg_pool2d(F.pad(tensor, pad, mode="replicate"), kernel_size=kernel_size, stride=1)
            return np.asarray(out[0, 0].detach().to("cpu").numpy(), dtype=np.float32)

        tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32)[None, None]
        pad = (
            kernel_size[2] // 2,
            kernel_size[2] // 2,
            kernel_size[1] // 2,
            kernel_size[1] // 2,
            kernel_size[0] // 2,
            kernel_size[0] // 2,
        )
        out = F.avg_pool3d(F.pad(tensor, pad, mode="replicate"), kernel_size=kernel_size, stride=1)
        return np.asarray(out[0, 0].detach().to("cpu").numpy(), dtype=np.float32)
    except Exception as exc:
        logger.debug("Torch uniform filter fallback to CPU: %s", exc)
        return None


def torch_maximum_filter(
    volume: np.ndarray,
    size: tuple[int, ...],
    preferred_backend: str = "auto",
) -> np.ndarray | None:
    runtime = _torch_runtime(preferred_backend)
    if runtime is None:
        return None
    arr = np.asarray(volume, dtype=np.float32)
    if arr.ndim not in {2, 3}:
        return None

    torch, F, device, _backend = runtime
    kernel_size = tuple(max(1, int(v)) for v in size)
    try:
        if arr.ndim == 2:
            tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32)[None, None]
            pad = (
                kernel_size[1] // 2,
                kernel_size[1] // 2,
                kernel_size[0] // 2,
                kernel_size[0] // 2,
            )
            out = F.max_pool2d(F.pad(tensor, pad, mode="replicate"), kernel_size=kernel_size, stride=1)
            return np.asarray(out[0, 0].detach().to("cpu").numpy(), dtype=np.float32)

        tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32)[None, None]
        pad = (
            kernel_size[2] // 2,
            kernel_size[2] // 2,
            kernel_size[1] // 2,
            kernel_size[1] // 2,
            kernel_size[0] // 2,
            kernel_size[0] // 2,
        )
        out = F.max_pool3d(F.pad(tensor, pad, mode="replicate"), kernel_size=kernel_size, stride=1)
        return np.asarray(out[0, 0].detach().to("cpu").numpy(), dtype=np.float32)
    except Exception as exc:
        logger.debug("Torch maximum filter fallback to CPU: %s", exc)
        return None


def torch_tubeness_slicewise(
    volume: np.ndarray,
    sigmas: list[float],
    preferred_backend: str = "auto",
    batch_slices: int = 12,
) -> np.ndarray | None:
    runtime = _torch_runtime(preferred_backend)
    if runtime is None:
        return None
    arr = np.asarray(volume, dtype=np.float32)
    if arr.ndim != 3 or arr.size == 0:
        return None

    torch, F, device, backend = runtime
    logger.info("Torch tubeness backend=%s slices=%d sigmas=%s", backend, int(arr.shape[0]), sigmas)
    try:
        out = np.zeros_like(arr, dtype=np.float32)
        for start in range(0, int(arr.shape[0]), max(1, int(batch_slices))):
            end = min(int(arr.shape[0]), start + max(1, int(batch_slices)))
            batch = torch.from_numpy(arr[start:end]).to(device=device, dtype=torch.float32)[:, None]
            best = torch.zeros_like(batch)
            for sigma_val in sigmas:
                sigma_float = float(max(sigma_val, 0.25))
                blurred = _gaussian_blur_2d_batch(torch, F, batch, sigma_float)
                vesselness = _frangi_like_response(torch, F, blurred, sigma_float)
                best = torch.maximum(best, vesselness)
            out[start:end] = np.asarray(best[:, 0].detach().to("cpu").numpy(), dtype=np.float32)
        return out
    except Exception as exc:
        logger.debug("Torch tubeness fallback to CPU: %s", exc)
        return None


def _gaussian_kernel1d(torch, sigma: float, device, dtype):
    radius = max(1, int(math.ceil(3.0 * float(max(sigma, 1.0e-6)))))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coords / float(max(sigma, 1.0e-6))) ** 2)
    kernel /= torch.sum(kernel)
    return kernel


def _gaussian_blur_2d_batch(torch, F, batch, sigma: float):
    kernel = _gaussian_kernel1d(torch, sigma, batch.device, batch.dtype)
    radius = kernel.numel() // 2
    out = F.conv2d(
        F.pad(batch, (radius, radius, 0, 0), mode="replicate"),
        kernel.view(1, 1, 1, -1),
    )
    out = F.conv2d(
        F.pad(out, (0, 0, radius, radius), mode="replicate"),
        kernel.view(1, 1, -1, 1),
    )
    return out


def _frangi_like_response(torch, F, batch, sigma: float):
    kernel_xx = torch.tensor([[1.0, -2.0, 1.0]], device=batch.device, dtype=batch.dtype).view(1, 1, 1, 3)
    kernel_yy = torch.tensor([[1.0], [-2.0], [1.0]], device=batch.device, dtype=batch.dtype).view(1, 1, 3, 1)
    kernel_xy = torch.tensor(
        [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]],
        device=batch.device,
        dtype=batch.dtype,
    ).view(1, 1, 3, 3) * 0.25

    ixx = F.conv2d(F.pad(batch, (1, 1, 0, 0), mode="replicate"), kernel_xx) * (sigma ** 2)
    iyy = F.conv2d(F.pad(batch, (0, 0, 1, 1), mode="replicate"), kernel_yy) * (sigma ** 2)
    ixy = F.conv2d(F.pad(batch, (1, 1, 1, 1), mode="replicate"), kernel_xy) * (sigma ** 2)

    trace = ixx + iyy
    delta = torch.sqrt(torch.clamp((ixx - iyy) ** 2 + (4.0 * (ixy ** 2)), min=1.0e-12))
    l1 = 0.5 * (trace + delta)
    l2 = 0.5 * (trace - delta)

    swap = torch.abs(l1) > torch.abs(l2)
    small = torch.where(swap, l2, l1)
    large = torch.where(swap, l1, l2)

    beta = 0.5
    eps = torch.tensor(1.0e-6, device=batch.device, dtype=batch.dtype)
    rb = torch.abs(small) / torch.clamp(torch.abs(large), min=float(eps))
    s2 = (small ** 2) + (large ** 2)
    scale = torch.clamp(torch.amax(torch.sqrt(s2)), min=1.0e-3)
    vesselness = torch.exp(-(rb ** 2) / (2.0 * (beta ** 2))) * (1.0 - torch.exp(-s2 / (2.0 * (scale ** 2))))
    vesselness = vesselness * (large < 0.0).to(batch.dtype)
    return vesselness
