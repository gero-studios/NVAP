from __future__ import annotations

from functools import lru_cache
import logging
import math
import os

import numpy as np

logger = logging.getLogger(__name__)
_VALID_BACKENDS = {"auto", "cpu", "cuda", "rocm", "directml", "mps"}


def _normalize_backend(value: str | None, default: str = "auto") -> str:
    backend = str(value or default).strip().lower()
    return backend if backend in _VALID_BACKENDS else default


def _backend_candidates(preferred_backend: str) -> tuple[str, ...]:
    preferred = _normalize_backend(preferred_backend)
    if preferred == "cpu":
        return ()
    if preferred != "auto":
        return (preferred,)
    # CUDA covers NVIDIA CUDA and AMD ROCm builds of PyTorch. DirectML is the
    # practical automatic Windows path for many AMD/Intel GPUs. MPS is Apple.
    return ("cuda", "rocm", "directml", "mps")


def _runtime_supports_filters(torch, F, device, backend: str) -> bool:
    """Probe the torch ops NVAP uses before selecting a GPU backend."""
    try:
        with torch.inference_mode():
            x2 = torch.ones((1, 1, 4, 4), device=device, dtype=torch.float32)
            k2 = torch.ones((1, 1, 1, 1), device=device, dtype=torch.float32)
            y2 = F.conv2d(F.pad(x2, (1, 1, 1, 1), mode="replicate"), k2)
            _ = float(y2.detach().to("cpu").sum())

            if backend == "directml":
                x3 = torch.ones((3, 3, 3), device=device, dtype=torch.float32)
                y3 = _gaussian_filter_3d_via_conv2d(torch, F, x3, (1.0, 1.0, 1.0), device)
                _ = float(y3.detach().to("cpu").sum())
                return True

            x3 = torch.ones((1, 1, 3, 3, 3), device=device, dtype=torch.float32)
            k3 = torch.ones((1, 1, 1, 1, 1), device=device, dtype=torch.float32)
            y3 = F.conv3d(F.pad(x3, (1, 1, 1, 1, 1, 1), mode="replicate"), k3)
            _ = float(y3.detach().to("cpu").sum())
        return True
    except Exception as exc:
        logger.info("Ignoring GPU backend %s: required filter probe failed (%s).", backend, exc)
        return False


@lru_cache(maxsize=4)
def _torch_runtime(preferred_backend: str = "auto"):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return None

    for candidate in _backend_candidates(preferred_backend):
        if candidate in {"cuda", "rocm"}:
            if not torch.cuda.is_available():
                continue
            backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
            if candidate != backend:
                continue
            device = torch.device("cuda")
        elif candidate == "directml":
            try:
                import torch_directml  # type: ignore

                device = torch_directml.device()
                backend = "directml"
            except Exception as exc:
                logger.debug("Torch DirectML runtime unavailable: %s", exc)
                continue
        elif candidate == "mps":
            try:
                mps = getattr(getattr(torch, "backends", None), "mps", None)
                if mps is None or not mps.is_available():
                    continue
                device = torch.device("mps")
                backend = "mps"
            except Exception:
                continue
        else:
            continue

        if _runtime_supports_filters(torch, F, device, backend):
            logger.info("Selected GPU compute backend: %s", backend)
            return torch, F, device, backend

    return None


def preferred_acceleration_backend(default: str = "auto") -> str:
    return _normalize_backend(os.environ.get("NVAP_GPU_BACKEND"), default=default)


def detect_torch_gpu_backend(preferred_backend: str = "auto") -> str | None:
    if preferred_backend == "auto":
        preferred_backend = preferred_acceleration_backend("auto")
    runtime = _torch_runtime(preferred_backend)
    if runtime is None:
        return None
    return str(runtime[3])


def torch_gaussian_filter(
    volume: np.ndarray,
    sigma: tuple[float, ...],
    preferred_backend: str = "auto",
) -> np.ndarray | None:
    if preferred_backend == "auto":
        preferred_backend = preferred_acceleration_backend("auto")
    runtime = _torch_runtime(preferred_backend)
    if runtime is None:
        return None
    arr = np.asarray(volume, dtype=np.float32)
    if arr.ndim not in {2, 3}:
        return None

    torch, F, device, backend = runtime
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

        if backend == "directml":
            tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
            tensor = _gaussian_filter_3d_via_conv2d(torch, F, tensor, sigma_vals, device)
            return np.asarray(tensor.detach().to("cpu").numpy(), dtype=np.float32)

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





def _gaussian_kernel1d(torch, sigma: float, device, dtype):
    radius = max(1, int(math.ceil(3.0 * float(max(sigma, 1.0e-6)))))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coords / float(max(sigma, 1.0e-6))) ** 2)
    kernel /= torch.sum(kernel)
    return kernel


def _gaussian_filter_3d_via_conv2d(torch, F, tensor, sigma_vals: tuple[float, ...], device):
    """Exact separable 3D Gaussian using DirectML-supported 2D convolutions."""
    out = tensor
    for axis, sigma_val in enumerate(sigma_vals[:3]):
        if sigma_val <= 1.0e-6:
            continue
        kernel = _gaussian_kernel1d(torch, sigma_val, device, out.dtype)
        radius = kernel.numel() // 2
        if axis == 0:
            z, y, x = out.shape
            columns = out.reshape(1, 1, z, y * x)
            out = F.conv2d(
                F.pad(columns, (0, 0, radius, radius), mode="replicate"),
                kernel.view(1, 1, -1, 1),
            ).reshape(z, y, x)
        elif axis == 1:
            slices = out[:, None]
            out = F.conv2d(
                F.pad(slices, (0, 0, radius, radius), mode="replicate"),
                kernel.view(1, 1, -1, 1),
            )[:, 0]
        else:
            slices = out[:, None]
            out = F.conv2d(
                F.pad(slices, (radius, radius, 0, 0), mode="replicate"),
                kernel.view(1, 1, 1, -1),
            )[:, 0]
    return out



