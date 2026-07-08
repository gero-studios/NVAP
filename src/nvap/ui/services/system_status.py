"""Lightweight system telemetry for the workspace header indicators.

Avoids fake "● GPU" labels by actually querying VTK + a tiny psutil/optional
fallback.  All probes are best-effort; failures degrade to a neutral status.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Status = Literal["good", "warn", "bad", "idle", "unknown"]


@dataclass(frozen=True)
class SystemStatus:
    label: str
    status: Status
    detail: str = ""


_gpu_status_cache: SystemStatus | None = None


def _compute_backend_label(compute_backend: str | None) -> str:
    backend = str(compute_backend or "").strip().lower()
    if backend == "cuda":
        return "GPU CUDA"
    if backend == "rocm":
        return "GPU ROCm"
    if backend == "directml":
        return "GPU DirectML"
    if backend == "mps":
        return "GPU MPS"
    return "CPU compute"


def gpu_status() -> SystemStatus:
    """Probe GPU rendering capability via VTK's render window.

    Result is cached after first call — GPU capabilities don't change at runtime.
    """
    global _gpu_status_cache
    if _gpu_status_cache is not None:
        return _gpu_status_cache
    compute_backend = None
    profile_detail = ""
    try:
        from nvap.runtime_optimization import plan_runtime_optimization

        profile = plan_runtime_optimization()
        compute_backend = None if profile.selected_backend == "cpu" else profile.selected_backend
        profile_detail = (
            f"backend={profile.selected_backend}; workers={profile.cpu_workers}; "
            f"numeric_threads={profile.numeric_threads}; memory={profile.memory_tier}"
        )
    except Exception:
        compute_backend = None
    label = _compute_backend_label(compute_backend)
    status: Status = "good" if compute_backend else "warn"
    try:
        import vtk  # noqa: F401  – just verify import works
        from vtk import vtkRenderWindow  # type: ignore

        rw = vtkRenderWindow()
        rw.SetOffScreenRendering(1)
        rw.SetSize(8, 8)
        rw.Render()
        renderer = rw.ReportCapabilities() or ""
        rw.Finalize()
        if not renderer:
            detail = "Render: no capability report"
            if compute_backend:
                detail = f"{detail}; compute={compute_backend}"
            else:
                detail = f"{detail}; compute=CPU fallback"
            if profile_detail:
                detail = f"{detail}; {profile_detail}"
            _gpu_status_cache = SystemStatus(label, status, detail)
        else:
            first_line = renderer.splitlines()[0] if renderer else "GPU available"
            detail = f"Render: {first_line[:120]}"
            if compute_backend:
                detail = f"{detail}; compute={compute_backend}"
            else:
                detail = f"{detail}; compute=CPU fallback"
            if profile_detail:
                detail = f"{detail}; {profile_detail}"
            _gpu_status_cache = SystemStatus(label, status, detail)
        return _gpu_status_cache
    except Exception as exc:  # vtk not installed, render fails, etc.
        detail = f"Render: software fallback ({type(exc).__name__})"
        if compute_backend:
            detail = f"{detail}; compute={compute_backend}"
            if profile_detail:
                detail = f"{detail}; {profile_detail}"
            _gpu_status_cache = SystemStatus(label, "good", detail)
            return _gpu_status_cache
        detail = f"{detail}; compute=CPU fallback"
        if profile_detail:
            detail = f"{detail}; {profile_detail}"
        _gpu_status_cache = SystemStatus(label, "warn", detail)
        return _gpu_status_cache


def memory_status() -> SystemStatus:
    """Report RAM headroom; psutil is optional and gracefully skipped."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return SystemStatus("Memory", "idle", "psutil not installed")
    try:
        vm = psutil.virtual_memory()
        free_gb = vm.available / (1024**3)
        pct = vm.percent
        if pct >= 90:
            level: Status = "bad"
        elif pct >= 75:
            level = "warn"
        else:
            level = "good"
        return SystemStatus("Memory", level, f"{free_gb:.1f} GB free ({100 - pct:.0f}%)")
    except Exception as exc:
        return SystemStatus("Memory", "unknown", str(exc))



