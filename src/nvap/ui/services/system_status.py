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


def gpu_status() -> SystemStatus:
    """Probe GPU rendering capability via VTK's render window."""
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
            return SystemStatus("GPU", "warn", "No capability report")
        first_line = renderer.splitlines()[0] if renderer else "GPU available"
        return SystemStatus("GPU", "good", first_line[:120])
    except Exception as exc:  # vtk not installed, render fails, etc.
        return SystemStatus("GPU", "warn", f"Software fallback ({type(exc).__name__})")


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


def runtime_status() -> SystemStatus:
    """Always-good "system ready" indicator – set after the UI completes init."""
    return SystemStatus("Ready", "good", "Idle")


def busy_status(message: str = "Working") -> SystemStatus:
    return SystemStatus("Busy", "warn", message)
