from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import logging
import os
import platform
from functools import lru_cache

logger = logging.getLogger(__name__)

_VALID_BACKENDS = {"auto", "cpu", "cuda", "rocm", "directml", "mps"}


@dataclass(frozen=True)
class RuntimeOptimization:
    requested_backend: str
    selected_backend: str
    gpu_available: bool
    cpu_count: int
    cpu_workers: int
    numeric_threads: int
    total_memory_gb: float | None
    memory_tier: str
    platform_label: str
    notes: tuple[str, ...] = ()

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def normalize_backend(value: str | None, default: str = "auto") -> str:
    backend = str(value or default).strip().lower()
    return backend if backend in _VALID_BACKENDS else default


def _total_memory_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().total / (1024**3))
    except Exception:
        return None


def _memory_tier(total_gb: float | None) -> str:
    if total_gb is None:
        return "unknown"
    if total_gb < 8.0:
        return "low"
    if total_gb < 24.0:
        return "standard"
    return "large"


def recommended_cpu_workers(cpu_count: int | None = None, memory_gb: float | None = None) -> int:
    cpus = max(1, int(cpu_count or os.cpu_count() or 1))
    # Leave at least one core for the UI/OS and avoid oversubscribing scipy,
    # VTK, and the shared preprocessing executor on large workstations.
    worker_cap = 12 if cpus >= 16 else 8
    workers = max(1, min(worker_cap, cpus - 1 if cpus > 2 else cpus))
    if memory_gb is not None and memory_gb < 8.0:
        workers = min(workers, 2)
    return int(workers)


def recommended_numeric_threads(cpu_count: int | None = None, cpu_workers: int | None = None) -> int:
    cpus = max(1, int(cpu_count or os.cpu_count() or 1))
    workers = max(1, int(cpu_workers or recommended_cpu_workers(cpus)))
    # Numeric libraries can spawn their own threads; keep them small so parallel
    # slice-level work does not multiply into dozens of runnable threads.
    if workers >= max(1, cpus // 2):
        return 1
    return max(1, min(4, cpus // max(1, workers)))


@lru_cache(maxsize=8)
def plan_runtime_optimization(
    requested_backend: str = "auto",
    requested_cpu_workers: int | None = None,
) -> RuntimeOptimization:
    requested = normalize_backend(requested_backend, default="auto")
    memory_gb = _total_memory_gb()
    cpus = max(1, int(os.cpu_count() or 1))
    workers = (
        max(1, int(requested_cpu_workers))
        if requested_cpu_workers is not None and int(requested_cpu_workers) > 0
        else recommended_cpu_workers(cpus, memory_gb)
    )
    numeric_threads = recommended_numeric_threads(cpus, workers)
    notes: list[str] = []
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        os.environ.setdefault(name, str(numeric_threads))

    if requested == "auto":
        from nvap.accelerate import preferred_acceleration_backend

        env_backend = preferred_acceleration_backend("auto")
    else:
        env_backend = requested

    if env_backend == "cpu":
        gpu_backend = None
        notes.append("CPU backend forced by configuration.")
    else:
        from nvap.accelerate import detect_torch_gpu_backend

        gpu_backend = detect_torch_gpu_backend(env_backend)
        if gpu_backend is None:
            notes.append("No usable Torch GPU backend detected; CPU path selected.")
        elif env_backend != "auto" and gpu_backend != env_backend:
            notes.append(f"Requested {env_backend}, selected {gpu_backend}.")

    selected = gpu_backend or "cpu"
    return RuntimeOptimization(
        requested_backend=requested,
        selected_backend=selected,
        gpu_available=bool(gpu_backend),
        cpu_count=cpus,
        cpu_workers=workers,
        numeric_threads=numeric_threads,
        total_memory_gb=memory_gb,
        memory_tier=_memory_tier(memory_gb),
        platform_label=f"{platform.system()} {platform.machine()} Python {platform.python_version()}",
        notes=tuple(notes),
    )


def configure_runtime_environment(
    *,
    requested_backend: str = "auto",
    requested_cpu_workers: int | None = None,
) -> RuntimeOptimization:
    profile = plan_runtime_optimization(requested_backend, requested_cpu_workers)
    os.environ["NVAP_GPU_BACKEND"] = profile.selected_backend
    os.environ["NVAP_CPU_WORKERS"] = str(profile.cpu_workers)
    os.environ["NVAP_NUMERIC_THREADS"] = str(profile.numeric_threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        os.environ.setdefault(name, str(profile.numeric_threads))
    logger.info(
        "Runtime optimization: backend=%s workers=%d numeric_threads=%d cpu=%d memory=%s tier=%s",
        profile.selected_backend,
        profile.cpu_workers,
        profile.numeric_threads,
        profile.cpu_count,
        f"{profile.total_memory_gb:.1f}GB" if profile.total_memory_gb is not None else "unknown",
        profile.memory_tier,
    )
    for note in profile.notes:
        logger.info("Runtime optimization note: %s", note)
    return profile


def configured_cpu_workers(default: int | None = None) -> int:
    value = os.environ.get("NVAP_CPU_WORKERS")
    try:
        if value is not None and int(value) > 0:
            return int(value)
    except ValueError:
        pass
    return recommended_cpu_workers(default or os.cpu_count() or 1)
