from __future__ import annotations

import os

from nvap.runtime_optimization import (
    configure_runtime_environment,
    plan_runtime_optimization,
    recommended_cpu_workers,
    recommended_numeric_threads,
)


def test_recommended_cpu_workers_leaves_headroom() -> None:
    assert recommended_cpu_workers(cpu_count=1, memory_gb=16.0) == 1
    assert recommended_cpu_workers(cpu_count=8, memory_gb=16.0) == 7
    assert recommended_cpu_workers(cpu_count=32, memory_gb=64.0) == 12
    assert recommended_cpu_workers(cpu_count=16, memory_gb=4.0) == 2


def test_numeric_threads_avoids_oversubscription() -> None:
    assert recommended_numeric_threads(cpu_count=8, cpu_workers=7) == 1
    assert recommended_numeric_threads(cpu_count=16, cpu_workers=2) == 4


def test_cpu_profile_does_not_require_gpu_probe() -> None:
    plan_runtime_optimization.cache_clear()
    profile = plan_runtime_optimization("cpu", 3)

    assert profile.selected_backend == "cpu"
    assert profile.gpu_available is False
    assert profile.cpu_workers == 3
    assert "CPU backend forced" in " ".join(profile.notes)


def test_configure_runtime_environment_sets_shared_knobs(monkeypatch) -> None:
    plan_runtime_optimization.cache_clear()
    for name in (
        "NVAP_GPU_BACKEND",
        "NVAP_CPU_WORKERS",
        "NVAP_NUMERIC_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)

    profile = configure_runtime_environment(requested_backend="cpu", requested_cpu_workers=2)

    assert profile.selected_backend == "cpu"
    assert os.environ["NVAP_GPU_BACKEND"] == "cpu"
    assert os.environ["NVAP_CPU_WORKERS"] == "2"
    assert os.environ["NVAP_NUMERIC_THREADS"] == str(profile.numeric_threads)
    assert os.environ["OMP_NUM_THREADS"] == str(profile.numeric_threads)

