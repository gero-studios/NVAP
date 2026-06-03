"""Shared thread-pool executor for preprocessing stages.

Reusing a single ThreadPoolExecutor across sequential preprocessing stages
avoids repeated thread-creation overhead (especially heavy on Windows).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None
_executor_workers: int = 0


class _ExecutorHandle:
    """Wraps the shared executor so context-manager usage doesn't shut it down."""

    def __init__(self, pool: ThreadPoolExecutor) -> None:
        self._pool = pool

    def __enter__(self) -> ThreadPoolExecutor:
        return self._pool

    def __exit__(self, *args: object) -> None:
        pass

    def map(self, *args, **kwargs):
        return self._pool.map(*args, **kwargs)

    def submit(self, *args, **kwargs):
        return self._pool.submit(*args, **kwargs)


def get_executor(workers: int, prefix: str = "nvap-pre") -> _ExecutorHandle:
    """Return a handle to the shared executor, creating one if needed.

    The returned handle can be used as a context manager; exiting the
    context does *not* shut down the executor so it remains available
    for subsequent pipeline stages.
    """
    global _executor, _executor_workers
    workers = max(1, int(workers))
    if _executor is not None and _executor_workers == workers:
        return _ExecutorHandle(_executor)
    if _executor is not None:
        logger.info("Shutting down previous executor (was %d workers)", _executor_workers)
        _executor.shutdown(wait=False)
    _executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=prefix)
    _executor_workers = workers
    return _ExecutorHandle(_executor)


def shutdown_executor() -> None:
    """Shut down the shared executor if running."""
    global _executor, _executor_workers
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None
        _executor_workers = 0
