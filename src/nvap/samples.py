"""Sample datasets bundled with NVAP.

Bundled samples live in a top-level ``samples/`` directory (one subfolder per
dataset). On first run they are registered in the recent-projects store as pinned
entries so they appear on the Home page and open like any other project. If the
user later removes a sample from the list it is not re-added.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

from nvap.ui.services.recent_projects import RecentProjectsStore

logger = logging.getLogger(__name__)


def bundled_samples_dir() -> Path:
    """Return the bundled ``samples/`` directory for both source and frozen runs."""
    if getattr(sys, "frozen", False):
        # PyInstaller: data is unpacked under _MEIPASS (onefile) or next to the
        # executable (onedir).
        for base in (
            Path(getattr(sys, "_MEIPASS", "")) if getattr(sys, "_MEIPASS", "") else None,
            Path(sys.executable).parent,
        ):
            if base is not None and (base / "samples").is_dir():
                return base / "samples"
        return Path(sys.executable).parent / "samples"
    # Source layout: src/nvap/samples.py -> src/nvap -> src -> repo root.
    return Path(__file__).resolve().parents[2] / "samples"


def discover_bundled_samples() -> list[tuple[str, Path]]:
    """List ``(name, path)`` for each bundled sample dataset that contains images."""
    root = bundled_samples_dir()
    if not root.is_dir():
        return []
    samples: list[tuple[str, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if next(child.rglob("*.png"), None) is not None or next(child.rglob("*.tif"), None) is not None:
            samples.append((child.name, child.resolve()))
    return samples


def register_bundled_samples(store: RecentProjectsStore) -> int:
    """Register any not-yet-known bundled samples as pinned recent-project entries.

    Returns the number of samples newly added. Existing entries (including ones the
    user has removed and therefore should stay gone) are left untouched.
    """
    added = 0
    for name, path in discover_bundled_samples():
        if store.find(path) is not None:
            continue
        store.upsert(path, name=f"{name} (sample)", status="Sample")
        store.set_pinned(str(path), True)
        added += 1
        logger.info("Registered bundled sample: %s -> %s", name, path)
    return added
