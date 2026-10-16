"""Sample datasets bundled with NVAP.

Bundled samples live in a top-level ``samples/`` directory (one subfolder per
dataset). On first run they are registered in the recent-projects store as pinned
entries so they appear on the Home page and open like any other project. Frozen
builds keep their images in the user data directory, outside temporary extraction.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import logging
from pathlib import Path
import shutil
import sys

from nvap.ui.services.recent_projects import ProjectEntry, RecentProjectsStore

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
    """Register stable sample identities and repair old temporary-folder entries."""
    added = 0
    entries = store.all()
    previous = sorted((entry.to_dict() for entry in entries), key=lambda item: item["path"])
    for name, source in discover_bundled_samples():
        path = source
        if getattr(sys, "frozen", False):
            # One-file builds extract into a different _MEI directory each run.
            # Store image data outside that directory before persisting a path.
            path = store.data_dir / "samples" / name
            for image in source.rglob("*"):
                if not image.is_file() or image.suffix.lower() not in {".png", ".tif", ".tiff", ".czi"}:
                    continue
                destination = path / image.relative_to(source)
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image, destination)
        path = path.resolve()

        def matches(entry: ProjectEntry) -> bool:
            old = Path(entry.path)
            legacy_temp = old.parent.name == "samples" and old.parent.parent.name.startswith("_MEI")
            return entry.sample_id == name or (
                old.name == name and (
                    old in {source, path} or legacy_temp
                    or (entry.status == "Sample" and old.parent.name == "samples")
                )
            )

        aliases = [entry for entry in entries if matches(entry)]
        if aliases:
            latest = max(aliases, key=lambda entry: entry.last_opened)
            canonical = replace(
                latest, path=str(path), name=f"{name} (sample)", sample_id=name,
                pinned=any(entry.pinned for entry in aliases),
                samples=max(entry.samples for entry in aliases),
                tags=list(dict.fromkeys(tag for entry in aliases for tag in entry.tags)),
            )
            entries = [entry for entry in entries if not matches(entry)]
            entries.append(canonical)
        else:
            entries.append(ProjectEntry(
                name=f"{name} (sample)", path=str(path), sample_id=name,
                status="Sample", pinned=True,
                last_opened_iso=datetime.now().isoformat(timespec="seconds"),
            ))
            added += 1
            logger.info("Registered bundled sample: %s -> %s", name, path)
    if sorted((entry.to_dict() for entry in entries), key=lambda item: item["path"]) != previous:
        store.replace_all(sorted(entries, key=lambda entry: entry.last_opened, reverse=True))
    return added
