"""Persistent recent-projects store.

Stores project entries in ``~/.nvap/recent_projects.json`` so the list
survives restarts. Newest first, deduplicated by path, capped at MAX_ENTRIES.

A "project" here is the dataset root the user opened (a directory).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

MAX_ENTRIES = 20


def _store_path() -> Path:
    base = Path(os.environ.get("NVAP_HOME") or (Path.home() / ".nvap"))
    return base / "recent_projects.json"


@dataclass
class ProjectEntry:
    name: str
    path: str
    last_opened_iso: str
    samples: int = 0
    status: str = "Ready"
    pinned: bool = False
    tags: list[str] = field(default_factory=list)

    @property
    def last_opened(self) -> datetime:
        try:
            return datetime.fromisoformat(self.last_opened_iso)
        except ValueError:
            return datetime.fromtimestamp(0)

    @property
    def exists(self) -> bool:
        try:
            return Path(self.path).exists()
        except OSError:
            return False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectEntry":
        return cls(
            name=str(data.get("name", "")),
            path=str(data.get("path", "")),
            last_opened_iso=str(data.get("last_opened_iso", "")),
            samples=int(data.get("samples", 0) or 0),
            status=str(data.get("status", "Ready")),
            pinned=bool(data.get("pinned", False)),
            tags=list(data.get("tags") or []),
        )


class RecentProjectsStore:
    """File-backed list of recent projects."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _store_path()
        self._entries: list[ProjectEntry] = []
        self._load()

    # ── persistence ────────────────────────────────────────────────────
    def _load(self) -> None:
        if not self._path.exists():
            self._entries = []
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            entries = [ProjectEntry.from_dict(item) for item in raw if isinstance(item, dict)]
            self._entries = entries[:MAX_ENTRIES]
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Recent projects store unreadable, starting empty: %s", exc)
            self._entries = []

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps([e.to_dict() for e in self._entries], indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Recent projects store unwritable: %s", exc)

    # ── queries ────────────────────────────────────────────────────────
    def all(self) -> list[ProjectEntry]:
        # Pinned first, then newest first.
        pinned = [e for e in self._entries if e.pinned]
        unpinned = [e for e in self._entries if not e.pinned]
        unpinned.sort(key=lambda e: e.last_opened, reverse=True)
        pinned.sort(key=lambda e: e.last_opened, reverse=True)
        return pinned + unpinned

    def count(self) -> int:
        return len(self._entries)

    def find(self, path: str | os.PathLike) -> ProjectEntry | None:
        target = str(Path(path))
        for entry in self._entries:
            if str(Path(entry.path)) == target:
                return entry
        return None

    # ── mutations ──────────────────────────────────────────────────────
    def upsert(
        self,
        path: str | os.PathLike,
        *,
        name: str | None = None,
        samples: int | None = None,
        status: str | None = None,
    ) -> ProjectEntry:
        target = str(Path(path))
        existing = self.find(target)
        now_iso = datetime.now().isoformat(timespec="seconds")
        if existing is None:
            entry = ProjectEntry(
                name=name or Path(target).name or target,
                path=target,
                last_opened_iso=now_iso,
                samples=int(samples or 0),
                status=status or "Ready",
            )
            self._entries.insert(0, entry)
        else:
            entry = existing
            entry.last_opened_iso = now_iso
            if name:
                entry.name = name
            if samples is not None:
                entry.samples = int(samples)
            if status is not None:
                entry.status = status
            # Move to front (preserving pinned status)
            self._entries = [entry] + [e for e in self._entries if e is not entry]
        self._entries = self._entries[:MAX_ENTRIES]
        self._save()
        return entry

    def remove(self, path: str | os.PathLike) -> bool:
        target = str(Path(path))
        before = len(self._entries)
        self._entries = [e for e in self._entries if str(Path(e.path)) != target]
        if len(self._entries) != before:
            self._save()
            return True
        return False

    def set_pinned(self, path: str | os.PathLike, pinned: bool) -> None:
        entry = self.find(path)
        if entry is None:
            return
        entry.pinned = bool(pinned)
        self._save()

    def clear(self) -> None:
        self._entries = []
        self._save()

    def replace_all(self, entries: Iterable[ProjectEntry]) -> None:
        self._entries = list(entries)[:MAX_ENTRIES]
        self._save()
