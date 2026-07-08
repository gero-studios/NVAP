"""Project metadata persistence for NVAP datasets."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

from nvap.config.types import PSFConfig, PreprocessConfig, VoxelSpacing

logger = logging.getLogger(__name__)

PROJECT_FILE_NAME = ".nvap_project.json"
PROJECT_FILE_VERSION = 1


def project_file_path(root: str | Path) -> Path:
    return Path(root).resolve() / PROJECT_FILE_NAME


def _dataclass_payload(value: object | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    return None


def save_project_state(
    root: str | Path,
    *,
    channel_sources: dict[str, str | Path],
    dataset_signature: str | None,
    load_mode: str,
    spacing: VoxelSpacing,
    psf_config: PSFConfig,
    preprocess_config: PreprocessConfig,
    cache_key: str | None = None,
    enhancement_method: str | None = None,
) -> Path:
    """Write a small project descriptor next to the dataset."""
    root_path = Path(root).resolve()
    payload: dict[str, Any] = {
        "version": PROJECT_FILE_VERSION,
        "name": root_path.name,
        "root": str(root_path),
        "saved_at_iso": datetime.now().isoformat(timespec="seconds"),
        "load_mode": str(load_mode or "folder"),
        "channel_sources": {
            "green": str(Path(channel_sources["green"]).resolve()),
            "red": str(Path(channel_sources["red"]).resolve()),
        },
        "dataset_signature": dataset_signature,
        "processed_cache_key": cache_key,
        "microglia_enhancement_method": enhancement_method,
        "spacing": _dataclass_payload(spacing),
        "psf_config": _dataclass_payload(psf_config),
        "preprocess_config": _dataclass_payload(preprocess_config),
    }
    path = project_file_path(root_path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Saved NVAP project metadata: %s", path)
    return path


def load_project_state(root: str | Path) -> dict[str, Any] | None:
    path = project_file_path(root)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Project metadata unreadable: %s", exc)
        return None
    if not isinstance(raw, dict):
        return None
    if int(raw.get("version", 0) or 0) > PROJECT_FILE_VERSION:
        logger.info("Project metadata has newer version, continuing best-effort: %s", path)
    sources = raw.get("channel_sources")
    if not isinstance(sources, dict):
        return None
    if "green" not in sources or "red" not in sources:
        return None
    return raw


def project_channel_sources(state: dict[str, Any]) -> dict[str, str] | None:
    sources = state.get("channel_sources")
    if not isinstance(sources, dict):
        return None
    try:
        return {
            "green": str(Path(str(sources["green"])).resolve()),
            "red": str(Path(str(sources["red"])).resolve()),
        }
    except (KeyError, OSError, TypeError, ValueError):
        return None
