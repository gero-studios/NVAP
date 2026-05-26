from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from nvap.config.types import MeshExportConfig, PreprocessConfig, PSFConfig, RenderConfig, VoxelSpacing

HEADLESS_SETTINGS_VERSION = 1

T = TypeVar("T")


def _dataclass_payload(value: object) -> dict[str, Any]:
    if not is_dataclass(value):
        return {}
    return asdict(value)


def _dataclass_from_payload(cls: type[T], payload: object, default: T) -> T:
    if not isinstance(payload, dict):
        return default
    allowed = {field.name for field in fields(cls)}
    values = {key: value for key, value in payload.items() if key in allowed}
    try:
        return cls(**values)
    except (TypeError, ValueError):
        return default


def build_headless_settings_payload(
    *,
    dataset_root: str | Path | None,
    channel_sources: dict[str, str | Path] | None,
    load_mode: str,
    spacing: VoxelSpacing,
    psf_config: PSFConfig,
    preprocess_config: PreprocessConfig,
    render_config: RenderConfig,
    mesh_config: MeshExportConfig,
    microglia_enhancement_method: str = "microglia_preserve",
) -> dict[str, Any]:
    root_path = Path(dataset_root).resolve() if dataset_root is not None else None
    sources = None
    if channel_sources is not None:
        sources = {
            "green": str(Path(channel_sources["green"]).resolve()),
            "red": str(Path(channel_sources["red"]).resolve()),
        }
    return {
        "version": HEADLESS_SETTINGS_VERSION,
        "dataset": {
            "root": str(root_path) if root_path is not None else "",
            "load_mode": str(load_mode or "folder"),
            "channel_sources": sources,
        },
        "spacing": _dataclass_payload(spacing),
        "psf_config": _dataclass_payload(psf_config),
        "preprocess_config": _dataclass_payload(preprocess_config),
        "render_config": _dataclass_payload(render_config),
        "mesh_config": _dataclass_payload(mesh_config),
        "microglia_enhancement": {
            "method": str(microglia_enhancement_method or "microglia_preserve"),
        },
    }


def save_headless_settings(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def load_headless_settings(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("NVAP settings file must contain a JSON object.")
    return raw


def settings_dataset_root(settings: dict[str, Any], fallback: str | Path) -> Path:
    dataset = settings.get("dataset")
    if isinstance(dataset, dict):
        root = str(dataset.get("root") or "").strip()
        if root:
            return Path(root).resolve()
    return Path(fallback).resolve()


def settings_channel_overrides(settings: dict[str, Any]) -> dict[str, str] | None:
    dataset = settings.get("dataset")
    if not isinstance(dataset, dict):
        return None
    sources = dataset.get("channel_sources")
    if not isinstance(sources, dict):
        return None
    try:
        return {
            "green": str(Path(str(sources["green"])).resolve()),
            "red": str(Path(str(sources["red"])).resolve()),
        }
    except (KeyError, TypeError, ValueError, OSError):
        return None


def settings_spacing(settings: dict[str, Any]) -> VoxelSpacing:
    return _dataclass_from_payload(VoxelSpacing, settings.get("spacing"), VoxelSpacing())


def settings_psf_config(settings: dict[str, Any]) -> PSFConfig:
    return _dataclass_from_payload(PSFConfig, settings.get("psf_config"), PSFConfig(enabled=False, iterations=0))


def settings_preprocess_config(settings: dict[str, Any]) -> PreprocessConfig:
    return _dataclass_from_payload(PreprocessConfig, settings.get("preprocess_config"), PreprocessConfig(enabled=True))


def settings_render_config(settings: dict[str, Any]) -> RenderConfig:
    return _dataclass_from_payload(RenderConfig, settings.get("render_config"), RenderConfig())


def settings_mesh_config(settings: dict[str, Any], export_format: str | None = None) -> MeshExportConfig:
    config = _dataclass_from_payload(MeshExportConfig, settings.get("mesh_config"), MeshExportConfig())
    if export_format:
        return MeshExportConfig(**{**asdict(config), "export_format": export_format})
    return config


def settings_microglia_enhancement(settings: dict[str, Any]) -> dict[str, str]:
    raw = settings.get("microglia_enhancement")
    if not isinstance(raw, dict):
        raw = {}
    return {
        "method": str(raw.get("method") or "microglia_preserve"),
    }
