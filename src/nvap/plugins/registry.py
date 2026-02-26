from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
import logging
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PluginDescriptor:
    plugin_id: str
    target_channel: str
    status: str
    error: str = ""
    plugin: Any | None = None


def discover_plugins() -> list[PluginDescriptor]:
    discovered: list[PluginDescriptor] = []
    try:
        entries = entry_points(group="nvap.plugins")
    except TypeError:
        entries = entry_points().get("nvap.plugins", [])

    logger.debug("Discovered %d plugin entry point(s).", len(entries))
    for ep in entries:
        try:
            loaded = ep.load()
            plugin = loaded() if isinstance(loaded, type) else loaded
            plugin_id = str(getattr(plugin, "plugin_id", ep.name))
            target = str(getattr(plugin, "target_channel", "unknown"))
            discovered.append(
                PluginDescriptor(
                    plugin_id=plugin_id,
                    target_channel=target,
                    status="loaded",
                    plugin=plugin,
                )
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Plugin load failed for entry point '%s'.", ep.name)
            discovered.append(
                PluginDescriptor(
                    plugin_id=ep.name,
                    target_channel="unknown",
                    status="error",
                    error=str(exc),
                )
            )
    return discovered
