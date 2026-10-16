"""Best-effort startup check against the published release metadata.

Never blocks or fails startup: skipped entirely for dev/source runs, runs on
a background thread, and swallows any network/parsing error since a slow or
unreachable check is not something the user should ever see.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
import threading
import urllib.request
from typing import Callable

from nvap._build_metadata import BUILD_COMMIT, BUILD_VARIANT
from nvap import __version__

logger = logging.getLogger(__name__)

METADATA_URL = "https://api.github.com/repos/gero-studios/NVAP/releases/latest"
_TIMEOUT_SECONDS = 4.0


@dataclass(frozen=True)
class UpdateInfo:
    download_url: str
    build_description: str


def check_for_update_async(on_update_available: Callable[[UpdateInfo], None]) -> None:
    """Fire a background check; invokes the callback only if a newer build
    than this one is currently published. The callback may be called from a
    non-GUI thread — connect it via a Qt signal rather than calling it
    directly if it touches widgets."""
    if BUILD_COMMIT == "dev":
        logger.debug("Update check skipped: running from source.")
        return

    def _worker() -> None:
        try:
            request = urllib.request.Request(
                METADATA_URL, headers={
                    "User-Agent": f"NVAP-UpdateCheck/{BUILD_COMMIT}",
                    "Accept": "application/vnd.github+json",
                }
            )
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            # Compare release versions, not unequal commits: a stale download
            # index must never tell a new build to install an older release.
            released = re.match(r"v?(\d+)\.(\d+)\.(\d+)(?:-|$)", str(payload.get("tag_name", "")))
            current = re.match(r"(\d+)\.(\d+)\.(\d+)$", __version__)
            if released is None or current is None:
                return
            if tuple(map(int, released.groups())) <= tuple(map(int, current.groups())):
                return
            asset = next((
                item for item in payload.get("assets", [])
                if str(item.get("name", "")).endswith(f"-windows-{BUILD_VARIANT}.exe")
                and str(item.get("browser_download_url", "")).startswith("https://github.com/gero-studios/NVAP/releases/download/")
            ), None)
            if asset is None:
                return
            on_update_available(
                UpdateInfo(
                    download_url=str(asset["browser_download_url"]),
                    build_description=str(payload.get("name") or payload["tag_name"]),
                )
            )
        except Exception as exc:  # pragma: no cover - best-effort network path
            logger.debug("Update check skipped: %s", exc)

    threading.Thread(target=_worker, name="nvap-update-check", daemon=True).start()
