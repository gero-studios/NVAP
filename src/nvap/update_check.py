"""Best-effort startup check against the published release metadata.

Never blocks or fails startup: skipped entirely for dev/source runs, runs on
a background thread, and swallows any network/parsing error since a slow or
unreachable check is not something the user should ever see.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import threading
import urllib.request
from typing import Callable

from nvap._build_metadata import BUILD_COMMIT, BUILD_VARIANT

logger = logging.getLogger(__name__)

METADATA_URL = "https://nvap-download.restless-cloud-e25f.workers.dev/metadata"
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
            # Cloudflare's edge blocks the default urllib User-Agent outright
            # (403), so this must set an explicit one.
            request = urllib.request.Request(
                METADATA_URL, headers={"User-Agent": f"NVAP-UpdateCheck/{BUILD_COMMIT}"}
            )
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            variant = payload.get(BUILD_VARIANT)
            if not isinstance(variant, dict):
                return
            remote_commit = str(variant.get("commit") or "").strip()
            if not remote_commit or remote_commit == BUILD_COMMIT:
                return
            download_url = str(variant.get("url") or "")
            if not download_url:
                return
            on_update_available(
                UpdateInfo(
                    download_url=download_url,
                    build_description=str(variant.get("build") or "a new build"),
                )
            )
        except Exception as exc:  # pragma: no cover - best-effort network path
            logger.debug("Update check skipped: %s", exc)

    threading.Thread(target=_worker, name="nvap-update-check", daemon=True).start()
