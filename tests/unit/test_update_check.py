import io
import json

import pytest

from nvap import update_check


@pytest.mark.parametrize("latest,expected", [("v0.1.2-windows-cpu", False), ("v0.1.3", False), ("v0.1.4", True)])
@pytest.mark.parametrize("variant", ["cpu", "directml"])
def test_updates_only_offer_newer_matching_windows_release(monkeypatch, latest, expected, variant):
    monkeypatch.setattr(update_check, "BUILD_COMMIT", "release-commit")
    monkeypatch.setattr(update_check, "BUILD_VARIANT", variant)
    monkeypatch.setattr(update_check, "__version__", "0.1.3")
    assets = [{
        "name": f"NVAP-0.1.4-windows-{kind}.exe",
        "browser_download_url": f"https://github.com/gero-studios/NVAP/releases/download/v0.1.4/NVAP-0.1.4-windows-{kind}.exe",
    } for kind in ("cpu", "directml")]
    payload = {"tag_name": latest, "name": "New release", "assets": assets}
    monkeypatch.setattr(update_check.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))

    class ImmediateThread:
        def __init__(self, *, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(update_check.threading, "Thread", ImmediateThread)
    callbacks = []
    update_check.check_for_update_async(callbacks.append)
    assert bool(callbacks) == expected
    if expected:
        assert callbacks[0].download_url.endswith(f"-windows-{variant}.exe")
