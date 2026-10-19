"""Check public entry points and links after retiring the old website."""
from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "website"

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for attribute in ("href", "src"):
            if attribute in attrs:
                self.links.append(attrs[attribute])


def test_landing_assets_and_navigation_resolve():
    parser = Links()
    parser.feed((SITE / "landing.html").read_text(encoding="utf-8"))
    for link in parser.links:
        if link.startswith("https://"):
            continue
        if link.startswith("#"):
            assert not link[1:] or link[1:] in parser.ids
        else:
            assert (SITE / link.split("?", 1)[0]).is_file(), link


def test_all_entry_points_use_clean_landing():
    assert "website/landing.html" in (ROOT / "index.html").read_text(encoding="utf-8")
    assert 'location.replace("landing.html")' in (SITE / "index.html").read_text(encoding="utf-8")
    assert not (SITE / "app.js").exists()
    assert not (SITE / "styles.css").exists()


def test_landing_javascript_parses():
    node = shutil.which("node")
    if node:
        for script in ("landing.js", "segmentation.js"):
            result = subprocess.run([node, "--check", str(SITE / script)], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
