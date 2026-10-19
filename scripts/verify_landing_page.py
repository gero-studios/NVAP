"""Validate the active NVAP website entry points, links, and scripts."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    raise SystemExit(subprocess.call([sys.executable, "-B", "-m", "pytest", "tests/unit/test_landing_page.py", "-q"], cwd=root))
