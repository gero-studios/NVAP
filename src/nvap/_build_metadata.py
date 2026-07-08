"""Build identity used only by the update-check.

Overwritten with the real commit/variant by scripts/build_windows.ps1 right
before PyInstaller runs, then restored to these dev defaults afterward. When
running from source (pip install -e .), BUILD_COMMIT stays "dev" and the
update-check skips itself entirely.
"""

BUILD_COMMIT = "dev"
BUILD_VARIANT = "cpu"
BUILT_AT = ""
