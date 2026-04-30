"""About dialog."""
from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from nvap.ui.design import COLOR, ICON_XL, SPACE
from nvap.ui.icons import icon_pixmap


def _safe_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "—"


class AboutDialog(QDialog):
    """Modal "About NVAP" dialog."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("aboutDialog")
        self.setWindowTitle("About NVAP")
        self.setModal(True)
        self.setMinimumWidth(440)

        nvap_ver = _safe_version("nvap")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE.xxl, SPACE.xxl, SPACE.xxl, SPACE.xl)
        layout.setSpacing(SPACE.md)

        header = QHBoxLayout()
        header.setSpacing(SPACE.lg)
        logo = QLabel()
        logo.setObjectName("aboutLogo")
        logo.setPixmap(icon_pixmap("logo", size=ICON_XL * 2, color=COLOR.accent_hover))
        logo.setFixedSize(ICON_XL * 2, ICON_XL * 2)
        header.addWidget(logo, 0, Qt.AlignmentFlag.AlignTop)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title = QLabel("NVAP")
        title.setObjectName("aboutTitle")
        title_col.addWidget(title)
        ver = QLabel(f"NeuroVascular Analytics Program · v{nvap_ver}")
        ver.setObjectName("aboutVersion")
        title_col.addWidget(ver)
        header.addLayout(title_col, 1)
        layout.addLayout(header)

        layout.addSpacing(SPACE.md)

        body = QLabel(
            "Desktop 3D viewer and analyzer for confocal microscopy stacks. "
            "Reconstructs microglia and vasculature, separates individual cells, "
            "and quantifies morphology and cell-vessel interactions."
        )
        body.setObjectName("aboutBody")
        body.setWordWrap(True)
        layout.addWidget(body)

        layout.addSpacing(SPACE.lg)

        runtime = QLabel(
            f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} · "
            f"{platform.system()} {platform.release()}\n"
            f"PySide6 {_safe_version('PySide6')}  ·  VTK {_safe_version('vtk')}  ·  "
            f"NumPy {_safe_version('numpy')}  ·  SciPy {_safe_version('scipy')}"
        )
        runtime.setObjectName("aboutFooter")
        runtime.setWordWrap(True)
        layout.addWidget(runtime)

        layout.addSpacing(SPACE.md)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
