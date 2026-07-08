"""Shared loading-screen drawing for NVAP."""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap

LOADING_BG = "#07090c"
LOADING_SURFACE = "#0b0e12"
LOADING_BORDER = "#242b35"
LOADING_ACCENT = "#9fb7c8"
LOADING_TEXT = "#edf2f7"
LOADING_MUTED = "#8d98a5"


def build_loading_pixmap(
    *,
    title: str = "NVAP",
    message: str = "Preparing visualization workspace",
    detail: str = "Loading UI, VTK, and analysis modules...",
    progress_fraction: float = 0.45,
) -> QPixmap:
    """Build the minimal startup loading screen used by the executable."""
    progress = max(0.0, min(1.0, float(progress_fraction)))
    pixmap = QPixmap(420, 180)
    pixmap.fill(QColor(LOADING_BG))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

    painter.setPen(QColor(LOADING_BORDER))
    painter.drawRect(0, 0, 419, 179)

    title_font = QFont("Aptos")
    title_font.setPointSize(12)
    title_font.setWeight(QFont.Weight.Bold)
    painter.setFont(title_font)
    painter.setPen(QColor(LOADING_ACCENT))
    painter.drawText(QRect(28, 28, 364, 28), Qt.AlignmentFlag.AlignLeft, title)

    body_font = QFont("Aptos")
    body_font.setPointSize(9)
    painter.setFont(body_font)
    painter.setPen(QColor(LOADING_TEXT))
    painter.drawText(QRect(28, 66, 364, 24), Qt.AlignmentFlag.AlignLeft, message)

    detail_font = QFont("Cascadia Mono")
    detail_font.setPointSize(8)
    painter.setFont(detail_font)
    painter.setPen(QColor(LOADING_MUTED))
    painter.drawText(QRect(28, 106, 364, 18), Qt.AlignmentFlag.AlignLeft, detail)

    painter.setPen(QColor(LOADING_BORDER))
    painter.drawRect(24, 136, 372, 10)
    painter.fillRect(26, 138, int(368 * progress), 6, QColor(LOADING_ACCENT))
    painter.end()
    return pixmap
