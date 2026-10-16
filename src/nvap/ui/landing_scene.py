"""Lightweight, procedural artwork for the native landing screen."""
from __future__ import annotations

import math
import random
import time

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QPushButton, QWidget


class LandingScene(QWidget):
    """An illustrative vascular field; never presented as measured sample data.

    Geometry is generated once. Animation pauses when hidden and can be disabled
    with the keyboard-accessible motion control.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 390)
        self.setAccessibleName("Animated illustration of microglia and vasculature")
        self._summary = ("ILLUSTRATIVE VIEW", "Two channels. One perspective.", "Microglia / Vasculature")
        self._phase = 0.0
        self._last_frame = time.monotonic()
        self._paused = False
        self._paths = []
        rng = random.Random(28)

        def branch(x, y, angle, length, depth, channel):
            end_x = x + math.cos(angle) * length
            end_y = y + math.sin(angle) * length
            path = QPainterPath(QPointF(x, y))
            path.cubicTo(x + length * .42 * math.cos(angle + .4),
                         y + length * .42 * math.sin(angle + .4),
                         end_x - length * .3 * math.cos(angle - .25),
                         end_y - length * .3 * math.sin(angle - .25), end_x, end_y)
            self._paths.append((path, depth, channel, rng.random()))
            if depth > 0:
                for offset in (-.55, .48):
                    branch(end_x, end_y, angle + offset + rng.uniform(-.22, .22),
                           length * rng.uniform(.61, .78), depth - 1, channel)

        branch(265, 410, -1.7, 125, 5, 0)
        branch(420, 45, 2.2, 100, 4, 0)
        for x, y in ((100, 165), (335, 215), (165, 330)):
            for i in range(6):
                branch(x, y, i * math.tau / 6, rng.uniform(22, 37), 2, 1)

        self.motion_button = QPushButton("Pause motion", self)
        self.motion_button.setCheckable(True)
        self.motion_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.motion_button.setStyleSheet(
            "QPushButton {background: #10141a; color: #a9b4bf; border: 1px solid #343d49;"
            "border-radius: 4px; padding: 4px 10px; font-size: 10px;}"
            "QPushButton:hover, QPushButton:focus {border-color: #9fb7c8; color: #edf2f7;}"
        )
        self.motion_button.toggled.connect(self._toggle_motion)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._advance)

    def set_summary(self, kicker, title, body):
        self._summary = (kicker, title, body)
        self.setAccessibleDescription(". ".join(self._summary))
        self.update()

    def _toggle_motion(self, paused):
        self._paused = paused
        self.motion_button.setText("Resume motion" if paused else "Pause motion")
        self.motion_button.adjustSize()
        self.motion_button.move(self.width() - self.motion_button.width() - 12, 12)
        self._sync_timer()

    def _sync_timer(self):
        self._last_frame = time.monotonic()
        if self.isVisible() and not self._paused:
            self._timer.start()
        else:
            self._timer.stop()

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_timer()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.motion_button.adjustSize()
        self.motion_button.move(self.width() - self.motion_button.width() - 12, 12)

    def _advance(self):
        now = time.monotonic()
        self._phase += min(now - self._last_frame, .1)
        self._last_frame = now
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#0b0e12"))
        glow = QRadialGradient(w * .57, h * .42, w * .65)
        glow.setColorAt(0, QColor("#182229"))
        glow.setColorAt(1, QColor("#0b0e12"))
        p.fillRect(self.rect(), glow)
        p.save()
        p.setClipRect(0, 45, w, h - 132)
        scale = min(w / 490, (h - 50) / 440)
        p.translate(w / 2 - 245 * scale, -12 + math.sin(self._phase * .24) * 5)
        p.scale(scale, scale)
        for path, depth, channel, offset in self._paths:
            color = QColor("#87b9a4" if channel else "#b77d85")
            color.setAlphaF(.28 + depth * .095)
            p.setPen(QPen(color, (depth * 1.25 + .65) if not channel else .8 + depth * .5,
                          Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPath(path)
            if depth > 1:
                point = path.pointAtPercent((self._phase * .075 + offset) % 1)
                color.setAlpha(180)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(color)
                p.drawEllipse(point, 1.4, 1.4)
        p.restore()
        p.setPen(QColor("#82919d"))
        p.setFont(QFont("Aptos", 8))
        p.drawText(16, 29, "N V / 0 1")
        kicker, title, body = self._summary
        p.drawText(16, h - 69, kicker)
        p.setFont(QFont("Aptos", 12))
        p.setPen(QColor("#d9e1e7"))
        p.drawText(16, h - 44, p.fontMetrics().elidedText(title, Qt.TextElideMode.ElideRight, w - 32))
        p.setFont(QFont("Aptos", 9))
        p.setPen(QColor("#8d98a5"))
        p.drawText(16, h - 21, p.fontMetrics().elidedText(body.replace("\n", " · "), Qt.TextElideMode.ElideRight, w - 32))
        p.end()
