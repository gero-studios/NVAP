"""Lightweight QPainter-based charts for the Analytics tab.

Avoids external chart deps. Two widgets:
- ``HistogramWidget``  — distribution of a numeric series.
- ``BarListWidget``    — horizontal labeled bars (top-N rankings).

All widgets degrade to an empty-state message when no data is set, and
use the design tokens from ``nvap.ui.design`` so they match the rest of
the app.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from nvap.ui.design import COLOR, FONT, RADIUS, SPACE


def _qcolor(hex_str: str, alpha: int | None = None) -> QColor:
    c = QColor(hex_str)
    if alpha is not None:
        c.setAlpha(int(alpha))
    return c


def _draw_empty(painter: QPainter, rect: QRectF, message: str) -> None:
    painter.save()
    painter.setPen(_qcolor(COLOR.text_tertiary))
    font = QFont()
    font.setPointSize(FONT.sm)
    painter.setFont(font)
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, message)
    painter.restore()


class HistogramWidget(QWidget):
    """Mini histogram with a title, axis labels, and an empty-state message."""

    def __init__(
        self,
        title: str = "",
        *,
        unit: str = "",
        bins: int = 18,
        accent: str | None = None,
        empty_text: str = "No data",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._unit = unit
        self._bins = max(4, int(bins))
        self._accent = accent or COLOR.accent
        self._empty_text = empty_text
        self._values: list[float] = []
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(280, 160)

    def set_values(self, values: Sequence[float]) -> None:
        cleaned: list[float] = []
        for v in values:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv != fv or fv in (float("inf"), float("-inf")):
                continue
            cleaned.append(fv)
        self._values = cleaned
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())

        bg = _qcolor(COLOR.bg_surface)
        painter.setBrush(bg)
        painter.setPen(QPen(_qcolor(COLOR.border_subtle), 1))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), RADIUS.md, RADIUS.md)

        pad_l, pad_t, pad_r, pad_b = SPACE.md, SPACE.sm, SPACE.md, SPACE.md
        inner = rect.adjusted(pad_l, pad_t, -pad_r, -pad_b)

        # Title
        title_h = 0.0
        if self._title:
            title_font = QFont()
            title_font.setPointSize(FONT.sm)
            title_font.setBold(True)
            painter.setFont(title_font)
            painter.setPen(_qcolor(COLOR.text_secondary))
            title_rect = QRectF(inner.left(), inner.top(), inner.width(), 16)
            painter.drawText(
                title_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._title,
            )
            title_h = 18.0

        plot_rect = QRectF(
            inner.left(),
            inner.top() + title_h,
            inner.width(),
            inner.height() - title_h - 14.0,
        )

        if not self._values:
            _draw_empty(painter, plot_rect, self._empty_text)
            return

        lo = min(self._values)
        hi = max(self._values)
        if hi <= lo:
            hi = lo + 1.0

        counts = [0] * self._bins
        span = hi - lo
        for v in self._values:
            idx = int((v - lo) / span * self._bins)
            if idx >= self._bins:
                idx = self._bins - 1
            elif idx < 0:
                idx = 0
            counts[idx] += 1
        max_count = max(counts) or 1

        accent = _qcolor(self._accent)
        accent_dim = _qcolor(self._accent, alpha=70)
        bar_w = plot_rect.width() / self._bins
        for i, c in enumerate(counts):
            if c <= 0:
                continue
            h = (c / max_count) * (plot_rect.height() - 2.0)
            bar_rect = QRectF(
                plot_rect.left() + i * bar_w + 1.0,
                plot_rect.bottom() - h,
                max(1.0, bar_w - 2.0),
                h,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(accent_dim)
            painter.drawRoundedRect(bar_rect, 2.0, 2.0)
            painter.setBrush(accent)
            cap_rect = QRectF(bar_rect.left(), bar_rect.top(), bar_rect.width(), min(3.0, h))
            painter.drawRect(cap_rect)

        # Axis labels: min / max
        painter.setPen(_qcolor(COLOR.text_tertiary))
        axis_font = QFont()
        axis_font.setPointSize(FONT.xs)
        painter.setFont(axis_font)
        unit = f" {self._unit}" if self._unit else ""
        axis_rect = QRectF(
            inner.left(),
            plot_rect.bottom() + 2.0,
            inner.width(),
            12.0,
        )
        painter.drawText(
            axis_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            f"{lo:,.1f}{unit}",
        )
        painter.drawText(
            axis_rect,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            f"{hi:,.1f}{unit}",
        )
        n_text = f"n={len(self._values):,}"
        painter.drawText(
            axis_rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            n_text,
        )


@dataclass(frozen=True)
class BarListItem:
    label: str
    value: float
    annotation: str = ""


class BarListWidget(QWidget):
    """Top-N labeled horizontal bars."""

    def __init__(
        self,
        title: str = "",
        *,
        accent: str | None = None,
        empty_text: str = "No data",
        max_rows: int = 6,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._accent = accent or COLOR.accent
        self._empty_text = empty_text
        self._max_rows = max(1, int(max_rows))
        self._items: list[BarListItem] = []
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)

    def sizeHint(self) -> QSize:
        return QSize(320, 200)

    def set_items(self, items: Sequence[BarListItem]) -> None:
        clean = [it for it in items if it.value > 0]
        clean.sort(key=lambda it: it.value, reverse=True)
        self._items = clean[: self._max_rows]
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        painter.setBrush(_qcolor(COLOR.bg_surface))
        painter.setPen(QPen(_qcolor(COLOR.border_subtle), 1))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), RADIUS.md, RADIUS.md)

        pad_l, pad_t, pad_r, pad_b = SPACE.md, SPACE.sm, SPACE.md, SPACE.md
        inner = rect.adjusted(pad_l, pad_t, -pad_r, -pad_b)

        title_h = 0.0
        if self._title:
            title_font = QFont()
            title_font.setPointSize(FONT.sm)
            title_font.setBold(True)
            painter.setFont(title_font)
            painter.setPen(_qcolor(COLOR.text_secondary))
            title_rect = QRectF(inner.left(), inner.top(), inner.width(), 16)
            painter.drawText(
                title_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._title,
            )
            title_h = 20.0

        plot_rect = QRectF(
            inner.left(),
            inner.top() + title_h,
            inner.width(),
            inner.height() - title_h,
        )

        if not self._items:
            _draw_empty(painter, plot_rect, self._empty_text)
            return

        accent = _qcolor(self._accent)
        accent_dim = _qcolor(self._accent, alpha=60)
        row_count = len(self._items)
        row_h = min(28.0, plot_rect.height() / row_count)
        max_value = max(it.value for it in self._items) or 1.0
        label_w = min(plot_rect.width() * 0.42, 140.0)

        label_font = QFont()
        label_font.setPointSize(FONT.sm)
        value_font = QFont()
        value_font.setPointSize(FONT.xs)

        for i, item in enumerate(self._items):
            row_top = plot_rect.top() + i * row_h
            row_rect = QRectF(plot_rect.left(), row_top, plot_rect.width(), row_h)

            painter.setFont(label_font)
            painter.setPen(_qcolor(COLOR.text_primary))
            label_rect = QRectF(row_rect.left(), row_rect.top(), label_w, row_rect.height())
            painter.drawText(
                label_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                item.label,
            )

            bar_track = QRectF(
                row_rect.left() + label_w + 4,
                row_rect.center().y() - 4,
                row_rect.width() - label_w - 4,
                8,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_qcolor(COLOR.border_subtle))
            painter.drawRoundedRect(bar_track, 4, 4)

            fill_w = bar_track.width() * (item.value / max_value)
            bar_fill = QRectF(bar_track.left(), bar_track.top(), fill_w, bar_track.height())
            painter.setBrush(accent_dim)
            painter.drawRoundedRect(bar_fill, 4, 4)
            cap_fill = QRectF(bar_track.left(), bar_track.top(), min(fill_w, 3.0), bar_track.height())
            painter.setBrush(accent)
            painter.drawRoundedRect(cap_fill, 2, 2)

            painter.setFont(value_font)
            painter.setPen(_qcolor(COLOR.text_secondary))
            annot = item.annotation or f"{item.value:,.0f}"
            painter.drawText(
                row_rect,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                annot,
            )

