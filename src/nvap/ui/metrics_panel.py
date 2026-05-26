"""NVAP right-side quantitative summary panel."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from nvap.config.types import MetricsComputation


class _MetricRow(QWidget):
    def __init__(self, icon: str, label: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("metricRow")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(10)

        icon_lbl = QLabel(icon)
        icon_lbl.setObjectName("metricIcon")
        icon_lbl.setFixedWidth(22)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_lbl)

        name_lbl = QLabel(label)
        name_lbl.setObjectName("metricLabel")
        layout.addWidget(name_lbl, 1)

        self._value_lbl = QLabel("\u2014")
        self._value_lbl.setObjectName("metricValue")
        self._value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._value_lbl)

    def set_value(self, text: str) -> None:
        self._value_lbl.setText(text)


class _StatusRow(QWidget):
    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(8)

        lbl = QLabel(label)
        lbl.setObjectName("statusLabel")
        layout.addWidget(lbl, 1)

        self._dot = QLabel("\u25cf")
        self._dot.setObjectName("statusDotPending")
        layout.addWidget(self._dot)

        self._status_text = QLabel("Pending")
        self._status_text.setObjectName("statusTextPending")
        layout.addWidget(self._status_text)

    def set_complete(self) -> None:
        self._dot.setObjectName("statusDotComplete")
        self._dot.style().unpolish(self._dot)
        self._dot.style().polish(self._dot)
        self._status_text.setObjectName("statusTextComplete")
        self._status_text.style().unpolish(self._status_text)
        self._status_text.style().polish(self._status_text)
        self._status_text.setText("Complete")

    def set_pending(self) -> None:
        self._dot.setObjectName("statusDotPending")
        self._dot.style().unpolish(self._dot)
        self._dot.style().polish(self._dot)
        self._status_text.setObjectName("statusTextPending")
        self._status_text.style().unpolish(self._status_text)
        self._status_text.style().polish(self._status_text)
        self._status_text.setText("Pending")


class MetricsPanel(QWidget):
    """Right-side quantitative summary panel showing dataset metrics."""

    view_details_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("metricsPanel")
        self.setMinimumWidth(220)
        self.setMaximumWidth(310)

        scroll = QScrollArea(self)
        scroll.setObjectName("metricsPanelScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content.setObjectName("metricsPanelContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 24, 18, 24)
        layout.setSpacing(0)

        # ── Quantitative Summary ──────────────────────────────────────────
        summary_lbl = QLabel("Quantitative Summary")
        summary_lbl.setObjectName("panelSectionTitle")
        layout.addWidget(summary_lbl)
        self._summary_hint = QLabel("Load a dataset to populate this summary.")
        self._summary_hint.setObjectName("summaryHint")
        self._summary_hint.setWordWrap(True)
        layout.addWidget(self._summary_hint)
        layout.addSpacing(16)

        self._green_voxels = _MetricRow("\u2726", "Green Voxels")
        self._green_volume = _MetricRow("\u29c1", "Green Volume")
        self._red_voxels = _MetricRow("\u25c9", "Red Voxels")
        self._red_volume = _MetricRow("\u2442", "Red Volume")
        self._overlap = _MetricRow("\u2194", "Overlap Volume")
        self._components = _MetricRow("\u229d", "Components")
        self._dataset_loaded = False

        for row in (
            self._green_voxels,
            self._green_volume,
            self._red_voxels,
            self._red_volume,
            self._overlap,
            self._components,
        ):
            layout.addWidget(row)
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setObjectName("metricSeparator")
            layout.addWidget(sep)

        layout.addSpacing(24)

        # ── Processing Status ─────────────────────────────────────────────
        status_lbl = QLabel("Processing Status")
        status_lbl.setObjectName("panelSectionTitle")
        layout.addWidget(status_lbl)
        layout.addSpacing(12)

        status_card = QFrame()
        status_card.setObjectName("statusCard")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(14, 12, 14, 12)
        status_layout.setSpacing(4)

        self._status_reconstruction = _StatusRow("3D Reconstruction")
        self._status_summary = _StatusRow("Quantitative Summary")

        status_layout.addWidget(self._status_reconstruction)
        status_layout.addWidget(self._status_summary)

        layout.addWidget(status_card)
        layout.addSpacing(8)

        self.view_details_btn = QPushButton("Open Analytics  \u203a")
        self.view_details_btn.setObjectName("linkButton")
        self.view_details_btn.setFlat(True)
        self.view_details_btn.clicked.connect(self.view_details_requested.emit)
        layout.addWidget(self.view_details_btn, alignment=Qt.AlignmentFlag.AlignRight)

        layout.addStretch(1)

        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    @staticmethod
    def _format_volume_um3(value: float) -> str:
        return f"{value:,.1f} um^3"

    def _clear_values(self) -> None:
        for row in (
            self._green_voxels,
            self._green_volume,
            self._red_voxels,
            self._red_volume,
            self._overlap,
            self._components,
        ):
            row.set_value("\u2014")

    def update_from_metrics(self, metrics: MetricsComputation) -> None:
        """Update dataset-scale summary values."""
        self._dataset_loaded = bool(metrics.channel_results)
        if self._dataset_loaded:
            self._status_reconstruction.set_complete()
        else:
            self.clear()
            return

        self._status_summary.set_complete()
        self._summary_hint.setText("Summary derived from the current render thresholds.")

        by_channel = {item.channel.lower(): item for item in metrics.channel_results}
        green = by_channel.get("green")
        red = by_channel.get("red")
        if green is not None:
            self._green_voxels.set_value(f"{green.voxel_count:,}")
            self._green_volume.set_value(self._format_volume_um3(float(green.volume_um3)))
        if red is not None:
            self._red_voxels.set_value(f"{red.voxel_count:,}")
            self._red_volume.set_value(self._format_volume_um3(float(red.volume_um3)))
        self._overlap.set_value(self._format_volume_um3(float(metrics.overlap_volume_um3)))
        component_total = sum(int(item.component_count) for item in metrics.channel_results)
        self._components.set_value(f"{component_total:,}")

    def clear(self) -> None:
        """Reset all metrics to placeholder dashes."""
        self._dataset_loaded = False
        self._clear_values()
        self._status_reconstruction.set_pending()
        self._status_summary.set_pending()
        self._summary_hint.setText("Load a dataset to populate this summary.")
