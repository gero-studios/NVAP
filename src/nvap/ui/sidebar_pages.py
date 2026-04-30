"""Settings page – the last remaining sidebar destination beyond Home/Workspace.

Projects and Samples were folded into the Home page in the v0.6 UI refactor.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class _StatCard(QFrame):
    def __init__(self, label: str, value: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pageStatCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(6)

        self._value_label = QLabel(value)
        self._value_label.setObjectName("pageStatValue")
        self._value_label.setWordWrap(True)
        layout.addWidget(self._value_label)

        caption = QLabel(label)
        caption.setObjectName("pageStatLabel")
        caption.setWordWrap(True)
        layout.addWidget(caption)

    def set_value(self, value: str) -> None:
        self._value_label.setText(value)


class _ScrollableSectionPage(QWidget):
    """Shared layout: eyebrow + title + intro inside a scroll area."""

    def __init__(self, eyebrow: str, title: str, intro: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("sectionPage")

        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("sectionScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content.setObjectName("sectionContent")
        self._content_layout = QVBoxLayout(content)
        self._content_layout.setContentsMargins(56, 48, 56, 48)
        self._content_layout.setSpacing(0)

        eyebrow_label = QLabel(eyebrow)
        eyebrow_label.setObjectName("pageEyebrow")
        self._content_layout.addWidget(eyebrow_label)

        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        self._content_layout.addWidget(title_label)

        intro_label = QLabel(intro)
        intro_label.setObjectName("pageIntro")
        intro_label.setWordWrap(True)
        self._content_layout.addWidget(intro_label)
        self._content_layout.addSpacing(28)

        self._scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scroll)

    def focus_top(self) -> None:
        self._scroll.verticalScrollBar().setValue(0)


class SettingsPage(_ScrollableSectionPage):
    """Runtime overview + future preferences home."""

    open_workspace_requested = Signal()
    open_analytics_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(
            "RUNTIME OVERVIEW",
            "Settings",
            "Current dataset, cache target, and plugin/runtime status. "
            "Dedicated preferences UI lands in a future session.",
            parent=parent,
        )

        action_row = QHBoxLayout()
        action_row.setSpacing(12)

        workspace_btn = QPushButton("Open Workspace")
        workspace_btn.setObjectName("primaryActionHome")
        workspace_btn.clicked.connect(self.open_workspace_requested.emit)
        action_row.addWidget(workspace_btn)

        analytics_btn = QPushButton("Open Analytics")
        analytics_btn.setObjectName("secondaryActionHome")
        analytics_btn.clicked.connect(self.open_analytics_requested.emit)
        action_row.addWidget(analytics_btn)

        action_row.addStretch(1)
        self._content_layout.addLayout(action_row)
        self._content_layout.addSpacing(24)

        stats_layout = QGridLayout()
        stats_layout.setHorizontalSpacing(16)
        stats_layout.setVerticalSpacing(16)
        self._dataset_card = _StatCard("Active dataset", "No dataset loaded")
        self._auto_apply_card = _StatCard("Auto apply", "On")
        self._plugins_card = _StatCard("Plugins", "No plugins discovered")
        self._cache_card = _StatCard("Cache root", ".nvap_cache")
        stats_layout.addWidget(self._dataset_card, 0, 0)
        stats_layout.addWidget(self._auto_apply_card, 0, 1)
        stats_layout.addWidget(self._plugins_card, 1, 0)
        stats_layout.addWidget(self._cache_card, 1, 1)
        self._content_layout.addLayout(stats_layout)
        self._content_layout.addStretch(1)

    def set_runtime_details(
        self,
        *,
        dataset_name: str,
        auto_apply_enabled: bool,
        plugin_summary: str,
        cache_root: str,
    ) -> None:
        self._dataset_card.set_value(dataset_name)
        self._auto_apply_card.set_value("On" if auto_apply_enabled else "Manual")
        self._plugins_card.set_value(plugin_summary)
        self._cache_card.set_value(cache_root)
