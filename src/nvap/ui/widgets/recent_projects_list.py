"""Reusable recent-projects table backed by RecentProjectsStore.

Single component used everywhere recent projects are displayed.
Supports double-click to open, context-menu to pin/remove/reveal.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QMenu,
    QTableWidget,
    QTableWidgetItem,
)

from nvap.ui.services.recent_projects import ProjectEntry, RecentProjectsStore


def _format_relative(when: datetime) -> str:
    delta = datetime.now() - when
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "Just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hr ago"
    days = hours // 24
    if days < 7:
        return f"{days} day{'s' if days != 1 else ''} ago"
    return when.strftime("%b %d, %Y")


class RecentProjectsList(QTableWidget):
    """Table of recent projects with built-in actions."""

    project_open_requested = Signal(str)         # path
    project_remove_requested = Signal(str)       # path
    project_pin_toggled = Signal(str, bool)      # path, pinned

    def __init__(self, store: RecentProjectsStore, parent=None) -> None:
        super().__init__(0, 4, parent)
        self._store = store
        self.setObjectName("recentProjectsTable")
        self.setHorizontalHeaderLabels(["Name", "Last Opened", "Samples", "Status"])
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setHighlightSections(False)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.itemDoubleClicked.connect(self._on_double_click)

        self.refresh()

    # ── data ───────────────────────────────────────────────────────────
    def refresh(self) -> None:
        entries = self._store.all()
        self.setRowCount(0)
        for row, entry in enumerate(entries):
            self.insertRow(row)
            name = ("📌  " + entry.name) if entry.pinned else entry.name
            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.ItemDataRole.UserRole, entry.path)
            name_item.setToolTip(entry.path)
            self.setItem(row, 0, name_item)
            self.setItem(row, 1, QTableWidgetItem(_format_relative(entry.last_opened)))
            self.setItem(row, 2, QTableWidgetItem(str(entry.samples or "—")))
            status = entry.status if entry.exists else "Missing"
            status_item = QTableWidgetItem(f"●  {status}")
            self.setItem(row, 3, status_item)

    def selected_entry(self) -> ProjectEntry | None:
        row = self.currentRow()
        if row < 0:
            return None
        item = self.item(row, 0)
        if item is None:
            return None
        path = item.data(Qt.ItemDataRole.UserRole)
        return self._store.find(str(path)) if path else None

    # ── interaction ────────────────────────────────────────────────────
    def _on_double_click(self, item: QTableWidgetItem) -> None:
        entry = self.selected_entry()
        if entry is not None:
            self.project_open_requested.emit(entry.path)

    def _show_context_menu(self, _pos) -> None:
        entry = self.selected_entry()
        if entry is None:
            return
        menu = QMenu(self)
        open_act = QAction("Open project", menu)
        open_act.triggered.connect(lambda: self.project_open_requested.emit(entry.path))
        menu.addAction(open_act)

        pin_label = "Unpin" if entry.pinned else "Pin to top"
        pin_act = QAction(pin_label, menu)
        pin_act.triggered.connect(lambda: self._toggle_pin(entry))
        menu.addAction(pin_act)

        reveal_act = QAction("Copy path", menu)
        reveal_act.triggered.connect(lambda: self._copy_path(entry))
        menu.addAction(reveal_act)

        menu.addSeparator()
        remove_act = QAction("Remove from recents", menu)
        remove_act.triggered.connect(lambda: self.project_remove_requested.emit(entry.path))
        menu.addAction(remove_act)

        menu.exec(QCursor.pos())

    def _toggle_pin(self, entry: ProjectEntry) -> None:
        new_state = not entry.pinned
        self._store.set_pinned(entry.path, new_state)
        self.project_pin_toggled.emit(entry.path, new_state)
        self.refresh()

    def _copy_path(self, entry: ProjectEntry) -> None:
        from PySide6.QtGui import QGuiApplication

        QGuiApplication.clipboard().setText(str(Path(entry.path)))
