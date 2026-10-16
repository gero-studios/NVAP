"""Native NVAP landing screen and project library."""
from __future__ import annotations

from typing import NamedTuple

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from nvap.ui.landing_scene import LandingScene
from nvap.ui.design import COLOR, ICON_LG, ICON_MD, SPACE
from nvap.ui.icons import icon, icon_pixmap, icon_size
from nvap.ui.services.recent_projects import RecentProjectsStore
from nvap.ui.widgets.recent_projects_list import RecentProjectsList


class RecentProject(NamedTuple):
    """Legacy in-memory tuple. Persistent storage now lives in
    nvap.ui.services.recent_projects.ProjectEntry. Kept for back-compat with
    callers that haven't migrated yet.
    """
    name: str
    last_modified: str
    samples: int
    status: str


class HomePage(QWidget):
    """NVAP landing page. Hosts the projects library; replaces the old
    Projects + Samples sidebar pages.
    """

    # Quick actions
    new_project_requested = Signal()
    open_project_requested = Signal()
    import_stack_requested = Signal()
    browse_samples_requested = Signal()

    # Project library row interactions
    project_open_requested = Signal(str)     # path
    project_remove_requested = Signal(str)   # path
    project_pin_toggled = Signal(str, bool)  # path, pinned

    def __init__(self, store: RecentProjectsStore, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("homePage")
        self._store = store

        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("homeScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content.setObjectName("homeContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(40, 32, 40, 40)
        layout.setSpacing(0)

        # ─── Hero row ───────────────────────────────────────────────────
        hero_row = QHBoxLayout()
        self._hero_row = hero_row
        hero_row.setSpacing(SPACE.xl)

        hero_left = QVBoxLayout()
        hero_left.setSpacing(0)

        brand_row = QHBoxLayout()
        brand_row.setSpacing(14)
        brand_mark = QLabel()
        brand_mark.setObjectName("homeBrandMark")
        brand_mark.setPixmap(icon_pixmap("logo", 36, "#c3d5b9"))
        brand_mark.setFixedSize(36, 36)
        brand_row.addWidget(brand_mark)
        eyebrow = QLabel("N V A P   /   NEUROVASCULAR ANALYTICS")
        eyebrow.setStyleSheet("color: #9fb7c8; font-size: 10px; letter-spacing: 2px;")
        brand_row.addWidget(eyebrow, 1)
        hero_left.addLayout(brand_row)
        hero_left.addSpacing(28)
        title = QLabel("A closer look.\nA clearer picture.")
        title.setObjectName("heroTitle")
        title.setWordWrap(True)
        title.setStyleSheet("font-size: 42px; font-weight: 500; color: #edf2f7; background: transparent;")
        hero_left.addWidget(title)

        subtitle = QLabel("NeuroVascular Analytics Program")
        subtitle.setObjectName("heroSubtitle")
        subtitle.setWordWrap(True)
        hero_left.addWidget(subtitle)
        hero_left.addSpacing(SPACE.md)

        desc = QLabel(
            "Explore microglia and vasculature in three dimensions. "
            "Turn microscopy stacks into measurable insight."
        )
        desc.setObjectName("heroDesc")
        desc.setWordWrap(True)
        desc.setMaximumWidth(400)
        hero_left.addWidget(desc)
        hero_left.addSpacing(SPACE.xl)

        # Quick action grid (2×2)
        actions = QGridLayout()
        actions.setSpacing(SPACE.md)
        actions.setColumnStretch(0, 1)
        actions.setColumnStretch(1, 1)

        self.new_project_btn = self._make_action_btn(
            "plus", "New Project", primary=True
        )
        self.new_project_btn.clicked.connect(self.new_project_requested.emit)

        self.open_project_btn = self._make_action_btn("folder-open", "Open Project")
        self.open_project_btn.clicked.connect(self.open_project_requested.emit)

        self.import_btn = self._make_action_btn("upload", "Import Image Stack")
        self.import_btn.clicked.connect(self.import_stack_requested.emit)

        self.browse_btn = self._make_action_btn("grid", "Browse Samples")
        self.browse_btn.clicked.connect(self.browse_samples_requested.emit)

        actions.addWidget(self.new_project_btn, 0, 0)
        actions.addWidget(self.open_project_btn, 0, 1)
        actions.addWidget(self.import_btn, 1, 0)
        actions.addWidget(self.browse_btn, 1, 1)

        hero_left.addLayout(actions)
        hero_row.addLayout(hero_left, 1)

        self._preview = LandingScene()
        hero_row.addWidget(self._preview, 1)
        layout.addLayout(hero_row)
        layout.addSpacing(SPACE.xl)

        workflow = QLabel("01  RECONSTRUCT     /     02  QUANTIFY     /     03  DISCOVER")
        workflow.setStyleSheet("color: #8d98a5; font-size: 10px; letter-spacing: 1px; padding: 20px 0; border-top: 1px solid #242b35; border-bottom: 1px solid #242b35;")
        workflow.setWordWrap(True)
        layout.addWidget(workflow)
        layout.addSpacing(32)

        # ─── Projects library ──────────────────────────────────────────
        lib_header = QHBoxLayout()
        lib_title = QLabel("Your projects")
        lib_title.setObjectName("sectionTitle")
        lib_header.addWidget(lib_title)

        self._project_count = QLabel("0 projects")
        self._project_count.setObjectName("sectionSubtle")
        self._project_count.setContentsMargins(SPACE.md, 0, 0, 0)
        lib_header.addWidget(self._project_count)
        lib_header.addStretch(1)

        new_link = QPushButton("New Project")
        new_link.setObjectName("linkButton")
        new_link.setIcon(icon("plus", size=ICON_MD, color=COLOR.accent_hover))
        new_link.setIconSize(icon_size(ICON_MD))
        new_link.clicked.connect(self.new_project_requested.emit)
        lib_header.addWidget(new_link)

        layout.addLayout(lib_header)
        layout.addSpacing(SPACE.md)

        self.recent_table = RecentProjectsList(self._store)
        self.recent_table.setMinimumHeight(260)
        self.recent_table.project_open_requested.connect(self.project_open_requested.emit)
        self.recent_table.project_remove_requested.connect(self._on_remove_project)
        self.recent_table.project_pin_toggled.connect(self.project_pin_toggled.emit)
        layout.addWidget(self.recent_table)

        self._empty_state = QLabel(
            "Your next discovery starts here.\nCreate a project or open an existing image stack."
        )
        self._empty_state.setObjectName("pageBody")
        self._empty_state.setWordWrap(True)
        self._empty_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_state.setContentsMargins(0, SPACE.xl, 0, SPACE.xl)
        layout.addWidget(self._empty_state)

        layout.addStretch(1)

        self._scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scroll)

        self._entrance_effect = QGraphicsOpacityEffect(content)
        content.setGraphicsEffect(self._entrance_effect)
        self._entrance = QPropertyAnimation(self._entrance_effect, b"opacity", self)
        self._entrance.setDuration(850)
        self._entrance.setStartValue(0.0)
        self._entrance.setEndValue(1.0)
        self._entrance.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._entrance.finished.connect(lambda: self._entrance_effect.setEnabled(False))
        self._has_entered = False

        self.refresh_projects()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._has_entered:
            self._has_entered = True
            self._entrance.start()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        from PySide6.QtWidgets import QBoxLayout
        compact = self.width() < 850
        self._hero_row.setDirection(
            QBoxLayout.Direction.TopToBottom if compact else QBoxLayout.Direction.LeftToRight
        )
        self._preview.setMinimumHeight(260 if compact else 390)

    # ── builders ───────────────────────────────────────────────────────
    def _make_action_btn(self, icon_name: str, label: str, *, primary: bool = False) -> QPushButton:
        btn = QPushButton(f"  {label}")
        btn.setObjectName("primaryActionHome" if primary else "secondaryActionHome")
        btn.setFixedHeight(44)
        btn.setIcon(icon(icon_name, size=ICON_MD, color="#FFFFFF" if primary else COLOR.text_primary))
        btn.setIconSize(icon_size(ICON_MD))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        return btn

    # ── public API ─────────────────────────────────────────────────────
    def refresh_projects(self) -> None:
        self.recent_table.refresh()
        count = self._store.count()
        self._project_count.setText(f"{count} project{'s' if count != 1 else ''}")
        has_any = count > 0
        self.recent_table.setVisible(has_any)
        self._empty_state.setVisible(not has_any)
        if not has_any:
            self._preview.set_summary("ILLUSTRATIVE VIEW", "Two channels. One perspective.", "Microglia / Vasculature")
        if has_any:
            entries = self._store.all()
            top = entries[0]
            from datetime import datetime
            try:
                ts = datetime.fromisoformat(top.last_opened_iso).strftime("%b %d, %Y · %H:%M")
            except ValueError:
                ts = "Unknown"
            self._preview.set_summary(
                "RECENT",
                top.name,
                f"Last opened {ts}\n{top.samples or 0} sample(s) • {top.status}",
            )

    def focus_top(self) -> None:
        self._scroll.verticalScrollBar().setValue(0)

    def focus_recent_projects(self) -> None:
        self._scroll.ensureWidgetVisible(self.recent_table, 0, 64)
        self.recent_table.setFocus(Qt.FocusReason.OtherFocusReason)

    def set_preview_summary(self, kicker: str, title: str, body: str) -> None:
        self._preview.set_summary(kicker, title, body)

    # ── handlers ───────────────────────────────────────────────────────
    def _on_remove_project(self, path: str) -> None:
        self._store.remove(path)
        self.refresh_projects()
        self.project_remove_requested.emit(path)
