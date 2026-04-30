"""NVAP home page – hero, quick actions, feature cards, projects library."""
from __future__ import annotations

from typing import NamedTuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

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


class _FeatureCard(QWidget):
    def __init__(self, icon_name: str, title: str, description: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("featureCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(SPACE.sm)

        icon_lbl = QLabel()
        icon_lbl.setObjectName("featureCardIcon")
        icon_lbl.setPixmap(icon_pixmap(icon_name, size=ICON_LG, color=COLOR.accent_hover))
        layout.addWidget(icon_lbl)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("featureCardTitle")
        layout.addWidget(title_lbl)

        desc_lbl = QLabel(description)
        desc_lbl.setObjectName("featureCardDesc")
        desc_lbl.setWordWrap(True)
        layout.addWidget(desc_lbl)

        layout.addStretch(1)


class _PreviewPanel(QFrame):
    """Hero preview frame – placeholder text until VTK snapshot is wired in."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("homePreviewBox")
        self.setFixedSize(420, 300)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(8)

        self._kicker = QLabel("WORKSPACE")
        self._kicker.setObjectName("previewKicker")
        layout.addWidget(self._kicker)

        self._title = QLabel("3D Workspace ready")
        self._title.setObjectName("previewTitle")
        self._title.setWordWrap(True)
        layout.addWidget(self._title)

        self._body = QLabel(
            "Load a dataset to render microglia and vasculature.\n"
            "Two channels • live metrics • mesh export."
        )
        self._body.setObjectName("previewBody")
        self._body.setWordWrap(True)
        layout.addWidget(self._body)
        layout.addStretch(1)

        # Subtle channel legend
        legend = QHBoxLayout()
        legend.setSpacing(SPACE.lg)
        for label, color in (("Microglia", COLOR.channel_green), ("Vasculature", COLOR.channel_red)):
            row = QHBoxLayout()
            row.setSpacing(6)
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 14px; background: transparent;")
            text = QLabel(label)
            text.setStyleSheet(f"color: {COLOR.text_tertiary}; background: transparent;")
            row.addWidget(dot)
            row.addWidget(text)
            container = QWidget()
            container.setLayout(row)
            legend.addWidget(container)
        legend.addStretch(1)
        layout.addLayout(legend)

    def set_summary(self, kicker: str, title: str, body: str) -> None:
        self._kicker.setText(kicker)
        self._title.setText(title)
        self._body.setText(body)


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
        layout.setContentsMargins(SPACE.huge, 56, SPACE.huge, 56)
        layout.setSpacing(0)

        # ─── Hero row ───────────────────────────────────────────────────
        hero_row = QHBoxLayout()
        hero_row.setSpacing(SPACE.xxxl)

        hero_left = QVBoxLayout()
        hero_left.setSpacing(0)

        title = QLabel("NVAP")
        title.setObjectName("heroTitle")
        hero_left.addWidget(title)

        subtitle = QLabel("Microglia + Vasculature Analysis Platform")
        subtitle.setObjectName("heroSubtitle")
        hero_left.addWidget(subtitle)
        hero_left.addSpacing(SPACE.lg)

        desc = QLabel(
            "Reconstruct, segment, and quantify microglia and brain vasculature\n"
            "from confocal microscopy stacks. Built for neuroscience research."
        )
        desc.setObjectName("heroDesc")
        hero_left.addWidget(desc)
        hero_left.addSpacing(SPACE.xxl)

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

        self._preview = _PreviewPanel()
        hero_row.addWidget(self._preview, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(hero_row)
        layout.addSpacing(SPACE.xxxl)

        # ─── Feature cards ──────────────────────────────────────────────
        cards = QGridLayout()
        cards.setSpacing(SPACE.lg)
        for col in range(4):
            cards.setColumnStretch(col, 1)
        feature_specs = [
            ("box",      "3D Reconstruction", "Reconstruct volumes from 2D stacks with interpolation and channel sync."),
            ("scissors", "Segmentation",      "Microglia separation and vessel masking with branch-preserving denoise."),
            ("bar-chart","Analytics",         "Quantify morphology, density, and cell-vessel interactions."),
            ("share",    "Export",            "Metrics CSV, PNG snapshots, and PLY/OBJ/STL meshes."),
        ]
        for i, (ic, title_text, desc_text) in enumerate(feature_specs):
            cards.addWidget(_FeatureCard(ic, title_text, desc_text), 0, i)
        layout.addLayout(cards)
        layout.addSpacing(SPACE.xxxl)

        # ─── Projects library ──────────────────────────────────────────
        lib_header = QHBoxLayout()
        lib_title = QLabel("Projects Library")
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
            "No projects yet. Use New Project or Open Project to get started."
        )
        self._empty_state.setObjectName("pageBody")
        self._empty_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_state.setContentsMargins(0, SPACE.xl, 0, SPACE.xl)
        layout.addWidget(self._empty_state)

        layout.addStretch(1)

        self._scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scroll)

        self.refresh_projects()

    # ── builders ───────────────────────────────────────────────────────
    def _make_action_btn(self, icon_name: str, label: str, *, primary: bool = False) -> QPushButton:
        btn = QPushButton(f"  {label}")
        btn.setObjectName("primaryActionHome" if primary else "secondaryActionHome")
        btn.setFixedHeight(46)
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
