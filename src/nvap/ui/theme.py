"""Scientific Dark Theme for NVAP — researcher-grade, modern."""

# Palette
# Background layers:  #111318 (deepest)  #191c24 (base)  #1f232e (panel)  #252a38 (surface)
# Borders:            #2e3347 (subtle)    #3d4461 (mid)    #4f5880 (accent-dim)
# Text:               #cdd5f0 (primary)   #8a93b8 (secondary)  #515a7a (muted)
# Accent teal/cyan:   #3dd6c8 (bright)    #27a99e (mid)    #1a7a72 (deep)
# Accent blue:        #4d90fe (info)      #2c6fe0 (press)
# Green channel:      #3dffa0 (indicator)
# Red channel:        #ff5f7a (indicator)
# Warning:            #f0c040
# Error:              #e05050

DARK_THEME_QSS = """

/* ═══════════════════════════════════════════
   BASE
   ═══════════════════════════════════════════ */
QWidget {
    background-color: #191c24;
    color: #cdd5f0;
    font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
    font-size: 12px;
}

QMainWindow {
    background-color: #111318;
}

/* ═══════════════════════════════════════════
   SPLITTER
   ═══════════════════════════════════════════ */
QSplitter::handle {
    background-color: #2e3347;
    width: 3px;
}
QSplitter::handle:hover {
    background-color: #3dd6c8;
}
QSplitter::handle:horizontal {
    width: 3px;
}
QSplitter::handle:vertical {
    height: 3px;
}

/* ═══════════════════════════════════════════
   SCROLL AREA — used around control panel
   ═══════════════════════════════════════════ */
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollArea > QWidget > QWidget {
    background-color: transparent;
}

/* ═══════════════════════════════════════════
   BUTTONS
   ═══════════════════════════════════════════ */
QPushButton {
    background-color: #252a38;
    color: #cdd5f0;
    border: 1px solid #3d4461;
    padding: 5px 11px;
    border-radius: 3px;
    letter-spacing: 0.2px;
}
QPushButton:hover {
    background-color: #2e3347;
    border: 1px solid #3dd6c8;
    color: #ffffff;
}
QPushButton:pressed {
    background-color: #1a7a72;
    border: 1px solid #3dd6c8;
    color: #ffffff;
}
QPushButton:disabled {
    background-color: #181c26;
    color: #515a7a;
    border: 1px solid #262c3e;
}

QPushButton#primaryAction {
    background-color: #27a99e;
    color: #ffffff;
    border: 1px solid #1a7a72;
    font-weight: 600;
    letter-spacing: 0.3px;
}
QPushButton#primaryAction:hover {
    background-color: #3dd6c8;
    border: 1px solid #27a99e;
}
QPushButton#primaryAction:pressed {
    background-color: #1a7a72;
}
QPushButton#primaryAction:disabled {
    background-color: #1a3330;
    color: #4a6b68;
    border: 1px solid #1e2e2d;
}

/* Small navigation prev/next buttons */
QPushButton[text="<"], QPushButton[text=">"] {
    padding: 4px 7px;
    font-weight: bold;
    background-color: #1f232e;
    border: 1px solid #3d4461;
}
QPushButton[text="<"]:hover, QPushButton[text=">"]:hover {
    border-color: #3dd6c8;
    background-color: #252a38;
}

/* ═══════════════════════════════════════════
   GROUP BOXES  (panel sections)
   ═══════════════════════════════════════════ */
QGroupBox {
    background-color: #1f232e;
    border: 1px solid #2e3347;
    border-radius: 4px;
    margin-top: 18px;
    padding-top: 8px;
    padding-bottom: 4px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: -1px;
    padding: 1px 6px;
    background-color: #191c24;
    color: #3dd6c8;
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    border-radius: 2px;
}
QGroupBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #3d4461;
    background-color: #252a38;
    border-radius: 2px;
}
QGroupBox::indicator:checked {
    background-color: #27a99e;
    border: 1px solid #1a7a72;
}

/* ═══════════════════════════════════════════
   FORM LAYOUT row labels
   ═══════════════════════════════════════════ */
QLabel {
    color: #8a93b8;
    font-size: 11px;
}

/* ═══════════════════════════════════════════
   INPUTS: SpinBox, DoubleSpinBox, LineEdit
   ═══════════════════════════════════════════ */
QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #141720;
    border: 1px solid #2e3347;
    border-radius: 3px;
    padding: 3px 6px;
    color: #e8edff;
    font-family: "Consolas", "JetBrains Mono", "Fira Code", monospace;
    font-size: 12px;
    selection-background-color: #27a99e;
    selection-color: #ffffff;
}
QTextEdit:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #3dd6c8;
    background-color: #161a24;
}
QTextEdit:read-only, QLineEdit:read-only {
    background-color: #161921;
    color: #8a93b8;
    border-color: #272d42;
}

/* SpinBox buttons */
QSpinBox::up-button, QDoubleSpinBox::up-button {
    width: 18px;
    border-left: 1px solid #2e3347;
    background-color: #1f232e;
    subcontrol-origin: border;
    subcontrol-position: top right;
    border-top-right-radius: 3px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    width: 18px;
    border-left: 1px solid #2e3347;
    background-color: #1f232e;
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    border-bottom-right-radius: 3px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #27a99e;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 6px;
    height: 6px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 6px;
    height: 6px;
}

/* ═══════════════════════════════════════════
   CHECKBOXES
   ═══════════════════════════════════════════ */
QCheckBox {
    spacing: 7px;
    color: #cdd5f0;
    font-size: 12px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    background-color: #141720;
    border: 1px solid #3d4461;
    border-radius: 2px;
}
QCheckBox::indicator:hover {
    border-color: #3dd6c8;
}
QCheckBox::indicator:checked {
    background-color: #27a99e;
    border-color: #1a7a72;
    /* Unicode checkmark rendered via color — no image needed */
}
QCheckBox::indicator:checked:hover {
    background-color: #3dd6c8;
}
QCheckBox::indicator:disabled {
    background-color: #181c26;
    border-color: #262c3e;
}
QCheckBox:disabled {
    color: #515a7a;
}

/* ═══════════════════════════════════════════
   SCROLLBARS
   ═══════════════════════════════════════════ */
QScrollBar:vertical {
    border: none;
    background-color: #141720;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background-color: #2e3347;
    min-height: 22px;
    border-radius: 5px;
    margin: 1px 2px;
}
QScrollBar::handle:vertical:hover {
    background-color: #3d4461;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    border: none;
    background-color: #141720;
    height: 10px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background-color: #2e3347;
    min-width: 22px;
    border-radius: 5px;
    margin: 2px 1px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #3d4461;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* ═══════════════════════════════════════════
   PROGRESS DIALOG / PROGRESS BAR
   ═══════════════════════════════════════════ */
QProgressBar {
    background-color: #141720;
    border: 1px solid #2e3347;
    border-radius: 3px;
    text-align: center;
    color: #cdd5f0;
    font-family: "Consolas", monospace;
    font-size: 11px;
    height: 14px;
}
QProgressBar::chunk {
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #1a7a72, stop:1 #3dd6c8
    );
    border-radius: 2px;
}

/* ═══════════════════════════════════════════
   STATUS BAR
   ═══════════════════════════════════════════ */
QStatusBar {
    background-color: #111318;
    color: #8a93b8;
    border-top: 1px solid #2e3347;
    font-size: 11px;
    padding: 0 6px;
}
QStatusBar::item {
    border: none;
}

/* ═══════════════════════════════════════════
   TOOLTIPS
   ═══════════════════════════════════════════ */
QToolTip {
    background-color: #1f232e;
    color: #e8edff;
    border: 1px solid #3d4461;
    padding: 5px 8px;
    border-radius: 3px;
    font-size: 11px;
}

/* ═══════════════════════════════════════════
   DIALOG BOXES (QProgressDialog, QMessageBox)
   ═══════════════════════════════════════════ */
QDialog {
    background-color: #191c24;
    border: 1px solid #3d4461;
}
QMessageBox {
    background-color: #191c24;
}
QMessageBox QLabel {
    color: #cdd5f0;
    font-size: 12px;
}

/* ═══════════════════════════════════════════
   FILE DIALOG
   ═══════════════════════════════════════════ */
QFileDialog {
    background-color: #191c24;
}

/* ═══════════════════════════════════════════
   MENU (for QMenuBar / context menus)
   ═══════════════════════════════════════════ */
QMenuBar {
    background-color: #111318;
    color: #cdd5f0;
    border-bottom: 1px solid #2e3347;
    padding: 1px 4px;
}
QMenuBar::item:selected {
    background-color: #1f232e;
    color: #3dd6c8;
}
QMenu {
    background-color: #1f232e;
    border: 1px solid #3d4461;
    padding: 3px;
}
QMenu::item {
    padding: 5px 18px 5px 10px;
    border-radius: 2px;
    color: #cdd5f0;
}
QMenu::item:selected {
    background-color: #27a99e;
    color: #ffffff;
}
QMenu::separator {
    height: 1px;
    background: #2e3347;
    margin: 3px 8px;
}

/* ═══════════════════════════════════════════
   TABLES (used by QTableWidget if added)
   ═══════════════════════════════════════════ */
QTableWidget, QTableView {
    background-color: #141720;
    alternate-background-color: #191c24;
    gridline-color: #2e3347;
    border: 1px solid #2e3347;
    border-radius: 3px;
    color: #cdd5f0;
    font-family: "Consolas", monospace;
    font-size: 11px;
}
QHeaderView::section {
    background-color: #1f232e;
    color: #3dd6c8;
    font-weight: 600;
    font-size: 10px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    border: none;
    border-bottom: 1px solid #3d4461;
    padding: 4px 8px;
}
QTableWidget::item:selected, QTableView::item:selected {
    background-color: #27a99e;
    color: #ffffff;
}

/* ═══════════════════════════════════════════
   CHANNEL-SPECIFIC LABELS & CHECKBOXES
   ═══════════════════════════════════════════ */
QLabel#channelGreen, QCheckBox#channelGreen {
    color: #3dffa0;
    font-weight: 500;
}
QCheckBox#channelGreen::indicator:checked {
    background-color: #27c47a;
    border-color: #1a7a4a;
}

QLabel#channelRed, QCheckBox#channelRed {
    color: #ff7a90;
    font-weight: 500;
}
QCheckBox#channelRed::indicator:checked {
    background-color: #c43050;
    border-color: #8a1e38;
}

/* ═══════════════════════════════════════════
   MODE HINT  (pass-through status line)
   ═══════════════════════════════════════════ */
QLabel#modeHint {
    color: #515a7a;
    font-size: 10px;
    font-style: italic;
    padding: 1px 4px;
}

/* ═══════════════════════════════════════════
   PENDING-CHANGE WARNING LABEL
   ═══════════════════════════════════════════ */
QLabel#pendingWarning {
    color: #f0c040;
    font-style: italic;
    font-size: 11px;
}
"""
