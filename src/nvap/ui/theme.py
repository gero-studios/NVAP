"""Minimal scientific UI theme for NVAP."""

DARK_THEME_QSS = """
QWidget {
    background-color: #101216;
    color: #d9dee7;
    font-family: "Aptos", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 12px;
}

QMainWindow {
    background-color: #080a0d;
}

QSplitter::handle {
    background-color: #0b0d11;
    width: 6px;
}

QSplitter::handle:hover {
    background-color: #c8a45d;
}

QScrollArea,
QScrollArea > QWidget > QWidget {
    background-color: transparent;
    border: none;
}

QLabel {
    color: #9aa4b2;
    font-size: 11px;
}

QLabel#panelTitle {
    color: #f2f5f8;
    font-size: 19px;
    font-weight: 650;
    letter-spacing: 1.5px;
}

QLabel#panelSubtitle {
    color: #7f8a99;
    font-size: 10px;
    letter-spacing: 1.0px;
    text-transform: uppercase;
}

QLabel#modeHint {
    color: #778293;
    font-size: 10px;
    padding: 6px 2px 2px 2px;
}

QLabel#pendingWarning {
    color: #d2a94d;
    font-style: italic;
}

QGroupBox {
    background-color: #151922;
    border: 1px solid #272e3b;
    border-radius: 10px;
    margin-top: 18px;
    padding: 14px 10px 10px 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 1px 8px;
    color: #d8bd78;
    background-color: #101216;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
}

QPushButton {
    background-color: #1b202a;
    color: #e5e9f0;
    border: 1px solid #313947;
    border-radius: 8px;
    padding: 7px 12px;
    min-height: 22px;
    font-weight: 520;
}

QPushButton:hover {
    background-color: #222936;
    border-color: #d8bd78;
}

QPushButton:pressed {
    background-color: #2a2418;
    border-color: #f0c56b;
}

QPushButton:disabled {
    background-color: #12161d;
    color: #56606f;
    border-color: #202633;
}

QPushButton#primaryAction {
    background-color: #d8bd78;
    color: #111318;
    border-color: #f0d98d;
    font-weight: 700;
}

QPushButton#primaryAction:hover {
    background-color: #f0d98d;
}

QPushButton[text="<"],
QPushButton[text=">"] {
    padding: 3px 7px;
    min-height: 18px;
}

QTextEdit,
QLineEdit,
QSpinBox,
QDoubleSpinBox,
QComboBox {
    background-color: #0c0f14;
    border: 1px solid #2a3342;
    border-radius: 7px;
    color: #eef2f6;
    padding: 5px 7px;
    selection-background-color: #d8bd78;
    selection-color: #101216;
}

QTextEdit,
QLineEdit,
QSpinBox,
QDoubleSpinBox {
    font-family: "Cascadia Mono", "Consolas", monospace;
}

QTextEdit:focus,
QLineEdit:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QComboBox:focus {
    border-color: #d8bd78;
    background-color: #11161e;
}

QTextEdit:read-only {
    color: #a6b0bd;
}

QComboBox::drop-down {
    border: none;
    width: 22px;
}

QCheckBox {
    color: #d7dde6;
    spacing: 8px;
}

QCheckBox::indicator,
QGroupBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 4px;
    border: 1px solid #3a4352;
    background-color: #0c0f14;
}

QCheckBox::indicator:hover,
QGroupBox::indicator:hover {
    border-color: #d8bd78;
}

QCheckBox::indicator:checked,
QGroupBox::indicator:checked {
    background-color: #d8bd78;
    border-color: #f0d98d;
}

QLabel#channelGreen,
QCheckBox#channelGreen {
    color: #69d9a3;
}

QLabel#channelRed,
QCheckBox#channelRed {
    color: #e56f78;
}

QTableWidget,
QTableView {
    background-color: #0c0f14;
    alternate-background-color: #11161e;
    color: #d9dee7;
    gridline-color: #252d3a;
    border: 1px solid #272e3b;
    border-radius: 8px;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 10px;
}

QHeaderView::section {
    background-color: #171c25;
    color: #d8bd78;
    border: none;
    border-bottom: 1px solid #303847;
    padding: 6px 8px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.9px;
    text-transform: uppercase;
}

QTableWidget::item:selected,
QTableView::item:selected {
    background-color: #d8bd78;
    color: #111318;
}

QProgressBar {
    background-color: #0c0f14;
    border: 1px solid #2a3342;
    border-radius: 7px;
    color: #d9dee7;
    text-align: center;
    min-height: 16px;
}

QProgressBar::chunk {
    background-color: #d8bd78;
    border-radius: 6px;
}

QScrollBar:vertical {
    background-color: #0c0f14;
    width: 10px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background-color: #303847;
    border-radius: 5px;
    min-height: 28px;
}

QScrollBar::handle:vertical:hover {
    background-color: #d8bd78;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background-color: #0c0f14;
    height: 10px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background-color: #303847;
    border-radius: 5px;
    min-width: 28px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #d8bd78;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0;
}

QStatusBar {
    background-color: #080a0d;
    color: #8b95a4;
    border-top: 1px solid #202633;
}

QStatusBar::item {
    border: none;
}

QToolTip {
    background-color: #171c25;
    color: #eef2f6;
    border: 1px solid #d8bd78;
    border-radius: 6px;
    padding: 6px 8px;
}

QDialog,
QMessageBox,
QFileDialog,
QMenu {
    background-color: #101216;
    color: #d9dee7;
}

QMenuBar {
    background-color: #080a0d;
    color: #d9dee7;
    border-bottom: 1px solid #202633;
}

QMenuBar::item:selected,
QMenu::item:selected {
    background-color: #d8bd78;
    color: #111318;
}
"""
