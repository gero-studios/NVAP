"""Modern Dark Theme for NVAP UI Overhaul."""

DARK_THEME_QSS = """
/* Global Application Settings */
QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}

/* Tooltips */
QToolTip {
    background-color: #2b2b2b;
    color: #ffffff;
    border: 1px solid #555555;
    padding: 4px;
    border-radius: 4px;
}

/* Splitter (between control panel and VTK viewer) */
QSplitter::handle {
    background-color: #333333;
    width: 6px;
    border-radius: 3px;
    margin: 2px;
}
QSplitter::handle:hover {
    background-color: #007acc;
}

/* Push Buttons */
QPushButton {
    background-color: #333333;
    color: #ffffff;
    border: 1px solid #555555;
    padding: 6px 12px;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #444444;
    border: 1px solid #007acc;
}
QPushButton:pressed {
    background-color: #007acc;
    border: 1px solid #005a9e;
}
QPushButton:disabled {
    background-color: #2a2a2a;
    color: #888888;
    border: 1px solid #444444;
}

/* Primary Action Button Overrides (Set dynamically via code if needed, or by objectName) */
QPushButton#primaryAction {
    background-color: #007acc;
    color: #ffffff;
    border: 1px solid #005a9e;
    font-weight: bold;
}
QPushButton#primaryAction:hover {
    background-color: #0098ff;
}

/* Group Boxes for better sectioning */
QGroupBox {
    font-weight: bold;
    border: 1px solid #444444;
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    left: 10px;
    color: #007acc;
}

/* Text Editors & Inputs */
QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #252526;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px;
    color: #ffffff;
    selection-background-color: #007acc;
}
QTextEdit:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #007acc;
}

/* SpinBox Arrows */
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 16px;
    border-left: 1px solid #555555;
    background-color: #333333;
    border-top-right-radius: 4px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 16px;
    border-left: 1px solid #555555;
    background-color: #333333;
    border-bottom-right-radius: 4px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #444444;
}

/* Checkboxes */
QCheckBox {
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    background-color: #252526;
    border: 1px solid #555555;
    border-radius: 3px;
}
QCheckBox::indicator:hover {
    border: 1px solid #007acc;
}
QCheckBox::indicator:checked {
    background-color: #007acc;
    border: 1px solid #005a9e;
}
QCheckBox::indicator:checked:hover {
    background-color: #0098ff;
}
QCheckBox::indicator:disabled {
    background-color: #2a2a2a;
    border: 1px solid #444444;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background-color: #1e1e1e;
    width: 12px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background-color: #424242;
    min-height: 20px;
    border-radius: 6px;
}
QScrollBar::handle:vertical:hover {
    background-color: #4f4f4f;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    border: none;
    background-color: #1e1e1e;
    height: 12px;
    margin: 0px;
}
QScrollBar::handle:horizontal {
    background-color: #424242;
    min-width: 20px;
    border-radius: 6px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #4f4f4f;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Progress Bar */
QProgressBar {
    background-color: #252526;
    border: 1px solid #444444;
    border-radius: 4px;
    text-align: center;
    color: #ffffff;
}
QProgressBar::chunk {
    background-color: #007acc;
    border-radius: 3px;
}

/* Main Window Status Bar */
QStatusBar {
    background-color: #007acc;
    color: #ffffff;
}
QStatusBar::item {
    border: none;
}
"""
