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

QTabWidget::pane {
    border: 1px solid #272e3b;
    border-radius: 8px;
    background-color: #101216;
    top: -1px;
}

QTabBar::tab {
    background-color: #151922;
    color: #7f8a99;
    border: 1px solid #272e3b;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 16px;
    margin-right: 2px;
    font-weight: 600;
}

QTabBar::tab:selected {
    background-color: #101216;
    color: #d8bd78;
    border-bottom: 1px solid #101216;
}

QTabBar::tab:hover:!selected {
    background-color: #1c222e;
    color: #eef2f6;
}

/* ── Inspector accordion (control panel) ──────────────────────────── */

QWidget#inspectorHeader {
    background-color: #101216;
}

QFrame#inspectorSeparator {
    background-color: #1e2530;
    max-height: 1px;
    border: none;
}

QWidget#inspectorSections {
    background-color: #101216;
}

QFrame#sectionHeader {
    background-color: #151922;
    border: 1px solid #1d2430;
    border-radius: 6px;
}

QFrame#sectionHeader:hover {
    background-color: #1b202a;
    border-color: #272e3b;
}

QLabel#sectionHeaderTitle {
    color: #d8bd78;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
}

QFrame#sectionBody {
    background-color: transparent;
    border: none;
}

/* ── App shell (sidebar + page stack) ─────────────────────────────── */

QFrame#appShell {
    background-color: #080a0d;
}

QFrame#appSidebar {
    background-color: #080a0d;
    border-right: 1px solid #1a1f28;
}

QFrame#sidebarLogo {
    background-color: transparent;
}

QLabel#sidebarAppName {
    color: #eef2f6;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 2.5px;
}

QLabel#sidebarVersion {
    color: #3a4352;
    font-size: 9px;
    letter-spacing: 0.5px;
    padding-bottom: 4px;
}

QFrame#sidebarSep {
    background-color: #181c24;
    max-height: 1px;
    border: none;
}

/* Nav buttons */
QPushButton#navButton {
    background-color: transparent;
    color: #6e7a8a;
    border: none;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 500;
    text-align: left;
}

QPushButton#navButton:hover {
    background-color: #141920;
    color: #d9dee7;
}

QPushButton#navButton:checked {
    background-color: #1b202a;
    color: #d8bd78;
}

/* Status pills in sidebar */
QFrame#statusPill {
    background-color: transparent;
    border-radius: 5px;
}

QLabel#statusPillLabel {
    color: #56606f;
    font-size: 10px;
    font-family: "Cascadia Mono", "Consolas", monospace;
}

/* Analytics / section pages */
QWidget#sectionPage {
    background-color: #101216;
}

QLabel#pageEyebrow {
    color: #56606f;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.0px;
    margin-bottom: 6px;
}

QLabel#pageTitle {
    color: #eef2f6;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 10px;
}

QLabel#pageIntro {
    color: #7f8a99;
    font-size: 13px;
    line-height: 1.5;
}

/* Metrics panel */
QWidget#metricsPanel,
QWidget#metricsPanelContent {
    background-color: #0d1018;
}

QLabel#panelSectionTitle {
    color: #d8bd78;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

QLabel#metricIcon {
    color: #d8bd78;
    font-size: 14px;
}

QLabel#metricLabel {
    color: #9aa4b2;
    font-size: 11px;
}

QLabel#metricValue {
    color: #eef2f6;
    font-size: 13px;
    font-weight: 600;
    font-family: "Cascadia Mono", "Consolas", monospace;
}

QFrame#metricSeparator {
    background-color: #1a1f28;
    max-height: 1px;
    border: none;
}

QLabel#summaryHint {
    color: #56606f;
    font-size: 11px;
    font-style: italic;
}

QFrame#statusCard {
    background-color: #131720;
    border: 1px solid #1e2530;
    border-radius: 8px;
}

QLabel#statusLabel {
    color: #7f8a99;
    font-size: 11px;
}

QLabel#statusDotPending {
    color: #3a4352;
    font-size: 10px;
}

QLabel#statusDotComplete {
    color: #69d9a3;
    font-size: 10px;
}

QLabel#statusTextPending {
    color: #56606f;
    font-size: 10px;
}

QLabel#statusTextComplete {
    color: #69d9a3;
    font-size: 10px;
}

QPushButton#linkButton {
    background-color: transparent;
    color: #d8bd78;
    border: none;
    padding: 2px 0;
    font-size: 11px;
}

QPushButton#linkButton:hover {
    color: #f0d98d;
}

/* Home page */
QWidget#homePage,
QWidget#homeContent,
QScrollArea#homeScroll {
    background-color: #101216;
}

QLabel#heroTitle {
    color: #eef2f6;
    font-size: 44px;
    font-weight: 750;
    letter-spacing: 0;
}

QLabel#heroSubtitle {
    color: #d8bd78;
    font-size: 16px;
    font-weight: 650;
}

QLabel#heroDesc {
    color: #9aa4b2;
    font-size: 13px;
    line-height: 1.45;
}

QPushButton#primaryActionHome {
    background-color: #d8bd78;
    color: #111318;
    border: 1px solid #f0d98d;
    border-radius: 8px;
    padding: 8px 14px;
    font-weight: 700;
    text-align: left;
}

QPushButton#primaryActionHome:hover {
    background-color: #f0d98d;
}

QPushButton#secondaryActionHome {
    background-color: #151922;
    color: #eef2f6;
    border: 1px solid #272e3b;
    border-radius: 8px;
    padding: 8px 14px;
    font-weight: 620;
    text-align: left;
}

QPushButton#secondaryActionHome:hover {
    background-color: #1b202a;
    border-color: #d8bd78;
}

QFrame#homePreviewBox {
    background-color: #0d1018;
    border: 1px solid #272e3b;
    border-radius: 8px;
}

QLabel#previewKicker {
    color: #d8bd78;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.8px;
    background-color: transparent;
}

QLabel#previewTitle {
    color: #eef2f6;
    font-size: 24px;
    font-weight: 700;
    background-color: transparent;
}

QLabel#previewBody {
    color: #9aa4b2;
    font-size: 12px;
    background-color: transparent;
}

QWidget#featureCard {
    background-color: #151922;
    border: 1px solid #272e3b;
    border-radius: 8px;
}

QWidget#featureCard:hover {
    background-color: #1b202a;
    border-color: #313947;
}

QLabel#featureCardIcon {
    background-color: transparent;
}

QLabel#featureCardTitle {
    color: #eef2f6;
    font-size: 13px;
    font-weight: 700;
    background-color: transparent;
}

QLabel#featureCardDesc {
    color: #9aa4b2;
    font-size: 11px;
    background-color: transparent;
}

QLabel#sectionTitle {
    color: #eef2f6;
    font-size: 18px;
    font-weight: 700;
}

QLabel#sectionSubtle {
    color: #56606f;
    font-size: 11px;
}

/* Settings / analytics cards */
QFrame#pageStatCard,
QFrame#analyticsMetricCard {
    background-color: #151922;
    border: 1px solid #272e3b;
    border-radius: 8px;
}

QLabel#pageStatValue,
QLabel#analyticsMetricValue {
    color: #eef2f6;
    font-size: 20px;
    font-weight: 700;
    font-family: "Cascadia Mono", "Consolas", monospace;
    background-color: transparent;
}

QLabel#pageStatLabel,
QLabel#analyticsMetricLabel {
    color: #d8bd78;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.0px;
    background-color: transparent;
}

QLabel#analyticsMetricHelper {
    color: #7f8a99;
    font-size: 11px;
    background-color: transparent;
}

QTableWidget#analyticsCellTable {
    background-color: #0c0f14;
    border: 1px solid #272e3b;
    border-radius: 8px;
}

QWidget:focus,
QPushButton:focus,
QTableWidget:focus,
QLineEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus {
    border-color: #d8bd78;
}
"""
