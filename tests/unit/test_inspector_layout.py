"""The narrow inspector must never trap controls beyond its right edge."""
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QFormLayout, QScrollArea

from nvap.ui.control_panel import ControlPanel
from nvap.ui.theme import DARK_THEME_QSS


@pytest.mark.parametrize("width", [260, 340, 520])
def test_narrow_inspector_controls_remain_reachable(qtbot, width):
    panel = ControlPanel()
    qtbot.addWidget(panel)
    panel.setStyleSheet(DARK_THEME_QSS)
    panel.resize(width, 800)
    panel.show()
    scroll = panel.findChild(QScrollArea)
    qtbot.wait(50)
    assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert all(form.rowWrapPolicy() == QFormLayout.RowWrapPolicy.WrapLongRows
               for form in scroll.widget().findChildren(QFormLayout))
    # Scroll to the far-right cell navigation button, previously clipped.
    scroll.ensureWidgetVisible(panel.microglia_next)
    qtbot.wait(50)
    corner = panel.microglia_next.mapTo(scroll.viewport(), QPoint(panel.microglia_next.width() - 1, 0))
    assert 0 <= corner.x() < scroll.viewport().width()
    assert scroll.horizontalScrollBar().maximum() >= scroll.widget().width() - scroll.viewport().width()
