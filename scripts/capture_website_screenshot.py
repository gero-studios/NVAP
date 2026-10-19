import sys
import os
import time
from pathlib import Path
from PySide6.QtCore import QUrl, QTimer, QSize, Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineWidgets import QWebEngineView

def capture():
    app = QApplication(sys.argv)
    view = QWebEngineView()
    view.resize(1440, 1080)

    root = Path(__file__).resolve().parents[1]
    html_url = QUrl.fromLocalFile(str(root / "website" / "index.html"))
    output_path = root / "manual_test_outputs" / "website-rendered-current.png"

    def on_load_finished(ok):
        if not ok:
            print("Failed to load page")
            app.quit()
            return

        # Wait a moment for canvas and Katex to initialize
        def do_capture():
            pixmap = view.grab()
            pixmap.save(str(output_path))
            print(f"Captured screenshot to {output_path}")
            app.quit()

        QTimer.singleShot(1500, do_capture)

    view.loadFinished.connect(on_load_finished)
    view.load(html_url)
    view.show()
    app.exec()

if __name__ == "__main__":
    capture()
