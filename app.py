"""LTX Studio — entrypoint. Avvia la GUI PySide6.

Esegui con il python del venv di ComfyUI:
  E:\\AI\\Comfy_Nexus\\ComfyUI\\venv\\Scripts\\python.exe E:\\AI\\ltxstudio\\app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtCore import Qt                # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from ui.main_window import MainWindow       # noqa: E402
from ui.theme import apply_theme            # noqa: E402


def main():
    # QtWebEngine (visore 3D three.js/WebGL) RICHIEDE il contesto OpenGL condiviso impostato PRIMA di creare la
    # QApplication; senza, su Windows il processo GPU di Chromium crasha silenziosamente al rendering del modello
    # → l'app si chiude senza log («modello 3D crasha dopo l'avvio»). Deve stare QUI, prima di tutto.
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    app.setApplicationName("LTX Studio")
    apply_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
