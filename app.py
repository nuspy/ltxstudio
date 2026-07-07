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
    # WebEngine (visore 3D) in RENDERING SOFTWARE: il processo GPU di Chromium NON deve competere con il
    # subprocess CUDA della generazione 3D (TRELLIS). Con entrambi sulla stessa GPU il driver andava in
    # TDR/reset → app E subprocess morivano insieme a metà sampling (log tagliato di netto, senza errore).
    # SwiftShader (CPU) rende comunque la mesh nel visore. Solo i 3 tab 3D usano WebEngine; immagini/video/musica
    # NON lo usano → inalterati. Override: QTWEBENGINE_CHROMIUM_FLAGS per rimettere la GPU se non serve il 3D.
    import os as _os
    _os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
    # QtWebEngine richiede comunque il contesto OpenGL condiviso impostato PRIMA della QApplication.
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    app.setApplicationName("LTX Studio")
    apply_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
