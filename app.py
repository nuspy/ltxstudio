"""LTX Studio — entrypoint. Avvia la GUI PySide6.

Esegui con il python del venv di ComfyUI:
  E:\\AI\\Comfy_Nexus\\ComfyUI\\venv\\Scripts\\python.exe E:\\AI\\ltxstudio\\app.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# SOLO il VISORE 3D (WebEngine/Chromium) in rendering SOFTWARE (SwiftShader). Motivo: la generazione 3D gira in
# un subprocess CUDA che satura la GPU; se l'app tiene processi GPU Chromium, il driver va in TDR (hang → reset)
# e freeza/uccide app + subprocess (il subprocess DA SOLO funziona). --disable-gpu tocca SOLO Chromium: la
# generazione immagini/video/3D usa la GPU PIENA a prescindere. NIENTE AA_ShareOpenGLContexts (su Blackwell dava
# kFatalFailure). Override: se non usi il 3D e vuoi il visore GPU, esporta QTWEBENGINE_CHROMIUM_FLAGS vuoto.
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS",
                      "--disable-gpu --disable-gpu-compositing --enable-unsafe-swiftshader")

from PySide6.QtWidgets import QApplication  # noqa: E402
from ui.main_window import MainWindow       # noqa: E402
from ui.theme import apply_theme            # noqa: E402


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("LTX Studio")
    apply_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
