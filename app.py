"""LTX Studio — entrypoint. Avvia la GUI PySide6.

Esegui con il python del venv di ComfyUI:
  E:\\AI\\Comfy_Nexus\\ComfyUI\\venv\\Scripts\\python.exe E:\\AI\\ltxstudio\\app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import QApplication  # noqa: E402
from ui.main_window import MainWindow       # noqa: E402
from ui.theme import apply_theme            # noqa: E402


def main():
    # NB: NON disabilitiamo la GPU e NON forziamo AA_ShareOpenGLContexts — su RTX 5090/Blackwell quel flag
    # causava "Failed to create shared context for virtualization" (kFatalFailure). La GPU serve piena alla
    # generazione ed era ampiamente libera. Il crash 3D era il subprocess CUDA (attention backend), non il visore.
    app = QApplication(sys.argv)
    app.setApplicationName("LTX Studio")
    apply_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
