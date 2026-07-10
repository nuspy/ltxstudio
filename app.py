"""LTX Studio — entrypoint. Avvia la GUI PySide6.

Esegui con il python del venv di ComfyUI:
  E:\\AI\\Comfy_Nexus\\ComfyUI\\venv\\Scripts\\python.exe E:\\AI\\ltxstudio\\app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# NB: il visore 3D WebGL (WebEngine) è ATTIVO di default. Il crash 3D NON era WebEngine (era l'accesso Qt
# cross-thread in _log, corretto): gli errori Chromium "kFatalFailure" all'avvio sono rumore innocuo (ripiega su
# GLES2/software). Paracadute manuale se una GPU/driver desse problemi al visore: avviare con LTX_NO_WEBGL=1.
from PySide6.QtWidgets import QApplication  # noqa: E402
from ui.main_window import MainWindow       # noqa: E402
from ui.theme import apply_theme            # noqa: E402


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("LTX Studio")
    apply_theme(app)
    win = MainWindow()
    win.show()
    # Primo avvio: wizard «Setup modelli» (cartelle → collega i locali → scarica i mancanti dell'edizione).
    try:
        from ui.setup import first_run
        first_run.maybe_show(win)
    except Exception:                                       # noqa: BLE001 — non deve mai bloccare l'avvio
        pass
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
