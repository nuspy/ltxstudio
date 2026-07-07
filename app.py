"""LTX Studio — entrypoint. Avvia la GUI PySide6.

Esegui con il python del venv di ComfyUI:
  E:\\AI\\Comfy_Nexus\\ComfyUI\\venv\\Scripts\\python.exe E:\\AI\\ltxstudio\\app.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Visore 3D SENZA WebEngine di default. PROVATO: su questa GPU (RTX 5090/Blackwell) il processo GPU di Chromium
# fallisce (kFatalFailure) e **crasha l'intera app** durante la generazione 3D — mentre il subprocess genera il
# GLB regolarmente (log proprio del subprocess arriva a OK: mentre l'app è già morta). Con il visore semplice
# (pulsanti Apri/Salva, GLB apribile nel visore 3D di Windows) l'app NON crasha. Per riprovare il visore WebGL
# interattivo: LTX_NO_WEBGL=0. Tocca SOLO i 3 tab 3D; immagini/video/musica non usano WebEngine.
os.environ.setdefault("LTX_NO_WEBGL", "1")

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
