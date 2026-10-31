"""LTX Studio — entrypoint. Avvia la GUI PySide6.

Esegui con il python del venv di ComfyUI:
  E:\\AI\\Comfy_Nexus\\ComfyUI\\venv\\Scripts\\python.exe E:\\AI\\ltxstudio\\app.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Verifica TLS contro lo store di certificati del sistema, non contro il bundle certifi: dietro un
# antivirus/proxy che ispeziona HTTPS certifi non riconosce la CA e ogni connessione fallisce (nel
# .exe compilato in modo sistematico). Prima di qualunque import che possa aprire connessioni.
from core.net_tls import use_system_ca                       # noqa: E402
use_system_ca()

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
    try:                                                   # messaggi non modali, selezionabili, scorrevoli (punto 8)
        from ui.widgets.msg_dialog import install as _install_msgs
        _install_msgs()
    except Exception:                                      # noqa: BLE001
        pass
    win = MainWindow()
    win.show()
    if os.environ.get("LTX_TEST_DRIVER") == "1" and not getattr(sys, "frozen", False) and "__compiled__" not in globals():              # solo per i test dal vivo (tests/app_live); MAI nelle build (frozen/Nuitka)
        try:
            import importlib.util
            _sp = importlib.util.spec_from_file_location("ltx_test_driver", str(Path(__file__).resolve().parent / "tests" / "app_live" / "driver.py"))
            _m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m)
            _m.attach(win)
        except Exception as e:                              # noqa: BLE001
            print(f"[test-driver] non attivato: {e}", flush=True)
    # Primo avvio: wizard «Setup modelli» (cartelle → collega i locali → scarica i mancanti dell'edizione).
    try:
        from ui.setup import first_run
        first_run.maybe_show(win)
    except Exception:                                       # noqa: BLE001 — non deve mai bloccare l'avvio
        pass
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
