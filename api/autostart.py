"""Autostart dell'API: avvia il server se non risponde, e attende che sia pronto.

Usato dallo skill/agente prima di lanciare job. L'API poi si spegne da sola dopo inattività
(vedi IDLE_TIMEOUT in api/app.py). 100% locale.

CLI:
  python -m api.autostart            # assicura che l'API sia su, stampa la base URL
  python -m api.autostart --status   # solo stato
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request

from api.config import BASE_URL, REPO

# Python che esegue l'API (venv con fastapi + accesso a core/*). Override: LTX_API_PYTHON.
# Default: lo STESSO interprete dell'app (prima puntava al venv ComfyUI legacy, poi rinominato → morto).
_legacy_py = r"E:\AI\Comfy_Nexus.bak\ComfyUI\venv\Scripts\python.exe"
API_PYTHON = os.environ.get("LTX_API_PYTHON") or (
    _legacy_py if os.path.exists(_legacy_py) else sys.executable)
_CREATE_NEW_CONSOLE = 0x00000010  # Windows: processo indipendente


def is_up(timeout: float = 1.5) -> bool:
    try:
        urllib.request.urlopen(BASE_URL + "/health", timeout=timeout)
        return True
    except Exception:                                        # noqa: BLE001
        return False


def ensure_api_up(wait: float = 90.0) -> bool:
    """Avvia l'API se giù e attende /health. Ritorna True se pronta."""
    if is_up():
        return True
    py = API_PYTHON if os.path.isfile(API_PYTHON) else sys.executable
    creation = _CREATE_NEW_CONSOLE if os.name == "nt" else 0
    subprocess.Popen([py, "-m", "api.app"], cwd=str(REPO),
                     creationflags=creation) if os.name == "nt" else \
        subprocess.Popen([py, "-m", "api.app"], cwd=str(REPO),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < wait:
        if is_up():
            return True
        time.sleep(1.5)
    return False


if __name__ == "__main__":
    if "--status" in sys.argv:
        print(f"{'UP' if is_up() else 'DOWN'} {BASE_URL}")
    else:
        ok = ensure_api_up()
        print(f"{'READY' if ok else 'FAILED'} {BASE_URL}")
        sys.exit(0 if ok else 1)
