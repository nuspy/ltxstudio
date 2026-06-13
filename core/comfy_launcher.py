"""Avvio automatico di ComfyUI se non è già in esecuzione.

L'app parla con ComfyUI su 127.0.0.1:8188. Se a startup non risponde, lanciamo il launcher
UFFICIALE dell'utente (`run_nvidia.bat`: venv + flag sage/fp8/reserve-vram/…) in una FINESTRA
SEPARATA, così ComfyUI resta vivo indipendentemente dall'app e mostra i suoi log. Riusare il .bat
evita di duplicare (e far divergere) la configurazione di avvio.

Il polling esistente in MainWindow (`_refresh_status`, ogni 3s) rileva da solo quando ComfyUI è
pronto e carica l'inventario: qui ci limitiamo a far partire il processo.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

COMFY_DIR = Path(r"E:\AI\Comfy_Nexus\ComfyUI")
LAUNCH_BAT = COMFY_DIR / "run_nvidia.bat"


def launch() -> tuple[bool, str]:
    """Lancia ComfyUI (`run_nvidia.bat`) in una console separata.

    Ritorna (avviato?, messaggio). NON attende che ComfyUI sia pronto (ci pensa il polling).
    """
    if sys.platform != "win32":
        return False, "auto-avvio ComfyUI supportato solo su Windows"
    if not LAUNCH_BAT.exists():
        return False, f"launcher non trovato: {LAUNCH_BAT}"
    try:
        # CREATE_NEW_CONSOLE → finestra propria (log ComfyUI visibili) e processo indipendente che
        # sopravvive alla chiusura di LTX Studio. CREATE_NEW_PROCESS_GROUP → non eredita il Ctrl-C.
        flags = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(["cmd", "/c", str(LAUNCH_BAT)], cwd=str(COMFY_DIR),
                         creationflags=flags, close_fds=True)
        return True, f"ComfyUI avviato ({LAUNCH_BAT.name}) — pronto tra ~30-90s"
    except Exception as e:                       # noqa: BLE001
        return False, f"avvio ComfyUI fallito: {e}"
