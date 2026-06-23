"""Avvio automatico del server NATIVO (InferenceServer, diffusers) se non già attivo.

Sostituisce `comfy_launcher` nella migrazione a zero-ComfyUI: l'app parla col backend nativo
su 127.0.0.1:8189. Se a startup non risponde, lanciamo `python -m inference_server.server`
(venv dedicato) in una finestra separata, con le ottimizzazioni attive (Sage attention auto +
FBCache). Il polling di MainWindow rileva quando è pronto.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

VENV_PY = Path(r"E:\AI\Tools\inference_server_venv\Scripts\python.exe")
PROJECT = Path(__file__).resolve().parents[1]


def launch() -> tuple[bool, str]:
    """Lancia l'InferenceServer nativo (porta 8189) in un processo indipendente.
    Ritorna (avviato?, messaggio). NON attende che sia pronto (ci pensa il polling)."""
    py = VENV_PY if VENV_PY.exists() else Path(sys.executable)
    if not Path(py).exists():
        return False, f"venv nativo non trovato: {VENV_PY}"
    try:
        env = dict(os.environ)
        env.setdefault("NATIVE_ATTENTION", "auto")          # Sage se disponibile, else SDPA
        env.setdefault("NATIVE_FBCACHE", "1")               # First Block Cache
        env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen([str(py), "-m", "inference_server.server"], cwd=str(PROJECT),
                         env=env, creationflags=flags, close_fds=True)
        return True, "Server nativo avviato (porta 8189) — pronto tra ~10-30s"
    except Exception as e:                                   # noqa: BLE001
        return False, f"avvio server nativo fallito: {e}"
