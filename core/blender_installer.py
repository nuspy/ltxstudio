"""Rilevamento e installazione automatica di Blender (per SkinTokens / rig).

SkinTokens, su Python 3.12, esporta SEMPRE tramite Blender headless: serve il
binario Blender 4.0+ raggiungibile via `SKINTOKEN_BLENDER_BIN` o sul PATH.

Questo modulo:
- `find_blender()`     → individua blender.exe (env, PATH, percorsi comuni, winget)
- `install_blender()`  → installa Blender via winget (blocking)
- `persist_blender_env(path)` → salva SKINTOKEN_BLENDER_BIN per i processi futuri
- `BlenderInstallWorker` → QThread per l'install non bloccante dalla GUI

NB: la variabile va vista dal processo ComfyUI (separato). Dopo l'install va
riavviato ComfyUI perché il nodo SkinToken la legga.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from PySide6.QtCore import QThread, Signal

WINGET_ID = "BlenderFoundation.Blender"
ENV_VAR = "SKINTOKEN_BLENDER_BIN"


def find_blender() -> str | None:
    """Ritorna il path a blender.exe se individuabile, altrimenti None."""
    # 1) variabile d'ambiente esplicita
    env = os.environ.get(ENV_VAR)
    if env and Path(env).exists():
        return env
    # 2) sul PATH
    which = shutil.which("blender")
    if which:
        return which
    # 3) percorsi d'installazione comuni (winget / installer ufficiale)
    candidates: list[Path] = []
    for root in (
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Blender Foundation",
        Path(os.environ.get("ProgramW6432", r"C:\Program Files")) / "Blender Foundation",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
    ):
        if root.exists():
            candidates += list(root.glob("**/blender.exe"))
    if candidates:
        # preferisci la versione più recente (path con numero più alto)
        candidates.sort(key=lambda p: str(p), reverse=True)
        return str(candidates[0])
    return None


def winget_available() -> bool:
    return shutil.which("winget") is not None


def install_blender(timeout: int = 1800) -> tuple[bool, str]:
    """Installa Blender via winget (blocking). Ritorna (ok, messaggio)."""
    if not winget_available():
        return False, "winget non disponibile su questo sistema."
    cmd = [
        "winget", "install", "-e", "--id", WINGET_ID,
        "--accept-source-agreements", "--accept-package-agreements",
        "--disable-interactivity",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "Timeout durante l'installazione di Blender."
    except Exception as e:
        return False, f"Errore winget: {e}"
    out = (proc.stdout or "") + (proc.stderr or "")
    # winget può restituire 0 anche con "già installato"; consideriamo ok se troviamo il binario
    path = find_blender()
    if path:
        return True, path
    if proc.returncode == 0:
        return True, out.strip()[-500:]
    return False, out.strip()[-800:] or f"winget exit {proc.returncode}"


def persist_blender_env(blender_path: str) -> bool:
    """Imposta SKINTOKEN_BLENDER_BIN nel processo corrente e in modo persistente (setx)."""
    os.environ[ENV_VAR] = blender_path
    try:
        # setx persiste per i processi FUTURI dell'utente (non per quelli già avviati)
        subprocess.run(["setx", ENV_VAR, blender_path],
                       capture_output=True, text=True, timeout=30)
        return True
    except Exception:
        return False


class BlenderInstallWorker(QThread):
    """Esegue l'installazione di Blender in un thread separato (non blocca la GUI)."""

    finished_ok = Signal(str)    # path blender.exe
    failed = Signal(str)         # messaggio errore

    def run(self):
        ok, msg = install_blender()
        if ok:
            path = find_blender() or msg
            persist_blender_env(path)
            self.finished_ok.emit(path)
        else:
            self.failed.emit(msg)
