"""Spina dorsale 3D: esegue job dentro Blender headless (binario, non modulo bpy).

LTX Studio gira nel venv di ComfyUI; per le operazioni mesh universali (conversione
di formato, separazione/cancellazione oggetti, retarget animazione) usiamo il
BINARIO di Blender — lo stesso individuato da `core.blender_installer.find_blender()`
e già usato da SkinTokens per l'export headless.

Pattern:
  run_blender_job("convert.py", {...}) ->
    blender --background --python scripts/blender/convert.py -- '<json>'
  Lo script Blender stampa una riga marcata `@@RESULT@@{...json...}` su stdout;
  qui la parsiamo e la restituiamo. Tutto il resto dello stdout/stderr finisce nel
  log (Blender è molto verboso).

`BlenderWorker(QThread)` esegue il job senza bloccare la GUI (sul modello di
`BlenderInstallWorker`): segnali `finished_ok(dict)` / `failed(str)`.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from core import blender_installer

# Cartella degli script eseguiti DENTRO Blender (questi usano `import bpy`).
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "blender"

# Marcatore della riga di risultato JSON emessa dagli script Blender.
RESULT_MARKER = "@@RESULT@@"

# Windows: niente finestra console per il sottoprocesso.
_CREATE_NO_WINDOW = 0x08000000


def blender_available() -> bool:
    return blender_installer.find_blender() is not None


def run_blender_job(script_name: str, args: dict, timeout: int = 900) -> dict:
    """Esegue `scripts/blender/<script_name>` in Blender headless con `args` (JSON).

    Ritorna il dict emesso dallo script sulla riga `@@RESULT@@{...}`.
    Solleva RuntimeError se Blender non è installato, il job fallisce o non
    produce una riga di risultato.
    """
    blender = blender_installer.find_blender()
    if not blender:
        raise RuntimeError(
            "Blender non trovato. Installa Blender (Strumenti → installa Blender) "
            "o imposta SKINTOKEN_BLENDER_BIN.")
    script = SCRIPTS_DIR / script_name
    if not script.is_file():
        raise RuntimeError(f"Script Blender mancante: {script}")

    cmd = [blender, "--background", "--factory-startup",
           "--python", str(script), "--", json.dumps(args)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            creationflags=_CREATE_NO_WINDOW)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Timeout ({timeout}s) nel job Blender '{script_name}'.")

    result: dict | None = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith(RESULT_MARKER):
            try:
                result = json.loads(line[len(RESULT_MARKER):])
            except json.JSONDecodeError:
                result = None
    if result is None:
        tail = ((proc.stderr or "") + "\n" + (proc.stdout or ""))[-1200:]
        raise RuntimeError(
            f"Job Blender '{script_name}' senza risultato (exit {proc.returncode}).\n{tail}")
    if not result.get("ok", False):
        raise RuntimeError(result.get("error", "errore sconosciuto nel job Blender"))
    return result


class BlenderWorker(QThread):
    """Esegue un job Blender in un thread separato (non blocca la GUI)."""

    finished_ok = Signal(dict)   # risultato JSON dello script
    failed = Signal(str)         # messaggio d'errore
    log = Signal(str)            # riga di log (best-effort)

    def __init__(self, script_name: str, args: dict, timeout: int = 900, parent=None):
        super().__init__(parent)
        self._script = script_name
        self._args = args
        self._timeout = timeout

    def run(self):
        try:
            res = run_blender_job(self._script, self._args, self._timeout)
            self.finished_ok.emit(res)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))
