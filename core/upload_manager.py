"""Staging input locale indipendente da ComfyUI (Fase E: eliminare ComfyUI).

Oggi gli input (immagini/maschere/video) vengono caricati nella cartella `ComfyUI/input` via REST.
I backend NATIVI (diffusers, inference_server) leggono direttamente da disco: basta uno staging locale.
Questo modulo copia/salva gli input in una cartella di lavoro dell'app e ritorna un PATH ASSOLUTO,
senza dipendere da ComfyUI. È additivo: i consumatori nativi lo useranno al posto di `to_comfy_image`.

API:
  staged_input_dir() -> str
  stage_file(path) -> str            # copia un file locale nello staging, ritorna il path assoluto
  stage_bytes(data, suffix) -> str   # salva bytes (es. upload HTTP) nello staging
  clear_old(max_age_h=24) -> int     # pulizia file vecchi (ritorna quanti rimossi)
"""
from __future__ import annotations

import os
import shutil
import time
import uuid

# cartella di staging (override con env LTXSTUDIO_INPUT_DIR); default sotto la root dell'app
_DEFAULT_DIR = os.environ.get("LTXSTUDIO_INPUT_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "inputs")


def staged_input_dir() -> str:
    os.makedirs(_DEFAULT_DIR, exist_ok=True)
    return _DEFAULT_DIR


def _unique(name: str) -> str:
    base, ext = os.path.splitext(os.path.basename(name) or "input")
    return os.path.join(staged_input_dir(), f"{base}_{uuid.uuid4().hex[:8]}{ext}")


def stage_file(path: str) -> str:
    """Copia un file locale nello staging e ritorna il path assoluto della copia.
    Se il file è già nello staging, lo ritorna invariato (idempotente)."""
    src = os.path.abspath(path)
    if not os.path.isfile(src):
        raise FileNotFoundError(path)
    if os.path.dirname(src) == os.path.abspath(staged_input_dir()):
        return src
    dst = _unique(src)
    shutil.copy2(src, dst)
    return dst


def stage_bytes(data: bytes, suffix: str = ".png") -> str:
    """Salva bytes (es. file caricato via HTTP) nello staging e ritorna il path assoluto."""
    if not suffix.startswith("."):
        suffix = "." + suffix
    dst = os.path.join(staged_input_dir(), f"upload_{uuid.uuid4().hex}{suffix}")
    with open(dst, "wb") as f:
        f.write(data)
    return dst


def clear_old(max_age_h: float = 24.0) -> int:
    """Rimuove dallo staging i file più vecchi di `max_age_h` ore. Ritorna il numero rimossi."""
    d = staged_input_dir()
    cutoff = time.time() - max_age_h * 3600.0
    n = 0
    for fn in os.listdir(d):
        fp = os.path.join(d, fn)
        try:
            if os.path.isfile(fp) and os.path.getmtime(fp) < cutoff:
                os.remove(fp); n += 1
        except OSError:
            pass
    return n
