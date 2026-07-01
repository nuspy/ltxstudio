"""Separazione voce/strumentale via demucs (htdemucs) — per il lip-sync 'Canta'.

Isolando la VOCE dalla canzone, il modello a2v sincronizza il labiale sul canto e non su
batteria/synth → lip-sync molto più pulito. demucs è già nel venv dell'app; il modello
htdemucs (~80MB) si scarica da torchhub al PRIMO uso.

Gira in-process (il main dell'app è il venv comfy che contiene demucs+torch). Se qualcosa va
storto, ritorna l'audio originale (fail-safe: la generazione continua comunque).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("demucs") is not None
    except Exception:                                        # noqa: BLE001
        return False


def separate_vocals(audio_path: str, out_dir: str | None = None, on_log=None) -> str:
    """Estrae la traccia vocale da `audio_path` (two-stems=vocals). Ritorna il path del
    vocals.wav prodotto, oppure l'audio originale se demucs non è disponibile/fallisce."""
    if not audio_path or not os.path.isfile(audio_path) or not available():
        return audio_path
    out_dir = out_dir or tempfile.mkdtemp(prefix="demucs_")
    cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
           "-o", out_dir, audio_path]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:                             # stream progress
            if on_log:
                on_log(line.rstrip())
        proc.wait()
        stem = Path(audio_path).stem
        voc = Path(out_dir) / "htdemucs" / stem / "vocals.wav"
        return str(voc) if voc.is_file() else audio_path
    except Exception:                                        # noqa: BLE001 — fail-safe
        return audio_path
