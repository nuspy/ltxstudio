"""Canale preview/progresso su FILE per i worker VIDEO (sottoprocessi, venv diversi).

Le immagini usano il bus HTTP del server :8189; i video invece girano come sottoprocesso
(ltx2_infer / comfy-native) e non condividono quel bus. Qui il worker scrive progresso+anteprima
su un file noto (stdlib, nessuna dipendenza) e la GUI lo polla. Job seriali → un solo stato globale.

Scritture atomiche (tmp+replace). Best-effort: non deve MAI rompere la generazione.
"""
from __future__ import annotations

import json
import os
import tempfile

_DIR = os.path.join(tempfile.gettempdir(), "ltxstudio_preview")
_PNG = os.path.join(_DIR, "vpreview.png")
_META = os.path.join(_DIR, "vpreview.json")


def _ensure() -> None:
    os.makedirs(_DIR, exist_ok=True)


def reset() -> None:
    """Azzera all'inizio di un nuovo job video."""
    _ensure()
    for f in (_PNG, _META):
        try:
            if os.path.exists(f):
                os.remove(f)
        except OSError:
            pass


def _read_meta() -> dict:
    try:
        with open(_META, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                       # noqa: BLE001
        return {}


def _write_meta(m: dict) -> None:
    try:
        tmp = _META + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(m, fh)
        os.replace(tmp, _META)
    except OSError:
        pass


def write_message(msg: str) -> None:
    """Fase corrente (es. 'Caricamento componenti…', 'Generazione…')."""
    _ensure()
    m = _read_meta()
    m["msg"] = str(msg or "")
    m.setdefault("png_seq", 0)
    _write_meta(m)


def write_progress(step: int, total: int) -> None:
    """Aggiorna solo il progresso (ogni step)."""
    _ensure()
    m = _read_meta()
    m.update(step=int(step), total=int(total))
    m.setdefault("png_seq", 0)
    _write_meta(m)


def write_preview(png: bytes, step: int, total: int) -> None:
    """Scrive l'anteprima (PNG) + progresso; png_seq++ così la GUI ridisegna."""
    _ensure()
    m = _read_meta()
    seq = int(m.get("png_seq", 0)) + 1
    try:
        tmp = _PNG + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(png)
        os.replace(tmp, _PNG)
    except OSError:
        return
    m.update(step=int(step), total=int(total), png_seq=seq)
    try:
        tmp = _META + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(m, fh)
        os.replace(tmp, _META)
    except OSError:
        pass


def read() -> dict | None:
    """Meta corrente {step, total, png_seq} (senza PNG). None se assente."""
    m = _read_meta()
    return m or None


def read_png() -> bytes:
    """Ultimo PNG di anteprima (vuoto se assente)."""
    try:
        with open(_PNG, "rb") as fh:
            return fh.read()
    except Exception:                                       # noqa: BLE001
        return b""
