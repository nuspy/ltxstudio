"""Bus preview per-step tra i backend (image_flux) e l'endpoint /preview.

I job dell'InferenceServer girano SERIALI (un solo job GPU alla volta, coda del worker),
quindi un unico "latest preview" globale è sufficiente: il callback di denoising vi scrive il
PNG dell'ultimo step decodificato, la GUI lo polla via GET /preview e lo mostra nella finestra
di anteprima live. Thread-safe. La preview è best-effort: non deve MAI rompere la generazione.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_latest = {"seq": 0, "step": 0, "total": 0, "png": b""}


def set_preview(png: bytes, step: int, total: int) -> None:
    """Registra l'anteprima dell'ultimo step (chiamato dal callback del backend)."""
    with _lock:
        _latest["seq"] += 1
        _latest["step"] = int(step)
        _latest["total"] = int(total)
        _latest["png"] = png or b""


def get_preview() -> dict:
    """Ritorna l'ultima anteprima {seq, step, total, png}. seq cresce ad ogni nuovo frame."""
    with _lock:
        return dict(_latest)


def reset() -> None:
    """Azzera all'inizio di un nuovo job (evita di mostrare l'anteprima del job precedente)."""
    with _lock:
        _latest.update(seq=0, step=0, total=0, png=b"")
