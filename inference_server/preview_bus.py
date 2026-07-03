"""Bus preview/progresso per-step tra i backend (image_flux) e l'endpoint /preview.

I job dell'InferenceServer girano SERIALI (un solo job GPU alla volta), quindi un unico stato
globale basta. Due canali:
- PROGRESSO (step/total): aggiornato ad OGNI step (leggerissimo) → barra credibile.
- PREVIEW (png): aggiornato ogni ~1/8 degli step (decode latent2rgb) → immagine dell'anteprima.
La GUI polla GET /preview: legge sempre step/total dagli header; ridisegna il PNG solo quando
`png_seq` cambia. Best-effort: non deve MAI rompere la generazione.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_latest = {"step": 0, "total": 0, "png_seq": 0, "png": b""}


def set_progress(step: int, total: int) -> None:
    """Aggiorna solo il progresso (ogni step, senza decodificare l'anteprima)."""
    with _lock:
        _latest["step"] = int(step)
        _latest["total"] = int(total)


def set_preview(png: bytes, step: int, total: int) -> None:
    """Aggiorna l'anteprima (PNG) + il progresso. png_seq cresce → la GUI ridisegna."""
    with _lock:
        _latest["step"] = int(step)
        _latest["total"] = int(total)
        if png:
            _latest["png_seq"] += 1
            _latest["png"] = png


def get_preview() -> dict:
    """Ritorna {step, total, png_seq, png}."""
    with _lock:
        return dict(_latest)


def reset() -> None:
    """Azzera all'inizio di un nuovo job (niente anteprima/progresso del job precedente)."""
    with _lock:
        _latest.update(step=0, total=0, png_seq=0, png=b"")
