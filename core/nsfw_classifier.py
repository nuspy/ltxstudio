"""Classificatore NSFW per immagini di preview — Falconsai/nsfw_image_detection (#17).

ViT binario (normal/nsfw), ~98% acc., gira in locale/offline DOPO il primo download. Usato come step della
cascata SFW/NSFW dell'importer per i file locali senza corrispondenza Civitai.

Metered-friendly: NON scarica nulla implicitamente. `classify_image` usa il modello solo se GIÀ in cache
(o se `allow_download=True`, scelta esplicita dell'utente al primo uso). Se non disponibile → ritorna None
(il chiamante prosegue con la sola euristica).
"""
from __future__ import annotations

from pathlib import Path

_MODEL_ID = "Falconsai/nsfw_image_detection"
_pipe = None
_load_failed = False


def is_available() -> bool:
    """True se il modello è già in cache locale (nessun download)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        # il config.json presente in cache è sufficiente come segnale di snapshot scaricato
        hit = try_to_load_from_cache(_MODEL_ID, "config.json")
        return isinstance(hit, str) and Path(hit).exists()
    except Exception:                                        # noqa: BLE001
        return False


def _ensure_pipe(allow_download: bool):
    global _pipe, _load_failed
    if _pipe is not None or _load_failed:
        return _pipe
    if not allow_download and not is_available():
        return None
    try:
        import os
        if not allow_download:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import pipeline
        _pipe = pipeline("image-classification", model=_MODEL_ID)
    except Exception:                                        # noqa: BLE001
        _load_failed = True
        _pipe = None
    return _pipe


def classify_image(path: str | Path, allow_download: bool = False) -> tuple[str, float] | None:
    """Ritorna ("normal"|"nsfw", score) oppure None se il classificatore non è disponibile/fallisce."""
    pipe = _ensure_pipe(allow_download)
    if pipe is None:
        return None
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        res = pipe(img)                                      # [{'label':'nsfw','score':..}, ...]
        best = max(res, key=lambda r: r["score"])
        return str(best["label"]).lower(), float(best["score"])
    except Exception:                                        # noqa: BLE001
        return None


def is_nsfw(path: str | Path, allow_download: bool = False, thresh: float = 0.7) -> bool | None:
    """True/False se il classificatore decide con confidenza ≥ thresh, altrimenti None (indeciso/assente)."""
    r = classify_image(path, allow_download)
    if r is None:
        return None
    label, score = r
    if label == "nsfw" and score >= thresh:
        return True
    if label == "normal" and score >= thresh:
        return False
    return None
