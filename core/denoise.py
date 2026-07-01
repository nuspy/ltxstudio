"""Denoise classico (niente modello/diffusione, niente download) per il restauro immagini.

Il controllo "denoise" del tab Migliora qualità → Foto era scollegato in entrambi i path
(nativo: mai passato; ComfyUI: VAEEncode costruito ma non connesso, "serve un modello").
Qui forniamo un denoise REALE non-diffusivo: cv2.fastNlMeansDenoisingColored (Non-Local Means),
con fallback a PIL MedianFilter se OpenCV non è disponibile. Riduce grana/rumore preservando i bordi.

strength ∈ [0.05, 0.50] (come lo spinbox GUI) → intensità h del Non-Local Means.
"""
from __future__ import annotations


def denoise_image(in_path: str, out_path: str | None = None, strength: float = 0.2) -> str:
    """Applica un denoise Non-Local Means a `in_path`. Ritorna il path di output (in-place se out_path=None)."""
    out_path = out_path or in_path
    s = max(0.0, min(1.0, float(strength)))
    # mappa 0.05–0.50 → h ~3–17 (luminanza). Colore leggermente più forte per togliere chroma-noise.
    h = 3.0 + s * 28.0
    try:
        import cv2
        import numpy as np  # noqa: F401
        img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"immagine non leggibile: {in_path}")
        den = cv2.fastNlMeansDenoisingColored(img, None, h, h * 1.1, 7, 21)
        cv2.imwrite(out_path, den)
        return out_path
    except Exception:                                        # noqa: BLE001 — fallback senza OpenCV
        from PIL import Image, ImageFilter
        im = Image.open(in_path).convert("RGB")
        # MedianFilter con raggio proporzionale alla forza (3 o 5)
        radius = 5 if s >= 0.3 else 3
        im = im.filter(ImageFilter.MedianFilter(size=radius))
        im.save(out_path)
        return out_path
