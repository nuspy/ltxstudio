"""Mappe di controllo per i ControlNet delle immagini (FLUX.1 Union, Z-Image Union): foto → mappa alla forma finale.

Una sola funzione, `make_control_map`, usata dai backend (`inference_server/backends/image_control.py`) e dai test.
La foto guida viene RITAGLIATA al rapporto del risultato (mai stirata: regola «output = rapporto del canvas») e poi
trasformata:  canny | depth | pose | gray | none  («none» = l'immagine è già una mappa).
Nessun import di torch a livello modulo (importabile dal processo GUI); depth/pose caricano i loro modelli solo se
chiesti, su CPU di default (il processo GUI non apre mai un contesto CUDA).
"""
from __future__ import annotations

import os
import tempfile

KINDS = ("canny", "depth", "pose", "gray", "none")

# Valori consigliati dagli autori dei ControlNet (model card Shakker-Labs FLUX.1-dev-ControlNet-Union-Pro-2.0,
# verificati il 17/10/2026): kind → (controlnet_conditioning_scale, control_guidance_end)
FLUX_UNION_DEFAULTS = {
    "canny": (0.7, 0.8), "depth": (0.8, 0.8), "pose": (0.9, 0.65), "gray": (0.9, 0.8), "none": (0.7, 0.8),
}


def fit_crop(img, width: int, height: int):
    """Ritaglio centrato alla forma finale (mai stirato)."""
    from PIL import Image, ImageOps
    return ImageOps.fit(img.convert("RGB"), (int(width), int(height)), Image.LANCZOS)


def make_control_map(path: str, kind: str, width: int, height: int, *, device: int = -1):
    """Foto → mappa di controllo PIL RGB di `width`×`height`. `device`: -1 CPU, 0 prima GPU (solo nei processi figli)."""
    from PIL import Image
    kind = (kind or "none").lower()
    if kind not in KINDS:
        raise ValueError(f"preprocessore di controllo sconosciuto: {kind} (validi: {', '.join(KINDS)})")
    img = fit_crop(Image.open(path), width, height)
    if kind == "none":
        return img
    if kind == "gray":
        return img.convert("L").convert("RGB")
    if kind == "canny":
        import cv2
        import numpy as np
        e = cv2.Canny(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY), 100, 200)
        return Image.fromarray(e).convert("RGB")
    # depth / pose: modelli in core.controlnet_preprocess, che lavorano da file → file temporanei
    from core import controlnet_preprocess as cp
    with tempfile.TemporaryDirectory(prefix="ctlmap_") as td:
        src = os.path.join(td, "in.png")
        dst = os.path.join(td, "out.png")
        img.save(src)
        if kind == "depth":
            cp.depth(src, dst, device=device)
        else:
            cp.pose(src, dst)
        out = Image.open(dst).convert("RGB")
        out.load()
    return out.resize((int(width), int(height)), Image.LANCZOS) if out.size != (int(width), int(height)) else out
