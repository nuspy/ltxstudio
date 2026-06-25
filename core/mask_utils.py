"""Gestione maschere in PIL puro — sostituisce l'endpoint ComfyUI `/upload/mask` (alpha-merge).

ComfyUI, quando carichi una maschera, prende il canale ALFA della maschera e lo scrive come canale
ALFA dell'immagine di riferimento (poi `LoadImage` espone quell'alfa come MASK). Qui replichiamo la
stessa semantica SENZA ComfyUI, così inpaint/outpaint nativi possono usare le maschere create dalla GUI.

API:
  merge_mask_into_alpha(ref_image, mask, out) -> str   # RGBA con alpha = maschera (come ComfyUI)
  load_mask_array(mask) -> np.ndarray                   # [H,W] float 0..1 (1 = area mascherata)
  invert_mask(mask, out) -> str
"""
from __future__ import annotations

import os


def _mask_channel(mask_img):
    """Estrae il canale-maschera da un'immagine: usa l'ALFA se presente (convenzione ComfyUI),
    altrimenti la luminanza (L). Ritorna un'immagine 'L' (8-bit)."""
    if "A" in mask_img.getbands():
        return mask_img.getchannel("A")
    return mask_img.convert("L")


def merge_mask_into_alpha(ref_image_path: str, mask_path: str, out_path: str) -> str:
    """Scrive `out_path` = immagine di riferimento RGBA col canale ALFA preso dalla maschera
    (identico a ciò che fa il server ComfyUI su /upload/mask). La maschera viene ridimensionata
    alla risoluzione del riferimento se necessario."""
    from PIL import Image
    ref = Image.open(ref_image_path).convert("RGBA")
    mask = Image.open(mask_path)
    alpha = _mask_channel(mask)
    if alpha.size != ref.size:
        alpha = alpha.resize(ref.size, Image.LANCZOS)
    ref.putalpha(alpha)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ref.save(out_path)
    return out_path


def load_mask_array(mask_path: str):
    """Carica la maschera come array float [H,W] in 0..1 (1 = area selezionata/da inpaintare).
    Usa l'alfa se presente, altrimenti la luminanza."""
    import numpy as np
    from PIL import Image
    m = _mask_channel(Image.open(mask_path))
    return np.asarray(m, dtype="float32") / 255.0


def invert_mask(mask_path: str, out_path: str) -> str:
    """Inverte una maschera (utile per outpaint: la regione da generare è il complemento)."""
    from PIL import Image, ImageOps
    m = _mask_channel(Image.open(mask_path))
    inv = ImageOps.invert(m)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    inv.save(out_path)
    return out_path
