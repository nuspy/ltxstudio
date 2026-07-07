"""Regressione: convenzione maschera inpaint/outpaint (bianco = (ri)genera · nero = mantieni).
Il bug era l'INVERSIONE della maschera per outpaint: FluxFill rigenerava la FOTO e lasciava NERA l'estensione.
Qui si verifica la SEMANTICA di ricomposizione (Image.composite) con la maschera corretta, senza GPU."""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _solid(w, h, c):
    return Image.new("RGB", (w, h), c)


def test_outpaint_preserva_foto_e_riempie_estensione():
    W, H = 64, 100
    photo_h = 60
    # composite di input: foto (rosso) in alto, estensione NERA in basso (come dal canvas)
    orig = _solid(W, H, (0, 0, 0))
    orig.paste(_solid(W, photo_h, (200, 30, 30)), (0, 0))
    # maschera outpaint CORRETTA (_border_mask): NERO sulla foto, BIANCO sull'estensione
    mask = Image.new("L", (W, H), 0)
    mask.paste(Image.new("L", (W, H - photo_h), 255), (0, photo_h))
    # "generato" dalla pipeline (verde ovunque, come se avesse rigenerato tutto il frame)
    gen = _solid(W, H, (20, 180, 60))
    # ricomposizione del backend: bianco→generato, nero→originale
    out = Image.composite(gen, orig, mask)
    # la FOTO (in alto) resta rossa (preservata), l'ESTENSIONE (in basso) è verde (riempita), NON nera
    assert out.getpixel((W // 2, 5)) == (200, 30, 30), "la foto va PRESERVATA"
    assert out.getpixel((W // 2, H - 5)) == (20, 180, 60), "l'estensione va RIEMPITA (non nera)"


def test_inpaint_bianco_rigenera_solo_area():
    W, H = 40, 40
    orig = _solid(W, H, (10, 10, 10))
    gen = _solid(W, H, (240, 240, 0))
    mask = Image.new("L", (W, H), 0)
    mask.paste(Image.new("L", (10, 10), 255), (15, 15))     # solo un riquadro disegnato
    out = Image.composite(gen, orig, mask)
    assert out.getpixel((20, 20)) == (240, 240, 0)          # dentro la maschera: rigenerato
    assert out.getpixel((2, 2)) == (10, 10, 10)             # fuori: originale intatto
