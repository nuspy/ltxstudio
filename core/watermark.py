"""Watermark OBBLIGATORIO sugli output (pilotato dalla feature di matrice `cap.watermark`).

Scopo: limitare i tier gratuito/Supporter ($5). L'utente NON ha alcun controllo — l'attivazione vive
solo nel manifest bakeato dell'edizione (core/edition.py, prodotto dal feature_matrix build). Su OGNI
immagine/video generato: banda semitrasparente sul fondo con logo B/W "AI Tools Forge" + link atf.tadamoo.com.

ANTI-BYPASS (best practice applicate):
- Testo + logo DISEGNATI PROCEDURALMENTE (PIL/ffmpeg) → nessun file-immagine da sostituire con uno vuoto.
- PIL (non Qt) → applicabile nel processo BACKEND che PRODUCE i pixel (finestra di intercettazione minima).
- Gate dal manifest bakeato (compilato Nuitka nel prodotto) → non un config runtime editabile.
- forced=True → FAIL-CLOSED: se il watermark non si applica, l'output viene RIMOSSO (niente file non marchiato).
- Verifica d'integrità: dopo il disegno controlla che i pixel della banda siano davvero cambiati.
- Banda a tutta larghezza sul fondo → più difficile da ritagliare di un angolo. Complementare alla
  steganografia invisibile (#107) per la provenienza anche se la banda venisse tagliata.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from core import features

_IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_VID_EXT = {".mp4", ".webm", ".mkv", ".mov", ".gif"}

_BRAND = "AI Tools Forge"
_LINK = "atf.tadamoo.com"


def _cfg() -> dict | None:
    """Config watermark SOLO se l'edizione lo abilita ESPLICITAMENTE nel manifest (dev = niente watermark)."""
    info = features._info("cap.watermark")                   # esplicito: non usare il default permissivo
    if not info or not info.get("enabled", False):
        if os.environ.get("ATF_WATERMARK_TEST") != "1":      # override solo per test interni
            return None
    prm = (info or {}).get("params", {}) if info else {}
    return {
        "brand": prm.get("brand", _BRAND),
        "link": prm.get("link", _LINK),
        "opacity": float(prm.get("opacity", 0.38)),
        "forced": bool(prm.get("forced", True)),             # per demo/supporter: fail-closed
    }


def should_watermark() -> bool:
    return _cfg() is not None


def apply_to_file(path: str) -> str:
    """Marchia il file se il watermark è attivo per l'edizione. Ritorna il path.
    forced=True → se la marchiatura fallisce, RIMUOVE l'output (fail-closed): nessun file non marchiato."""
    cfg = _cfg()
    if not cfg:
        return path
    p = Path(path)
    ext = p.suffix.lower()
    ok = False
    try:
        if ext in _IMG_EXT:
            ok = _img(p, cfg)
        elif ext in _VID_EXT:
            return _video(p, cfg)
        else:
            return path                                      # tipo non marchiabile → lascia
    except Exception:                                        # noqa: BLE001
        ok = False
    if not ok and cfg["forced"]:
        try:
            p.unlink(missing_ok=True)                        # FAIL-CLOSED: niente output non marchiato
        except Exception:                                    # noqa: BLE001
            pass
        raise RuntimeError("watermark obbligatorio non applicabile: output rimosso")
    return path


# ---------------- font procedurale ----------------
def _font(size: int):
    from PIL import ImageFont
    _cands = (
        r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\seguisb.ttf",
        "arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for name in _cands:
        try:
            return ImageFont.truetype(name, size)
        except Exception:                                    # noqa: BLE001
            continue
    return ImageFont.load_default()


def _draw_logo(draw, x, y, s, color):
    """Logo B/W procedurale semplificato (incudine+martello 'forge'): niente file esterno."""
    # incudine: corpo trapezoidale + base
    draw.polygon([(x, y + s * 0.35), (x + s * 0.7, y + s * 0.35), (x + s * 0.7, y + s * 0.5),
                  (x + s * 0.15, y + s * 0.5)], fill=color)
    draw.rectangle([x + s * 0.28, y + s * 0.5, x + s * 0.5, y + s * 0.72], fill=color)
    draw.rectangle([x + s * 0.12, y + s * 0.72, x + s * 0.66, y + s * 0.82], fill=color)
    # martello: manico + testa
    draw.line([(x + s * 0.55, y + s * 0.3), (x + s * 0.9, y + s * 0.02)], fill=color, width=max(2, int(s * 0.07)))
    draw.rectangle([x + s * 0.8, y - s * 0.03, x + s * 1.02, y + s * 0.12], fill=color)


# ---------------- immagini (PIL, portabile in ogni venv) ----------------
def _img(p: Path, cfg: dict) -> bool:
    from PIL import Image, ImageDraw
    im = Image.open(str(p)).convert("RGB")
    W, H = im.size
    band_h = max(22, H // 16)
    fs = max(12, int(band_h * 0.42))
    a = int(255 * cfg["opacity"])
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rectangle([0, H - band_h, W, H], fill=(0, 0, 0, int(a * 0.7)))   # banda scura semitrasparente
    lg = int(band_h * 0.6)
    ly = H - band_h + (band_h - lg) // 2
    _draw_logo(d, 12, ly, lg, (255, 255, 255, min(255, a + 60)))
    tx = 12 + int(lg * 1.15) + 8
    ty = H - band_h + (band_h - fs) // 2 - 2
    d.text((tx, ty), cfg["brand"], font=_font(fs), fill=(255, 255, 255, min(255, a + 70)))
    link = cfg["link"]
    lf = _font(max(10, int(fs * 0.8)))
    lw = d.textlength(link, font=lf)
    d.text((W - lw - 14, H - band_h + (band_h - int(fs * 0.8)) // 2 - 1), link, font=lf,
           fill=(255, 255, 255, min(255, a + 40)))
    out = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
    # verifica integrità: la banda inferiore deve essere cambiata rispetto all'originale
    import hashlib
    before = im.crop((0, H - band_h, W, H)).tobytes()
    after = out.crop((0, H - band_h, W, H)).tobytes()
    if hashlib.md5(before).digest() == hashlib.md5(after).digest():
        return False                                         # nulla è cambiato → fallito
    out.save(str(p))
    return True


# ---------------- video (ffmpeg drawtext + drawbox) ----------------
def _video(p: Path, cfg: dict) -> str:
    out = p.with_name(p.stem + "_wm" + p.suffix)
    op = cfg["opacity"]
    brand = cfg["brand"].replace(":", r"\:").replace("'", r"\'")
    link = cfg["link"].replace(":", r"\:").replace("'", r"\'")
    bh = "max(22\\,ih/16)"
    vf = (
        f"drawbox=x=0:y=ih-{bh}:w=iw:h={bh}:color=black@{op * 0.7:.2f}:t=fill,"
        f"drawtext=text='{brand}':fontcolor=white@{min(1.0, op + 0.3):.2f}:fontsize=ih/34:"
        f"x=16:y=ih-{bh}/2-th/2,"
        f"drawtext=text='{link}':fontcolor=white@{min(1.0, op + 0.2):.2f}:fontsize=ih/42:"
        f"x=iw-tw-14:y=ih-{bh}/2-th/2"
    )
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(p), "-vf", vf, "-c:a", "copy", str(out)],
                       capture_output=True, text=True)
    if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
        try:
            p.unlink(); out.rename(p)
            return str(p)
        except Exception:                                    # noqa: BLE001
            return str(out)
    if cfg["forced"]:                                        # fail-closed video
        try:
            out.unlink(missing_ok=True); p.unlink(missing_ok=True)
        except Exception:                                    # noqa: BLE001
            pass
        raise RuntimeError("watermark video obbligatorio non applicabile: output rimosso")
    return str(p)
