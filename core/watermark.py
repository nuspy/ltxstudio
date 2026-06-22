"""Watermark obbligatorio sugli output (pilotato dalla feature di matrice `cap.watermark`).

Applicato DOPO la generazione, sul file salvato. Immagini via Qt (QPainter),
video via ffmpeg (drawtext). Forzato per le edizioni che hanno
`cap.watermark.params.forced=true` (es. demo/supporter): non disattivabile dalla UI.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from core import features

_IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_VID_EXT = {".mp4", ".webm", ".mkv", ".mov", ".gif"}


def _cfg() -> dict | None:
    """Config watermark se attivo per l'edizione corrente, altrimenti None."""
    if not features.is_enabled("cap.watermark"):
        return None
    return {
        "text": features.param("cap.watermark", "text", "LTX Studio"),
        "opacity": float(features.param("cap.watermark", "opacity", 0.30)),
        "position": features.param("cap.watermark", "position", "bottom-right"),
        "forced": bool(features.param("cap.watermark", "forced", False)),
    }


def should_watermark() -> bool:
    return _cfg() is not None


def apply_to_file(path: str) -> str:
    """Marchia il file se il watermark è attivo per l'edizione. Ritorna il path
    (eventualmente invariato). Non solleva: in caso di errore lascia il file com'è."""
    cfg = _cfg()
    if not cfg:
        return path
    p = Path(path)
    ext = p.suffix.lower()
    try:
        if ext in _IMG_EXT:
            _img(p, cfg)
        elif ext in _VID_EXT:
            return _video(p, cfg)
    except Exception:
        pass
    return path


# ---------------- immagini (Qt) ----------------
def _img(p: Path, cfg: dict) -> None:
    from PySide6.QtCore import Qt, QRect
    from PySide6.QtGui import QImage, QPainter, QColor, QFont
    img = QImage(str(p))
    if img.isNull():
        return
    painter = QPainter(img)
    fs = max(14, img.height() // 28)
    painter.setFont(QFont("Arial", fs, QFont.Bold))
    painter.setPen(QColor(255, 255, 255, int(255 * cfg["opacity"])))
    m = fs // 2 + 4
    rect = QRect(m, m, img.width() - 2 * m, img.height() - 2 * m)
    align = _qt_align(cfg["position"])
    painter.drawText(rect, align, cfg["text"])
    painter.end()
    img.save(str(p))


def _qt_align(position: str):
    from PySide6.QtCore import Qt
    pos = (position or "bottom-right").lower()
    vert = Qt.AlignBottom if "bottom" in pos else Qt.AlignTop
    horiz = Qt.AlignRight if "right" in pos else (Qt.AlignLeft if "left" in pos else Qt.AlignHCenter)
    return vert | horiz


# ---------------- video (ffmpeg) ----------------
def _video(p: Path, cfg: dict) -> str:
    out = p.with_name(p.stem + "_wm" + p.suffix)
    x, y = _ff_xy(cfg["position"])
    text = cfg["text"].replace(":", r"\:").replace("'", r"\'")
    draw = (f"drawtext=text='{text}':fontcolor=white@{cfg['opacity']:.2f}:"
            f"fontsize=h/28:box=1:boxcolor=black@0.25:boxborderw=8:x={x}:y={y}")
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(p), "-vf", draw,
         "-c:a", "copy", str(out)], capture_output=True, text=True)
    if r.returncode == 0 and out.exists():
        try:
            p.unlink(); out.rename(p)          # sostituisci l'originale
            return str(p)
        except Exception:
            return str(out)
    return str(p)


def _ff_xy(position: str) -> tuple[str, str]:
    pos = (position or "bottom-right").lower()
    x = "w-tw-20" if "right" in pos else ("20" if "left" in pos else "(w-tw)/2")
    y = "h-th-20" if "bottom" in pos else ("20" if "top" in pos else "(h-th)/2")
    return x, y
