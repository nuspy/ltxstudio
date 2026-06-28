"""Precisione modelli (fp16/fp8/fp4/gguf) + fattibilità VRAM (verde/giallo/rosso).

Usato dai dropdown dei generatori (immagine: generazione/inpaint/outpaint/integra; video: tutte le
sezioni) per:
- FILTRARE i modelli per precisione scelta dall'utente (Tutte / fp16 / fp8 / fp4);
- COLORARE ogni voce per fattibilità sulla GPU corrente:
    🟢 verde  = entra tutto in VRAM (ottimale);
    🟡 giallo = il transformer entra ma TE/VAE vanno offloadati su CPU/shared (gestibile, ~ok);
    🔴 rosso  = nemmeno il transformer entra (troppo grande per questa GPU).

Zero dipendenze pesanti: solo stdlib + nvidia-smi (via core.hw_profiles). Vedi [[dynamic-vram-tuning]].
"""
from __future__ import annotations

import os
import re

# precisione → etichetta UI
PRECISIONS = ("fp16", "fp8", "fp4", "gguf")
PRECISION_LABEL = {"fp16": "FP16/BF16", "fp8": "FP8", "fp4": "FP4/NVFP4", "gguf": "GGUF"}

_RE_FP4 = re.compile(r"nvfp4|fp4|svdq|int4|nf4|\bw4a", re.I)
_RE_GGUF = re.compile(r"gguf|\bq[2-8](_k|_0|_1)", re.I)
_RE_FP8 = re.compile(r"fp8|float8|e4m3|e5m2", re.I)


def detect_precision(name: str) -> str:
    """Deduce la precisione dal nome file. Default: fp16 (full precision bf16/fp16)."""
    n = (name or "").lower()
    if _RE_FP4.search(n):
        return "fp4"
    if _RE_GGUF.search(n):
        return "gguf"
    if _RE_FP8.search(n):
        return "fp8"
    return "fp16"


def file_gb(path: str) -> float:
    try:
        return os.path.getsize(path) / 1024**3
    except Exception:                                        # noqa: BLE001
        return 0.0


def _overhead_gb(name: str) -> float:
    """Overhead tipico (text-encoder + VAE + working set) oltre ai pesi del transformer, in GB."""
    n = (name or "").lower()
    if "ltx" in n or "wan" in n or "hunyuan" in n:
        return 14.0                                          # Gemma/T5 video + VAE video + working
    if "zimage" in n or "z_image" in n or "z-image" in n or "lumina" in n:
        return 4.0                                           # encoder piccolo
    if "flux" in n or "ernie" in n or "qwen" in n:
        return 11.0                                          # T5/CLIP/Mistral + VAE + working
    return 9.0


# stima pesi quando il file non è locale (per precisione, su un 12B di riferimento)
_EST_WEIGHTS_GB = {"fp16": 24.0, "fp8": 12.0, "fp4": 7.0, "gguf": 10.0}


def classify(name_or_path: str, free_vram_gb: float, *, file_path: str | None = None) -> dict:
    """Ritorna {color, footprint_gb, weights_gb, precision}.
    color ∈ {"green","yellow","red",""}. "" = VRAM ignota (nessuna colorazione)."""
    name = os.path.basename(name_or_path or "")
    prec = detect_precision(name)
    fp = file_path or name_or_path
    w = file_gb(fp) if (fp and os.path.isfile(fp)) else _EST_WEIGHTS_GB.get(prec, 24.0)
    footprint = w + _overhead_gb(name)
    if not free_vram_gb or free_vram_gb <= 0:
        return {"color": "", "footprint_gb": round(footprint, 1), "weights_gb": round(w, 1), "precision": prec}
    if footprint <= free_vram_gb * 0.92:
        color = "green"
    elif w <= free_vram_gb * 0.92:                           # il transformer entra → offload TE/VAE ok
        color = "yellow"
    else:
        color = "red"
    return {"color": color, "footprint_gb": round(footprint, 1), "weights_gb": round(w, 1), "precision": prec}


def gpu_free_gb() -> float:
    """VRAM totale della GPU (usiamo il totale, non il libero istantaneo, per una stima stabile)."""
    try:
        from core import hw_profiles
        return hw_profiles.detect_vram_gb()
    except Exception:                                        # noqa: BLE001
        return 0.0


# colori Qt (foreground) per stato
QT_COLORS = {"green": "#3fb950", "yellow": "#d29922", "red": "#f85149", "": None}
