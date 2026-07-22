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


# overhead per FAMIGLIA rilevata dall'header (più affidabile del nome)
_OVERHEAD_FAMILY = {
    "ltxv": 14.0, "wan21": 14.0, "wan22": 14.0, "hunyuan_video": 14.0,   # Gemma/T5 video + VAE video
    # LingBot: snapshot diffusers self-contained. Il text-encoder è una VLM Qwen (~9GB) + VAE ~0.5GB.
    # NB per la MoE il "footprint file" (uint4 ~22GB) è ciò che occupa RAM/VRAM streamata, l'overhead qui
    # copre solo TE+VAE. La classify() usa comunque la dim file come proxy del transformer.
    "lingbot": 10.0,
    "zimage": 4.0,                                                       # encoder piccolo
    "flux1": 11.0, "flux2": 11.0, "ernie": 11.0, "qwen_image": 11.0,     # T5/CLIP/Mistral/Qwen + VAE
    "sd15": 3.0, "sdxl": 4.0, "sd3": 6.0,
}


def _overhead_gb(name: str, family: str | None = None) -> float:
    """Overhead tipico (text-encoder + VAE + working set) oltre ai pesi del transformer, in GB.
    Preferisce la FAMIGLIA rilevata dall'header; ricade sull'euristica per nome."""
    if family and family in _OVERHEAD_FAMILY:
        return _OVERHEAD_FAMILY[family]
    n = (name or "").lower()
    if "ltx" in n or "wan" in n or "hunyuan" in n:
        return 14.0                                          # Gemma/T5 video + VAE video + working
    if "zimage" in n or "z_image" in n or "z-image" in n or "lumina" in n:
        return 4.0                                           # encoder piccolo
    if "flux" in n or "ernie" in n or "qwen" in n:
        return 11.0                                          # T5/CLIP/Mistral + VAE + working
    return 9.0


def _detect(file_path: str) -> dict:
    """Detection header cache-aware (precisione+famiglia dall'header). Vuoto se non disponibile."""
    try:
        from core import model_detect
        return model_detect.get(file_path) or {}
    except Exception:                                        # noqa: BLE001
        return {}


# stima pesi quando il file non è locale (per precisione, su un 12B di riferimento)
_EST_WEIGHTS_GB = {"fp16": 24.0, "fp8": 12.0, "fp4": 7.0, "gguf": 10.0}


def classify(name_or_path: str, free_vram_gb: float, *, file_path: str | None = None) -> dict:
    """Ritorna {color, footprint_gb, weights_gb, precision, family}.
    color ∈ {"green","yellow","red",""}. "" = VRAM ignota (nessuna colorazione).
    Precisione e famiglia sono lette dall'HEADER del file (detector) quando disponibile —
    più affidabili del nome (es. 'FP8' nel nome che nasconde pesi bf16); fallback al nome."""
    name = os.path.basename(name_or_path or "")
    fp = file_path or name_or_path
    prec = None
    family = None
    if fp and os.path.isfile(fp):
        det = _detect(fp)
        p = det.get("precision")
        if p and p != "unknown":
            prec = p
        f = det.get("family")
        if f and f != "unknown":
            family = f
    if not prec:
        prec = detect_precision(name)
    # F6 unified-model-manager: preferisci il DB canonico (footprint misurato → header preciso);
    # la vecchia euristica file_gb + overhead-per-famiglia resta come ULTIMO fallback.
    w = footprint = 0.0
    if fp and os.path.isfile(fp):
        try:
            from core import vram_estimate as _ve
            _fp8 = (prec == "fp8")
            _tr = _ve.transformer_mb(fp, fp8_storage=_fp8)
            _tot, _src = _ve.estimate_mb(fp, fp8_storage=_fp8, working_mb=0)
            if _tr > 0:
                w = _tr / 1024.0
            if _tot > 0:
                footprint = _tot / 1024.0
        except Exception:                                    # noqa: BLE001
            pass
    if not w:
        w = file_gb(fp) if (fp and os.path.isfile(fp)) else _EST_WEIGHTS_GB.get(prec, 24.0)
    if not footprint:
        footprint = w + _overhead_gb(name, family)
    base = {"footprint_gb": round(footprint, 1), "weights_gb": round(w, 1), "precision": prec, "family": family}
    if not free_vram_gb or free_vram_gb <= 0:
        return {"color": "", **base}
    if footprint <= free_vram_gb * 0.92:
        color = "green"
    elif w <= free_vram_gb * 0.92:                           # il transformer entra → offload TE/VAE ok
        color = "yellow"
    else:
        color = "red"
    return {"color": color, **base}


def gpu_free_gb() -> float:
    """VRAM totale della GPU (usiamo il totale, non il libero istantaneo, per una stima stabile)."""
    try:
        from core import hw_profiles
        return hw_profiles.detect_vram_gb()
    except Exception:                                        # noqa: BLE001
        return 0.0


# colori Qt (foreground) per stato
QT_COLORS = {"green": "#3fb950", "yellow": "#d29922", "red": "#f85149", "": None}
