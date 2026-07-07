"""Stima VRAM CONDIVISA (immagini + video): byte reali dell'header safetensors per dtype + cache su file del
footprint reale. stdlib-only → usabile da QUALSIASI venv (server immagini, worker video ltx2, 3D…).

Ordine di precisione: cache_get (footprint REALE misurato al 1° load) → header_mb (byte reali per dtype) → 0
(il chiamante ripiega sulla dimensione file grezza). estimate_mb() somma anche il working-set (attivazioni).
"""
from __future__ import annotations

import json
import os
import struct

_CACHE = r"E:\AI\ltxstudio\data\model_vram_cache.json"
# byte in MEMORIA per dtype safetensors (fp8 = 1B se resta quantizzato, 2B se upcastato a bf16).
_DTYPE_BYTES = {"F8_E4M3": 1, "F8_E5M2": 1, "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
                "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8, "BOOL": 1}
_TE_KEYS = ("text_encoder", "t5xxl", "t5.", "cond_stage", "clip_l", "text_model", "gemma", "mistral")


def _read_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def header_mb(path: str, *, fp8_storage: bool = False) -> int:
    """MB reali in memoria di un checkpoint safetensors (0 se illeggibile / non-safetensors). fp8_storage=True
    → i pesi F8 restano 1B (layerwise casting); altrimenti diffusers li upcasta a bf16 (2B). Se il file NON
    include il text-encoder (transformer-only) → aggiunge T5/CLIP/VAE (~9.7GB)."""
    if not path or path.lower().endswith(".gguf"):
        return 0
    try:
        h = _read_header(path)
    except Exception:                                       # noqa: BLE001
        return 0
    total, has_te = 0.0, False
    for k, t in h.items():
        if k == "__metadata__":
            continue
        kl = k.lower()
        if any(s in kl for s in _TE_KEYS):
            has_te = True
        n = 1
        for s in t.get("shape", []):
            n *= int(s)
        dt = str(t.get("dtype", "F16"))
        b = _DTYPE_BYTES.get(dt, 2)
        if dt.startswith("F8") and not fp8_storage:
            b = 2
        total += n * b
    mb = total / 1024**2
    if not has_te:
        mb += 9700
    return int(mb)


def _key(path: str) -> str:
    try:
        return f"{os.path.normpath(path).lower()}|{os.stat(path).st_size}"   # invalida se il file cambia
    except Exception:                                       # noqa: BLE001
        return os.path.normpath(path).lower()


def _load_cache() -> dict:
    try:
        with open(_CACHE, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                       # noqa: BLE001
        return {}


def cache_get(path: str) -> int:
    return int(_load_cache().get(_key(path), {}).get("footprint_mb", 0) or 0)


def cache_put(path: str, footprint_mb: int) -> None:
    if footprint_mb <= 0:
        return
    c = _load_cache()
    c[_key(path)] = {"footprint_mb": int(footprint_mb), "name": os.path.basename(path)}
    try:
        os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
        with open(_CACHE, "w", encoding="utf-8") as fh:
            json.dump(c, fh, indent=1)
    except Exception:                                       # noqa: BLE001
        pass


def estimate_mb(path: str, *, fp8_storage: bool = False, working_mb: int = 3200) -> tuple[int, str]:
    """(stima_MB, fonte) — cache reale → header preciso → 0. working_mb = attivazioni/VAE/workspace
    (immagini ~3200; video molto più alto, ~9000: attivazioni temporali su molti frame)."""
    c = cache_get(path)
    if c:
        return c + working_mb, "cache"
    h = header_mb(path, fp8_storage=fp8_storage)
    if h:
        return h + working_mb, "header"
    return 0, "none"
