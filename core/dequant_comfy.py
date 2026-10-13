"""Dequantizzazione AUTOMATICA dei modelli comfy_quant (ComfyUI fp8/nvfp4/int8) → bf16 standard.

Quando l'app deve usare nel backend NATIVO un checkpoint quantizzato di ComfyUI (chiavi `.comfy_quant`) con un
loader che vuole pesi pieni, lo converte UNA VOLTA in bf16 standard (cache) e ritorna il path convertito.

NATIVO (13/10/2026): la conversione la fa core/native_convert (converter «comfy_bf16»: un tensore alla volta,
verifica, scrittura atomica) nel processo chiamante. Prima girava nel venv ComfyUI (comfy.quant_ops), che non
esiste più: senza di esso la funzione ritornava il path originale e il loader caricava i pesi quantizzati crudi.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

CACHE = Path(r"E:\AI\Models\_dequant_cache")


def is_comfy_quantized(path: str) -> bool:
    """True se il safetensors contiene chiavi `.comfy_quant` (quantizzazione ComfyUI)."""
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            return any(k.endswith(".comfy_quant") for k in f.keys())
    except Exception:                                       # noqa: BLE001
        return False


def ensure_dequantized(path: str, remap: str = "none", timeout: int = 1800) -> str:
    """Se `path` è comfy_quant → dequantizza (cache, una volta sola) e ritorna il bf16; altrimenti `path`.
    `timeout` resta per compatibilità di firma (la conversione è in-process). Errori di conversione (es. formato
    non supportato) emergono come eccezione: meglio di un caricamento silenzioso di pesi quantizzati crudi."""
    path = str(path)
    if not path.lower().endswith(".safetensors") or not is_comfy_quantized(path):
        return path
    from core import native_convert as nc
    CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5((path + "|" + remap).encode()).hexdigest()[:12]
    dst = CACHE / f"{Path(path).stem}.{remap}.{h}.safetensors"
    if dst.exists() and nc.is_native(dst):
        return str(dst)
    conv = "comfy_bf16_gemma" if remap == "gemma" else "comfy_bf16"
    nc.convert(conv, path, dst, delete_source=False, log=lambda m: print(m, flush=True))
    return str(dst)
