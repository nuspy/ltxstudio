"""Dequantizzazione AUTOMATICA dei modelli comfy_quant (ComfyUI fp8/nvfp4/mxfp8) → bf16 standard.

Quando l'app deve usare nel backend NATIVO un checkpoint quantizzato di ComfyUI (chiavi
`.comfy_quant`), lo converte UNA VOLTA in bf16 standard (cache) e ritorna il path convertito.
Così un modello scaricato/inserito in formato ComfyUI funziona automaticamente col motore nativo.

La conversione gira nel venv ComfyUI (ha i kernel di dequant fp8/nvfp4); se assente, ritorna il
path originale (il loader tenterà as-is). Vedi scripts/dequant_comfy_worker.py.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

COMFY_PY = r"E:\AI\Comfy_Nexus\ComfyUI\venv\Scripts\python.exe"
WORKER = str(Path(__file__).resolve().parent.parent / "scripts" / "dequant_comfy_worker.py")
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
    """Se `path` è comfy_quant → dequantizza (cache) e ritorna il bf16; altrimenti ritorna `path`."""
    path = str(path)
    if not path.lower().endswith(".safetensors") or not is_comfy_quantized(path):
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5((path + "|" + remap).encode()).hexdigest()[:12]
    dst = CACHE / f"{Path(path).stem}.{remap}.{h}.safetensors"
    if dst.exists():
        return str(dst)
    if not os.path.exists(COMFY_PY):
        return path
    subprocess.run([COMFY_PY, WORKER, "--src", path, "--dst", str(dst), "--remap", remap],
                   capture_output=True, text=True, timeout=timeout)
    return str(dst) if dst.exists() else path
