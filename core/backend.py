"""Astrazione di backend: per ogni modulo sceglie ComfyUI o il backend nativo.

Feature-flag in `data/backend_config.json` (default: tutto "comfy" → zero regressioni).
Le pipeline/UI interrogano `is_native(module)` per decidere se usare `core.native_driver`
invece del driver ComfyUI. Migrazione incrementale (strangler).
"""
from __future__ import annotations

import json
from pathlib import Path

_CONFIG = Path(__file__).resolve().parents[1] / "data" / "backend_config.json"
_DEFAULT = "comfy"


def _cfg() -> dict:
    try:
        return json.loads(_CONFIG.read_text("utf-8"))
    except Exception:                                   # noqa: BLE001
        return {}


def get(module: str) -> str:
    """Ritorna 'comfy' | 'native' per il modulo (default 'comfy')."""
    return _cfg().get(module, _DEFAULT)


def is_native(module: str) -> bool:
    return get(module) == "native"


def set_backend(module: str, backend: str) -> None:
    cfg = _cfg(); cfg[module] = backend
    _CONFIG.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", "utf-8")


def strict_native() -> bool:
    """STRICT: se True, le feature NON ancora native NON ripiegano su ComfyUI — vanno bloccate
    con errore esplicito. Serve a non usare ComfyUI 'per errore' durante la migrazione, SENZA
    eliminare l'installazione (reversibile). Default False (fallback ComfyUI attivo)."""
    return bool(_cfg().get("strict_native", False))


def set_strict(on: bool) -> None:
    cfg = _cfg(); cfg["strict_native"] = bool(on)
    _CONFIG.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", "utf-8")


def comfy_blocked(module: str) -> bool:
    """True se il modulo userebbe ComfyUI ma STRICT lo vieta → la UI deve mostrare errore
    'feature non ancora migrata al nativo' invece di usare ComfyUI."""
    return strict_native() and not is_native(module)
