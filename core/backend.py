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
