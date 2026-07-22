"""DEPRECATO (F1 unified-model-manager, 22/07/2026) — usare `core.vram_estimate`.

Questo registro (data/model_footprint.json) era WRITE-ONLY: scritto solo dal ComponentStore e letto
da nessuno. Il DB canonico dei footprint è ora `data/model_vram_cache.json` via `core/vram_estimate`
(chiavi `comp:<key>` per i componenti condivisi; `vram_estimate.migrate_legacy()` importa le voci
vecchie). Questo shim resta per compatibilità di eventuali import esterni e delega al canonico.
"""
from __future__ import annotations

from core import vram_estimate as _ve


def get(key: str) -> float | None:
    """Footprint (MB) del componente se tracciato, altrimenti None. Delegato a vram_estimate.comp_get."""
    mb = _ve.comp_get(key)
    return float(mb) if mb else None


def record(key: str, vram_mb: float, *, dtype: str | None = None, category: str | None = None) -> None:
    """Delegato a vram_estimate.comp_put (scrittura atomica sul DB canonico)."""
    _ve.comp_put(key, float(vram_mb), category=category or "aux", dtype=dtype or "")


def measure_module_mb(module) -> float:
    """Footprint REALE di un nn.Module: byte di parametri + buffer, DTYPE-AWARE (fp8=1B, bf16=2B, fp32=4B)."""
    total = 0
    try:
        for p in module.parameters(recurse=True):
            total += p.numel() * p.element_size()
        for b in module.buffers(recurse=True):
            total += b.numel() * b.element_size()
    except Exception:                                        # noqa: BLE001
        return 0.0
    return total / 1024**2


def ensure(key: str, module=None, *, mb: float | None = None, category: str | None = None) -> float | None:
    """Se `key` non è tracciato nel DB canonico, registra il footprint (da `mb` o misurando `module`)."""
    cur = get(key)
    if cur is not None:
        return cur
    val = mb if mb is not None else (measure_module_mb(module) if module is not None else None)
    if val is None:
        return None
    dt = None
    if module is not None:
        try:
            dt = str(next(module.parameters()).dtype).replace("torch.", "")
        except Exception:                                    # noqa: BLE001
            pass
    record(key, val, dtype=dt, category=category)
    return val


def all_entries() -> dict:
    """Voci comp:* del DB canonico, nel formato legacy {key: {vram_mb, dtype, category}}."""
    out = {}
    for k, e in _ve.all_entries().items():
        if k.startswith("comp:"):
            out[k[5:]] = {"vram_mb": e.get("footprint_mb", 0), "dtype": e.get("dtype"),
                          "category": e.get("category")}
    return out
