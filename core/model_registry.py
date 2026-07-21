"""Registro PERSISTENTE del footprint VRAM per-modello/componente (dtype-aware) — base della smart-memory.

Idea (direttiva utente): tenere un registro di quanto occupa OGNI modello, così l'eviction può fare un
**calcolo combinatorio size-aware** di cosa scaricare e quando (più economico sfrattare un modello da 2GB che
uno da 10GB se il buco richiesto è piccolo), e il placement può stimare senza tabelle statiche.

Auto-popolato: al primo uso di un modello NON tracciato se ne misura il peso reale (byte parametri+buffer,
dtype-aware: fp8=1B, bf16=2B, fp32=4B) e si aggiorna il registro su disco. Il file è condiviso tra i processi
(GUI/dispatch che decide, e figli torch che misurano): scrittura atomica via file temporaneo + os.replace.

Opzionale: `measure_all()` per una passata di pre-popolamento all'avvio (carica i modelli uno alla volta e
registra i pesi); non necessaria perché il popolamento incrementale converge da solo con l'uso.
"""
from __future__ import annotations

import json
import os
import threading

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "model_footprint.json")
_LOCK = threading.Lock()
_CACHE: dict | None = None


def _load() -> dict:
    global _CACHE
    if _CACHE is None:
        try:
            with open(_PATH, encoding="utf-8") as f:
                _CACHE = json.load(f)
        except Exception:                                    # noqa: BLE001 — file assente/corrotto → vuoto
            _CACHE = {}
    return _CACHE


def get(key: str) -> float | None:
    """Footprint (MB) del modello se tracciato, altrimenti None."""
    e = _load().get(key)
    return float(e["vram_mb"]) if e and "vram_mb" in e else None


def record(key: str, vram_mb: float, *, dtype: str | None = None, category: str | None = None) -> None:
    """Registra/aggiorna il footprint di `key` e persiste su disco (atomico)."""
    with _LOCK:
        d = _load()
        prev = d.get(key, {})
        d[key] = {
            "vram_mb": round(float(vram_mb), 1),
            "dtype": dtype or prev.get("dtype"),
            "category": category or prev.get("category"),
        }
        try:
            os.makedirs(os.path.dirname(_PATH), exist_ok=True)
            tmp = _PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=1, ensure_ascii=False)
            os.replace(tmp, _PATH)
        except Exception:                                    # noqa: BLE001 — best-effort, mai fatale
            pass


def measure_module_mb(module) -> float:
    """Footprint REALE di un nn.Module: byte di parametri + buffer, DTYPE-AWARE (fp8 conta 1B, bf16 2B, fp32 4B).
    È lo spazio che si RECUPERA sfrattandolo → la metrica giusta per la scelta combinatoria della vittima."""
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
    """Se `key` NON è tracciato, registra il footprint: da `mb` (già misurato, es. ComponentStore._measure_mb)
    oppure misurando `module`. Se già tracciato, non ricarica nulla. Ritorna il footprint MB (o None)."""
    cur = get(key)
    if cur is not None:
        return cur
    val = mb if mb is not None else (measure_module_mb(module) if module is not None else None)
    if val is None:
        return None
    dt = None
    if module is not None:
        try:
            dt = str(next(module.parameters()).dtype)
        except Exception:                                    # noqa: BLE001
            pass
    record(key, val, dtype=dt, category=category)
    return val


def all_entries() -> dict:
    """Copia dell'intero registro (per il planner/eviction e la diagnostica)."""
    return dict(_load())
