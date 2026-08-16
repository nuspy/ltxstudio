"""Risoluzione path dei modelli DAL registro unico (weights_catalog v2) — punto unico.

Perché: ~45 file hardcodano `E:\\AI\\Models\\…` come stringhe letterali, con 5 convenzioni
diverse; il risultato è che «Cartella base» del tab Pesi e `AI_MODELS_DIR` non spostano nulla
di ciò che i motori caricano davvero. Da qui in poi i consumer migrano a QUESTO modulo:
il path di un peso si chiede per ID di catalogo, non si re-inventa in ogni file.

stdlib-only, nessun torch (importabile dal processo GUI). La migrazione dei letterali nei
motori è graduale (FASE C/D del piano weights-unification): ogni file toccato passa di qui.
"""
from __future__ import annotations

import os
from pathlib import Path

from core import weights_catalog as wc


def models_root() -> Path:
    """Radice modelli EFFETTIVA, in ordine di precedenza: env AI_MODELS_DIR (l'unica finora
    onorata solo da offline_inventory) → base_dir del tab Pesi (weights_config.json) → default."""
    env = os.environ.get("AI_MODELS_DIR", "").strip()
    if env:
        return Path(env)
    return Path(wc.get_base_dir())


def entry(entry_id: str) -> dict | None:
    return next((e for e in wc.load() if e.get("id") == entry_id), None)


def path_of(entry_id: str) -> Path | None:
    """Path COMPLETO del peso (file → path del file; snapshot → directory), o None se l'id
    non è a catalogo. Rispetta tool_dest/family vuota/`~` esattamente come il downloader:
    un solo posto decide dove vive un peso."""
    e = entry(entry_id)
    if not e:
        return None
    d = wc.dest_for(e, str(models_root()))
    if e.get("kind") == "file" and e.get("filename"):
        return d / e["filename"]
    return d


def exists(entry_id: str) -> bool:
    e = entry(entry_id)
    return bool(e and wc.is_present(e, str(models_root())))


def p(entry_id: str, fallback: str) -> str:
    """Risoluzione COMPATTA per la migrazione dei letterali: path dal registro se l'id esiste,
    altrimenti il fallback storico. MAI solleva (i moduli che definiscono costanti a import-time
    non devono morire per un catalogo assente/rotto)."""
    try:
        q = path_of(entry_id)
        return str(q) if q else fallback
    except Exception:                                    # noqa: BLE001
        return fallback
