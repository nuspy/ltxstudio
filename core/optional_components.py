"""Funzioni che un'edizione può NON includere (file esclusi dalla build per via della feature matrix): import opzionale.

Oggi: MagicQuill V2 (`inference_server/backends/magicquill_v2.py` + `mq2/`). Se `cap.image_magicquill` è spenta in
un'edizione il `runtime_bundle` non spedisce quei file: il job_runner / persistent_runner NON devono rompersi. Qui:
  · `magicquill_installed()`  esiste il modulo? (find_spec, senza importarlo: niente torch nel processo GUI);
  · `load_magicquill()`       il modulo o None (mai solleva; il motivo in `last_error()`);
  · `image_mq_handler()`      `run` del modulo oppure un handler che risponde «non incluso» (mai un crash del runner).
`LTX_OPTIONAL_DISABLE=magicquill` simula un'edizione senza il componente (test).
Stdlib pura.
"""
from __future__ import annotations

import importlib
import importlib.util
import os

MODULE = "inference_server.backends.magicquill_v2"
ENV_DISABLE = "LTX_OPTIONAL_DISABLE"
NOT_INSTALLED_MSG = "La funzione MagicQuill V2 non è inclusa in questa edizione dell'app (componente assente)."
_last_error = ""


def _disabled() -> bool:
    return "magicquill" in os.environ.get(ENV_DISABLE, "").lower().split(",")


def magicquill_installed() -> bool:
    if _disabled():
        return False
    try:
        return importlib.util.find_spec(MODULE) is not None
    except (ImportError, ValueError):
        return False


def load_magicquill():
    """Modulo `magicquill_v2` o None se non c'è / non si importa. Mai solleva."""
    global _last_error
    if not magicquill_installed():
        _last_error = NOT_INSTALLED_MSG
        return None
    try:
        return importlib.import_module(MODULE)
    except Exception as e:                                 # noqa: BLE001
        _last_error = f"MagicQuill V2 non caricabile ({type(e).__name__}: {e})"
        return None


def last_error() -> str:
    return _last_error or NOT_INSTALLED_MSG


def missing_run(params, mm, output_dir):
    """Handler `image_mq` quando il modulo manca: errore chiaro nel risultato del job, mai un crash del runner."""
    return {"status": "error", "error": last_error()}


def image_mq_handler():
    m = load_magicquill()
    return m.run if m is not None else missing_run
