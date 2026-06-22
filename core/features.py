"""Facade runtime del feature-gating (self-contained, nessuna dipendenza da
feature_matrix). Legge il manifest BAKEATO in `core/edition.py` (prodotto dal
build); se assente (sviluppo, app non compilata) ricade su `core/edition_default.py`.

API:
  edition()            -> id edizione effettiva
  is_enabled(fid)      -> bool   (permissivo: feature non nel manifest = True, per dev)
  param(fid, key, dflt)-> valore parametro di una feature
  assert_enabled(fid)  -> solleva PermissionError se disabilitata (per il backend/API)
  allows_adult()/allows_studio()/manifest()
"""
from __future__ import annotations

try:                                   # binario compilato: edition.py bakeato
    from core import edition as _ed     # type: ignore
except Exception:                      # sviluppo: profilo permissivo
    from core import edition_default as _ed   # type: ignore

_MANIFEST: dict = getattr(_ed, "MANIFEST", {}) or {}
_FEATURES: dict = _MANIFEST.get("features", {}) or {}


def edition() -> str:
    return _MANIFEST.get("edition", getattr(_ed, "EDITION_ID", "dev"))


def manifest() -> dict:
    return _MANIFEST


def allows_adult() -> bool:
    return bool(_MANIFEST.get("allows_adult", False))


def allows_studio() -> bool:
    return bool(_MANIFEST.get("allows_studio", False))


def _info(fid: str) -> dict | None:
    return _FEATURES.get(fid)


def is_enabled(fid: str) -> bool:
    info = _info(fid)
    if info is None:
        return True                    # permissivo: non gestita = consentita (dev/fail-open non-sensibile)
    return bool(info.get("enabled", False))


def param(fid: str, key: str, default=None):
    info = _info(fid)
    if not info:
        return default
    return info.get("params", {}).get(key, default)


def assert_enabled(fid: str) -> None:
    if not is_enabled(fid):
        raise PermissionError(f"feature '{fid}' non disponibile in questa edizione ({edition()})")
