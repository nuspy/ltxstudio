"""Componenti OPZIONALI con licenza restrittiva, fuori dalle build: oggi solo «MagicQuill V2» (codice derivato, CC BY-NC 4.0).

Il codice sta in `optional/magicquill/` (repo, per lo sviluppo; ESCLUSO SEMPRE da build/installer: vedi
`build/runtime_bundle.EXCLUDE_ALWAYS`) oppure in `E:\\AI\\Models\\LoRA\\MagicQuill\\_code\\` (voce di catalogo `mq2-code`,
scaricata dall'utente con avviso di licenza). Qui si trova la cartella e si importa con importlib come pacchetto
`ltx_magicquill` (i moduli usano import relativi). Se manca, NIENTE crash: `load_magicquill()` ritorna None e i
chiamanti mostrano «componente non installato». Stdlib pura (importabile dal processo GUI).
"""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

PKG = "ltx_magicquill"
ENV = "LTX_MAGICQUILL_CODE"                                # override esplicito (test/sviluppo)
ENV_DISABLE = "LTX_OPTIONAL_DISABLE"                       # es. «magicquill»: simula un'edizione/build SENZA il componente
CATALOG_ID = "mq2-code"
NOT_INSTALLED_MSG = ("Componente MagicQuill V2 non installato: scaricalo dal tab «Pesi modelli» (voce «mq2-code»; "
                     "licenza non commerciale, il download chiede conferma).")
_last_error = ""


def _installed_dir() -> Path:
    try:
        from core.model_paths import p as _cat_p
        return Path(_cat_p(CATALOG_ID, r"E:\AI\Models\LoRA\MagicQuill\_code"))
    except Exception:                                      # noqa: BLE001
        return Path(r"E:\AI\Models\LoRA\MagicQuill\_code")


def repo_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "optional" / "magicquill"


def _valid(d: Path) -> bool:
    return (d / "magicquill_v2.py").is_file() and (d / "mq2" / "__init__.py").is_file()


def candidate_dirs() -> list[Path]:
    """Ordine: override env → repo (sviluppo) → cartella installata dall'utente."""
    out = []
    if "magicquill" in os.environ.get(ENV_DISABLE, "").lower().split(","):
        return out
    if os.environ.get(ENV):
        out.append(Path(os.environ[ENV]))
    out += [repo_dir(), _installed_dir()]
    return out


def pack_dir() -> Path | None:
    for d in candidate_dirs():
        if _valid(d):
            return d
    return None


def magicquill_installed() -> bool:
    """True se il codice del componente c'è (senza importarlo)."""
    return pack_dir() is not None


def load_magicquill():
    """Modulo `magicquill_v2` (pacchetto `ltx_magicquill`) o None se il componente non c'è / non si importa.
    Mai solleva: la ragione dell'eventuale fallimento è in `last_error()`."""
    global _last_error
    root = pack_dir()
    if root is None:
        _last_error = NOT_INSTALLED_MSG
        return None
    pkg = sys.modules.get(PKG)
    if pkg is None or [str(p) for p in getattr(pkg, "__path__", [])] != [str(root)]:
        for k in [k for k in sys.modules if k == PKG or k.startswith(PKG + ".")]:
            del sys.modules[k]
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [str(root)]
        sys.modules[PKG] = pkg
    try:
        return importlib.import_module(PKG + ".magicquill_v2")
    except Exception as e:                                 # noqa: BLE001
        _last_error = f"Componente MagicQuill V2 non caricabile ({type(e).__name__}: {e})"
        return None


def require_magicquill():
    m = load_magicquill()
    if m is None:
        raise ModuleNotFoundError(_last_error or NOT_INSTALLED_MSG)
    return m


def last_error() -> str:
    return _last_error or NOT_INSTALLED_MSG


def missing_run(params, mm, output_dir):
    """Handler `image_mq` quando il componente manca: errore chiaro nel risultato del job, mai un crash del runner."""
    return {"status": "error", "error": last_error()}


def image_mq_handler():
    """`run` del componente se c'è, altrimenti il handler che risponde «non installato»."""
    m = load_magicquill()
    return m.run if m is not None else missing_run
