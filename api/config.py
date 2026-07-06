"""Configurazione della REST API pubblica di LTX Studio.

L'API è 100% LOCALE (bind 127.0.0.1): nessuna autenticazione, nessun cloud. Espone le funzioni
native (immagini, video, film, musica, 3D/rig/anim) come job asincroni e restituisce link di
download per i file prodotti. Vedi [[local-only-target]].
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Porta pubblica dell'API (l'InferenceServer interno resta su :8189).
HOST = os.environ.get("LTX_API_HOST", "127.0.0.1")
PORT = int(os.environ.get("LTX_API_PORT", "8200"))
BASE_URL = os.environ.get("LTX_API_BASE_URL", f"http://{HOST}:{PORT}")

# Token opzionale: se il bind NON è locale (accesso da telefono/rete) e LTX_API_TOKEN è impostato,
# le richieste devono portare `Authorization: Bearer <token>` (eccetto /health e /docs).
API_TOKEN = os.environ.get("LTX_API_TOKEN", "").strip()
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", ""}


def auth_required() -> bool:
    """True se va verificato il bearer token (bind non-locale + token impostato)."""
    return bool(API_TOKEN) and HOST not in _LOCAL_HOSTS


def token_ok(header: str | None) -> bool:
    if not auth_required():
        return True
    if not header:
        return False
    parts = header.split(" ", 1)
    return len(parts) == 2 and parts[0].lower() == "bearer" and parts[1] == API_TOKEN

# Cartelle: output dei job e upload dei file portati dall'utente.
OUTPUT_DIR = Path(os.environ.get("LTX_OUTPUT_DIR", str(REPO / "output")))
UPLOAD_DIR = Path(os.environ.get("LTX_UPLOAD_DIR", str(OUTPUT_DIR / "api_uploads")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Cartelle da cui è consentito servire file via /download (anti path-traversal).
ALLOWED_ROOTS = [OUTPUT_DIR.resolve(), UPLOAD_DIR.resolve()]


def is_allowed_path(p: str | os.PathLike) -> bool:
    """True se `p` è dentro una cartella servibile (output/upload)."""
    try:
        rp = Path(p).resolve()
    except Exception:                                        # noqa: BLE001
        return False
    return any(str(rp).startswith(str(root)) for root in ALLOWED_ROOTS)
