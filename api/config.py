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
BASE_URL = f"http://{HOST}:{PORT}"

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
