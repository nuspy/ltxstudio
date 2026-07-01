"""Gestione file: link di download + upload dei file portati dall'utente.

Ogni file prodotto da un job viene esposto come `{path, download_url}`. `download_url` è un
`GET /download?path=<abs>` servito solo se il path è dentro le cartelle consentite (anti-traversal).
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from api.config import BASE_URL, UPLOAD_DIR, is_allowed_path


def download_url(path: str) -> str:
    """URL di download per un file locale (o stringa vuota se non servibile)."""
    from urllib.parse import quote
    if not path or not is_allowed_path(path):
        return ""
    return f"{BASE_URL}/download?path={quote(str(path))}"


def file_entry(path: str) -> dict:
    """Rappresentazione API di un file prodotto: path locale + link di download."""
    p = Path(path)
    return {"path": str(p), "name": p.name, "download_url": download_url(str(p))}


def files_payload(paths: list[str]) -> list[dict]:
    return [file_entry(p) for p in paths if p]


def save_upload(upload: UploadFile) -> str:
    """Salva un UploadFile sotto UPLOAD_DIR con nome univoco. Ritorna il path locale."""
    suffix = Path(upload.filename or "file").suffix or ".bin"
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return str(dest)
