"""Collegamento SENZA privilegi dei modelli locali già presenti — così l'installer/wizard NON li riscarica.

Strategia per OS (nessun admin / Developer Mode richiesto):
- **Windows**: file → **hardlink** (`os.link`, stesso volume); directory → **junction** (`mklink /J`). Entrambi
  NON richiedono elevazione (a differenza dei symlink su Windows).
- **macOS/Linux**: **symlink** nativo.
- **Fallback**: **copia** (con callback di conferma/progress) se il link è impossibile — es. volume diverso su
  Windows (hardlink solo stesso volume), o filesystem senza symlink.

`unlink_safe` rimuove SOLO il collegamento (mai i dati sorgente dell'utente).
Puro stdlib; nessuna dipendenza da torch/hf.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# FILE_ATTRIBUTE_REPARSE_POINT (junction/symlink su Windows)
_REPARSE = 0x400


def _same_volume(src: str, dst_parent: str) -> bool:
    """True se src e la cartella di destinazione stanno sullo STESSO volume (hardlink lo richiede)."""
    try:
        if os.name == "nt":
            a = os.path.splitdrive(os.path.abspath(src))[0].lower()
            b = os.path.splitdrive(os.path.abspath(dst_parent))[0].lower()
            return bool(a) and a == b
        return os.stat(src).st_dev == os.stat(dst_parent or ".").st_dev
    except Exception:                                       # noqa: BLE001
        return False


def is_linked(path: str) -> bool:
    """True se `path` è un collegamento (symlink o junction/reparse-point su Windows), non un file/dir reale."""
    p = Path(path)
    try:
        if p.is_symlink():
            return True
        if os.name == "nt" and p.exists():
            attrs = getattr(os.stat(str(p), follow_symlinks=False), "st_file_attributes", 0)
            return bool(attrs & _REPARSE)
    except Exception:                                       # noqa: BLE001
        pass
    return False


def would_copy(src: str, dst: str) -> bool:
    """True se `link_or_copy` dovrebbe RICORRERE ALLA COPIA perché il link non è possibile: su Windows un FILE
    su volume DIVERSO dalla destinazione (l'hardlink richiede lo stesso volume). Le dir usano junction (ok) e
    mac/Linux i symlink (ok) → False. Serve al chiamante per AVVISARE prima di una copia grande."""
    if os.name != "nt" or os.path.isdir(src):
        return False
    return not _same_volume(src, os.path.dirname(os.path.abspath(dst)))


def link_or_copy(src: str, dst: str, *, on_copy=None, progress=None) -> str:
    """Collega `src`→`dst` senza admin (o copia in fallback). `src` file o directory.
    Ritorna: 'exists' | 'hardlink' | 'junction' | 'symlink' | 'copy' | 'skipped' | 'error'.
    `on_copy(src, dst) -> bool`: se il link non è possibile, chiede conferma prima di copiare (None = copia sempre)."""
    src, dst = str(src), str(dst)
    if not os.path.exists(src):
        return "error"
    if os.path.exists(dst) or is_linked(dst):
        return "exists"
    parent = os.path.dirname(os.path.abspath(dst))
    os.makedirs(parent, exist_ok=True)
    is_dir = os.path.isdir(src)
    # 1) link nativo (no admin)
    try:
        if os.name == "nt":
            if is_dir:
                r = subprocess.run(["cmd", "/c", "mklink", "/J", dst, src],
                                   capture_output=True, text=True)
                if r.returncode == 0 and os.path.exists(dst):
                    return "junction"
            elif _same_volume(src, parent):
                os.link(src, dst)
                return "hardlink"
        else:
            os.symlink(src, dst)
            return "symlink"
    except Exception:                                       # noqa: BLE001 — cade sulla copia
        pass
    # 2) fallback copia (volume diverso / link non permesso)
    if on_copy is not None and not on_copy(src, dst):
        return "skipped"
    try:
        if is_dir:
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
            if progress:
                try:
                    progress(os.path.getsize(dst))
                except Exception:                           # noqa: BLE001
                    pass
        return "copy"
    except Exception:                                       # noqa: BLE001
        return "error"


def unlink_safe(dst: str) -> bool:
    """Rimuove SOLO il collegamento `dst` (junction/symlink/hardlink), MAI i dati sorgente dell'utente.
    Ritorna True se rimosso. Se `dst` è un file/dir REALE (non un link), NON tocca nulla e ritorna False.
    (Un hardlink-file è un secondo nome dello stesso inode: rimuoverlo non cancella l'originale.)"""
    p = Path(dst)
    try:
        if p.is_symlink():
            p.unlink()
            return True
        if os.name == "nt" and is_linked(dst) and p.is_dir():
            os.rmdir(dst)                                   # junction: rimuove il punto, non il target
            return True
        # hardlink (qualsiasi OS): st_nlink ≥ 2 → rimuovere QUESTO nome è sicuro (l'inode resta sull'altro link)
        if p.is_file() and getattr(os.stat(dst), "st_nlink", 1) >= 2:
            os.remove(dst)
            return True
        return False                                        # dato reale non-linkato → non cancellare
    except Exception:                                       # noqa: BLE001
        return False
