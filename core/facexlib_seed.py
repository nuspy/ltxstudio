"""Pre-seed dei pesi facexlib (FASE H weights-unification) — l'ultimo download silenzioso.

GFPGANer/facexlib scaricano `detection_Resnet50_Final.pth` e `parsing_parsenet.pth` da GitHub
al PRIMO restauro volti, dentro `<cwd>/gfpgan/weights` (cwd = repo per i nostri processi) —
senza consenso, senza resume, su connessione metered. Ora i due file sono voci di CATALOGO
(facexlib-detection/-parsing → facerestore_models\\facexlib) e questo helper li COLLEGA
(hardlink/junction, senza copiare) nel punto esatto dove facexlib li cerca, PRIMA di istanziare
GFPGANer. Se le voci non sono scaricate, non fa nulla: facexlib segue il suo corso storico
(il tab Pesi le offre comunque).
"""
from __future__ import annotations

import os
from pathlib import Path

_IDS = ("facexlib-detection", "facexlib-parsing")


def ensure(cwd: str | None = None, log=print) -> list[str]:
    """Collega i pesi facexlib dal catalogo in `<cwd>/gfpgan/weights`. Ritorna i nomi seedati.
    Best-effort: mai solleva (il restauro deve poter continuare comunque)."""
    seeded: list[str] = []
    try:
        from core import linkutil, model_paths
        dest_dir = Path(cwd or os.getcwd()) / "gfpgan" / "weights"
        for eid in _IDS:
            src = model_paths.path_of(eid)
            if not src or not Path(src).is_file():
                continue                                 # non scaricato: niente da seedare
            dst = dest_dir / Path(src).name
            if dst.exists():
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            r = linkutil.link_or_copy(str(src), str(dst))
            if r in ("hardlink", "junction", "symlink", "copy", "exists"):
                seeded.append(dst.name)
                log(f"[facexlib] pre-seed {dst.name} ({r}) → niente download al primo restauro")
    except Exception as e:                               # noqa: BLE001
        log(f"[facexlib] pre-seed saltato ({e})")
    return seeded
