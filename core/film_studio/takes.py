"""Storia dei «take» (non-distruttiva) per keyframe/video degli shot. La generazione scrive sul path canonico
(shot_N_kf.png) che la GUI mostra; qui, ad ogni generazione, se ne SNAPSHOTTA una copia versionata in
takes/ tracciata come artifact_version. Così rigenerare non perde i tentativi precedenti e si può ripristinare
un take vecchio come corrente. Riusa il versioning già presente in db (new_artifact/add_version/versions)."""
from __future__ import annotations

import os
import shutil

from core.film_studio import db


def _artifact(owner_kind: str, owner_id: int, kind: str) -> int:
    rows = db.query("artifact", "owner_kind=? AND owner_id=? AND kind=?", (owner_kind, owner_id, kind))
    return rows[0]["id"] if rows else db.new_artifact(kind, owner_kind, owner_id)


def record_take(owner_kind: str, owner_id: int, kind: str, src_file: str, params: dict | None = None) -> str | None:
    """Snapshotta il file appena generato come nuovo take versionato. Ritorna il path del take (o None)."""
    if not (src_file and os.path.exists(src_file)):
        return None
    aid = _artifact(owner_kind, owner_id, kind)
    n = len(db.versions(aid)) + 1
    ext = os.path.splitext(src_file)[1] or (".png" if kind == "keyframe" else ".mp4")
    tdir = os.path.join(os.path.dirname(src_file), "takes")
    os.makedirs(tdir, exist_ok=True)
    dst = os.path.join(tdir, f"{owner_kind}{owner_id}_{kind}_v{n}{ext}")
    shutil.copyfile(src_file, dst)
    db.add_version(aid, dst, params)
    return dst


def list_takes(owner_kind: str, owner_id: int, kind: str) -> list[dict]:
    """[{version, file, params, created}] dei take di uno shot per tipo (keyframe/video)."""
    rows = db.query("artifact", "owner_kind=? AND owner_id=? AND kind=?", (owner_kind, owner_id, kind))
    return db.versions(rows[0]["id"]) if rows else []


def restore_take(owner_kind: str, owner_id: int, kind: str, version: int, canonical_file: str) -> str | None:
    """Ripristina un take come CORRENTE: lo copia sul path canonico che la GUI/il montaggio usano."""
    match = next((t for t in list_takes(owner_kind, owner_id, kind) if t["version"] == version), None)
    if not match or not os.path.exists(match["file"]):
        return None
    shutil.copyfile(match["file"], canonical_file)
    return canonical_file
