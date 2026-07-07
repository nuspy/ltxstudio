"""Persistenza AI Film Studio — SQLite (stdlib, zero dipendenze) con versioning non distruttivo.

Schema (spec §14): Series, Project, Asset, AssetVariation, Wardrobe, Image, Lora, Voice, Script,
Scene, Shot, Artifact (+ versions), Job, AudioClip, MontageTimeline.

DB: output/film_studio/studio.db. Artefatti binari su filesystem (output/film_studio/<project>/...).
Ogni Artifact ha cronologia versioni (tabella artifact_version con seed/params/prompt/file).
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "output" / "film_studio"
DB_PATH = ROOT / "studio.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS series(
  id INTEGER PRIMARY KEY, name TEXT, notes TEXT, created REAL);
CREATE TABLE IF NOT EXISTS project(
  id INTEGER PRIMARY KEY, series_id INTEGER, name TEXT, kind TEXT,
  duration_s REAL, genres TEXT, styles TEXT, path_mode TEXT, run_mode TEXT,
  config TEXT, created REAL, updated REAL);
CREATE TABLE IF NOT EXISTS asset(
  id INTEGER PRIMARY KEY, type TEXT, name TEXT, description TEXT, fields TEXT,
  tags TEXT, scope TEXT, series_id INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS asset_variation(
  id INTEGER PRIMARY KEY, asset_id INTEGER, name TEXT, kind TEXT, description TEXT, created REAL);
CREATE TABLE IF NOT EXISTS wardrobe(
  id INTEGER PRIMARY KEY, variation_id INTEGER, name TEXT, description TEXT, images TEXT, created REAL);
CREATE TABLE IF NOT EXISTS image(
  id INTEGER PRIMARY KEY, owner_kind TEXT, owner_id INTEGER, angle TEXT, repr TEXT,
  file TEXT, seed INTEGER, prompt TEXT, version INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS lora(
  id INTEGER PRIMARY KEY, asset_id INTEGER, variation_id INTEGER, flavor TEXT,
  trigger TEXT, scope TEXT, version INTEGER, file TEXT, dataset TEXT, created REAL);
CREATE TABLE IF NOT EXISTS voice(
  id INTEGER PRIMARY KEY, origin TEXT, name TEXT, params TEXT, sample TEXT,
  actor_id INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS script(
  id INTEGER PRIMARY KEY, project_id INTEGER, text TEXT, summary TEXT, version INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS scene(
  id INTEGER PRIMARY KEY, project_id INTEGER, ord INTEGER, description TEXT, style TEXT,
  duration_s REAL, continuity TEXT, assets TEXT, created REAL);
CREATE TABLE IF NOT EXISTS shot(
  id INTEGER PRIMARY KEY, scene_id INTEGER, ord INTEGER, description TEXT, duration_s REAL,
  dialogues TEXT, keyframes TEXT, refs TEXT, camera TEXT, continuity TEXT, created REAL);
CREATE TABLE IF NOT EXISTS artifact(
  id INTEGER PRIMARY KEY, kind TEXT, owner_kind TEXT, owner_id INTEGER,
  current_version INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS artifact_version(
  id INTEGER PRIMARY KEY, artifact_id INTEGER, version INTEGER, file TEXT,
  params TEXT, created REAL);
CREATE TABLE IF NOT EXISTS job(
  id INTEGER PRIMARY KEY, type TEXT, status TEXT, priority INTEGER, gpu INTEGER,
  params TEXT, est TEXT, artifact_id INTEGER, log TEXT, created REAL, updated REAL);
CREATE TABLE IF NOT EXISTS audioclip(
  id INTEGER PRIMARY KEY, type TEXT, owner_kind TEXT, owner_id INTEGER, source TEXT,
  in_s REAL, out_s REAL, volume REAL, ducking INTEGER, file TEXT, version INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS montage(
  id INTEGER PRIMARY KEY, project_id INTEGER, timeline TEXT, audio_tracks TEXT,
  fps REAL, aspect TEXT, target TEXT, created REAL);
CREATE TABLE IF NOT EXISTS lora_pipeline(
  id INTEGER PRIMARY KEY, asset_id INTEGER, variation_id INTEGER, name TEXT, state TEXT,
  mode TEXT, spec TEXT, img_lora TEXT, img_dataset TEXT, review_photos TEXT, selected_photos TEXT,
  vid_lora TEXT, vid_dataset TEXT, error TEXT, created REAL, updated REAL);
"""

_conn: sqlite3.Connection | None = None


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        ROOT.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        try:
            _conn.execute("PRAGMA journal_mode=WAL")        # WAL: letture concorrenti (UI+scheduler thread)
            _conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:                                   # noqa: BLE001
            pass
        _conn.executescript(_SCHEMA)
        for ddl in ["ALTER TABLE shot ADD COLUMN seed INTEGER DEFAULT 0",
                    "ALTER TABLE shot ADD COLUMN transition TEXT",           # transizione verso lo shot dopo
                    "ALTER TABLE shot ADD COLUMN in_s REAL DEFAULT 0",       # trim montaggio (in/out)
                    "ALTER TABLE shot ADD COLUMN out_s REAL DEFAULT 0",
                    "ALTER TABLE lora ADD COLUMN weight REAL DEFAULT 0.9",    # B6: peso composizione
                    "ALTER TABLE lora ADD COLUMN base_arch TEXT"]:           # arch compatibile (flux/ltx2)
            try:
                _conn.execute(ddl)                          # migration leggera per DB esistenti
            except Exception:                               # noqa: BLE001 (colonna già presente)
                pass
        _conn.commit()
    return _conn


def _now() -> float:
    return time.time()


def insert(table: str, **fields) -> int:
    fields.setdefault("created", _now())
    cols = ", ".join(fields)
    ph = ", ".join("?" for _ in fields)
    c = conn().execute(f"INSERT INTO {table}({cols}) VALUES({ph})", tuple(fields.values()))
    conn().commit()
    return c.lastrowid


def update(table: str, row_id: int, **fields) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    conn().execute(f"UPDATE {table} SET {sets} WHERE id=?", (*fields.values(), row_id))
    conn().commit()


def get(table: str, row_id: int) -> dict | None:
    r = conn().execute(f"SELECT * FROM {table} WHERE id=?", (row_id,)).fetchone()
    return dict(r) if r else None


def query(table: str, where: str = "", params: tuple = (), order: str = "id") -> list[dict]:
    sql = f"SELECT * FROM {table}"
    if where:
        sql += f" WHERE {where}"
    sql += f" ORDER BY {order}"
    return [dict(r) for r in conn().execute(sql, params).fetchall()]


def delete(table: str, row_id: int) -> None:
    conn().execute(f"DELETE FROM {table} WHERE id=?", (row_id,))
    conn().commit()


# ─────────── versioning artefatti (non distruttivo) ───────────
def new_artifact(kind: str, owner_kind: str, owner_id: int) -> int:
    return insert("artifact", kind=kind, owner_kind=owner_kind, owner_id=owner_id, current_version=0)


def add_version(artifact_id: int, file: str, params: dict | None = None) -> int:
    n = len(query("artifact_version", "artifact_id=?", (artifact_id,))) + 1
    vid = insert("artifact_version", artifact_id=artifact_id, version=n, file=file,
                 params=json.dumps(params or {}, ensure_ascii=False))
    update("artifact", artifact_id, current_version=n)
    return vid


def versions(artifact_id: int) -> list[dict]:
    return query("artifact_version", "artifact_id=?", (artifact_id,), order="version")


def current_file(artifact_id: int) -> str | None:
    vs = versions(artifact_id)
    return vs[-1]["file"] if vs else None


# ─────────── helper @Reference ───────────
def find_assets(name_like: str = "", type_: str = "") -> list[dict]:
    w, p = [], []
    if name_like:
        w.append("(name LIKE ? OR tags LIKE ?)"); p += [f"%{name_like}%", f"%{name_like}%"]
    if type_:
        w.append("type=?"); p.append(type_)
    return query("asset", " AND ".join(w), tuple(p))


def project_dir(project_id: int) -> Path:
    d = ROOT / f"project_{project_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _delete_owned(owner_kind: str, owner_id: int) -> None:
    """Elimina gli artefatti/audio posseduti da uno shot (evita righe orfane — B10)."""
    for art in query("artifact", "owner_kind=? AND owner_id=?", (owner_kind, owner_id)):
        for av in query("artifact_version", "artifact_id=?", (art["id"],)):
            delete("artifact_version", av["id"])
        delete("artifact", art["id"])
    for ac in query("audioclip", "owner_kind=? AND owner_id=?", (owner_kind, owner_id)):
        delete("audioclip", ac["id"])


def delete_project(project_id: int) -> None:
    """Elimina il progetto e TUTTE le entità collegate: scene/shot + artifact/artifact_version/audioclip degli
    shot + script/montage. NB asset/lora/voice NON si toccano: sono condivisi (per-serie/globali), non
    per-progetto (B10: prima restavano righe orfane di artifact/audioclip)."""
    for sc in query("scene", "project_id=?", (project_id,)):
        for sh in query("shot", "scene_id=?", (sc["id"],)):
            _delete_owned("shot", sh["id"])
            delete("shot", sh["id"])
        delete("scene", sc["id"])
    for t in ("script", "montage"):
        for r in query(t, "project_id=?", (project_id,)):
            delete(t, r["id"])
    delete("project", project_id)


def duplicate_project(project_id: int) -> int:
    """Duplica progetto + copione + scene + shot (gli artefatti binari NON sono ri-renderizzati)."""
    p = get("project", project_id)
    if not p:
        raise ValueError("progetto inesistente")
    p.pop("id", None)
    p["name"] = (p.get("name") or "progetto") + " (copia)"
    nid = insert("project", **p)
    for sc in query("scene", "project_id=?", (project_id,), order="ord"):
        old_scid = sc.pop("id")
        sc["project_id"] = nid
        new_scid = insert("scene", **sc)
        for sh in query("shot", "scene_id=?", (old_scid,), order="ord"):
            sh.pop("id", None); sh["scene_id"] = new_scid
            insert("shot", **sh)
    for s in query("script", "project_id=?", (project_id,)):
        s.pop("id", None); s["project_id"] = nid
        insert("script", **s)
    for mt in query("montage", "project_id=?", (project_id,)):    # B10: copia anche il montaggio
        mt.pop("id", None); mt["project_id"] = nid
        insert("montage", **mt)
    return nid
