"""Persistenza progetti VIDEOMUSICALI — SQLite (stdlib) con fasi salvate e riapribili.

DB: output/music_video/mvprojects.db. Tre tabelle:
- mv_project: metadati + parametri input + fase corrente + canzone attiva
- mv_song:    tutte le canzoni generate per il progetto (lista selezionabile)
- mv_scene:   elenco riprese (start/length/prompt immagine+video/immagine/video) della canzone attiva

Fasi: 'song' → 'storyboard' (immagini+prompt) → 'video' → 'done'. Salvataggio esplicito per fase.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "output" / "music_video"
DB_PATH = ROOT / "mvprojects.db"

PHASES = ["song", "storyboard", "video", "done"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mv_project(
  id INTEGER PRIMARY KEY, name TEXT, lyrics TEXT, tags TEXT, image_hint TEXT,
  scene_seconds REAL, video_style TEXT, song_duration REAL, base_image TEXT,
  song_temperature REAL, song_cfg REAL,
  active_song_id INTEGER, phase TEXT, created REAL, updated REAL);
CREATE TABLE IF NOT EXISTS mv_song(
  id INTEGER PRIMARY KEY, project_id INTEGER, file TEXT, params TEXT, created REAL);
CREATE TABLE IF NOT EXISTS mv_scene(
  id INTEGER PRIMARY KEY, project_id INTEGER, ord INTEGER, start_s REAL, length_s REAL,
  img_prompt TEXT, video_prompt TEXT, image_path TEXT, video_path TEXT);
"""

_conn: sqlite3.Connection | None = None


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        ROOT.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(_SCHEMA)
        _conn.commit()
    return _conn


def _now() -> float:
    return time.time()


# ───────────── progetti ─────────────
def create_project(name: str, *, lyrics: str = "", tags: str = "", image_hint: str = "",
                   scene_seconds: float = 4.0, video_style: str = "", song_duration: float = 120.0,
                   base_image: str = "", song_temperature: float = 1.0, song_cfg: float = 1.5) -> int:
    c = conn().execute(
        "INSERT INTO mv_project(name,lyrics,tags,image_hint,scene_seconds,video_style,song_duration,"
        "base_image,song_temperature,song_cfg,active_song_id,phase,created,updated) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (name, lyrics, tags, image_hint, scene_seconds, video_style, song_duration,
         base_image, song_temperature, song_cfg, 0, "song", _now(), _now()))
    conn().commit()
    return c.lastrowid


def list_projects() -> list[dict]:
    return [dict(r) for r in conn().execute("SELECT * FROM mv_project ORDER BY updated DESC").fetchall()]


def get_project(pid: int) -> dict | None:
    r = conn().execute("SELECT * FROM mv_project WHERE id=?", (pid,)).fetchone()
    return dict(r) if r else None


def update_project(pid: int, **fields) -> None:
    if not fields:
        return
    fields["updated"] = _now()
    sets = ", ".join(f"{k}=?" for k in fields)
    conn().execute(f"UPDATE mv_project SET {sets} WHERE id=?", (*fields.values(), pid))
    conn().commit()


def delete_project(pid: int) -> None:
    for t in ("mv_song", "mv_scene"):
        conn().execute(f"DELETE FROM {t} WHERE project_id=?", (pid,))
    conn().execute("DELETE FROM mv_project WHERE id=?", (pid,))
    conn().commit()


def duplicate_project(pid: int) -> int:
    p = get_project(pid)
    if not p:
        raise ValueError("progetto inesistente")
    nid = create_project(p["name"] + " (copia)", lyrics=p["lyrics"], tags=p["tags"],
                         image_hint=p["image_hint"], scene_seconds=p["scene_seconds"],
                         video_style=p["video_style"], song_duration=p["song_duration"])
    for s in list_scenes(pid):
        add_scene(nid, s)
    return nid


# ───────────── canzoni ─────────────
def add_song(pid: int, file: str, params: str = "") -> int:
    c = conn().execute("INSERT INTO mv_song(project_id,file,params,created) VALUES(?,?,?,?)",
                       (pid, file, params, _now()))
    conn().commit()
    sid = c.lastrowid
    if not get_project(pid).get("active_song_id"):
        update_project(pid, active_song_id=sid)
    return sid


def list_songs(pid: int) -> list[dict]:
    return [dict(r) for r in conn().execute(
        "SELECT * FROM mv_song WHERE project_id=? ORDER BY id", (pid,)).fetchall()]


def set_active_song(pid: int, song_id: int) -> None:
    update_project(pid, active_song_id=song_id)


def active_song(pid: int) -> dict | None:
    p = get_project(pid)
    if not p or not p.get("active_song_id"):
        return None
    r = conn().execute("SELECT * FROM mv_song WHERE id=?", (p["active_song_id"],)).fetchone()
    return dict(r) if r else None


def delete_song(song_id: int) -> None:
    conn().execute("DELETE FROM mv_song WHERE id=?", (song_id,))
    conn().commit()


# ───────────── scene (riprese) ─────────────
def save_scenes(pid: int, scenes: list[dict]) -> None:
    """Sovrascrive l'elenco riprese del progetto (snapshot della fase storyboard/video)."""
    conn().execute("DELETE FROM mv_scene WHERE project_id=?", (pid,))
    for s in scenes:
        add_scene(pid, s)
    conn().commit()


def add_scene(pid: int, s: dict) -> int:
    c = conn().execute(
        "INSERT INTO mv_scene(project_id,ord,start_s,length_s,img_prompt,video_prompt,image_path,video_path)"
        " VALUES(?,?,?,?,?,?,?,?)",
        (pid, int(s.get("order", 0)), float(s.get("start_s", 0)), float(s.get("length_s", 0)),
         s.get("img_prompt", ""), s.get("video_prompt", ""), s.get("image_path", ""), s.get("video_path", "")))
    conn().commit()
    return c.lastrowid


def list_scenes(pid: int) -> list[dict]:
    rows = conn().execute("SELECT * FROM mv_scene WHERE project_id=? ORDER BY ord", (pid,)).fetchall()
    return [{"order": r["ord"], "start_s": r["start_s"], "length_s": r["length_s"],
             "img_prompt": r["img_prompt"], "video_prompt": r["video_prompt"],
             "image_path": r["image_path"], "video_path": r["video_path"]} for r in rows]


def project_dir(pid: int) -> Path:
    d = ROOT / f"project_{pid}"
    d.mkdir(parents=True, exist_ok=True)
    return d
