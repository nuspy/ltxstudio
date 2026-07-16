"""Costruzione della lista di transizioni per il master, a DUE LIVELLI (tra scene e tra riprese).

Condiviso tra il montaggio manuale di Cinema Studio (`ui/film_studio/montage_tab.py`) e Lucky Director
(`core/film_studio/auto_run.py`) → STESSO codice, nessuna duplicazione.

`build_transitions()` produce la lista attesa da `otio_export.render_master(transitions=[...])`:
un elemento per ogni confine tra clip consecutive (len = n_clip - 1). Per ogni confine:
  - se la ripresa corrente ha una transizione esplicita (override `shot.transition`) → quella;
  - altrimenti, se la clip successiva appartiene a una SCENA diversa → transizione TRA SCENE;
  - altrimenti (stessa scena) → transizione TRA RIPRESE.
Tipo vuoto ("") = stacco netto (dur 0).
"""
from __future__ import annotations


def build_transitions(clips_meta: list[dict], *, scene_trans: str = "", shot_trans: str = "",
                      scene_dur: float = 0.7, shot_dur: float = 0.5) -> list[dict]:
    """clips_meta: clip ORDINATE del master, ogni dict con almeno `scene_id`; opzionale `transition`
    (override per-ripresa). Ritorna [{type, dur}, ...] lungo n-1. Vedi docstring del modulo."""
    out: list[dict] = []
    n = len(clips_meta)
    for i in range(n - 1):
        cur, nxt = clips_meta[i], clips_meta[i + 1]
        override = str(cur.get("transition") or "").strip()
        if override:
            out.append({"type": override, "dur": float(shot_dur)})
        elif cur.get("scene_id") != nxt.get("scene_id"):
            out.append({"type": scene_trans, "dur": float(scene_dur)} if scene_trans else {"type": "", "dur": 0})
        else:
            out.append({"type": shot_trans, "dur": float(shot_dur)} if shot_trans else {"type": "", "dur": 0})
    return out


def has_any(transitions: list[dict] | None) -> bool:
    """True se almeno una transizione reale (type non vuoto e dur>0) → altrimenti conviene passare None."""
    return bool(transitions) and any((t or {}).get("type") and (t or {}).get("dur") for t in transitions)
