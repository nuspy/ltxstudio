"""Breakdown Scene→Shot via LLM (JSON) — logica CORE riusata da Cinema Studio (storyboard) e Lucky Director.

`SYS` è lo stesso system-prompt usato dallo storyboard; `build()` popola scene/shot nel DB (con propagazione
della continuity), sostituendo l'eventuale breakdown precedente. Headless: nessuna dipendenza Qt.
"""
from __future__ import annotations

import json

from core import llm_service as L
from core.film_studio import db, continuity as ct

# Schema INTERNO (invisibile all'utente): shot arricchito con camera+lighting per prompt più cinematografici.
SYS = ("You are a film script supervisor. Break the screenplay into SCENES and SHOTS. "
       "Output STRICT JSON: {\"scenes\":[{\"description\":str,\"style\":str,\"duration_s\":number,"
       "\"continuity\":{\"time_of_day\":str,\"weather\":str,\"mood\":str},"
       "\"shots\":[{\"description\":str,\"duration_s\":number,"
       "\"camera\":str,\"lighting\":str,"
       "\"dialogues\":[{\"actor\":str,\"text\":str}],\"keyframe\":str}]}]}. "
       "camera = concrete shot type + angle (e.g. 'wide shot, low angle', 'close-up'). "
       "lighting = light + mood (e.g. 'soft golden hour', 'harsh neon'). "
       "description = the visual content of the shot (subject + action + setting). "
       "Sum of shot durations = scene duration; sum of scene durations = film duration. "
       "Keep @Name references. No prose outside JSON.")


def build(pid: int, *, max_tokens: int = 6000) -> int:
    """Genera il breakdown dal copione (o Racconto) del progetto e lo salva nel DB. Ritorna il n. di scene.
    Solleva se manca il copione o l'LLM non produce JSON valido."""
    proj = db.get("project", pid) or {}
    copione = db.latest_script(pid, "script") or db.latest_script(pid, "narrative")
    if not copione:
        raise RuntimeError("Copione mancante: genera prima il copione")
    racconto = db.latest_script(pid, "narrative")
    ctx = ("Cinematic treatment:\n" + racconto["text"] + "\n\n") if racconto else ""
    # VALIDAZIONE DI FORMA + retry: su output troncato il parser di complete_json può ripiegare
    # sull'ARRAY interno (lista, non dict) o su un dict senza 'scenes' → prima finiva in
    # "0 scene salvate" SENZA errore e auto_run moriva con "Nessuna ripresa da generare".
    data = {}
    for _att in range(3):
        data = L.complete_json(SYS, f"Film duration: {proj.get('duration_s', 60)}s.\n\n{ctx}" + copione["text"],
                               max_tokens=int(max_tokens * (1 + 0.5 * _att)))
        if isinstance(data, dict) and data.get("scenes"):
            break
        print(f"[breakdown] risposta senza 'scenes' (tentativo {_att + 1}: "
              f"{type(data).__name__}) → riprovo con più budget", flush=True)
    else:
        raise RuntimeError("breakdown: l'LLM non ha prodotto scene valide dopo 3 tentativi")
    # sostituisci scene/shot precedenti del progetto
    for sc in db.query("scene", "project_id=?", (pid,)):
        for sh in db.query("shot", "scene_id=?", (sc["id"],)):
            db.delete("shot", sh["id"])
        db.delete("scene", sc["id"])
    prev_state = None
    scenes = data.get("scenes", []) if isinstance(data, dict) else []
    for i, sc in enumerate(scenes, 1):
        state = ct.propagate(prev_state, sc.get("continuity"))
        prev_state = state
        sid = db.insert("scene", project_id=pid, ord=i, description=sc.get("description", ""),
                        style=sc.get("style", ""), duration_s=float(sc.get("duration_s", 0) or 0),
                        continuity=json.dumps(state, ensure_ascii=False), assets="")
        for j, sh in enumerate(sc.get("shots", []), 1):
            cam = ", ".join(x for x in (sh.get("camera", ""), sh.get("lighting", "")) if x)
            db.insert("shot", scene_id=sid, ord=j, description=sh.get("description", ""),
                      duration_s=float(sh.get("duration_s", 0) or 0),
                      dialogues=json.dumps(sh.get("dialogues", []), ensure_ascii=False),
                      keyframes=json.dumps([sh.get("keyframe", "")], ensure_ascii=False),
                      refs="", camera=cam, continuity=json.dumps(state, ensure_ascii=False))
    return len(scenes)


def normalize_durations(pid: int, target_s: float, *, min_shot: float = 2.0, max_shot: float = 12.0) -> None:
    """Scala proporzionalmente le durate di scene/shot così che il totale ≈ target_s (distribuzione della
    durata richiesta da Lucky Director). Clamp per-shot per restare in un range ragionevole per LTX."""
    if target_s <= 0:
        return
    shots = []
    for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
        shots += db.query("shot", "scene_id=?", (sc["id"],), order="ord")
    total = sum(float(s["duration_s"] or 0) for s in shots)
    if total <= 0 or not shots:
        return
    k = target_s / total
    for s in shots:
        nd = max(min_shot, min(max_shot, float(s["duration_s"] or 0) * k))
        db.update("shot", s["id"], duration_s=round(nd, 2))
    # aggiorna le durate di scena come somma degli shot
    for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
        ss = sum(float(x["duration_s"] or 0) for x in db.query("shot", "scene_id=?", (sc["id"],)))
        db.update("scene", sc["id"], duration_s=round(ss, 2))
