"""Moonshot — gli agenti «autoriali» del Film Studio (LLM). Non generano pixel: leggono il breakdown già
prodotto e RILANCIANO come farebbe una troupe. Quattro ruoli, tutti best-effort e NON distruttivi (propongono,
non impongono — l'utente applica ciò che vuole):

- `script_supervisor(pid)`  — il segretario di edizione: rilegge scene/shot e segnala i SALTI di raccordo
   (180°/eyeline, oggetti che compaiono/spariscono, ora del giorno che torna indietro, condizioni perse).
   Unisce i warning DETERMINISTICI del continuity engine con quelli semantici dell'LLM.
- `dop_suggestions(scene_id, style_bible)` — il direttore della fotografia: dalla Style Bible propone
   camera+luce per OGNI shot della scena (dialoga: ritorna suggerimenti che l'utente può applicare shot-per-shot).
- `second_look(pid)` — «il secondo sguardo»: il critico post-visione che dice il «no» creativo finale
   («scene 4-6 corrette ma anonime»), con 2-4 rilievi concreti e un giudizio complessivo.
- `emotional_rhythm(pid, tension)` — il montatore a ritmo emotivo: da una curva di tensione (0..1 per scena)
   modula le DURATE degli shot (tensione alta → tagli più corti; bassa → respiro), mantenendo il totale.

Tutte le funzioni degradano con grazia se l'LLM non è disponibile (le due deterministiche restano utili).
"""
from __future__ import annotations

import json

from core.film_studio import continuity as ct, db


# ─────────────────────────────── contesto per gli LLM ───────────────────────────────
def collect_breakdown(pid: int) -> list[dict]:
    """Struttura leggibile del film: [{ord, description, continuity, shots:[{ord, description, camera,
    dialogues, duration_s}]}]. Base condivisa dagli agenti (niente pixel, solo il piano)."""
    scenes = []
    for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
        shots = []
        for sh in db.query("shot", "scene_id=?", (sc["id"],), order="ord"):
            shots.append({"id": sh["id"], "ord": sh["ord"], "description": sh.get("description", ""),
                          "camera": sh.get("camera", ""), "duration_s": float(sh.get("duration_s") or 0),
                          "dialogues": json.loads(sh.get("dialogues") or "[]")})
        scenes.append({"id": sc["id"], "ord": sc["ord"], "description": sc.get("description", ""),
                       "continuity": json.loads(sc.get("continuity") or "{}"), "shots": shots})
    return scenes


def _breakdown_text(scenes: list[dict]) -> str:
    """Rende il breakdown come testo compatto per il prompt (numerato scene→shot, con camera e dialoghi)."""
    out = []
    for sc in scenes:
        st = sc.get("continuity") or {}
        head = f"SCENE {sc['ord']}: {sc['description']}".strip()
        tag = ", ".join(f"{k}={st[k]}" for k in ("time_of_day", "weather", "light", "mood") if st.get(k))
        out.append(head + (f"  [{tag}]" if tag else ""))
        for sh in sc["shots"]:
            dl = "; ".join(f"{d.get('actor', '')}:{d.get('text', '')}" for d in sh.get("dialogues") or [])
            cam = f" <{sh['camera']}>" if sh.get("camera") else ""
            out.append(f"  {sc['ord']}.{sh['ord']} ({sh['duration_s']:.0f}s){cam} {sh['description']}"
                       + (f"  | {dl}" if dl else ""))
    return "\n".join(out)


# ─────────────────────────────── 1) segretario di edizione ───────────────────────────────
_SUP_SYS = (
    "You are a virtual SCRIPT SUPERVISOR (continuity/script supervisor) for a film. "
    "Read the numbered scene/shot breakdown and flag CONTINUITY problems a supervisor would catch: "
    "180-degree rule / eyeline mismatches in dialogue coverage, objects or props that appear or vanish "
    "between shots without cause, time-of-day or weather that jumps backward, a character's condition "
    "(wounded, wet, dressed) that resets, screen-direction reversals. "
    "Output STRICT JSON: {\"issues\":[{\"where\":\"S.shot or scene\",\"kind\":str,\"note\":str,\"severity\":\"low|med|high\"}]}. "
    "Only real, specific issues grounded in the text. If none, return {\"issues\":[]}. No prose outside JSON."
)


def script_supervisor(pid: int) -> dict:
    """{'deterministic':[str], 'llm':[{where,kind,note,severity}], 'available':bool}. Non solleva."""
    scenes = collect_breakdown(pid)
    det = []
    prev = None
    for sc in scenes:                                        # warning deterministici del continuity engine
        cur = sc.get("continuity") or {}
        for w in ct.warnings(prev, cur):
            det.append(f"S{sc['ord']}: {w}")
        prev = cur
    llm_issues, available = [], False
    try:
        from core import llm_service as L
        if L.available() and scenes:
            available = True
            data = L.complete_json(_SUP_SYS, _breakdown_text(scenes), max_tokens=1400)
            llm_issues = [i for i in (data.get("issues") or []) if isinstance(i, dict) and i.get("note")]
    except Exception:                                        # noqa: BLE001
        pass
    return {"deterministic": det, "llm": llm_issues, "available": available}


# ─────────────────────────────── 2) direttore della fotografia ───────────────────────────────
_DOP_SYS = (
    "You are a DIRECTOR OF PHOTOGRAPHY. Given a film's Style Bible and a scene's shots, propose for EACH shot "
    "a concrete camera framing+angle+movement and a lighting setup consistent with the Style Bible and the "
    "scene mood. Be specific and cinematic but SERVE the story (do not overpower it). "
    "Output STRICT JSON: {\"shots\":[{\"ord\":number,\"camera\":str,\"lighting\":str,\"rationale\":str}]}. "
    "camera e.g. 'medium close-up, slight low angle, slow push-in'; lighting e.g. 'hard key from window, deep "
    "shadows, teal ambient'. One entry per shot given, same ord. No prose outside JSON."
)


def _style_bible_text(pid: int) -> str:
    cfg = json.loads((db.get("project", pid) or {}).get("config") or "{}")
    sb = cfg.get("style_bible") or {}
    bits = [f"{k}: {sb[k]}" for k in ("logline", "palette", "look", "directors") if sb.get(k)]
    return "; ".join(bits) or "(no style bible set — infer a coherent cinematic look)"


def dop_suggestions(scene_id: int) -> list[dict]:
    """[{shot_id, ord, camera, lighting, rationale}] proposte dal DoP per gli shot della scena. [] se non fattibile."""
    sc = db.get("scene", scene_id)
    if not sc:
        return []
    shots = db.query("shot", "scene_id=?", (scene_id,), order="ord")
    if not shots:
        return []
    st = json.loads(sc.get("continuity") or "{}")
    user = (f"Style Bible: {_style_bible_text(sc['project_id'])}\n"
            f"Scene mood: {st.get('mood', 'neutral')}, {st.get('time_of_day', 'day')}, "
            f"{st.get('light', 'natural')} light.\nShots:\n"
            + "\n".join(f"  ord {sh['ord']}: {sh.get('description', '')}" for sh in shots))
    try:
        from core import llm_service as L
        if not L.available():
            return []
        data = L.complete_json(_DOP_SYS, user, max_tokens=1200)
    except Exception:                                        # noqa: BLE001
        return []
    by_ord = {int(sh["ord"]): sh["id"] for sh in shots}
    out = []
    for e in (data.get("shots") or []):
        try:
            o = int(e.get("ord"))
        except Exception:                                   # noqa: BLE001
            continue
        if o in by_ord and (e.get("camera") or e.get("lighting")):
            out.append({"shot_id": by_ord[o], "ord": o, "camera": (e.get("camera") or "").strip(),
                        "lighting": (e.get("lighting") or "").strip(), "rationale": (e.get("rationale") or "").strip()})
    return out


def apply_dop(suggestions: list[dict]) -> int:
    """Applica le proposte del DoP: scrive shot.camera = 'camera, lighting'. Ritorna quanti shot aggiornati."""
    n = 0
    for s in suggestions:
        cam = ", ".join(x for x in (s.get("camera", ""), s.get("lighting", "")) if x)
        if cam and s.get("shot_id"):
            db.update("shot", int(s["shot_id"]), camera=cam)
            n += 1
    return n


# ─────────────────────────────── 3) il secondo sguardo (critico) ───────────────────────────────
_CRIT_SYS = (
    "You are a demanding film CRITIC giving the final creative 'no' before release. You have the shot breakdown "
    "(you cannot see the pixels — judge the plan: rhythm, coverage, visual variety, emotional arc, cliché). "
    "Be honest and specific, not flattering. Name scenes by number. "
    "Output STRICT JSON: {\"verdict\":str,\"notes\":[{\"where\":str,\"note\":str}],\"strengths\":[str]}. "
    "verdict = one-sentence overall judgement. notes = 2-5 concrete critiques. No prose outside JSON."
)


def second_look(pid: int) -> dict:
    """{'verdict':str,'notes':[{where,note}],'strengths':[str],'available':bool}. Non solleva."""
    scenes = collect_breakdown(pid)
    try:
        from core import llm_service as L
        if not (L.available() and scenes):
            return {"verdict": "", "notes": [], "strengths": [], "available": False}
        sb = _style_bible_text(pid)
        data = L.complete_json(_CRIT_SYS, f"Style Bible: {sb}\n\n{_breakdown_text(scenes)}", max_tokens=1200)
        return {"verdict": (data.get("verdict") or "").strip(),
                "notes": [n for n in (data.get("notes") or []) if isinstance(n, dict) and n.get("note")],
                "strengths": [s for s in (data.get("strengths") or []) if s], "available": True}
    except Exception:                                       # noqa: BLE001
        return {"verdict": "", "notes": [], "strengths": [], "available": False}


# ─────────────────────────────── 4) montaggio a ritmo emotivo ───────────────────────────────
def emotional_rhythm(pid: int, tension: list[float]) -> list[dict]:
    """Da una curva di tensione per SCENA (0..1) rimodula le durate degli shot: tensione alta → tagli più
    corti (ritmo incalzante), bassa → respiro. Mantiene invariata la durata TOTALE di ogni scena (redistribuisce
    tra i suoi shot). Ritorna [{shot_id, old, new}] SENZA scrivere (l'utente conferma). Deterministico, no LLM.

    Modello: fattore di lunghezza per shot = 1/(0.4 + 1.2*t) → a t=0 shot ~1.79× la media, a t=1 ~0.62×.
    """
    scenes = collect_breakdown(pid)
    out = []
    for i, sc in enumerate(scenes):
        shots = sc["shots"]
        if not shots:
            continue
        t = float(tension[i]) if i < len(tension) else 0.5
        t = max(0.0, min(1.0, t))
        total = sum(float(sh["duration_s"] or 0) for sh in shots) or (len(shots) * 4.0)
        factor = 1.0 / (0.4 + 1.2 * t)                      # curva monotòna decrescente in t
        weights = [factor] * len(shots)                     # (uniforme entro la scena: la tensione è per-scena)
        wsum = sum(weights) or 1.0
        for sh, w in zip(shots, weights):
            new = round(total * w / wsum, 1)
            new = max(0.6, new)                             # nessuno shot sotto ~0.6s
            out.append({"shot_id": sh["id"], "ord": f"{sc['ord']}.{sh['ord']}",
                        "old": float(sh["duration_s"] or 0), "new": new})
    return out


def apply_rhythm(changes: list[dict]) -> int:
    """Scrive le nuove durate proposte da emotional_rhythm. Ritorna quanti shot aggiornati."""
    n = 0
    for c in changes:
        if c.get("shot_id") and c.get("new"):
            db.update("shot", int(c["shot_id"]), duration_s=float(c["new"]))
            n += 1
    return n
