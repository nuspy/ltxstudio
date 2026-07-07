"""Copertura dialoghi (shot-reverse-shot): un'inquadratura con più battute è cinematograficamente sbagliata
(una sola ripresa, un solo lip-sync). Qui la si «copre»: la si espande in N inquadrature, una per battuta,
alternando gli angoli tra i due interlocutori (campo/controcampo).

plan_coverage(dialogues, total_dur) → lista di unità {speaker, text, camera, dur}.
expand_shot_to_coverage(shot_id) → crea gli shot di copertura nella scena (il primo riusa lo shot originale).
"""
from __future__ import annotations

import json

from core.film_studio import db

# angoli alternati per il campo/controcampo (indice speaker → angolo)
_ANGLES = ["medium close-up, over-the-shoulder favoring the speaker",
           "medium close-up, reverse angle over-the-shoulder favoring the speaker",
           "medium shot, clean single", "close-up, clean single"]


def plan_coverage(dialogues: list, total_dur: float) -> list[dict]:
    """Piano di copertura: una unità per battuta con testo, angolo alternato per interlocutore, durata equa."""
    lines = [d for d in (dialogues or []) if (d.get("text") or "").strip()]
    if not lines:
        return []
    speakers = []
    for d in lines:
        s = (d.get("actor") or "").lstrip("@")
        if s and s not in speakers:
            speakers.append(s)
    units = []
    per = float(total_dur or 4) / len(lines)
    for d in lines:
        s = (d.get("actor") or "").lstrip("@")
        idx = speakers.index(s) if s in speakers else 0
        units.append({"speaker": s, "text": d["text"], "camera": _ANGLES[idx % len(_ANGLES)],
                      "dur": round(per, 2)})
    return units


_WARD_ANGLES = ["front full-body", "three-quarter left", "three-quarter right", "profile side",
                "back view", "outfit detail close-up"]


def wardrobe_prompts(char_desc: str, wardrobe_desc: str, n: int = 6) -> list[tuple[str, str]]:
    """[(angle_id, prompt)] per generare il personaggio con un abbigliamento da N angoli (turnaround outfit).
    Prima l'abbigliamento era solo un metadato: qui diventa un set di immagini coerenti."""
    base = f"{(char_desc or 'the character').strip()} wearing {(wardrobe_desc or 'the outfit').strip()}"
    return [(a.replace(" ", "_"), f"{base}, {a}, full outfit visible, neutral studio background, consistent look")
            for a in _WARD_ANGLES[:max(1, n)]]


_SHEET_ANGLES = {"front": "front view, facing camera", "back": "back view",
                 "left": "left profile side view", "right": "right profile side view",
                 "three_quarter": "three-quarter view", "closeup": "face close-up portrait",
                 "full": "full-body shot", "detail": "detail close-up"}


def turnaround_prompt(subj_desc: str, angle_id: str) -> str:
    """Prompt per RIGENERARE UN SINGOLO angolo del character/asset sheet mantenendo l'angolo (curatela)."""
    a = _SHEET_ANGLES.get(angle_id, angle_id.replace("_", " "))
    return f"{(subj_desc or 'the subject').strip()}, {a}, neutral studio background, consistent design"


def expand_shot_to_coverage(shot_id: int) -> list[int]:
    """Espande uno shot-dialogo in N inquadrature di copertura (campo/controcampo). Il primo shot riusa
    l'originale; i successivi vengono inseriti dopo, spostando in avanti gli shot seguenti. Ritorna gli id."""
    sh = db.get("shot", shot_id)
    if not sh:
        return []
    dl = json.loads(sh.get("dialogues") or "[]")
    units = plan_coverage(dl, float(sh.get("duration_s") or 4))
    if len(units) < 2:
        return []                                            # 0-1 battute: niente da coprire
    scene_id, base = sh["scene_id"], sh["ord"]
    for s in db.query("shot", "scene_id=? AND ord>?", (scene_id, base)):   # fai spazio
        db.update("shot", s["id"], ord=s["ord"] + len(units) - 1)
    base_desc = sh.get("description") or ""
    cont = sh.get("continuity") or "{}"
    ids = []
    for i, u in enumerate(units):
        one = json.dumps([{"actor": "@" + u["speaker"] if u["speaker"] else "", "text": u["text"]}],
                         ensure_ascii=False)
        if i == 0:                                           # riusa lo shot originale
            db.update("shot", shot_id, ord=base, camera=u["camera"], duration_s=u["dur"], dialogues=one)
            ids.append(shot_id)
        else:
            ids.append(db.insert("shot", scene_id=scene_id, ord=base + i, description=base_desc,
                                 duration_s=u["dur"], camera=u["camera"], dialogues=one, keyframes="[]",
                                 refs="", continuity=cont, seed=0))
    return ids
