"""Continuity Engine (spec §12): la coerenza come STATO esplicito, non effetto del prompt.

Continuity State per scena/shot: orario, meteo, luce, color grade/mood + per ogni asset presente
(variazione attiva, abbigliamento, condizione, emozione). Propagazione in avanti, iniezione nei
prompt, warning incoerenze. Stato serializzato come JSON nelle colonne `continuity` di scene/shot.
"""
from __future__ import annotations

import copy

DEFAULT_STATE = {
    "time_of_day": "day", "weather": "clear", "light": "natural soft",
    "color_grade": "neutral", "mood": "neutral",
    "lens": "", "film_look": "",           # continuità fotografica: lente/ottica + resa pellicola/LUT (persistono)
    "assets": {},                          # name -> {variation, wardrobe, condition, emotion}
}


def new_state(**overrides) -> dict:
    s = copy.deepcopy(DEFAULT_STATE)
    s.update(overrides)
    return s


def propagate(prev: dict | None, scene_overrides: dict | None = None) -> dict:
    """Stato di una scena/shot = stato precedente + override locali (le condizioni asset persistono)."""
    s = copy.deepcopy(prev) if prev else new_state()
    for k, v in (scene_overrides or {}).items():
        if k == "assets" and isinstance(v, dict):
            for name, st in v.items():
                s["assets"].setdefault(name, {}).update(st or {})
        else:
            s[k] = v
    return s


def to_prompt(state: dict) -> str:
    """Frammento di prompt che esprime lo stato (iniettato in ogni gen immagine/video)."""
    if not state:
        return ""
    parts = [f"{state.get('time_of_day','day')}", f"{state.get('weather','clear')} weather",
             f"{state.get('light','natural')} lighting", f"{state.get('color_grade','neutral')} color grade",
             f"{state.get('mood','neutral')} mood"]
    if state.get("lens"):
        parts.append(f"shot on {state['lens']} lens")        # continuità ottica (persiste tra scene)
    if state.get("film_look"):
        parts.append(state["film_look"])                     # resa pellicola/LUT
    for name, st in (state.get("assets") or {}).items():
        bits = [b for b in (st.get("condition"), st.get("emotion"), st.get("wardrobe")) if b]
        if bits:
            parts.append(f"{name}: " + ", ".join(bits))
    return ", ".join(p for p in parts if p)


def warnings(prev: dict | None, cur: dict) -> list[str]:
    """Segnala salti sospetti tra stato precedente e corrente (luce/condizione che torna indietro…)."""
    out = []
    if prev:
        if prev.get("time_of_day") != cur.get("time_of_day"):
            out.append(f"orario cambia: {prev.get('time_of_day')} → {cur.get('time_of_day')}")
        for name, st in (prev.get("assets") or {}).items():
            cs = (cur.get("assets") or {}).get(name, {})
            if st.get("condition") and not cs.get("condition"):
                out.append(f"{name}: condizione '{st['condition']}' persa senza motivo")
    return out
