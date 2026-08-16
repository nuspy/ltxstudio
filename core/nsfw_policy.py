"""Politica NSFW CENTRALIZZATA per i contenuti/modelli (FASE D weights-unification).

Prima esistevano QUATTRO euristiche keyword diverse in quattro file (video_tab, lora_catalog,
lora_tab, frame1 via catalogo): stessi modelli classificati in modo diverso per tab. Qui: UNA
lista keyword + la precedenza corretta (flag di catalogo/override utente VINCE sull'euristica)
+ il gate d'edizione (cap.adult_mode) in un posto solo.
"""
from __future__ import annotations

# unione delle liste storiche (video_tab, lora_catalog._NSFW_HINTS, lora_tab) + sensitivity v2
KEYWORDS = ("nsfw", "eros", "xxx", "adult", "porn", "porno", "hentai", "nude", "nudity",
            "naked", "erotic", "uncensored", "abliterated", "explicit", "sex", "lewd",
            "deepthroat", "penile", "pussy", "anal", "blowjob")


def looks_nsfw(*parts: object) -> bool:
    """Euristica keyword sull'unione dei testi (nome file, path, tag…)."""
    text = " ".join(str(p or "") for p in parts).lower()
    return any(k in text for k in KEYWORDS)


def is_nsfw(name: str, *, catalog_flag: bool | None = None, extra: object = "") -> bool:
    """Verdetto per un modello: il flag esplicito (catalogo/override utente) VINCE quando è
    presente; altrimenti euristica keyword su nome+extra."""
    if catalog_flag is not None:
        return bool(catalog_flag) or looks_nsfw(name, extra)
    return looks_nsfw(name, extra)


def adult_allowed() -> bool:
    """Il gate d'edizione (cap.adult_mode) — unico punto interrogato dalla GUI."""
    try:
        from core import features
        return bool(features.is_enabled("cap.adult_mode") and features.allows_adult())
    except Exception:                                    # noqa: BLE001
        return True                                      # dev fail-open (come features)
