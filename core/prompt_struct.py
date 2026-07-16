"""Strutturazione semantica dei prompt di generazione in JSON (schema PromptBuilder).

Riusato sia dalla UI (`ui/widgets/prompt_builder.py`) sia dalla pipeline di generazione
(`core/film_studio/gen_service.py`), così che OGNI prompt di generazione (foto/video) sia rappresentato
nello STESSO schema JSON → riproducibilità, editabilità, coerenza tra Cinema Studio e Lucky Director.

Schema (identico ai campi di PromptBuilder):
    subject, action, setting, style, camera, light, mood, details, negative

- `compose_fields(d)`  → funzione PURA che ricompone (prompt, negative) dallo schema. È l'UNICA
  implementazione della composizione: sia il widget UI sia la pipeline la chiamano.
- `structure(text)`    → usa l'LLM locale (core/llm_service) per mappare testo libero → schema JSON.
- `structure_and_compose(text, ...)` → struttura, ricompone, e RI-APPENDE i token critici (trigger LoRA,
  tag di continuity/stile) così la ricomposizione non perde nulla di essenziale.

Disattivabile con env `LTX_PROMPT_JSON=0` (default attivo). Cache in-memory per prompt identici (evita
ri-chiamate LLM costose nei loop di «Genera tutti» / Lucky Director).
"""
from __future__ import annotations

import os

FIELDS = ("subject", "action", "setting", "style", "camera", "light", "mood", "details", "negative")

_cache: dict[tuple[str, str], dict] = {}

_SYS = (
    "You convert a freeform image/video generation prompt into STRICT JSON describing a single shot. "
    "Use EXACTLY these keys: subject, action, setting, style, camera, light, mood, details, negative. "
    "Keep the user's original language. Do not invent content that is not implied by the prompt. "
    "Put an empty string for any field you cannot fill. 'negative' = things to avoid (may be empty). "
    "Output ONLY the JSON object, nothing else."
)


def enabled() -> bool:
    """True se la strutturazione JSON è attiva (default). Disattiva con LTX_PROMPT_JSON=0."""
    v = os.environ.get("LTX_PROMPT_JSON")
    if v is not None:
        return v.strip().lower() not in ("0", "false", "no", "off", "")
    return True


def compose_fields(d: dict) -> tuple[str, str]:
    """Ricompone (prompt, negative) in linguaggio naturale dallo schema. Condivisa UI ⇄ pipeline."""
    d = d or {}
    parts: list[str] = []
    for k in ("subject", "action", "setting"):
        t = str(d.get(k, "") or "").strip()
        if t:
            parts.append(t)
    style = str(d.get("style", "") or "").strip()
    if style:
        parts.append(f"{style} style")
    for k in ("camera", "light"):
        t = str(d.get(k, "") or "").strip()
        if t:
            parts.append(t)
    for k in ("mood", "details"):
        t = str(d.get(k, "") or "").strip()
        if t:
            parts.append(t)
    return ", ".join(parts), str(d.get("negative", "") or "").strip()


def empty() -> dict:
    return {k: "" for k in FIELDS}


def structure(text: str, *, kind: str = "image", nsfw: bool = False) -> dict:
    """Mappa il testo libero nello schema JSON via LLM locale. Robusto: mai solleva, fallback su subject=text."""
    text = (text or "").strip()
    if not text:
        return empty()
    key = (kind, text)
    if key in _cache:
        return dict(_cache[key])
    d = empty()
    try:
        from core import llm_service
        raw = llm_service.complete_json(_SYS, text, max_tokens=400)
        if isinstance(raw, dict):
            d = {k: str(raw.get(k, "") or "").strip() for k in FIELDS}
            if not any(d.values()):                          # LLM ha risposto vuoto → fallback
                d = {**empty(), "subject": text}
        else:
            d = {**empty(), "subject": text}
    except Exception:                                        # noqa: BLE001 — non deve mai bloccare la generazione
        d = {**empty(), "subject": text}
    _cache[key] = dict(d)
    return d


def structure_and_compose(text: str, *, kind: str = "image", nsfw: bool = False,
                          triggers: tuple | list = (), extra_suffix: str = "") -> tuple[str, str, dict]:
    """Struttura → ricompone → ri-appende trigger LoRA e suffissi critici. Ritorna (prompt, negative, json)."""
    d = structure(text, kind=kind, nsfw=nsfw)
    prompt, negative = compose_fields(d)
    tail = [str(t).strip() for t in (triggers or []) if str(t).strip() and str(t).strip() not in prompt]
    es = (extra_suffix or "").strip()
    if es and es not in prompt:
        tail.append(es)
    if tail:
        prompt = (prompt + (", " if prompt else "") + ", ".join(tail)).strip(", ")
    if not prompt.strip():                                   # fallback totale: mai perdere il prompt originale
        prompt = text or ""
    return prompt, negative, d


def to_lingbot_json(text: str, *, nsfw: bool = False, first_frame: str = "") -> tuple[str, str]:
    """Composer JSON per LingBot-Video (#16): LingBot consuma caption JSON strutturate, non prosa.

    Riusa `structure()` (LLM locale) per riempire lo schema a 9 campi, poi lo serializza nel formato
    caption che il DiT LingBot si aspetta. Ritorna (json_prompt_str, negative). Fallback-safe.

    ⚠️ VALIDATE: lo schema esatto delle caption LingBot va confermato contro il pacchetto ufficiale.
    Qui produciamo un JSON descrittivo (caption + campi) accettato da un rewriter/loader tollerante; la
    versione proprietaria a 2 stadi vive in core/prompt_rewrite_lingbot."""
    import json as _json
    d = structure(text, kind="video", nsfw=nsfw)
    caption, negative = compose_fields(d)
    obj = {
        "caption": caption or (text or ""),
        "subject": d.get("subject", ""), "action": d.get("action", ""),
        "setting": d.get("setting", ""), "style": d.get("style", ""),
        "camera": d.get("camera", ""), "light": d.get("light", ""),
        "mood": d.get("mood", ""), "details": d.get("details", ""),
    }
    if first_frame:
        obj["has_first_frame"] = True
    return _json.dumps(obj, ensure_ascii=False), negative
