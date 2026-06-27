"""Voice Studio (spec §8): estende la voce esistente con (a) voce da DESCRIZIONE e (b) binding
voce↔attore + sintesi dialoghi per shot. Riusa core/tts_engine (Edge preset + CosyVoice clone) e il
DB Film Studio. Zero ComfyUI.

- voice_from_description: l'LLM mappa la descrizione → {gender, age, language, style} → sceglie il
  miglior preset Edge corrispondente; salva un VoicePreset riutilizzabile (come una voce clonata).
- bind/voice_for_actor: lega una voce a un Attore (DB voice.actor_id).
- synth_dialogue: genera l'audio di una battuta con la voce legata (Edge o clone CosyVoice).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from core import llm_service as L, tts_engine as T
from core.film_studio import db

_LANG = {"it": "it-IT", "en": "en-US", "es": "es-ES", "fr": "fr-FR", "de": "de-DE",
         "pt": "pt-BR", "ru": "ru-RU", "zh": "zh-CN", "ja": "ja-JP", "ar": "ar-EG"}

_SYS = ("Map a voice description to JSON: {\"gender\":\"male|female\",\"age\":\"child|young|adult|old\","
        "\"language\":\"it|en|es|fr|de|pt|ru|zh|ar\",\"style\":\"short adjectives\"}. "
        "Infer from the description. Output ONLY the JSON object.")


def voice_from_description(description: str, name: str = "", *, actor_id: int = 0) -> dict:
    """Crea una voce da descrizione testuale. Ritorna {voice_id(db), preset_slug, attrs}."""
    try:
        attrs = L.complete_json(_SYS, description, max_tokens=200)
    except Exception:                                       # noqa: BLE001
        attrs = {"gender": "female", "age": "adult", "language": "it", "style": ""}
    lang = _LANG.get(str(attrs.get("language", "it")).lower(), "it-IT")
    want_gender = str(attrs.get("gender", "")).lower()
    # scegli il miglior preset Edge per lingua+genere
    presets = T.edge_list_voices(lang.split("-")[0]) if T.edge_available() else []
    chosen = None
    for p in presets:
        disp = (p.name or "").lower()
        if want_gender and (("female" in disp) == (want_gender == "female")):
            chosen = p; break
    chosen = chosen or (presets[0] if presets else None)
    vid_str = chosen.voice_id if chosen else ""
    disp_name = (name or "").strip() or f"voce_{want_gender or 'auto'}_{lang}"
    vp = T.VoicePreset(name=disp_name, engine="edge", voice_id=vid_str, language=lang,
                       note=f"da descrizione: {attrs.get('style','')}".strip())
    T.save_voice(vp)
    db_id = db.insert("voice", origin="described", name=disp_name,
                      params=json.dumps({**attrs, "engine": "edge", "voice_id": vid_str}, ensure_ascii=False),
                      sample="", actor_id=actor_id or 0)
    return {"voice_id": db_id, "preset_slug": vp.slug(), "attrs": attrs, "edge_voice": vid_str}


# mappe parametri → prosodia Edge (rate/pitch/volume)
_RATE = {"molto_lento": "-30%", "lento": "-15%", "normale": "+0%", "veloce": "+15%", "molto_veloce": "+30%"}
_PITCH = {"molto_basso": "-50Hz", "basso": "-25Hz", "medio": "+0Hz", "alto": "+25Hz", "molto_alto": "+50Hz"}
_AGE_PITCH = {"child": "+40Hz", "young": "+15Hz", "adult": "+0Hz", "old": "-15Hz"}
EMOTIONS = ["neutral", "happy", "sad", "angry", "whisper", "calm"]


def voice_from_params(name: str, *, gender: str = "female", age: str = "adult", language: str = "it",
                      rate: str = "normale", pitch: str = "medio", emotion: str = "neutral",
                      actor_id: int = 0) -> dict:
    """Crea una voce dai PARAMETRI espliciti (spec §8.2): genere/età/lingua/ritmo/tono/emozione.
    Sceglie il preset Edge per lingua+genere e salva la prosodia (rate/pitch/volume) nel profilo."""
    from core import tts_engine as T
    lang = _LANG.get(str(language).lower(), "it-IT")
    presets = T.edge_list_voices(lang.split("-")[0]) if T.edge_available() else []
    chosen = None
    for p in presets:
        disp = (p.name or "").lower()
        if ("female" in disp) == (str(gender).lower() == "female"):
            chosen = p; break
    chosen = chosen or (presets[0] if presets else None)
    vid_str = chosen.voice_id if chosen else ""
    # prosodia: combina ritmo/tono scelti + offset per età + preset emozione
    emo = T._EMOTION_PRESETS.get(emotion, {})
    pr = {"rate": _RATE.get(rate, "+0%"), "pitch": _PITCH.get(pitch, "+0Hz"), "volume": emo.get("volume", "+0%")}
    if emotion in T._EMOTION_PRESETS:
        pr["rate"] = emo.get("rate", pr["rate"])
    # somma offset età al pitch (in Hz)
    def _hz(s):
        try: return int(s.replace("Hz", "").replace("+", ""))
        except Exception: return 0
    pr["pitch"] = f"{_hz(pr['pitch']) + _hz(_AGE_PITCH.get(age, '+0Hz')):+d}Hz"
    disp_name = (name or "").strip() or f"voce_{gender}_{age}_{lang}"
    vp = T.VoicePreset(name=disp_name, engine="edge", voice_id=vid_str, language=lang,
                       note=f"params: {gender},{age},{rate},{pitch},{emotion}")
    T.save_voice(vp)
    db_id = db.insert("voice", origin="params", name=disp_name,
                      params=json.dumps({"engine": "edge", "voice_id": vid_str, "gender": gender, "age": age,
                                         "emotion": emotion, "prosody": pr}, ensure_ascii=False),
                      sample="", actor_id=actor_id or 0)
    return {"voice_id": db_id, "preset_slug": vp.slug(), "edge_voice": vid_str, "prosody": pr}


def register_clone(name: str, sample_wav: str, *, transcript: str = "", actor_id: int = 0,
                   language: str = "it-IT") -> dict:
    """Registra una voce CLONATA (CosyVoice) da un campione reale, legabile a un attore."""
    vp = T.VoicePreset(name=name, engine="cosyvoice2", voice_id=sample_wav, language=language,
                       sample_path=sample_wav, sample_transcript=transcript)
    T.save_voice(vp, copy_sample_from=sample_wav)
    db_id = db.insert("voice", origin="cloned", name=name,
                      params=json.dumps({"engine": "cosyvoice2", "language": language}, ensure_ascii=False),
                      sample=sample_wav, actor_id=actor_id or 0)
    return {"voice_id": db_id, "preset_slug": vp.slug()}


def bind(voice_id: int, actor_id: int) -> None:
    db.update("voice", voice_id, actor_id=actor_id)


def voice_for_actor(actor_id: int) -> dict | None:
    rows = db.query("voice", "actor_id=?", (actor_id,), order="id DESC")
    return rows[0] if rows else None


def synth_dialogue(text: str, voice_row: dict, out_wav: str, *, engine: str = "") -> str:
    """Sintetizza una battuta. Il MOTORE è scelto dall'utente (param `engine`, o quello salvato nella
    voce): 'kokoro'/'handy_clone' = Handy-API; 'edge'/'cosyvoice2'/... = motore predisposto. Nessun
    automatismo nascosto. `engine` esplicito (dal selettore GUI al momento del Genera) ha precedenza."""
    params = json.loads(voice_row.get("params") or "{}")
    engine = (engine or params.get("engine") or "edge").strip()
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    if engine in ("kokoro", "handy_clone"):                 # Handy-API (scelto esplicitamente)
        from core import handy_api
        if engine == "handy_clone" or voice_row.get("origin") == "cloned":
            r = handy_api.tts(text, voice=voice_row.get("name", ""), out_wav=out_wav, model="chatterbox-multilingual")
        else:
            r = handy_api.tts(text, voice=params.get("voice_id") or "if_sara",
                              lang=params.get("language", "it")[:2], out_wav=out_wav)
        if r and Path(out_wav).exists():
            return out_wav
        raise RuntimeError("Handy-API non disponibile (:8720).")
    if engine == "edge":
        pr = params.get("prosody") or {}
        T.edge_synthesize(text, params.get("voice_id", ""), Path(out_wav),
                          rate=pr.get("rate", "+0%"), volume=pr.get("volume", "+0%"),
                          pitch=pr.get("pitch", "+0Hz"))
    else:                                                   # clone CosyVoice via job
        job = __import__("core.cosyvoice_service", fromlist=["make_job"]).make_job(
            text, voice_row.get("sample", ""), out_name=Path(out_wav).stem)
        job.run() if hasattr(job, "run") else None
    return out_wav
