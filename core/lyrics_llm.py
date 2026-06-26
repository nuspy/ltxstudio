"""Generatore di testi-canzone via LLM LOCALE già nel progetto: Gemma-3 12B IT (transformers).

Da un breve prompt l'LLM scrive liriche cantabili con sezioni [Verse]/[Chorus]/[Bridge] nella
lingua scelta. Il prompt è ARRICCHITO automaticamente spiegando la finalità (paroliere, struttura,
cantabilità, stile/tag). Nessun servizio esterno (no ollama), nessuna nuova dipendenza:
usa `transformers` (già installato) + i pesi Gemma-3 già su disco (stesso modello del text-encoder LTX).

API:
  available() -> bool
  build_prompt(prompt, language, tags) -> str    # prompt arricchito (loggabile)
  generate(prompt, language=, tags=, max_new_tokens=) -> str
"""
from __future__ import annotations

import os

# repo HF Gemma-3 12B IT (config + chat_template + tokenizer + pesi). Fallback al TE LTX.
GEMMA_DIRS = [
    r"E:\AI\Models\TextEncoders\gemma-3-ltx",
    r"E:\AI\Models\TextEncoders\gemma3",
]

LANG_NAMES = {
    "it": "Italian", "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "pt": "Portuguese", "ru": "Russian", "uk": "Ukrainian", "zh": "Chinese", "ko": "Korean",
    "ar": "Arabic", "he": "Hebrew", "hi": "Hindi", "el": "Greek", "hu": "Hungarian",
}

_SYSTEM = (
    "You are a professional songwriter. Write original, singable song lyrics. "
    "Use clear section tags on their own lines: [Verse], [Chorus], [Bridge], [Outro]. "
    "Make the lines rhythmic and easy to sing, with natural rhymes. "
    "Output ONLY the lyrics with their section tags — no title, no explanations, no chords, no notes."
)

_CACHE: dict = {}


def _gemma_dir() -> str | None:
    for d in GEMMA_DIRS:
        if os.path.exists(os.path.join(d, "config.json")):
            return d
    return None


def available() -> bool:
    return _gemma_dir() is not None


def build_prompt(user_prompt: str, language: str = "it", tags: str = "") -> str:
    """Prompt utente ARRICCHITO con la finalità (mostrabile/loggabile)."""
    lang = LANG_NAMES.get(language, language or "Italian")
    parts = [f"Write the complete lyrics of a song in {lang}."]
    if (user_prompt or "").strip():
        parts.append(f"Theme / brief: {user_prompt.strip()}")
    if (tags or "").strip():
        parts.append(f"Musical style / mood (tags): {tags.strip()}")
    parts.append("Structure it with [Verse] and [Chorus] sections (add [Bridge]/[Outro] if it fits). "
                 "Keep the lyrics singable and rhythmic, suitable to be sung over music.")
    return "\n".join(parts)


def _load():
    if "model" in _CACHE:
        return _CACHE["model"], _CACHE["proc"]
    import torch
    from transformers import Gemma3ForConditionalGeneration, AutoProcessor
    d = _gemma_dir()
    if not d:
        raise RuntimeError("Gemma-3 non trovata in E:\\AI\\Models\\TextEncoders (gemma-3-ltx / gemma3).")
    proc = AutoProcessor.from_pretrained(d)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        d, torch_dtype=torch.bfloat16, device_map="auto")     # offload CPU automatico se la VRAM è poca
    model.eval()
    _CACHE["model"], _CACHE["proc"] = model, proc
    return model, proc


_SB_SYSTEM = (
    "You are a music-video art director. Given song lyrics in ANY language, design a sequence of "
    "cinematic visual SCENES for a music video, in the SAME order as the lyrics, coherent with the "
    "narrative and mood. Translate/understand the lyrics yourself. "
    "Output ONLY a JSON array (no prose, no markdown fences). Each item: "
    '{"prompt": "<vivid ENGLISH visual description ~30 words, cinematic, coherent with this part of the song>", '
    '"weight": <integer 1-10 = relative on-screen duration, roughly proportional to how many lyric lines '
    "this scene covers>}. Cover the WHOLE song in order. Use 6 to 12 scenes. "
    "Keep a consistent setting/characters across scenes (it is one continuous music video)."
)


def storyboard(lyrics: str, image_hint: str = "", target_scenes: int | None = None,
               video_style: str = "", max_new_tokens: int = 1600) -> list[dict]:
    """Gemma-3 locale → storyboard per il videomusicale: lista ORDINATA di scene {prompt, weight}
    coerenti col testo (l'LLM traduce/interpreta). Parametri:
      image_hint    = descrizione dell'immagine iniziale/sfondo da mantenere coerente.
      target_scenes = numero scene desiderato (più scene = cambi più rapidi); None = 6-12 a scelta LLM.
      video_style   = stile/resa del video (es. 'cartone animato divertente', 'bianco e nero malinconico',
                      'realistico romantico'): NON è contenuto, va applicato a TUTTE le scene.
    Ritorna [] se l'LLM non è disponibile/non produce JSON valido."""
    import json
    import re
    import torch
    model, proc = _load()
    user = "Song lyrics (keep scene order matching the song):\n" + (lyrics or "").strip()
    if (video_style or "").strip():
        user += (f"\n\nVISUAL STYLE (apply to EVERY scene, this is the look/mood NOT the content): "
                 f"{video_style.strip()}. Every scene description must reflect this style consistently.")
    if target_scenes:
        n = max(2, int(target_scenes))
        user += (f"\n\nProduce EXACTLY about {n} scenes (one every few seconds of the song) so the music "
                 "video has frequent, dynamic scene changes. Each scene visually DISTINCT "
                 "(different shot/angle/action) yet coherent with the same world, characters and style.")
    if (image_hint or "").strip():
        user += ("\n\nThe music video STARTS from and stays visually consistent with this background "
                 f"image: {image_hint.strip()}. Keep that setting/lighting coherent across scenes.")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": _SB_SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=int(max_new_tokens),
                             do_sample=True, temperature=0.8, top_p=0.95)
    text = proc.decode(out[0][n_in:], skip_special_tokens=True).strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)              # estrai l'array JSON anche se circondato da testo
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:                                       # noqa: BLE001
        return []
    scenes = []
    for it in arr:
        if isinstance(it, dict) and it.get("prompt"):
            scenes.append({"prompt": str(it["prompt"]).strip(),
                           "weight": max(1, int(it.get("weight", 1) or 1))})
    return scenes


def generate(user_prompt: str, language: str = "it", tags: str = "", max_new_tokens: int = 600) -> str:
    """Genera le liriche con Gemma-3 locale. Solleva RuntimeError se il modello non è disponibile."""
    import torch
    model, proc = _load()
    messages = [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": build_prompt(user_prompt, language, tags)}]},
    ]
    inputs = proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=int(max_new_tokens),
                             do_sample=True, temperature=0.9, top_p=0.95)
    text = proc.decode(out[0][n_in:], skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("L'LLM non ha prodotto testo.")
    return text
