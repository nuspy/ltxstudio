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


def unload() -> None:
    """Libera Gemma dalla VRAM (da chiamare dopo lo storyboard, prima del video GGUF)."""
    import gc
    m = _CACHE.pop("model", None)
    _CACHE.pop("proc", None)
    del m
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:                                       # noqa: BLE001
        pass


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


def _cast_block(cast: list[dict] | None) -> str:
    """Descrizione testuale del cast per lo storyboard: nomi, ruoli, strumenti, frequenza di apparizione.
    Istruisce l'LLM a taggare le scene con @Nome (rispettando la frequenza) + assegnare una location."""
    cast = [c for c in (cast or []) if (c.get("name") or "").strip()]
    if not cast:
        return ""
    lines = []
    for c in cast:
        name = str(c["name"]).strip()
        role = (c.get("role") or "performer").strip()
        instr = (c.get("instrument") or "").strip()
        freq = int(c.get("frequency", 60) or 60)
        bit = f'  - @{name}: {role}'
        if instr:
            bit += f", plays {instr} (ALWAYS show the {instr} with them)"
        bit += f" — appears in ~{freq}% of scenes"
        lines.append(bit)
    return (
        "\n\nCAST (recurring people — reuse them CONSISTENTLY across scenes):\n" + "\n".join(lines) +
        "\n\nFor EACH scene ALSO output:\n"
        '  "chars": [ "@Name", ... ]  = which cast members appear in THIS scene (respect their % frequency; '
        "the singer appears most; some scenes can be b-roll/location with nobody),\n"
        '  "location": "<one short place, reuse a small set of recurring locations>".\n'
        "When a musician appears, their instrument MUST be described in the scene prompt. "
        "Keep each named person visually identical every time they appear."
    )


def storyboard(lyrics: str, image_hint: str = "", target_scenes: int | None = None,
               video_style: str = "", cast: list[dict] | None = None,
               max_new_tokens: int = 1600) -> list[dict]:
    """Gemma-3 locale → storyboard per il videomusicale: lista ORDINATA di scene {prompt, weight}
    coerenti col testo (l'LLM traduce/interpreta). Parametri:
      image_hint    = descrizione dell'immagine iniziale/sfondo da mantenere coerente.
      target_scenes = numero scene desiderato (più scene = cambi più rapidi); None = 6-12 a scelta LLM.
      video_style   = stile/resa del video (es. 'cartone animato divertente', 'bianco e nero malinconico',
                      'realistico romantico'): NON è contenuto, va applicato a TUTTE le scene.
      cast          = personaggi ricorrenti [{name, role, instrument, frequency}]; se dato, ogni scena
                      riceve anche `chars` (@Nome presenti) e `location`, per coerenza volto tra i clip.
    Ritorna [] se l'LLM non è disponibile/non produce JSON valido. Le scene includono `chars`/`location`
    solo quando c'è un cast."""
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
    user += _cast_block(cast)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": _SB_SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    # scala i token col numero scene richiesto (ogni scena ~70 tok) per non troncare l'array JSON
    mnt = int(max_new_tokens)
    if target_scenes:
        mnt = min(4096, max(mnt, 500 + 80 * int(target_scenes)))
    inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=mnt, do_sample=True, temperature=0.8, top_p=0.95)
    text = proc.decode(out[0][n_in:], skip_special_tokens=True).strip()
    scenes = []
    m = re.search(r"\[.*\]", text, re.DOTALL)              # array completo se ben chiuso
    arr = None
    if m:
        try:
            arr = json.loads(m.group(0))
        except Exception:                                   # noqa: BLE001
            arr = None
    if arr is None:                                         # PARSING ROBUSTO: array troncato → estrai i singoli oggetti
        arr = []
        for obj in re.findall(r'\{[^{}]*?"prompt"\s*:\s*".*?"[^{}]*?\}', text, re.DOTALL):
            try:
                arr.append(json.loads(obj))
            except Exception:                               # noqa: BLE001
                pass
    for it in arr:
        if isinstance(it, dict) and it.get("prompt"):
            sc = {"prompt": str(it["prompt"]).strip(),
                  "weight": max(1, int(it.get("weight", 1) or 1))}
            ch = it.get("chars")
            if isinstance(ch, list):                        # @Nome presenti nella scena (solo con cast)
                sc["chars"] = [str(x).lstrip("@").strip() for x in ch if str(x).strip()]
            elif isinstance(ch, str) and ch.strip():
                sc["chars"] = [x.lstrip("@").strip() for x in re.split(r"[,\s]+", ch) if x.strip()]
            if isinstance(it.get("location"), str) and it["location"].strip():
                sc["location"] = it["location"].strip()
            scenes.append(sc)
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
