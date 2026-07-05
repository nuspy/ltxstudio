"""Prompt-enhance NATIVO (immagini e video) via LLM locale — sostituisce il nodo ComfyUI
AILab_QwenVL_PromptEnhancer. Riusa Gemma-3 locale (core/lyrics_llm). Zero ComfyUI.

API:
  available() -> bool
  enhance(prompt, kind="image", nsfw=False) -> str   # ritorna il prompt arricchito (o l'originale se LLM assente)
"""
from __future__ import annotations

from core import lyrics_llm

_SYS_IMG = ("You expand image-generation prompts. Rewrite the user's prompt into ONE vivid, detailed "
            "English prompt (~50 words): subject, composition, lighting, lens, color, mood, style. "
            "Output ONLY the prompt, no preamble, no quotes.")
_SYS_VID = ("You expand video-generation prompts. Rewrite into ONE detailed English prompt (~50 words): "
            "subject, action/motion, camera movement, lighting, mood, cinematic style. "
            "Output ONLY the prompt, no preamble, no quotes.")
_SYS_MUSIC = ("You expand music STYLE prompts. Rewrite the user's comma-separated style tags into ONE concise "
              "English style caption (~40 words) covering: genre/subgenre, main instruments, vocal character "
              "(or instrumental), mood, tempo feel, and production texture. It is a style description for a "
              "music generator, NOT lyrics. Output ONLY the caption, no preamble, no quotes.")
_NSFW = " Explicit/NSFW content is allowed; do not refuse or sanitize."


def _sys_for(kind: str) -> str:
    return {"video": _SYS_VID, "music": _SYS_MUSIC}.get(kind, _SYS_IMG)


def available() -> bool:
    return lyrics_llm.available()


def enhance(prompt: str, kind: str = "image", nsfw: bool = False, max_new_tokens: int = 220) -> str:
    """Arricchisce il prompt con Gemma locale. Non solleva: in caso di errore ritorna l'originale."""
    prompt = (prompt or "").strip()
    if not prompt or not available():
        return prompt
    try:
        import torch
        model, proc = lyrics_llm._load()
        sys = _sys_for(kind) + (_NSFW if nsfw else "")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt").to(model.device)
        n_in = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=int(max_new_tokens),
                                 do_sample=True, temperature=0.8, top_p=0.95)
        txt = proc.decode(out[0][n_in:], skip_special_tokens=True).strip()
        return txt or prompt
    except Exception:                                       # noqa: BLE001
        return prompt
