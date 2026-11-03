"""Prompt-enhance NATIVO (immagini e video) via LLM locale — sostituisce il nodo ComfyUI
AILab_QwenVL_PromptEnhancer. Riusa Gemma-3 locale (core/lyrics_llm). Zero ComfyUI.

API:
  available() -> bool
  enhance(prompt, kind="image", nsfw=False) -> str   # ritorna il prompt arricchito (o l'originale se LLM assente)
"""
from __future__ import annotations

import os

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
_SYS_H3 = ("You receive ONE JSON object whose values are short texts written by a user in any language. Reply with ONE JSON object on a single line with EXACTLY the same keys "
           "and nothing else (no markdown, no preamble). Per key: 'subject' is a segmentation label for a person: translate it FAITHFULLY into short English (1-6 words), "
           "never add details or adjectives that are not there ('persona' -> 'person', 'la ragazza in rosso' -> 'the girl in red'). 'setting', 'picture_desc' and 'shots' "
           "are descriptions: translate into English and enrich with concrete visual detail (30-50 words, one or two sentences: place, materials, lighting, atmosphere), "
           "keeping every fact the user gave and inventing no people. Escape quotes so the JSON is valid.")
_NSFW = " Explicit/NSFW content is allowed; do not refuse or sanitize."


def _sys_for(kind: str) -> str:
    return {"video": _SYS_VID, "music": _SYS_MUSIC, "h3_fields": _SYS_H3}.get(kind, _SYS_IMG)


def available() -> bool:
    return lyrics_llm.available()


def enhance(prompt: str, kind: str = "image", nsfw: bool = False, max_new_tokens: int = 220) -> str:
    """Arricchisce il prompt con Gemma locale. Non solleva: in caso di errore ritorna l'originale.

    ⚠ SUBPROCESS: Gemma-3 12B (~23GB) NON deve entrare nel processo GUI — caricarla in-process
    (transformers → safetensors mmap, NON coperto dal patch no-mmap della GUI) causava ACCESS
    VIOLATION in torch_cpu.dll = crash secco dell'app al click di Genera inpaint/integra con
    «Migliora prompt» attivo (visto live 22/07, Event Viewer 20:17 e 20:32). Qui si spawna lo
    STESSO interprete in un processo separato (LS_ENHANCE_INPROC=1 → ramo in-process sotto):
    se il figlio muore, si ritorna il prompt originale e l'app resta viva."""
    prompt = (prompt or "").strip()
    if not prompt or not available():
        return prompt
    if os.environ.get("LS_ENHANCE_INPROC") != "1":
        import json as _json
        import subprocess
        import sys
        # COORDINAMENTO CONDIZIONALE (mancava): Gemma-3 12B (~23GB, device_map='auto') si
        # auto-offloada su CPU se la VRAM è piena → enhance lentissimo. Se la VRAM libera non
        # basta si DEMOTANO gli altri motori (parcheggio RAM: rientrano subito dopo — mai destroy,
        # il motore che genera dopo l'enhance deve restare caldo) e si aspetta il rilascio.
        try:
            from core import gpu_mem, mem_coordinator
            if gpu_mem.free_vram_mb() < 12000:
                mem_coordinator.free_for("llm", timeout=15)
                gpu_mem.wait_free_vram_mb(12000, timeout_s=20, tag="enhance")
        except Exception:                                   # noqa: BLE001
            pass
        _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, LS_ENHANCE_INPROC="1", HF_HUB_OFFLINE="1")
        try:
            r = subprocess.run([sys.executable, "-m", "core.prompt_enhance"],
                               input=_json.dumps({"prompt": prompt, "kind": kind, "nsfw": bool(nsfw),
                                                  "max_new_tokens": int(max_new_tokens)}),
                               capture_output=True, text=True, encoding="utf-8",
                               cwd=_repo, env=env, timeout=420)
            for line in reversed((r.stdout or "").splitlines()):
                if line.startswith("ENHANCED:"):
                    out = line[len("ENHANCED:"):].strip()
                    return out or prompt
        except Exception:                                   # noqa: BLE001 — enhance è best-effort
            pass
        return prompt
    return _enhance_inproc(prompt, kind, nsfw, max_new_tokens)


def _enhance_inproc(prompt: str, kind: str, nsfw: bool, max_new_tokens: int) -> str:
    """Implementazione reale (SOLO nel subprocess dedicato: mai nel processo GUI)."""
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
            if kind == "h3_fields":                         # traduzione fedele: deterministica (niente campionamento)
                out = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
            else:
                out = model.generate(**inputs, max_new_tokens=int(max_new_tokens),
                                     do_sample=True, temperature=0.8, top_p=0.95)
        txt = proc.decode(out[0][n_in:], skip_special_tokens=True).strip()
        if kind == "h3_fields":
            txt = " ".join(txt.split())                      # una sola riga: il genitore legge la riga ENHANCED:
        return txt or prompt
    except Exception:                                       # noqa: BLE001
        return prompt


if __name__ == "__main__":                                  # child del subprocess di enhance()
    import json as _json
    import sys as _sys
    try:                                                    # commit-safe anche nel figlio
        from core import st_direct
        st_direct.patch_safetensors_torch()
        st_direct.patch_safe_open()
    except Exception:                                       # noqa: BLE001
        pass
    _req = _json.loads(_sys.stdin.read() or "{}")
    _out = _enhance_inproc(str(_req.get("prompt", "")), str(_req.get("kind", "image")),
                           bool(_req.get("nsfw")), int(_req.get("max_new_tokens", 220)))
    print("ENHANCED:" + _out, flush=True)
