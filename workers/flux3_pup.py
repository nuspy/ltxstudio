"""Prompt upsampler (PUP) di FLUX.3 Image Klein con transformers — niente vLLM (non gira su Windows).

Il PUP è un Qwen3.5-9B (VLM) fine-tuned da BFL: trasforma una richiesta breve nel caption completo che il DiT vuole
(BFL: «do not use the model without it»). Quattro compiti, scelti dalla chiave del system prompt (`system_prompts.json`
che viaggia coi pesi): t2i · i2i (con riferimenti) · t2i_bbox · i2i_bbox (layout con riquadri bloccati; la risposta
si riconcilia con `core.flux3_caption.reconcile` così i riquadri dell'utente restano com'erano).

Pesi: cartella formato Hugging Face (config + model.safetensors bf16 — upsampler ufficiale o una variante) oppure GGUF
(Sakura abliterata: `flux3_gguf.read_state`, solo il modello di linguaggio; senza torre visiva i riferimenti non si
vedono e il compito resta t2i / i2i senza immagini — dichiarato nel log). Si carica PRIMA del DiT e si libera subito.
Campionamento: quello del codice ufficiale (temperatura 0 = greedy, presence_penalty 1,5 solo in t2i, max 3584 token).
"""
from __future__ import annotations

import gc
import io
import json
import time
from pathlib import Path


def _log(m: str) -> None:
    print(f"[flux3-pup] {m}", flush=True)


def policy_reference(image, n_refs: int):
    """Preprocessing ufficiale dei riferimenti per l'upsampler: RGB su bianco, LANCZOS al budget di pixel, JPEG q92."""
    from PIL import Image, ImageOps
    from core.flux3_caption import pup_reference_size
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    image = image.convert("RGB")
    size = pup_reference_size(image.width, image.height, n_refs)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


class _PresencePenalty:
    """presence_penalty come in vLLM: sottrae `penalty` ai logit dei token GIÀ GENERATI (non quelli del prompt)."""

    def __init__(self, penalty: float, prompt_len: int):
        self.penalty, self.n0 = float(penalty), int(prompt_len)

    def __call__(self, input_ids, scores):
        import torch
        gen = input_ids[:, self.n0:]
        if gen.shape[1] == 0 or self.penalty == 0.0:
            return scores
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, gen, True)
        return scores - self.penalty * mask.to(scores.dtype)


def load(weights: str, cfg_dir: str, want_vision: bool = True):
    """(modello, processor, system_prompts) — pesi senza mmap, un tensore alla volta, direttamente in GPU."""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    import _qlinear as Q
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(cfg_dir)
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(cfg, dtype=torch.bfloat16)
    if str(weights).lower().endswith(".gguf"):
        import flux3_gguf
        sd, vision = flux3_gguf.read_state(weights, "cuda"), False
    else:
        sd, vision = Q.read_safetensors(weights, "cuda"), True
    if not vision:                                         # solo linguaggio: la torre visiva non c'è nel file
        model.model.visual = torch.nn.Identity()
    miss, unex = Q.load_into(model, sd)
    miss = [m for m in miss if not m.endswith("inv_freq")]
    if miss or unex:
        raise RuntimeError(f"upsampler: chiavi mancanti {miss[:6]} / inattese {unex[:6]}")
    del sd
    model = model.to("cuda").eval()
    proc = AutoProcessor.from_pretrained(cfg_dir)
    prompts = json.loads((Path(cfg_dir) / "system_prompts.json").read_text("utf-8"))
    _log(f"pronto in {time.time() - t0:.0f}s — VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GB"
         + ("" if vision else " (solo linguaggio: niente lettura dei riferimenti)"))
    model._flux3_vision = vision
    return model, proc, prompts


def upsample(model, proc, prompts: dict, prompt: str, images, rows) -> str:
    """Caption completo (con tabella dei riquadri in coda se `rows`). Alza ValueError se un riferimento non è valido."""
    import torch
    from core import flux3_caption as C
    from core import flux3_spec as S
    n = len(images)
    if n and not getattr(model, "_flux3_vision", True):
        _log("ATTENZIONE: questo upsampler non vede le immagini: i riferimenti sono citati solo per ordinale")
        images, n = [], 0
    key, text = C.user_message(prompt, rows, n)
    system = prompts[key]
    refs = [policy_reference(im, n) for im in images]
    if refs:
        content = [{"type": "image", "image": im} for im in refs] + [{"type": "text", "text": text}]
    else:
        content = [{"type": "text", "text": text}]
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]
    inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                      return_tensors="pt", enable_thinking=False).to(model.device)
    n0 = int(inputs["input_ids"].shape[1])
    task = "i2i" if n else "t2i"
    from transformers import LogitsProcessorList
    lp = LogitsProcessorList([_PresencePenalty(S.PUP_PRESENCE_PENALTY[task], n0)])
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=S.PUP_MAX_NEW_TOKENS, do_sample=False, logits_processor=lp)
    new = out[0, n0:]
    caption = proc.tokenizer.decode(new, skip_special_tokens=True).strip()
    _log(f"{key}: {len(new)} token in {time.time() - t0:.0f}s")
    if not caption:
        raise RuntimeError("l'upsampler ha restituito una risposta vuota")
    return C.reconcile(caption, rows) if rows else caption


def run(weights: str, cfg_dir: str, prompt: str, images, rows) -> str:
    """Carica, riscrive, libera la VRAM. Se il compito fallisce con ValueError (es. riferimento con rapporto
    estremo) alza: il chiamante usa il prompt com'è e lo dice nel log."""
    import torch
    model, proc, prompts = load(weights, cfg_dir)
    try:
        return upsample(model, proc, prompts, prompt, images, rows)
    finally:
        del model, proc
        gc.collect()
        torch.cuda.empty_cache()
