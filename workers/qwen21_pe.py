"""Arricchitori di prompt UFFICIALI di Qwen-Image 2.1 (Qwen/Qwen-Image-2.1-PE-T2I e -PE-I2I).

Due Qwen3.5-VL 9B fine-tunati: T2I trasforma un brief (qualunque lingua) in un lungo prompt inglese che descrive
l'immagine finita; I2I guarda le immagini e trasforma un'istruzione vaga in un'istruzione di editing precisa.
In modalità approfondita ragionano in un blocco <think> (2-4k token), in quella rapida no; poi rispondono con
un JSON {"rewritten_prompt", "wh_ratio"[, "ratio_follow"]}.
Il formato richiesto dal canvas vince sempre: wh_ratio/ratio_follow vengono solo registrati nel log.

Pesi: INT8 ConvRot di Comfy-Org convertiti in nativo (core/native_convert «qwen21_pe»), config/tokenizer/system
prompt dal repo ufficiale. Si carica PRIMA del DiT e si libera subito dopo: con VRAM a sufficienza i Linear INT8
vengono materializzati una volta in bf16 (de-ruotati), perché la generazione token per token con la dequantizzazione
a ogni passo sarebbe ~3 volte più lenta.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def _log(m: str) -> None:
    print(f"[qwen21-pe] {m}", flush=True)


def _materialize_bf16(model) -> int:
    """RowInt8Linear → nn.Linear bf16 (peso dequantizzato e de-ruotato una volta sola)."""
    import torch
    import torch.nn as nn
    import _qlinear as Q
    from core.native_convert import rotate
    n = 0
    for name, m in list(model.named_modules()):
        if not isinstance(m, Q.RowInt8Linear):
            continue
        w = (m.weight.to(torch.bfloat16) - m.zero_point.to(torch.bfloat16)) * m.scale.to(torch.bfloat16)
        if m.convrot:                            # peso salvato come W·H: (W·H)·H = W (Hadamard normalizzata)
            w = rotate(w, m.convrot)
        lin = nn.Linear(m.in_features, m.out_features, bias=m.bias is not None, device=w.device,
                        dtype=torch.bfloat16)
        with torch.no_grad():
            lin.weight.copy_(w)
            if m.bias is not None:
                lin.bias.copy_(m.bias.to(torch.bfloat16))
        Q._set_module(model, name, lin)
        del w
        n += 1
    torch.cuda.empty_cache()
    return n


def load(weights: str, cfg_dir: str):
    """Modello + processor dal file nativo e dalla cartella di config del repo ufficiale."""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    import _qlinear as Q
    t0 = time.time()
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(cfg_dir), dtype=torch.bfloat16)
    sd = Q.quantize_from_state_dict(model, Q.read_safetensors(weights, "cuda"))
    miss, unex = Q.load_into(model, sd)
    miss = [m for m in miss if not m.endswith("inv_freq")]
    if miss or unex:
        raise RuntimeError(f"arricchitore prompt: chiavi mancanti {miss[:6]} / inattese {unex[:6]}")
    del sd
    model = model.to("cuda").eval()                      # buffer non persistenti (RoPE) restano in CPU altrimenti
    free = torch.cuda.mem_get_info()[0] / 2**30
    if free > 14:                                        # spazio per i pesi bf16 (~+9 GB) con margine
        _log(f"materializzo in bf16 ({_materialize_bf16(model)} Linear, {free:.0f} GB liberi)")
    proc = AutoProcessor.from_pretrained(cfg_dir)
    _log(f"pronto in {time.time() - t0:.0f}s — VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GB")
    return model, proc


def rewrite(model, proc, cfg_dir: str, prompt: str, images=(), *, max_new_tokens: int = 8192,
            seed: int = 0, think: bool = False) -> tuple[str, dict]:
    """Prompt riscritto (o l'originale se il modello non chiude il JSON) + il JSON completo per il log."""
    import torch
    system = (Path(cfg_dir) / "system_prompt.txt").read_text("utf-8").strip()
    user = [{"type": "image", "image": im.convert("RGB")} for im in images] + [{"type": "text", "text": prompt}]
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": user}]
    inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                      return_tensors="pt", enable_thinking=bool(think)).to(model.device)
    torch.manual_seed(seed)
    t0 = time.time()
    with torch.no_grad():
        # con ragionamento: campionamento del README ufficiale; senza: quello consigliato da Qwen per il
        # modo non-thinking (misurato: 580 token / 40 s contro 2-4k token / 2-4 min, JSON sempre valido)
        sampling = dict(temperature=1.0, top_p=0.95) if think else dict(temperature=0.7, top_p=0.8)
        out = model.generate(**inputs, max_new_tokens=int(max_new_tokens if think else 3000), do_sample=True,
                             top_k=20, **sampling)
    new = out[0, inputs["input_ids"].shape[1]:]
    text = proc.tokenizer.decode(new, skip_special_tokens=True)
    _log(f"{len(new)} token in {time.time() - t0:.0f}s")
    _thinking, sep, answer = text.partition("</think>")
    answer = answer if sep else text
    s, e = answer.find("{"), answer.rfind("}")
    try:
        data = json.loads(answer[s:e + 1]) if s >= 0 and e > s else {}
    except ValueError:
        data = {}
    new_prompt = str(data.get("rewritten_prompt") or "").strip()
    if not new_prompt:
        _log("risposta senza JSON valido (troncata?): tengo il prompt originale")
        return prompt, data
    return new_prompt, data


def enhance(kind: str, weights: str, cfg_dir: str, prompt: str, images=(), *, seed: int = 0,
            think: bool = False) -> str:
    """Carica, riscrive, libera la VRAM. `kind` = t2i | i2i; `think` = modalità approfondita (ragionamento)."""
    import gc
    import torch
    from core.qwen21_spec import PE_MAX_NEW_TOKENS
    model, proc = load(weights, cfg_dir)
    try:
        new, data = rewrite(model, proc, cfg_dir, prompt, images if kind == "i2i" else (),
                            max_new_tokens=PE_MAX_NEW_TOKENS, seed=seed, think=think)
    finally:
        del model, proc
        gc.collect()
        torch.cuda.empty_cache()
    extra = {k: v for k, v in data.items() if k != "rewritten_prompt" and v}
    _log(f"prompt riscritto ({kind}{', ' + json.dumps(extra, ensure_ascii=False) if extra else ''}): {new[:400]}")
    return new
