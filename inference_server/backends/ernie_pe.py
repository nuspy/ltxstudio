"""Prompt enhancer ufficiale di ERNIE-Image (Ministral-3 3B causale, 6.4 GB bf16) — caricato SOLO se l'utente attiva
«Migliora prompt» e scaricato subito dopo aver riscritto il prompt (mai residente accanto al DiT).

Pesi: `<repo ERNIE-Image>/pe/model.safetensors` (config, tokenizer e chat template sono già in `pe/` e `pe_tokenizer/`).
Il file «ernie-image-prompt-enhancer.safetensors» (stile Comfy: `model.*`, senza lm_head, + blob `tekken_model` che si
ignora) va SPOSTATO lì dall'utente; il file ufficiale HF ha anche lm_head: entrambi caricano (tie_word_embeddings).
Stessa logica della pipeline diffusers (`ErnieImagePipeline._enhance_prompt_with_pe`), con lo shim transformers <5
(Ministral3 → Ministral, sliding_window pieno) usato anche da `_assemble_ernie`.
"""
from __future__ import annotations

import gc
import json
import os
from types import SimpleNamespace


def pe_weights(repo_dir: str) -> str:
    return os.path.join(repo_dir, "pe", "model.safetensors")


def available(repo_dir: str) -> bool:
    return os.path.isfile(pe_weights(repo_dir)) and os.path.isfile(os.path.join(repo_dir, "pe", "config.json")) \
        and os.path.isfile(os.path.join(repo_dir, "pe_tokenizer", "tokenizer.json"))


def load_pe(repo_dir: str, dtype=None, weights: str | None = None):
    """-> (modello su CPU, tokenizer). `weights` solo per i test (file in un'altra posizione)."""
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import PreTrainedTokenizerFast
    dtype = dtype or torch.bfloat16
    cfg = json.load(open(os.path.join(repo_dir, "pe", "config.json"), encoding="utf-8"))
    if hasattr(transformers, "Ministral3ForCausalLM"):
        Model = transformers.Ministral3ForCausalLM
        Cfg = transformers.Ministral3Config
    else:                                                       # transformers 4.x: ministral3 non esiste
        Model, Cfg = transformers.MinistralForCausalLM, transformers.MinistralConfig
        if cfg.get("model_type") == "ministral3":
            cfg["model_type"] = "ministral"
        if cfg.get("sliding_window") is None:
            cfg["sliding_window"] = cfg.get("max_position_embeddings", 262144)
    cfg.pop("architectures", None)
    with torch.device("meta"):
        model = Model(Cfg(**{k: v for k, v in cfg.items() if k not in ("model_type", "transformers_version")}))
    raw = load_file(weights or pe_weights(repo_dir))
    sd = {k: v.to(dtype) for k, v in raw.items() if not k.startswith("tekken_model")}
    del raw
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in missing if k != "lm_head.weight"]      # lm_head legato a embed_tokens (tie)
    if missing or unexpected:
        raise ValueError(f"PE: chiavi non combaciano — mancano {missing[:3]}, in più {unexpected[:3]}")
    model.tie_weights()
    if any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers()):
        # i buffer non persistenti (rotary inv_freq) nati su meta vanno ricreati
        model = _rebuild_nonpersistent(model, Model, Cfg, cfg, dtype, sd)
    tdir = os.path.join(repo_dir, "pe_tokenizer")
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tdir, "tokenizer.json"), bos_token="<s>",
                                  eos_token="</s>", pad_token="<pad>", unk_token="<unk>", model_max_length=2048)
    tok.model_input_names = ["input_ids", "attention_mask"]      # niente token_type_ids: generate() lo rifiuta
    with open(os.path.join(tdir, "chat_template.jinja"), encoding="utf-8") as f:
        tok.chat_template = f.read()
    return model.eval().requires_grad_(False), tok


def _rebuild_nonpersistent(model, Model, Cfg, cfg, dtype, sd):
    """Costruisce il modello su CPU (buffer reali) e ci carica i pesi: fallback se rimane qualcosa su meta."""
    import torch
    m = Model(Cfg(**{k: v for k, v in cfg.items() if k not in ("model_type", "transformers_version")})).to(dtype)
    m.load_state_dict(sd, strict=False)
    m.tie_weights()
    return m


def _generate(prompt, width, height, repo_dir, device, weights, log):
    import time
    import torch
    t0 = time.perf_counter()
    model, tok = load_pe(repo_dir, weights=weights)
    if device is None:
        device = "cpu"
        if torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 2**30
            if free >= 9.0:
                device = "cuda"
            else:
                log(f"[ernie-pe] VRAM libera {free:.1f} GB < 9: l'enhancer gira su CPU (lento)")
    model.to(device)
    from diffusers import ErnieImagePipeline
    out = ErnieImagePipeline._enhance_prompt_with_pe(SimpleNamespace(pe=model, pe_tokenizer=tok), prompt,
                                                     torch.device(device), width, height)
    log(f"[ernie-pe] prompt riscritto in {time.perf_counter() - t0:.1f}s su {device}: {out[:120]!r}...")
    return out


def enhance(prompt: str, width: int, height: int, repo_dir: str, *, device: str | None = None,
            weights: str | None = None, log=print) -> str:
    """Riscrive `prompt` col PE ufficiale e SCARICA il modello (le sue referenze muoiono con `_generate`).
    Ritorna il prompt originale se qualcosa fallisce (mai bloccare la generazione per l'enhancer). GPU se ci
    stanno ~9 GB liberi, altrimenti CPU (lento: avvisa)."""
    try:
        out = _generate(prompt, width, height, repo_dir, device, weights, log)
    except Exception as e:                                       # noqa: BLE001
        log(f"[ernie-pe] enhancer non riuscito ({type(e).__name__}: {e}) -> prompt originale")
        out = prompt
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                            # noqa: BLE001
        pass
    return out or prompt
