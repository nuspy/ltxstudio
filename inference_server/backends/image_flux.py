"""Handler immagini del backend nativo (diffusers): text-to-image, editing (Kontext), inpaint.

Dispatch per `arch`/`task`. Dove la pipeline diffusers supporta `from_single_file` si riusano
i pesi LOCALI ComfyUI (nessun download); altrimenti serve un repo diffusers (`from_pretrained`).

| arch     | pipeline             | single_file | input        |
|----------|----------------------|-------------|--------------|
| flux     | FluxPipeline         | sì          | —            |
| flux2    | Flux2Pipeline        | NO (repo)   | —            |
| zimage   | ZImagePipeline       | sì          | —            |
| ernie    | ErnieImagePipeline   | NO (repo)   | —            |
| kontext  | FluxKontextPipeline  | sì          | image        |
| inpaint  | FluxFillPipeline     | sì          | image + mask |
"""
from __future__ import annotations

import os
from pathlib import Path

# (nome classe diffusers, supporta from_single_file, richiede immagine input)
SPEC = {
    "flux":    ("FluxPipeline",        True,  False),
    "flux2":   ("Flux2Pipeline",       False, False),
    "zimage":  ("ZImagePipeline",      True,  False),
    "ernie":   ("ErnieImagePipeline",  False, False),
    "kontext": ("FluxKontextPipeline", True,  True),
    "inpaint": ("FluxFillPipeline",    True,  True),
    # #14 — FLUX.2 per editing/inpaint/outpaint/kontext (assemblati via _assemble_flux2; assemblaggio
    # locale = supports_sf irrilevante, gestito da arch.startswith('flux2') in _load). I nomi delle
    # classi diffusers vanno VALIDATI su GPU (diffusers evolve): qui le best-guess attuali. L'outpaint
    # riusa flux2_inpaint con maschera INVERTITA (core.mask_utils.invert_mask).
    # FLUX.2 editing per istruzione (no maschera) = Klein (Flux2KontextPipeline NON esiste in diffusers
    # 0.38; l'editing FLUX.2 è la famiglia Klein). API da validare su GPU.
    "flux2_kontext": ("Flux2KleinPipeline",        False, True),
    "flux2_inpaint": ("Flux2KleinInpaintPipeline", False, True),
}
# Stima VRAM realistica (tutto-su-GPU) per decidere GPU vs offload. FLUX.1 dev/Fill/Kontext bf16
# pesano ~24GB il solo transformer: con TE(T5)+VAE il totale sfora i 32GB → DEVE andare in
# enable_model_cpu_offload, altrimenti spilla nella shared-RAM di Windows (lentissimo, vedi outpaint).
VRAM_MB = {"flux": 24000, "flux2": 24000, "zimage": 12000, "ernie": 16000,
           "kontext": 26000, "inpaint": 26000, "flux2_kontext": 24000, "flux2_inpaint": 24000}
# archi che richiedono una maschera (inpaint-family)
_MASK_ARCHS = ("inpaint", "flux2_inpaint")

# Text encoder convertiti (pesi locali + config/tokenizer) e VAE locale, per i file solo-transformer
TE_DIR = r"E:\AI\Models\diffusers\_te"
AE_FLUX = r"E:\AI\Models\VAE\Flux\ae.safetensors"
# Ernie: repo diffusers (config/vae/tokenizer/scheduler) + transformer e text-encoder come single-file locali
ERNIE_REPO = r"E:\AI\Models\diffusers\Ernie\ERNIE-Image"
ERNIE_TRANSFORMER = r"E:\AI\Models\diffusion_models\Ernie\ernie-image.safetensors"
ERNIE_TE = r"E:\AI\Models\TextEncoders\Ernie\ministral-3-3b.safetensors"
FLUX2_DIR = r"E:\AI\Models\diffusers\flux2"          # TE Mistral HF + tokenizer + config (gated, scaricati)
FLUX2_VAE = r"E:\AI\Models\VAE\Flux\flux2-vae.safetensors"
# FLUX.2 KLEIN (edit/inpaint): encoder Qwen3 (NON Mistral). 4B ufficiale → hidden 2560 (jad 7680),
# miraclein NSFW 8B → hidden 4096 (jad 12288). Il dir giusto è scelto dal joint_attention_dim del
# transformer (vedi _assemble_flux2_klein). VAE = lo stesso flux2-vae.
KLEIN_QWEN_DIRS = {7680: os.path.join(TE_DIR, "qwen3_4b"), 12288: os.path.join(TE_DIR, "qwen3_8b")}


def _place(pipe, vram_mb: int) -> str:
    """Posizionamento VRAM: tieni il modello SU GPU se c'è spazio (resta caricato tra le generazioni
    → niente reload/offload ripetuto), altrimenti offload CPU. L'eviction quando serve spazio per un
    ALTRO modello è gestita dal ModelManager (make_room/unload LRU).

    NB Windows: forzare GPU con VRAM stretta innesca il *system memory fallback* del driver NVIDIA
    (RAM come VRAM → lentissimo). Quindi: GPU solo se free ≥ vram_mb·margine + floor di sistema;
    sotto quella soglia → enable_model_cpu_offload (portabile, schede piccole). Override:
    NATIVE_FORCE_OFFLOAD=1 (sempre offload), NATIVE_GPU_MARGIN (default 1.35), NATIVE_GPU_FLOOR_MB."""
    import os
    import torch
    if os.environ.get("NATIVE_FORCE_OFFLOAD") == "1" or not torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        return "offload"
    free_mb = torch.cuda.mem_get_info()[0] / 1024**2
    margin = float(os.environ.get("NATIVE_GPU_MARGIN", "1.35"))   # modello + attivazioni
    floor_mb = float(os.environ.get("NATIVE_GPU_FLOOR_MB", "1500"))  # lascia margine al sistema
    if free_mb >= vram_mb * margin + floor_mb:
        pipe.to("cuda")                                 # RESTA su GPU; la cache ModelManager lo riusa
        return "gpu"
    pipe.enable_model_cpu_offload()
    return "offload"


def _place_flux2_blockswap(pipe) -> str:
    """Placement memory-bounded per FLUX.2 (~32B): block-swap manuale (no accelerate, che segfaulta coi
    buffer float8). I Linear quantizzati (Fp8Linear/Nvfp4Linear) tengono i pesi su CPU e li spostano su
    GPU 1 layer alla volta nel forward; i moduli leggeri restano su GPU. VAE intero su GPU (piccolo).
    Forza pipe._execution_device='cuda' perché coi pesi fp8 su CPU il device della pipe sarebbe ambiguo."""
    import torch
    from inference_server.backends.flux2_transformer import enable_blockswap
    dev = torch.device("cuda")
    enable_blockswap(pipe.transformer, dev)
    te = getattr(pipe, "text_encoder", None)
    if te is not None:
        try:
            enable_blockswap(te, dev)
        except Exception:                                   # noqa: BLE001
            te.to(dev)
    try:
        pipe.vae.to(dev)
    except Exception:                                       # noqa: BLE001
        pass
    # coi pesi fp8 su CPU, DiffusionPipeline.device (→ _execution_device) può risolvere 'cpu' e rompere la
    # creazione di latenti/embed. Forziamo cuda sulla classe della pipe (l'app esegue FLUX.2 solo su GPU).
    try:
        pipe.__class__._execution_device = property(lambda self: dev)
    except Exception:                                       # noqa: BLE001
        pass
    return "blockswap"


def _set_attention(pipe) -> str:
    """Backend di attention configurabile con FALLBACK (per la distribuzione su GPU diverse).
    NATIVE_ATTENTION: auto|sage|sage3|sdpa. 'auto' prova Sage e ricade su SDPA (sempre disponibile)."""
    import os
    want = (os.environ.get("NATIVE_ATTENTION") or "auto").lower()
    if want in ("auto", "sage", "sage3"):
        try:
            pipe.transformer.set_attention_backend("sage")
            return "sage"
        except Exception:                               # noqa: BLE001
            if want in ("sage", "sage3"):
                print("[image] Sage non disponibile su questa GPU → SDPA", flush=True)
    return "sdpa"                                       # default torch (sempre presente)


def _apply_fbcache(pipe) -> bool:
    """First Block Cache (come WaveSpeed di ComfyUI): salta i blocchi DiT quando il residuo
    cambia poco → ~2x sui modelli multi-step. Env NATIVE_FBCACHE=0|1 (default 1),
    NATIVE_FBCACHE_THRESHOLD (default 0.12: più alto = più veloce, meno fedele)."""
    import os
    if os.environ.get("NATIVE_FBCACHE", "1") != "1":
        return False
    try:
        from diffusers import FirstBlockCacheConfig            # cache NATIVO diffusers (compatibile)
        thr = float(os.environ.get("NATIVE_FBCACHE_THRESHOLD", "0.12"))
        pipe.transformer.enable_cache(FirstBlockCacheConfig(threshold=thr))
        return True
    except Exception as e:                              # noqa: BLE001
        print(f"[image] FBCache non applicato: {e}", flush=True)
        return False


def _maybe_compile(pipe) -> bool:
    """torch.compile del transformer (+~1.5x). Warm-up lungo, ma il server è persistente.
    Env NATIVE_COMPILE=0|1 (default 0: opt-in, può fallire su alcuni checkpoint fp8)."""
    import os
    import torch
    if os.environ.get("NATIVE_COMPILE", "0") != "1":
        return False
    try:
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=False)
        return True
    except Exception as e:                              # noqa: BLE001
        print(f"[image] compile non applicato: {e}", flush=True)
        return False


def _assemble_flux(Pipe, transformer_path: str, arch: str = "flux"):
    """Assembla una pipeline FLUX-family da: transformer LOCALE (single_file) + VAE LOCALE
    (ae.safetensors) + text encoder convertiti (CLIP-L, T5-xxl). Per i file 'solo transformer'
    (diffusion_models) che non contengono i text encoder."""
    import os
    import torch
    from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    bf16 = torch.bfloat16
    # config transformer LOCALE (no dipendenza HF a runtime → funziona offline/gated; target 100% locale).
    # FLUX Fill (inpaint/outpaint) ha in_channels=384 (latente+immagine mascherata concatenati): usa il
    # config dedicato 'flux_fill_transformer', altrimenti x_embedder mismatch (64 vs 384).
    _cfg_name = "flux_fill_transformer" if arch == "inpaint" else "flux_transformer"
    _tr_cfg = os.path.join(TE_DIR, _cfg_name)
    if os.path.isdir(_tr_cfg):
        tr = FluxTransformer2DModel.from_single_file(
            transformer_path, config=_tr_cfg, torch_dtype=bf16, local_files_only=True)
    else:
        tr = FluxTransformer2DModel.from_single_file(transformer_path, torch_dtype=bf16)
    # config esplicito (VAE FLUX = 16 canali latenti; senza, from_single_file inferisce SD-VAE a 8)
    vae = AutoencoderKL.from_single_file(AE_FLUX, config=os.path.join(TE_DIR, "flux_vae"), torch_dtype=bf16)
    te = CLIPTextModel.from_pretrained(os.path.join(TE_DIR, "clip_l"), torch_dtype=bf16)
    tok = CLIPTokenizer.from_pretrained(os.path.join(TE_DIR, "clip_l"))
    te2 = T5EncoderModel.from_pretrained(os.path.join(TE_DIR, "t5xxl"), torch_dtype=bf16)
    tok2 = T5TokenizerFast.from_pretrained(os.path.join(TE_DIR, "t5xxl"))
    sched = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True,
        base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096)
    return Pipe(transformer=tr, vae=vae, text_encoder=te, tokenizer=tok,
                text_encoder_2=te2, tokenizer_2=tok2, scheduler=sched)


def _assemble_zimage(transformer_path: str):
    """Assembla ZImagePipeline da: transformer LOCALE + VAE FLUX LOCALE + Qwen3 convertito."""
    import os
    import torch
    from diffusers import (ZImageTransformer2DModel, AutoencoderKL,
                           FlowMatchEulerDiscreteScheduler, ZImagePipeline)
    from transformers import AutoModel, AutoTokenizer
    bf16 = torch.bfloat16
    tr = ZImageTransformer2DModel.from_single_file(transformer_path, torch_dtype=bf16)
    vae = AutoencoderKL.from_single_file(AE_FLUX, config=os.path.join(TE_DIR, "flux_vae"), torch_dtype=bf16)
    te = AutoModel.from_pretrained(os.path.join(TE_DIR, "qwen3_4b"), torch_dtype=bf16)
    tok = AutoTokenizer.from_pretrained(os.path.join(TE_DIR, "qwen3_4b"))
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    return ZImagePipeline(transformer=tr, vae=vae, text_encoder=te, tokenizer=tok, scheduler=sched)


def _assemble_flux2_klein(Pipe, transformer_path: str):
    """Assembla una pipeline FLUX.2 KLEIN (edit/inpaint/kontext) — variante DISTILLATA, encoder Qwen3
    (NON Mistral), transformer fp8 puro col prefisso `model.diffusion_model.`. Config transformer
    DEDOTTA dai dim del file → 4B (jad 7680, qwen3_4b) o miraclein 8B (jad 12288, qwen3_8b). VAE = flux2-vae.
    Block-swap memory-bounded come Dev. Niente ComfyUI."""
    import json
    import torch
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file
    from inference_server.backends.flux2_transformer import (klein_config_from_header,
                                                             load_flux2_klein_transformer)
    bf16 = torch.bfloat16
    cfg = klein_config_from_header(transformer_path)
    jad = cfg["joint_attention_dim"]
    qdir = KLEIN_QWEN_DIRS.get(jad)
    if not qdir or not os.path.exists(os.path.join(qdir, "model.safetensors")):
        need = {7680: "Qwen3-4B", 12288: "Qwen3-8B"}.get(jad, f"Qwen3 (jad={jad})")
        raise RuntimeError(
            f"FLUX.2 Klein richiede l'encoder {need} in '{qdir}'. Manca su disco: scaricalo "
            f"col wizard pesi (è l'unico file mancante per Klein nativo).")
    tr = load_flux2_klein_transformer(transformer_path, cfg)
    vae = AutoencoderKLFlux2.from_config(json.load(open(os.path.join(FLUX2_DIR, "vae", "config.json"))))
    vae.load_state_dict(load_file(FLUX2_VAE), assign=True)
    vae = vae.to(bf16)
    # Qwen3 (Qwen3ForCausalLM): la pipeline Klein chiama text_encoder(..., output_hidden_states=True) e
    # concatena i layer [9,18,27]. Caricato in bf16 dai pesi locali (no repo runtime, no ComfyUI).
    te = AutoModelForCausalLM.from_pretrained(qdir, torch_dtype=bf16)
    tok = AutoTokenizer.from_pretrained(qdir)
    try:
        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(FLUX2_DIR, "scheduler"))
    except Exception:                                   # noqa: BLE001
        sched = FlowMatchEulerDiscreteScheduler()
    return Pipe(transformer=tr, vae=vae, text_encoder=te, tokenizer=tok, scheduler=sched)


def _assemble_flux2(Pipe, transformer_path: str):
    """Assembla una pipeline FLUX.2 (t2i/inpaint/kontext) da: transformer LOCALE (single_file) +
    VAE FLUX.2 LOCALE + TE Mistral HF + tokenizer (scaricati gated, solo questi). Pipe è la
    classe diffusers concreta (Flux2Pipeline, Flux2KleinInpaintPipeline, …)."""
    import os
    import torch
    # Klein è un'architettura DISTILLATA con encoder Qwen3 (≠ Dev/Mistral): instradata a parte.
    if "Klein" in getattr(Pipe, "__name__", ""):
        return _assemble_flux2_klein(Pipe, transformer_path)
    from diffusers import Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from transformers import AutoModel, AutoTokenizer
    bf16 = torch.bfloat16
    import json
    from safetensors.torch import load_file
    # Transformer FLUX.2: loader fp8-aware memory-bounded (applica le weight_scale che from_single_file
    # ignora → qualità corretta; streaming+assign-and-free → niente picco RAM ~50GB). Fallback a
    # from_single_file solo se il config transformer non è presente.
    tr_cfg_dir = os.path.join(FLUX2_DIR, "transformer")
    if os.path.exists(os.path.join(tr_cfg_dir, "config.json")):
        from inference_server.backends.flux2_transformer import load_flux2_transformer
        tr = load_flux2_transformer(transformer_path, tr_cfg_dir)
    else:
        tr = Flux2Transformer2DModel.from_single_file(transformer_path, torch_dtype=bf16)
    # AutoencoderKLFlux2 NON supporta from_single_file → from_config + load_state_dict (chiavi 1:1)
    vae = AutoencoderKLFlux2.from_config(json.load(open(os.path.join(FLUX2_DIR, "vae", "config.json"))))
    vae.load_state_dict(load_file(FLUX2_VAE), assign=True)
    vae = vae.to(bf16)
    # Text encoder FLUX.2 = Mistral3 nvfp4 (~12GB, file LOCALE) — dequant→bf16 (no repo 48GB, no ComfyUI)
    te_dir = os.path.join(FLUX2_DIR, "text_encoder")
    if os.path.exists(os.path.join(te_dir, "model.safetensors")):
        te = AutoModel.from_pretrained(te_dir, torch_dtype=bf16)      # se già convertito su disco
    else:
        from inference_server.backends.flux2_te import load_flux2_text_encoder
        te = load_flux2_text_encoder(te_dir)                          # dequant nvfp4 dal file locale
    # FLUX.2 vuole il PROCESSOR (PixtralProcessor), NON un tokenizer puro: Flux2Pipeline.encode_prompt usa
    # tokenizer.apply_chat_template col formato multimodale + system message. Con AutoTokenizer
    # (LlamaTokenizerFast) il chat template/i token speciali sono diversi → embedding sbagliati → RUMORE.
    from transformers import AutoProcessor
    try:
        tok = AutoProcessor.from_pretrained(os.path.join(FLUX2_DIR, "tokenizer"))
    except Exception:                                   # noqa: BLE001
        tok = AutoTokenizer.from_pretrained(os.path.join(FLUX2_DIR, "tokenizer"))
    try:
        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(FLUX2_DIR, "scheduler"))
    except Exception:                                   # noqa: BLE001
        sched = FlowMatchEulerDiscreteScheduler()
    return Pipe(transformer=tr, vae=vae, text_encoder=te, tokenizer=tok, scheduler=sched)


def _assemble_ernie():
    """Assembla ErnieImagePipeline (Baidu ERNIE-Image) da pesi LOCALI, con shim per transformers <5
    (il repo diffusers vuole transformers 5: TokenizersBackend / Ministral3 / model_type 'ministral3').
    transformer (single-file, chiavi 1:1) + text-encoder Mistral3 (single-file, remap model.→language_model.)
    + vae/tokenizer/scheduler dal repo. NIENTE ComfyUI."""
    import json
    import torch
    import transformers
    from transformers import (Mistral3Model, Mistral3Config, MinistralForCausalLM, PreTrainedTokenizerFast)
    from safetensors.torch import load_file
    from diffusers import (ErnieImageTransformer2DModel, AutoencoderKLFlux2,
                           FlowMatchEulerDiscreteScheduler, ErnieImagePipeline)
    bf16 = torch.bfloat16
    if not hasattr(transformers, "Ministral3ForCausalLM"):
        transformers.Ministral3ForCausalLM = MinistralForCausalLM
    if not hasattr(transformers, "TokenizersBackend"):
        transformers.TokenizersBackend = PreTrainedTokenizerFast
    # transformer: config dal repo + pesi dal single-file (chiavi identiche, 409/409)
    tr = ErnieImageTransformer2DModel.from_config(json.load(open(os.path.join(ERNIE_REPO, "transformer", "config.json"))))
    tr.load_state_dict(load_file(ERNIE_TRANSFORMER), assign=True)
    tr = tr.to(bf16)
    # text encoder Mistral3: 'ministral3' (sconosciuto a transformers 4) → 'ministral' + sliding_window=full
    tcfg = json.load(open(os.path.join(ERNIE_REPO, "text_encoder", "config.json")))
    tc = tcfg.get("text_config", {})
    if tc.get("model_type") == "ministral3":
        tc["model_type"] = "ministral"
    if tc.get("sliding_window") is None:
        tc["sliding_window"] = tc.get("max_position_embeddings", 262144)
    te = Mistral3Model(Mistral3Config(**tcfg))
    raw = load_file(ERNIE_TE)
    rem = {("language_model." + k[6:]) if k.startswith("model.") else k: v
           for k, v in raw.items() if not k.startswith("tekken_model")}
    te.load_state_dict(rem, strict=False)
    te = te.to(bf16)
    vae = AutoencoderKLFlux2.from_pretrained(os.path.join(ERNIE_REPO, "vae"), torch_dtype=bf16)
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(ERNIE_REPO, "tokenizer", "tokenizer.json"))
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(ERNIE_REPO, "scheduler"))
    return ErnieImagePipeline(transformer=tr, vae=vae, text_encoder=te, tokenizer=tok, scheduler=sched)


def _arch_of(params: dict) -> str:
    a = (params.get("task") or params.get("arch") or "flux").lower()
    if a == "flux" and any(t in str(params.get("model", "")).lower()
                           for t in ("flux2", "flux_2", "flux 2", "flux.2")):
        return "flux2"                                   # FLUX_2_Dev/FLUX.2 → loader flux2 (config+TE diversi)
    return a if a in SPEC else "flux"


def _load(model_path: str, arch: str):
    """Loader per il ModelManager: ritorna (pipe, vram_mb)."""
    import torch
    import diffusers
    # Risolvi il path RELATIVO (es. "Flux\\FLUX_Krea_Dev.safetensors" dall'inventario filesystem) →
    # assoluto sotto diffusion_models. Senza, Path.is_file() è False e si cade in from_pretrained che
    # interpreta il nome come repo_id HF ("Repo id must use alphanumeric..."). Tollera spazi↔underscore.
    if not Path(model_path).is_file():
        dm = Path(ERNIE_TRANSFORMER).parent.parent              # ...\Models\diffusion_models
        # cerca in diffusion_models E in Checkpoints (i checkpoint completi stanno lì, es. flux1-schnell-fp8)
        bases = [dm, dm.parent / "Checkpoints", dm.parent / "checkpoints", dm.parent]
        _name = Path(model_path).name
        cands = []
        for b in bases:
            cands += [b / model_path, (b / model_path).with_name(_name.replace(" ", "_")),
                      b / _name, b / "Flux" / _name]
        for cand in cands:
            if cand.is_file():
                model_path = str(cand)
                break
    # AUTO-DEQUANT: se è un checkpoint quantizzato ComfyUI (comfy_quant), convertilo in bf16 (cache).
    try:
        from core.dequant_comfy import ensure_dequantized
        model_path = ensure_dequantized(model_path)
    except Exception:                                       # noqa: BLE001
        pass
    cls_name, supports_sf, _ = SPEC[arch]
    Pipe = getattr(diffusers, cls_name, None)
    if Pipe is None:
        # es. Flux2KontextPipeline non esiste in diffusers 0.38 → messaggio chiaro + suggerimento
        alt = "kontext" if arch == "flux2_kontext" else "flux2_inpaint" if arch.startswith("flux2") else ""
        raise RuntimeError(
            f"La classe diffusers '{cls_name}' per arch '{arch}' non esiste in diffusers "
            f"{diffusers.__version__}." + (f" Usa '{alt}' per questa funzione." if alt else
            " Aggiorna diffusers o scegli un'altra pipeline."))
    is_file = Path(model_path).is_file()
    if arch == "ernie":
        pipe = _assemble_ernie()                            # assemblaggio locale (shim transformers <5)
    elif arch.startswith("flux2") and is_file:
        # FLUX.2: transformer locale + VAE locale + TE Mistral HF (assemblaggio, anche inpaint/kontext)
        pipe = _assemble_flux2(Pipe, model_path)
    elif is_file and not supports_sf:
        raise RuntimeError(
            f"{cls_name} non supporta from_single_file: per '{arch}' serve un repo diffusers "
            f"(scaricalo col wizard pesi), non il file singolo {Path(model_path).name}.")
    elif is_file:
        try:
            pipe = Pipe.from_single_file(model_path, torch_dtype=torch.bfloat16)
        except Exception:                               # noqa: BLE001
            # file 'solo transformer' (manca TE/VAE nel checkpoint) → assembla coi componenti locali
            if arch in ("flux", "kontext", "inpaint"):
                pipe = _assemble_flux(Pipe, model_path, arch=arch)
            elif arch == "zimage":
                pipe = _assemble_zimage(model_path)
            else:
                raise
    else:
        pipe = Pipe.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    try:
        pipe.vae.enable_tiling()
    except Exception:                                   # noqa: BLE001
        pass
    attn = _set_attention(pipe)
    # FBCache rompe gli arch image-conditioned (Kontext/inpaint): la pipeline non entra nel context
    # manager dell'hook → "No context is set". Lo applichiamo solo ai t2i puri.
    fb = _apply_fbcache(pipe) if not SPEC.get(arch, ("", False, False))[2] else False
    comp = _maybe_compile(pipe)
    # Stima footprint REALE dalla dimensione del file pesi (robusto per fp16/fp8/gguf, ogni arch):
    # tutto-su-GPU ≈ pesi transformer + TE/VAE/working. Se sovrastima la VRAM, _place sceglie
    # enable_model_cpu_offload invece di spillare nella shared-RAM (vedi outpaint lento). Vale per
    # generazione/inpaint/outpaint/integra/kontext con QUALSIASI modello, non solo i preset.
    try:
        sz_mb = os.path.getsize(model_path) / 1024**2 if os.path.isfile(model_path) else 0
    except Exception:                                       # noqa: BLE001
        sz_mb = 0
    overhead = 4000 if arch == "zimage" else 11000          # T5/CLIP/Mistral + VAE + working
    est_mb = max(VRAM_MB.get(arch, 16000), int(sz_mb + overhead)) if sz_mb else VRAM_MB.get(arch, 16000)
    # FLUX.2 (transformer ~32B fp8 = 33GB + TE Mistral): NON entra tutto-su-GPU nei 32GB e nemmeno in
    # enable_model_cpu_offload (offload a livello-modello = transformer intero su GPU). Serve offload
    # SEQUENZIALE (granularità per-modulo: un layer alla volta su GPU). Memory-bounded e imposta
    # correttamente pipe._execution_device=cuda. Override: NATIVE_FLUX2_PLACE=offload|sequential.
    if arch.startswith("flux2"):
        # FLUX.2 (transformer ~32B fp8 = 33GB + TE Mistral nvfp4): NON entra tutto-su-GPU nei 32GB.
        #  - 'blockswap' (DEFAULT): i pesi fp8/nvfp4 restano su CPU, spostati 1 layer alla volta nel forward
        #    (Fp8Linear/Nvfp4Linear), i moduli leggeri (norm/embed/modulation/bias) stabilmente su GPU.
        #    Picco VRAM ≈ moduli leggeri + 1 layer pesante → entra nei 32GB ed è veloce (no spill shared-RAM).
        #  - 'offload'    : enable_model_cpu_offload (su Windows spilla in shared-RAM → praticamente bloccato).
        #  - 'sequential' : enable_sequential_cpu_offload (Segmentation fault coi buffer float8).
        # Override: NATIVE_FLUX2_PLACE=blockswap|offload|sequential.
        mode = (os.environ.get("NATIVE_FLUX2_PLACE") or "blockswap").lower()
        if mode == "blockswap":
            where = _place_flux2_blockswap(pipe)
        elif mode == "sequential":
            try:
                pipe.enable_sequential_cpu_offload(); where = "sequential"
            except Exception as _e:                         # noqa: BLE001
                print(f"[image] flux2 sequential offload fallito ({_e}) → blockswap", flush=True)
                where = _place_flux2_blockswap(pipe)
        else:
            pipe.enable_model_cpu_offload(); where = "offload"
    else:
        where = _place(pipe, est_mb)
    print(f"[image] {arch}: {where} (est {est_mb}MB, file {int(sz_mb)}MB), attention={attn}, fbcache={fb}, compile={comp}", flush=True)
    return pipe, VRAM_MB.get(arch, 16000)


def register(mm, model: str, arch: str) -> str:
    key = f"{arch}:{model}"
    mm.register(key, lambda: _load(model, arch))
    return key


def run(params: dict, mm, output_dir: str) -> dict:
    import torch
    from PIL import Image
    arch = _arch_of(params)
    model = params["model"]
    key = register(mm, model, arch)
    pipe = mm.ensure(key, need_mb=VRAM_MB.get(arch, 16000))
    mm.apply_loras(pipe, params.get("loras"))

    # Face adapter PuLID (identità del volto da una foto) — FLUX/kontext
    _pulid_h = None
    if params.get("pulid_face") and arch in ("flux", "kontext"):
        try:
            from core import pulid_adapter
            _pl = pulid_adapter.load(pipe.transformer)
            _emb = pulid_adapter.extract_id(_pl, params["pulid_face"])
            _pulid_h = pulid_adapter.attach(pipe.transformer, _pl, _emb,
                                            float(params.get("pulid_strength", 0.8)))
            print(f"[pulid] attivo ({len(_pulid_h)} hook, strength={params.get('pulid_strength', 0.8)})", flush=True)
        except Exception as _e:                          # noqa: BLE001
            print(f"[pulid] disattivato (errore): {_e}", flush=True)
            _pulid_h = None

    g = torch.Generator(device="cuda").manual_seed(int(params.get("seed", 0)))
    kw = dict(prompt=params.get("prompt", ""),
              num_inference_steps=int(params.get("steps", 20)),
              guidance_scale=float(params.get("guidance", 3.5)),
              generator=g)
    neg = params.get("negative")
    # immagine/maschera per kontext e inpaint
    _, _, needs_image = SPEC[arch]
    if needs_image:
        img_p = params.get("image_path")
        if not img_p or not Path(img_p).exists():
            return {"status": "error", "error": f"'{arch}' richiede 'image_path' valido."}
        kw["image"] = Image.open(img_p).convert("RGB")
        if arch in _MASK_ARCHS:
            mk = params.get("mask_path")
            if not mk or not Path(mk).exists():
                return {"status": "error", "error": f"'{arch}' richiede 'mask_path' valido."}
            # OUTPAINT = inpaint della regione COMPLEMENTARE → maschera invertita (riusa mask_utils)
            if params.get("outpaint"):
                try:
                    from core import mask_utils
                    inv = str(Path(output_dir) / "_outpaint_mask.png")
                    os.makedirs(output_dir, exist_ok=True)
                    mk = mask_utils.invert_mask(mk, inv)
                except Exception as _e:                      # noqa: BLE001
                    print(f"[inpaint] invert_mask fallito ({_e}); uso maschera originale", flush=True)
            kw["mask_image"] = Image.open(mk).convert("RGB")
    else:
        kw["width"] = int(params.get("width", 1024))
        kw["height"] = int(params.get("height", 1024))
    # i modelli "guidance-distilled" ignorano/non espongono il negative_prompt:
    # zimage turbo, Flux2Pipeline (flux2) e FluxFillPipeline (inpaint/outpaint).
    if neg and arch not in ("zimage", "flux2", "inpaint"):
        kw["negative_prompt"] = neg

    try:
        out = pipe(**kw)
    finally:
        if _pulid_h:
            from core import pulid_adapter
            pulid_adapter.detach(_pulid_h)
    os.makedirs(output_dir, exist_ok=True)
    path = str(Path(output_dir) / f"native_{arch}_{params.get('seed', 0)}.png")
    out.images[0].save(path)
    return {"status": "success", "output": [path]}
