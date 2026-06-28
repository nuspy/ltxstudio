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
VRAM_MB = {"flux": 16000, "flux2": 24000, "zimage": 12000, "ernie": 16000,
           "kontext": 16000, "inpaint": 16000, "flux2_kontext": 24000, "flux2_inpaint": 24000}
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


def _place(pipe, vram_mb: int) -> str:
    """Posizionamento VRAM (sicuro e portabile, default).

    NB Windows: forzare tutto su GPU quando la VRAM è stretta innesca il *system memory fallback*
    del driver NVIDIA (usa la RAM come VRAM) → lentezza estrema. Quindi DEFAULT =
    enable_model_cpu_offload (funziona su ogni GPU/VRAM, dalle 5090 alle schede minori).
    Il "full GPU" è opt-in solo con VRAM molto abbondante (NATIVE_FULL_GPU=1) e margine ampio."""
    import os
    import torch
    full = os.environ.get("NATIVE_FULL_GPU") == "1"
    if full and torch.cuda.is_available():
        free_mb = torch.cuda.mem_get_info()[0] / 1024**2
        if free_mb > vram_mb * 2.5:                     # margine ampio per evitare sysmem fallback
            pipe.to("cuda")
            return "gpu"
    pipe.enable_model_cpu_offload()
    return "offload"


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


def _assemble_flux(Pipe, transformer_path: str):
    """Assembla una pipeline FLUX-family da: transformer LOCALE (single_file) + VAE LOCALE
    (ae.safetensors) + text encoder convertiti (CLIP-L, T5-xxl). Per i file 'solo transformer'
    (diffusion_models) che non contengono i text encoder."""
    import os
    import torch
    from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    bf16 = torch.bfloat16
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


def _assemble_flux2(Pipe, transformer_path: str):
    """Assembla una pipeline FLUX.2 (t2i/inpaint/kontext) da: transformer LOCALE (single_file) +
    VAE FLUX.2 LOCALE + TE Mistral HF + tokenizer (scaricati gated, solo questi). Pipe è la
    classe diffusers concreta (Flux2Pipeline, Flux2KleinInpaintPipeline, …)."""
    import os
    import torch
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
    if a == "flux" and "flux2" in str(params.get("model", "")).lower():
        return "flux2"
    return a if a in SPEC else "flux"


def _load(model_path: str, arch: str):
    """Loader per il ModelManager: ritorna (pipe, vram_mb)."""
    import torch
    import diffusers
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
                pipe = _assemble_flux(Pipe, model_path)
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
    where = _place(pipe, VRAM_MB.get(arch, 16000))
    print(f"[image] {arch}: {where}, attention={attn}, fbcache={fb}, compile={comp}", flush=True)
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
    # i modelli "guidance-distilled" (flux schnell/zimage turbo) ignorano il negative
    if neg and arch not in ("zimage",):
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
