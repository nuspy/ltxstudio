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
}
VRAM_MB = {"flux": 16000, "flux2": 24000, "zimage": 12000, "ernie": 16000,
           "kontext": 16000, "inpaint": 16000}

# Text encoder convertiti (pesi locali + config/tokenizer) e VAE locale, per i file solo-transformer
TE_DIR = r"E:\AI\Models\diffusers\_te"
AE_FLUX = r"E:\AI\Models\VAE\Flux\ae.safetensors"


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


def _arch_of(params: dict) -> str:
    a = (params.get("task") or params.get("arch") or "flux").lower()
    if a == "flux" and "flux2" in str(params.get("model", "")).lower():
        return "flux2"
    return a if a in SPEC else "flux"


def _load(model_path: str, arch: str):
    """Loader per il ModelManager: ritorna (pipe, vram_mb)."""
    import torch
    import diffusers
    cls_name, supports_sf, _ = SPEC[arch]
    Pipe = getattr(diffusers, cls_name)
    is_file = Path(model_path).is_file()
    if is_file and not supports_sf:
        raise RuntimeError(
            f"{cls_name} non supporta from_single_file: per '{arch}' serve un repo diffusers "
            f"(scaricalo col wizard pesi), non il file singolo {Path(model_path).name}.")
    if is_file:
        try:
            pipe = Pipe.from_single_file(model_path, torch_dtype=torch.bfloat16)
        except Exception:                               # noqa: BLE001
            # file 'solo transformer' (manca TE/VAE nel checkpoint) → assembla coi componenti locali
            if arch in ("flux", "kontext", "inpaint"):
                pipe = _assemble_flux(Pipe, model_path)
            else:
                raise
    else:
        pipe = Pipe.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_tiling()
    except Exception:                                   # noqa: BLE001
        pass
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
        if arch == "inpaint":
            mk = params.get("mask_path")
            if not mk or not Path(mk).exists():
                return {"status": "error", "error": "inpaint richiede 'mask_path' valido."}
            kw["mask_image"] = Image.open(mk).convert("RGB")
    else:
        kw["width"] = int(params.get("width", 1024))
        kw["height"] = int(params.get("height", 1024))
    # i modelli "guidance-distilled" (flux schnell/zimage turbo) ignorano il negative
    if neg and arch not in ("zimage",):
        kw["negative_prompt"] = neg

    out = pipe(**kw)
    os.makedirs(output_dir, exist_ok=True)
    path = str(Path(output_dir) / f"native_{arch}_{params.get('seed', 0)}.png")
    out.images[0].save(path)
    return {"status": "success", "output": [path]}
