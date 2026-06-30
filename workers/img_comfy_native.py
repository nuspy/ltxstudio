"""Motore IMMAGINI che usa il LOADER + MEMORY-MANAGEMENT di ComfyUI come LIBRERIA (no server/UI).

Stesso approccio di workers/ltx_comfy_native.py (video): il model_management di ComfyUI regola la VRAM
DINAMICAMENTE (tiene su GPU ciò che entra, offload per-blocco con buffer riusabile quando serve) →
scala da 8GB a 48GB senza OOM. fp8 nativo, niente upcast. Selezionabile come motore alternativo al
path diffusers (image_flux.py) via il setting `image_engine`.

Eseguito nel venv ComfyUI. Uso: python workers/img_comfy_native.py params.json
params: arch(flux|flux2), model(rel a diffusion_models), clip1, clip2, clip_type, vae, prompt, negative,
        width, height, steps, guidance, sampler, scheduler, seed, output.
"""
import asyncio
import json
import os
import sys

COMFY = r"E:\AI\Comfy_Nexus.bak\ComfyUI"
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402
import folder_paths  # noqa: E402

for _folder, _path in (("diffusion_models", r"E:\AI\Models\diffusion_models"),
                       ("checkpoints", r"E:\AI\Models\Checkpoints"),
                       ("text_encoders", r"E:\AI\Models\TextEncoders"),
                       ("clip", r"E:\AI\Models\TextEncoders"),
                       ("vae", r"E:\AI\Models\VAE")):
    try:
        folder_paths.add_model_folder_path(_folder, _path)
    except Exception:  # noqa: BLE001
        pass


def _vals(x):
    if isinstance(x, tuple):
        return list(x)
    items = getattr(x, "result", None) or getattr(x, "items", None)
    return list(items) if items is not None else [x]


def _unwrap(x):
    return _vals(x)[0]


async def _init():
    import nodes
    await nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False)


def run(p: dict) -> str:
    import time
    import nodes
    from comfy_extras.nodes_flux import FluxGuidance

    arch = (p.get("arch") or "flux").lower()
    t0 = time.time()
    # transformer: UNETLoader per i single-file (la maggioranza), CheckpointLoaderSimple per i checkpoint completi
    model_name = p.get("model")
    if p.get("checkpoint"):
        model, clip, vae = (_vals(nodes.CheckpointLoaderSimple().load_checkpoint(p["checkpoint"]))[:3])
    else:
        model = _unwrap(nodes.UNETLoader().load_unet(model_name, p.get("weight_dtype", "default")))
        # CLIP: flux = (clip_l, t5xxl, 'flux'); flux2 = encoder mistral/qwen, type 'flux2'
        ctype = p.get("clip_type") or ("flux2" if arch == "flux2" else "flux")
        c1 = p.get("clip1") or ("mistral_3_small_flux2.safetensors" if arch == "flux2" else "clip_l.safetensors")
        c2 = p.get("clip2") or ("" if arch == "flux2" else "t5xxl_fp16.safetensors")
        if c2:
            clip = _unwrap(nodes.DualCLIPLoader().load_clip(c1, c2, ctype))
        else:
            clip = _unwrap(nodes.CLIPLoader().load_clip(c1, ctype))
        vae = _unwrap(nodes.VAELoader().load_vae(
            p.get("vae") or ("flux2-vae.safetensors" if arch == "flux2" else "ae.safetensors")))
    print(f"[load] {time.time()-t0:.0f}s (arch={arch})", flush=True)

    enc = nodes.CLIPTextEncode()
    pos = _unwrap(enc.encode(clip, p.get("prompt", "")))
    neg = _unwrap(enc.encode(clip, p.get("negative", "")))
    g = float(p.get("guidance", 3.5))
    pos = _unwrap(FluxGuidance().append(pos, g))             # FLUX usa la guidance distillata
    W, H = int(p.get("width", 1024)), int(p.get("height", 1024))
    try:
        latent = _unwrap(nodes.EmptySD3LatentImage().generate(W, H, 1))
    except Exception:                                        # noqa: BLE001
        latent = _unwrap(nodes.EmptyLatentImage().generate(W, H, 1))

    tg = time.time()
    out = nodes.KSampler().sample(
        model, int(p.get("seed", 0)), int(p.get("steps", 20)), 1.0,
        p.get("sampler", "euler"), p.get("scheduler", "simple"), pos, neg, latent, 1.0)
    denoised = _unwrap(out)
    if not isinstance(denoised, dict):
        denoised = {"samples": getattr(denoised, "samples", denoised)}
    print(f"[gen] {time.time()-tg:.0f}s", flush=True)
    images = _unwrap(nodes.VAEDecode().decode(vae, denoised))
    return _save_png(images, p["output"])


def _save_png(images, out_path: str) -> str:
    import numpy as np
    from PIL import Image
    arr = (images[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)   # (H,W,3)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Image.fromarray(arr).save(out_path)
    return out_path


def main():
    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    asyncio.run(_init())
    with torch.inference_mode():
        print("OK:" + run(p), flush=True)


if __name__ == "__main__":
    main()
