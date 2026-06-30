"""Motore video LTX-2.3 che usa il LOADER + MEMORY-MANAGEMENT di ComfyUI come LIBRERIA (no server/UI).

Il memory-management di ComfyUI regola la VRAM DINAMICAMENTE: su 32GB tiene il transformer in VRAM
(massima velocità + qualità), offloada per-blocco solo se necessario. fp8-native, niente upcast.
Pesi dal workflow reale Pluto_AllInOne (transformer-only fp8 + Gemma + projection ltxv + VAE).

Eseguito nel venv ComfyUI: E:\AI\Comfy_Nexus\ComfyUI\venv\Scripts\python.exe
Uso: python workers/ltx_comfy_native.py params.json
"""
import asyncio
import json
import os
import subprocess
import sys

COMFY = r"E:\AI\Comfy_Nexus.bak\ComfyUI"   # repo allineato al venv attivo (torch 2.8+cu128 + SageAttention)
sys.path.insert(0, COMFY)
os.chdir(COMFY)                                          # cwd corretto: import relativi (utils) + extra paths

import torch  # noqa: E402
import folder_paths  # noqa: E402

# Registra i percorsi modelli reali (E:\AI\Models) oltre ai default di ComfyUI.
for _folder, _path in (("diffusion_models", r"E:\AI\Models\diffusion_models"),
                       ("text_encoders", r"E:\AI\Models\TextEncoders"),
                       ("clip", r"E:\AI\Models\TextEncoders"),
                       ("vae", r"E:\AI\Models\VAE"),
                       ("checkpoints", r"E:\AI\Models\Checkpoints")):
    try:
        folder_paths.add_model_folder_path(_folder, _path)
    except Exception:  # noqa: BLE001
        pass


def _vals(x):
    """Tutti i valori di ritorno (tuple o io.NodeOutput)."""
    if isinstance(x, tuple):
        return list(x)
    items = getattr(x, "result", None) or getattr(x, "items", None)
    return list(items) if items is not None else [x]


def _unwrap(x):
    """Primo valore di ritorno (tuple o io.NodeOutput)."""
    return _vals(x)[0]


async def _init():
    import nodes
    await nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False)


def run(p: dict) -> str:
    import time
    import nodes
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider, CFGGuider, KSamplerSelect, RandomNoise, SamplerCustomAdvanced)
    from comfy_extras.nodes_lt import EmptyLTXVLatentVideo, LTXVScheduler

    t0 = time.time()
    model = _unwrap(nodes.UNETLoader().load_unet(
        p.get("model", "LTXVideo/v2/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"),
        "default"))
    clip = _unwrap(nodes.DualCLIPLoader().load_clip(
        p.get("clip1", "gemma_3_12B_it.safetensors"),
        p.get("clip2", "ltx-2.3_text_projection_bf16.safetensors"), "ltxv"))
    vae = _unwrap(nodes.VAELoader().load_vae(p.get("vae", "LTX23_video_vae_bf16.safetensors")))
    print(f"[load] {time.time()-t0:.0f}s", flush=True)
    if torch.cuda.is_available():
        print(f"[vram] post-load {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    enc = nodes.CLIPTextEncode()
    pos = _unwrap(enc.encode(clip, p.get("prompt", "")))
    neg = _unwrap(enc.encode(clip, p.get("negative", "worst quality, blurry")))
    W, H, F = int(p.get("width", 768)), int(p.get("height", 512)), int(p.get("num_frames", 97))
    if p.get("image_path"):                              # i2v: condiziona sul primo frame
        import numpy as np
        from PIL import Image
        from comfy_extras.nodes_lt import LTXVImgToVideo
        im = Image.open(p["image_path"]).convert("RGB")
        img_t = torch.from_numpy(np.asarray(im, dtype="float32") / 255.0)[None]   # [1,H,W,3]
        v = _vals(LTXVImgToVideo().execute(pos, neg, img_t, vae, W, H, F, 1,
                                           float(p.get("img_strength", 1.0))))
        pos, neg, latent = v[0], v[1], v[2]
    else:
        latent = _unwrap(EmptyLTXVLatentVideo().execute(W, H, F))
    if not isinstance(latent, dict):
        latent = {"samples": getattr(latent, "samples", latent)}
    print(f"[cond] ok, latent={tuple(latent['samples'].shape)}", flush=True)

    cfg = float(p.get("guidance", 1.0))
    if cfg <= 1.0:
        guider = _unwrap(BasicGuider().get_guider(model, pos))
    else:
        guider = _unwrap(CFGGuider().get_guider(model, pos, neg, cfg))
    sampler = _unwrap(KSamplerSelect().execute(p.get("sampler", "euler")))
    sigmas = _unwrap(LTXVScheduler().execute(int(p.get("steps", 30)), 2.05, 0.95, True, 0.1, latent))
    noise = _unwrap(RandomNoise().execute(int(p.get("seed", 42))))

    tg = time.time()
    out = SamplerCustomAdvanced().execute(noise, guider, sampler, sigmas, latent)
    denoised = out[1] if isinstance(out, tuple) and len(out) > 1 else _unwrap(out)
    if not isinstance(denoised, dict):
        denoised = {"samples": getattr(denoised, "samples", denoised)}
    print(f"[gen] {time.time()-tg:.0f}s", flush=True)

    images = _unwrap(nodes.VAEDecode().decode(vae, denoised))     # (frames,H,W,3) [0,1]
    return _save_mp4(images, p["output"], float(p.get("fps", 24.0)))


def _save_mp4(images, out_path: str, fps: float) -> str:
    import numpy as np
    arr = (images.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # (N,H,W,3)
    n, h, w, _ = arr.shape
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path],
        stdin=subprocess.PIPE)
    proc.stdin.write(arr.tobytes())
    proc.stdin.close()
    proc.wait()
    return out_path


def main():
    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    asyncio.run(_init())
    with torch.inference_mode():                          # come ComfyUI: niente grafo autograd
        print("OK:" + run(p), flush=True)


if __name__ == "__main__":
    main()
