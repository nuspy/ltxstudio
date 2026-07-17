"""Worker Self-Forcing 1.3B — variante VIDEO **realtime** di krea_rt (autoregressivo, leggero per la 5090).

Self-Forcing (github.com/guandeh17/Self-Forcing, Apache-2.0) distilla Wan 2.1 1.3B in un generatore
autoregressivo (Self-Forcing/DMD, 4 step) → t2v/i2v/estensione in streaming, ~realtime su una singola 4090.
NON è diffusers: si usa il runner del repo come libreria (come LingBot). Girato nel venv `krea_rt_venv`
(torch + FA2 + deps leggere), con cwd = repo (i wrapper cercano `wan_models/Wan2.1-T2V-1.3B/` RELATIVO).

Pesi (catalogo): `self_forcing_dmd` (checkpoints/self_forcing_dmd.pt, 5.6GB) + base **formato Wan nativo**
`wan21_13b_original` (Wan-AI/Wan2.1-T2V-1.3B: models_t5_umt5-xxl-enc-bf16.pth + Wan2.1_VAE.pth + WanModel).

VRAM: `DynamicSwapInstaller` (block-swap nativo del repo) sotto i 40GB → entra comodo sui 32GB.
⚠️ LIVE-TEST PENDENTE: forma del noise per risoluzioni ≠480×832, ritorno di inference(), Windows-compat
(# VALIDATE) — attesa iterazione come per gli altri.
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SF_REPO = r"E:\AI\Tools\self-forcing"
if _ROOT not in sys.path:                       # core.* importabile fin da subito (prima del chdir sul repo)
    sys.path.insert(0, _ROOT)


def _resolve_paths(p: dict) -> tuple[str, str]:
    """(checkpoint_dmd.pt, wan_native_dir) dal catalogo pesi."""
    from core import weights_catalog as wc
    ent = {e["id"]: e for e in wc.load()}

    def dmd():
        e = ent.get("self_forcing_dmd")
        if not e:
            return ""
        d = wc.dest_for(e)
        return str(d / e.get("filename", "")) if e.get("kind") == "file" else str(d)

    def wan():
        e = ent.get("wan21_13b_original")
        return str(wc.dest_for(e)) if e else ""

    ckpt = p.get("checkpoint") or dmd()
    wan_dir = p.get("wan_dir") or wan()
    if not (ckpt and os.path.isfile(ckpt)):
        raise RuntimeError("Checkpoint Self-Forcing (self_forcing_dmd.pt) non trovato. Scaricalo dal wizard 'Pesi'.")
    if not (wan_dir and os.path.isdir(wan_dir)):
        raise RuntimeError("Base Wan 2.1 1.3B (formato nativo) non trovata. Scaricala dal wizard 'Pesi'.")
    return ckpt, wan_dir


def _ensure_wan_link(wan_dir: str) -> None:
    """I wrapper cercano `wan_models/Wan2.1-T2V-1.3B/` RELATIVO al repo → creo una junction verso la base scaricata."""
    link = os.path.join(SF_REPO, "wan_models", "Wan2.1-T2V-1.3B")
    if os.path.isdir(link) and os.path.isfile(os.path.join(link, "Wan2.1_VAE.pth")):
        return
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if os.path.exists(link):
        try:
            os.remove(link)
        except OSError:
            pass
    try:
        os.symlink(wan_dir, link, target_is_directory=True)          # richiede privilegio/developer-mode
    except OSError:
        import subprocess                                            # fallback: junction Windows (no admin)
        subprocess.run(["cmd", "/c", "mklink", "/J", link, wan_dir], check=False)


def _snap_lat(px: int) -> int:
    """spaziale latente = pixel / 8 (VAE Wan)."""
    return max(16, int(px) // 8)


def run_generation(p: dict) -> str:
    out = p["output"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    ckpt, wan_dir = _resolve_paths(p)
    _ensure_wan_link(wan_dir)

    # cwd + sys.path del repo (i moduli pipeline/utils/wan sono locali) + root app (core.stego/watermark)
    os.chdir(SF_REPO)
    for pth in (SF_REPO, _ROOT):
        if pth not in sys.path:
            sys.path.insert(0, pth)

    import torch
    from omegaconf import OmegaConf
    from einops import rearrange
    from pipeline import CausalInferencePipeline
    from utils.misc import set_seed
    from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    set_seed(int(p.get("seed", 0)))

    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/self_forcing_dmd.yaml"))
    pipeline = CausalInferencePipeline(cfg, device=device)          # few-step (denoising_step_list) → causale
    sd = torch.load(ckpt, map_location="cpu")
    key = "generator_ema" if "generator_ema" in sd else "generator"
    pipeline.generator.load_state_dict(sd[key])                    # VALIDATE: chiave stato (ema vs no)
    pipeline = pipeline.to(dtype=torch.bfloat16)

    low = get_cuda_free_memory_gb(gpu) < 40                        # 32GB → block-swap del text-encoder
    if low:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)

    # noise autoregressivo: [1, num_output_frames(latenti), 16, H/8, W/8]. Default 21 latenti ≈ 81px @16fps ≈ 5s.
    # Estensione/long-video: aumentare num_output_frames (rolling KV-cache) → memory-bounded.
    W = int(p.get("width", 832)); H = int(p.get("height", 480))
    lat_h, lat_w = _snap_lat(H), _snap_lat(W)
    n_out = int(p.get("num_output_frames") or 0)
    if n_out <= 0:                                                 # da durata pixel: ~4 px per latente
        n_out = max(3, ((int(p.get("num_frames", 81)) - 1) // 4) + 1)
    fps = int(p.get("fps", 16))
    prompt = p.get("prompt", "")

    initial_latent = None                                         # i2v/estensione: VALIDATE (encode_to_latent)
    if p.get("image_path") and os.path.exists(str(p["image_path"])):
        from PIL import Image
        import numpy as np
        img = Image.open(p["image_path"]).convert("RGB").resize((W, H))
        t = torch.from_numpy(np.asarray(img)).float().div(127.5).sub(1.0)     # [-1,1]
        t = t.permute(2, 0, 1)[None, :, None].to(device=device, dtype=torch.bfloat16)   # (1,C,1,H,W)
        initial_latent = pipeline.vae.encode_to_latent(t).to(device=device, dtype=torch.bfloat16)

    noise = torch.randn([1, n_out, 16, lat_h, lat_w], device=device, dtype=torch.bfloat16)
    print(f"[self_forcing] t2v realtime {W}x{H} ~{n_out} latenti (~{(n_out-1)*4+1}px @ {fps}fps), 4 step", flush=True)
    video, _lat = pipeline.inference(noise=noise, text_prompts=[prompt], return_latents=True,
                                     initial_latent=initial_latent, low_memory=low)
    # (b,t,c,h,w) in [0,1] → (t,h,w,c) uint8
    import numpy as np
    vid = rearrange(video, "b t c h w -> b t h w c")[0].float().cpu().numpy()
    if np.isnan(vid).any() or np.isinf(vid).any():
        raise RuntimeError("Frame NaN/inf: pesi Self-Forcing/Wan corrotti o incompleti.")
    frames = (vid * 255.0).clip(0, 255).astype("uint8")

    import imageio
    imageio.mimsave(out, list(frames), fps=fps, codec="libx264", quality=8, macro_block_size=16)
    try:
        pipeline.vae.model.clear_cache()
    except Exception:                                             # noqa: BLE001
        pass

    try:                                                          # #107 stego + #106 watermark
        from core import stego
        if stego.enabled():
            stego.sign_file(out)
    except Exception as _se:                                      # noqa: BLE001
        print(f"[stego] {_se}", flush=True)
    try:
        from core import watermark
        out = watermark.apply_to_file(out)
    except Exception as _we:                                     # noqa: BLE001
        print(f"[watermark] {_we}", flush=True)
    return out


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    print("OK:" + run_generation(p), flush=True)


if __name__ == "__main__":
    main()
