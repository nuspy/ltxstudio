"""Handler PoC: text-to-image FLUX.1 / FLUX.2 via diffusers, usando il ModelManager.

Prova di concetto della Fase 0: genera un'immagine senza ComfyUI. La parità completa
(scheduler/guidance/encoder per-variante, inpaint, ecc.) arriva in F1.

Eseguito DENTRO l'InferenceServer (venv torch+diffusers).
"""
from __future__ import annotations

import os
from pathlib import Path


def _load_flux(model_path: str, is_flux2: bool):
    """Loader per il ModelManager: ritorna (pipe, vram_mb stimati)."""
    import torch
    if is_flux2:
        from diffusers import Flux2Pipeline as Pipe
    else:
        from diffusers import FluxPipeline as Pipe
    pipe = Pipe.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()                     # gestione VRAM: tiene su GPU solo il modulo attivo
    try:
        pipe.vae.enable_tiling()
    except Exception:                                   # noqa: BLE001
        pass
    vram = 24000 if is_flux2 else 16000                 # stima grezza (MB); affinabile
    return pipe, vram


def run(params: dict, mm, output_dir: str) -> dict:
    """params: {model, prompt, negative, width, height, steps, guidance, seed, loras, arch}.
    Ritorna {status, output:[path]}.
    """
    import torch
    model = params["model"]
    is_flux2 = "flux2" in str(model).lower() or params.get("arch") == "flux2"
    key = register(mm, model, is_flux2)                 # idempotente: registra il loader
    pipe = mm.ensure(key, need_mb=(24000 if is_flux2 else 16000))
    mm.apply_loras(pipe, params.get("loras"))

    g = torch.Generator(device="cuda").manual_seed(int(params.get("seed", 0)))
    out = pipe(
        prompt=params.get("prompt", ""),
        negative_prompt=params.get("negative") or None,
        width=int(params.get("width", 1024)), height=int(params.get("height", 1024)),
        num_inference_steps=int(params.get("steps", 20)),
        guidance_scale=float(params.get("guidance", 3.5)),
        generator=g,
    )
    os.makedirs(output_dir, exist_ok=True)
    path = str(Path(output_dir) / f"native_gen_{params.get('seed', 0)}.png")
    out.images[0].save(path)
    return {"status": "success", "output": [path]}


def register(mm, model: str, is_flux2: bool) -> str:
    """Registra il loader del modello nel ModelManager e ritorna la chiave."""
    key = f"flux2:{model}" if is_flux2 else f"flux:{model}"
    mm.register(key, lambda: _load_flux(model, is_flux2))
    return key
