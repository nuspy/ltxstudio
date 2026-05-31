"""Aumento definizione (upscale) con modello AI, con opzionale RIFINITURA AI. ID numerici.

- Upscale: UpscaleModelLoader + ImageUpscaleWithModel (modelli installati: 4xNomos8kSCHAT-L,
  4xNomos2_hq_dat2/drct-l, 4xRealWebPhoto_v4, 4x-UltraSharpV2, …). Se il fattore richiesto
  differisce dal nativo (di solito 4x), ridimensiona con lanczos → AUMENTO RISOLUZIONE.
- Rifinitura AI (opzionale): passata diffusion FLUX img2img a denoise basso sull'immagine
  già ingrandita → inventa/ricostruisce dettaglio fine (più potente, più lento/pesante).
"""
from __future__ import annotations

import re

from core.gen_pipeline import is_flux2, FLUX2_VAE, FLUX2_CLIP

FLUX_DEFAULTS = {"t5": "t5xxl_fp16.safetensors", "clip_l": "long_clip_l_hi_dream.safetensors",
                 "vae": "Flux\\ae.safetensors"}


def model_native_scale(model_name: str) -> int:
    """Ricava il fattore nativo dal nome (es. '4x-UltraSharp' → 4, '2xNomos' → 2)."""
    m = re.match(r"\s*(\d+)\s*x", (model_name or "").lower())
    return int(m.group(1)) if m else 4


class UpscalePipeline:
    def __init__(self, flux_models: dict | None = None):
        self.fx = dict(FLUX_DEFAULTS)
        if flux_models:
            self.fx.update({k: v for k, v in flux_models.items() if v})

    def build(self, image_file: str, model_name: str, *, target_scale: float = 4.0,
              refine: bool = False, refine_model: str | None = None,
              refine_denoise: float = 0.3, refine_steps: int = 12, refine_guidance: float = 3.5,
              refine_prompt: str = "", refine_negative: str = "", seed: int = 0) -> dict:
        native = model_native_scale(model_name)
        p: dict = {}
        p["1"] = {"class_type": "LoadImage", "inputs": {"image": image_file}}
        p["2"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model_name}}
        p["3"] = {"class_type": "ImageUpscaleWithModel",
                  "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}}
        ref = ["3", 0]
        if target_scale and abs(target_scale - native) > 0.05:
            p["4"] = {"class_type": "ImageScaleBy",
                      "inputs": {"image": ["3", 0], "upscale_method": "lanczos",
                                 "scale_by": float(target_scale) / float(native)}}
            ref = ["4", 0]

        # --- rifinitura AI (FLUX img2img a denoise basso sull'immagine ingrandita) ---
        if refine and refine_model:
            p["10"] = {"class_type": "UNETLoader",
                       "inputs": {"unet_name": refine_model, "weight_dtype": "default"}}
            if is_flux2(refine_model):
                p["11"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": FLUX2_CLIP, "type": "flux2"}}
                p["12"] = {"class_type": "VAELoader", "inputs": {"vae_name": FLUX2_VAE}}
            else:
                p["11"] = {"class_type": "DualCLIPLoader",
                           "inputs": {"clip_name1": self.fx["t5"], "clip_name2": self.fx["clip_l"], "type": "flux"}}
                p["12"] = {"class_type": "VAELoader", "inputs": {"vae_name": self.fx["vae"]}}
            p["13"] = {"class_type": "CLIPTextEncode", "inputs": {"text": refine_prompt or "", "clip": ["11", 0]}}
            p["14"] = {"class_type": "FluxGuidance",
                       "inputs": {"conditioning": ["13", 0], "guidance": float(refine_guidance)}}
            p["15"] = {"class_type": "CLIPTextEncode", "inputs": {"text": refine_negative or "", "clip": ["11", 0]}}
            p["16"] = {"class_type": "VAEEncode", "inputs": {"pixels": ref, "vae": ["12", 0]}}
            p["17"] = {"class_type": "KSampler",
                       "inputs": {"model": ["10", 0], "positive": ["14", 0], "negative": ["15", 0],
                                  "latent_image": ["16", 0], "seed": int(seed), "steps": int(refine_steps),
                                  "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                                  "denoise": float(refine_denoise)}}
            p["18"] = {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["12", 0]}}
            ref = ["18", 0]

        p["5"] = {"class_type": "SaveImage", "inputs": {"images": ref, "filename_prefix": "ltxstudio_upscale"}}
        return p
