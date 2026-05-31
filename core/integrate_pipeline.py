"""Integrazione immagini posizionate nella scena: FLUX img2img MASCHERATO. ID numerici.

A differenza dell'inpaint (FLUX Fill, che IGNORA il contenuto mascherato e lo rigenera dal
contesto), qui si fa img2img con un modello FLUX normale + SetLatentNoiseMask: il composito
(base + immagini incollate) viene codificato e SOLO le regioni mascherate vengono denoise a
bassa intensità → il contenuto incollato è PRESERVATO e armonizzato nella scena, guidato dal
prompt di integrazione. La forza (denoise) regola quanto preservare vs reinterpretare.
"""
from __future__ import annotations

from core.gen_pipeline import is_flux2, FLUX2_VAE, FLUX2_CLIP

FLUX_DEFAULTS = {"t5": "t5xxl_fp16.safetensors", "clip_l": "long_clip_l_hi_dream.safetensors",
                 "vae": "Flux\\ae.safetensors"}


class IntegratePipeline:
    def __init__(self, flux_models: dict | None = None):
        self.fx = dict(FLUX_DEFAULTS)
        if flux_models:
            self.fx.update({k: v for k, v in flux_models.items() if v})

    def build(self, image_file: str, mask_file: str, model: str, prompt: str, *,
              denoise: float = 0.55, steps: int = 20, guidance: float = 3.5,
              negative: str = "", seed: int = 0, lora: str | None = None,
              lora_strength: float = 1.0) -> dict:
        p: dict = {}
        p["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}}
        mref = ["1", 0]
        if lora:
            p["20"] = {"class_type": "LoraLoaderModelOnly",
                       "inputs": {"model": mref, "lora_name": lora, "strength_model": float(lora_strength)}}
            mref = ["20", 0]
        if is_flux2(model):                          # FLUX.2: encoder/VAE dedicati
            p["2"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": FLUX2_CLIP, "type": "flux2"}}
            p["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": FLUX2_VAE}}
        else:
            p["2"] = {"class_type": "DualCLIPLoader",
                      "inputs": {"clip_name1": self.fx["t5"], "clip_name2": self.fx["clip_l"], "type": "flux"}}
            p["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": self.fx["vae"]}}
        p["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt or "", "clip": ["2", 0]}}
        p["5"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": float(guidance)}}
        p["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative or "", "clip": ["2", 0]}}
        p["7"] = {"class_type": "LoadImage", "inputs": {"image": image_file}}
        p["8"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_file, "channel": "red"}}
        p["9"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["7", 0], "vae": ["3", 0]}}
        p["10"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["9", 0], "mask": ["8", 0]}}
        p["11"] = {"class_type": "KSampler",
                   "inputs": {"model": mref, "positive": ["5", 0], "negative": ["6", 0],
                              "latent_image": ["10", 0], "seed": int(seed), "steps": int(steps),
                              "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                              "denoise": float(denoise)}}
        p["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}}
        p["13"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": "ltxstudio_integrate"}}
        return p
