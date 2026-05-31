"""Pipeline inpaint/outpaint (nodi standard ComfyUI). ID numerici (convenzione ComfyUI:
gli ID nominali rompono alcuni custom node / executor v3).

Due sorgenti modello:
- diffusion_model (default): UNETLoader(FLUX_DEV_Fill) + DualCLIPLoader + VAELoader → inpaint specializzato.
- checkpoint: CheckpointLoaderSimple(<ckpt>) → model/clip/vae dal checkpoint (es. getphat Flux Reality).

Inpaint: rigenera l'area mascherata. Outpaint: ImagePadForOutpaint + riempimento.
NB: FLUX Fill ~22 GB → serve RAM/VRAM adeguata.
"""
from __future__ import annotations

DEFAULTS = {
    "unet": "Flux\\FLUX_DEV_Fill.safetensors",
    "t5": "t5xxl_fp16.safetensors",
    "clip_l": "long_clip_l_hi_dream.safetensors",
    "vae": "Flux\\ae.safetensors",
}

# FLUX.2 (Klein) usa encoder/VAE/latent diversi da FLUX.1 → ramo "edit" separato.
# L'inpaint NON è basato su un modello Fill (canali extra), ma su edit-instruct:
# l'immagine intera è data come RIFERIMENTO (ReferenceLatent, stile Kontext) e la
# maschera limita la ri-pittura alla sola regione (SetLatentNoiseMask).
FLUX2_CLIP = "Flux\\mistral_3_small_flux2.safetensors"
FLUX2_VAE = "Flux\\flux2-vae.safetensors"


def is_flux2(name: str) -> bool:
    n = (name or "").lower()
    return "flux_2" in n or "flux2" in n or "flux 2" in n or "klein" in n


class InpaintPipeline:
    def __init__(self, models: dict | None = None):
        self.m = dict(DEFAULTS)
        if models:
            self.m.update({k: v for k, v in models.items() if v})

    def build(self, image_file: str, prompt: str, *, mode: str = "inpaint",
              mask_file: str | None = None, denoise: float = 1.0, steps: int = 20,
              guidance: float = 30.0, seed: int = 0,
              margins: tuple[int, int, int, int] = (0, 0, 0, 0),
              feather: int = 24, mask_grow: int = 0,
              lora: str | None = None, lora_strength: float = 1.0,
              negative: str = "", checkpoint: str | None = None,
              unet: str | None = None) -> dict:
        p: dict = {}
        unet = unet or self.m["unet"]
        # FLUX.2 (Klein NSFW): architettura edit, grafo dedicato (encoder Mistral + VAE flux2).
        if not checkpoint and is_flux2(unet):
            return self._build_flux2(image_file, prompt, mode=mode, mask_file=mask_file,
                                     denoise=denoise, steps=steps, guidance=guidance, seed=seed,
                                     margins=margins, feather=feather, mask_grow=mask_grow,
                                     lora=lora, lora_strength=lora_strength, negative=negative,
                                     unet=unet)
        # --- sorgente modello ---
        if checkpoint:
            p["10"] = {"class_type": "CheckpointLoaderSimple",
                       "inputs": {"ckpt_name": checkpoint}}
            model_ref, clip_ref, vae_ref = ["10", 0], ["10", 1], ["10", 2]
        else:
            if unet.lower().endswith(".gguf"):
                p["10"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}}
            else:
                p["10"] = {"class_type": "UNETLoader",
                           "inputs": {"unet_name": unet, "weight_dtype": "default"}}
            p["12"] = {"class_type": "DualCLIPLoader",
                       "inputs": {"clip_name1": self.m["t5"], "clip_name2": self.m["clip_l"],
                                  "type": "flux"}}
            p["13"] = {"class_type": "VAELoader", "inputs": {"vae_name": self.m["vae"]}}
            model_ref, clip_ref, vae_ref = ["10", 0], ["12", 0], ["13", 0]

        if lora:
            p["20"] = {"class_type": "LoraLoaderModelOnly",
                       "inputs": {"model": model_ref, "lora_name": lora,
                                  "strength_model": float(lora_strength)}}
            model_ref = ["20", 0]
        p["11"] = {"class_type": "DifferentialDiffusion", "inputs": {"model": model_ref}}

        p["14"] = {"class_type": "CLIPTextEncode",
                   "inputs": {"text": prompt or "", "clip": clip_ref}}
        p["15"] = {"class_type": "FluxGuidance",
                   "inputs": {"conditioning": ["14", 0], "guidance": float(guidance)}}
        p["16"] = {"class_type": "CLIPTextEncode",
                   "inputs": {"text": negative, "clip": clip_ref}}
        p["17"] = {"class_type": "LoadImage", "inputs": {"image": image_file}}

        # outpaint con margini espliciti e senza maschera → vero ImagePadForOutpaint;
        # altrimenti (flusso app: composito già a dimensione frame + maschera-bordo) →
        # inpaint della regione mascherata (bordo per outpaint, disegno per inpaint).
        if mode == "outpaint" and any(margins) and not mask_file:
            l, t, r, b = margins
            p["18"] = {"class_type": "ImagePadForOutpaint",
                       "inputs": {"image": ["17", 0], "left": l, "top": t, "right": r,
                                  "bottom": b, "feathering": feather}}
            pixels, mask = ["18", 0], ["18", 1]
        else:
            p["18"] = {"class_type": "LoadImageMask",
                       "inputs": {"image": mask_file or image_file, "channel": "red"}}
            pixels, mask = ["17", 0], ["18", 0]
            # sfuma/espandi la maschera → DifferentialDiffusion fonde la transizione
            # (elimina la linea di giunzione tra zona generata e foto esistente)
            if feather > 0 or mask_grow != 0:
                p["24"] = {"class_type": "GrowMaskWithBlur",
                           "inputs": {"mask": ["18", 0], "expand": int(mask_grow),
                                      "incremental_expandrate": 0.0, "tapered_corners": True,
                                      "flip_input": False, "blur_radius": float(feather),
                                      "lerp_alpha": 1.0, "decay_factor": 1.0,
                                      "fill_holes": False}}
                mask = ["24", 0]

        p["19"] = {"class_type": "InpaintModelConditioning",
                   "inputs": {"positive": ["15", 0], "negative": ["16", 0], "vae": vae_ref,
                              "pixels": pixels, "mask": mask, "noise_mask": True}}
        p["21"] = {"class_type": "KSampler",
                   "inputs": {"model": ["11", 0], "positive": ["19", 0], "negative": ["19", 1],
                              "latent_image": ["19", 2], "seed": int(seed), "steps": int(steps),
                              "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal",
                              "denoise": float(denoise)}}
        p["22"] = {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0], "vae": vae_ref}}
        p["23"] = {"class_type": "SaveImage",
                   "inputs": {"images": ["22", 0], "filename_prefix": "ltxstudio_inpaint"}}
        return p

    def _build_flux2(self, image_file: str, prompt: str, *, mode: str, mask_file: str | None,
                     denoise: float, steps: int, guidance: float, seed: int,
                     margins: tuple[int, int, int, int], feather: int, mask_grow: int,
                     lora: str | None, lora_strength: float, negative: str, unet: str) -> dict:
        """Inpaint/outpaint con FLUX.2 Klein (edit-instruct). Niente Fill-channels:
        l'immagine intera è RIFERIMENTO (ReferenceLatent) + la maschera limita la
        ri-pittura alla regione (SetLatentNoiseMask). DifferentialDiffusion fonde il bordo.
        Guidance FLUX.2 bassa (default 4.0); il 30 di FLUX.1 Fill non si applica qui."""
        g = guidance if (guidance and guidance <= 10) else 4.0
        st = steps or 12
        p: dict = {}
        if unet.lower().endswith(".gguf"):
            p["10"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}}
        else:
            p["10"] = {"class_type": "UNETLoader",
                       "inputs": {"unet_name": unet, "weight_dtype": "default"}}
        model_ref = ["10", 0]
        if lora:
            p["20"] = {"class_type": "LoraLoaderModelOnly",
                       "inputs": {"model": model_ref, "lora_name": lora,
                                  "strength_model": float(lora_strength)}}
            model_ref = ["20", 0]
        # DifferentialDiffusion → la maschera sfumata fonde la transizione senza giunzione.
        p["11"] = {"class_type": "DifferentialDiffusion", "inputs": {"model": model_ref}}

        p["12"] = {"class_type": "CLIPLoader",
                   "inputs": {"clip_name": FLUX2_CLIP, "type": "flux2"}}
        p["13"] = {"class_type": "VAELoader", "inputs": {"vae_name": FLUX2_VAE}}

        p["17"] = {"class_type": "LoadImage", "inputs": {"image": image_file}}

        # outpaint con margini espliciti e senza maschera → ImagePadForOutpaint;
        # altrimenti (flusso app: composito a dimensione frame + maschera) → inpaint regione.
        if mode == "outpaint" and any(margins) and not mask_file:
            l, t, r, b = margins
            p["18"] = {"class_type": "ImagePadForOutpaint",
                       "inputs": {"image": ["17", 0], "left": l, "top": t, "right": r,
                                  "bottom": b, "feathering": feather}}
            pixels, mask = ["18", 0], ["18", 1]
        else:
            p["18"] = {"class_type": "LoadImageMask",
                       "inputs": {"image": mask_file or image_file, "channel": "red"}}
            pixels, mask = ["17", 0], ["18", 0]
            if feather > 0 or mask_grow != 0:
                p["24"] = {"class_type": "GrowMaskWithBlur",
                           "inputs": {"mask": ["18", 0], "expand": int(mask_grow),
                                      "incremental_expandrate": 0.0, "tapered_corners": True,
                                      "flip_input": False, "blur_radius": float(feather),
                                      "lerp_alpha": 1.0, "decay_factor": 1.0,
                                      "fill_holes": False}}
                mask = ["24", 0]

        # latent sorgente: ri-codifica l'immagine; la maschera limita il denoise alla regione.
        p["25"] = {"class_type": "VAEEncode", "inputs": {"pixels": pixels, "vae": ["13", 0]}}
        p["26"] = {"class_type": "SetLatentNoiseMask",
                   "inputs": {"samples": ["25", 0], "mask": mask}}

        p["14"] = {"class_type": "CLIPTextEncode",
                   "inputs": {"text": prompt or "", "clip": ["12", 0]}}
        p["15"] = {"class_type": "FluxGuidance",
                   "inputs": {"conditioning": ["14", 0], "guidance": float(g)}}
        # ReferenceLatent: l'immagine intera (latent sorgente) è il riferimento Kontext-style
        # → il modello "vede" il contesto e ridipinge coerente solo la regione mascherata.
        p["27"] = {"class_type": "ReferenceLatent",
                   "inputs": {"conditioning": ["15", 0], "latent": ["25", 0]}}
        p["16"] = {"class_type": "CLIPTextEncode",
                   "inputs": {"text": negative, "clip": ["12", 0]}}
        p["21"] = {"class_type": "KSampler",
                   "inputs": {"model": ["11", 0], "positive": ["27", 0], "negative": ["16", 0],
                              "latent_image": ["26", 0], "seed": int(seed), "steps": int(st),
                              "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                              "denoise": float(denoise)}}
        p["22"] = {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0], "vae": ["13", 0]}}
        p["23"] = {"class_type": "SaveImage",
                   "inputs": {"images": ["22", 0], "filename_prefix": "ltxstudio_inpaint"}}
        return p
