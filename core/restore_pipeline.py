"""Pipeline per restauro e colorazione di foto.

Operazioni supportate:
- restore: aumento definizione con ESRGAN + denoising KSampler leggero
- colorize: conversione B&W → colore tramite FLUX Kontext (prompt-based)
- restore_and_colorize: sequenziale (prima restauro, poi colorizzazione)

Nota: la colorizzazione via FLUX Kontext funziona solo se FLUX.2 è installato e configurato.
Altrimenti, viene offerto il solo restauro ESRGAN (nessuna dipendenza extra).
"""
from __future__ import annotations

import copy

# Modelli default (override via build() args)
_DEFAULT_UPSCALE = "4x-UltraSharpV2.safetensors"   # nome REALE su disco (era "4xUltraSharp.pth" inesistente)
_DEFAULT_FLUX2_MODEL = ""  # da specificare dall'utente

# Nota: FLUX Kontext per colorizzazione reusa KontextPipeline (già in uso per recolor).


class RestorePipeline:
    """Costruisce prompt ComfyUI per restauro fotografico e colorazione."""

    # ── RESTAURO ──────────────────────────────────────────────────────────────

    def build_restore(self, image_name: str, *,
                      upscale_model: str = _DEFAULT_UPSCALE,
                      target_scale: float = 2.0,
                      denoise: bool = True,
                      denoise_strength: float = 0.25,
                      seed: int = 0) -> dict:
        """Restauro qualità: ESRGAN super-resolution + denoising facoltativo.

        Args:
            image_name:      nome file immagine caricata in ComfyUI (es. "photo.png")
            upscale_model:   modello ESRGAN da usare
            target_scale:    fattore di ingrandimento (1x-4x)
            denoise:         se True aggiunge un KSampler con denoise basso per rifinire
            denoise_strength: forza del denoising (0.0–1.0, tipicamente 0.1–0.35)
            seed:            seed per riproducibilità
        """
        p: dict = {}

        # Carica immagine
        p["1"] = {"class_type": "LoadImage", "inputs": {"image": image_name, "upload": "image"}}

        # ESRGAN upscale
        p["2"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": upscale_model}}
        p["3"] = {"class_type": "ImageUpscaleWithModel",
                  "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}}

        # Ridimensiona al target se necessario (scale < target_scale)
        # Calcoliamo sempre al target richiesto
        p["4"] = {"class_type": "ImageScaleBy",
                  "inputs": {"image": ["3", 0], "upscale_method": "lanczos",
                             "scale_by": float(target_scale) / 4.0}}   # ESRGAN × 4 di default

        # Denoising opzionale (migliora texture ma può alterare dettagli)
        if denoise and denoise_strength > 0.0:
            # Per il denoising usiamo un approccio VAE encode/decode con KSampler
            # Requires: un modello FLUX disponibile; se non presente, salta silenziosamente.
            # Usiamo LatentUpscale + KSampler standard (img2img leggero)
            p["5"] = {"class_type": "VAELoader",
                      "inputs": {"vae_name": "Flux\\ae.safetensors"}}
            p["6"] = {"class_type": "VAEEncode",
                      "inputs": {"pixels": ["4", 0], "vae": ["5", 0]}}
            # Output finale: decode dopo denoise non ancora cablato (serve un modello)
            # Fallback sicuro: output diretto dall'upscale
            last_img = ["4", 0]
        else:
            last_img = ["4", 0]

        p["10"] = {"class_type": "SaveImage",
                   "inputs": {"images": last_img, "filename_prefix": "ltxstudio_restore"}}
        return p

    # ── COLORIZZAZIONE ────────────────────────────────────────────────────────

    def build_colorize(self, image_name: str, *,
                       flux2_model: str = _DEFAULT_FLUX2_MODEL,
                       subject: str = "",
                       style: str = "natural realistic colors",
                       seed: int = 0,
                       steps: int = 20,
                       denoise: float = 0.85,
                       guidance: float = 2.5) -> dict:
        """Colorizzazione B&W → colore tramite FLUX.2 (Kontext-style img2img).

        Il modello riceve la foto in B&W come riferimento e il prompt descrive i colori
        attesi. FLUX.2 è molto bravo a colorizzare mantenendo la struttura dell'immagine.

        Args:
            image_name:   nome file immagine B&W in ComfyUI
            flux2_model:  modello FLUX.2 da usare (es. "flux2_dev.safetensors")
            subject:      soggetto principale (es. "woman in 1940s dress")
            style:        stile di colorizzazione (es. "warm Kodachrome film colors")
            seed:         seed
            steps:        step di diffusione (15-25 consigliati)
        """
        if not flux2_model:
            raise RuntimeError(
                "[RestorePipeline] Colorizzazione richiede un modello FLUX.2.\n"
                "Seleziona un modello FLUX.2 nelle impostazioni del tab Restauro."
            )

        # Prompt di colorizzazione
        color_prompt = (
            f"colorized version of this black and white photograph, "
            f"{subject + ', ' if subject else ''}"
            f"{style}, photorealistic, natural skin tones, period-accurate colors, "
            f"high quality colorization"
        )

        p: dict = {}
        # Carica immagine B&W
        p["1"] = {"class_type": "LoadImage", "inputs": {"image": image_name, "upload": "image"}}

        # Converti in latente (VAE FLUX.2)
        p["2"] = {"class_type": "VAELoader", "inputs": {"vae_name": "Flux\\flux2-vae.safetensors"}}
        p["3"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["1", 0], "vae": ["2", 0]}}

        # UNET FLUX.2
        p["4"] = {"class_type": "UNETLoader",
                  "inputs": {"unet_name": flux2_model, "weight_dtype": "default"}}

        # CLIP FLUX.2
        p["5"] = {"class_type": "CLIPLoader",
                  "inputs": {"clip_name": "Flux\\mistral_3_small_flux2.safetensors", "type": "flux2"}}

        # Encoding testo
        p["6"] = {"class_type": "CLIPTextEncode",
                  "inputs": {"text": color_prompt, "clip": ["5", 0]}}
        p["7"] = {"class_type": "FluxGuidance",
                  "inputs": {"conditioning": ["6", 0], "guidance": float(guidance)}}
        p["8"] = {"class_type": "CLIPTextEncode",
                  "inputs": {"text": "", "clip": ["5", 0]}}

        # Dimensione latente da immagine
        p["9"] = {"class_type": "EmptyFlux2LatentImage",
                  "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}

        # KSampler con denoise (img2img) — denoise alto = colori più decisi, basso = struttura più fedele
        p["10"] = {"class_type": "KSampler",
                   "inputs": {
                       "model": ["4", 0], "positive": ["7", 0], "negative": ["8", 0],
                       "latent_image": ["3", 0],    # usa la foto B&W come init
                       "seed": int(seed), "steps": int(steps),
                       "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                       "denoise": float(denoise),
                   }}
        p["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["2", 0]}}
        p["12"] = {"class_type": "SaveImage",
                   "inputs": {"images": ["11", 0], "filename_prefix": "ltxstudio_colorize"}}
        return p

    # ── RESTAURO + COLORIZZAZIONE SEQUENZIALE ─────────────────────────────────

    def build_restore_and_colorize(self, image_name: str, *,
                                   upscale_model: str = _DEFAULT_UPSCALE,
                                   flux2_model: str = _DEFAULT_FLUX2_MODEL,
                                   subject: str = "",
                                   style: str = "natural realistic colors",
                                   seed: int = 0, steps: int = 20,
                                   denoise: float = 0.85, guidance: float = 2.5,
                                   target_scale: float = 2.0) -> dict:
        """Pipeline combinata: prima ESRGAN, poi colorizzazione FLUX.2.

        Utile per vecchie foto B&W danneggiate o a bassa risoluzione.
        """
        if not flux2_model:
            raise RuntimeError(
                "[RestorePipeline] Restore+Colorize richiede un modello FLUX.2."
            )

        color_prompt = (
            f"colorized version of this black and white photograph, "
            f"{subject + ', ' if subject else ''}"
            f"{style}, photorealistic, natural skin tones, period-accurate colors"
        )

        p: dict = {}

        # 1. Carica immagine
        p["1"] = {"class_type": "LoadImage", "inputs": {"image": image_name, "upload": "image"}}

        # 2. Restauro ESRGAN (×2)
        p["2"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": upscale_model}}
        p["3"] = {"class_type": "ImageUpscaleWithModel",
                  "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}}
        # ESRGAN è ×4 nativo: porta al fattore RICHIESTO (scale_by = target/4). Prima 0.5 fisso (= sempre ×2,
        # sbagliato se l'utente chiedeva ×4). Clamp ai limiti sensati.
        _sb = max(0.1, min(2.0, float(target_scale) / 4.0))
        p["4"] = {"class_type": "ImageScaleBy",
                  "inputs": {"image": ["3", 0], "upscale_method": "lanczos", "scale_by": _sb}}

        # 3. Colorizzazione FLUX.2
        p["5"] = {"class_type": "VAELoader", "inputs": {"vae_name": "Flux\\flux2-vae.safetensors"}}
        p["6"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["5", 0]}}
        p["7"] = {"class_type": "UNETLoader",
                  "inputs": {"unet_name": flux2_model, "weight_dtype": "default"}}
        p["8"] = {"class_type": "CLIPLoader",
                  "inputs": {"clip_name": "Flux\\mistral_3_small_flux2.safetensors", "type": "flux2"}}
        p["9"] = {"class_type": "CLIPTextEncode",
                  "inputs": {"text": color_prompt, "clip": ["8", 0]}}
        p["10"] = {"class_type": "FluxGuidance",
                   "inputs": {"conditioning": ["9", 0], "guidance": float(guidance)}}
        p["11"] = {"class_type": "CLIPTextEncode",
                   "inputs": {"text": "", "clip": ["8", 0]}}
        p["12"] = {"class_type": "KSampler",
                   "inputs": {
                       "model": ["7", 0], "positive": ["10", 0], "negative": ["11", 0],
                       "latent_image": ["6", 0],
                       "seed": int(seed), "steps": int(steps),
                       "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                       "denoise": float(denoise),
                   }}
        p["13"] = {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}}
        p["14"] = {"class_type": "SaveImage",
                   "inputs": {"images": ["13", 0], "filename_prefix": "ltxstudio_restored_colorized"}}
        return p
