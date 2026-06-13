"""Editing per riferimento stile MagicQuill V2: FLUX **Kontext** (o FLUX.2) in ComfyUI.

Sostituisce il vecchio "integra" (FLUX img2img + noise-mask a denoise basso, che PRESERVAVA
l'incollato → "incollava la foto originale"). Qui l'immagine composita (scena + oggetto
incollato) è data come RIFERIMENTO Kontext + prompt: il modello **rigenera/armonizza** la
regione (luce, ombre, prospettiva) invece di incollare. Stesso schema per il CAMBIO COLORE
(prompt "ricolora X in …", mantieni texture/luce).

MagicQuill usa FLUX.1-Kontext + LoRA dedicate (puzzle/color/...). Qui:
- engine="kontext": FLUX.1-Kontext-dev + (opz.) puzzle_lora come LoRA normale.
- engine="flux2":   FLUX.2 (componenti flux2) con lo stesso ReferenceLatent (sperimentale).
ComfyUI espone i nodi nativi: FluxKontextImageScale + ReferenceLatent.
"""
from __future__ import annotations

from core.gen_pipeline import is_flux2, FLUX2_VAE, FLUX2_CLIP

KONTEXT_MODEL = "Flux\\FLUX_Kontext_Dev.safetensors"
FLUX1_DEFAULTS = {"t5": "t5xxl_fp16.safetensors", "clip_l": "long_clip_l_hi_dream.safetensors",
                  "vae": "Flux\\ae.safetensors"}


class KontextPipeline:
    def __init__(self, flux1: dict | None = None):
        self.fx = dict(FLUX1_DEFAULTS)
        if flux1:
            self.fx.update({k: v for k, v in flux1.items() if v})
        self._text_encoders: list[str] = []      # encoder testo installati (FLUX.2 per-variante)

    def set_inventory(self, text_encoders: list[str] | None = None) -> None:
        """Inventario encoder live (per scegliere l'encoder FLUX.2 corretto per-variante)."""
        if text_encoders is not None:
            self._text_encoders = list(text_encoders)

    def build(self, image_file: str, prompt: str, *, engine: str = "kontext",
              model: str | None = None, negative: str = "", steps: int = 20,
              guidance: float = 2.5, seed: int = 0,
              aux_lora: str | None = None, aux_strength: float = 1.0) -> dict:
        unet = model or KONTEXT_MODEL
        # flux2-ness DERIVATA DAL MODELLO (non dal flag engine): così VAE/encoder/latent combaciano
        # SEMPRE col modello caricato. Selezionare engine FLUX.2 con un modello FLUX.1 (o il
        # fallback Kontext) dava `img_in` 512-vs-64 (VAE flux2 → modello flux1). Ora impossibile.
        flux2 = is_flux2(unet)
        p: dict = {}
        p["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}
        mref = ["1", 0]
        if aux_lora:                                  # es. puzzle_lora per l'integrazione
            p["20"] = {"class_type": "LoraLoaderModelOnly",
                       "inputs": {"model": mref, "lora_name": aux_lora, "strength_model": float(aux_strength)}}
            mref = ["20", 0]
        if flux2:
            from core.flux2_encoder import resolve_encoder
            encoder, err = resolve_encoder(unet, self._text_encoders, default=FLUX2_CLIP)
            if err:
                raise RuntimeError("[ERR] integra FLUX.2: " + err)
            p["2"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": encoder, "type": "flux2"}}
            p["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": FLUX2_VAE}}
        else:
            p["2"] = {"class_type": "DualCLIPLoader",
                      "inputs": {"clip_name1": self.fx["t5"], "clip_name2": self.fx["clip_l"], "type": "flux"}}
            p["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": self.fx["vae"]}}
        # immagine di riferimento → scala Kontext → encode
        p["7"] = {"class_type": "LoadImage", "inputs": {"image": image_file}}
        p["8"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": ["7", 0]}}
        p["9"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["8", 0], "vae": ["3", 0]}}
        # condizionamento: prompt + riferimento iniettato (ReferenceLatent)
        p["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt or "", "clip": ["2", 0]}}
        p["5"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": float(guidance)}}
        p["6"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["5", 0], "latent": ["9", 0]}}
        p["10"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative or "", "clip": ["2", 0]}}
        p["11"] = {"class_type": "KSampler",
                   "inputs": {"model": mref, "positive": ["6", 0], "negative": ["10", 0],
                              "latent_image": ["9", 0], "seed": int(seed), "steps": int(steps),
                              "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
        p["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}}
        p["13"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": "ltxstudio_kontext"}}
        return p

    def integrate(self, image_file: str, prompt: str, **kw) -> dict:
        """Integra l'oggetto incollato nella scena (armonizza luce/prospettiva)."""
        base = ("Seamlessly integrate the added object into the scene: match the lighting, "
                "shadows, color and perspective so it looks naturally part of the photo.")
        full = f"{base} {prompt}".strip() if prompt else base
        return self.build(image_file, full, **kw)

    def recolor(self, image_file: str, target: str, subject: str = "", **kw) -> dict:
        """Cambia colore (prompt-based): ricolora il soggetto mantenendo texture e luce."""
        what = subject.strip() or "the indicated subject"
        prompt = (f"Recolor {what} to {target}. Keep the original texture, material, "
                  f"lighting and shadows; change only the color.")
        return self.build(image_file, prompt, **kw)
