# Pesi per il backend nativo — SONO GIÀ TUTTI SUL DISCO (no download GB)

**Scoperta (2026-06-23)**: tutti i pesi pesanti che servono erano già in `E:\AI\Models` (o `Comfy_Nexus.bak`),
perché ComfyUI li usava. I download da HF erano inutili (e bloccati dal CDN). Restano da risolvere
solo **config piccoli (KB-MB)** o **adattamenti di formato** (single-file ComfyUI ≠ cartella diffusers/HF).

## Pesi trovati in locale
| Modello | Peso locale | Formato | Cosa serve ancora |
|---|---|---|---|
| **Gemma-3-12B** (TE video) | `E:\AI\Models\TextEncoders\gemma3\` | cartella HF completa ✅ | NIENTE — usabile diretto |
| **LTX-2.3 transformer** | `E:\AI\Models\diffusion_models\LTX\ltx-2.3-22b-dev-fp8.safetensors` | single-file ComfyUI | **config 2.3** (il config diffusers scaricato è per LTX-2 base: audio_scale_shift_table 6≠9) |
| LTX-2 componenti (audio_vae/connectors/vocoder/upsampler) | `E:\AI\Models\diffusers\ltx2\` | diffusers ✅ | NIENTE |
| LTX-2 video VAE | `E:\AI\Models\VAE\LTX\LTX2_video_vae.safetensors` | single-file | config vae ✅ (presente) |
| **FLUX.2 transformer** | `E:\AI\Models\diffusion_models\Flux\miraclein_flux2klein_nsfw_v2_fp8.safetensors` | single-file | config transformer flux2 ✅ |
| **FLUX.2 TE (Mistral)** | `E:\AI\Models\TextEncoders\Flux\mistral_3_small_flux2.safetensors` | single-file | **config Mistral pruned** (da FLUX.2-dev, ~KB) |
| FLUX.2 VAE | `E:\AI\Models\VAE\Flux\flux2-vae.safetensors` | single-file | config ✅ |
| **SAM2** | `E:\AI\Models\Segmentation\sam2\sam2_hiera_large.safetensors` (+ `.pt`) | originale facebook/kijai | **lib `sam2` originale** o conversione HF |
| **GroundingDINO** | `E:\AI\Models\Segmentation\grounding-dino\groundingdino_swinb_cogcoor.pth` | originale IDEA-Research | **lib `groundingdino` originale** o config+conversione HF |
| **Ernie** | `E:\AI\Models\diffusion_models\Ernie\ernie-image.safetensors` | single-file | `ErnieImageTransformer2DModel` NON ha from_single_file → serve config/conversione |
| SAM v1 / SAM-HQ | `E:\AI\Models\Segmentation\sam_vit_*.pth`, `Comfy_Nexus.bak\...\sams\sam_hq_vit_l.pth` | originale | lib segment_anything |

## Azioni (config/adattamenti, NON download GB)
1. **Video LTX-2.3**: ottenere/derivare il config transformer per la 2.3 (audio_scale_shift_table 9). Opzioni: cercare repo diffusers LTX-2.3, oppure patchare il config, oppure `ignore_mismatched_sizes` (audio non garantito).
2. **FLUX.2**: scaricare SOLO il config Mistral pruned (`text_encoder/config.json`+tokenizer da black-forest-labs/FLUX.2-dev, gated, ~KB) → `image_flux._assemble_flux2` carica il TE dal single-file locale.
3. **Segmentazione**: `pip install groundingdino-py segment-anything-2` nel venv nativo → riscrivere `image_segment.py` con le lib originali sui pesi locali (come faceva comfyui_segment_anything). NIENTE download pesi.
4. **Ernie**: serve config/conversione del transformer (single-file non caricabile da diffusers as-is).
