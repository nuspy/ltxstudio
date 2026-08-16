# -*- coding: utf-8 -*-
"""Migrazione weights_catalog.json → SCHEMA v2 (registro unico dei pesi).

Cosa fa (idempotente, backup automatico):
  1. aggiunge a OGNI voce `category` (image|video|music|tts|threed|aux) e `arch` — gli assi
     tassonomici che mancavano (prima solo `module` funzionale);
  2. BONIFICA le voci rotte: le 9 non scaricabili ricevono repo_id/url VERIFICATI su HF
     (o `no_remote: true` quando la fonte remota non esiste/non è identificabile — mai inventata);
     il tool_dest con il profilo utente hardcoded diventa `~`; i type con case incoerente
     vengono normalizzati (LoRA/TextEncoders/TTS/Checkpoints) — su NTFS è lo stesso path,
     su macOS/Linux evita la doppia cartella;
  3. aggiunge le voci MANCANTI richieste dai motori (censimento build/extract_models.py):
     stack LTX-2.3 completo (checkpoint fp8, Gemma TE, distilled-LoRA, upscaler, IC-LoRA),
     pack LTX-2.5 split, FLUX single-file, SAM2/DINO/DepthAnything, insightface, ESRGAN,
     SeedVR2, FlashVSR, JoyCaption, wav2vec2 MultiTalk, TTS F5/XTTS, GGUF QuantStack.
     Tutti i repo/filename remoti sono stati VERIFICATI su HuggingFace il 16/08/2026;
  4. `install_role: "base"` marca i pesi che l'installer scarica di default (il resto è
     on-demand); `sensitivity: "adult"` marca i pesi NSFW (il filtro edizione ora ha i dati).

Campi nuovi dello schema v2 (retro-compatibili: i lettori v1 li ignorano):
  category, arch, install_role, sensitivity, remote_filename (nome nel repo ≠ nome locale),
  allow_patterns (subset di uno snapshot), no_remote (voce solo scan/link, niente download).

Uso:  python build/catalog_v2.py [--dry]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "weights_catalog.json"

# ── category+arch per le voci ESISTENTI ──────────────────────────────────────
CAT_ARCH = {
    "flux1-dev": ("image", "flux1"), "flux1-fill": ("image", "flux1"),
    "flux1-kontext": ("image", "flux1"), "flux1-krea": ("image", "flux1"),
    "flux2-dev": ("image", "flux2"), "mistral-flux2-te": ("image", "flux2"),
    "qwen3-8b-klein-te": ("image", "flux2"), "qwen3-4b-klein-te": ("image", "flux2"),
    "zimage-turbo": ("image", "zimage"), "ernie-image": ("image", "ernie"),
    "ernie-ministral": ("image", "ernie"),
    "krea2_turbo": ("image", "krea2"), "krea2_raw": ("image", "krea2"),
    "pulid-flux": ("image", "flux1"), "flux-vae-ae": ("image", "flux1"),
    "flux2-vae": ("image", "flux2"),
    "ltx23": ("video", "ltx23"),
    "ltx-iclora-canny": ("video", "ltx23"), "ltx-iclora-depth": ("video", "ltx23"),
    "ltx-iclora-pose": ("video", "ltx23"),
    "lingbot_dense": ("video", "lingbot"), "lingbot_moe": ("video", "lingbot"),
    "lingbot_moe_uint4": ("video", "lingbot"), "lingbot_dense_fp8": ("video", "lingbot"),
    "lingbot_rewriter_lora": ("video", "lingbot"), "lingbot_rewriter_base": ("video", "qwen"),
    "krea_rt": ("video", "krea_rt"), "wan21_14b_diffusers": ("video", "wan"),
    "wan21_13b_diffusers": ("video", "wan"), "wan21_13b_original": ("video", "wan"),
    "self_forcing_dmd": ("video", "self_forcing"),
    "da2": ("threed", "worldgen"), "svdq-fp4-fluxfill": ("threed", "worldgen"),
    "worldgen-lora": ("threed", "worldgen"),
    "trellis2-4b": ("threed", "trellis"), "dinov3-vitl16": ("threed", "trellis"),
    "birefnet": ("threed", "trellis"), "skintoken-grpo": ("threed", "skintoken"),
    "mmaudio": ("music", "mmaudio"), "acestep-3.5b": ("music", "acestep"),
    "heartmula-gen": ("music", "heartmula"), "heartmula-3b": ("music", "heartmula"),
    "heartcodec": ("music", "heartmula"),
    "voice-fish": ("tts", "fish"), "voice-vibevoice": ("tts", "vibevoice"),
    "voice-cosyvoice2": ("tts", "cosyvoice"),
    "gfpgan-14": ("aux", "gfpgan"), "codeformer": ("aux", "codeformer"),
}

# ── bonifiche puntuali delle voci esistenti (id → update) ────────────────────
PATCH = {
    # verificato 16/08: il repo Lightricks/LTX-2.3 NON è più gated
    "ltx23": {"gated": False,
              "note": "snapshot diffusers (40GB) — i motori NATIVI usano i file singoli "
                      "(voci ltx23-*); tenerlo solo per pipeline diffusers pure"},
    # dest allineato al motore: i file vivono alla RADICE del type (family vuota = nessuna sottocartella)
    "gfpgan-14": {"family": ""},
    "codeformer": {"family": ""},
    "pulid-flux": {"family": ""},
    # le 3 IC-LoRA di controllo: repo ufficiali Lightricks VERIFICATI (canny: filename esatto)
    "ltx-iclora-canny": {"repo_id": "Lightricks/LTX-2-19b-IC-LoRA-Canny-Control",
                         "url": "https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Canny-Control",
                         "size_mb": 624},
    "ltx-iclora-depth": {"repo_id": "Lightricks/LTX-2-19b-IC-LoRA-Depth-Control",
                         "url": "https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Depth-Control"},
    "ltx-iclora-pose": {"repo_id": "Lightricks/LTX-2-19b-IC-LoRA-Pose-Control",
                        "url": "https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Pose-Control"},
    # VAE FLUX.1: sta nel repo gated BFL (file ae.safetensors alla radice)
    "flux-vae-ae": {"repo_id": "black-forest-labs/FLUX.1-dev", "gated": True,
                    "url": "https://huggingface.co/black-forest-labs/FLUX.1-dev"},
    # VAE FLUX.2: mirror Comfy-Org NON gated, path remoto verificato
    "flux2-vae": {"repo_id": "Comfy-Org/flux2-dev",
                  "remote_filename": "split_files/vae/flux2-vae.safetensors",
                  "url": "https://huggingface.co/Comfy-Org/flux2-dev", "size_mb": 321},
    # fonte remota NON identificata → onesto: niente download, solo scan/link (il CRUD potrà completarle)
    "ernie-ministral": {"no_remote": True,
                        "note": "fonte remota non identificata (16/08) — file presente localmente"},
    "skintoken-grpo": {"no_remote": True,
                       "note": "fonte remota non identificata (16/08) — pesi installati da install_skintokens"},
    # MAI il profilo utente hardcoded nel catalogo distribuito
    "acestep-3.5b": {"tool_dest": "~/.cache/ace-step/checkpoints"},
    # normalizzazione vocabolario type (case NTFS-neutra, macOS/Linux-safe) + dest=motore
    "voice-cosyvoice2": {"type": "TTS", "family": ""},
    "voice-fish": {"type": "TTS", "family": ""},
    "voice-vibevoice": {"type": "TTS", "family": ""},
    "lingbot_rewriter_lora": {"type": "LoRA"},
    "worldgen-lora": {"type": "LoRA"},
    "mistral-flux2-te": {"type": "TextEncoders"},
    "lingbot_rewriter_base": {"type": "TextEncoders"},
    "self_forcing_dmd": {"type": "Checkpoints"},
    "wan21_13b_original": {"type": "Checkpoints"},
}

# ── voci NUOVE (repo/filename verificati su HF il 16/08/2026, salvo no_remote) ──
def N(id, module, label, *, category, arch, kind="file", repo=None, filename=None,
      remote=None, type=None, family="", tool_dest=None, gated=False, gb=0.0,
      scan=None, deps=None, role=None, sens=None, no_remote=False, note=None,
      allow=None, used_by=None):
    e = {"id": id, "module": module, "label": label, "kind": kind,
         "type": type or "misc", "family": family, "gated": gated,
         "approx_gb": gb, "scan_names": scan or ([filename] if filename else []),
         "category": category, "arch": arch}
    if repo:
        e["repo_id"] = repo
        e["url"] = "https://huggingface.co/" + repo
    if filename:
        e["filename"] = filename
    if remote:
        e["remote_filename"] = remote
    if tool_dest:
        e["tool_dest"] = tool_dest
    if deps:
        e["deps"] = deps
    if role:
        e["install_role"] = role
    if sens:
        e["sensitivity"] = sens
    if no_remote:
        e["no_remote"] = True
    if note:
        e["note"] = note
    if allow:
        e["allow_patterns"] = allow
    if used_by:
        e["used_by"] = used_by
    return e


NEW = [
    # ───────── LTX-2.3: lo stack che i motori usano DAVVERO (prima: zero voci) ─────────
    N("ltx23-dev-fp8", "Video", "LTX-2.3 22B dev fp8 (checkpoint nativo)",
      category="video", arch="ltx23", repo="Lightricks/LTX-2.3-fp8",
      filename="ltx-2.3-22b-dev-fp8.safetensors", type="diffusion_models", family="LTX",
      gb=27.2, scan=["ltx-2.3-22b-dev-fp8"]),
    N("ltx23-distilled-fp8", "Video", "LTX-2.3 22B distilled fp8 (veloce, 8 step)",
      category="video", arch="ltx23", repo="Lightricks/LTX-2.3-fp8",
      filename="ltx-2.3-22b-distilled-fp8.safetensors", type="diffusion_models", family="LTX",
      gb=27.5, role="base", scan=["ltx-2.3-22b-distilled-fp8"]),
    N("ltx23-distilled-lora-384", "Video", "LTX-2.3 distilled-LoRA 384 (OBBLIGATORIA coi dev)",
      category="video", arch="ltx23", repo="Lightricks/LTX-2.3",
      filename="ltx-2.3-22b-distilled-lora-384-1.1.safetensors", type="LoRA", family="LTX-2",
      gb=7.1, role="base", scan=["distilled-lora-384-1.1"]),
    N("ltx23-spatial-upscaler", "Video", "LTX-2.3 spatial upscaler x2 1.1",
      category="video", arch="ltx23", repo="Lightricks/LTX-2.3",
      filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors", type="latent_upscale_models",
      gb=0.93, role="base", scan=["spatial-upscaler-x2-1.1"]),
    N("ltx23-temporal-upscaler", "Video", "LTX-2.3 temporal upscaler x2 1.0",
      category="video", arch="ltx23", repo="Lightricks/LTX-2.3",
      filename="ltx-2.3-temporal-upscaler-x2-1.0.safetensors", type="latent_upscale_models",
      gb=0.25, scan=["temporal-upscaler-x2"]),
    N("ltx23-gemma-te", "Video", "Gemma-3 12B TE (formato ltx-core, cartella gemma-3-ltx)",
      category="video", arch="ltx23", kind="snapshot", repo="Lightricks/LTX-2",
      tool_dest=r"E:\AI\Models\TextEncoders\gemma-3-ltx",
      allow=["text_encoder/*", "tokenizer*", "*.json"], gb=24.0, role="base",
      scan=["gemma-3-ltx"], note="subset text_encoder/ del repo LTX-2 (formato ltx-core)"),
    N("ltx23-gguf-q4-distilled", "Video", "LTX-2.3 distilled GGUF Q4_K_M (GPU piccole)",
      category="video", arch="ltx23", repo="QuantStack/LTX-2.3-GGUF",
      remote="LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
      filename="ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf",
      type="diffusion_models", family="LTX", gb=16.6, scan=["distilled-1.1-Q4_K_M"]),
    # IC-LoRA effetto (module Video → cap.video) e controllo (ControlNet → cap.controlnet)
    N("ltx-iclora-detailer", "Video", "IC-LoRA Detailer", category="video", arch="ltx23",
      repo="Lightricks/LTX-2-19b-IC-LoRA-Detailer",
      filename="ltx-2-19b-ic-lora-detailer.safetensors", type="LoRA", family="LTX-2", gb=2.4),
    N("ltx-iclora-union", "ControlNet", "IC-LoRA Union Control ref0.5", category="video",
      arch="ltx23", repo="Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
      filename="ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
      type="LoRA", family="LTX-2", gb=0.66),
    N("ltx-iclora-motion-track", "ControlNet", "IC-LoRA Motion-Track ref0.5", category="video",
      arch="ltx23", repo="Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
      filename="ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
      type="LoRA", family="LTX-2", gb=0.33),
    N("ltx-iclora-outpaint", "Video", "IC-LoRA In/Outpainting", category="video", arch="ltx23",
      repo="Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting", gated=True,
      remote="ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
      filename="ltx-2.3-22b-ic-lora-outpaint.safetensors", type="LoRA", family="LTX-2", gb=1.3),
    N("ltx-iclora-refocus", "Video", "IC-LoRA ReFocus", category="video", arch="ltx23",
      repo="oumoumad/LTX-2.3-22b-IC-LoRA-ReFocus",
      filename="ltx-2.3-22b-ic-lora-refocus.safetensors", type="LoRA", family="LTX-2", gb=1.3),
    N("ltx-iclora-hdr", "Video", "IC-LoRA HDR 0.9", category="video", arch="ltx23",
      repo="Lightricks/LTX-2.3-22b-IC-LoRA-HDR", gated=True,
      filename="ltx-2.3-22b-ic-lora-hdr-0.9.safetensors", type="LoRA", family="LTX-2", gb=0.33),
    N("ltx-iclora-lipdub", "Video", "IC-LoRA DubIt/LipDub 0.9", category="video", arch="ltx23",
      repo="Lightricks/LTX-2.3-22b-IC-LoRA-DubIt", gated=True,
      remote="ltx-2.3-22b-ic-lora-dubit-0.9.safetensors",
      filename="ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors", type="LoRA", family="LTX-2", gb=2.5),
    N("ltx-iclora-motiondeblur", "Video", "IC-LoRA Deblur 0.9", category="video", arch="ltx23",
      repo="Lightricks/LTX-2.3-22b-IC-LoRA-Deblur", gated=True,
      remote="ltx-2.3-22b-ic-lora-deblur-0.9.safetensors",
      filename="ltx-2.3-22b-ic-lora-motiondeblur.safetensors", type="LoRA", family="LTX-2", gb=0.9),
    N("ltx-iclora-uncompress", "Video", "IC-LoRA Uncompress (fonte remota ignota)",
      category="video", arch="ltx23", filename="ltx-2.3-22b-ic-lora-uncompress.safetensors",
      type="LoRA", family="LTX-2", gb=1.0, no_remote=True),
    N("ltx-iclora-deinterlace", "Video", "IC-LoRA Deinterlace (fonte remota ignota)",
      category="video", arch="ltx23", filename="ltx-2.3-22b-ic-lora-deinterlace.safetensors",
      type="LoRA", family="LTX-2", gb=1.0, no_remote=True),
    N("ltx-iclora-restyle", "Video", "IC-LoRA ReStyle (community)", category="video",
      arch="ltx23", filename="852654_LTX2.3-22B_ReStyle_IC-LoRA_8000_v0.1.safetensors",
      type="LoRA", family="LTX-2", gb=0.8, no_remote=True),
    N("ltx23-transformer-only-fp8scaled", "Video",
      "LTX-2.3 distilled transformer-only fp8_scaled (comfy, fonte ignota)",
      category="video", arch="ltx23",
      filename="ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors",
      type="diffusion_models", family="LTXVideo", gb=21.0, no_remote=True,
      note="su disco in LTXVideo\\v2\\ — usato da extend_pipeline/ltx2_service"),
    N("gemma3-12b-fp8-comfy", "Video", "Gemma-3 12B it fp8_scaled (single-file comfy)",
      category="video", arch="ltx23", repo="Comfy-Org/ltx-2",
      remote="split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
      filename="gemma_3_12B_it_fp8_scaled.safetensors", type="TextEncoders", gb=12.3),
    N("gemma-abliterated", "Video", "Gemma-3 12B abliterated (TE NSFW)", category="video",
      arch="ltx23", repo="FusionCow/Gemma-3-12b-Abliterated-LTX2",
      remote="gemma_ablit_fixed_bf16.safetensors",
      filename="gemma_3_12B_it_abliterated.safetensors", type="TextEncoders",
      gb=21.9, sens="adult", role="base", scan=["abliterated"]),
    N("ltx23-eros", "Video", "LTX-2.3 10Eros (checkpoint NSFW, fonte community)",
      category="video", arch="ltx23", filename="ltx2310eros_v1.safetensors",
      type="diffusion_models", family="LTX", gb=27.0, sens="adult", no_remote=True),

    # ───────── LTX-2.5: il pack split (repo GATED Lightricks/LTX-2.5) ─────────
    N("ltx25-transformer-nvfp4", "Video", "LTX-2.5 22B distilled transformer nvfp4",
      category="video", arch="ltx25", repo="Lightricks/LTX-2.5", gated=True,
      remote="diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
      filename="ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
      type="diffusion_models", family="LTX", gb=17.4),
    N("ltx25-transformer-int8", "Video", "LTX-2.5 22B distilled transformer int8 (comfy)",
      category="video", arch="ltx25", repo="Lightricks/LTX-2.5", gated=True,
      remote="diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
      filename="ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
      type="diffusion_models", family="LTX", gb=20.0),
    N("ltx25-te-bf16", "Video", "LTX-2.5 Gemma-4 TE bf16", category="video", arch="ltx25",
      repo="Lightricks/LTX-2.5", gated=True,
      remote="text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
      filename="gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", type="text_encoders", gb=24.5),
    N("ltx25-te-nvfp4-local", "Video", "LTX-2.5 Gemma-4 TE nvfp4-torchao (derivato localmente)",
      category="video", arch="ltx25",
      filename="gemma4-12b-with-proj-nvfp4-torchao.safetensors", type="text_encoders",
      gb=6.9, no_remote=True, note="conversione locale del TE bf16 (6.9GB vs 24.5)"),
    N("ltx25-video-vae-conv", "Video", "LTX-2.5 video VAE Conv", category="video", arch="ltx25",
      repo="Lightricks/LTX-2.5", gated=True,
      remote="vae/ltx-2.5-video-vae-conv-bf16.safetensors",
      filename="ltx-2.5-video-vae-conv-bf16.safetensors", type="vae", gb=1.36),
    N("ltx25-audio-vae", "Video", "LTX-2.5 audio VAE (+vocoder)", category="video", arch="ltx25",
      repo="Lightricks/LTX-2.5", gated=True,
      remote="vae/ltx-2.5-audio-vae-bf16.safetensors",
      filename="ltx-2.5-audio-vae-bf16.safetensors", type="vae", gb=0.34),
    N("ltx25-diffusion-decoder", "Video", "LTX-2.5 video VAE (diffusion decoder)",
      category="video", arch="ltx25", repo="Lightricks/LTX-2.5", gated=True,
      remote="vae/ltx-2.5-video-vae-bf16.safetensors",
      filename="ltx-2.5-video-vae-bf16.safetensors", type="vae", gb=1.37),
    N("ltx25-spatial-upscaler", "Video", "LTX-2.5 latent spatial upscaler x2",
      category="video", arch="ltx25", repo="Lightricks/LTX-2.5", gated=True,
      remote="latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
      filename="ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
      type="latent_upscale_models", gb=0.93),
    N("ltx25-temporal-upscaler", "Video", "LTX-2.5 latent temporal upscaler x2",
      category="video", arch="ltx25", repo="Lightricks/LTX-2.5", gated=True,
      remote="latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
      filename="ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
      type="latent_upscale_models", gb=0.25),
    N("ltx25-distilled-lora-450", "Video", "LTX-2.5 distilled-LoRA 450", category="video",
      arch="ltx25", repo="Lightricks/LTX-2.5", gated=True,
      remote="loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
      filename="ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
      type="LoRA", family="LTX-2.5", gb=8.3),
    N("ltx25-duration-head", "Video", "LTX-2.5 duration head", category="video", arch="ltx25",
      repo="Lightricks/LTX-2.5", gated=True,
      remote="model_patches/ltx-2.5-duration-head-bf16.safetensors",
      filename="ltx-2.5-duration-head-bf16.safetensors", type="model_patches", gb=0.01),
    N("ltx25-iclora-pixel-upscaler", "Video", "LTX-2.5 IC-LoRA Pixel Spatial Upscaler",
      category="video", arch="ltx25",
      repo="Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler", gated=True,
      filename="ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
      type="LoRA", family="LTX-2.5", gb=1.0,
      note="filename remoto da verificare col resolver (repo verificato)"),

    # ───────── FLUX single-file (quelli che il backend immagini carica DAVVERO) ─────────
    N("flux1-dev-single", "Immagini", "FLUX.1 dev (single-file nativo)", category="image",
      arch="flux1", repo="black-forest-labs/FLUX.1-dev", gated=True,
      remote="flux1-dev.safetensors", filename="FLUX_Dev.safetensors",
      type="diffusion_models", family="Flux", gb=23.0, scan=["FLUX_Dev", "flux1-dev"]),
    N("flux1-fill-single", "Inpaint", "FLUX.1 Fill dev (single-file nativo)", category="image",
      arch="flux1", repo="black-forest-labs/FLUX.1-Fill-dev", gated=True,
      remote="flux1-fill-dev.safetensors", filename="FLUX_DEV_Fill.safetensors",
      type="diffusion_models", family="Flux", gb=23.0, scan=["FLUX_DEV_Fill", "flux1-fill"]),
    N("flux1-kontext-single", "Editing", "FLUX.1 Kontext dev (single-file nativo)",
      category="image", arch="flux1", repo="black-forest-labs/FLUX.1-Kontext-dev", gated=True,
      remote="flux1-kontext-dev.safetensors", filename="FLUX_Kontext_Dev.safetensors",
      type="diffusion_models", family="Flux", gb=23.0, scan=["FLUX_Kontext", "kontext-dev"]),
    N("flux1-krea-single", "Immagini", "FLUX.1 Krea dev (single-file nativo)", category="image",
      arch="flux1", repo="black-forest-labs/FLUX.1-Krea-dev",
      remote="flux1-krea-dev.safetensors", filename="FLUX_Krea_Dev.safetensors",
      type="diffusion_models", family="Flux", gb=23.0, scan=["FLUX_Krea", "flux1-krea"]),
    N("flux2-dev-fp8mixed", "Immagini", "FLUX.2 dev fp8mixed (single-file comfy)",
      category="image", arch="flux2", repo="Comfy-Org/flux2-dev",
      remote="split_files/diffusion_models/flux2_dev_fp8mixed.safetensors",
      filename="flux2_dev_fp8mixed.safetensors", type="diffusion_models", family="Flux",
      gb=33.0, scan=["flux2_dev_fp8", "FLUX_2_Dev"]),
    N("mistral-te-fp8-single", "Immagini", "Mistral-3 TE FLUX.2 fp8 (single-file)",
      category="image", arch="flux2", repo="Comfy-Org/flux2-dev",
      remote="split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors",
      filename="mistral_3_small_flux2.safetensors", type="TextEncoders", family="Flux",
      gb=16.8, scan=["mistral_3_small_flux2"]),
    N("t5xxl-fp16", "TextEncoder", "T5-XXL fp16 (FLUX.1 nativo)", category="image",
      arch="flux1", repo="comfyanonymous/flux_text_encoders",
      remote="t5xxl_fp16.safetensors", filename="t5xxl_fp16.safetensors",
      type="TextEncoders", gb=9.2),
    N("long-clip-hi-dream", "TextEncoder", "Long-CLIP L (fonte remota ignota)",
      category="image", arch="flux1", filename="long_clip_l_hi_dream.safetensors",
      type="TextEncoders", gb=0.9, no_remote=True),

    # ───────── Segmentazione / ControlNet / Face / Upscale (aux) ─────────
    N("sam2-hiera-large", "Segmentazione", "SAM2 Hiera Large (tracking/maschere)",
      category="aux", arch="sam", kind="snapshot", repo="facebook/sam2-hiera-large",
      type="Segmentation", family="hf", gb=0.9, scan=["sam2-hiera-large"]),
    N("grounding-dino-base", "Segmentazione", "GroundingDINO base (detect da testo)",
      category="aux", arch="dino", kind="snapshot", repo="IDEA-Research/grounding-dino-base",
      type="Segmentation", family="hf", gb=0.9, scan=["grounding-dino-base"]),
    N("depth-anything-v2", "ControlNet", "DepthAnything V2 Large (depth control)",
      category="aux", arch="depthanything", kind="snapshot",
      repo="depth-anything/Depth-Anything-V2-Large-hf",
      tool_dest=r"E:\AI\Models\Segmentation\hf\depth-anything-v2-large",
      gb=1.3, scan=["Depth-Anything-V2-Large", "depth-anything-v2-large"]),
    N("inswapper-128", "Face", "InsightFace inswapper (face-swap)", category="aux",
      arch="insightface", repo="ezioruan/inswapper_128.onnx",
      remote="inswapper_128.onnx", filename="inswapper_128.onnx",
      tool_dest=r"E:\AI\Models\insightface", gb=0.5),
    N("antelopev2", "Face", "InsightFace antelopev2 (detector)",
      category="aux", arch="insightface", kind="snapshot", repo="DIAMONIK7777/antelopev2",
      tool_dest=r"E:\AI\Models\insightface\models\antelopev2", gb=0.4,
      note="mirror HF (i 5 onnx alla radice); dest = insightface/models/antelopev2 dove "
           "FaceAnalysis li cerca"),
    N("gfpgan-13", "FaceRestore", "GFPGAN v1.3 (restauro volti)", category="aux",
      arch="gfpgan", filename="GFPGANv1.3.pth", type="facerestore_models", gb=0.3,
      note="url GitHub (non-HF): gestito dal downloader unico FASE B"),
    N("ultrasharp-v2", "Upscale", "4x-UltraSharpV2 (ESRGAN)", category="aux", arch="esrgan",
      repo="Kim2091/UltraSharpV2", remote="4x-UltraSharpV2.safetensors",
      filename="4x-UltraSharpV2.safetensors", type="Upscale", gb=0.14),
    N("nomos8k-schat-l", "Upscale", "4xNomos8kSCHAT-L (ESRGAN)", category="aux", arch="esrgan",
      repo="Phips/4xNomos8kSCHAT-L", remote="4xNomos8kSCHAT-L.pth",
      filename="4xNomos8kSCHAT-L.pth", type="Upscale", gb=0.3),

    # ───────── Upscaler video / caption / lip-sync ─────────
    N("seedvr2-3b-fp8", "Video", "SeedVR2 3B fp8 (super-risoluzione video)", category="video",
      arch="seedvr2", repo="numz/SeedVR2_comfyUI",
      remote="seedvr2_ema_3b_fp8_e4m3fn.safetensors",
      filename="seedvr2_ema_3b_fp8_e4m3fn.safetensors", type="SeedVR2", gb=3.2,
      deps=["seedvr2-vae"]),
    N("seedvr2-vae", "Video", "SeedVR2 VAE", category="video", arch="seedvr2",
      repo="numz/SeedVR2_comfyUI", remote="ema_vae_fp16.safetensors",
      filename="ema_vae_fp16.safetensors", type="SeedVR2", gb=0.5),
    N("flashvsr-v11", "Video", "FlashVSR v1.1 (super-risoluzione video)", category="video",
      arch="flashvsr", kind="snapshot", repo="JunhaoZhuang/FlashVSR-v1.1",
      type="FlashVSR", gb=6.5, scan=["FlashVSR-v1.1"]),
    N("joycaption-nf4", "LoRA", "JoyCaption beta-one nf4 (caption dataset)", category="aux",
      arch="joycaption", kind="snapshot",
      repo="John6666/llama-joycaption-beta-one-hf-llava-nf4",
      tool_dest=r"E:\AI\Models\JoyCaption", gb=5.6,
      used_by=["cap.lora_image_train", "cap.lora_video_train"]),
    N("wav2vec2-multitalk", "Video", "wav2vec2 cinese (MultiTalk lip-sync)", category="video",
      arch="wan", kind="snapshot", repo="TencentGameMate/chinese-wav2vec2-base",
      type="audio", family="Wan", gb=0.8, used_by=["cap.video"]),

    # ───────── TTS locali (oggi auto-download nella cache HF al primo uso) ─────────
    N("f5-tts", "Voce (clone)", "F5-TTS v1 Base", category="tts", arch="f5",
      kind="snapshot", repo="SWivid/F5-TTS", type="TTS", gb=1.4, scan=["F5-TTS", "F5TTS"]),
    N("xtts-v2", "Voce (clone)", "XTTS v2 (Coqui)", category="tts", arch="xtts",
      kind="snapshot", repo="coqui/XTTS-v2", type="TTS", gb=2.0, scan=["XTTS-v2", "xtts"]),
]

# url GitHub per voci file non-HF (il downloader FASE B le gestirà)
GH_URLS = {
    "gfpgan-13": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
}

DEFAULT_CAT_BY_MODULE = {
    "Immagini": ("image", ""), "Inpaint": ("image", ""), "Editing": ("image", ""),
    "Video": ("video", ""), "ControlNet": ("video", ""),
    "Modello 3D (TRELLIS)": ("threed", ""), "Rig 3D": ("threed", ""),
    "Mondo 3D (WorldGen)": ("threed", ""),
    "Musica (HeartMuLa)": ("music", ""), "Musica (ACE-Step)": ("music", ""),
    "Musica (MMAudio)": ("music", ""), "Voce (clone)": ("tts", ""),
    "FaceRestore": ("aux", ""), "Face": ("aux", ""), "Upscale": ("aux", ""),
    "Segmentazione": ("aux", ""), "TextEncoder": ("aux", ""), "VAE": ("aux", ""),
    "LoRA": ("aux", ""),
}


def main() -> int:
    dry = "--dry" in sys.argv
    data = json.loads(CATALOG.read_text("utf-8"))
    weights = data.get("weights", [])
    by_id = {e["id"]: e for e in weights}

    # 1) category/arch + patch sulle esistenti
    for e in weights:
        cat, arch = CAT_ARCH.get(e["id"], (None, None))
        if cat is None:
            cat, arch = DEFAULT_CAT_BY_MODULE.get(e.get("module", ""), ("aux", ""))
        e.setdefault("category", cat)
        e.setdefault("arch", arch or "generic")
        for k, v in PATCH.get(e["id"], {}).items():
            e[k] = v

    # 2) nuove voci (idempotente: non duplica id già presenti)
    added = 0
    for n in NEW:
        if n["id"] in by_id:
            continue
        if n["id"] in GH_URLS:
            n["url"] = GH_URLS[n["id"]]
        weights.append(n)
        by_id[n["id"]] = n
        added += 1

    data["schema"] = 2
    data["_comment_v2"] = ("SCHEMA v2 — campi nuovi: category (image|video|music|tts|threed|aux), "
                           "arch, install_role (base|on_demand), sensitivity (adult), "
                           "remote_filename (nome nel repo se ≠ dal locale), allow_patterns "
                           "(subset snapshot), no_remote (voce solo scan/link). Fonti remote "
                           "verificate su HF il 2026-08-16. Gestione: tools/catalog_manager.")
    data["weights"] = weights

    n_tot = len(weights)
    n_dl = sum(1 for e in weights if e.get("repo_id") or e.get("url"))
    n_nr = sum(1 for e in weights if e.get("no_remote"))
    print(f"voci: {n_tot} (aggiunte {added}) — scaricabili {n_dl}, no_remote {n_nr}")
    if dry:
        return 0
    shutil.copy2(CATALOG, CATALOG.with_suffix(".json.v1.bak"))
    CATALOG.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")
    json.loads(CATALOG.read_text("utf-8"))
    print(f"scritto {CATALOG} (backup .v1.bak)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
