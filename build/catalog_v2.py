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
      allow=None, used_by=None, convert=None, sha256=None, url=None):
    e = {"id": id, "module": module, "label": label, "kind": kind,
         "type": type or "misc", "family": family, "gated": gated,
         "approx_gb": gb, "scan_names": scan or ([filename] if filename else []),
         "category": category, "arch": arch}
    if repo:
        e["repo_id"] = repo
        e["url"] = "https://huggingface.co/" + repo
    elif url:                                       # fonte diretta (es. Civitai: chiave API nel tab Pesi)
        e["url"] = url
    if filename:
        e["filename"] = filename
    if remote:
        e["remote_filename"] = remote
    if tool_dest:
        e["tool_dest"] = tool_dest
    if sha256:                                      # verificato da core/downloader.verify_file dopo il download
        e["sha256"] = sha256
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
    if convert:                                   # file ComfyUI → convertito in nativo e cancellato
        e["convert"] = convert                    # (core/native_convert: `filename` è il NATIVO)
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

    # ───────── LTX-2.5: pack split. DEFAULT = mirror UNGATE dummy9996/LTX-2.5-22b-ungate
    # (file BYTE-IDENTICI all'ufficiale, verificato 16/08; scelta utente: niente attrito
    # token/licenza). Restano sul repo GATED Lightricks solo nvfp4 e diffusion-decoder
    # (assenti dal mirror). Ufficiale: Lightricks/LTX-2.5. ─────────
    N("ltx25-transformer-nvfp4", "Video", "LTX-2.5 22B distilled transformer nvfp4",
      category="video", arch="ltx25", repo="Lightricks/LTX-2.5", gated=True,
      remote="diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
      filename="ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
      type="diffusion_models", family="LTX", gb=17.4,
      note="solo repo ufficiale gated (assente dal mirror ungate)"),
    N("ltx25-transformer-int8", "Video", "LTX-2.5 22B distilled transformer int8 (comfy)",
      category="video", arch="ltx25", repo="dummy9996/LTX-2.5-22b-ungate",
      remote="ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
      filename="ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
      type="diffusion_models", family="LTX", gb=20.0,
      note="mirror ungate (byte-identico); ufficiale gated: Lightricks/LTX-2.5"),
    N("ltx25-te-bf16", "Video", "LTX-2.5 Gemma-4 TE bf16", category="video", arch="ltx25",
      repo="dummy9996/LTX-2.5-22b-ungate",
      remote="gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
      filename="gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", type="text_encoders", gb=24.5,
      note="mirror ungate (byte-identico); il default di DOWNLOAD del TE 2.5 è il GGUF Q5_K_M"),
    N("ltx25-te-nvfp4-local", "Video", "LTX-2.5 Gemma-4 TE nvfp4-torchao (derivato localmente)",
      category="video", arch="ltx25",
      filename="gemma4-12b-with-proj-nvfp4-torchao.safetensors", type="text_encoders",
      gb=6.9, no_remote=True, note="conversione locale del TE bf16 (6.9GB vs 24.5)"),
    N("ltx25-video-vae-conv", "Video", "LTX-2.5 video VAE Conv", category="video", arch="ltx25",
      repo="dummy9996/LTX-2.5-22b-ungate",
      remote="ltx-2.5-video-vae-conv-bf16.safetensors",
      filename="ltx-2.5-video-vae-conv-bf16.safetensors", type="vae", gb=1.36,
      note="mirror ungate (byte-identico)"),
    N("ltx25-audio-vae", "Video", "LTX-2.5 audio VAE (+vocoder)", category="video", arch="ltx25",
      repo="dummy9996/LTX-2.5-22b-ungate",
      remote="ltx-2.5-audio-vae-bf16.safetensors",
      filename="ltx-2.5-audio-vae-bf16.safetensors", type="vae", gb=0.34,
      note="mirror ungate (byte-identico)"),
    N("ltx25-diffusion-decoder", "Video", "LTX-2.5 video VAE (diffusion decoder)",
      category="video", arch="ltx25", repo="Lightricks/LTX-2.5", gated=True,
      remote="vae/ltx-2.5-video-vae-bf16.safetensors",
      filename="ltx-2.5-video-vae-bf16.safetensors", type="vae", gb=1.37,
      note="solo repo ufficiale gated (assente dal mirror ungate)"),
    N("ltx25-spatial-upscaler", "Video", "LTX-2.5 latent spatial upscaler x2",
      category="video", arch="ltx25", repo="dummy9996/LTX-2.5-22b-ungate",
      remote="ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
      filename="ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
      type="latent_upscale_models", gb=0.93, note="mirror ungate (byte-identico)"),
    N("ltx25-temporal-upscaler", "Video", "LTX-2.5 latent temporal upscaler x2",
      category="video", arch="ltx25", repo="dummy9996/LTX-2.5-22b-ungate",
      remote="ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
      filename="ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
      type="latent_upscale_models", gb=0.25, note="mirror ungate (byte-identico)"),
    N("ltx25-distilled-lora-450", "Video", "LTX-2.5 distilled-LoRA 450", category="video",
      arch="ltx25", repo="dummy9996/LTX-2.5-22b-ungate",
      remote="ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
      filename="ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
      type="LoRA", family="LTX-2.5", gb=8.3, note="mirror ungate (byte-identico)"),
    N("ltx25-duration-head", "Video", "LTX-2.5 duration head", category="video", arch="ltx25",
      repo="dummy9996/LTX-2.5-22b-ungate",
      remote="ltx-2.5-duration-head-bf16.safetensors",
      filename="ltx-2.5-duration-head-bf16.safetensors", type="model_patches", gb=0.01,
      note="mirror ungate (byte-identico)"),
    N("ltx25-iclora-pixel-upscaler", "Video", "LTX-2.5 IC-LoRA Pixel Spatial Upscaler",
      category="video", arch="ltx25", repo="dummy9996/LTX-2.5-22b-ungate",
      remote="ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
      filename="ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
      type="LoRA", family="LTX-2.5", gb=0.33,
      note="filename CONFERMATO; mirror ungate (ufficiale gated: Lightricks IC-LoRA repo)"),
    # ── ecosistema COMMUNITY 2.5 (12-16/08, verificato su HF il 16/08) ──
    N("ltx25-gguf-q4-distilled", "Video", "LTX-2.5 distilled GGUF Q4_K_M (GPU piccole)",
      category="video", arch="ltx25", repo="Abiray/LTX-2.5-Distilled-GGUF",
      remote="LTX-2.5-Distilled-Q4_K_M.gguf", filename="LTX-2.5-Distilled-Q4_K_M.gguf",
      type="diffusion_models", family="LTX", gb=14.6,
      note="community (28K dl al 16/08); 1a gen GGUF 2.5 da validare live"),
    N("ltx25-gguf-q6-distilled", "Video", "LTX-2.5 distilled GGUF Q6_K (qualità, 24GB+)",
      category="video", arch="ltx25", repo="Abiray/LTX-2.5-Distilled-GGUF",
      remote="LTX-2.5-Distilled-Q6_K.gguf", filename="LTX-2.5-Distilled-Q6_K.gguf",
      type="diffusion_models", family="LTX", gb=17.3),
    N("ltx25-gguf-q2-distilled", "Video", "LTX-2.5 distilled GGUF Q2_K (8.2GB, GPU 12GB)",
      category="video", arch="ltx25", repo="realrebelai/LTX-2.5_GGUFs",
      remote="LTX-2.5-Distilled-Q2_K.gguf", filename="LTX-2.5-Distilled-Q2_K.gguf",
      type="diffusion_models", family="LTX", gb=8.2,
      note="quant estremo per GPU piccole; qualità da valutare"),
    N("ltx25-te-gemma4-gguf", "Video", "LTX-2.5 Gemma-4 TE GGUF Q5_K_M (8.9GB vs 24.5 bf16)",
      category="video", arch="ltx25", repo="elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF",
      remote="gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf",
      filename="gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf", type="text_encoders", gb=8.9,
      note="TE quantizzato community; loader TE GGUF nel worker 2.5 da validare"),
    N("ltx25-lora-multisubject", "Video", "LTX-2.5 Multiple Subject Reference (LiconStudio)",
      category="video", arch="ltx25", repo="LiconStudio/LTX-2.5-Multiple-Subject-Reference",
      remote="LTX-2.5-Licon-MSR-V1.safetensors", filename="LTX-2.5-Licon-MSR-V1.safetensors",
      type="LoRA", family="LTX-2.5", gb=1.22,
      note="riferimenti multipli (max 5 immagini) per la 2.5 — apache-2.0; V1 del 14/08, molto "
           "meno provata della V2 per 2.3. NON funziona sulla 2.3: le due LoRA sono addestrate "
           "ciascuna sulla propria architettura"),
    N("ltx23-lora-multisubject", "Video", "LTX-2.3 Multiple Subject Reference V2 (LiconStudio)",
      category="video", arch="ltx23", repo="LiconStudio/LTX-2.3-Multiple-Subject-Reference",
      remote="LTX-2.3-Licon-MSR-V2.safetensors", filename="LTX-2.3-Licon-MSR-V2.safetensors",
      type="LoRA", family="LTX-2", gb=0.65,
      note="riferimenti multipli (max 5 immagini) per la 2.3 — apache-2.0; 28K download dal "
           "1/06: la variante più matura. Il file 2.5 NON è intercambiabile"),

    # ───────── MiniMax H3 (3ª architettura video, pesi aperti dal 3/08/2026) ─────────
    # Omni-transformer 33.1B: video + audio stereo 32kHz denoisati INSIEME. Due checkpoint
    # DISTINTI: fl2va (t2v/i2v/primo-ultimo frame) e ref2va (riferimenti multipli: 9 img +
    # 3 video + 3 audio). TE = Qwen3-VL-32B. File verificati su HF il 17/08/2026.
    # ⚠ LICENZA "MiniMax H3 Community": Territori Esclusi = UE, UK, USA, Corea del Sud; obbligo
    # di mostrare "MiniMax H3" nella UI del prodotto commerciale; vietato usare gli output per
    # allenare ALTRI modelli (⇒ mai dataset LoRA non-H3 da clip H3). Noi distribuiamo SOLO il
    # codice: i pesi li scarica l'utente finale, che valuta la licenza per il proprio territorio.
    N("mmh3-fl2va-int8", "Video", "MiniMax H3 FL2VA int8-ConvRot (t2v/i2v, consigliato 32GB)",
      category="video", arch="minimax_h3", repo="Comfy-Org/MiniMax-H3",
      remote="diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
      filename="minimax_h3_fl2va_pruned_int8_convrot-native.safetensors", convert="mmh3_dit",
      type="diffusion_models", family="MiniMax", gb=20.97,
      deps=["mmh3-configs", "mmh3-te-nvfp4", "mmh3-video-vae", "mmh3-audio-vae"],
      used_by=["cap.video_minimax_h3"],
      note="il checkpoint del workflow ComfyUI ufficiale; picco ~31.8GB VRAM col TE nvfp4. "
           "Licenza MiniMax H3 Community: territori esclusi UE/UK/USA/Corea"),
    N("mmh3-configs", "Video", "MiniMax H3 config + tokenizer (ufficiali, ~25MB)",
      category="video", arch="minimax_h3", kind="snapshot", repo="MiniMaxAI/MiniMax-H3",
      tool_dest=r"E:\AI\Models\configs\minimax-h3", gb=0.03,
      allow=["model_index.json", "modular_model_index.json", "scheduler/*", "audio_scheduler/*",
             "transformer/config.json", "transformer_ref/config.json", "text_encoder/config.json",
             "vae/config.json", "audio_vae/config.json", "processor/*", "tokenizer/*", "docs/*.md"],
      used_by=["cap.video_minimax_h3"],
      note="SOLO config/tokenizer/processor + guide ai prompt dal repo ufficiale (layout diffusers): le classi diffusers si costruiscono "
           "da qui e i pesi arrivano dal single-file quantizzato (stesso schema di FLUX.2). "
           "Evita i 124GB dello snapshot completo"),
    N("mmh3-fl2va-nvfp4", "Video", "MiniMax H3 FL2VA nvfp4 (12.5GB, ideale RTX 50xx)",
      category="video", arch="minimax_h3", repo="Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot",
      remote="MiniMax_H3_FL2VA_pruned_nvfp4.safetensors",
      filename="MiniMax_H3_FL2VA_pruned_nvfp4-native.safetensors", convert="mmh3_dit",
      type="diffusion_models", family="MiniMax", gb=12.53,
      deps=["mmh3-configs", "mmh3-te-nvfp4", "mmh3-video-vae", "mmh3-audio-vae"],
      used_by=["cap.video_minimax_h3"],
      note="il più piccolo dei non-GGUF: kernel FP4 nativi su Blackwell, si carica col nostro "
           "Nvfp4Linear (già validato su FLUX.2). Licenza con territori esclusi"),
    N("mmh3-ref2va-nvfp4", "Video", "MiniMax H3 Ref2VA nvfp4 (riferimenti, 12.5GB)",
      category="video", arch="minimax_h3", repo="Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot",
      remote="MiniMax_H3_Ref2VA_pruned_nvfp4.safetensors",
      filename="MiniMax_H3_Ref2VA_pruned_nvfp4-native.safetensors", convert="mmh3_dit",
      type="diffusion_models", family="MiniMax", gb=12.53,
      deps=["mmh3-configs", "mmh3-te-nvfp4", "mmh3-video-vae", "mmh3-audio-vae"],
      used_by=["cap.video_minimax_h3"],
      note="variante nvfp4 del checkpoint riferimenti; licenza con territori esclusi"),
    N("mmh3-fl2va-fp8", "Video", "MiniMax H3 FL2VA fp8-scaled (t2v/i2v)",
      category="video", arch="minimax_h3", repo="Comfy-Org/MiniMax-H3",
      remote="diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
      filename="minimax_h3_fl2va_pruned_fp8_scaled-native.safetensors", convert="mmh3_dit",
      type="diffusion_models", family="MiniMax", gb=20.96,
      deps=["mmh3-configs", "mmh3-te-nvfp4", "mmh3-video-vae", "mmh3-audio-vae"],
      used_by=["cap.video_minimax_h3"],
      note="alternativa fp8 all'int8; licenza con territori esclusi (UE/UK/USA/Corea)"),
    N("mmh3-fl2va-bf16", "Video", "MiniMax H3 FL2VA bf16 pruned (qualità piena, 40GB)",
      category="video", arch="minimax_h3", repo="Comfy-Org/MiniMax-H3",
      remote="diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors",
      filename="minimax_h3_fl2va_pruned_bf16-native.safetensors", convert="mmh3_dit",
      type="diffusion_models", family="MiniMax", gb=40.23,
      deps=["mmh3-configs", "mmh3-te-nvfp4", "mmh3-video-vae", "mmh3-audio-vae"],
      used_by=["cap.video_minimax_h3"],
      note="non entra in 32GB senza offload; licenza con territori esclusi"),
    N("mmh3-ref2va-int8", "Video", "MiniMax H3 Ref2VA int8-ConvRot (riferimenti multipli)",
      category="video", arch="minimax_h3", repo="Comfy-Org/MiniMax-H3",
      remote="diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
      filename="minimax_h3_ref2va_pruned_int8_convrot-native.safetensors", convert="mmh3_dit",
      type="diffusion_models", family="MiniMax", gb=20.97,
      deps=["mmh3-configs", "mmh3-te-nvfp4", "mmh3-video-vae", "mmh3-audio-vae"],
      used_by=["cap.video_minimax_h3"],
      note="checkpoint SEPARATO dal fl2va: serve SOLO per i riferimenti multipli "
           "(9 immagini + 3 video + 3 audio). Licenza con territori esclusi"),
    N("mmh3-ref2va-fp8", "Video", "MiniMax H3 Ref2VA fp8-scaled (riferimenti multipli)",
      category="video", arch="minimax_h3", repo="Comfy-Org/MiniMax-H3",
      remote="diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
      filename="minimax_h3_ref2va_pruned_fp8_scaled-native.safetensors", convert="mmh3_dit",
      type="diffusion_models", family="MiniMax", gb=20.96,
      deps=["mmh3-configs", "mmh3-te-nvfp4", "mmh3-video-vae", "mmh3-audio-vae"],
      used_by=["cap.video_minimax_h3"],
      note="alternativa fp8 al ref2va int8; licenza con territori esclusi"),
    N("mmh3-te-nvfp4", "Video", "MiniMax H3 TE Qwen3-VL-32B nvfp4-AWQ (15.7GB vs 51.5 bf16)",
      category="video", arch="minimax_h3", repo="Comfy-Org/MiniMax-H3",
      remote="text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
      filename="qwen3vl_32b_minimax_h3_nvfp4_awq-native.safetensors", convert="mmh3_te",
      type="text_encoders", gb=15.69, used_by=["cap.video_minimax_h3"],
      note="TE consigliato su 32GB; è un VLM completo: legge anche le immagini di riferimento"),
    N("mmh3-te-int8", "Video", "MiniMax H3 TE Qwen3-VL-32B int8-ConvRot (qualità, 27GB)",
      category="video", arch="minimax_h3", repo="Comfy-Org/MiniMax-H3",
      remote="text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
      filename="qwen3vl_32b_minimax_h3_int8_convrot-native.safetensors", convert="mmh3_te",
      type="text_encoders", gb=27.14, used_by=["cap.video_minimax_h3"],
      note="serve parcheggio VRAM→RAM fra TE e transformer (il nostro handoff lo fa)"),
    N("mmh3-video-vae", "Video", "MiniMax H3 video VAE fp16 (f16t4d24)",
      category="video", arch="minimax_h3", repo="Comfy-Org/MiniMax-H3",
      remote="vae/minimax_h3_video_vae_fp16.safetensors",
      filename="minimax_h3_video_vae_fp16-native.safetensors", convert="mmh3_vae", type="vae", gb=5.21,
      used_by=["cap.video_minimax_h3"],
      note="16x spaziale, 4x temporale, 24 canali, decoder ViT"),
    N("mmh3-audio-vae", "Video", "MiniMax H3 audio VAE (ufficiale diffusers, stereo 32kHz)",
      category="video", arch="minimax_h3", repo="MiniMaxAI/MiniMax-H3",
      remote="audio_vae/diffusion_pytorch_model.safetensors",
      filename="minimax_h3_audio_vae.safetensors", type="vae", gb=0.61,
      used_by=["cap.video_minimax_h3"],
      note="l'audio nasce dalla STESSA passata del video: già sincronizzato. File ufficiale in formato "
           "diffusers (nessuna conversione)"),
    N("mmh3-turbo-lora-fl2v", "Video", "MiniMax H3 Turbo LoRA 4 step FL2V (lightx2v)",
      category="video", arch="minimax_h3", repo="Comfy-Org/MiniMax-H3",
      remote="loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
      filename="minimax_h3_fl2v_turbo_4step_v1.0_768p-native.safetensors", convert="mmh3_lora",
      type="LoRA", family="MiniMax-H3", gb=1.96, used_by=["cap.video_minimax_h3"],
      note="distillazione lightx2v/ModelTC: da ~20 step a 4 (768p)"),
    N("mmh3-turbo-lora-ref2v", "Video", "MiniMax H3 Turbo LoRA 4 step Ref2V (anteprima v0.1)",
      category="video", arch="minimax_h3", repo="Comfy-Org/MiniMax-H3",
      remote="loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
      filename="minimax_h3_ref2v_turbo_4step_v0.1-native.safetensors", convert="mmh3_lora",
      type="LoRA", family="MiniMax-H3", gb=1.96, used_by=["cap.video_minimax_h3"],
      note="anteprima: la distillazione dei riferimenti è meno matura della FL2V"),
    N("mmh3-te-fp8", "Video", "MiniMax H3 TE Qwen3-VL-32B fp8 (per GPU senza nvfp4, 26.5GB)",
      category="video", arch="minimax_h3", repo="rzgar/qwen3-vl-32b-minimax-h3-fp8-comfyui",
      remote="qwen3-vl-32b-minimax-h3-fp8-comfyui.safetensors",
      filename="qwen3vl_32b_minimax_h3_fp8-native.safetensors", convert="mmh3_te",
      type="text_encoders", gb=26.5, used_by=["cap.video_minimax_h3"],
      note="50 layer (quelli che H3 legge), fp8 e4m3: per RTX 40xx. Si carica da solo e si libera "
           "prima del transformer"),
    N("mmh3-te-heretic-nvfp4", "Video", "MiniMax H3 TE Qwen3-VL-32B Heretic nvfp4 (senza censura)",
      category="video", arch="minimax_h3", repo="sakamakismile/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4",
      remote="qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
      filename="qwen3vl_32b_heretic_minimax_h3_nvfp4-native.safetensors", convert="mmh3_te",
      type="text_encoders", gb=15.68, sens="adult", used_by=["cap.video_minimax_h3"],
      note="text encoder «abliterated»: non rifiuta i prompt 18+ (solo edizioni adulte)"),
    N("mmh3-turbo-lora-fl2v-8", "Video", "MiniMax H3 Turbo LoRA 8 step FL2V (qualità)",
      category="video", arch="minimax_h3", repo="Comfy-Org/MiniMax-H3",
      remote="loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
      filename="minimax_h3_fl2v_turbo_8step_v1.0-native.safetensors", convert="mmh3_lora",
      type="LoRA", family="MiniMax-H3", gb=1.96, used_by=["cap.video_minimax_h3"],
      note="8 passi: più dettaglio del 4 passi, sempre ~3x più veloce del modello base"),

    # ───────── LongCat-Video-Avatar 1.5 (tab Avatar: foto + voce → persona che parla) ─────────
    # MIT. Codice ufficiale nel venv E:\AI\Tools\LongCatAvatar (scripts/install_longcat_avatar.py).
    # Text encoder UMT5-XXL, VAE Wan e tokenizer vengono dal modello BASE LongCat-Video; il DiT
    # avatar si sceglie per precisione (INT8 ufficiale / FP8 wavespeed / bf16). File verificati su
    # HF il 26/09/2026 (header safetensors letti via HTTP range).
    N("lcav-configs", "Avatar", "LongCat-Video config + tokenizer (json, ~25MB)",
      category="video", arch="longcat_avatar", kind="snapshot", repo="meituan-longcat/LongCat-Video",
      type="configs", gb=0.03, used_by=["cap.avatar_longcat"],
      allow=["tokenizer/*", "text_encoder/*.json", "vae/config.json", "scheduler/*",
             "model_index.json", "config.json"],
      note="solo json e tokenizer del modello base: il worker ci costruisce text encoder e VAE"),
    N("lcav-te", "Avatar", "LongCat-Video text encoder UMT5-XXL (fp32, caricato in bf16)",
      category="video", arch="longcat_avatar", kind="snapshot", repo="meituan-longcat/LongCat-Video",
      type="TextEncoders", family="LongCat", gb=22.7, used_by=["cap.avatar_longcat"],
      allow=["text_encoder/*.safetensors", "text_encoder/*.json"], scan=["LongCat-Video"],
      note="lavora solo all'inizio (codifica il prompt) e poi si libera: non occupa VRAM "
           "durante la generazione"),
    N("lcav-vae", "Avatar", "LongCat-Video VAE (Wan 2.1)", category="video", arch="longcat_avatar",
      repo="meituan-longcat/LongCat-Video", remote="vae/diffusion_pytorch_model.safetensors",
      filename="diffusion_pytorch_model.safetensors", type="VAE", family="LongCat", gb=0.51,
      used_by=["cap.avatar_longcat"]),
    N("lcav15-core", "Avatar", "LongCat Avatar 1.5 base (LoRA DMD 8 step + Whisper + separatore voce)",
      category="video", arch="longcat_avatar", kind="snapshot",
      repo="meituan-longcat/LongCat-Video-Avatar-1.5", type="LongCat", gb=5.7,
      deps=["lcav-configs", "lcav-te", "lcav-vae"], used_by=["cap.avatar_longcat"],
      allow=["config.json", "model_index.json", "scheduler/*", "lora/dmd_lora.safetensors",
             "vocal_separator/*", "whisper-large-v3/*.json", "whisper-large-v3/*.txt",
             "whisper-large-v3/model.safetensors"],
      note="componenti comuni a tutte le precisioni: Whisper-large-v3 SOLO fp16 (3.1GB, non i "
           "12GB fp32/flax/bin del repo)"),
    N("lcav15-int8", "Avatar", "LongCat Avatar 1.5 DiT INT8 ufficiale (15.9GB, ogni GPU)",
      category="video", arch="longcat_avatar", kind="snapshot",
      repo="meituan-longcat/LongCat-Video-Avatar-1.5", type="LongCat", family="int8", gb=15.9,
      deps=["lcav15-core"], used_by=["cap.avatar_longcat"], allow=["base_model_int8/*"],
      note="quantizzazione ufficiale weight-only per canale: gira su qualunque GPU NVIDIA; "
           "scelta consigliata su 32GB (picco ~28GB)"),
    N("lcav15-fp8", "Avatar", "LongCat Avatar 1.5 DiT FP8 e4m3 (15.9GB, RTX 40xx/50xx)",
      category="video", arch="longcat_avatar", kind="snapshot",
      repo="wavespeed/LongCat-Video-Avatar-1.5-e4m3", type="LongCat", family="fp8", gb=15.9,
      deps=["lcav15-core"], used_by=["cap.avatar_longcat"],
      note="FP8 per canale (non ufficiale, MIT): matmul FP8 nativo dove supportato, altrimenti "
           "dequantizza al volo come l'INT8"),
    N("lcav15-bf16", "Avatar", "LongCat Avatar 1.5 DiT bf16 (31.7GB, qualità di riferimento)",
      category="video", arch="longcat_avatar", kind="snapshot",
      repo="meituan-longcat/LongCat-Video-Avatar-1.5", type="LongCat", family="bf16", gb=31.7,
      deps=["lcav15-core"], used_by=["cap.avatar_longcat"], allow=["base_model/*"], role="on_demand",
      note="non entra in 32GB insieme al resto: serve solo per confronti di qualità"),

    # ───────── Qwen-Image 2.1 (immagini + editing, ≤10 riferimenti, RGBA nativo) ─────────
    # Venv condiviso E:\AI\Tools\diffusers_next_venv (diffusers main). Profili che stanno in 32GB: nvfp4
    # (BennyDaBall, formato ComfyUI) · fp8/int8 (unsloth, torchao per riga). Header verificati il 26/09/2026.
    # ⚠ Licenza Qwen Research: uso NON commerciale (per usi commerciali serve una licenza Alibaba): i pesi li scarica l'utente.
    N("qwen21-configs", "Immagini", "Qwen-Image 2.1 config + processor + scheduler (json, ~20MB)",
      category="image", arch="qwen_image21", kind="snapshot", repo="Qwen/Qwen-Image-2.1", type="configs",
      gb=0.02, used_by=["cap.image_qwen21"],
      allow=["model_index.json", "scheduler/*", "processor/*", "transformer/config.json",
             "text_encoder/config.json", "text_encoder/generation_config.json", "vae/config.json", "LICENSE"],
      note="solo json/tokenizer dal repo ufficiale. Licenza Qwen Research: uso NON commerciale (per usi commerciali serve una licenza Alibaba): i pesi li scarica l'utente"),
    N("qwen21-vae", "Immagini", "Qwen-Image 2.1 VAE RGBA bf16 (0.68GB)", category="image",
      arch="qwen_image21", repo="unsloth/Qwen-Image-2.1-FP8", remote="vae/qwen_image_2.1_vae_bf16.safetensors",
      filename="qwen_image_2.1_vae_bf16.safetensors", type="VAE", family="Qwen21", gb=0.68,
      used_by=["cap.image_qwen21"], note="stesse chiavi del VAE ufficiale fp32 (1.35GB), metà spazio"),
    N("qwen21-te-fp8", "Immagini", "Qwen-Image 2.1 text encoder Qwen3-VL-8B FP8 (9.4GB)", category="image",
      arch="qwen_image21", repo="unsloth/Qwen-Image-2.1-FP8",
      remote="Qwen-Image-2.1-text_encoder-FP8.safetensors", filename="Qwen-Image-2.1-text_encoder-FP8.safetensors",
      type="TextEncoders", family="Qwen21", gb=9.39, used_by=["cap.image_qwen21"],
      note="VLM: legge anche le immagini di riferimento. Licenza Qwen Research: uso NON commerciale (per usi commerciali serve una licenza Alibaba): i pesi li scarica l'utente"),
    N("qwen21-te-nvfp4", "Immagini", "Qwen-Image 2.1 text encoder Qwen3-VL-8B NVFP4 (7.5GB)", category="image",
      arch="qwen_image21", repo="BennyDaBall/Qwen-Image-2.1-NVFP4",
      remote="text_encoders/qwen3vl_8b_nvfp4.safetensors", filename="qwen3vl_8b_nvfp4-native.safetensors", convert="qwen21_te",
      type="TextEncoders", family="Qwen21", gb=7.55, used_by=["cap.image_qwen21"],
      note="kernel FP4 delle RTX 50xx (dequant comfy-kitchen). Licenza Qwen Research: uso NON commerciale (per usi commerciali serve una licenza Alibaba): i pesi li scarica l'utente"),
    N("qwen21-te-heretic-nvfp4", "Immagini", "Qwen-Image 2.1 text encoder «heretic» NVFP4 (18+, 6.3GB)",
      category="image", arch="qwen_image21", repo="pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-NVFP4",
      remote="qwen3vl_8b_nvfp4_heretic.safetensors", filename="qwen3vl_8b_nvfp4_heretic-native.safetensors", convert="qwen21_te",
      type="TextEncoders", family="Qwen21", gb=6.31, sens="adult", used_by=["cap.image_qwen21"],
      note="encoder senza rifiuti per l'edizione 18+ (non ufficiale; formato da validare al primo uso). Licenza Qwen Research: uso NON commerciale (per usi commerciali serve una licenza Alibaba): i pesi li scarica l'utente"),
    N("qwen21-dit-nvfp4", "Immagini", "Qwen-Image 2.1 DiT NVFP4 (4.2GB, RTX 50xx)", category="image",
      arch="qwen_image21", repo="BennyDaBall/Qwen-Image-2.1-NVFP4",
      remote="diffusion_models/qwen_image_2.1_nvfp4.safetensors", filename="qwen_image_2.1_nvfp4-native.safetensors", convert="qwen21_dit",
      type="diffusion_models", family="Qwen21", gb=4.2,
      deps=["qwen21-configs", "qwen21-vae", "qwen21-te-nvfp4|qwen21-te-fp8"], used_by=["cap.image_qwen21"],
      note="profilo più piccolo: catena completa 12.4GB. Licenza Qwen Research: uso NON commerciale (per usi commerciali serve una licenza Alibaba): i pesi li scarica l'utente"),
    N("qwen21-dit-fp8", "Immagini", "Qwen-Image 2.1 DiT FP8 (7.1GB, RTX 40xx/50xx)", category="image",
      arch="qwen_image21", repo="unsloth/Qwen-Image-2.1-FP8", remote="Qwen-Image-2.1-FP8.safetensors",
      filename="Qwen-Image-2.1-FP8.safetensors", type="diffusion_models", family="Qwen21", gb=7.12,
      deps=["qwen21-configs", "qwen21-vae", "qwen21-te-fp8|qwen21-te-nvfp4"], used_by=["cap.image_qwen21"],
      note="FP8 per riga (torchao): catena completa 17.2GB. Licenza Qwen Research: uso NON commerciale (per usi commerciali serve una licenza Alibaba): i pesi li scarica l'utente"),
    N("qwen21-dit-int8", "Immagini", "Qwen-Image 2.1 DiT INT8 (7.3GB, ogni GPU, qualità migliore)",
      category="image", arch="qwen_image21", repo="unsloth/Qwen-Image-2.1-FP8",
      remote="Qwen-Image-2.1-INT8.safetensors", filename="Qwen-Image-2.1-INT8.safetensors",
      type="diffusion_models", family="Qwen21", gb=7.26,
      deps=["qwen21-configs", "qwen21-vae", "qwen21-te-fp8|qwen21-te-nvfp4"], used_by=["cap.image_qwen21"],
      note="INT8 per riga: LPIPS 0.064 contro 0.112 dell'FP8 (dati unsloth). Licenza Qwen Research: uso NON commerciale (per usi commerciali serve una licenza Alibaba): i pesi li scarica l'utente"),
    N("qwen21-turbo-lora-4", "Immagini", "Qwen-Image 2.1 Turbo LoRA 4 passi (Viggle, 0.34GB)", category="image",
      arch="qwen_image21", repo="Viggle/Qwen-Image-2.1-viggle-turbo",
      remote="Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors",
      filename="Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors", type="LoRA", family="Qwen21",
      gb=0.34, used_by=["cap.image_qwen21"], note="4 passi invece di 40 (non ufficiale). Licenza Qwen Research: uso NON commerciale (per usi commerciali serve una licenza Alibaba): i pesi li scarica l'utente"),
    N("qwen21-turbo-lora-6", "Immagini", "Qwen-Image 2.1 Turbo LoRA 6 passi v0.2.1 (Viggle, 0.68GB)",
      category="image", arch="qwen_image21", repo="Viggle/Qwen-Image-2.1-viggle-turbo",
      remote="Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors",
      filename="Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors", type="LoRA", family="Qwen21",
      gb=0.68, used_by=["cap.image_qwen21"], note="6 passi, qualità più vicina ai 40 passi (non ufficiale). Licenza Qwen Research: uso NON commerciale (per usi commerciali serve una licenza Alibaba): i pesi li scarica l'utente"),
    # Turbo UFFICIALE (Qwen/Qwen-Image-2.1-Turbo, 8 passi con la sua schedule, CFG 1): stesso text encoder del base
    # (tensori identici, verificato il 12/10/2026) → riusa i TE esistenti. Single-file ComfyUI convertiti in nativo.
    N("qwen21-turbo-dit-nvfp4", "Immagini", "Qwen-Image 2.1 Turbo DiT NVFP4 (5.3GB, 8 passi, RTX 50xx)",
      category="image", arch="qwen_image21", repo="BennyDaBall/Qwen-Image-2.1-Turbo-NVFP4",
      remote="diffusion_models/qwen_image_2.1_turbo_nvfp4.safetensors",
      filename="qwen_image_2.1_turbo_nvfp4-native.safetensors", convert="qwen21_dit",
      type="diffusion_models", family="Qwen21", gb=5.3,
      deps=["qwen21-configs", "qwen21-vae", "qwen21-te-nvfp4|qwen21-te-fp8"], used_by=["cap.image_qwen21"],
      note="checkpoint Turbo ufficiale distillato a 8 passi (NVFP4 misto di BennyDaBall). Licenza Qwen Research: uso NON commerciale (per usi commerciali serve una licenza Alibaba): i pesi li scarica l'utente"),
    N("qwen21-turbo-dit-int8", "Immagini", "Qwen-Image 2.1 Turbo DiT INT8 (6.8GB, 8 passi, ogni GPU)",
      category="image", arch="qwen_image21", repo="Comfy-Org/Qwen-Image-2.1",
      remote="diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
      filename="qwen_image_2.1_turbo_int8_convrot-native.safetensors", convert="qwen21_dit",
      type="diffusion_models", family="Qwen21", gb=6.76,
      deps=["qwen21-configs", "qwen21-vae", "qwen21-te-fp8|qwen21-te-nvfp4"], used_by=["cap.image_qwen21"],
      note="checkpoint Turbo ufficiale distillato a 8 passi (INT8 ConvRot di Comfy-Org). Licenza Qwen Research: uso NON commerciale (per usi commerciali serve una licenza Alibaba): i pesi li scarica l'utente"),
    # 18+ — checkpoint senza censura (nudo ed esplicito senza LoRA). Visibili SOLO in modalità NSFW e nelle
    # edizioni 18+ (sensitivity=adult). Header verificati il 12/10/2026: stessi nomi del DiT diffusers.
    N("qwen21-uc-dit-nvfp4", "Immagini", "Qwen-Image 2.1 Uncensored DiT NVFP4 (18+, 3.9GB, RTX 50xx)",
      category="image", arch="qwen_image21", repo="abenzerps/Qwen-Image-2.1-Uncensored-Quantized",
      remote="qwen-image-2.1-UC-NVFP4.safetensors", filename="qwen-image-2.1-UC-NVFP4-native.safetensors",
      convert="qwen21_dit", type="diffusion_models", family="Qwen21", gb=3.91, sens="adult",
      deps=["qwen21-configs", "qwen21-vae", "qwen21-te-nvfp4|qwen21-te-fp8"], used_by=["cap.image_qwen21"],
      note="versione senza censura dei pesi base (abenzerps). Licenza Qwen Research: uso NON commerciale: i pesi li scarica l'utente"),
    N("qwen21-uc-dit-int8", "Immagini", "Qwen-Image 2.1 Uncensored DiT INT8 (18+, 6.8GB, ogni GPU)",
      category="image", arch="qwen_image21", repo="abenzerps/Qwen-Image-2.1-Uncensored-Quantized",
      remote="qwen-image-2.1-UC-int8_convrot.safetensors", filename="qwen-image-2.1-UC-int8_convrot-native.safetensors",
      convert="qwen21_dit", type="diffusion_models", family="Qwen21", gb=6.76, sens="adult",
      deps=["qwen21-configs", "qwen21-vae", "qwen21-te-fp8|qwen21-te-nvfp4"], used_by=["cap.image_qwen21"],
      note="versione senza censura dei pesi base (abenzerps), INT8 ConvRot. Licenza Qwen Research: uso NON commerciale: i pesi li scarica l'utente"),
    N("qwen21-noctq-v4-int8", "Immagini", "Noct-Q V4 Uncensored realismo (Qwen-Image 2.1, 18+, 6.8GB)",
      category="image", arch="qwen_image21", repo="Noctaluna/Noct-Q-Uncensored-Qwen-Image-2.1",
      remote="NoctQ_V4_int8_convrot.safetensors", filename="NoctQ_V4_int8_convrot-native.safetensors",
      convert="qwen21_dit", type="diffusion_models", family="Qwen21", gb=6.76, sens="adult",
      deps=["qwen21-configs", "qwen21-vae", "qwen21-te-fp8|qwen21-te-nvfp4"], used_by=["cap.image_qwen21"],
      note="merge fotorealistico senza censura (Noctaluna): 25 passi, CFG 3 col negativo; prompt in prosa. Licenza Qwen Research: uso NON commerciale: i pesi li scarica l'utente"),
    N("qwen21-noctq-anime-int8", "Immagini", "Noct-Q Anime V1 Uncensored (Qwen-Image 2.1, 18+, 6.8GB)",
      category="image", arch="qwen_image21", repo="Noctaluna/Noct-Q-Anime-Uncensored-Qwen-Image-2.1",
      remote="NoctQA_V1_int8_convrot.safetensors", filename="NoctQA_V1_int8_convrot-native.safetensors",
      convert="qwen21_dit", type="diffusion_models", family="Qwen21", gb=6.76, sens="adult",
      deps=["qwen21-configs", "qwen21-vae", "qwen21-te-fp8|qwen21-te-nvfp4"], used_by=["cap.image_qwen21"],
      note="merge anime/illustrazione senza censura (Noctaluna): 25 passi, CFG 3 col negativo. Licenza Qwen Research: uso NON commerciale: i pesi li scarica l'utente"),
    # ControlNet-Union ufficiale (alibaba-pai «Fun»): ramo di controllo (control_img_in + 16 control_blocks) sopra il
    # DiT: canny/depth/pose/lineart/hed/mlsd/scribble/gray + inpaint. INT8 ConvRot di Comfy-Org (bf16 ufficiale 7GB).
    N("qwen21-controlnet-int8", "Immagini", "Qwen-Image 2.1 ControlNet Union (Fun, INT8, 3.5GB)",
      category="image", arch="qwen_image21", repo="Comfy-Org/Qwen-Image-2.1",
      remote="model_patches/qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors",
      filename="qwen_image_2.1_fun_controlnet_union_int8_convrot-native.safetensors", convert="qwen21_cn",
      type="controlnet", family="Qwen21", gb=3.52, used_by=["cap.qwen21_controlnet"],
      note="una sola rete per 8 condizioni di struttura + inpaint (alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union). Licenza Qwen Research: uso NON commerciale"),
    # Arricchitori di prompt UFFICIALI (Qwen3.5-VL 9B fine-tunati): T2I = brief → prompt lungo in inglese;
    # I2I = istruzione vaga + immagini → istruzione di editing precisa. Pesi INT8 ConvRot di Comfy-Org, config dai repo Qwen.
    N("qwen21-pe-t2i-configs", "Immagini", "Qwen-Image 2.1 PE testo→immagine: config + tokenizer + system prompt",
      category="image", arch="qwen_image21", kind="snapshot", repo="Qwen/Qwen-Image-2.1-PE-T2I", type="configs",
      gb=0.02, used_by=["cap.qwen21_prompt_enhancer"], allow=["*.json", "*.txt", "*.jinja", "LICENSE"],
      note="solo json/tokenizer/system_prompt.txt del repo ufficiale"),
    N("qwen21-pe-i2i-configs", "Immagini", "Qwen-Image 2.1 PE editing: config + tokenizer + system prompt",
      category="image", arch="qwen_image21", kind="snapshot", repo="Qwen/Qwen-Image-2.1-PE-I2I", type="configs",
      gb=0.02, used_by=["cap.qwen21_prompt_enhancer"], allow=["*.json", "*.txt", "*.jinja", "LICENSE"],
      note="solo json/tokenizer/system_prompt.txt del repo ufficiale"),
    N("qwen21-pe-t2i-int8", "Immagini", "Qwen-Image 2.1 arricchitore prompt testo→immagine (Qwen3.5-VL 9B INT8, 8.8GB)",
      category="image", arch="qwen_image21", repo="Comfy-Org/Qwen-Image-2.1",
      remote="text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors",
      filename="qwen3.5_9b_qwen_image_2.1_pe_t2i-native.safetensors", convert="qwen21_pe",
      type="TextEncoders", family="Qwen21", gb=8.82, deps=["qwen21-pe-t2i-configs"], used_by=["cap.qwen21_prompt_enhancer"],
      note="riscrive il brief in un prompt lungo e dettagliato (ufficiale Qwen). Licenza Qwen Research: uso NON commerciale"),
    N("qwen21-pe-i2i-int8", "Immagini", "Qwen-Image 2.1 arricchitore prompt editing (Qwen3.5-VL 9B INT8, 8.8GB)",
      category="image", arch="qwen_image21", repo="Comfy-Org/Qwen-Image-2.1",
      remote="text_encoders/qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors",
      filename="qwen3.5_9b_qwen_image_2.1_pe_i2i-native.safetensors", convert="qwen21_pe",
      type="TextEncoders", family="Qwen21", gb=8.82, deps=["qwen21-pe-i2i-configs"], used_by=["cap.qwen21_prompt_enhancer"],
      note="guarda le immagini e trasforma un'istruzione vaga in una di editing precisa (ufficiale Qwen). Licenza Qwen Research: uso NON commerciale"),
    # 18+ — LoRA (pannello LoRA del tab Immagini, solo in modalità NSFW). Il loader nativo aggancia sia il formato
    # peft «.default» sia l'MLP fuso gate_up (anche sui DiT a rami separati).
    N("qwen21-lora-nsfw-alpacas-v1", "Immagini", "LoRA NSFW Qwen-Image 2.1 v1 (TheseAlpacas, 18+, 0.15GB)",
      category="image", arch="qwen_image21", repo="Wickedlizerd/NSFW-Qwen-Image-2.1-LoRA",
      remote="nsfw_qwen_21.safetensors", filename="NSFW_Qwen21_TheseAlpacas_v1.safetensors", type="LoRA",
      family="Qwen21", gb=0.15, sens="adult", used_by=["cap.image_qwen21"],
      note="rank 32, nessuna trigger word; l'autore consiglia ≥25 passi. Mirror con licenza di civitai 2958918. Licenza Qwen Research: uso NON commerciale"),
    N("qwen21-lora-nsfw-alpacas-v2", "Immagini", "LoRA NSFW Qwen-Image 2.1 v2 (TheseAlpacas, 18+, 0.07GB)",
      category="image", arch="qwen_image21", repo="IMOON007/qwenimage-2.1-nsfw-lora",
      remote="NSFW Qwen by TheseAlpacas V2.safetensors", filename="NSFW_Qwen21_TheseAlpacas_v2.safetensors",
      type="LoRA", family="Qwen21", gb=0.07, sens="adult", used_by=["cap.image_qwen21"],
      note="seconda versione (rank 16, anche sull'MLP). Licenza Qwen Research: uso NON commerciale"),
    N("qwen21-lora-nsfw-edit", "Immagini", "LoRA NSFW Image Edit Qwen-Image 2.1 (18+, 0.08GB)",
      category="image", arch="qwen_image21", repo="RunningHubAI/rh-qwen-image-2.1-aio-nsfw-lora",
      remote="Qwen-Image-2.1 NSFW Image Edit.safetensors", filename="Qwen21_NSFW_Image_Edit.safetensors",
      type="LoRA", family="Qwen21", gb=0.08, sens="adult", used_by=["cap.image_qwen21"],
      note="per l'EDITING con riferimento (spogliare, pose): addestrata con istruzioni brevi in CINESE, in inglese rende poco. Licenza dell'autore"),
    N("qwen21-lora-uncensored", "Immagini", "LoRA Uncensored Qwen-Image 2.1 (abenzerps, 18+, 0.03GB)",
      category="image", arch="qwen_image21", repo="abenzerps/Qwen-Image-2.1-Uncensored-Quantized",
      remote="qwen-image-2.1-uncensored-lora.safetensors", filename="Qwen21_Uncensored_abenzerps.safetensors",
      type="LoRA", family="Qwen21", gb=0.03, sens="adult", used_by=["cap.image_qwen21"],
      note="toglie la censura a qualunque DiT base (solo attenzione, rank 16). Licenza Qwen Research: uso NON commerciale"),

    # ───────── 18+ (13/10/2026, scelti dall'utente con domanda a scelta multipla) — header verificati via Range ─────────
    # LTX-2.3: monoliti fp8 ComfyUI (weight_scale scalare + comfy_quant, NIENTE input_scale → ltx2_infer usa
    # fp8_cast). Modelli DEV: la distilled-LoRA della release resta automatica.
    N("ltx23-eros15", "Video", "10Eros v1.5 LTX-2.3 (18+, fp8, 30.2GB)", category="video", arch="ltx23",
      repo="LokkenJP/10EROS_1.5_fp8_exp_learned", remote="10Eros_v1.5_fp8mixed_experimental_learned.safetensors",
      filename="10Eros_v1.5_fp8mixed_experimental_learned.safetensors", type="diffusion_models", family="LTX",
      gb=30.24, sens="adult", sha256="192c8f544fe1694bd974e30217ca0f006be9d629b502313921989ee101bfe483",
      deps=["ltx23-gemma-te", "ltx23-distilled-lora-384", "ltx23-spatial-upscaler"],
      note="aggiornamento di 10Eros v1 (TenStrip): finetune NSFW con VAE inclusi; blocchi 0,1,46,47 in bf16. Licenza LTX-2 Community"),
    N("ltx23-pinkcherry18", "Video", "PinkCherry v1.8 LTX-2.3 (18+, fp8, 27.6GB)", category="video", arch="ltx23",
      repo="SexGod1979/PinkCherry_NSFW_LTX23", remote="v1.8/PinkCherry_FineTune_fp8scaled_v1_8_LTX23.safetensors",
      filename="PinkCherry_FineTune_fp8scaled_v1_8_LTX23.safetensors", type="diffusion_models", family="LTX",
      gb=27.62, sens="adult", deps=["ltx23-gemma-te", "ltx23-distilled-lora-384", "ltx23-spatial-upscaler"],
      note="finetune NSFW con VAE inclusi; l'autore consiglia la distilled-LoRA 384 ufficiale. Derivato LTX-2"),
    # LTX-2.5: solo transformer, int8 per-riga + ConvRot ComfyUI (fold nativo in ltx_core fp8_cast). Distill
    # già fuso → nome locale con «distilled»: il worker NON aggiunge la distilled-LoRA.
    N("ltx25-te-uncensored", "Video", "LTX-2.5 Gemma-4 TE Uncensored v1.1 (18+, 13.2GB → bf16 24.5GB)",
      category="video", arch="ltx25", repo="ChrisColeTech/LTX-2.5-uncensored-v1.1-FP8",
      remote="split/text_encoders/gemma4_12b_ltx25_uncensored-int8.safetensors",
      filename="gemma4-12b-with-proj-ltx-2.5-uncensored-v1.1-bf16-native.safetensors", convert="ltx25_te",
      type="text_encoders", gb=13.16, sens="adult",
      note="Gemma 4 12B senza rifiuti con le proiezioni LTX: int8 → bf16 (il loader Gemma di ltx-core non legge "
           "pesi quantizzati), asset/gemma_config innestati dal TE ufficiale. Vale con ogni transformer 2.5"),
    # Stesso modello del GGUF Q8_0 del repo, ma in safetensors int8: passa dal fold nativo di ltx-core (fp8_cast) e
    # porta i connector degli embedding nello stesso file (il percorso GGUF 2.5 non li sa leggere).
    N("ltx25-uncensored-v11-int8", "Video", "LTX-2.5 Uncensored Turbo v1.1 (18+, INT8, distilled, 21.5GB)",
      category="video", arch="ltx25", repo="ChrisColeTech/LTX-2.5-uncensored-v1.1-FP8",
      remote="split/diffusion_models/ltx25_uncensored_v1.1-int8.safetensors",
      filename="ltx2.5-uncensored-v1.1-distilled-int8-native.safetensors", convert="ltx25_dit",
      type="diffusion_models", family="LTX", gb=21.47, sens="adult",
      deps=["ltx25-te-uncensored|ltx25-te-bf16", "ltx25-video-vae-conv", "ltx25-audio-vae"],
      note="10Eros (0.7) + DMD distill + IC-LoRA + detailer + img2vid fusi: 4-8 passi, CFG 1. Prompt opzionale "
           "[VISUAL]/[SPEECH]/[SOUNDS]"),
    N("ltx25-stubelius-remix-v1", "Video", "Stubelius Remix v1 LTX-2.5 (18+, INT8, distilled, 21.5GB)",
      category="video", arch="ltx25", repo="Stuubs/Stubelius_Remix_Ltx2.5",
      remote="ltx2.5-Stubelius_remix_v1_int8_convrot.safetensors",
      filename="ltx2.5-Stubelius_remix_v1_distilled_int8_convrot.safetensors", type="diffusion_models",
      family="LTX", gb=21.5, sens="adult",
      deps=["ltx25-te-gemma4-gguf|ltx25-te-bf16", "ltx25-video-vae-conv", "ltx25-audio-vae"],
      note="10Eros v1.5 + distilled-LoRA 450 + OmniNFT RL fusi: 8 passi, CFG 1. Licenza LTX-2.x Community"),
    # MiniMax H3: DiT «hybrid» (fl2va+ref2va fusi, stesso layout pruned int8 ConvRot) e LoRA kohya.
    # ⚠ licenza H3 Community: territori esclusi UE/UK/USA/Corea, anche per i derivati.
    N("mmh3-dit-pinkcherry18", "Video", "PinkCherry H3 v1.8 hybrid (18+, INT8, 21.1GB)", category="video",
      arch="minimax_h3", repo="SexGod1979/PinkCherry-NSFW_MiniMax-H3",
      remote="1.8-hybrid/PinkCherry_hybrid_v1.8_int8_minimax_h3_pruned.safetensors",
      filename="PinkCherry_hybrid_v1.8_int8_minimax_h3_pruned-native.safetensors", convert="mmh3_dit",
      type="diffusion_models", family="MiniMax-H3", gb=21.06, sens="adult", deps=["mmh3-configs", "mmh3-te-nvfp4|mmh3-te-fp8", "mmh3-video-vae", "mmh3-audio-vae"],
      used_by=["cap.video_minimax_h3"],
      note="finetune NSFW per FL2VA e Ref2VA (stesse chiavi del DiT ufficiale pruned). Licenza H3 Community (no UE/UK/USA/Corea)"),
    N("mmh3-dit-10erosmax-b6", "Video", "10Eros-Max H3 beta6 Turbo 8 passi (18+, INT8, 21.7GB)", category="video",
      arch="minimax_h3", repo="TenStrip/10Eros-Max", remote="10Eros_Max_h3_8stepTURBO_beta6_int8.safetensors",
      filename="10Eros_Max_h3_8stepTURBO_beta6_int8-native.safetensors", convert="mmh3_dit",
      type="diffusion_models", family="MiniMax-H3", gb=21.71, sens="adult", deps=["mmh3-configs", "mmh3-te-nvfp4|mmh3-te-fp8", "mmh3-video-vae", "mmh3-audio-vae"],
      used_by=["cap.video_minimax_h3"],
      note="turbo già fuso: 8-12 passi, shift 10/3, NIENTE LoRA turbo. Licenza H3 Community (no UE/UK/USA/Corea)"),
    N("mmh3-lora-unlocked-v2", "Video", "LoRA H3 NSFW Unlocked V2 (18+, 0.3GB)", category="video", arch="minimax_h3",
      repo="lynaNSFW/minimaxH3_Collection", remote="Minimax_H3_Unlocked_V2.safetensors",
      filename="Minimax_H3_Unlocked_V2-native.safetensors", convert="mmh3_lora", type="LoRA", family="MiniMax-H3",
      gb=0.3, sens="adult", used_by=["cap.video_minimax_h3"],
      note="la LoRA H3 più scaricata (Civitai 2925727), kohya rank 16"),
    N("mmh3-veda-r2va", "Video", "Veda attenzione sparsa H3 Ref2VA (predittore fp8, 0.28GB)", category="video",
      arch="minimax_h3", repo="Veda-Sparse/Minimax-H3-R2VA-Veda-Preview",
      remote="minimax_h3_r2va_veda_preview_fp8.safetensors",
      filename="minimax_h3_r2va_veda_preview_fp8.safetensors", type="model_patches", family="MiniMax-H3",
      gb=0.28, used_by=["cap.video_h3_veda"],
      note="predittore di attenzione sparsa appresa (Veda, ICML 2026): ~1.4-2.2× sul denoise in Ref2VA + Turbo 8 "
           "passi. Non è una LoRA: i pesi del DiT non cambiano. Licenza H3 Community"),
    N("mmh3-lora-hmcumshot-v2", "Video", "LoKr H3 HMCumshot V2 (18+, 0.63GB)", category="video", arch="minimax_h3",
      repo="lynaNSFW/minimaxH3_Collection", remote="HMCumshot_V2.safetensors",
      filename="HMCumshot_V2-native.safetensors", convert="mmh3_lora", type="LoRA", family="MiniMax-H3",
      gb=0.63, sens="adult", used_by=["cap.video_minimax_h3"],
      note="LoKr ai-toolkit (w1 8×8 ⊗ w2 piena): applicata al volo sui Linear quantizzati da _qlinear"),
    N("mmh3-lora-aftermidnight", "Video", "LoRA H3 AfterMidnight Ref2VA (18+, 1.2GB)", category="video",
      arch="minimax_h3", repo="SexGod1979/AfterMidnight-MiniMax-H3-NSFW",
      remote="AfterMidnight_ref2va_h3_sexytime_rank64-v1.2.safetensors",
      filename="AfterMidnight_ref2va_h3_rank64_v1.2-native.safetensors", convert="mmh3_lora", type="LoRA",
      family="MiniMax-H3", gb=1.19, sens="adult", used_by=["cap.video_minimax_h3"],
      note="solo modalità Ref2VA; l'autore usa euler + scheduler beta"),
    # Krea 2: LoRA ai-toolkit (diffusion_model.blocks… + txtfusion → text_fusion)
    # Krea 2 single-file community: convertito in nativo (nomi diffusers, Linear int8 ConvRot compressi) e montato
    # sul text encoder/VAE dello snapshot Turbo (workers/krea2_custom).
    N("krea2-noct-v4", "Immagini", "Krea Noct V4 (18+, INT8, Turbo 8 passi, 13.5GB)", category="image", arch="krea2",
      repo="Noctaluna/Krea2-Noct-Uncensored", remote="KreaNoct_V4_int8_convrot.safetensors",
      filename="KreaNoct_V4_int8_convrot-native.safetensors", convert="krea2_dit", type="diffusion_models",
      family="Krea2", gb=13.49, sens="adult", deps=["krea2_turbo"],
      note="merge fotorealistico NSFW su Krea 2 Turbo, niente LoRA di sblocco: 8 passi, CFG 1. Krea 2 Community License"),
    N("krea2-kroma-v031", "Immagini", "Krea Kroma v0.3.1 Turbo OPD (INT8 ConvRot, 8 passi, 14.1GB)", category="image",
      arch="krea2", repo="jakethessmake/Krea-Kroma-v0.3.1-Turbo-OPD-ComfyUI-Quants",
      remote="krea_kroma_v0.3.1_turbo_opd_int8_convrot.safetensors",
      filename="krea_kroma_v0.3.1_turbo_opd_int8_convrot-native.safetensors", convert="krea2_dit",
      type="diffusion_models", family="Krea2", gb=14.14, deps=["krea2_turbo"],
      note="finetune Kroma (lodestones) di Krea 2, quant ComfyUI int8 ConvRot: stesso layout di Krea Noct V4. "
           "8-12 passi, CFG 1-1.5. Krea 2 Community License"),
    N("krea2-te-uncensored", "Immagini", "Krea 2 text encoder Uncensored v1.1 (18+, fp8, 4.9GB)", category="image",
      arch="krea2", repo="ChrisColeTech/krea2-turbo-uncensored-v1.1-FP8",
      remote="split/text_encoders/qwen3vl_4b_uncensored-fp8_scaled.safetensors",
      filename="qwen3vl_4b_uncensored-fp8-native.safetensors", convert="krea2_te", type="TextEncoders",
      family="Krea2", gb=4.85, sens="adult", deps=["krea2_turbo|krea2_raw"],
      note="Qwen3-VL-4B senza rifiuti: stesse chiavi del TE ufficiale, Linear fp8 (scala per-tensore)"),
    N("krea2-lora-knp-v4", "Immagini", "LoRA Krea 2 NSFW V4 (KNP, 18+, 0.46GB)", category="image", arch="krea2",
      repo="randomforest/krea-2-nsfw-v4", remote="KNP_000003000.safetensors",
      filename="Krea2_NSFW_KNP_v4.safetensors", type="LoRA", family="Krea2", gb=0.46, sens="adult",
      note="Civitai 2725430 (105k download), rank 64. Apache-2.0"),
    N("krea2-lora-mysticxxx-v3", "Immagini", "LoRA Krea 2 Mystic XXX v3 (18+, 0.23GB)", category="image",
      arch="krea2", repo="yufusoft/MysticXXX_KREA2_v3", remote="MysticXXX_KREA2_v3.safetensors",
      filename="Krea2_MysticXXX_v3.safetensors", type="LoRA", family="Krea2", gb=0.23, sens="adult",
      note="Civitai 2728644 (54k download), rank 32"),
    # ── fonte CIVITAI (login obbligatorio: chiave API nel tab «Pesi modelli»; sha256 dall'API pubblica) ──
    N("krea2-lora-snofs-v15", "Immagini", "LoRA Krea 2 SNOFS v1.5 (18+, 1.5GB)", category="image", arch="krea2",
      url="https://civitai.com/api/download/models/3390681?fileId=3280058", filename="snofs_krea_v1_5.safetensors", type="LoRA", family="Krea2",
      gb=1.53, sens="adult", sha256="326a1f23ed4f728238d37d5c84b02086871d2a1e4c5c2d9e2b8b17ae17e14d1c",
      note="Civitai 1972981 «Sex, Nudes, Other Fun Stuff» per Krea 2 (bf16)"),
    N("krea2-lora-refusal-v2", "Immagini", "LoRA Krea 2 TextFusion Refusal-Reduction v2 (18+, 0.03GB)",
      category="image", arch="krea2", url="https://civitai.com/api/download/models/3359021?fileId=3246655", filename="refusal_reduction_v2.safetensors",
      type="LoRA", family="Krea2", gb=0.03, sens="adult",
      sha256="2b5ce7ce81a5b6f407e7aa6f5b631527abadb89c0a90a3cdec1da2ff18173f40",
      note="Civitai 2775340: agisce sul TextFusion, toglie i rifiuti sui prompt espliciti"),
    N("flux2-klein-realdream-v2", "Immagini", "Real Dream Klein 9B V2 (18+, fp8, 9.2GB)", category="image",
      arch="flux2", url="https://civitai.com/api/download/models/3108757?fileId=2988715", filename="realDream_klein9BV2.safetensors",
      type="diffusion_models", family="Flux", gb=9.21, sens="adult",
      sha256="346273ab9bc9b535689b42af7a597d7e2bad32ef11a06a4140196a0b311764d4",
      note="Civitai 153568: finetune fotorealistico di FLUX.2 Klein 9B"),
    N("zimage-biglove-zora1", "Immagini", "Big Love Zora1 Z-Image Turbo (18+, bf16, 12GB)", category="image",
      arch="zimage", url="https://civitai.com/api/download/models/3296380?fileId=3184279", filename="bigLove_zora1_bf16.safetensors",
      type="diffusion_models", family="Z-Image", gb=12.02, sens="adult",
      sha256="85bd0540418b01bb03115502cd060fbc39e094eb4b12ac5ed3f88ee4981dc7a5",
      note="Civitai 897413, file bf16 (il primario è fp8_mixed)"),
    N("zimage-lora-zitnsfw-v3", "Immagini", "LoRA Z-Image Turbo NSFW v3 (18+, 0.01GB)", category="image",
      arch="zimage", url="https://civitai.com/api/download/models/2898951?fileId=2776902", filename="ZITnsfwLoRAv3.safetensors", type="LoRA",
      family="Z-Image", gb=0.01, sens="adult",
      sha256="a3425142dcbe76bb5bbd226a0f23fe51b28c5c3c53c56a2e7aa20c8706a03042",
      note="Civitai 2279079: aggiorna la v2 (ZITnsfwLoRAv2, da cancellare dopo il download)"),
    # Z-Image Turbo: LoRA ai-toolkit con nomi diffusers (le «ZIB» sono per Z-Image Base: escluse)
    *[N(f"zimage-lora-{lid}", "Immagini", f"LoRA Z-Image NSFW {lbl} (18+, {gb}GB)", category="image",
        arch="zimage", repo="qqnyanddld/nsfw-z-image-lora", remote=f"{rem}.safetensors",
        filename=f"ZIT_NSFW_{lid}.safetensors", type="LoRA", family="Z-Image", gb=gb, sens="adult", note=note)
      for lid, rem, lbl, gb, note in (
          ("pussy", "lora-pussy", "anatomia", 0.34, "Civitai 41k download"),
          ("bukkake", "lora-bukkake", "bukkake", 0.17, "trigger «pornmaster bukkake,»"),
          ("open-pussy", "lora-open-pussy", "anatomia aperta", 0.34, "trigger «zitpussy»"),
          ("tentacled", "lora-tentacled", "tentacoli", 0.17, "versione Turbo"))],
    # Wan 2.1: encoder UMT5 senza censura (bf16: il loader MultiTalk rifiuta l'fp8 «scaled»)
    N("wan-umt5-nsfw-bf16", "Video", "Wan UMT5-XXL NSFW (18+, bf16, 11.4GB)", category="video", arch="wan21",
      repo="NSFW-API/NSFW-Wan-UMT5-XXL", remote="nsfw_wan_umt5-xxl_bf16.safetensors",
      filename="nsfw_wan_umt5-xxl_bf16.safetensors", type="TextEncoders", gb=11.36, sens="adult",
      note="sostituisce l'umt5 di MultiTalk (stesse chiavi). ⚠ MultiTalk oggi non parte: dipende da WanVideoWrapper, spostato in ToTrash"),

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
    N("facexlib-detection", "FaceRestore", "facexlib detection (RetinaFace R50)", category="aux",
      arch="gfpgan", filename="detection_Resnet50_Final.pth", type="facerestore_models",
      family="facexlib", gb=0.11, used_by=["cap.restore"],
      note="pre-seed: core/facexlib_seed lo collega in gfpgan/weights dove facexlib lo cerca"),
    N("facexlib-parsing", "FaceRestore", "facexlib parsing (ParseNet)", category="aux",
      arch="gfpgan", filename="parsing_parsenet.pth", type="facerestore_models",
      family="facexlib", gb=0.09, used_by=["cap.restore"],
      note="pre-seed: core/facexlib_seed lo collega in gfpgan/weights dove facexlib lo cerca"),
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

# url GitHub per voci file non-HF (il downloader FASE B le gestisce)
GH_URLS = {
    "gfpgan-13": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
    "facexlib-detection": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "facexlib-parsing": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
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


# Famiglie in cui NEW è la fonte autorevole anche per le voci già nel json (campi di NEW vincono).
REFRESH = ("mmh3-", "qwen21-", "lcav")

# Voci ritirate (tolte dal catalogo alla prossima rigenerazione).
REMOVED = {
    # GGUF di MiniMax H3: nessun loader nativo (i profili nvfp4/fp8/int8-ConvRot coprono tutte le GPU)
    "mmh3-fl2va-gguf-q4", "mmh3-fl2va-gguf-q3", "mmh3-ref2va-gguf-q4", "mmh3-te-gguf-q4",
    # 10Eros v1 (LTX-2.3) sostituito da 10Eros v1.5 (ltx23-eros15): il file vecchio va cancellato
    "ltx23-eros",
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

    # 2) voci NEW: nuove → aggiunte; esistenti → aggiornate SOLO per le famiglie in REFRESH (le altre
    #    possono avere ritocchi a mano nel json — type/family/note — che non vanno persi)
    added = updated = 0
    for n in NEW:
        if n["id"] in GH_URLS:
            n["url"] = GH_URLS[n["id"]]
        if n["id"] in by_id and not n["id"].startswith(REFRESH):
            continue
        if n["id"] in by_id:
            old = by_id[n["id"]]
            merged = dict(old)
            merged.update(n)                                    # i campi di NEW vincono
            for k in ("scan_names", "deps"):                    # liste: unione (curate a mano / use_deps)
                vals = list(n.get(k) or []) + [v for v in (old.get(k) or []) if v not in (n.get(k) or [])]
                if vals:
                    merged[k] = vals
            if merged != old:
                weights[weights.index(old)] = merged
                updated += 1
            n = merged
        else:
            weights.append(n)
            added += 1
        by_id[n["id"]] = n

    # 3) voci ritirate
    removed = [e["id"] for e in weights if e["id"] in REMOVED]
    weights = [e for e in weights if e["id"] not in REMOVED]

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
    print(f"voci: {n_tot} (aggiunte {added}, aggiornate {updated}, ritirate {removed}) — "
          f"scaricabili {n_dl}, no_remote {n_nr}")
    if dry:
        return 0
    shutil.copy2(CATALOG, CATALOG.with_suffix(".json.v1.bak"))
    CATALOG.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")
    json.loads(CATALOG.read_text("utf-8"))
    print(f"scritto {CATALOG} (backup .v1.bak)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
