"""Percorsi dei pesi MultiTalk nativo — SOLO stdlib (importabile dalla GUI e dal servizio senza torch)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_WORKERS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = Path(os.environ.get("AI_MODELS_DIR", r"E:\AI\Models"))

def _first_existing(cands):
    for c in cands:
        if c and Path(c).exists():
            return str(c)
    return None


def resolve_paths(over: dict | None = None) -> dict:
    """Percorsi dei pesi (registro pesi -> posizioni storiche). `over` ha la precedenza (chiavi: dit, multitalk, vae,
    t5, t5_tokenizer, clip, wav2vec, lora). Ritorna {chiave: path|None}."""
    over = over or {}

    def cat(i, fb):
        try:
            sys.path.insert(0, os.path.dirname(_WORKERS))
            from core import model_paths as mp
            p = mp.path_of(i)
            return str(p) if p else fb
        except Exception:                             # noqa: BLE001
            return fb
    wan13 = cat("wan21_13b_original", str(MODELS / "Checkpoints" / "Wan13B" / "Wan2.1-T2V-1.3B"))
    p = {
        "dit": _first_existing([over.get("dit"), cat("wan21-i2v-14b-480p-fp8", ""),
                                MODELS / "diffusion_models" / "Wan2.1" / "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"]),
        "multitalk": _first_existing([over.get("multitalk"), cat("wan21-multitalk-14b-fp8", ""),
                                      MODELS / "diffusion_models" / "Wan2.1" / "WanVideo_2_1_Multitalk_14B_fp8_e4m3fn.safetensors",
                                      MODELS / "Checkpoints" / "Wan" / "WanVideo_2_1_Multitalk_14B_fp32.safetensors"]),
        "vae": _first_existing([over.get("vae"), Path(wan13) / "Wan2.1_VAE.pth", MODELS / "VAE" / "Wan2.1" / "Wan2.1_VAE.pth"]),
        "t5": _first_existing([over.get("t5"), Path(wan13) / "models_t5_umt5-xxl-enc-bf16.pth"]),
        "t5_tokenizer": _first_existing([over.get("t5_tokenizer"), Path(wan13) / "google" / "umt5-xxl"]),
        "clip": _first_existing([over.get("clip"), cat("wan-clip-vision-h", ""), MODELS / "VisionEncoders" / "clip_vision_h.safetensors"]),
        "wav2vec": _first_existing([over.get("wav2vec"), cat("wav2vec2-multitalk", "") and Path(cat("wav2vec2-multitalk", "")) / "pytorch_model.bin",
                                    MODELS / "Audio" / "Wan" / "chinese-wav2vec2-base" / "pytorch_model.bin",
                                    MODELS / "Audio" / "Wan2.1" / "pytorch_model.bin"]),
        "lora": _first_existing([over.get("lora"), cat("lightx2v-i2v-480p-rank64", ""), MODELS / "LoRA" / "Wan" / "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"]),
    }
    return p


def missing_required(paths: dict) -> list[str]:
    return [k for k in ("dit", "multitalk", "vae", "t5", "t5_tokenizer", "clip", "wav2vec") if not paths.get(k)]


