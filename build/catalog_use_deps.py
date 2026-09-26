# -*- coding: utf-8 -*-
"""Assegna a OGNI voce del catalogo v2 i TAG D'USO (`use`, vocabolario core/weights_tooltips)
e le CATENE DI DIPENDENZE (`deps`): un checkpoint tira i componenti che gli servono e il
downloader/li scarica in automatico (deps-first, già supportato da plan_ids/CLI/dropdown).

Idempotente: sovrascrive use/deps con la mappa qui sotto (fonte di verità curatoriale;
la gestione fine passa poi dal CRUD tools/catalog_manager). Uso: python build/catalog_use_deps.py
"""
from __future__ import annotations

import json
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[1] / "data" / "weights_catalog.json"

# id → (use_tags, deps)  — deps SOLO verso voci scaricabili (mai no_remote)
MAP: dict[str, tuple[list[str], list[str]]] = {
    # ── LTX 2.3 ──
    "ltx23-dev-fp8": (["ckpt_video", "dev_quality"],
                      ["ltx23-gemma-te", "ltx23-distilled-lora-384", "ltx23-spatial-upscaler"]),
    "ltx23-distilled-fp8": (["ckpt_video", "distilled"],
                            ["ltx23-gemma-te", "ltx23-spatial-upscaler"]),
    "ltx23-gguf-q4-distilled": (["ckpt_video", "distilled", "gguf_small"],
                                ["ltx23-gemma-te", "ltx23-spatial-upscaler"]),
    "ltx23-transformer-only-fp8scaled": (["ckpt_video", "int8_comfy"], []),
    "ltx23": (["ckpt_video", "big_quality"], []),
    "ltx23-gemma-te": (["te"], []),
    "gemma3-12b-fp8-comfy": (["te", "te_quant", "int8_comfy"], []),
    "gemma-abliterated": (["te"], []),
    "ltx23-spatial-upscaler": (["upscaler_latent"], []),
    "ltx23-temporal-upscaler": (["upscaler_temporal"], []),
    "ltx23-distilled-lora-384": (["distill_lora"], []),
    "ltx23-eros": (["ckpt_video"], []),
    "ltx-iclora-canny": (["iclora_control"], []),
    "ltx-iclora-depth": (["iclora_control"], []),
    "ltx-iclora-pose": (["iclora_control"], []),
    "ltx-iclora-union": (["iclora_control"], []),
    "ltx-iclora-motion-track": (["iclora_control"], []),
    "ltx-iclora-detailer": (["iclora_edit"], []),
    "ltx-iclora-outpaint": (["iclora_edit"], []),
    "ltx-iclora-refocus": (["iclora_edit"], []),
    "ltx-iclora-uncompress": (["iclora_edit"], []),
    "ltx-iclora-deinterlace": (["iclora_edit"], []),
    "ltx-iclora-motiondeblur": (["iclora_edit"], []),
    "ltx-iclora-hdr": (["iclora_edit"], []),
    "ltx-iclora-lipdub": (["iclora_edit"], []),
    "ltx-iclora-restyle": (["iclora_edit"], []),
    # ── LTX 2.5 (catena: transformer → TE gguf + VAE video/audio) ──
    "ltx25-transformer-nvfp4": (["ckpt_video", "distilled", "nvfp4"],
                                ["ltx25-te-gemma4-gguf", "ltx25-video-vae-conv", "ltx25-audio-vae"]),
    "ltx25-transformer-int8": (["ckpt_video", "distilled", "int8_comfy"],
                               ["ltx25-te-gemma4-gguf", "ltx25-video-vae-conv", "ltx25-audio-vae"]),
    "ltx25-gguf-q2-distilled": (["ckpt_video", "distilled", "gguf_small", "q_extreme"],
                                ["ltx25-te-gemma4-gguf", "ltx25-video-vae-conv", "ltx25-audio-vae"]),
    "ltx25-gguf-q4-distilled": (["ckpt_video", "distilled", "gguf_small"],
                                ["ltx25-te-gemma4-gguf", "ltx25-video-vae-conv", "ltx25-audio-vae"]),
    "ltx25-gguf-q6-distilled": (["ckpt_video", "distilled", "gguf_small"],
                                ["ltx25-te-gemma4-gguf", "ltx25-video-vae-conv", "ltx25-audio-vae"]),
    "ltx25-te-bf16": (["te", "big_quality"], []),
    "ltx25-te-nvfp4-local": (["te", "te_quant", "nvfp4"], []),
    "ltx25-te-gemma4-gguf": (["te", "te_quant", "gguf_small"], []),
    "ltx25-video-vae-conv": (["vae"], []),
    "ltx25-audio-vae": (["audio_vae"], []),
    "ltx25-diffusion-decoder": (["vae", "big_quality"], []),
    "ltx25-spatial-upscaler": (["upscaler_latent"], []),
    "ltx25-temporal-upscaler": (["upscaler_temporal"], []),
    "ltx25-distilled-lora-450": (["distill_lora"], []),
    "ltx25-duration-head": ([], []),
    "ltx25-iclora-pixel-upscaler": (["iclora_edit", "upscaler_latent"], []),
    "ltx25-lora-multisubject": (["lora_style", "multisubject"], []),
    # ── immagini ──
    "flux1-dev-single": (["ckpt_image"], ["flux-vae-ae", "t5xxl-fp16"]),
    "flux1-fill-single": (["ckpt_image"], ["flux-vae-ae", "t5xxl-fp16"]),
    "flux1-kontext-single": (["ckpt_image"], ["flux-vae-ae", "t5xxl-fp16"]),
    "flux1-krea-single": (["ckpt_image"], ["flux-vae-ae", "t5xxl-fp16"]),
    "flux2-dev-fp8mixed": (["ckpt_image"], ["flux2-vae", "mistral-te-fp8-single"]),
    "flux1-dev": (["ckpt_image", "big_quality"], []),
    "flux1-fill": (["ckpt_image", "big_quality"], []),
    "flux1-kontext": (["ckpt_image", "big_quality"], []),
    "flux1-krea": (["ckpt_image", "big_quality"], []),
    "flux2-dev": (["ckpt_image", "big_quality"], []),
    "zimage-turbo": (["ckpt_image", "distilled"], []),
    "ernie-image": (["ckpt_image"], []),
    "krea2_turbo": (["ckpt_image", "distilled"], []),
    "krea2_raw": (["ckpt_image"], []),
    "mistral-flux2-te": (["te", "big_quality"], []),
    "mistral-te-fp8-single": (["te", "te_quant"], []),
    "qwen3-8b-klein-te": (["te"], []),
    "qwen3-4b-klein-te": (["te"], []),
    "flux2-klein-te-4b": (["te"], []),
    "flux2-klein-te-8b": (["te", "te_quant"], []),
    "t5xxl-fp16": (["te"], []),
    "long-clip-hi-dream": (["te"], []),
    "ernie-ministral": (["te"], []),
    "flux-vae-ae": (["vae"], []),
    "flux2-vae": (["vae"], []),
    "pulid-flux": (["face_tools"], []),
    "svdq-fp4-fluxfill": (["ckpt_image", "nvfp4"], []),
    # ── video: altre architetture ──
    "lingbot_dense": (["ckpt_video"], []),
    "lingbot_dense_fp8": (["ckpt_video"], []),
    "lingbot_moe": (["ckpt_video", "big_quality"], []),
    "lingbot_moe_uint4": (["ckpt_video", "gguf_small"], []),
    "lingbot_rewriter_lora": (["lora_style"], []),
    "lingbot_rewriter_base": (["te", "big_quality"], []),
    "krea_rt": (["ckpt_video"], []),
    "self_forcing_dmd": (["ckpt_video", "distilled"], []),
    "wan21_14b_diffusers": ([], []),
    "wan21_13b_diffusers": ([], []),
    "wan21_13b_original": ([], []),
    "wav2vec2-multitalk": ([], []),
    "seedvr2-3b-fp8": (["upscale_video"], []),
    "seedvr2-vae": (["upscale_video", "vae"], []),
    "flashvsr-v11": (["upscale_video"], []),
    # ── aux ──
    "sam2-hiera-large": (["segment"], []),
    "grounding-dino-base": (["segment"], []),
    "depth-anything-v2": (["depth_ctrl"], []),
    "inswapper-128": (["face_tools"], []),
    "antelopev2": (["face_tools"], []),
    "gfpgan-14": (["faceres"], []),
    "gfpgan-13": (["faceres"], []),
    "codeformer": (["faceres"], []),
    "facexlib-detection": (["faceres"], []),
    "facexlib-parsing": (["faceres"], []),
    "ultrasharp-v2": (["upscale_image"], []),
    "nomos8k-schat-l": (["upscale_image"], []),
    "joycaption-nf4": (["caption"], []),
    # ── musica / voce / 3D ──
    "heartmula-gen": (["music"], []),
    "heartmula-3b": (["music"], []),
    "heartcodec": (["music"], []),
    "mmaudio": (["music"], []),
    "acestep-3.5b": (["music"], []),
    "voice-fish": (["tts"], []),
    "voice-vibevoice": (["tts"], []),
    "voice-cosyvoice2": (["tts"], []),
    "f5-tts": (["tts"], []),
    "xtts-v2": (["tts"], []),
    "trellis2-4b": (["threed"], []),
    "dinov3-vitl16": (["threed"], []),
    "birefnet": (["threed"], []),
    "skintoken-grpo": (["threed"], []),
    "da2": (["threed", "depth_ctrl"], []),
    "worldgen-lora": (["threed", "lora_style"], []),
    "ltx23-lora-multisubject": (["lora_style", "multisubject"], []),
    # ── MiniMax H3 (deps già scritte da catalog_v2; qui i tag d'uso) ──
    # ── LongCat Avatar 1.5 ──
    "lcav-configs": (["te"], []),
    "lcav-te": (["te"], []),
    "lcav-vae": (["vae"], []),
    "lcav15-core": (["avatar", "distill_lora", "audio_enc"], ["lcav-configs", "lcav-te", "lcav-vae"]),
    "lcav15-int8": (["avatar", "ckpt_video", "q_int8"], ["lcav15-core"]),
    "lcav15-fp8": (["avatar", "ckpt_video", "fp8"], ["lcav15-core"]),
    "lcav15-bf16": (["avatar", "ckpt_video", "big_quality"], ["lcav15-core"]),
    "mmh3-configs": (["te"], []),                    # config+tokenizer: componente obbligatoria
    "mmh3-fl2va-nvfp4": (["ckpt_video", "av_joint", "nvfp4"], []),
    "mmh3-ref2va-nvfp4": (["ckpt_video", "av_joint", "ref_omni", "nvfp4"], []),
    "mmh3-fl2va-int8": (["ckpt_video", "av_joint", "q_int8"], []),
    "mmh3-fl2va-fp8": (["ckpt_video", "av_joint"], []),
    "mmh3-fl2va-bf16": (["ckpt_video", "av_joint", "big_quality"], []),
    "mmh3-fl2va-gguf-q4": (["ckpt_video", "av_joint", "gguf_small"], []),
    "mmh3-fl2va-gguf-q3": (["ckpt_video", "av_joint", "gguf_small", "q_extreme"], []),
    "mmh3-ref2va-int8": (["ckpt_video", "av_joint", "ref_omni", "q_int8"], []),
    "mmh3-ref2va-fp8": (["ckpt_video", "av_joint", "ref_omni"], []),
    "mmh3-ref2va-gguf-q4": (["ckpt_video", "av_joint", "ref_omni", "gguf_small"], []),
    "mmh3-te-nvfp4": (["te", "te_quant", "nvfp4"], []),
    "mmh3-te-int8": (["te", "te_quant", "q_int8"], []),
    "mmh3-te-gguf-q4": (["te", "te_quant", "gguf_small"], []),
    "mmh3-video-vae": (["vae"], []),
    "mmh3-audio-vae": (["audio_vae"], []),
    "mmh3-turbo-lora-fl2v": (["distill_lora", "lora_style"], []),
    "mmh3-turbo-lora-ref2v": (["distill_lora", "lora_style", "ref_omni"], []),
}


def main() -> int:
    data = json.loads(CATALOG.read_text("utf-8"))
    unknown = []
    n_use = n_deps = 0
    ids = {e["id"] for e in data["weights"]}
    for e in data["weights"]:
        got = MAP.get(e["id"])
        if got is None:
            unknown.append(e["id"])
            continue
        use, deps = got
        if use:
            e["use"] = use
            n_use += 1
        else:
            e.pop("use", None)
        # deps: unione con quelle già presenti (krea_rt→wan ecc. restano), mai auto-riferimenti
        merged = list(dict.fromkeys((e.get("deps") or []) + deps))
        bad = [d for d in merged if d not in ids]
        assert not bad, f"{e['id']}: deps inesistenti {bad}"
        if merged:
            e["deps"] = merged
            n_deps += 1
        # niente else-pop: non tocco deps preesistenti se la mappa è vuota
    CATALOG.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")
    json.loads(CATALOG.read_text("utf-8"))
    print(f"use su {n_use} voci, deps su {n_deps}; NON mappate: {unknown or 'nessuna'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
