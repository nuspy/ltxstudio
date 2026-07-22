"""Definizione VIDEO: restore + super-risoluzione + (opz.) aumento FPS, via ComfyUI.

Motori selezionabili:
- **seedvr2**: SeedVR2 (ByteDance, ICLR2026) one-step diffusion+GAN — SOTA open per video
  AI-generati/compressi. Nodi `numz/ComfyUI-SeedVR2_VideoUpscaler` (auto-download in
  ComfyUI/models/SEEDVR2). Modelli 3B/7B fp16/fp8/gguf.
- **esrgan**: upscale per-fotogramma (UpscaleModelLoader + ImageUpscaleWithModel) — veloce,
  niente coerenza temporale (fallback/anteprima).
- **flashvsr**: VSR streaming real-time (CVPR2026) — richiede il nodo FlashVSR (non installato).

Opzionale **RIFE** (comfyui-frame-interpolation, `RIFE VFI`) per aumentare gli FPS.
Carica con VHS_LoadVideo (file copiato nella input dir di ComfyUI), salva con VHS_VideoCombine
mantenendo l'audio originale.
"""
from __future__ import annotations

import re

# nomi modello SeedVR2 (auto-download in ComfyUI/models/SEEDVR2)
SEEDVR2_MODELS = {
    # fp8 3B per PRIMO = default GUI (#156): resta residente in VRAM, niente offload. I fp16 pieni
    # restano come scelta esplicita dell'utente (7B fp16 su 32GB = offload lentissimo).
    "SeedVR2 3B FP8":                 "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "SeedVR2 7B FP8 (leggero)":       "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "SeedVR2 3B FP16 (veloce)":       "seedvr2_ema_3b_fp16.safetensors",
    "SeedVR2 7B FP16 (qualità max)":  "seedvr2_ema_7b_fp16.safetensors",
}
SEEDVR2_VAE = "ema_vae_fp16.safetensors"


def native_scale(model_name: str) -> int:
    m = re.match(r"\s*(\d+)\s*x", (model_name or "").lower())
    return int(m.group(1)) if m else 4


class VideoEnhancePipeline:
    def build(self, video_name: str, *, engine: str = "seedvr2",
              # default = 3B fp8: ESISTE ed entra in VRAM; il 7b_fp16 è su disco solo come
              # .download INCOMPLETO (9.3GB parziali) → usarlo come default falliva sempre.
              seedvr2_model: str = "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
              resolution: int = 1080, blocks_to_swap: int = 0, batch_size: int = 5,
              esrgan_model: str = "4x-UltraSharpV2.safetensors", target_scale: float = 2.0,
              fps: int = 24, rife: bool = False, rife_multiplier: int = 2,
              rife_ckpt: str = "rife47.pth", seed: int = 0,
              out_format: str = "video/h264-mp4") -> dict:
        p: dict = {}
        # 1) carica i fotogrammi (file già copiato nella input dir di ComfyUI) + audio
        p["1"] = {"class_type": "VHS_LoadVideo",
                  "inputs": {"video": video_name, "force_rate": 0, "custom_width": 0,
                             "custom_height": 0, "frame_load_cap": 0, "skip_first_frames": 0,
                             "select_every_nth": 1, "format": "AnimateDiff"}}
        frames = ["1", 0]; audio = ["1", 2]

        if engine == "esrgan":                       # upscale per-fotogramma (no coerenza temporale)
            p["2"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": esrgan_model}}
            p["3"] = {"class_type": "ImageUpscaleWithModel",
                      "inputs": {"upscale_model": ["2", 0], "image": frames}}
            out = ["3", 0]
            nat = native_scale(esrgan_model)
            if target_scale and abs(target_scale - nat) > 0.05:
                p["4"] = {"class_type": "ImageScaleBy",
                          "inputs": {"image": ["3", 0], "upscale_method": "lanczos",
                                     "scale_by": float(target_scale) / float(nat)}}
                out = ["4", 0]
        elif engine == "flashvsr":
            raise RuntimeError("FlashVSR non installato: aggiungi il nodo ComfyUI FlashVSR, "
                               "oppure usa SeedVR2 / ESRGAN.")
        else:                                        # seedvr2 (default)
            p["2"] = {"class_type": "SeedVR2LoadDiTModel",
                      "inputs": {"model": seedvr2_model, "device": "cuda",
                                 "blocks_to_swap": int(blocks_to_swap), "swap_io_components": False,
                                 "offload_device": "cpu", "cache_model": False, "attention_mode": "sdpa"}}
            p["3"] = {"class_type": "SeedVR2LoadVAEModel",
                      "inputs": {"model": SEEDVR2_VAE, "device": "cuda",
                                 "encode_tiled": False, "encode_tile_size": 1024, "encode_tile_overlap": 128,
                                 "decode_tiled": False, "decode_tile_size": 256, "decode_tile_overlap": 32,
                                 "tile_debug": False, "offload_device": "cpu", "cache_model": False}}
            p["4"] = {"class_type": "SeedVR2VideoUpscaler",
                      "inputs": {"image": frames, "dit": ["2", 0], "vae": ["3", 0],
                                 "seed": int(seed), "resolution": int(resolution), "max_resolution": 4096,
                                 "batch_size": int(batch_size), "uniform_batch_size": True,
                                 "color_correction": "wavelet", "temporal_overlap": 0, "prepend_frames": 0,
                                 "input_noise_scale": 0.0, "latent_noise_scale": 0.0,
                                 "offload_device": "cpu", "enable_debug": False}}
            out = ["4", 0]

        out_fps = int(fps)
        if rife:                                     # aumento FPS (interpolazione)
            p["8"] = {"class_type": "RIFE VFI",
                      "inputs": {"ckpt_name": rife_ckpt, "frames": out,
                                 "clear_cache_after_n_frames": 10, "multiplier": int(rife_multiplier),
                                 "fast_mode": True, "ensemble": True, "scale_factor": 1.0}}
            out = ["8", 0]; out_fps = int(fps) * int(rife_multiplier)

        p["9"] = {"class_type": "VHS_VideoCombine",
                  "inputs": {"images": out, "frame_rate": out_fps, "loop_count": 0,
                             "filename_prefix": "ltxstudio_enhanced", "format": out_format,
                             "pingpong": False, "save_output": True, "audio": audio}}
        return p
