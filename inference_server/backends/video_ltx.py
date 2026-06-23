"""Handler VIDEO nativo (diffusers LTX-2) — sostituisce ComfyUI per i video.

Genera text-to-video (+audio) con LTX2Pipeline assemblata da:
- transformer LOCALE 2.3 (from_single_file) + VAE video LOCALE (from_single_file)
- audio_vae / connectors / vocoder / scheduler / tokenizer / processor: componenti diffusers LTX-2
- text_encoder: Gemma-3 (cartella diffusers in TE_GEMMA)

NB: le classi connectors/vocoder sono interne al package (diffusers.pipelines.ltx2.*).
Stato: PoC t2v. i2v/extend/long/controlnet/audio-sync → fasi successive (task #18).
"""
from __future__ import annotations

import os
from pathlib import Path

LTX2_DIR = r"E:\AI\Models\diffusers\ltx2"                       # componenti + config diffusers LTX-2
TRANSFORMER = r"E:\AI\Models\diffusion_models\LTX\ltx-2.3-22b-dev-fp8.safetensors"
VAE_VIDEO = r"E:\AI\Models\VAE\LTX\LTX2_video_vae.safetensors"
TE_GEMMA = r"E:\AI\Models\diffusers\_te\gemma3_ltx_fp8"        # Gemma-3 12B fp8 (MISHANM, ~12GB)
VRAM_MB = 30000


def _load(_model: str = "", _arch: str = "video"):
    """Assembla LTX2Pipeline. Ritorna (pipe, vram_mb) per il ModelManager."""
    import torch
    from diffusers import (LTX2VideoTransformer3DModel, AutoencoderKLLTX2Video,
                           AutoencoderKLLTX2Audio, FlowMatchEulerDiscreteScheduler)
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
    from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline
    from transformers import Gemma3ForConditionalGeneration, AutoTokenizer
    bf16 = torch.bfloat16
    tr = LTX2VideoTransformer3DModel.from_single_file(
        TRANSFORMER, config=os.path.join(LTX2_DIR, "transformer"), torch_dtype=bf16)
    vae = AutoencoderKLLTX2Video.from_single_file(
        VAE_VIDEO, config=os.path.join(LTX2_DIR, "vae"), torch_dtype=bf16)
    avae = AutoencoderKLLTX2Audio.from_pretrained(os.path.join(LTX2_DIR, "audio_vae"), torch_dtype=bf16)
    conn = LTX2TextConnectors.from_pretrained(os.path.join(LTX2_DIR, "connectors"), torch_dtype=bf16)
    voc = LTX2Vocoder.from_pretrained(os.path.join(LTX2_DIR, "vocoder"), torch_dtype=bf16)
    te = Gemma3ForConditionalGeneration.from_pretrained(TE_GEMMA, torch_dtype=bf16)
    tok = AutoTokenizer.from_pretrained(os.path.join(LTX2_DIR, "tokenizer"))
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(LTX2_DIR, "scheduler"))
    proc = None
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(os.path.join(LTX2_DIR, "processor"))
    except Exception:                                          # noqa: BLE001
        pass
    pipe = LTX2Pipeline(scheduler=sched, vae=vae, audio_vae=avae, text_encoder=te, tokenizer=tok,
                        connectors=conn, transformer=tr, vocoder=voc, processor=proc)
    pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_tiling()
    except Exception:                                          # noqa: BLE001
        pass
    print("[video] LTX2 assemblata (transformer 2.3 + vae locali + componenti LTX-2)", flush=True)
    return pipe, VRAM_MB


def run(params: dict, mm, output_dir: str) -> dict:
    import torch
    from diffusers.utils import export_to_video
    key = "video:ltx2"
    mm.register(key, _load)
    pipe = mm.ensure(key, need_mb=VRAM_MB)            # LTX2Pipeline (t2v); componenti condivisi
    seed = int(params.get("seed", 0))
    g = torch.Generator(device="cuda").manual_seed(seed)
    kw = dict(
        prompt=params.get("prompt", ""),
        negative_prompt=params.get("negative") or None,
        width=int(params.get("width", 768)),
        height=int(params.get("height", 768)),
        num_frames=int(params.get("num_frames", 49)),
        frame_rate=int(params.get("fps", 24)),
        num_inference_steps=int(params.get("steps", 30)),
        guidance_scale=float(params.get("guidance", 3.0)),
        generator=g,
    )
    # i2v: stessa pipeline-family, RIUSA i componenti già caricati (no ricarica pesi/VRAM)
    image_path = params.get("image_path")
    if image_path:
        from PIL import Image
        from diffusers import LTX2ImageToVideoPipeline
        pipe = LTX2ImageToVideoPipeline(
            scheduler=pipe.scheduler, vae=pipe.vae, audio_vae=pipe.audio_vae,
            text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer, connectors=pipe.connectors,
            transformer=pipe.transformer, vocoder=pipe.vocoder, processor=pipe.processor)
        kw["image"] = Image.open(image_path).convert("RGB")
    out = pipe(**kw)
    os.makedirs(output_dir, exist_ok=True)
    path = str(Path(output_dir) / f"native_ltx2_{seed}.mp4")
    frames = out.frames[0] if hasattr(out, "frames") else out[0]
    export_to_video(frames, path, fps=int(params.get("fps", 24)))
    # TODO fase audio: muxare out.audio (LTX2 genera audio joint) nel file finale.
    return {"status": "success", "output": [path]}
