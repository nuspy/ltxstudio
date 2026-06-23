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
TE_GEMMA = r"E:\AI\Models\TextEncoders\gemma3"                 # Gemma-3 12B bf16 HF locale (già presente, 13.7GB)
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
    # ── #24 video editing nativo (roadmap, da abilitare a Gemma scaricato + test API):
    #   ControlNet / ReTake → diffusers.LTX2ConditionPipeline(**{comp meno 'processor'}),
    #     call(conditions=[...], prompt=...). 'conditions' = riferimenti video/immagine (depth/canny/pose
    #     per controlnet; sezione+maschera per ReTake). Stessi 8 componenti già caricati.
    #   Durata illimitata → diffusers.LTXI2VLongMultiPromptPipeline (prompt_segments + cond_image/
    #     cond_strength) OPPURE catena autoregressiva LTX2 (riprogettare: era instabile su ComfyUI).
    #   i2v keyframe multipli → LTX2ImageToVideoPipeline con più frame guida.
    out = pipe(**kw)
    os.makedirs(output_dir, exist_ok=True)
    path = str(Path(output_dir) / f"native_ltx2_{seed}.mp4")
    frames = out.frames[0] if hasattr(out, "frames") else out[0]
    export_to_video(frames, path, fps=int(params.get("fps", 24)))
    # TODO fase audio: muxare out.audio (LTX2 genera audio joint) nel file finale.
    return {"status": "success", "output": [path]}


def run_condition(params: dict, mm, output_dir: str) -> dict:
    """#24 — video editing CONDIZIONATO (keyframe multipli / ReTake / extend / ControlNet) via
    LTX2ConditionPipeline. params['conditions'] = [{'image_path', 'index', 'strength'}] (frame guida
    a posizioni date). ControlNet: passare frame pre-processati (depth/canny/pose). Test a Gemma."""
    import torch
    from PIL import Image
    from diffusers import LTX2ConditionPipeline
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
    from diffusers.utils import export_to_video
    key = "video:ltx2"
    mm.register(key, _load)
    pipe = mm.ensure(key, need_mb=VRAM_MB)
    cpipe = LTX2ConditionPipeline(
        scheduler=pipe.scheduler, vae=pipe.vae, audio_vae=pipe.audio_vae,
        text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer, connectors=pipe.connectors,
        transformer=pipe.transformer, vocoder=pipe.vocoder)
    conds = []
    for c in params.get("conditions", []):
        img = Image.open(c["image_path"]).convert("RGB")
        conds.append(LTX2VideoCondition(frames=[img], index=int(c.get("index", 0)),
                                        strength=float(c.get("strength", 1.0))))
    seed = int(params.get("seed", 0))
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = cpipe(conditions=conds, prompt=params.get("prompt", ""),
                negative_prompt=params.get("negative") or None,
                width=int(params.get("width", 768)), height=int(params.get("height", 768)),
                num_frames=int(params.get("num_frames", 49)), frame_rate=int(params.get("fps", 24)),
                num_inference_steps=int(params.get("steps", 30)),
                guidance_scale=float(params.get("guidance", 3.0)), generator=g)
    os.makedirs(output_dir, exist_ok=True)
    path = str(Path(output_dir) / f"native_ltxcond_{seed}.mp4")
    frames = out.frames[0] if hasattr(out, "frames") else out[0]
    export_to_video(frames, path, fps=int(params.get("fps", 24)))
    return {"status": "success", "output": [path]}


def run_long(params: dict, mm, output_dir: str) -> dict:
    """#24 — DURATA ILLIMITATA riprogettata (era instabile su ComfyUI): catena di segmenti, ognuno
    i2v condizionato sull'ULTIMO frame del precedente; stitch ffmpeg. Memory-bounded (1 seg/volta)."""
    import os
    import subprocess
    import tempfile
    from pathlib import Path
    from inference_server.backends.video_postprocess import _ffmpeg
    total = int(params.get("total_frames", 200))
    seg_len = int(params.get("num_frames", 97))
    n_seg = max(1, -(-total // seg_len))                    # ceil
    td = tempfile.mkdtemp(prefix="ltxlong_")
    ff = _ffmpeg()
    segs, last_image = [], params.get("image_path")
    for i in range(n_seg):
        p = dict(params); p["num_frames"] = seg_len; p["seed"] = int(params.get("seed", 0)) + i
        if last_image:
            p["image_path"] = last_image
        else:
            p.pop("image_path", None)                       # primo segmento: t2v
        res = run(p, mm, td)
        if res.get("status") != "success":
            return res
        segs.append(res["output"][0])
        last_image = os.path.join(td, f"last_{i}.png")       # ultimo frame → guida del prossimo
        subprocess.run([ff, "-y", "-sseof", "-0.2", "-i", segs[-1], "-frames:v", "1", last_image],
                       capture_output=True, timeout=60)
    os.makedirs(output_dir, exist_ok=True)
    out = str(Path(output_dir) / f"native_long_{params.get('seed', 0)}.mp4")
    lst = os.path.join(td, "list.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for s in segs:
            f.write(f"file '{s}'\n")
    subprocess.run([ff, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out],
                   capture_output=True, timeout=300)
    return {"status": "success", "output": [out], "segments": len(segs)}
