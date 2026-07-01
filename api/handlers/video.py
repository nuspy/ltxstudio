"""Endpoint VIDEO — generazione via workers/ltx2_infer.py (venv ltx2) + post-process nativi.

generate (t2v/i2v/a2v) · keyframes · extend · retake · v2v-edit · controlnet · lipsync
· enhance (SeedVR2/FlashVSR) · rife (fps) · faceswap. Tutto async → GET /jobs/{id}.
"""
from __future__ import annotations

import os
import subprocess

from fastapi import FastAPI
from pydantic import BaseModel, Field

from api.jobs import MANAGER
from api.handlers._common import run_ltx2, parse_loras, out_path


def _submit(kind, fn):
    return {"job_id": MANAGER.submit(kind, fn), "status": "pending"}


def _base(req) -> dict:
    p = {"prompt": req.prompt, "negative": req.negative or "worst quality, blurry",
         "width": req.width, "height": req.height, "fps": req.fps, "steps": req.steps,
         "guidance": req.guidance, "seed": req.seed,
         "num_frames": max(9, int(req.length_s * req.fps) // 8 * 8 + 1)}
    if getattr(req, "loras", None):
        p["loras"] = parse_loras(req.loras)
    return p


class GenerateReq(BaseModel):
    prompt: str
    negative: str | None = None
    width: int = 768
    height: int = 512
    fps: int = 25
    length_s: float = 5.0
    steps: int = 40
    guidance: float = 3.0
    seed: int = 42
    loras: list | None = None
    image_path: str | None = Field(None, description="i2v: foto di partenza")
    ref_strength: float = 1.0
    i2v_method: str = Field("addguide", description="addguide | inplace (soft)")
    audio_path: str | None = Field(None, description="a2v: audio → video (lip-sync/canto)")
    audio_scale: float = 3.0


class KeyframesReq(BaseModel):
    prompt: str
    keyframes: list = Field(..., description="[{path, frame:int, strength:float}]")
    negative: str | None = None
    width: int = 768
    height: int = 512
    fps: int = 25
    length_s: float = 5.0
    steps: int = 40
    guidance: float = 3.0
    seed: int = 42
    loras: list | None = None


class ExtendReq(BaseModel):
    prompt: str
    video_path: str
    add_seconds: float = 5.0
    seg_seconds: float = 0.0
    steps: int = 8
    negative: str | None = None


class RetakeReq(BaseModel):
    prompt: str
    video_path: str
    start_time: float = 0.0
    end_time: float = 0.0
    width: int = 768
    height: int = 512
    fps: int = 25
    steps: int = 40
    seed: int = 42


class V2VEditReq(BaseModel):
    prompt: str
    control_video: str
    edit_lora: str = Field(..., description="LoRA editing, es. ltx23_edit_anything.safetensors")
    control_strength: float = 1.0
    width: int = 768
    height: int = 512
    fps: int = 25
    length_s: float = 5.0
    steps: int = 40
    seed: int = 42


class ControlNetReq(BaseModel):
    prompt: str
    control_video: str
    control_type: str = Field("canny", description="canny | depth | pose")
    control_strength: float = 0.8
    image_path: str | None = None
    width: int = 768
    height: int = 512
    fps: int = 25
    length_s: float = 5.0
    steps: int = 40
    seed: int = 42


class LipsyncReq(BaseModel):
    image_path: str
    audio_path: str
    prompt: str = ""
    negative: str | None = None
    width: int = 512
    height: int = 512
    fps: int = 25
    audio_scale: float = 3.0
    separate_vocals: bool = False
    talk_region: str | None = Field(None, description="mouth | face (composita solo la regione)")
    talk_mask_offset: int = 0


class EnhanceReq(BaseModel):
    video_path: str
    engine: str = Field("seedvr2", description="seedvr2 | flashvsr")
    seedvr2_model: str = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    resolution: int = 1080
    batch_size: int = 5
    block_swap: int = 0
    scale: float = 4.0


class RifeReq(BaseModel):
    video_path: str
    factor: int = 2


class FaceSwapVideoReq(BaseModel):
    video_path: str
    source_face_path: str


def register(app: FastAPI):

    @app.post("/video/generate", tags=["video"])
    def video_generate(req: GenerateReq):
        p = _base(req)
        if req.audio_path:                                   # a2v (lip-sync/canto)
            p.update(audio_path=req.audio_path, audio_scale=req.audio_scale,
                     ref_strength=req.ref_strength)
            if req.image_path:
                p["image_path"] = req.image_path
        elif req.image_path:                                 # i2v
            p.update(image_path=req.image_path, ref_strength=req.ref_strength,
                     i2v_method=req.i2v_method)
        p["output"] = out_path("video_gen")
        return _submit("video.generate", lambda job: run_ltx2(p, job))

    @app.post("/video/keyframes", tags=["video"])
    def video_keyframes(req: KeyframesReq):
        p = _base(req); p["keyframes"] = req.keyframes; p["output"] = out_path("video_kf")
        return _submit("video.keyframes", lambda job: run_ltx2(p, job))

    @app.post("/video/extend", tags=["video"])
    def video_extend(req: ExtendReq):
        def fn(job):
            src = req.video_path
            fps = 25
            # ultimo frame → i2v per add_seconds → concat originale + continuazione
            lastf = out_path("ext_last", "png")
            subprocess.run(["ffmpeg", "-v", "error", "-sseof", "-0.12", "-i", src,
                            "-vframes", "1", lastf, "-y"], check=False)
            if not os.path.exists(lastf):
                raise RuntimeError("impossibile estrarre l'ultimo frame")
            nf = max(9, int(req.add_seconds * fps) // 8 * 8 + 1)
            seg = int(req.seg_seconds * fps) // 8 * 8 + 1 if req.seg_seconds > 0 else 121
            cont = out_path("ext_cont")
            p = {"image_path": lastf, "prompt": req.prompt,
                 "negative": req.negative or "worst quality, blurry",
                 "num_frames": nf, "fps": fps, "steps": req.steps, "ref_strength": 1.0,
                 "output": cont}
            if nf > seg:
                p["chunk_frames"] = max(9, seg)
            run_ltx2(p, job)
            final = out_path("video_extended")
            lst = out_path("_concat", "txt")
            with open(lst, "w") as f:
                f.write(f"file '{src}'\nfile '{cont}'\n")
            subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                            "-c", "copy", final, "-y"], check=False)
            return [final if os.path.exists(final) else cont]
        return _submit("video.extend", fn)

    @app.post("/video/retake", tags=["video"])
    def video_retake(req: RetakeReq):
        p = {"retake": True, "video_path": req.video_path, "prompt": req.prompt,
             "start_time": req.start_time, "end_time": req.end_time,
             "width": req.width, "height": req.height, "fps": req.fps, "steps": req.steps,
             "seed": req.seed, "output": out_path("video_retake")}
        return _submit("video.retake", lambda job: run_ltx2(p, job))

    @app.post("/video/v2v_edit", tags=["video"])
    def video_v2v_edit(req: V2VEditReq):
        p = {"control_video": req.control_video, "edit_lora": req.edit_lora,
             "control_strength": req.control_strength, "prompt": req.prompt,
             "width": req.width, "height": req.height, "fps": req.fps, "steps": req.steps,
             "seed": req.seed, "num_frames": max(9, int(req.length_s * req.fps) // 8 * 8 + 1),
             "output": out_path("video_v2v")}
        return _submit("video.v2v_edit", lambda job: run_ltx2(p, job))

    @app.post("/video/controlnet", tags=["video"])
    def video_controlnet(req: ControlNetReq):
        p = {"control_video": req.control_video, "control_type": req.control_type,
             "control_strength": req.control_strength, "prompt": req.prompt,
             "width": req.width, "height": req.height, "fps": req.fps, "steps": req.steps,
             "seed": req.seed, "num_frames": max(9, int(req.length_s * req.fps) // 8 * 8 + 1),
             "output": out_path("video_cnet")}
        if req.image_path:
            p["image_path"] = req.image_path
        return _submit("video.controlnet", lambda job: run_ltx2(p, job))

    @app.post("/video/lipsync", tags=["video"])
    def video_lipsync(req: LipsyncReq):
        def fn(job):
            audio = req.audio_path
            if req.separate_vocals:
                from core import vocal_separation as vs
                job.log("separazione voce (demucs)…", 0.1)
                audio = vs.separate_vocals(audio, on_log=lambda l: job.log(l))
            p = {"image_path": req.image_path, "audio_path": audio, "prompt": req.prompt,
                 "negative": req.negative or "worst quality, blurry",
                 "width": req.width, "height": req.height, "fps": req.fps,
                 "audio_scale": req.audio_scale, "ref_strength": 1.0,
                 "output": out_path("video_lipsync")}
            outs = run_ltx2(p, job)
            if req.talk_region and outs:
                from core import talk_region as tr
                comp = outs[0].rsplit(".", 1)[0] + "_region.mp4"
                return [tr.composite_talk_region(outs[0], req.image_path, comp,
                        region=req.talk_region, mask_offset=req.talk_mask_offset)]
            return outs
        return _submit("video.lipsync", fn)

    @app.post("/video/enhance", tags=["video"])
    def video_enhance(req: EnhanceReq):
        def fn(job):
            out = out_path(f"{req.engine}")
            if req.engine == "flashvsr":
                from core import flashvsr_service as fv
                if fv.preflight():
                    raise RuntimeError("; ".join(fv.preflight()))
                fv.upscale_video(req.video_path, out, scale=req.scale,
                                 on_log=lambda l: job.log(l))
            else:
                from core import seedvr2_service as sv
                if sv.preflight():
                    raise RuntimeError("; ".join(sv.preflight()))
                sv.upscale_video(req.video_path, out, resolution=req.resolution,
                                 model=req.seedvr2_model, batch_size=req.batch_size,
                                 block_swap=req.block_swap, on_log=lambda l: job.log(l))
            return [out]
        return _submit("video.enhance", fn)

    @app.post("/video/rife", tags=["video"])
    def video_rife(req: RifeReq):
        def fn(job):
            from core import rife
            if not rife.available():
                raise RuntimeError("Practical-RIFE non installato.")
            out = out_path("rife")
            return [rife.interpolate_video(req.video_path, out, factor=req.factor)]
        return _submit("video.rife", fn)

    @app.post("/video/faceswap", tags=["video"])
    def video_faceswap(req: FaceSwapVideoReq):
        def fn(job):
            from core import faceswap
            if not faceswap.available():
                raise RuntimeError("Face-swap non disponibile (inswapper/insightface).")
            out = out_path("faceswap_video")
            return [faceswap.swap_video(req.video_path, req.source_face_path, out)]
        return _submit("video.faceswap", fn)
