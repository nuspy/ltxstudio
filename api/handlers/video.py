"""Endpoint VIDEO — generazione via workers/ltx2_infer.py (venv ltx2) + post-process nativi.

generate (t2v/i2v/a2v) · keyframes · extend · retake · v2v-edit · controlnet · lipsync
· enhance (SeedVR2/FlashVSR) · rife (fps) · faceswap. Tutto async → GET /jobs/{id}.
"""
from __future__ import annotations

import os
import subprocess

from fastapi import FastAPI
from pydantic import BaseModel, Field

from api import gate
from api.jobs import MANAGER
from api.handlers._common import run_ltx2, parse_loras, out_path, apply_ltx_release


def _submit(kind, fn):
    return {"job_id": MANAGER.submit(kind, fn), "status": "pending"}


def _r64(x: int) -> int:
    """LTX-2 (pipeline two-stage) richiede lati multipli di 64."""
    return max(64, int(round(x / 64)) * 64)


def _base(req) -> dict:
    nf = max(9, int(req.length_s * req.fps) // 8 * 8 + 1)
    p = {"prompt": req.prompt, "negative": req.negative or "worst quality, blurry",
         "width": _r64(req.width), "height": _r64(req.height), "fps": req.fps, "steps": req.steps,
         "guidance": req.guidance, "seed": req.seed, "num_frames": nf}
    if nf > 150:                                          # video lungo → chunk memory-bounded (no OOM)
        p["chunk_frames"] = 121
    if getattr(req, "loras", None):
        p["loras"] = parse_loras(req.loras)
    # Release/checkpoint e riferimenti multipli (MSR): assenti → resta il default 2.3 di sempre.
    apply_ltx_release(p, getattr(req, "ltx_release", None), getattr(req, "model", "") or "",
                      getattr(req, "references", None),
                      float(getattr(req, "msr_strength", 1.0) or 1.0),
                      getattr(req, "msr_background", "") or "")
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
    ltx_release: str | None = Field(None, description="release LTX: 2.3 (default) | 2.5")
    model: str | None = Field(None, description="transformer esplicito; auto per release != 2.3")
    references: list | None = Field(None, description="MSR: 2-5 riferimenti [{path,strength,role}] "
                                                      "per la coerenza di personaggi/luoghi")
    msr_strength: float = 1.0
    msr_background: str = Field("", description="MSR 2.3 (pseudo-video): sfondo obbligatorio")


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
    ltx_release: str | None = Field(None, description="release LTX: 2.3 (default) | 2.5")
    model: str | None = Field(None, description="transformer esplicito; auto per release != 2.3")
    references: list | None = Field(None, description="MSR: 2-5 riferimenti per la coerenza")
    msr_strength: float = 1.0
    msr_background: str = ""


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


class IcLoraReq(BaseModel):
    """Modalita' IC-LoRA di «Modifica Video AI» (stessa logica della GUI: core/video_edit_ic.py)."""
    kind: str = Field(..., description="refocus | uncompress | deinterlace | motiondeblur | detailer | "
                                       "outpaint | restyle | cameraman")
    video_path: str = Field(..., description="video sorgente (cameraman: video di RIFERIMENTO del movimento)")
    prompt: str = ""
    strength: float = 1.0
    attention: float = Field(1.0, ge=0.1, le=1.0, description="attenzione della guida")
    keep_audio: bool = True
    res_mode: str = Field("source", description="source | half | custom (restauro/restyle)")
    res_long: int = 0
    # restyle
    style_image: str | None = None
    style_text: str = Field("", description="stile a parole: prompt «Make it {style} style.» se prompt vuoto")
    # outpaint
    canvas: str = Field("16:9", description="16:9 | 9:16 | 1:1 | 4:3 | 3:4 | 21:9 | custom")
    anchor: str = "center"
    margins: list[int] | None = Field(None, description="custom: [sopra, destra, sotto, sinistra] px sorgente")
    dark_source: bool = False
    paste_original: bool = True
    # cameraman
    image_path: str | None = None
    image_strength: float = 0.7
    width: int = 960
    height: int = 576
    seconds: float = 0.0
    # checkpoint: ESATTAMENTE come scelto (mai sostituito); vuoto = default del worker (dev)
    model: str | None = None
    ltx_release: str | None = None
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
    scale: float = 2.0


class RifeReq(BaseModel):
    video_path: str
    factor: int = 2


class FaceSwapVideoReq(BaseModel):
    video_path: str
    source_face_path: str


def register(app: FastAPI):

    @app.post("/video/generate", tags=["video"])
    def video_generate(req: GenerateReq):
        gate.require("cap.video")                             # rispetta l'edizione come la GUI (403 se off)
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
        gate.require("cap.video")
        p = _base(req); p["keyframes"] = req.keyframes; p["output"] = out_path("video_kf")
        return _submit("video.keyframes", lambda job: run_ltx2(p, job))

    @app.post("/video/extend", tags=["video"])
    def video_extend(req: ExtendReq):
        gate.require("cap.extend_retake")
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
        gate.require("cap.extend_retake")
        p = {"retake": True, "video_path": req.video_path, "prompt": req.prompt,
             "start_time": req.start_time, "end_time": req.end_time,
             "width": req.width, "height": req.height, "fps": req.fps, "steps": req.steps,
             "seed": req.seed, "output": out_path("video_retake")}
        return _submit("video.retake", lambda job: run_ltx2(p, job))

    @app.post("/video/v2v_edit", tags=["video"])
    def video_v2v_edit(req: V2VEditReq):
        gate.require("cap.video")
        p = {"control_video": req.control_video, "edit_lora": req.edit_lora,
             "control_strength": req.control_strength, "prompt": req.prompt,
             "width": req.width, "height": req.height, "fps": req.fps, "steps": req.steps,
             "seed": req.seed, "num_frames": max(9, int(req.length_s * req.fps) // 8 * 8 + 1),
             "output": out_path("video_v2v")}
        return _submit("video.v2v_edit", lambda job: run_ltx2(p, job))

    @app.post("/video/controlnet", tags=["video"])
    def video_controlnet(req: ControlNetReq):
        gate.require("cap.controlnet")
        p = {"control_video": req.control_video, "control_type": req.control_type,
             "control_strength": req.control_strength, "prompt": req.prompt,
             "width": req.width, "height": req.height, "fps": req.fps, "steps": req.steps,
             "seed": req.seed, "num_frames": max(9, int(req.length_s * req.fps) // 8 * 8 + 1),
             "output": out_path("video_cnet")}
        if req.image_path:
            p["image_path"] = req.image_path
        return _submit("video.controlnet", lambda job: run_ltx2(p, job))

    @app.post("/video/ic_lora", tags=["video"])
    def video_ic_lora(req: IcLoraReq):
        """Restauro (refocus/uncompress/deinterlace/motiondeblur/detailer), outpaint, restyle, cameraman.
        Risposta: job_id + `warnings` (es. checkpoint non distilled: MAI sostituito)."""
        from fastapi import HTTPException
        from core import ltx_release as lrel
        from core import video_edit_ic as vic
        from core import video_edit_modes as vem
        from core.video_ic_prep import OUTPAINT_FORMATS, ANCHORS
        spec = vem.spec_for_kind(req.kind)
        if spec is None or not spec.cap or req.kind == "union":
            valid = sorted(s.kind for s in vem.MODE_SPECS.values() if s.enabled and s.cap)
            raise HTTPException(422, {"error": "invalid_kind", "kind": req.kind, "valid": valid})
        gate.require(spec.cap)
        if req.kind == "outpaint":
            if req.canvas != "custom" and req.canvas not in OUTPAINT_FORMATS:
                raise HTTPException(422, {"error": "invalid_canvas", "valid": [*OUTPAINT_FORMATS, "custom"]})
            if req.anchor not in ANCHORS:
                raise HTTPException(422, {"error": "invalid_anchor", "valid": list(ANCHORS)})
            if req.canvas == "custom" and not (req.margins and len(req.margins) == 4 and any(req.margins)):
                raise HTTPException(422, {"error": "margins_required",
                                          "detail": "custom richiede 4 margini [sopra,destra,sotto,sinistra], almeno uno > 0"})
        if req.kind == "restyle" and not (req.style_image and os.path.exists(req.style_image)):
            raise HTTPException(422, {"error": "style_image_required"})
        if not os.path.exists(req.video_path):
            raise HTTPException(422, {"error": "video_not_found", "path": req.video_path})
        warnings: list[str] = []
        model = (req.model or "").strip()
        rel = (req.ltx_release or (lrel.detect_release(model) if model else None) or "")
        if rel and not lrel.ic_lora_supported(rel):
            raise HTTPException(422, {"error": "release_unsupported", "release": rel,
                                      "detail": "le IC-LoRA locali sono per LTX 2.3"})
        if not lrel.is_distilled(model):
            if spec.distilled == vem.DISTILLED_REQUIRED:
                raise HTTPException(422, {"error": "distilled_required", "mode": spec.key})
            warnings.append("checkpoint non distilled" + ("" if model else " (default del worker: dev)")
                            + ": qualita' ridotta, la distilled-LoRA vale solo per lo stage 1")
        miss = vic.missing_weight(spec)
        if miss:
            raise HTTPException(409, {"error": "weight_missing", "detail": miss})
        prompt = req.prompt.strip()
        if req.kind == "restyle" and not prompt and req.style_text.strip():
            prompt = f"Make it {req.style_text.strip()} style."
        cfg = {"mode": spec.key, "video": req.video_path, "prompt": prompt, "strength": req.strength,
               "conditioning_attention": req.attention, "keep_audio": req.keep_audio,
               "res_mode": req.res_mode, "res_long": req.res_long, "style_image": req.style_image,
               "out_format": req.canvas, "out_anchor": req.anchor,
               "out_margins": tuple(req.margins or (0, 0, 0, 0)), "out_dark": req.dark_source,
               "out_paste": req.paste_original, "cam_ref": req.video_path, "cam_img": req.image_path or "",
               "cam_img_strength": req.image_strength, "cam_seconds": req.seconds,
               "width": req.width, "height": req.height, "seed": req.seed}
        if model:
            cfg["gguf_transformer" if model.lower().endswith(".gguf") else "model"] = model
        if rel:
            cfg["ltx_release"] = rel

        def fn(job):
            import time as _t
            output = out_path(f"video_ic_{spec.key}")
            ij = vic.build_job(cfg, os.path.dirname(output), int(_t.time()), output, log=lambda m: job.log(m))
            outs = run_ltx2(ij.params, job)
            return [vic.finalize(ij, outs[0], log=lambda m: job.log(m))]
        r = _submit("video.ic_lora", fn)
        r["warnings"] = warnings
        return r

    @app.post("/video/lipsync", tags=["video"])
    def video_lipsync(req: LipsyncReq):
        gate.require("cap.video")
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
        gate.require("cap.restore")
        def fn(job):
            out = out_path(f"{req.engine}")
            if req.engine == "flashvsr":
                from core import flashvsr_service as fv
                _errs = fv.preflight()                        # una sola chiamata (era invocata 2×)
                if _errs:
                    raise RuntimeError("; ".join(_errs))
                fv.upscale_video(req.video_path, out, scale=req.scale,
                                 on_log=lambda l: job.log(l))
            else:
                from core import seedvr2_service as sv
                _errs = sv.preflight()                        # una sola chiamata (era invocata 2×)
                if _errs:
                    raise RuntimeError("; ".join(_errs))
                sv.upscale_video(req.video_path, out, resolution=req.resolution,
                                 model=req.seedvr2_model, batch_size=req.batch_size,
                                 block_swap=req.block_swap, on_log=lambda l: job.log(l))
            return [out]
        return _submit("video.enhance", fn)

    @app.post("/video/rife", tags=["video"])
    def video_rife(req: RifeReq):
        gate.require("cap.restore")
        def fn(job):
            from core import rife
            if not rife.available():
                raise RuntimeError("Practical-RIFE non installato.")
            out = out_path("rife")
            out = rife.interpolate_video(req.video_path, out, factor=req.factor)
            try:  # #107 firma invisibile di provenienza (default ON) — PRIMA del watermark, video esterno
                from core import stego
                if stego.enabled():
                    stego.sign_file(out)
            except Exception:                               # noqa: BLE001
                pass
            # #106 watermark obbligatorio: core/rife.py NON marchia di suo (nelle catene GUI
            # gen→RIFE / enhance→RIFE l'input è GIÀ marchiato alla fonte → marcarlo lì sarebbe un
            # doppio watermark). Questo endpoint standalone lavora su video esterni → si marchia
            # QUI l'output consegnato. RuntimeError → errore del job (fail-closed).
            try:
                from core import watermark
                out = watermark.apply_to_file(out)
            except RuntimeError:
                raise
            except Exception:                               # noqa: BLE001
                pass
            return [out]
        return _submit("video.rife", fn)

    @app.post("/video/faceswap", tags=["video"])
    def video_faceswap(req: FaceSwapVideoReq):
        gate.require("cap.video")
        def fn(job):
            from core import faceswap
            if not faceswap.available():
                raise RuntimeError("Face-swap non disponibile (inswapper/insightface).")
            out = out_path("faceswap_video")
            return [faceswap.swap_video(req.video_path, req.source_face_path, out)]
        return _submit("video.faceswap", fn)
