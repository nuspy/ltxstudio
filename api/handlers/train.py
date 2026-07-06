"""Endpoint TRAINING LoRA (immagini e video) — job async, con gating feature-matrix (anti-bypass).

- POST /train/lora_image → dataset (immagini+caption) → core/lora_image_service.train.
- POST /train/lora_video → clip video → core/lora_video_service.make_lora_from_clips.
I file dataset arrivano già caricati via /files/upload (path in output/api_uploads). Gate: cap.lora_image_train
/ cap.lora_video_train (STESSE attivazioni del desktop).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from api import gate
from api.config import OUTPUT_DIR
from api.jobs import MANAGER


class _Img(BaseModel):
    path: str
    caption: str = ""


class LoraImageReq(BaseModel):
    name: str
    images: list[_Img] = Field(..., description="immagini dataset già caricate (+caption)")
    base: str = "flux2-dev"
    trigger: str = ""
    rank: int = 16
    steps: int = 2000


class _Clip(BaseModel):
    path: str
    caption: str = ""


class LoraVideoReq(BaseModel):
    name: str
    clips: list[_Clip] = Field(..., description="clip video già caricati")
    trigger: str = ""
    global_caption: str = ""
    rank: int = 16
    steps: int = 2000


def register(app: FastAPI):

    @app.post("/train/lora_image", tags=["training"])
    def train_lora_image(req: LoraImageReq):
        gate.require("cap.lora_image_train")

        def fn(job):
            from core import lora_image_service as svc
            ds = OUTPUT_DIR / "api_datasets" / f"img_{req.name}_{job.id[:8]}"
            ds.mkdir(parents=True, exist_ok=True)
            for i, item in enumerate(req.images):
                ext = Path(item.path).suffix or ".png"
                shutil.copy2(item.path, ds / f"{i:04d}{ext}")
                (ds / f"{i:04d}.txt").write_text(item.caption or req.trigger, encoding="utf-8")
            out = str(OUTPUT_DIR / "loras")
            lora = svc.train(str(ds), req.name, out, base=req.base, trigger=req.trigger,
                             rank=req.rank, steps=req.steps, progress=lambda m: job.log(m))
            return [lora]
        return {"job_id": MANAGER.submit("train.lora_image", fn), "status": "pending"}

    @app.post("/train/lora_video", tags=["training"])
    def train_lora_video(req: LoraVideoReq):
        gate.require("cap.lora_video_train")

        def fn(job):
            from core import lora_video_service as svc
            clips = [{"path": c.path, "caption": c.caption} for c in req.clips]
            out = str(OUTPUT_DIR / "loras")
            res = svc.make_lora_from_clips(clips, req.name, out, global_caption=req.global_caption,
                                           trigger=req.trigger, rank=req.rank, steps=req.steps,
                                           progress=lambda m: job.log(m))
            return [res.get("lora")] if isinstance(res, dict) else [res]
        return {"job_id": MANAGER.submit("train.lora_video", fn), "status": "pending"}
