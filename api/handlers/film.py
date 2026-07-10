"""Endpoint FILM STUDIO — primitive generative a livello di inquadratura.

script (trattamento+scene via LLM) · asset_image (character/keyframe) · shot_video (storyboard shot).
L'orchestrazione di PROGETTO completa (DB, scheduler, montaggio, doppiaggio, modalità Auto) resta
nella GUI Film Studio (stateful). Il videoclip canzone→video è in /music/music_video.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from api import gate
from api.jobs import MANAGER
from api.handlers._common import out_path, parse_loras


def _submit(kind, fn):
    return {"job_id": MANAGER.submit(kind, fn), "status": "pending"}


class ScriptReq(BaseModel):
    subject: str = Field(..., description="soggetto/logline del film")
    style: str = ""
    max_tokens: int = 2000


class AssetImageReq(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    seed: int = 7
    arch: str = "flux"
    face_ref: str = Field("", description="foto identità (PuLID)")
    ref_image: str = Field("", description="immagine di riferimento (Kontext)")
    loras: list | None = None
    negative: str = ""


class ShotVideoReq(BaseModel):
    prompt: str
    width: int = 768
    height: int = 512
    fps: int = 25
    duration_s: float = 4.0
    seed: int = 7
    keyframes: list | None = None
    loras: list | None = None
    audio_path: str = ""
    negative: str = ""


def register(app: FastAPI):

    @app.post("/film/script", tags=["film"])
    def film_script(req: ScriptReq):
        gate.require("cap.filmstudio")
        def fn(job):
            from core import llm_service
            if not llm_service.available():
                raise RuntimeError("LLM non disponibile (Gemma locale o LM Studio :1234).")
            style = f" Stile: {req.style}." if req.style else ""
            system = ("Sei uno sceneggiatore. Produci un trattamento con logline, personaggi e "
                      "una scaletta di scene numerate, ognuna con luogo, azione e battute chiave.")
            txt = llm_service.complete(system, req.subject + style, max_tokens=req.max_tokens)
            out = out_path("film_script", "txt")
            with open(out, "w", encoding="utf-8") as f:
                f.write(txt or "")
            return [out]
        return _submit("film.script", fn)

    @app.post("/film/asset_image", tags=["film"])
    def film_asset_image(req: AssetImageReq):
        gate.require("cap.filmstudio")
        def fn(job):
            from core.film_studio import gen_service
            out = out_path("film_asset", "png")
            gen_service.gen_image(prompt=req.prompt, out_png=out, width=req.width, height=req.height,
                                  seed=req.seed, face_ref=req.face_ref, ref_image=req.ref_image,
                                  loras=parse_loras(req.loras), arch=req.arch, negative=req.negative)
            return [out] if os.path.exists(out) else []
        return _submit("film.asset_image", fn)

    @app.post("/film/shot_video", tags=["film"])
    def film_shot_video(req: ShotVideoReq):
        gate.require("cap.filmstudio")
        def fn(job):
            from core.film_studio import gen_service
            out = out_path("film_shot")
            gen_service.gen_shot_video(prompt=req.prompt, out_mp4=out, width=req.width,
                                       height=req.height, fps=req.fps, duration_s=req.duration_s,
                                       seed=req.seed, keyframes=req.keyframes or None,
                                       loras=parse_loras(req.loras), audio_path=req.audio_path,
                                       negative=req.negative)
            return [out] if os.path.exists(out) else []
        return _submit("film.shot_video", fn)
