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
    face_strength: float = Field(0.85, description="quanto pesa il volto di `face_ref`. A 0.85 "
                                                   "l'identità è fedelissima ma arriva con tutto il "
                                                   "volto di partenza, ETÀ COMPRESA, e il prompt "
                                                   "non riesce più a ringiovanire il personaggio. "
                                                   "0.5-0.7 tiene i tratti e lascia contare il "
                                                   "prompt")
    ref_image: str = Field("", description="immagine di riferimento (Kontext)")
    ref_images: list | None = Field(None, description="riferimenti MULTIPLI (Kontext multi-ref, o "
                                                      "in-context con coherence_engine=krea2_edit)")
    coherence_engine: str = Field("flux", description="flux (PuLID/Kontext) | krea2_edit (in-context)")
    loras: list | None = None
    negative: str = ""
    structure: bool = Field(True, description="riscrive il prompt in JSON strutturato con l'LLM "
                                              "locale; False = il prompt passa così com'è")


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
    ltx_release: str | None = Field(None, description="release LTX: 2.3 (default) | 2.5")
    model: str | None = Field(None, description="transformer esplicito; auto per release != 2.3")
    references: list | None = Field(None, description="MSR: 2-5 riferimenti [{path,strength,role}]. "
                                                      "Sulla 2.5 entrano come slot embedding a frame "
                                                      "negativi: coerenza di cast e luoghi fra shot")
    ref_images: list | None = Field(None, description="ritratti/luoghi della scena (path). Diventano "
                                                      "riferimenti MSR da soli, con lo sfondo ricavato "
                                                      "dal keyframe: e' la via comoda per `references`")
    msr_strength: float = 1.0
    msr_background: str = Field("", description="MSR 2.3 (pseudo-video): sfondo obbligatorio")
    msr_spacing: int = Field(1, description="MSR 2.5: distanza temporale dei riferimenti dal primo "
                                            "fotogramma. 1 = ricetta del plugin (istanti contigui "
                                            "alla scena, il modello può generare una transizione "
                                            "dai riferimenti al video); valori maggiori li "
                                            "allontanano dalla timeline")
    structure: bool = Field(True, description="riscrive il prompt in JSON strutturato con l'LLM "
                                              "locale; False = il prompt passa così com'è")
    steps: int = Field(8, description="passi di denoise. 8 bastano al video su un checkpoint "
                                      "distilled, ma l'AUDIO viene denoisato dagli stessi passi: "
                                      "alzarli (20-30) dà una traccia più pulita, al costo di tempo")
    guidance: float = Field(3.0, description="CFG del video")
    native_audio: bool = Field(False, description="tiene la traccia audio che LTX-2 genera insieme "
                                                  "al video (suoni/parlato dal prompt + lip-sync). "
                                                  "Senza, il video esce muto: la traccia non "
                                                  "condizionata sarebbe rumore")
    audio_lowpass: int | None = Field(None, description="taglio in Hz della banda che il vocoder "
                                                        "BWE della 2.5 sintetizza sopra i 16 kHz "
                                                        "interni: è materiale inventato dal mel e "
                                                        "si sente come alone metallico. None = "
                                                        "8000 sulla 2.5 e nessun filtro sulla 2.3; "
                                                        "0 disattiva e restituisce la banda piena")


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
                                  ref_images=[r for r in (req.ref_images or []) if r],
                                  coherence_engine=req.coherence_engine, structure=req.structure,
                                  loras=parse_loras(req.loras), arch=req.arch, negative=req.negative,
                                  face_strength=req.face_strength)
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
                                       negative=req.negative, ltx_release=req.ltx_release,
                                       model=req.model or "", references=req.references,
                                       ref_images=[r for r in (req.ref_images or []) if r],
                                       msr_strength=req.msr_strength,
                                       msr_background=req.msr_background, structure=req.structure,
                                       native_audio=req.native_audio, msr_spacing=req.msr_spacing,
                                       steps=req.steps, guidance=req.guidance,
                                       audio_lowpass=req.audio_lowpass)
            return [out] if os.path.exists(out) else []
        return _submit("film.shot_video", fn)
