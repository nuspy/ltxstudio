"""Endpoint MUSICA / AUDIO — servizi nativi (venv isolati).

song (HeartMuLa/ACE-Step) · sonify (MMAudio) · music_video · tts · voice_clone · separate_vocals · transcribe.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from api import gate
from api.jobs import MANAGER
from api.handlers._common import out_path


def _submit(kind, fn):
    return {"job_id": MANAGER.submit(kind, fn), "status": "pending"}


class SongReq(BaseModel):
    name: str = "song"
    tags: str = Field("pop, upbeat, catchy", description="stile/genere/strumenti")
    lyrics: str = ""
    duration_s: int = 120
    instrumental: bool = False
    engine: str = Field("heartmula", description="heartmula | acestep")
    version: str = Field("3B", description="HeartMuLa: 3B | 7B")
    temperature: float = 1.0
    cfg_scale: float = 1.5
    guidance: float = Field(15.0, description="ACE-Step guidance")
    lazy_load: bool = False


class SonifyReq(BaseModel):
    name: str = "sfx"
    video_path: str = Field("", description="video da sonorizzare (vuoto = solo prompt→audio)")
    prompt: str = ""
    negative_prompt: str = ""
    duration_s: float = 8.0
    cfg_strength: float = 4.5
    num_steps: int = 25
    seed: int = 42


class MusicVideoReq(BaseModel):
    lyrics: str
    song_path: str = Field("", description="canzone già pronta (vuoto = genera prima)")
    image_path: str = Field(..., description="immagine iniziale/sfondo")
    width: int = 704
    height: int = 512
    fps: int = 25
    scene_seconds: float = 6.0
    video_style: str = ""
    steps: int = 8


class TTSReq(BaseModel):
    text: str
    engine: str = Field("edge", description="edge | kokoro | f5tts | xtts2 | fish | vibevoice | cosyvoice2")
    voice_id: str = "it-IT-ElsaNeural"
    language: str = "it-IT"
    sample_path: str | None = Field(None, description="campione voce per clone (f5/xtts/fish/vibevoice)")
    sample_transcript: str = ""
    speed: float = Field(1.0, ge=0.5, le=2.0, description="velocità (kokoro; edge → rate)")


class SeparateVocalsReq(BaseModel):
    audio_path: str


class TranscribeReq(BaseModel):
    audio_path: str


def register(app: FastAPI):

    @app.post("/music/song", tags=["musica"])
    def music_song(req: SongReq):
        gate.require("cap.music_song")
        def fn(job):
            if req.engine == "acestep":
                from core import acestep_service as ax
                _errs = ax.preflight(req.tags)               # una sola chiamata (era invocata 2×)
                if _errs:
                    raise RuntimeError("; ".join(_errs))
                j = ax.make_job(req.tags, "" if req.instrumental else req.lyrics, req.name,
                                duration=float(req.duration_s), guidance=req.guidance,
                                lazy=req.lazy_load, on_log=job.log)
                out = ax.OUT_DIR / f"{ax._safe(req.name)}.wav"
                j.run()
                return [str(out)] if out.exists() else []
            from core import music_service as ms
            p = ms.SongParams(name=req.name, lyrics=req.lyrics, tags=req.tags,
                              duration_s=req.duration_s, instrumental=req.instrumental,
                              version=req.version, temperature=req.temperature,
                              cfg_scale=req.cfg_scale, lazy_load=req.lazy_load)
            errs = ms.preflight_song(p)
            if errs:
                raise RuntimeError("; ".join(errs))
            j = ms.make_song_job(p, on_log=job.log)
            j.run()
            out = p.out_mp3()
            return [str(out)] if os.path.exists(str(out)) else []
        return _submit("music.song", fn)

    @app.post("/music/sonify", tags=["musica"])
    def music_sonify(req: SonifyReq):
        gate.require("cap.music_sonify")
        def fn(job):
            from core import music_service as ms
            p = ms.SonifyParams(name=req.name, video_path=req.video_path, prompt=req.prompt,
                                negative_prompt=req.negative_prompt, duration_s=req.duration_s,
                                cfg_strength=req.cfg_strength, num_steps=req.num_steps, seed=req.seed)
            errs = ms.preflight_sonify(p)
            if errs:
                raise RuntimeError("; ".join(errs))
            import time as _t
            _t0 = _t.time()
            j = ms.make_sonify_job(p, on_log=job.log)
            j.run()
            # SonifyParams non ha out_path(): MMAudio scrive in MUSIC_OUTPUT_DIR con nome dinamico →
            # prendi l'ultimo audio prodotto DOPO l'avvio del job (prima: p.out_path() inesistente → sempre []).
            out = ms.find_latest_audio(after=_t0)
            return [str(out)] if out and os.path.exists(str(out)) else []
        return _submit("music.sonify", fn)

    @app.post("/music/music_video", tags=["musica"])
    def music_video(req: MusicVideoReq):
        gate.require("cap.music_video")
        def fn(job):
            from core import music_video_service as mv
            out = out_path("music_video")
            mv.make_music_video(lyrics=req.lyrics, song_path=req.song_path,
                                image_path=req.image_path, out_path=out,
                                width=req.width, height=req.height, fps=req.fps,
                                scene_seconds=req.scene_seconds, video_style=req.video_style,
                                steps=req.steps)
            return [out] if os.path.exists(out) else []
        return _submit("music.music_video", fn)

    @app.post("/audio/tts", tags=["musica"])
    def audio_tts(req: TTSReq):
        gate.require("cap.audio_tts_clone")
        def fn(job):
            from core import tts_engine as tts
            voice = tts.VoicePreset(name="api", engine=req.engine, voice_id=req.voice_id,
                                    language=req.language, sample_path=req.sample_path,
                                    sample_transcript=req.sample_transcript)
            res = tts.synthesize(req.engine, voice, req.text, speed=req.speed)
            return [str(res)] if res and os.path.exists(str(res)) else []
        return _submit("audio.tts", fn)

    @app.post("/audio/separate_vocals", tags=["musica"])
    def audio_separate_vocals(req: SeparateVocalsReq):
        gate.require("cap.audio_tts_clone")                   # utility audio → stessa cap della voce
        def fn(job):
            from core import vocal_separation as vs
            if not vs.available():
                raise RuntimeError("demucs non disponibile.")
            out = vs.separate_vocals(req.audio_path, on_log=job.log)
            if not out or out == req.audio_path:
                return []
            # demucs scrive in una cartella temporanea (fuori da ALLOWED_ROOTS → download_url vuoto):
            # copia il vocals.wav in output/ così il link di download funziona.
            import shutil
            dst = out_path("vocals", "wav")
            shutil.copyfile(out, dst)
            return [dst]
        return _submit("audio.separate_vocals", fn)

    @app.post("/audio/transcribe", tags=["musica"])
    def audio_transcribe(req: TranscribeReq):
        gate.require("cap.audio_tts_clone")                   # utility audio → stessa cap della voce
        def fn(job):
            from core import handy_api
            txt = handy_api.transcribe(req.audio_path)
            out = out_path("transcript", "txt")
            with open(out, "w", encoding="utf-8") as f:
                f.write(txt or "")
            return [out]
        return _submit("audio.transcribe", fn)
