"""Endpoint AVATAR PARLANTE — foto/personaggio + voce → video di una persona che parla.

POST /avatar/generate  job asincrono (immagine A/B/C/D → voce/dialogo/musica/effetti → video → post)
GET  /avatar/options   motori, voci, lingue, formati e profili pesi INSTALLATI (per gli agenti)

Immagine: A `text` (prompt → immagine) · B `image` (file) · C `character` (1 riferimento o asset «@Nome» +
modifiche: pettinatura, abiti, sfondo…) · D `character_multi` (più riferimenti) · `none` (solo testo).
Motore immagine sempre esplicito (default qwen_image21): pesi mancanti = errore chiaro, mai sostituzioni.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from api import gate
from api.jobs import MANAGER


class CharacterSpec(BaseModel):
    ref_images: list[str] = Field(default_factory=list, description="riferimenti (≤10 Qwen-Image 2.1, ≤2 Krea 2 Edit)")
    asset: str = Field("", description="personaggio di Film Studio: «@Nome» o «Nome»")
    changes: dict = Field(default_factory=dict,
                          description="hair, outfit, background, expression, pose, lighting, extra (testo libero)")
    keep_background: bool = Field(True, description="se background non è indicato, mantiene quello originale")


class ImageSpec(BaseModel):
    mode: str = Field("image", description="text | image | character | character_multi | none")
    prompt: str = Field("", description="A: descrizione; C/D: istruzione aggiuntiva")
    negative: str = ""
    engine: str = Field("qwen_image21", description="qwen_image21 | krea2_edit | krea2 | flux | flux2 | zimage")
    image_path: str = Field("", description="B: file (via /files/upload)")
    character: CharacterSpec | None = None
    aspect: str = Field("", description="Qwen-Image 2.1: formato 2K 1:1 4:3 3:4 3:2 2:3 16:9 9:16")
    width: int = 0
    height: int = 0
    seed: int = -1
    steps: int = 0
    profile: str = Field("fp8", description="Qwen-Image 2.1: nvfp4 | fp8 | int8 (vedi /avatar/options)")
    turbo: str = Field("", description="Qwen-Image 2.1: '' | '4' | '6' passi (LoRA turbo)")
    enhance_prompt: bool = False
    model: str = Field("", description="FLUX/Z-Image: checkpoint esplicito")


class SpeakerSpec(BaseModel):
    name: str = "A"
    engine: str = Field("edge", description="kokoro | edge | handy_clone | cosyvoice2 | f5tts | xtts2 | fish | vibevoice")
    voice_id: str = Field("", description="voce della libreria o id del motore (it-IT-ElsaNeural, if_sara…)")
    voice_description: str = Field("", description="se voice_id è vuoto: l'LLM sceglie la voce dalla descrizione")
    clone: dict | None = Field(None, description="{sample_path, transcript} per clonare da un campione")
    language: str = Field("", description=(
        "lingua del TESTO che questo speaker legge (it-IT, en-US, fr…). Vuota = quella della voce scelta "
        "(voce della libreria, id Edge 'xx-XX-…' o Kokoro 'if_…'); obbligatoria con voice_description, clone o "
        "senza voce. È distinta dalla lingua della VOCE: Edge/Kokoro sono monolingua (testo in un'altra lingua "
        "= accento), clone e voci Edge 'Multilingual' sono cross-lingua. Frasi in altra lingua: [lang:en]…[/lang]"))
    allow_accent: bool = Field(False, description="testo in una lingua diversa da quella della voce con accento "
                                                  "VOLUTO: niente avviso")
    speed: float = Field(1.0, ge=0.5, le=2.0)
    pitch: float = Field(0.0, ge=-12, le=12, description="semitoni")
    volume_db: float = Field(0.0, ge=-30, le=12)
    emotion: str = Field("", description="neutral | happy | sad | angry | whisper | calm")


class AudioSpec(BaseModel):
    mode: str = Field("file", description="file | tts | dialogue (| generated_by_engine con minimax_h3)")
    audio_path: str = ""
    audio2_path: str = Field("", description="persona 2 (file)")
    trim_start: float = 0.0
    trim_end: float = 0.0
    text: str = Field("", description="tts: testo; dialogue: copione «A: …» / «B: …» ([pause:500ms] ammesso)")
    language: str = Field("", description="lingua del testo predefinita per gli speaker senza language "
                                           "(e per le battute H3 senza language)")
    speakers: list[SpeakerSpec] = Field(default_factory=list)
    turn_mode: str = Field("add", description="due file: add = a turno | para = insieme")
    gap_ms: int = 350
    pad_start_ms: int = 0
    pad_end_ms: int = 0
    target_duration_s: float = 0.0
    fit: str = Field("pad", description="pad | speed | error")
    music: dict = Field(default_factory=dict, description="{prompt | path, engine: acestep|heartmula, volume_db, ducking}")
    sfx: dict = Field(default_factory=dict, description="{mode: none|prompt|auto, prompt, volume_db}")
    normalize_lufs: float = 0.0
    vocal_separation: bool = True
    voice_reference_path: str = Field("", description="minimax_h3: timbro voce di riferimento")


class VideoSpec(BaseModel):
    prompt: str = ""
    negative: str = ""
    enhance_prompt: bool = False
    resolution: str = Field("480p", description="480p | 720p")
    aspect: str = Field("16:9", description="senza immagine: 16:9 9:16 1:1 4:3 3:4")
    resize_mode: str = Field("crop", description="crop | default")
    max_duration_s: float = 0.0
    seed: int = -1
    ref_img_index: int = Field(10, ge=0, le=30)
    mask_frame_range: int = Field(3, ge=1, le=8)
    precision_profile: str = Field("int8", description="LongCat: int8 | fp8 | bf16 · MiniMax H3: nvfp4 | fp8 | int8")
    offload_kv_cache: bool = False
    multi_person: dict = Field(default_factory=dict, description="{bboxes: {person1:[y0,x0,y1,x1], person2:[…]}} frazioni 0–1")
    h3: dict = Field(default_factory=dict, description=(
        "minimax_h3: duration_s (5–15), dialogue_lines [{speaker, language, text}], soundscape, music_desc, "
        "turbo_steps ('' | '4' | '8'), text_encoder (id catalogo, es. mmh3-te-nvfp4)"))
    postprocess: dict = Field(default_factory=dict, description="{face_restore: codeformer|gfpgan, interpolate_fps: 50}")


class AvatarReq(BaseModel):
    engine: str = Field("longcat", description="longcat | minimax_h3")
    image: ImageSpec = Field(default_factory=ImageSpec)
    audio: AudioSpec = Field(default_factory=AudioSpec)
    video: VideoSpec = Field(default_factory=VideoSpec)
    return_intermediates: bool = True


def _submit(kind, fn):
    return {"job_id": MANAGER.submit(kind, fn), "status": "pending"}


def _languages() -> dict:
    """Lingue: quelle del TESTO che ogni motore sa leggere + come si comporta la VOCE (mono/cross-lingua)."""
    from core import lang_codes as LC
    from core.avatar.audio_chain import DEFAULT_VOICES
    return {
        "codes": {k: {"bcp47": b, "name": n} for k, (b, n, _a) in LC.LANGS.items()},
        "text_languages_by_engine": {e: list(v) for e, v in LC.ENGINE_TEXT_LANGS.items()},
        "minimax_h3": [LC.LANGS[k][1] for k in LC.H3_LANGS],
        "cross_lingual_voices": sorted(LC.CLONE_ENGINES) + ["edge: voci '…MultilingualNeural'"],
        "monolingual_voices": "edge (lingua = prefisso dell'id, es. it-IT-…), kokoro (prima lettera: a/b=en, "
                              "i=it, e=es, f=fr, p=pt, h=hi, j=ja, z=zh)",
        "default_voice_by_language": DEFAULT_VOICES,
        "mixed_language_markup": "[lang:en]…[/lang] dentro la battuta",
    }


def _h3_options(ew) -> dict:
    import os
    from core import minimax_h3_spec as S
    from core.ltx_video_job import MINIMAX_H3_PY
    profs = [p for p, (fl, _r, te) in S.PROFILES.items() if not ew.missing([fl, te, *S.COMMON_IDS])]
    return {"installed": os.path.exists(MINIMAX_H3_PY) and bool(profs), "profiles_installed": profs,
            "profiles": list(S.PROFILES), "duration_s": [S.MIN_S, S.MAX_S], "fps": S.FPS,
            "dialogue_languages": list(S.DIALOGUE_LANGS), "turbo_steps": ["", *S.TURBO],
            "voice_reference": "audio.voice_reference_path → Ref2VA (timbro della voce)",
            "license": "MiniMax H3 Community: territori esclusi UE, UK, USA, Corea del Sud"}


def options() -> dict:
    from core import longcat_service as lc
    from core import tts_engine as T
    from core import ensure_weights as ew
    from core.qwen21_spec import PROFILES, ASPECTS_2K
    try:
        voices = [{"id": v.slug(), "name": v.name, "engine": v.engine, "language": v.language}
                  for v in T.list_user_voices()]
    except Exception:                                                # noqa: BLE001
        voices = []
    return {
        "engines": {
            "longcat": {"installed": lc.is_installed(), "precisions_installed": lc.installed_precisions(),
                        "resolutions": list(lc.RESOLUTIONS), "aspects": list(lc.ASPECTS),
                        "duration": "decisa dall'audio (nessun limite)"},
            "minimax_h3": _h3_options(ew),
        },
        "image_engines": {
            "qwen_image21": {"profiles_installed": [p for p, (d, _t) in PROFILES.items() if not ew.missing([d])],
                             "aspects_2k": list(ASPECTS_2K), "max_refs": 10,
                             "license": "Qwen Research: uso non commerciale"},
            "krea2_edit": {"max_refs": 2},
        },
        "tts_engines": [{"id": e[0], "name": e[1], "clone": e[2], "emotions": e[3], "installed": e[4]}
                        for e in T.available_engines()],
        "voices": voices,
        "languages": _languages(),
        "emotions": ["neutral", "happy", "sad", "angry", "whisper", "calm"],
    }


def register(app: FastAPI) -> None:
    @app.get("/avatar/options", tags=["avatar"])
    def avatar_options():
        gate.require("cap.avatar_longcat")
        return options()

    @app.post("/avatar/generate", tags=["avatar"])
    def avatar_generate(req: AvatarReq):
        gate.require("cap.avatar_h3" if req.engine == "minimax_h3" else "cap.avatar_longcat")
        if req.image.mode in ("text", "character", "character_multi") and req.image.engine == "qwen_image21":
            gate.require("cap.image_qwen21")
        payload = req.model_dump()

        def fn(job):
            from core.avatar import pipeline
            res = pipeline.run(payload, log=job.log)
            job.warnings = list(res.get("warnings") or [])
            job.info = {"languages": res.get("languages") or {}}
            outs = [res["video"]]
            if req.return_intermediates:
                outs += [p for p in [res.get("image"), res.get("audio")] if p and os.path.isfile(p)]
            return outs
        return _submit("avatar.generate", fn)
