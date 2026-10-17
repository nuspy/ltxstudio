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
                                       "outpaint | restyle | cameraman | lipdub (Doppiaggio da testo) | "
                                       "lipdub_audio (Doppiaggio da audio, SPERIMENTALE) | hdr (SDR -> HDR) | "
                                       "motion_track (movimento da tracce disegnate)")
    video_path: str = Field("", description="video sorgente (cameraman: video di RIFERIMENTO del movimento; "
                                            "motion_track: non serve)")
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
    anchor: str = Field("center", description="dove sta il video nel canvas: center (si estende su tutti i lati) | "
                        "left (area nuova a destra) | right (a sinistra) | top (in basso) | bottom (in alto)")
    margins: list[int] | None = Field(None, description="custom: [sopra, destra, sotto, sinistra] px sorgente")
    dark_source: bool = False
    paste_original: bool = True
    feather_px: int = Field(24, ge=0, le=256, description="outpaint: sfumatura (px) del bordo fra originale incollato "
                            "e parte generata, uguale per ogni fotogramma; 0 = taglio netto. Conta se paste_original")
    min_bar: float = Field(0.0, ge=0.0, le=0.45, description="outpaint (opt-in, solo canvas non custom): se una barra "
                           "nuova e' piu' stretta, rimpicciolisce il video nel canvas finche' ogni barra e' >= questa "
                           "frazione (consigliato 0.15); 0 = spento")
    fill_coverage: bool = Field(True, description="outpaint «garantisci riempimento» (default SI): se le barre nuove sono "
                                "simmetriche sullo stesso asse (due barre opposte) e l'area nuova e' < 60%, il video "
                                "originale viene rimpicciolito (cornice su 4 lati) perche' il modello altrimenti le "
                                "lascia nere; false = layout letterale (originale piu' grande, barre forse non riempite)")
    # lipdub = «Doppiaggio da testo» (JUST-DUB-IT): il video CON audio e' il riferimento; la NUOVA battuta (testo) genera
    # voce e labbra nuove, per la stessa durata del video. `prompt` (se presente) vince su line/language/scene.
    line: str = Field("", description="lipdub: la nuova battuta, nell'alfabeto della lingua parlata (il modello non traduce)")
    language: str = Field("", description="lipdub: lingua parlata (codice: en es fr de ru zh it pt ja ko). Altre: avviso nei warnings")
    scene: str = Field("", description="lipdub: scena / chi parla, in inglese (opzionale)")
    translate_to: str = Field("", description="lipdub: se `line` e' vuota, estrae il parlato del video (STT locale), lo traduce "
                              "in questa lingua (codice di `language`) con l'LLM locale e lo usa come battuta; richiede STT e LLM attivi")
    source_text: str = Field("", description="lipdub + translate_to: testo originale gia' noto (salta l'STT)")
    # lipdub_audio = «Doppiaggio da audio» SPERIMENTALE: l'audio dell'utente e' tenuto fisso nei latenti (l'audio comanda la durata)
    audio_path: str | None = Field(None, description="lipdub_audio: file audio da usare (canzone, voce, TTS)")
    audio_reference: str = Field("external", description="lipdub_audio: external (l'audio scelto fa da riferimento voce) | original (la voce del video)")
    # hdr (SDR -> HDR, IC-LoRA HDR LTX-2.3): nessun prompt (il testo e' il file scene-emb). Uscita = anteprima SDR
    # tonemappata (output[0]) + HEVC 10 bit PQ/BT.2020 e/o cartella EXR lineare (output[1:]), vedi core/hdr_export.py
    hdr_format: str = Field("pq", description="hdr: pq (HEVC 10 bit PQ/BT.2020) | exr (sequenza EXR lineare) | both. "
                            "I file scaricabili sono l'anteprima SDR e l'HEVC; la cartella EXR sta accanto all'anteprima "
                            "(<anteprima senza .mp4>_EXR, un .exr per frame) e non e' un file scaricabile")
    hdr_exposure_ev: float = Field(0.0, ge=-6.0, le=6.0, description="hdr: esposizione dell'anteprima SDR in stop (non tocca HDR/EXR)")
    hdr_exr_half: bool = Field(True, description="hdr: EXR in float16 (False = float32, file doppi)")
    hdr_hq: bool = Field(False, description="hdr: alta qualita' (2x frame interni, circa 2x tempo)")
    # cameraman (image_path/image_strength/width/height/seconds valgono anche per motion_track, con default propri)
    image_path: str | None = Field(None, description="cameraman: immagine iniziale (opz.) · motion_track: immagine di partenza "
                                   "= frame 0 (OBBLIGATORIA)")
    image_strength: float = Field(0.7, description="forza dell'immagine (motion_track, se omessa: 1.0)")
    width: int = Field(960, description="motion_track: se omessa si ricava dall'aspetto dell'immagine (lato corto 576)")
    height: int = 576
    seconds: float = Field(0.0, description="cameraman: durata · motion_track: durata (default 121 frame = 5 s a 24 fps; "
                           "tetto 257 frame, ~10,7 s a 24 fps)")
    # motion_track (IC-LoRA motion-track LTX-2.3): le tracce sono punti sull'immagine; l'ordine dei punti e' il TEMPO
    tracks: str | list | dict | None = Field(None, description="motion_track: tracce. JSON/oggetto: punti di controllo "
        "[[{x,y},...],...] (uno o piu' punti per traccia; 1 punto = ferma), oppure {points_store, coordinates, size}, oppure "
        "tracce dense (un punto per frame) oppure il workflow ComfyUI Lightricks con LTXVSparseTrackEditor. Coordinate: "
        "`size` del JSON o `tracks_size`; se mancano = pixel dell'immagine ORIGINALE")
    tracks_size: list[int] | None = Field(None, description="motion_track: [larghezza, altezza] dello spazio delle coordinate")
    tracks_format: str = Field("auto", description="motion_track: auto | control (punti di controllo) | dense (un punto per frame)")
    tracks_video_path: str | None = Field(None, description="motion_track: video di controllo gia' pronto (alternativa a `tracks`; "
                                          "stesso disegno di LTXVDrawTracks)")
    num_frames: int | None = Field(None, description="motion_track: frame (8k+1; alternativa a `seconds`)")
    fps: int = Field(24, description="motion_track: fps (8-60, ufficiale 24)")
    native_audio: bool = Field(True, description="motion_track: tiene l'audio generato insieme al video (voce/suoni dal "
                               "prompt, come il workflow ufficiale); False = video muto")
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
    engine: str = Field("seedvr2", description="seedvr2 | flashvsr | h3_upsampler (MiniMax H3, nativo)")
    seedvr2_model: str = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    resolution: int = 1080
    batch_size: int = 5
    block_swap: int = 0
    scale: float = 2.0
    # solo engine=h3_upsampler: il checkpoint H3 Ref2VA lo sceglie il chiamante (nessun ripiego automatico)
    h3_profile: str | None = Field(None, description="nvfp4 | int4 | int8 | fp8 (obbligatorio per h3_upsampler)")
    h3_te: str | None = Field(None, description="mmh3-te-nvfp4 | mmh3-te-fp8 | mmh3-te-int8 (vuoto = quello del profilo)")
    megapixels: float = Field(2.0, description="uscita come area, rapporto invariato (1 MP = 1024x1024)")
    denoise: float = Field(0.1, description="forza del ri-disegno, 0 = solo upscaler latente")
    guide: bool = Field(True, description="guida con la LoRA h3upscale x2")
    lora_strength: float = 1.0
    tile_tokens: int = Field(0, description="budget di token per forward di tile; 0 = automatico dalla VRAM libera (tetto 70000)")


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
        """Restauro (refocus/uncompress/deinterlace/motiondeblur/detailer), outpaint, restyle, cameraman, lipdub, hdr,
        motion_track (immagine + tracce -> video; output = video + video delle tracce di controllo).
        Risposta: job_id + `warnings` (es. checkpoint non distilled: MAI sostituito)."""
        from fastapi import HTTPException
        from core import ltx_release as lrel
        from core import video_edit_ic as vic
        from core import video_edit_modes as vem
        from core.video_ic_prep import OUTPAINT_FORMATS, ANCHORS
        spec = vem.MODE_SPECS.get(req.kind)                       # `kind` = chiave della modalita' (lipdub_audio inclusa)
        if spec is None or not spec.enabled or not spec.cap or req.kind == "union":
            valid = sorted(s.key for s in vem.MODE_SPECS.values() if s.enabled and s.cap)
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
        if req.kind == "hdr":
            from core import hdr_export as _hx
            if req.hdr_format not in _hx.FORMATS:
                raise HTTPException(422, {"error": "invalid_hdr_format", "valid": list(_hx.FORMATS)})
        if req.kind == "motion_track":
            from core import motion_tracks as _mt
            if not (req.image_path and os.path.exists(req.image_path)):
                raise HTTPException(422, {"error": "image_required", "path": req.image_path,
                                          "detail": "motion_track: image_path = immagine di partenza (frame 0)"})
            if req.tracks_format not in ("auto", "control", "dense"):
                raise HTTPException(422, {"error": "invalid_tracks_format", "valid": ["auto", "control", "dense"]})
            if req.tracks_video_path:
                if not os.path.exists(req.tracks_video_path):
                    raise HTTPException(422, {"error": "tracks_video_not_found", "path": req.tracks_video_path})
            elif req.tracks in (None, "", [], {}):
                raise HTTPException(422, {"error": "tracks_required",
                                          "detail": "motion_track: indica `tracks` (JSON) o `tracks_video_path`"})
            else:
                try:
                    _mt.parse_track_input(req.tracks, req.num_frames, req.tracks_format)
                except (ValueError, TypeError, KeyError) as e:
                    raise HTTPException(422, {"error": "invalid_tracks", "detail": str(e)})
            if not 8 <= req.fps <= 60:
                raise HTTPException(422, {"error": "invalid_fps", "detail": "fps tra 8 e 60"})
            _n = (vem.snap_frames(req.num_frames) if req.num_frames
                  else _mt.frames_for_seconds(req.seconds or (_mt.DEFAULT_FRAMES / _mt.DEFAULT_FPS), req.fps))
            if _n > spec.max_frames_per_segment:
                raise HTTPException(422, {"error": "too_long", "frames": _n, "max_frames": spec.max_frames_per_segment,
                                          "detail": "le tracce non si possono spezzare in segmenti: accorcia il clip"})
        elif not os.path.exists(req.video_path):
            raise HTTPException(422, {"error": "video_not_found", "path": req.video_path})
        warnings: list[str] = []
        if req.kind in ("lipdub", "lipdub_audio"):
            from core import video_ic_prep as _prep
            if req.kind == "lipdub":
                if not (req.prompt.strip() or req.line.strip() or req.translate_to.strip()):
                    raise HTTPException(422, {"error": "line_required",
                                              "detail": "lipdub: indica `line` (la nuova battuta), `translate_to` o un `prompt` completo"})
                if not _prep.has_audio(req.video_path):
                    raise HTTPException(422, {"error": "video_no_audio",
                                              "detail": "lipdub: il video deve avere una traccia audio (riferimento della voce)"})
                if req.translate_to.strip() and req.translate_to.strip().lower() not in vic.SPOKEN_CODES:
                    raise HTTPException(422, {"error": "invalid_translate_to", "valid": list(vic.SPOKEN_CODES)})
            else:
                if not (req.audio_path and os.path.exists(req.audio_path)):
                    raise HTTPException(422, {"error": "audio_not_found", "path": req.audio_path})
                if not _prep.has_audio(req.audio_path):
                    raise HTTPException(422, {"error": "audio_no_stream", "path": req.audio_path})
                if req.audio_reference not in ("external", "original"):
                    raise HTTPException(422, {"error": "invalid_audio_reference", "valid": ["external", "original"]})
                if req.audio_reference == "original" and not _prep.has_audio(req.video_path):
                    raise HTTPException(422, {"error": "video_no_audio",
                                              "detail": "audio_reference=original richiede una traccia audio nel video"})
                warnings.append("SPERIMENTALE: audio dell'utente tenuto fisso nei latenti, qualita' non garantita; per "
                                "canzoni/voci la via collaudata e' Talk/Canta del tab Video")
            if req.language.strip() and req.language.strip().lower() not in vic.SPOKEN_CODES:
                warnings.append(f"lingua «{req.language}» senza prove di supporto (note: {', '.join(vic.SPOKEN_CODES)}): "
                                "il modello potrebbe parlarla male")
        model = (req.model or "").strip()
        rel = (req.ltx_release or (lrel.detect_release(model) if model else None) or "")
        if rel and not lrel.ic_lora_supported(rel):
            raise HTTPException(422, {"error": "release_unsupported", "release": rel,
                                      "detail": "le IC-LoRA locali sono per LTX 2.3"})
        if not lrel.is_distilled(model):
            # col dev il worker aggiunge la distilled-LoRA ufficiale (strength 0.5, stage 1+2): si avvisa, non si blocca
            if vic.distilled_lora_ok(rel or "2.3"):
                warnings.append("checkpoint non distilled" + ("" if model else " (default del worker: dev)")
                                + ": verra' aggiunta la distilled-LoRA ufficiale (il checkpoint resta quello scelto)")
            else:
                warnings.append("checkpoint non distilled" + ("" if model else " (default del worker: dev)")
                                + f": la distilled-LoRA manca su disco ({vic.distilled_lora_path(rel or '2.3')}), "
                                  "qualita' molto ridotta; scaricala dal tab Pesi modelli (id ltx23-distilled-lora-384)")
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
               "out_paste": req.paste_original, "out_feather": req.feather_px, "out_min_bar": req.min_bar,
               "out_fill": req.fill_coverage,
               "cam_ref": req.video_path, "cam_img": req.image_path or "",
               "cam_img_strength": req.image_strength, "cam_seconds": req.seconds,
               "width": req.width, "height": req.height, "seed": req.seed,
               "lipdub_line": req.line, "lipdub_lang": req.language, "lipdub_scene": req.scene,
               "translate_to": req.translate_to, "source_text": req.source_text,
               "lipdub_audio": req.audio_path or "", "lipdub_audio_ref": req.audio_reference,
               "hdr_format": req.hdr_format, "hdr_exposure_ev": req.hdr_exposure_ev,
               "hdr_exr_half": req.hdr_exr_half, "hdr_hq": req.hdr_hq}
        if req.kind == "motion_track":
            _given = req.model_fields_set
            for _k in ("width", "height"):                       # senza valori espliciti: dall'aspetto dell'immagine
                if _k not in _given:
                    cfg.pop(_k, None)
            cfg.update(track_image=req.image_path, tracks=req.tracks, tracks_size=req.tracks_size,
                       tracks_format=req.tracks_format, tracks_video_path=req.tracks_video_path or "",
                       fps=req.fps, num_frames=req.num_frames, seconds=req.seconds or None,
                       image_strength=(req.image_strength if "image_strength" in _given else 1.0), keep_audio=False,
                       native_audio=req.native_audio)
        if model:
            cfg["gguf_transformer" if model.lower().endswith(".gguf") else "model"] = model
        if rel:
            cfg["ltx_release"] = rel

        def fn(job):
            import time as _t
            output = out_path(f"video_ic_{spec.key}")
            if cfg.get("translate_to"):                          # lipdub: estrai il parlato e traduci (STT+LLM locali)
                from core import video_dub_translate as vdt
                try:
                    vdt.fill_cfg(cfg, log=lambda m: job.log(m))
                except vdt.DubTranslateError as e:
                    raise RuntimeError(str(e)) from None
            ij = vic.build_job(cfg, os.path.dirname(output), int(_t.time()), output, log=lambda m: job.log(m))
            outs = run_ltx2(ij.params, job)
            allo = ij.outputs(vic.finalize(ij, outs[0], log=lambda m: job.log(m)))
            for d in (o for o in allo if os.path.isdir(o)):      # hdr: la cartella EXR non e' scaricabile come file
                job.log(f"EXR: {d}")
            return [o for o in allo if os.path.isfile(o)]
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
        if req.engine == "h3_upsampler":
            gate.require("cap.video_h3_upsampler")
        def fn(job):
            out = out_path(f"{req.engine}")
            if req.engine == "h3_upsampler":
                from core import h3_upsample_service as hs
                hs.upscale_video(req.video_path, out, h3_profile=req.h3_profile or "", h3_te=req.h3_te or "",
                                 megapixels=req.megapixels, denoise=req.denoise, guide=req.guide,
                                 lora_strength=req.lora_strength, tile_tokens=req.tile_tokens,
                                 on_log=lambda l: job.log(l))
            elif req.engine == "flashvsr":
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
