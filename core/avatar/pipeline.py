"""Orchestrazione dell'avatar parlante per l'API /avatar/generate: immagine → audio → video → post.

Motore «longcat»: l'audio è dato (file o TTS/dialogo creato qui) e decide la durata.
Motore «minimax_h3»: voce generata dal modello insieme al video (4–15 s) — disponibile quando il motore H3
è installato (fase H3); fino ad allora la richiesta fallisce con un messaggio chiaro.

Ogni passo registra i file intermedi (immagine, voce, mix) così l'API può restituirli con download_url.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from core.avatar import audio_chain as A
from core.avatar import image_step as IS

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "avatar"


def _p(prefix: str, ext: str) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    return str(OUT / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.{ext}")


def build_audio(a: dict, *, log=None) -> A.AudioResult:
    """Sezione `audio` della richiesta → voce (1–2 tracce) + musica/effetti + normalizzazione."""
    mode = a.get("mode", "file")
    if mode == "file":
        path = a.get("audio_path") or ""
        if not os.path.isfile(path):
            raise ValueError(f"audio inesistente: {path}")
        if a.get("trim_start") or a.get("trim_end"):
            path = A.trim_file(path, float(a.get("trim_start") or 0), float(a.get("trim_end") or 0))
        res = A.AudioResult(path=path, tracks={"person1": path}, audio_type="add", duration_s=A.duration(path))
        if a.get("audio2_path"):
            if not os.path.isfile(a["audio2_path"]):
                raise ValueError(f"audio persona 2 inesistente: {a['audio2_path']}")
            res.tracks["person2"] = a["audio2_path"]
            res.audio_type = a.get("turn_mode", "add")
    elif mode in ("tts", "dialogue"):
        text = (a.get("text") or "").strip()
        if not text:
            raise ValueError("audio.text vuoto")
        speakers = [A.Speaker.from_dict(s) for s in (a.get("speakers") or [{}])]
        for s in speakers:
            if a.get("language") and s.language == "it-IT":
                s.language = a["language"]
        if mode == "tts":
            speakers = speakers[:1]
        if log:
            log(f"[avatar] voce: {len(speakers)} speaker, {len(text)} caratteri")
        res = A.build_speech(text, speakers, gap_ms=int(a.get("gap_ms", 350)),
                             pad_start_ms=int(a.get("pad_start_ms", 0)), pad_end_ms=int(a.get("pad_end_ms", 0)))
    else:
        raise ValueError(f"audio.mode sconosciuto: {mode} (file | tts | dialogue)")
    if a.get("target_duration_s"):
        fitted = A.fit_duration(res.path, float(a["target_duration_s"]), a.get("fit", "pad"))
        if fitted != res.path:
            res.path = fitted
            if len(res.tracks) == 1:
                res.tracks = {"person1": fitted}
    voice_only = res.path
    music = a.get("music") or {}
    if music.get("prompt") or music.get("path"):
        bed = music.get("path") or A.generate_music(music["prompt"], A.duration(voice_only),
                                                    music.get("engine", "acestep"), on_log=log)
        res.path = A.mix_bed(res.path, bed, bed_db=float(music.get("volume_db", -18)),
                             ducking=bool(music.get("ducking", True)))
    sfx = a.get("sfx") or {}
    if sfx.get("mode") == "prompt" and sfx.get("prompt"):
        fx = A.generate_sfx(sfx["prompt"], A.duration(voice_only), on_log=log)
        res.path = A.mix_bed(res.path, fx, bed_db=float(sfx.get("volume_db", -14)), ducking=False)
    if a.get("normalize_lufs"):
        res.path = A.loudnorm(res.path, float(a["normalize_lufs"]))
    # le tracce per il lip-sync restano la voce pulita; il mix con musica/effetti va nel video finale
    res.duration_s = A.duration(res.path)
    return res


def _mux(video: str, audio: str, out: str) -> str:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg non trovato per il mix audio finale")
    r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"mix audio finale fallito: {r.stderr[-600:]}")
    return out


def run(req: dict, *, log=None) -> dict:
    """Esegue la richiesta completa. Ritorna {'video', 'image', 'audio', 'intermediates': [...]}."""
    engine = req.get("engine", "longcat")
    v = dict(req.get("video") or {})
    inter: list[str] = []
    if engine == "minimax_h3":
        from core import minimax_avatar                       # arriva con la fase H3
        return minimax_avatar.run(req, log=log)
    if engine != "longcat":
        raise ValueError(f"motore avatar sconosciuto: {engine} (longcat | minimax_h3)")
    from core import longcat_service as lc
    img_req = IS.ImageRequest.from_dict(req.get("image") or {"mode": "none"})
    image = IS.make_image(img_req, _p("start", "png"), log=log)
    if image:
        inter.append(image)
    audio = build_audio(req.get("audio") or {}, log=log)
    inter += [p for p in {audio.path, *audio.tracks.values()} if p]
    two = "person2" in audio.tracks
    params = {
        "mode": "ai2v" if image else "at2v", "image_path": image,
        "prompt": v.get("prompt", ""), "resolution": v.get("resolution", "480p"),
        "aspect": v.get("aspect", "16:9"), "resize_mode": v.get("resize_mode", "crop"),
        "audio_path": audio.tracks.get("person1", audio.path),
        "audio2_path": audio.tracks.get("person2", "") if two else "",
        "audio_type": audio.audio_type if two else "add",
        "bbox": (v.get("multi_person") or {}).get("bboxes"),
        "max_duration": float(v.get("max_duration_s") or 0), "precision": v.get("precision_profile", "int8"),
        "seed": int(v.get("seed", 42) if v.get("seed", -1) >= 0 else int(time.time()) % 2**31),
        "ref_img_index": int(v.get("ref_img_index", 10)), "mask_frame_range": int(v.get("mask_frame_range", 3)),
        "vocal_separation": bool((req.get("audio") or {}).get("vocal_separation", True)),
        "offload_kv_cache": True if v.get("offload_kv_cache") else "auto",   # False = decide il piano VRAM
        "output": _p("avatar_longcat", "mp4"),
    }
    if v.get("negative"):
        params["negative"] = v["negative"]
    if v.get("enhance_prompt") and params["prompt"]:
        from core import prompt_enhance
        params["prompt"] = prompt_enhance.enhance(params["prompt"], kind="video") or params["prompt"]
    errs = lc.preflight(params)
    if errs:
        raise RuntimeError("; ".join(errs))
    lines: list[str] = []

    def _l(s):
        lines.append(s)
        if log:
            log(s)
    job = lc.make_job(params, on_log=_l)
    rc = job.run()
    video = params["output"]
    if rc != 0 or not os.path.isfile(video):
        err = next((x for x in reversed(lines) if x.startswith("ERRORE")), "") or "\n".join(lines[-10:])
        raise RuntimeError(err or f"LongCat terminato con codice {rc}")
    # il worker mette nel video la voce del lip-sync: se c'è un mix (musica/effetti) lo sostituisco
    lip_audio = audio.tracks.get("person1")
    if audio.path and audio.path != lip_audio and not two:
        video = _mux(video, audio.path, _p("avatar_mix", "mp4"))
    sfx = (req.get("audio") or {}).get("sfx") or {}
    if sfx.get("mode") == "auto":                            # effetti dalla scena del video finale (MMAudio)
        fx = A.generate_sfx(sfx.get("prompt", ""), A.duration(audio.path), video_path=video, on_log=log)
        bed = A.mix_bed(audio.path, fx, bed_db=float(sfx.get("volume_db", -14)), ducking=True)
        video = _mux(video, bed, _p("avatar_sfx", "mp4"))
    post = v.get("postprocess") or {}
    if post.get("face_restore"):
        from core import native_driver
        model = {"codeformer": "codeformer-v0.1.0.pth", "gfpgan": "GFPGANv1.4.pth"}.get(
            post["face_restore"], post["face_restore"])
        r = native_driver.run_and_wait("video_face", {"image_path": video, "model": model, "fidelity": 0.6},
                                       timeout=3600)
        out = (r.get("output") or [None])[0]
        if r.get("status") == "success" and out and os.path.isfile(out):
            video = out
        elif log:
            log(f"[avatar] restauro volto fallito: {str(r.get('error', ''))[:200]}")
    if int(post.get("interpolate_fps") or 0) >= 50:
        from core import rife
        if rife.available():
            video = rife.interpolate_video(video, _p("avatar_rife", "mp4"), factor=2)
        elif log:
            log("[avatar] RIFE non disponibile: tengo 25 fps")
    return {"video": video, "image": image, "audio": audio.path, "intermediates": inter}
