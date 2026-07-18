"""Videomusicale AUTOMATICO (1 workflow): canzone (HeartMuLa) → storyboard LLM (Gemma traduce +
crea scene coerenti col testo) → mappa temporale proporzionale al contenuto → chunk video NATIVI
(LTX-2.3 fp8 in auto, i2v concatenati, un prompt-scena per sezione) → concat + mux dell'audio.

Coerenza testo↔immagini: lo storyboard dell'LLM segue l'ordine del testo; i pesi delle scene
distribuiscono il tempo (HeartMuLa canta le sezioni in ordine → mappa proporzionale = buon
allineamento). Whisper/forced-align = upgrade futuro per precisione al secondo.

Zero ComfyUI. Pensato per essere chiamato da un worker GUI (callback progress) o da CLI.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

from core import ltx_video_job, lyrics_llm

REPO = Path(__file__).resolve().parents[1]
LTX2_PY = r"E:\AI\Tools\ltx2_venv\Scripts\python.exe"
# NB: nessuna costante GGUF di default. Il GGUF si usa SOLO se scelto dall'utente (passa da
# core/ltx_video_job): forzarlo costava ~500 s/step (dequant per-forward) e OOM col modello residente.
NEG = "blurry, low quality, deformed, gore, watermark, text, distorted faces, extra limbs"


HEARTLIB = Path(r"E:\AI\Tools\heartlib")
HEART_PY = str(HEARTLIB / "venv" / "Scripts" / "python.exe")
HEART_WRAP = str(REPO / "workers" / "heartmula_run.py")
HEART_CKPT = str(HEARTLIB / "ckpt")


def generate_song(*, lyrics: str, tags: str, out_mp3: str, duration_s: float = 180.0,
                  version: str = "3B", temperature: float = 1.0, cfg_scale: float = 1.5,
                  topk: int = 50, progress=None) -> str:
    """Genera la canzone con HeartMuLa (subprocess venv heartlib). Scrive i file liriche/tag accanto
    all'mp3. NB: questa versione di HeartMuLa NON ha --seed (non riproducibile per seed)."""
    def log(m):
        if progress:
            progress(m)
    base = os.path.splitext(out_mp3)[0]
    lyr, tag = base + ".lyrics.txt", base + ".tags.txt"
    os.makedirs(os.path.dirname(out_mp3), exist_ok=True)
    open(lyr, "w", encoding="utf-8").write(lyrics)
    open(tag, "w", encoding="utf-8").write(tags)
    log("Canzone (HeartMuLa)…")
    cmd = [HEART_PY, "-u", HEART_WRAP, "--model_path", HEART_CKPT, "--version", version,
           "--lyrics", lyr, "--tags", tag, "--save_path", out_mp3,
           "--max_audio_length_ms", str(int(max(5, duration_s) * 1000)),
           "--temperature", str(temperature), "--cfg_scale", str(cfg_scale), "--topk", str(topk),
           "--lazy_load", "True"]
    r = subprocess.run(cmd, cwd=str(HEARTLIB), capture_output=True, text=True)
    if not os.path.exists(out_mp3):
        raise RuntimeError("canzone fallita: " + (r.stdout[-500:] + r.stderr[-500:]))
    return out_mp3


def generate_song_and_video(*, lyrics: str, tags: str, image_path: str, out_video: str,
                            song_duration_s: float = 180.0, width: int = 704, height: int = 512,
                            scene_seconds: float = 6.0, video_style: str = "",
                            image_hint: str = "", progress=None) -> dict:
    """Workflow COMPLETO: canzone → storyboard LLM → video a chunk → mux. Ritorna {song, video}."""
    base = os.path.splitext(out_video)[0]
    song = generate_song(lyrics=lyrics, tags=tags, out_mp3=base + ".mp3",
                         duration_s=song_duration_s, progress=progress)
    video = make_music_video(lyrics=lyrics, song_path=song, image_path=image_path, out_path=out_video,
                             width=width, height=height, scene_seconds=scene_seconds,
                             video_style=video_style, image_hint=image_hint, progress=progress)
    return {"song": song, "video": video}


def _dur(path: str) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except Exception:                                       # noqa: BLE001
        return 0.0


def _scene_for_chunk(i: int, nch: int, scenes: list[dict]) -> dict:
    """Mappa il chunk i (0..nch-1) alla scena giusta in base ai PESI cumulativi (= tempo proporzionale)."""
    total = sum(max(1, s.get("weight", 1)) for s in scenes) or 1
    frac = (i + 0.5) / nch
    acc = 0.0
    for s in scenes:
        acc += max(1, s.get("weight", 1)) / total
        if frac <= acc:
            return s
    return scenes[-1]


# ───────────────────── CAST (personaggi ricorrenti col volto coerente) ─────────────────────
def _cast_by_name(cast: list[dict] | None) -> dict:
    """Indice case-insensitive del cast per nome. Ogni voce:
    {name, role, instrument, ref_image, instrument_image, frequency}."""
    return {str(c["name"]).strip().lower(): c
            for c in (cast or []) if (c.get("name") or "").strip()}


def _apply_cast_to_scene(scene: dict, by_name: dict) -> None:
    """Aggancia il cast alla scena dello storyboard: sceglie il personaggio-hero fra i `chars`, imposta
    `char_ref` (foto identità) + eventuale 2° riferimento `instrument_ref` (foto strumento) e AGGIUNGE il
    tag testuale dello strumento al prompt. Il volto coerente avviene poi in gen_scene_image (krea2_edit).
    Nessun cast/nessun match → scena invariata (b-roll/location)."""
    names = scene.get("chars") or []
    hero = next((by_name[n.lower()] for n in names if n.lower() in by_name), None)
    if not hero:
        return
    ref = (hero.get("ref_image") or "").strip()
    if ref and os.path.exists(ref):
        scene["char_ref"] = ref
    instr = (hero.get("instrument") or "").strip()
    if instr and instr.lower() not in scene["img_prompt"].lower():
        tag = f", holding/playing a {instr}"
        scene["img_prompt"] += tag
        scene["video_prompt"] += tag
    instr_img = (hero.get("instrument_image") or "").strip()
    if instr_img and os.path.exists(instr_img):
        scene["instrument_ref"] = instr_img            # 2° riferimento Krea 2 Edit per i clip-hero


def gen_portrait(prompt: str, out_png: str, *, arch: str = "flux", model: str | None = None,
                 width: int = 768, height: int = 768) -> str:
    """Ritratto di riferimento per un membro del cast (t2i, il motore immagine nativo). È la SORGENTE
    dell'identità: la coerenza vera nei clip avviene poi via gen_scene_image(coherence_engine='krea2_edit').
    Riusa native_driver come gen_scene_image → stesso percorso, nessuna duplicazione."""
    from core import native_driver
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    _p = {"arch": (arch or "flux"), "prompt": prompt, "negative": NEG,
          "width": int(width), "height": int(height), "steps": 20, "guidance": 3.5, "seed": 12,
          "output": out_png}
    if str(arch).lower() == "krea2":                     # Turbo distillato: 8 step, guidance 0.0 (no CFG)
        _p["steps"] = 8; _p["guidance"] = 0.0
    if model and (arch or "flux") == "flux":
        _p["model"] = model
    res = native_driver.run_and_wait("image", _p, timeout=600)
    p = res.get("output") or out_png
    p = p[0] if isinstance(p, list) else p
    if p and os.path.exists(p) and p != out_png:
        import shutil
        shutil.copyfile(p, out_png)
    if not os.path.exists(out_png):
        raise RuntimeError(res.get("error", "ritratto non generato"))
    return out_png


# ───────────────────── workflow INTERATTIVO (storyboard editabile) ─────────────────────
def plan_scenes(*, lyrics: str, song_path: str, scene_seconds: float = 4.0, video_style: str = "",
                image_hint: str = "", cast: list[dict] | None = None, progress=None) -> list[dict]:
    """Crea l'ELENCO delle riprese SENZA generare: per ogni scena {order, start_s, length_s,
    img_prompt, video_prompt, image_path, video_path, chars?, location?, char_ref?, instrument_ref?}.
    start_s distribuito sul tempo (pesi LLM); length_s = distanza dalla scena successiva (auto).
    `cast` = personaggi ricorrenti col volto coerente (vedi _apply_cast_to_scene). Editabile dall'utente."""
    def log(m):
        if progress:
            progress(m)
    dur = _dur(song_path)
    target = max(2, round(dur / max(1.0, scene_seconds)))
    log("Storyboard (LLM)…")
    try:
        scenes = lyrics_llm.storyboard(lyrics, image_hint=image_hint,
                                       target_scenes=min(target, MAX_SCENES), video_style=video_style,
                                       cast=cast)
    except Exception:                                       # noqa: BLE001
        scenes = []
    finally:
        try:
            lyrics_llm.unload()
        except Exception:                                   # noqa: BLE001
            pass
    if not scenes:
        scenes = [{"prompt": "cinematic music video, dynamic, atmospheric", "weight": 1}]
    sfx = (", " + video_style.strip()) if (video_style or "").strip() else ""
    by_name = _cast_by_name(cast)
    total_w = sum(max(1, s.get("weight", 1)) for s in scenes) or 1
    out, acc = [], 0.0
    for i, s in enumerate(scenes):
        start = round(acc / total_w * dur, 2)
        vp = s["prompt"].strip() + sfx
        loc = (", " + s["location"].strip()) if s.get("location") else ""
        scn = {"order": i, "start_s": start, "length_s": 0.0,
               "video_prompt": vp, "img_prompt": vp + loc + ", cinematic still frame",
               "image_path": "", "video_path": "",
               "chars": s.get("chars") or [], "location": s.get("location", "")}
        _apply_cast_to_scene(scn, by_name)                 # → char_ref / instrument_ref / tag strumento
        out.append(scn)
        acc += max(1, s.get("weight", 1))
    recompute_lengths(out, dur)
    return out


def recompute_lengths(scenes: list[dict], song_dur: float) -> None:
    """length_s di ogni scena = distanza dall'inizio della successiva (l'ultima fino a fine canzone).
    Le scene vengono ordinate per start_s e ri-numerate (order)."""
    scenes.sort(key=lambda s: s["start_s"])
    for i, s in enumerate(scenes):
        s["order"] = i
        nxt = scenes[i + 1]["start_s"] if i + 1 < len(scenes) else song_dur
        s["length_s"] = round(max(0.5, nxt - s["start_s"]), 2)


def gen_scene_image(scene: dict, out_png: str, *, width: int = 704, height: int = 512,
                    model: str | None = None, placement_mode: str | None = None,
                    arch: str = "flux", coherence_engine: str = "flux", char_ref: str = "",
                    grounding_px: int = 1024, krea2_variant: str = "turbo") -> str:
    """Genera l'immagine di apertura della scena (modulo immagini nativo).
    arch: architettura immagine ('flux' default o 'krea2' → worker dedicato via native_driver).
    coherence_engine + char_ref: se 'krea2_edit' e c'è un personaggio di riferimento (foto), la scena è
    generata con Krea 2 Edit in-context (istruzione = img_prompt) preservando l'identità del personaggio tra
    le scene. La scena può specificare un proprio riferimento (scene['char_ref']) → persone diverse in scene
    diverse; altrimenti si usa il char_ref globale. NON sostituisce FLUX: switch dell'utente."""
    from core import native_driver
    _ref = scene.get("char_ref") or char_ref             # per-scena → globale (attori diversi per scena)
    if str(coherence_engine).lower() == "krea2_edit" and _ref:
        # 2° riferimento opzionale = foto strumento del musicista-hero (clip-hero: 1 persona + strumento)
        _refs = [_ref] + ([scene["instrument_ref"]] if scene.get("instrument_ref") else [])
        _p = {"arch": "krea2_edit", "krea2_variant": krea2_variant, "prompt": scene["img_prompt"],
              "negative": NEG, "images": _refs, "image_path": _ref, "grounding_px": int(grounding_px),
              "seed": 7, "output": out_png, "steps": 8 if krea2_variant != "raw" else 20,
              "guidance": 0.0 if krea2_variant != "raw" else 2.0}
        res = native_driver.run_and_wait("image", _p, timeout=600)
        p = res.get("output") or out_png; p = p[0] if isinstance(p, list) else p
        if p and os.path.exists(p) and p != out_png:
            import shutil; shutil.copyfile(p, out_png)
        if not os.path.exists(out_png):
            raise RuntimeError(res.get("error", "immagine non generata"))
        scene["image_path"] = out_png
        return out_png
    _p = {
        "arch": (arch or "flux"), "prompt": scene["img_prompt"], "negative": NEG,
        "width": width, "height": height, "steps": 20, "guidance": 3.5, "seed": 7, "output": out_png,
    }
    if str(arch).lower() == "krea2":                     # Turbo distillato: 8 step, guidance 0.0 (no CFG)
        _p["steps"] = 8 if krea2_variant != "raw" else 52
        _p["guidance"] = 0.0 if krea2_variant != "raw" else 3.5
    if model and (arch or "flux") == "flux":             # eredita il modello del tab Foto (solo FLUX ha un file)
        _p["model"] = model
    if placement_mode:                                   # 'sticky' → transformer residente per il batch
        _p["placement_mode"] = placement_mode
    res = native_driver.run_and_wait("image", _p, timeout=600)
    p = res.get("output") or out_png
    p = p[0] if isinstance(p, list) else p
    if p and os.path.exists(p) and p != out_png:
        import shutil
        shutil.copyfile(p, out_png)
    if not os.path.exists(out_png):
        raise RuntimeError(res.get("error", "immagine non generata"))
    scene["image_path"] = out_png
    return out_png


def gen_scene_video(scene: dict, out_mp4: str, *, width: int = 704, height: int = 512, fps: int = 25,
                    model: str | None = None, model_gguf: str | None = None,
                    arch: str = "ltx", lingbot_variant: str = "dense", run_refiner: bool = False) -> str:
    """Genera lo spezzone video della scena (i2v dall'immagine di apertura, durata = length_s).
    model_gguf/model: transformer scelto nel tab Video; se None il worker sceglie da sé (fp8).
    arch: 'ltx' (default) o 'lingbot' (2ª architettura); per LingBot si passano variante e refiner."""
    nf = _valid_nf(min(float(scene["length_s"]), 6.7), fps)
    _extra = {"ref_strength": 1.0}
    if arch == "lingbot":
        _extra["lingbot_variant"] = lingbot_variant
        if run_refiner:
            _extra["run_refiner"] = True
    # Un solo posto per la policy del video (#158: niente GGUF forzato) → core/ltx_video_job.
    ltx_video_job.generate(
        output=out_mp4, image_path=scene.get("image_path", ""), prompt=scene["video_prompt"],
        negative=NEG, width=width, height=height, num_frames=nf, fps=fps,
        steps=8, guidance=3.0, seed=8, model=model, model_gguf=model_gguf, arch=arch,
        extra=_extra,
    )
    scene["video_path"] = out_mp4
    return out_mp4


def assemble_scenes(scenes: list[dict], song_path: str, out_path: str, *, fps: int = 25,
                    transition: str = "cut", xfade_s: float = 0.4) -> str:
    """Monta i video delle scene (nell'ordine) + traccia audio della canzone.
    transition: 'cut' (stacco netto, default) | 'crossfade'/'fade'/'dissolve'... (xfade ffmpeg).
    NB il nuovo Studio usa clip i2v SEPARATE → niente ghosting tra clip (lo stacco è netto by-design);
    la transizione xfade è opzionale per ammorbidire i tagli (#41/#46)."""
    ordered = sorted(scenes, key=lambda s: s["order"])
    clips = [s["video_path"] for s in ordered if s.get("video_path") and os.path.exists(s["video_path"])]
    if not clips:
        raise RuntimeError("nessun video di scena da montare")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    silent = out_path + ".silent.mp4"
    if transition == "cut" or len(clips) == 1:
        lst = out_path + ".list.txt"
        open(lst, "w", encoding="utf-8").write("".join(f"file '{c}'\n" for c in clips))
        subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), silent, "-y"], check=False)
    else:                                                   # xfade a catena tra clip consecutive
        xkind = {"crossfade": "fade", "dissolve": "dissolve", "fade": "fade"}.get(transition, transition)
        cmd = ["ffmpeg", "-v", "error"]
        for c in clips:
            cmd += ["-i", c]
        durs = [_dur(c) for c in clips]
        fc, prev, offset = [], "[0:v]", 0.0
        for i in range(1, len(clips)):
            offset += durs[i - 1] - xfade_s
            lbl = f"[x{i}]"
            fc.append(f"{prev}[{i}:v]xfade=transition={xkind}:duration={xfade_s}:offset={max(0,offset):.3f}{lbl}")
            prev = lbl
        subprocess.run(cmd + ["-filter_complex", ";".join(fc), "-map", prev,
                              "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), silent, "-y"], check=False)
        if not os.path.exists(silent):                     # fallback cut se xfade fallisce
            lst = out_path + ".list.txt"
            open(lst, "w", encoding="utf-8").write("".join(f"file '{c}'\n" for c in clips))
            subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), silent, "-y"], check=False)
    subprocess.run(["ffmpeg", "-v", "error", "-i", silent, "-i", song_path, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-shortest", out_path, "-y"], check=False)
    try:  # #107 firma invisibile di provenienza (default ON) — PRIMA del watermark visibile
        from core import stego
        if stego.enabled():
            stego.sign_file(out_path)
    except Exception:                                       # noqa: BLE001
        pass
    # #106 watermark obbligatorio sull'MP4 FINALE del montaggio (workflow interattivo). Punto scelto:
    # dopo il mux, UNA sola volta a fine catena — NON dentro gen_scene_video (le clip di scena sono
    # intermedi e possono venire da cache/riusi non marchiati, quindi la banda dei chunk NON è
    # garantita alla fonte). RuntimeError (forced non applicabile) → errore del job.
    try:
        from core import watermark
        out_path = watermark.apply_to_file(out_path)
    except RuntimeError:
        raise
    except Exception:                                       # noqa: BLE001
        pass
    return out_path


def whisper_available() -> bool:
    """STT disponibile: preferito Handy-API (REST :8720), fallback faster-whisper locale."""
    try:
        from core import handy_api
        if handy_api.available():
            return True
    except Exception:                                       # noqa: BLE001
        pass
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:                                       # noqa: BLE001
        return False


def align_scenes_with_whisper(song_path: str, scenes: list[dict], *, model_size: str = "base") -> list[dict]:
    """#2.2 Allineamento temporale: STT sull'mp3 → onset cantati → ridistribuisce gli start_s delle
    scene sui confini reali del canto. Backend: Handy-API (/transcribe) se attivo, altrimenti
    faster-whisper locale (timestamp per segmento). Se nessuno è disponibile → scene invariate."""
    if not scenes:
        return scenes
    dur = _dur(song_path)
    starts: list[float] = []
    # 1) Handy-API (preferito): trascrizione testo (per ora senza timestamp di segmento esposti via REST)
    try:
        from core import handy_api
        if handy_api.available():
            txt = handy_api.transcribe(song_path)
            if txt.strip():
                # senza timestamp espliciti, distribuiamo sul numero di righe/frasi trascritte
                lines = [ln for ln in txt.replace(". ", ".\n").splitlines() if ln.strip()]
                k = max(1, len(lines))
                starts = [round(dur * i / k, 2) for i in range(k)]
    except Exception:                                       # noqa: BLE001
        starts = []
    # 2) faster-whisper (fallback, timestamp di segmento reali)
    if not starts:
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(model_size, device="cuda" if _has_cuda() else "cpu",
                                 compute_type="float16" if _has_cuda() else "int8")
            segments, _info = model.transcribe(song_path, word_timestamps=False)
            starts = [float(s.start) for s in segments]
        except Exception:                                   # noqa: BLE001
            starts = []
    if not starts:
        return scenes
    n = len(scenes)
    ordered = sorted(scenes, key=lambda s: s["start_s"])
    for i, sc in enumerate(ordered):
        idx = min(len(starts) - 1, round(i * (len(starts) - 1) / max(1, n - 1)))
        sc["start_s"] = round(starts[idx], 2)
    recompute_lengths(ordered, dur)
    return ordered


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:                                       # noqa: BLE001
        return False


def export_nle(scenes: list[dict], out_dir: str, *, fmt: str = "fcpxml", name: str = "music_video") -> str:
    """#2.7 Export NLE del videomusicale (FCPXML/AAF/EDL/OTIO) riusando core/film_studio/otio_export."""
    from core.film_studio import otio_export
    ordered = sorted(scenes, key=lambda s: s["order"])
    clips = [{"file": s["video_path"], "in_s": 0.0, "out_s": float(s.get("length_s", 4)),
              "name": _mmss_safe(s.get("start_s", 0))}
             for s in ordered if s.get("video_path") and os.path.exists(s["video_path"])]
    if not clips:
        raise RuntimeError("nessun video da esportare")
    return otio_export.export_nle(clips, out_dir, fmt=fmt, name=name)


def _mmss_safe(s) -> str:
    s = int(s or 0); return f"{s // 60:02d}_{s % 60:02d}"


def _valid_nf(seconds: float, fps: int) -> int:
    """Numero frame valido per LTX (8·k+1), corrispondente a ~`seconds`, clamp ~2s..6.7s."""
    k = round((seconds * fps - 1) / 8)
    k = max(6, min(21, k))           # 8*6+1=49 (~2s) … 8*21+1=169 (~6.7s)
    return k * 8 + 1


MAX_SCENES = 40                       # tetto scene LLM (oltre, mappatura proporzionale)


def make_music_video(*, lyrics: str, song_path: str, image_path: str, out_path: str,
                     width: int = 704, height: int = 512, fps: int = 25, scene_seconds: float = 6.0,
                     video_style: str = "", chunk_frames: int | None = None,
                     steps: int = 8, image_hint: str = "", scratch: str | None = None,
                     progress=None) -> str:
    """Genera il videomusicale dato l'mp3 GIÀ generato + le liriche + l'immagine iniziale.
    `scene_seconds` = durata media di una scena = ritmo dei cambi (più piccolo = stacchi più rapidi).
    `video_style` = stile/resa applicato a OGNI segmento (es. 'cartone animato', 'bianco e nero triste').
    Ritorna il path del video finale (video+audio)."""
    def log(m):
        if progress:
            progress(m)

    scratch = scratch or str(REPO / "output" / "_mv_chunks")
    os.makedirs(scratch, exist_ok=True)
    dur = _dur(song_path)
    if dur <= 0:
        raise RuntimeError(f"durata canzone non valida: {song_path}")
    # chunk corto per qualità/VRAM (≤5s); le scene possono durare più chunk se il ritmo è lento
    nf = chunk_frames or _valid_nf(min(scene_seconds, 5.0), fps)
    seg = nf / fps
    nch = max(1, math.ceil(dur / seg))
    target = max(2, round(dur / max(1.0, scene_seconds)))  # numero scene = ritmo dei cambi
    log("Storyboard (LLM)…")
    try:
        scenes = lyrics_llm.storyboard(lyrics, image_hint=image_hint,
                                       target_scenes=min(target, MAX_SCENES), video_style=video_style)
    except Exception as e:                                  # noqa: BLE001
        log(f"[storyboard] LLM non disponibile ({e}) → scena unica")
        scenes = []
    try:
        lyrics_llm.unload()                                # libera Gemma dalla VRAM prima del video
    except Exception:                                      # noqa: BLE001
        pass
    if not scenes:
        base = "cinematic music video coherent with the song, dynamic, atmospheric"
        scenes = [{"prompt": base, "weight": 1}]
    style_sfx = (", " + video_style.strip()) if (video_style or "").strip() else ""
    log(f"Storyboard: {len(scenes)} scene · {nch} chunk da {seg:.1f}s ({dur:.0f}s)"
        + (f" · stile: {video_style.strip()}" if style_sfx else ""))

    prev = image_path
    prev_prompt = None
    parts: list[str] = []
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    for i in range(nch):
        scene = scenes[min(len(scenes) - 1, i * len(scenes) // nch)]   # scene distribuite uniformemente nel tempo
        # STACCO NETTO al cambio scena: riparti dall'immagine di sfondo (non dall'ultimo frame).
        cut = scene["prompt"] != prev_prompt
        cond = image_path if cut else prev
        out_i = os.path.join(scratch, f"chunk_{i:03d}.mp4")
        if not os.path.exists(out_i):
            log(f"Chunk {i + 1}/{nch}{' ✂' if cut else ''}…")
            ltx_video_job.generate(                          # policy #158 centralizzata
                output=out_i, image_path=cond, prompt=scene["prompt"] + style_sfx, negative=NEG,
                width=width, height=height, num_frames=nf, fps=fps,
                steps=steps, guidance=3.0, seed=8, extra={"ref_strength": 1.0},
            )
        prev_prompt = scene["prompt"]
        last = os.path.join(scratch, f"last_{i:03d}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-sseof", "-0.12", "-i", out_i, "-update", "1",
                        "-q:v", "2", last, "-y"])
        if os.path.exists(last):
            prev = last
        parts.append(out_i)

    log("Montaggio (concat + audio)…")
    lst = os.path.join(scratch, "list.txt")
    open(lst, "w", encoding="utf-8").write("".join(f"file '{p}'\n" for p in parts))
    silent = os.path.join(scratch, "silent.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent, "-y"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", silent, "-i", song_path, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-shortest", out_path, "-y"])
    try:  # #107 firma invisibile di provenienza (default ON) — PRIMA del watermark visibile
        from core import stego
        if stego.enabled():
            stego.sign_file(out_path)
    except Exception:                                       # noqa: BLE001
        pass
    # #106 watermark obbligatorio sull'MP4 FINALE consegnato (workflow automatico). Punto scelto:
    # dopo il mux finale, UNA sola volta a fine catena — NON sui chunk (intermedi, riusati da cache:
    # la marchiatura alla fonte via ltx2_infer non è garantita per chunk pre-esistenti).
    # RuntimeError (forced non applicabile) → errore del job, non nascosto.
    try:
        from core import watermark
        out_path = watermark.apply_to_file(out_path)
    except RuntimeError:
        raise
    except Exception:                                       # noqa: BLE001
        pass
    log(f"Fatto: {out_path}")
    return out_path
