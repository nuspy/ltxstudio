"""Videomusicale AUTOMATICO (1 workflow): canzone (HeartMuLa) → storyboard LLM (Gemma traduce +
crea scene coerenti col testo) → mappa temporale proporzionale al contenuto → chunk video NATIVI
(LTX-2.3 GGUF, i2v concatenati, un prompt-scena per sezione) → concat + mux dell'audio.

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

from core import lyrics_llm

REPO = Path(__file__).resolve().parents[1]
LTX2_PY = r"E:\AI\Tools\ltx2_venv\Scripts\python.exe"
LTX2_WORKER = str(REPO / "workers" / "ltx2_infer.py")
GGUF = r"E:\AI\Models\diffusion_models\LTXVideo\LTX-2\quantstack\ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf"
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
        lyrics_llm.unload()                                # libera Gemma dalla VRAM prima del video GGUF
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
            params = {
                "image_path": cond, "gguf_transformer": GGUF,
                "prompt": scene["prompt"] + style_sfx, "negative": NEG,
                "width": width, "height": height, "num_frames": nf, "fps": fps,
                "steps": steps, "guidance": 3.0, "seed": 8, "offload": "cpu",
                "ref_strength": 1.0, "output": out_i,
            }
            jf = os.path.join(scratch, f"c_{i:03d}.json")
            json.dump(params, open(jf, "w"))
            log(f"Chunk {i + 1}/{nch}{' ✂' if cut else ''}…")
            r = subprocess.run([LTX2_PY, "-u", LTX2_WORKER, jf], env=env, capture_output=True, text=True)
            if not os.path.exists(out_i):
                raise RuntimeError(f"chunk {i} fallito: " + (r.stdout[-500:] + r.stderr[-500:]))
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
    log(f"Fatto: {out_path}")
    return out_path
