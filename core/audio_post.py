"""Post-produzione audio via ffmpeg — normalizzazione loudness, trim silenzio, fade, conversione formato.

Nessun peso/modello: usa solo ffmpeg (già presente). Riusa core.music_service.Job per il subprocess+log.
Le operazioni sono filtri audio ffmpeg concatenati:
 - normalize  → loudnorm (EBU R128, target -14 LUFS / -1.5 dBTP): livello uniforme "da streaming".
 - trim_silence → silenceremove: taglia i silenzi a inizio/fine (e i lunghi interni).
 - fade_in/fade_out → afade (il fade-out usa areverse per non dover conoscere la durata).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from core.music_service import Job, MUSIC_OUTPUT_DIR, _safe


def _ffmpeg() -> str:
    p = Path(r"E:\AI\Tools\ffmpeg-shared\bin\ffmpeg.exe")
    return str(p) if p.exists() else (shutil.which("ffmpeg") or "ffmpeg")


def ffmpeg_available() -> bool:
    return bool(shutil.which(_ffmpeg()) or Path(_ffmpeg()).exists())


def _filters(opts: dict) -> list[str]:
    f: list[str] = []
    if opts.get("trim_silence"):
        f.append("silenceremove=start_periods=1:start_threshold=-50dB:"
                 "stop_periods=-1:stop_threshold=-50dB:stop_duration=0.5")
    if opts.get("normalize"):
        f.append("loudnorm=I=-14:TP=-1.5:LRA=11")
    fin = float(opts.get("fade_in", 0.0) or 0.0)
    if fin > 0:
        f.append(f"afade=t=in:st=0:d={fin}")
    fout = float(opts.get("fade_out", 0.0) or 0.0)
    if fout > 0:                                              # fade-out senza conoscere la durata: reverse→fade-in→reverse
        f.append(f"areverse,afade=t=in:st=0:d={fout},areverse")
    return f


def out_path(out_name: str, fmt: str) -> Path:
    return MUSIC_OUTPUT_DIR / f"{_safe(out_name)}_processed.{fmt or 'wav'}"


def make_job(src: str, out_name: str, opts: dict, **cbs) -> Job:
    MUSIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fmt = opts.get("format") or Path(src).suffix.lstrip(".") or "wav"
    out = out_path(out_name, fmt)
    cmd = [_ffmpeg(), "-y", "-i", str(src)]
    filt = _filters(opts)
    if filt:
        cmd += ["-af", ",".join(filt)]
    cmd.append(str(out))
    return Job(cmd, MUSIC_OUTPUT_DIR, **cbs)


def preflight(src: str, opts: dict) -> list[str]:
    errs = []
    if not ffmpeg_available():
        errs.append("ffmpeg non trovato: installalo o mettilo nel PATH.")
    if not src or not Path(src).is_file():
        errs.append("Seleziona un file audio sorgente valido.")
    return errs
