"""Export timeline (spec §11.4): master ffmpeg + export NLE via OpenTimelineIO (FCPXML/AAF/EDL/OTIO).
OTIO è opzionale: se assente, si esporta un manifest JSON + EDL minimale (fallback graceful)."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _watermark_master(path: str) -> str:
    """#106 watermark obbligatorio sul MASTER consegnato: unico punto post-montaggio del Film
    Studio (gli shot sorgente sono intermedi e possono provenire da cache/riusi non marchiati →
    la marchiatura autoritativa è qui, UNA volta a fine catena, mai dentro gen_service).
    RuntimeError (forced non applicabile) → errore del job, non nascosto."""
    try:
        from core import watermark
        return watermark.apply_to_file(path)
    except RuntimeError:
        raise
    except Exception:                                       # noqa: BLE001
        return path


def render_master(clips: list[str], out_path: str, *, audio: str = "",
                  audio_tracks: list[dict] | None = None, fps: float = 25.0) -> str:
    """Concatena le clip in un master unico (re-encode) + audio.
    audio_tracks=[{path, start_s, volume}] = mix multitraccia (dialoghi/musica/ambient/SFX) con
    ducking semplice via volume. `audio` singolo = retro-compat."""
    if not clips:
        raise RuntimeError("nessuna clip da montare")
    work = Path(out_path).with_suffix(".list.txt")
    work.write_text("".join(f"file '{c}'\n" for c in clips if os.path.exists(c)), "utf-8")
    silent = str(Path(out_path).with_suffix(".silent.mp4"))
    subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(work),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), silent, "-y"], check=False)
    tracks = list(audio_tracks or ([] if not audio else [{"path": audio, "start_s": 0, "volume": 1.0}]))
    tracks = [t for t in tracks if t.get("path") and os.path.exists(t["path"])]
    if not tracks:
        os.replace(silent, out_path); return _watermark_master(out_path)
    # mix multitraccia: ogni traccia ritardata (adelay) e attenuata (volume) → amix
    cmd = ["ffmpeg", "-v", "error", "-i", silent]
    for t in tracks:
        cmd += ["-i", t["path"]]
    filt = []
    for i, t in enumerate(tracks, start=1):
        delay = int(float(t.get("start_s", 0)) * 1000)
        filt.append(f"[{i}:a]adelay={delay}|{delay},volume={float(t.get('volume',1.0))}[a{i}]")
    mix_in = "".join(f"[a{i}]" for i in range(1, len(tracks) + 1))
    filt.append(f"{mix_in}amix=inputs={len(tracks)}:duration=longest:dropout_transition=0[aout]")
    cmd += ["-filter_complex", ";".join(filt), "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-shortest", out_path, "-y"]
    subprocess.run(cmd, check=False)
    if not os.path.exists(out_path):
        os.replace(silent, out_path)
    return _watermark_master(out_path)


def export_nle(clips: list[dict], out_dir: str, *, fmt: str = "fcpxml", fps: float = 25.0,
               name: str = "film") -> str:
    """Esporta la timeline per un NLE. clips=[{file, in_s, out_s, name}]. Usa OTIO se disponibile;
    altrimenti scrive manifest JSON + EDL minimale. Ritorna il path del file progetto."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        import opentimelineio as otio
        tl = otio.schema.Timeline(name=name)
        track = otio.schema.Track(name="V1")
        tl.tracks.append(track)
        for c in clips:
            dur = float(c.get("out_s", 0)) - float(c.get("in_s", 0))
            rng = otio.opentime.TimeRange(
                otio.opentime.RationalTime(0, fps),
                otio.opentime.RationalTime(max(1, round(dur * fps)), fps))
            mref = otio.schema.ExternalReference(target_url=Path(c["file"]).as_uri())
            track.append(otio.schema.Clip(name=c.get("name", "clip"), media_reference=mref, source_range=rng))
        ext = {"fcpxml": ".fcpxml", "aaf": ".aaf", "edl": ".edl", "otio": ".otio"}.get(fmt, ".otio")
        out = os.path.join(out_dir, name + ext)
        otio.adapters.write_to_file(tl, out)
        return out
    except Exception:                                       # noqa: BLE001
        # Fallback: manifest JSON + EDL minimale
        manifest = os.path.join(out_dir, name + ".manifest.json")
        json.dump({"name": name, "fps": fps, "clips": clips}, open(manifest, "w", encoding="utf-8"), indent=2)
        edl = os.path.join(out_dir, name + ".edl")
        with open(edl, "w", encoding="utf-8") as fh:
            fh.write(f"TITLE: {name}\n")
            t = 0.0
            for i, c in enumerate(clips, 1):
                dur = float(c.get("out_s", 0)) - float(c.get("in_s", 0))
                fh.write(f"{i:03d}  AX  V  C  {t:.2f} {t+dur:.2f}  {Path(c['file']).name}\n")
                t += dur
        return manifest
