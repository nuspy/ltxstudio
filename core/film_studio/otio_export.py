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
    try:  # #107 firma invisibile di provenienza (default ON) — PRIMA del watermark visibile
        from core import stego
        if stego.enabled():
            stego.sign_file(path)
    except Exception:                                       # noqa: BLE001
        pass
    try:
        from core import watermark
        path = watermark.apply_to_file(path)
    except RuntimeError:
        raise
    except Exception:                                       # noqa: BLE001
        pass
    _embed_master_metadata(path)                            # task 21: per ULTIMI (stego/watermark riscrivono il file)
    return path


def _embed_master_metadata(path: str) -> None:
    try:
        from core import gen_metadata
        gen_metadata.embed(path, {"label": "Film Studio (master)", "output_kind": "mp4"}, force=False)
    except Exception:                                       # noqa: BLE001
        pass


def _aspect_filter(aspect: str) -> str:
    """Filtro ffmpeg per letterbox/pillarbox all'aspetto scelto SENZA crop (B7: prima l'aspect era ignorato)."""
    R = {"16:9": 16 / 9, "2.39:1": 2.39, "9:16": 9 / 16, "1:1": 1.0}.get(aspect or "")
    if not R:
        return ""
    return (f"pad=w=ceil(max(iw\\,ih*{R:.5f})/2)*2:h=ceil(max(ih\\,iw/{R:.5f})/2)*2:"
            f"x=(ow-iw)/2:y=(oh-ih)/2:color=black")


def _probe_dur(path: str) -> float:
    """Durata (s) dello stream VIDEO via ffprobe (per gli offset delle transizioni).
    NON quella del container: nelle clip a2v l'audio paddato è più lungo del video
    (12.0 vs 11.88) → offset xfade oltre la fine del video → la catena non scattava
    mai e il master si troncava alla prima clip (film di test 12/07)."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=duration",
                            "-of", "default=nk=1:nw=1", path], capture_output=True, text=True)
        d = float((r.stdout or "0").strip() or 0)
        if d > 0:
            return d
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nk=1:nw=1", path], capture_output=True, text=True)
        return float((r.stdout or "0").strip() or 0)
    except Exception:                                       # noqa: BLE001
        return 0.0


def _build_silent_xfade(clips: list[str], transitions: list, silent: str, fps: float, vf: str) -> bool:
    """Master con TRANSIZIONI reali (xfade) invece di stacchi netti. transitions[i] = {type,dur} tra clip i e i+1
    (None/dur=0 = stacco). Ritorna True se ha prodotto il file."""
    durs = [_probe_dur(c) for c in clips]
    if any(d <= 0 for d in durs):
        return False                                        # durata ignota → ricadi sul concat
    inp = []
    for c in clips:
        inp += ["-i", c]
    filt = []
    for i in range(len(clips)):                             # normalizza ogni input (aspect+fps+formato)
        # setpts=PTS-STARTPTS: le clip a2v (audio incorporato) hanno timestamp che NON partono da 0
        # → la catena xfade collassava alla prima transizione (master 11.96s su ~200, "ok" rc=0).
        chain = (vf + "," if vf else "") + f"fps={fps},format=yuv420p,setsar=1,setpts=PTS-STARTPTS"
        filt.append(f"[{i}:v]{chain}[n{i}]")
    prev, total = "[n0]", durs[0]
    for i in range(1, len(clips)):
        tr = transitions[i - 1] or {}
        d = float(tr.get("dur", 0.0) or 0.0)
        ttype = tr.get("type") or "fade"
        d = max(0.05, min(d, durs[i] - 0.05, total - 0.05)) if d > 0 else 0.05
        off = max(0.0, total - d)
        out_lbl = f"[x{i}]"
        filt.append(f"{prev}[n{i}]xfade=transition={ttype}:duration={d:.3f}:offset={off:.3f}{out_lbl}")
        prev, total = out_lbl, off + durs[i]
    r = subprocess.run(["ffmpeg", "-v", "error", *inp, "-filter_complex", ";".join(filt),
                        "-map", prev, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
                        silent, "-y"], check=False, capture_output=True, text=True)
    if r.returncode != 0:                                    # PRIMA falliva muto → fallback silenzioso
        print(f"[master] xfade fallito → fallback concat. ffmpeg: {(r.stderr or '')[-300:]}", flush=True)
    return r.returncode == 0 and os.path.exists(silent)


def _apply_trims(clips: list[str], trims: list | None, out_path: str, fps: float) -> list[str]:
    """Pre-taglia le clip con in_s/out_s (trim=montaggio). Ritorna la lista clip effettiva (tagliate dove
    richiesto, passthrough altrove). trims[i] = {in_s, out_s} o None."""
    if not trims:
        return clips
    res = []
    for i, c in enumerate(clips):
        t = trims[i] if i < len(trims) else None
        a = float((t or {}).get("in_s", 0) or 0)
        b = float((t or {}).get("out_s", 0) or 0)
        if t and b > a:
            dst = str(Path(out_path).with_suffix(f".trim{i}.mp4"))
            subprocess.run(["ffmpeg", "-v", "error", "-i", c, "-ss", f"{a:.3f}", "-to", f"{b:.3f}",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), dst, "-y"], check=False)
            res.append(dst if os.path.exists(dst) else c)
        else:
            res.append(c)
    return res


def render_master(clips: list[str], out_path: str, *, audio: str = "",
                  audio_tracks: list[dict] | None = None, fps: float = 25.0, aspect: str = "",
                  transitions: list | None = None, trims: list | None = None) -> str:
    """Concatena le clip in un master unico (re-encode) + audio.
    audio_tracks=[{path, start_s, volume}] = mix multitraccia (dialoghi/musica/ambient/SFX) con
    ducking semplice via volume. `audio` singolo = retro-compat. aspect: 16:9/2.39:1/9:16/1:1 (letterbox).
    transitions=[{type,dur}] tra clip consecutive → dissolvenze reali (xfade); vuoto = stacchi netti (concat)."""
    if not clips:
        raise RuntimeError("nessuna clip da montare")
    clips = _apply_trims(clips, trims, out_path, fps)         # trim in/out prima di tutto (montaggio)
    silent = str(Path(out_path).with_suffix(".silent.mp4"))
    _vf = _aspect_filter(aspect)
    _has_tr = bool(transitions) and any((t or {}).get("dur", 0) for t in transitions)
    if not (_has_tr and _build_silent_xfade(clips, transitions, silent, fps, _vf)):
        # stacchi netti: concat A FILTRO (v=1:a=0), NON il demuxer -f concat — quello esige stream
        # OMOGENEI e con clip miste (a2v con audio incorporato + ti2v mute) TRONCAVA il master alla
        # prima clip (film di test 12/07: 18 clip → 11.96s). Il filtro ignora l'audio per design:
        # dialoghi e musica arrivano dal mix multitraccia sotto.
        _ex = [c for c in clips if os.path.exists(c)]
        _inp = []
        for c in _ex:
            _inp += ["-i", c]
        _filt = []
        for i in range(len(_ex)):
            chain = (_vf + "," if _vf else "") + f"fps={fps},format=yuv420p,setsar=1,setpts=PTS-STARTPTS"
            _filt.append(f"[{i}:v]{chain}[n{i}]")
        _filt.append("".join(f"[n{i}]" for i in range(len(_ex))) + f"concat=n={len(_ex)}:v=1:a=0[vout]")
        _r = subprocess.run(["ffmpeg", "-v", "error", *_inp, "-filter_complex", ";".join(_filt),
                             "-map", "[vout]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
                             silent, "-y"], check=False, capture_output=True, text=True)
        if _r.returncode != 0:
            print(f"[master] concat a filtro fallito: {(_r.stderr or '')[-300:]}", flush=True)
    tracks = list(audio_tracks or ([] if not audio else [{"path": audio, "start_s": 0, "volume": 1.0}]))
    tracks = [t for t in tracks if t.get("path") and os.path.exists(t["path"])]
    if not tracks:
        os.replace(silent, out_path); return _watermark_master(out_path)
    # mix multitraccia: ogni traccia ritardata (adelay) e attenuata (volume) → amix.
    # La durata del MASTER è quella del VIDEO ('-t vdur', NON '-shortest': tagliava il film
    # alla fine dell'ultima battuta — 147s su 196 nel film di test). track['loop']=True fa
    # ripetere la traccia (musica corta) fino a fine film.
    vdur = _probe_dur(silent)
    cmd = ["ffmpeg", "-v", "error", "-i", silent]
    for t in tracks:
        if t.get("loop"):
            cmd += ["-stream_loop", "-1"]
        cmd += ["-i", t["path"]]
    filt = []
    for i, t in enumerate(tracks, start=1):
        delay = int(float(t.get("start_s", 0)) * 1000)
        filt.append(f"[{i}:a]adelay={delay}|{delay},volume={float(t.get('volume',1.0))}[a{i}]")
    mix_in = "".join(f"[a{i}]" for i in range(1, len(tracks) + 1))
    filt.append(f"{mix_in}amix=inputs={len(tracks)}:duration=longest:dropout_transition=0[aout]")
    cmd += ["-filter_complex", ";".join(filt), "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac",
            *(["-t", f"{vdur:.3f}"] if vdur > 0 else ["-shortest"]), out_path, "-y"]
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
