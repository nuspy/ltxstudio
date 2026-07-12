"""Orchestratore HEADLESS del film completo — condiviso tra Cinema Studio («Genera film completo») e
Lucky Director. Nessuna dipendenza Qt: pilota il Job Scheduler e attende via polling dello stato job.

Sequenza (identica alla modalità Auto della UI): keyframe di tutte le riprese → attesa → video → attesa →
dialoghi (voce legata all'attore, temporizzati) → musica strumentale (opz.) → MASTER con transizioni a
DUE LIVELLI (scene vs riprese) e formato/aspect scelti.

Riusa: gen_service.compose_shot_prompt / handlers, scheduler, voice_studio, music_service, montage_render,
otio_export. Emette avanzamento via callback on_progress(phase, index, total, prompt, pct).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from core.film_studio import db, gen_service, scheduler, handlers, montage_render, otio_export


def _wav_dur(path: str) -> float:
    """Durata reale di un wav (per sfasare le battute). Fallback 1.5s se illeggibile."""
    try:
        import wave
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:                                        # noqa: BLE001
        return 1.5


def _emit(cb, phase: str, index: int, total: int, prompt: str, pct: float) -> None:
    if not cb:
        return
    try:
        cb(phase, int(index), int(total), prompt or "", float(pct))
    except Exception:                                        # noqa: BLE001
        pass


def _iter_shots(pid: int) -> list[dict]:
    """Riprese ORDINATE del progetto con scena/stato/outdir già risolti."""
    out = []
    for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
        state = json.loads(sc.get("continuity") or "{}")
        outdir = db.project_dir(pid) / f"scene_{sc['ord']}"
        outdir.mkdir(parents=True, exist_ok=True)
        for sh in db.query("shot", "scene_id=?", (sc["id"],), order="ord"):
            out.append({"sc": sc, "sh": sh, "state": state, "outdir": str(outdir)})
    return out


def _wait(jids: list[int], jmap: dict, phase: str, on_progress, cancel,
          base: float, span: float) -> None:
    """Attende il completamento (done/error) di tutti i jid, emettendo progresso. Rispetta cancel()."""
    total = len(jids) or 1
    done: set[int] = set()
    while len(done) < len(jids):
        if cancel and cancel():
            raise RuntimeError("Operazione annullata")
        running = ""
        for jid in jids:
            job = db.get("job", jid) or {}
            st = job.get("status")
            if st in ("done", "error"):
                done.add(jid)
            elif st == "running":
                running = jmap.get(jid, "")
        idx = len(done)
        _emit(on_progress, phase, min(idx + 1, len(jids)), len(jids), running,
              base + span * (idx / total))
        if len(done) >= len(jids):
            break
        time.sleep(0.5)


def _synth_dialogue_tracks(pid: int) -> list[dict]:
    """Sintetizza i dialoghi (voce legata all'attore) temporizzati sull'intero film → tracce audio per il mix.
    Stessa logica di montage_tab._gen_audio, headless. Non fatale."""
    try:
        from core.film_studio import voice_studio as vs
    except Exception:                                        # noqa: BLE001
        return []
    tracks: list[dict] = []
    t = 0.0
    for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
        for sh in db.query("shot", "scene_id=?", (sc["id"],), order="ord"):
            dur = float(sh["duration_s"] or 4)
            local = 0.0
            for di, d in enumerate(json.loads(sh.get("dialogues") or "[]")):
                actor = (d.get("actor") or "").lstrip("@")
                text = d.get("text") or ""
                if not text:
                    continue
                assets = [a for a in db.find_assets(actor, "actor") if a["name"].lower() == actor.lower()]
                vrow = vs.voice_for_actor(assets[0]["id"]) if assets else None
                if not vrow:
                    continue
                out = str(db.project_dir(pid) / f"scene_{sc['ord']}" / f"shot_{sh['ord']}_d{di}.wav")
                try:
                    vs.synth_dialogue(text, vrow, out)
                    tracks.append({"path": out, "start_s": t + local, "volume": 1.0})
                    local += _wav_dur(out) + 0.15
                except Exception:                            # noqa: BLE001
                    pass
            t += dur
    return tracks


def _shot_dialogue_audio(sh: dict, outdir: str) -> str:
    """PRIMO dialogo dello shot → wav (per il lip-sync a2v del video). Headless. Vedi storyboard equivalente.
    Il wav viene PADDATO DI SILENZIO alla durata esatta dello shot: la pipeline a2v esige che
    l'audio copra l'intera clip (battuta 5s su shot 12s → AssertionError 'Latent shape [125]
    does not match [297]' = i 7 shot col parlato persi nel film di test 12/07)."""
    try:
        from core.film_studio import voice_studio as vs
        for d in json.loads(sh.get("dialogues") or "[]"):
            actor = (d.get("actor") or "").lstrip("@")
            text = d.get("text") or ""
            if not text:
                continue
            assets = [a for a in db.find_assets(actor, "actor") if a["name"].lower() == actor.lower()]
            vrow = vs.voice_for_actor(assets[0]["id"]) if assets else None
            if not vrow:
                continue
            wav = str(Path(outdir) / f"shot_{sh['ord']}_dlg.wav")
            vs.synth_dialogue(text, vrow, wav)
            dur = float(sh.get("duration_s") or 0)
            if dur > 0:
                import shutil as _sh
                import subprocess as _sp
                padded = str(Path(outdir) / f"shot_{sh['ord']}_dlg_pad.wav")
                _ff = _sh.which("ffmpeg") or "ffmpeg"
                _sp.run([_ff, "-y", "-v", "error", "-i", wav, "-af", "apad",
                         "-t", f"{dur:.3f}", padded], capture_output=True, timeout=120)
                if Path(padded).exists() and Path(padded).stat().st_size > 1000:
                    return padded
            return wav
    except Exception:                                        # noqa: BLE001
        pass
    return ""


def _film_mood(pid: int) -> str:
    moods = []
    for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
        st = json.loads(sc.get("continuity") or "{}")
        if st.get("mood"):
            moods.append(st["mood"])
    return ", ".join(dict.fromkeys(moods)) or "cinematic, atmospheric"


def _make_music(pid: int, total_s: float) -> dict | None:
    """Colonna sonora strumentale (blocking, headless). Ritorna una traccia audio o None. Non fatale."""
    try:
        from core import music_service as ms
        dest = str(db.project_dir(pid) / "music.mp3")
        p = ms.SongParams(name=f"fs_music_{pid}", lyrics="",
                          tags=f"{_film_mood(pid)}, instrumental, soundtrack",
                          duration_s=min(max(total_s, 20.0), 180.0), instrumental=True)
        if ms.preflight_song(p):
            return None
        ev = threading.Event()
        job = ms.make_song_job(p, on_done=lambda *_a: ev.set())
        if hasattr(job, "start"):
            job.start()
        ev.wait(timeout=600)
        src = p.out_mp3()
        if src and Path(str(src)).exists():
            try:
                import shutil
                shutil.copy2(str(src), dest)
                src = dest
            except Exception:                                # noqa: BLE001
                pass
            # loop=True: se ACE consegna una traccia più corta del film (visto 23.9s su 180 richiesti,
            # instabilità #15) la musica si ripete fino a fine video invece di lasciare il silenzio.
            return {"path": str(src), "start_s": 0, "volume": 0.3, "loop": True}   # ducking sotto i dialoghi
    except Exception:                                        # noqa: BLE001
        pass
    return None


def run(pid: int, *, aspect: str = "", scene_trans: str = "", shot_trans: str = "",
        scene_dur: float = 0.8, shot_dur: float = 0.5, fps: float = 25.0, make_music: bool = True,
        width: int = 0, height: int = 0, on_progress=None, cancel=None) -> dict:
    """Esegue l'intera pipeline (keyframe→video→audio→musica→master) in modo headless. Ritorna
    {'project_id', 'master'}. on_progress(phase, index, total, prompt, pct); cancel() → bool."""
    handlers.register_default_handlers()                     # assicura gli handler anche senza finestra UI
    proj = db.get("project", pid) or {}
    guard = json.loads(proj.get("config") or "{}")
    pmode = proj.get("path_mode", "fast")
    img_arch, img_model = guard.get("img_arch", "flux"), guard.get("img_model", "")
    vid_gguf, forbidden = guard.get("vid_gguf", ""), guard.get("forbidden", "")
    shots = _iter_shots(pid)
    if not shots:
        raise RuntimeError("Nessuna ripresa da generare (breakdown mancante)")

    # ── casting voci automatico: senza voce legata all'attore le battute verrebbero SALTATE
    # (film muto) e il lip-sync a2v della FASE 2 non avrebbe l'audio. Voci distinte per attore.
    try:
        from core.film_studio import voice_studio as vs
        n_cast = vs.auto_cast_voices(pid)
        if n_cast:
            print(f"[auto_run] casting voci automatico: {n_cast} attori doppiati", flush=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"[auto_run] casting voci fallito (non fatale): {e}", flush=True)

    # ── FASE 1: keyframe ──────────────────────────────────────────────────────
    kf_jids, jmap = [], {}
    for it in shots:
        sh, state, outdir = it["sh"], it["state"], it["outdir"]
        prompt, refs = gen_service.compose_shot_prompt(sh, state, guard, path_mode=pmode, flavor="image")
        seed = int(sh.get("seed", 0) or 0) or (int(sh["id"]) * 2654435761 % (2 ** 31 - 1))
        out = str(Path(outdir) / f"shot_{sh['ord']}_kf.png")
        if Path(out).exists() and Path(out).stat().st_size > 1000:   # IDEMPOTENTE: ripresa/riparazione
            continue                                                 # (rilanciare run() rigenera SOLO i mancanti)
        jid = scheduler.enqueue("image", {"prompt": prompt, "out": out,
                                          "face_ref": (refs["ref_images"] or [""])[0],
                                          "ref_images": refs["ref_images"], "loras": refs["loras"],
                                          "arch": img_arch, "model": img_model,
                                          "seed": seed, "negative": forbidden,
                                          "width": width or 0, "height": height or 0},
                                owner_kind="shot", owner_id=sh["id"])
        kf_jids.append(jid); jmap[jid] = prompt
    _wait(kf_jids, jmap, "keyframe", on_progress, cancel, base=0.0, span=0.40)

    # ── FASE 2: video ─────────────────────────────────────────────────────────
    # Worker video PERSISTENTE (:8190): avviato QUI (dopo i keyframe, così non contende la VRAM
    # al FLUX) se non è già su. Il modello 22B si carica UNA volta al primo shot e resta per
    # tutti gli altri — senza, ogni shot rilegge 29GB da disco (visto live: 4 min/clip di cui
    # 2 di solo load). free_for('video') in gen_shot_video svuota prima i server immagine.
    try:
        import os as _os
        if _os.environ.get("LTX2_NO_PERSISTENT") == "1":
            raise RuntimeError("disabilitato da LTX2_NO_PERSISTENT (bug #160 in corso)")
        if not gen_service._ltx2_server_up():
            import subprocess as _sp
            _flags = getattr(_sp, "DETACHED_PROCESS", 0) | getattr(_sp, "CREATE_NEW_PROCESS_GROUP", 0)
            _srv_log = open(str(db.ROOT / "ltx2_server_8190.log"), "a", encoding="utf-8")
            _sp.Popen([gen_service.LTX2_PY, "-u",
                       str(Path(__file__).resolve().parents[2] / "workers" / "ltx2_server.py")],
                      stdout=_srv_log, stderr=_sp.STDOUT, creationflags=_flags,
                      env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"})
            for _ in range(20):                              # attesa best-effort (poi fallback subprocess)
                time.sleep(1)
                if gen_service._ltx2_server_up():
                    print("[auto_run] worker video persistente :8190 avviato", flush=True)
                    break
    except Exception as _e:                                  # noqa: BLE001
        print(f"[auto_run] server :8190 non avviato ({_e}) → fallback subprocess per-shot", flush=True)
    vid_jids, jmap = [], {}
    for it in shots:
        sh, state, outdir = it["sh"], it["state"], it["outdir"]
        prompt, refs = gen_service.compose_shot_prompt(sh, state, guard, path_mode=pmode, flavor="video")
        seed = int(sh.get("seed", 0) or 0) or (int(sh["id"]) * 2654435761 % (2 ** 31 - 1))
        out = str(Path(outdir) / f"shot_{sh['ord']}.mp4")
        if Path(out).exists() and Path(out).stat().st_size > 100_000:   # IDEMPOTENTE (vedi keyframe)
            continue
        kf = str(Path(outdir) / f"shot_{sh['ord']}_kf.png")
        kfs = [{"path": kf, "frame": 0, "strength": 1.0}] if Path(kf).exists() else []
        jid = scheduler.enqueue("video", {"prompt": prompt, "out": out,
                                          "duration_s": sh["duration_s"] or 4, "keyframes": kfs,
                                          "loras": refs["loras"],
                                          "audio_path": _shot_dialogue_audio(sh, outdir),
                                          "model_gguf": vid_gguf, "seed": seed, "negative": forbidden,
                                          "width": width or 0, "height": height or 0},
                                owner_kind="shot", owner_id=sh["id"])
        vid_jids.append(jid); jmap[jid] = prompt
    _wait(vid_jids, jmap, "video", on_progress, cancel, base=0.40, span=0.40)

    # ── FASE 3: audio dialoghi ────────────────────────────────────────────────
    _emit(on_progress, "audio", 1, 1, "", 0.82)
    audio_tracks = _synth_dialogue_tracks(pid)

    # ── FASE 4: musica (opzionale) ────────────────────────────────────────────
    clips_meta = []
    for it in shots:
        sc, sh, outdir = it["sc"], it["sh"], it["outdir"]
        f = str(Path(outdir) / f"shot_{sh['ord']}.mp4")
        if Path(f).exists():
            clips_meta.append({"file": f, "scene_id": sc["id"], "transition": sh.get("transition") or ""})
    total_s = sum(float(it["sh"]["duration_s"] or 4) for it in shots)
    if make_music:
        _emit(on_progress, "music", 1, 1, "", 0.88)
        mt = _make_music(pid, total_s)
        if mt:
            audio_tracks.append(mt)

    # ── FASE 5: master con transizioni a due livelli ─────────────────────────
    _emit(on_progress, "master", 1, 1, "", 0.95)
    if not clips_meta:
        raise RuntimeError("Nessuna clip video prodotta: master non generabile")
    trans = montage_render.build_transitions([c for c in clips_meta], scene_trans=scene_trans,
                                             shot_trans=shot_trans, scene_dur=scene_dur, shot_dur=shot_dur)
    trans = trans if montage_render.has_any(trans) else None
    out_master = str(db.project_dir(pid) / "master.mp4")
    otio_export.render_master([c["file"] for c in clips_meta], out_master,
                              audio_tracks=audio_tracks or None, fps=fps, aspect=aspect, transitions=trans)
    _emit(on_progress, "done", 1, 1, "", 1.0)
    return {"project_id": pid, "master": out_master}
