"""Tab Montaggio/Master (spec §11): timeline dalle clip generate → (A) master ffmpeg + (B) export NLE.
Versione funzionale: raccoglie i video degli shot in ordine, render master e/o export OTIO."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QLabel, QListWidget, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

from core.film_studio import db, otio_export
from ui.i18n_manager import tr

_FMT = [("fcpxml", "FCPXML (Filmora/Resolve)"), ("otio", "OpenTimelineIO"),
        ("aaf", "AAF (Premiere/Avid)"), ("edl", "EDL")]
_ASPECT = [("16:9", "16:9"), ("2.39:1", "2.39:1 Cinemascope"), ("9:16", "9:16 Vertical"), ("1:1", "1:1")]


class MontageTab(QWidget):
    def __init__(self, get_project):
        super().__init__()
        self._get_project = get_project
        v = QVBoxLayout(self)
        h = QHBoxLayout()
        self.btn_refresh = QPushButton(tr("fs.mt.refresh", "↻ Aggiorna timeline")); self.btn_refresh.clicked.connect(self._refresh)
        h.addWidget(self.btn_refresh); h.addStretch(1)
        v.addLayout(h)
        self.lst = QListWidget(); v.addWidget(self.lst, 1)
        box = QGroupBox(tr("fs.mt.output", "Output")); bv = QVBoxLayout(box)
        r1 = QHBoxLayout()
        self.aspect = QComboBox()
        for code, lab in _ASPECT:
            self.aspect.addItem(lab, code)
        r1.addWidget(QLabel(tr("fs.mt.aspect", "Aspetto"))); r1.addWidget(self.aspect)
        self.fmt = QComboBox()
        for code, lab in _FMT:
            self.fmt.addItem(lab, code)
        r1.addWidget(QLabel(tr("fs.mt.nle", "Formato NLE"))); r1.addWidget(self.fmt); r1.addStretch(1)
        bv.addLayout(r1)
        r2 = QHBoxLayout()
        self.btn_audio = QPushButton(tr("fs.mt.audio", "🔊 Genera dialoghi")); self.btn_audio.clicked.connect(self._gen_audio)
        self.btn_music = QPushButton(tr("fs.mt.music", "🎵 Colonna sonora")); self.btn_music.clicked.connect(self._gen_music)
        self.btn_ambient = QPushButton(tr("fs.mt.ambient", "🌧 Ambient/SFX")); self.btn_ambient.clicked.connect(self._gen_ambient)
        r2.addWidget(self.btn_music); r2.addWidget(self.btn_ambient)
        self.btn_master = QPushButton(tr("fs.mt.master", "🎞 Master (A)")); self.btn_master.clicked.connect(self._master)
        self.btn_export = QPushButton(tr("fs.mt.export", "📦 Export NLE (B)")); self.btn_export.clicked.connect(self._export)
        r2.addWidget(self.btn_audio); r2.addWidget(self.btn_master); r2.addWidget(self.btn_export); r2.addStretch(1)
        bv.addLayout(r2)
        r3 = QHBoxLayout()
        self.btn_animatic = QPushButton(tr("fs.mt.animatic", "🎞 Animatic (anteprima)")); self.btn_animatic.clicked.connect(self._animatic)
        self.btn_dub = QPushButton(tr("fs.mt.dub", "🌐 Doppia in lingua…")); self.btn_dub.clicked.connect(self._dub)
        r3.addWidget(self.btn_animatic); r3.addWidget(self.btn_dub); r3.addStretch(1)
        bv.addLayout(r3)
        self.info = QLabel(""); bv.addWidget(self.info)
        v.addWidget(box)
        self._refresh()

    def _animatic(self):
        """1.3 Animatic: sequenza keyframe coi tempi (pass rapido a basso costo) + audio dialoghi se presente."""
        import subprocess
        pid = self._get_project()
        if not pid:
            return
        frames = []   # (img, dur)
        for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
            for sh in db.query("shot", "scene_id=?", (sc["id"],), order="ord"):
                kf = str(db.project_dir(pid) / f"scene_{sc['ord']}" / f"shot_{sh['ord']}_kf.png")
                if Path(kf).exists():
                    frames.append((kf, float(sh["duration_s"] or 3)))
        if not frames:
            QMessageBox.information(self, tr("fs.mt.animatic", "Animatic"), tr("fs.mt.no_kf", "Genera prima i keyframe.")); return
        lst = str(db.project_dir(pid) / "_animatic.txt")
        with open(lst, "w", encoding="utf-8") as fh:
            for img, dur in frames:
                fh.write(f"file '{img}'\nduration {dur}\n")
            fh.write(f"file '{frames[-1][0]}'\n")           # ultimo frame ripetuto (richiesto da concat demuxer)
        out = str(db.project_dir(pid) / "animatic.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                        "-vf", "scale=768:-2,format=yuv420p", "-r", "25", out, "-y"], check=False)
        self.info.setText(tr("fs.mt.animatic_done", "Animatic: {f}").format(f=out))
        try:
            import os as _os; _os.startfile(out)             # noqa: S606
        except Exception:                                   # noqa: BLE001
            pass

    def _dub(self):
        """1.5 Multilingua/doppiaggio: LLM traduce i dialoghi nella lingua scelta + risintesi voce legata."""
        from PySide6.QtWidgets import QInputDialog
        from core import llm_service as L
        from core.film_studio import voice_studio as vs
        pid = self._get_project()
        if not pid:
            return
        lang, ok = QInputDialog.getText(self, tr("fs.mt.dub", "Doppia"), tr("fs.mt.dub_lang", "Lingua di destinazione (es. English, Español):"))
        if not ok or not lang.strip():
            return
        import json
        n = 0
        for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
            for sh in db.query("shot", "scene_id=?", (sc["id"],), order="ord"):
                for di, d in enumerate(json.loads(sh.get("dialogues") or "[]")):
                    actor = (d.get("actor") or "").lstrip("@"); text = d.get("text") or ""
                    if not text:
                        continue
                    try:
                        tr_text = L.complete("Translate to " + lang.strip() + ". Output only the translation.", text, max_tokens=200)
                    except Exception:                       # noqa: BLE001
                        tr_text = text
                    assets = [a for a in db.find_assets(actor, "actor") if a["name"].lower() == actor.lower()]
                    vrow = vs.voice_for_actor(assets[0]["id"]) if assets else None
                    if not vrow:
                        continue
                    out = str(db.project_dir(pid) / f"scene_{sc['ord']}" / f"shot_{sh['ord']}_d{di}_{lang.strip()[:2]}.wav")
                    try:
                        vs.synth_dialogue(tr_text, vrow, out); n += 1
                    except Exception:                       # noqa: BLE001
                        pass
        self.info.setText(tr("fs.mt.dub_done", "Doppiaggio {l}: {n} battute (re-lip-sync rigenerando i video).").format(l=lang.strip(), n=n))

    def _clips(self) -> list[dict]:
        pid = self._get_project()
        if not pid:
            return []
        out = []
        for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
            for sh in db.query("shot", "scene_id=?", (sc["id"],), order="ord"):
                f = str(db.project_dir(pid) / f"scene_{sc['ord']}" / f"shot_{sh['ord']}.mp4")
                if Path(f).exists():
                    out.append({"file": f, "in_s": 0.0, "out_s": float(sh["duration_s"] or 4),
                                "name": f"S{sc['ord']}.{sh['ord']}"})
        return out

    def _refresh(self):
        self.lst.clear()
        self._cur_clips = self._clips()
        for c in self._cur_clips:
            self.lst.addItem(f"{c['name']}  ({c['out_s']:.0f}s)  {Path(c['file']).name}")
        self.info.setText(tr("fs.mt.count", "{n} clip in timeline").format(n=len(self._cur_clips)))

    def _gen_audio(self):
        """Sintetizza i dialoghi (voce legata all'attore) per ogni shot → tracce audio temporizzate."""
        import json
        from core.film_studio import voice_studio as vs
        pid = self._get_project()
        if not pid:
            return
        self._audio_tracks = []
        t = 0.0
        n = 0
        for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
            for sh in db.query("shot", "scene_id=?", (sc["id"],), order="ord"):
                dur = float(sh["duration_s"] or 4)
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
                        self._audio_tracks.append({"path": out, "start_s": t, "volume": 1.0})
                        n += 1
                    except Exception:                       # noqa: BLE001
                        pass
                t += dur
        self.info.setText(tr("fs.mt.audio_done", "Audio: {n} battute sintetizzate").format(n=n))

    def _film_mood(self) -> str:
        import json
        pid = self._get_project()
        moods = []
        for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
            st = json.loads(sc.get("continuity") or "{}")
            if st.get("mood"):
                moods.append(st["mood"])
        return ", ".join(dict.fromkeys(moods)) or "cinematic, atmospheric"

    def _gen_music(self):
        """Colonna sonora strumentale (HeartMuLa) col mood del film → traccia musica (volume ridotto = ducking)."""
        import time as _t
        from core import music_service as ms
        pid = self._get_project()
        if not pid:
            return
        total = sum(float(c["out_s"]) for c in (self._cur_clips or [])) or 60.0
        out = str(db.project_dir(pid) / "music.mp3")
        p = ms.SongParams(name=f"fs_music_{pid}", lyrics="", tags=f"{self._film_mood()}, instrumental, soundtrack",
                          duration_s=min(total, 180), instrumental=True)
        errs = ms.preflight_song(p)
        if errs:
            QMessageBox.warning(self, tr("fs.mt.music", "Musica"), "; ".join(errs)); return
        self.btn_music.setEnabled(False); self.info.setText(tr("fs.working", "Generazione…"))
        self._mjob = ms.make_song_job(p, on_done=lambda rc: self._on_music(rc, p.out_mp3(), total))
        self._mjob.start() if hasattr(self._mjob, "start") else None

    def _on_music(self, rc, path, total):
        self.btn_music.setEnabled(True)
        if os.path.exists(str(path)):
            self._extra_tracks = getattr(self, "_extra_tracks", [])
            self._extra_tracks.append({"path": str(path), "start_s": 0, "volume": 0.3})  # ducking sotto i dialoghi
            self.info.setText(tr("fs.mt.music_done", "Colonna sonora pronta."))
        else:
            self.info.setText(tr("fs.mt.music_fail", "Colonna sonora non generata."))

    def _gen_ambient(self):
        """Ambient/SFX via MMAudio sul master muto (se generato), come traccia di fondo."""
        from core import music_service as ms
        pid = self._get_project()
        master = str(db.project_dir(pid) / "master.mp4")
        if not os.path.exists(master):
            QMessageBox.information(self, tr("fs.mt.ambient", "Ambient"), tr("fs.mt.need_master", "Genera prima il Master (A).")); return
        out = str(db.project_dir(pid) / "ambient")
        p = ms.SonifyParams(name=f"fs_amb_{pid}", video_path=master,
                            prompt=self._film_mood() + ", ambient background, room tone")
        errs = ms.preflight_sonify(p)
        if errs:
            QMessageBox.warning(self, tr("fs.mt.ambient", "Ambient"), "; ".join(errs)); return
        self.btn_ambient.setEnabled(False); self.info.setText(tr("fs.working", "Generazione…"))
        self._ajob = ms.make_sonify_job(p, on_done=lambda rc: self._on_ambient(rc))
        self._ajob.start() if hasattr(self._ajob, "start") else None

    def _on_ambient(self, rc):
        self.btn_ambient.setEnabled(True)
        from core import music_service as ms
        f = ms.find_latest_audio()
        if f and os.path.exists(str(f)):
            self._extra_tracks = getattr(self, "_extra_tracks", [])
            self._extra_tracks.append({"path": str(f), "start_s": 0, "volume": 0.4})
            self.info.setText(tr("fs.mt.ambient_done", "Ambient pronto."))
        else:
            self.info.setText(tr("fs.mt.ambient_fail", "Ambient non generato."))

    def _master(self):
        clips = [c["file"] for c in (self._cur_clips or [])]
        if not clips:
            QMessageBox.information(self, tr("fs.mt.master", "Master"), tr("fs.mt.no_clips", "Nessuna clip generata.")); return
        pid = self._get_project()
        out = str(db.project_dir(pid) / "master.mp4")
        tracks = list(getattr(self, "_audio_tracks", []) or []) + list(getattr(self, "_extra_tracks", []) or [])
        try:
            otio_export.render_master(clips, out, audio_tracks=tracks or None)
            self.info.setText(tr("fs.mt.master_done", "Master: {f}").format(f=out))
        except Exception as e:                              # noqa: BLE001
            QMessageBox.critical(self, tr("fs.mt.master", "Master"), str(e)[:600])

    def _export(self):
        if not self._cur_clips:
            QMessageBox.information(self, tr("fs.mt.export", "Export"), tr("fs.mt.no_clips", "Nessuna clip generata.")); return
        pid = self._get_project()
        out_dir = str(db.project_dir(pid) / "export")
        try:
            p = otio_export.export_nle(self._cur_clips, out_dir, fmt=self.fmt.currentData(),
                                       name=f"project_{pid}")
            self.info.setText(tr("fs.mt.export_done", "Export: {f}").format(f=p))
        except Exception as e:                              # noqa: BLE001
            QMessageBox.critical(self, tr("fs.mt.export", "Export"), str(e)[:600])
