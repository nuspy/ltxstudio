"""Tab Montaggio/Master (spec §11): timeline dalle clip generate → (A) master ffmpeg + (B) export NLE.
Versione funzionale: raccoglie i video degli shot in ordine, render master e/o export OTIO."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QLabel, QListWidget, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

from core.film_studio import db, otio_export
from ui.i18n_manager import tr

_FMT = [("fcpxml", "FCPXML (Filmora/Resolve)"), ("otio", "OpenTimelineIO"),
        ("aaf", "AAF (Premiere/Avid)"), ("edl", "EDL")]
_ASPECT = [("16:9", "16:9"), ("2.39:1", "2.39:1 Cinemascope"), ("9:16", "9:16 Vertical"), ("1:1", "1:1")]
_TRANS = [("", "Stacco netto"), ("fade", "Dissolvenza"), ("fadeblack", "Dissolvenza in nero"),
          ("dissolve", "Dissolve"), ("wipeleft", "Tendina ←"), ("slideright", "Scorri →"),
          ("circleopen", "Cerchio")]


def _wav_dur(path: str) -> float:
    """Durata reale di un wav (per sfasare le battute nello shot). Fallback 1.5s se illeggibile."""
    try:
        import wave
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:                                       # noqa: BLE001
        return 1.5


class MontageTab(QWidget):
    def __init__(self, get_project):
        super().__init__()
        self._get_project = get_project
        v = QVBoxLayout(self)
        h = QHBoxLayout()
        self.btn_refresh = QPushButton(tr("fs.mt.refresh", "↻ Aggiorna timeline")); self.btn_refresh.clicked.connect(self._refresh)
        h.addWidget(self.btn_refresh); h.addStretch(1)
        v.addLayout(h)
        # #53 timeline visuale (clip video trascinabili + tracce audio)
        from ui.film_studio.timeline_widget import TimelineWidget
        self.timeline = TimelineWidget()
        self.timeline.reordered.connect(self._on_reorder)
        v.addWidget(self.timeline)
        self.lst = QListWidget(); v.addWidget(self.lst, 1)
        box = QGroupBox(tr("fs.mt.output", "Output")); bv = QVBoxLayout(box)
        r1 = QHBoxLayout()
        self.aspect = QComboBox()
        for code, lab in _ASPECT:
            self.aspect.addItem(lab, code)
        r1.addWidget(QLabel(tr("fs.mt.aspect", "Aspetto"))); r1.addWidget(self.aspect)
        self.trans = QComboBox()                             # transizione di default tra le clip
        for code, lab in _TRANS:
            self.trans.addItem(lab, code)
        r1.addWidget(QLabel(tr("fs.mt.trans", "Transizione"))); r1.addWidget(self.trans)
        from PySide6.QtWidgets import QDoubleSpinBox as _DSB
        self.trans_dur = _DSB(); self.trans_dur.setRange(0.1, 3.0); self.trans_dur.setSingleStep(0.1)
        self.trans_dur.setValue(0.5); self.trans_dur.setSuffix(" s")
        r1.addWidget(self.trans_dur)
        self.fmt = QComboBox()
        for code, lab in _FMT:
            self.fmt.addItem(lab, code)
        r1.addWidget(QLabel(tr("fs.mt.nle", "Formato NLE"))); r1.addWidget(self.fmt); r1.addStretch(1)
        bv.addLayout(r1)
        r2 = QHBoxLayout()
        self.btn_audio = QPushButton(tr("fs.mt.audio", "🔊 Genera dialoghi")); self.btn_audio.clicked.connect(self._gen_audio)
        self.btn_music = QPushButton(tr("fs.mt.music", "🎵 Colonna sonora")); self.btn_music.clicked.connect(self._gen_music)
        self.btn_ambient = QPushButton(tr("fs.mt.ambient", "🌧 Ambient")); self.btn_ambient.clicked.connect(self._gen_ambient)
        self.btn_sfx = QPushButton(tr("fs.mt.sfx", "💥 SFX per-azione")); self.btn_sfx.clicked.connect(self._gen_sfx)
        r2.addWidget(self.btn_music); r2.addWidget(self.btn_ambient); r2.addWidget(self.btn_sfx)
        self.btn_master = QPushButton(tr("fs.mt.master", "🎞 Master (A)")); self.btn_master.clicked.connect(self._master)
        self.btn_export = QPushButton(tr("fs.mt.export", "📦 Export NLE (B)")); self.btn_export.clicked.connect(self._export)
        r2.addWidget(self.btn_audio); r2.addWidget(self.btn_master); r2.addWidget(self.btn_export); r2.addStretch(1)
        bv.addLayout(r2)
        r3 = QHBoxLayout()
        self.btn_animatic = QPushButton(tr("fs.mt.animatic", "🎞 Animatic (anteprima)")); self.btn_animatic.clicked.connect(self._animatic)
        self.dub_lang = QComboBox(); self.dub_lang.setEditable(True)
        self.dub_lang.addItems(["English", "Español", "Français", "Deutsch", "Português", "Italiano",
                                "中文", "日本語", "한국어", "Русский", "العربية", "हिन्दी"])
        from PySide6.QtWidgets import QCheckBox as _CK
        self.dub_relip = _CK(tr("fs.mt.relip", "re-lip-sync")); self.dub_relip.setChecked(True)
        self.btn_dub = QPushButton(tr("fs.mt.dub", "🌐 Doppia")); self.btn_dub.clicked.connect(self._dub)
        r3.addWidget(self.btn_animatic); r3.addStretch(1)
        r3.addWidget(QLabel(tr("fs.mt.dub_to", "Lingua"))); r3.addWidget(self.dub_lang)
        r3.addWidget(self.dub_relip); r3.addWidget(self.btn_dub)
        bv.addLayout(r3)
        # Moonshot — critico + montatore a ritmo emotivo (leggono il breakdown, propongono senza distruggere)
        r4 = QHBoxLayout()
        self.btn_secondlook = QPushButton(tr("fs.mt.secondlook", "🔍 Secondo sguardo (critico)"))
        self.btn_secondlook.setToolTip(tr("fs.mt.secondlook_tt", "Un critico rilegge il piano del film e dà "
                                          "il «no» creativo finale: rilievi concreti + punti di forza."))
        self.btn_secondlook.clicked.connect(self._second_look)
        self.btn_rhythm = QPushButton(tr("fs.mt.rhythm", "📉 Ritmo emotivo"))
        self.btn_rhythm.setToolTip(tr("fs.mt.rhythm_tt", "Disegna la curva di tensione per scena: modula le "
                                      "durate degli shot (tensione alta → tagli corti) mantenendo il totale scena."))
        self.btn_rhythm.clicked.connect(self._rhythm)
        r4.addWidget(self.btn_secondlook); r4.addWidget(self.btn_rhythm); r4.addStretch(1)
        bv.addLayout(r4)
        self.info = QLabel(""); bv.addWidget(self.info)
        v.addWidget(box)
        self._refresh()

    def _animatic(self):
        """#56 Animatic: sequenza keyframe coi tempi (pass rapido) + TRACCIA AUDIO DIALOGHI sincronizzata."""
        import os
        import subprocess
        pid = self._get_project()
        if not pid:
            return
        frames = []   # (img, dur, dialog_wav|"")
        for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
            scd = db.project_dir(pid) / f"scene_{sc['ord']}"
            for sh in db.query("shot", "scene_id=?", (sc["id"],), order="ord"):
                kf = str(scd / f"shot_{sh['ord']}_kf.png")
                if Path(kf).exists():
                    wavs = sorted(scd.glob(f"shot_{sh['ord']}_d*.wav"))
                    frames.append((kf, float(sh["duration_s"] or 3), str(wavs[0]) if wavs else ""))
        if not frames:
            QMessageBox.information(self, tr("fs.mt.animatic", "Animatic"), tr("fs.mt.no_kf", "Genera prima i keyframe.")); return
        lst = str(db.project_dir(pid) / "_animatic.txt")
        with open(lst, "w", encoding="utf-8") as fh:
            for img, dur, _w in frames:
                fh.write(f"file '{img}'\nduration {dur}\n")
            fh.write(f"file '{frames[-1][0]}'\n")           # ultimo frame ripetuto (richiesto da concat demuxer)
        vid = str(db.project_dir(pid) / "_animatic_mute.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                        "-vf", "scale=768:-2,format=yuv420p", "-r", "25", vid, "-y"], check=False)
        out = str(db.project_dir(pid) / "animatic.mp4")
        if any(w for _i, _d, w in frames):                  # traccia audio: 1 segmento/shot (wav o silenzio)
            segdir = db.project_dir(pid) / "_anim_seg"; segdir.mkdir(exist_ok=True)
            segs = []
            for i, (_img, dur, wav) in enumerate(frames):
                seg = str(segdir / f"a{i}.wav")
                if wav and Path(wav).exists():
                    subprocess.run(["ffmpeg", "-v", "error", "-i", wav, "-af", f"apad,atrim=0:{dur}",
                                    "-ar", "44100", "-ac", "2", seg, "-y"], check=False)
                else:
                    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                                    "-t", str(dur), seg, "-y"], check=False)
                segs.append(seg)
            alst = str(segdir / "list.txt")
            with open(alst, "w", encoding="utf-8") as fh:
                for s in segs:
                    fh.write(f"file '{s}'\n")
            atrk = str(segdir / "track.wav")
            subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", alst, "-c", "copy", atrk, "-y"], check=False)
            subprocess.run(["ffmpeg", "-v", "error", "-i", vid, "-i", atrk, "-c:v", "copy", "-c:a", "aac",
                            "-shortest", out, "-y"], check=False)
        elif os.path.exists(vid):
            os.replace(vid, out)
        self.info.setText(tr("fs.mt.animatic_done", "Animatic: {f}").format(f=out))
        try:
            import os as _os; _os.startfile(out)             # noqa: S606
        except Exception:                                   # noqa: BLE001
            pass

    def _dub(self):
        """#52 Doppiaggio COMPLETO: LLM traduce i dialoghi nella lingua scelta (combo) + risintesi voce
        legata + RE-LIP-SYNC (riaccoda i video shot con dialogo, a2v sul nuovo audio)."""
        import json
        from core import llm_service as L
        from core.film_studio import voice_studio as vs, gen_service, scheduler, continuity as ct
        pid = self._get_project()
        if not pid:
            return
        lang = self.dub_lang.currentText().strip() if hasattr(self, "dub_lang") else "English"
        relip = self.dub_relip.isChecked() if hasattr(self, "dub_relip") else True
        n = m = 0
        for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
            state = json.loads(sc.get("continuity") or "{}")
            for sh in db.query("shot", "scene_id=?", (sc["id"],), order="ord"):
                dlgs = json.loads(sh.get("dialogues") or "[]")
                first_wav = ""
                for di, d in enumerate(dlgs):
                    actor = (d.get("actor") or "").lstrip("@"); text = d.get("text") or ""
                    if not text:
                        continue
                    try:
                        tr_text = L.complete(f"Translate to {lang}. Output only the translation.", text, max_tokens=200)
                    except Exception:                       # noqa: BLE001
                        tr_text = text
                    assets = [a for a in db.find_assets(actor, "actor") if a["name"].lower() == actor.lower()]
                    vrow = vs.voice_for_actor(assets[0]["id"]) if assets else None
                    if not vrow:
                        continue
                    out = str(db.project_dir(pid) / f"scene_{sc['ord']}" / f"shot_{sh['ord']}_d{di}_{lang[:2].lower()}.wav")
                    try:
                        vs.synth_dialogue(tr_text, vrow, out); n += 1
                        first_wav = first_wav or out
                    except Exception:                       # noqa: BLE001
                        pass
                # RE-LIP-SYNC: riaccoda il video dello shot con il nuovo audio doppiato (a2v)
                if relip and first_wav:
                    kf = str(db.project_dir(pid) / f"scene_{sc['ord']}" / f"shot_{sh['ord']}_kf.png")
                    kfs = [{"path": kf, "frame": 0, "strength": 1.0}] if Path(kf).exists() else []
                    out_v = str(db.project_dir(pid) / f"scene_{sc['ord']}" / f"shot_{sh['ord']}_{lang[:2].lower()}.mp4")
                    scheduler.enqueue("video", {"prompt": (sh.get("description") or "") + ", " + ct.to_prompt(state),
                                                "out": out_v, "duration_s": sh["duration_s"] or 4,
                                                "keyframes": kfs, "audio_path": first_wav},
                                      owner_kind="shot", owner_id=sh["id"])
                    m += 1
        self.info.setText(tr("fs.mt.dub_done2", "Doppiaggio {l}: {n} battute; {m} video in re-lip-sync (coda).")
                          .format(l=lang, n=n, m=m))

    def _second_look(self):
        """Moonshot — «il secondo sguardo»: critico post-visione sul piano del film."""
        from core import llm_service as L
        pid = self._get_project()
        if not pid or not db.query("scene", "project_id=?", (pid,)):
            QMessageBox.information(self, tr("fs.mt.secondlook", "Secondo sguardo"),
                                   tr("fs.mt.no_break", "Serve un breakdown (scene/inquadrature).")); return
        if not L.available():
            QMessageBox.warning(self, tr("fs.mt.secondlook", "Secondo sguardo"), tr("fs.llm.unavailable", "LLM non disponibile.")); return
        self.btn_secondlook.setEnabled(False); self.info.setText(tr("fs.working", "Generazione…"))
        from PySide6.QtCore import QThread, Signal

        class _W(QThread):
            done = Signal(dict)

            def run(self):
                from core.film_studio import supervisor
                try:
                    self.done.emit(supervisor.second_look(pid))
                except Exception:                            # noqa: BLE001
                    self.done.emit({"verdict": "", "notes": [], "strengths": [], "available": False})
        self._sl_w = _W(); self._sl_w.done.connect(self._on_second_look); self._sl_w.start()

    def _on_second_look(self, res: dict):
        self.btn_secondlook.setEnabled(True)
        if not res.get("available") or not (res.get("verdict") or res.get("notes")):
            self.info.setText(tr("fs.mt.secondlook_none", "Nessun rilievo dal critico (LLM non disponibile o piano vuoto).")); return
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QPlainTextEdit as _PT
        dlg = QDialog(self); dlg.setWindowTitle(tr("fs.mt.secondlook", "🔍 Secondo sguardo")); lay = QVBoxLayout(dlg)
        txt = _PT(); txt.setReadOnly(True); txt.setMinimumSize(560, 360)
        body = ["◆ " + tr("fs.mt.sl_verdict", "Giudizio") + ": " + (res.get("verdict") or ""), ""]
        if res.get("notes"):
            body.append(tr("fs.mt.sl_notes", "Rilievi:"))
            for n in res["notes"]:
                body.append(f"  ⚠ [{n.get('where', '')}] {n.get('note', '')}")
            body.append("")
        if res.get("strengths"):
            body.append(tr("fs.mt.sl_strengths", "Punti di forza:"))
            for s in res["strengths"]:
                body.append(f"  ✓ {s}")
        txt.setPlainText("\n".join(body))
        lay.addWidget(txt)
        bb = QDialogButtonBox(QDialogButtonBox.Close); bb.rejected.connect(dlg.reject); bb.accepted.connect(dlg.accept)
        lay.addWidget(bb); dlg.exec()
        self.info.setText(tr("fs.mt.secondlook_done", "🔍 Secondo sguardo completato."))

    def _rhythm(self):
        """Moonshot — montaggio a ritmo emotivo: curva di tensione per scena → rimodula le durate degli shot."""
        pid = self._get_project()
        scenes = db.query("scene", "project_id=?", (pid,), order="ord") if pid else []
        if not scenes:
            QMessageBox.information(self, tr("fs.mt.rhythm", "Ritmo"), tr("fs.mt.no_break", "Serve un breakdown (scene/inquadrature).")); return
        from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QSlider, QVBoxLayout, QLabel as _L)
        from PySide6.QtCore import Qt as _Qt
        dlg = QDialog(self); dlg.setWindowTitle(tr("fs.mt.rhythm", "📉 Ritmo emotivo")); outer = QVBoxLayout(dlg)
        outer.addWidget(_L(tr("fs.mt.rhythm_head", "Tensione per scena (0 = respiro/lenta, 100 = incalzante):")))
        form = QFormLayout(); sliders = []
        for sc in scenes:
            s = QSlider(_Qt.Horizontal); s.setRange(0, 100); s.setValue(50); s.setMinimumWidth(240)
            form.addRow(f"S{sc['ord']}: {(sc.get('description') or '')[:36]}", s)
            sliders.append(s)
        outer.addLayout(form)
        bb = QDialogButtonBox(); bb.addButton(tr("fs.mt.rhythm_apply", "Anteprima + applica"), QDialogButtonBox.AcceptRole)
        bb.addButton(QDialogButtonBox.Cancel); bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        outer.addWidget(bb)
        if dlg.exec() != QDialog.Accepted:
            return
        from core.film_studio import supervisor
        tension = [s.value() / 100.0 for s in sliders]
        changes = supervisor.emotional_rhythm(pid, tension)
        if not changes:
            self.info.setText(tr("fs.mt.rhythm_none", "Nessuna modifica di durata.")); return
        prev = "\n".join(f"  {c['ord']}: {c['old']:.1f}s → {c['new']:.1f}s"
                         for c in changes if abs(c['new'] - c['old']) >= 0.1)[:1200]
        if QMessageBox.question(self, tr("fs.mt.rhythm", "Ritmo emotivo"),
                                tr("fs.mt.rhythm_confirm", "Applico queste durate?\n\n{p}").format(p=prev or "—")
                                ) != QMessageBox.Yes:
            return
        n = supervisor.apply_rhythm(changes)
        self._refresh()
        self.info.setText(tr("fs.mt.rhythm_done", "📉 Ritmo emotivo: {n} durate aggiornate.").format(n=n))

    def _clips(self) -> list[dict]:
        pid = self._get_project()
        if not pid:
            return []
        out = []
        for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
            for sh in db.query("shot", "scene_id=?", (sc["id"],), order="ord"):
                f = str(db.project_dir(pid) / f"scene_{sc['ord']}" / f"shot_{sh['ord']}.mp4")
                if Path(f).exists():
                    ins, outs = float(sh.get("in_s", 0) or 0), float(sh.get("out_s", 0) or 0)  # trim per-shot
                    if not (outs > ins):
                        ins, outs = 0.0, float(sh["duration_s"] or 4)
                    out.append({"file": f, "in_s": ins, "out_s": outs,
                                "name": f"S{sc['ord']}.{sh['ord']}", "shot_id": sh["id"], "sh_ord": sh["ord"]})
        return out

    def _refresh(self):
        self.lst.clear()
        self._cur_clips = self._clips()
        for c in self._cur_clips:
            self.lst.addItem(f"{c['name']}  ({c['out_s']:.0f}s)  {Path(c['file']).name}")
        # #53 popola la timeline visuale (clip video + tracce audio già generate)
        audio = []
        for tr_name, src in [("dialoghi", getattr(self, "_audio_tracks", [])),
                             ("musica", getattr(self, "_extra_tracks", [])),
                             ("ambient", getattr(self, "_ambient_tracks", []))]:
            for a in src:
                audio.append({"track": tr_name, "start_s": a.get("start_s", 0),
                              "dur_s": a.get("dur_s", 4), "label": tr_name})
        self.timeline.set_data([{"id": c["shot_id"], "label": c["name"], "duration_s": c["out_s"]}
                                for c in self._cur_clips], audio)
        self.info.setText(tr("fs.mt.count", "{n} clip in timeline").format(n=len(self._cur_clips)))

    def _on_reorder(self, shot_ids: list):
        """#53 Drop sulla timeline → riscrive l'ord degli shot secondo il nuovo ordine visuale."""
        for new_ord, sid in enumerate(shot_ids, start=1):
            try:
                db.update("shot", int(sid), ord=new_ord)
            except Exception:                               # noqa: BLE001
                pass
        self._refresh()
        self.info.setText(tr("fs.mt.reordered", "Ordine clip aggiornato dalla timeline."))

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
        _overflow = 0
        for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
            for sh in db.query("shot", "scene_id=?", (sc["id"],), order="ord"):
                dur = float(sh["duration_s"] or 4)
                local = 0.0                                   # B3: le battute dello shot si SUSSEGUONO (prima
                for di, d in enumerate(json.loads(sh.get("dialogues") or "[]")):  # partivano tutte allo stesso t)
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
                        self._audio_tracks.append({"path": out, "start_s": t + local, "volume": 1.0})
                        local += _wav_dur(out) + 0.15         # durata reale + pausa tra battute
                        n += 1
                    except Exception:                       # noqa: BLE001
                        pass
                if local > dur + 0.1:                          # le battute eccedono la durata dello shot
                    _overflow += 1
                t += dur
        _msg = tr("fs.mt.audio_done", "Audio: {n} battute sintetizzate").format(n=n)
        if _overflow:
            _msg += " · " + tr("fs.mt.audio_overflow", "⚠ {k} shot: dialoghi più lunghi della durata (allunga l'inquadratura o taglia il testo)").format(k=_overflow)
        self.info.setText(_msg)

    def _film_mood(self) -> str:
        import json
        pid = self._get_project()
        moods = []
        for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
            st = json.loads(sc.get("continuity") or "{}")
            if st.get("mood"):
                moods.append(st["mood"])
        return ", ".join(dict.fromkeys(moods)) or "cinematic, atmospheric"

    def _gen_music(self, then=None):
        """Colonna sonora strumentale (HeartMuLa) col mood del film → traccia musica (volume ridotto = ducking).
        `then`: callback opzionale eseguito DOPO che la musica è pronta (usato dall'Auto per il master)."""
        from core import music_service as ms
        pid = self._get_project()
        if not pid:
            if then:
                then()                                       # niente progetto: non bloccare la catena Auto
            return
        total = sum(float(c["out_s"]) for c in (self._cur_clips or [])) or 60.0
        dest = str(db.project_dir(pid) / "music.mp3")        # #95: copia nella cartella progetto (self-contained)
        p = ms.SongParams(name=f"fs_music_{pid}", lyrics="", tags=f"{self._film_mood()}, instrumental, soundtrack",
                          duration_s=min(total, 180), instrumental=True)
        errs = ms.preflight_song(p)
        if errs:
            QMessageBox.warning(self, tr("fs.mt.music", "Musica"), "; ".join(errs))
            if then:
                then()
            return
        self.btn_music.setEnabled(False); self.info.setText(tr("fs.working", "Generazione…"))
        self._mjob = ms.make_song_job(p, on_done=lambda rc: self._on_music(rc, p.out_mp3(), total, dest, then))
        self._mjob.start() if hasattr(self._mjob, "start") else None

    def _on_music(self, rc, path, total, dest=None, then=None):
        self.btn_music.setEnabled(True)
        if os.path.exists(str(path)):
            use = str(path)
            if dest and str(dest) != str(path):              # #95: copia nella cartella progetto
                try:
                    import shutil
                    shutil.copy2(str(path), str(dest)); use = str(dest)
                except Exception:                            # noqa: BLE001
                    pass
            self._extra_tracks = getattr(self, "_extra_tracks", [])
            self._extra_tracks.append({"path": use, "start_s": 0, "volume": 0.3})  # ducking sotto i dialoghi
            self.info.setText(tr("fs.mt.music_done", "Colonna sonora pronta."))
        else:
            self.info.setText(tr("fs.mt.music_fail", "Colonna sonora non generata."))
        if then:                                             # #95 Auto: il master parte SOLO ora (musica pronta)
            then()

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
            self._ambient_tracks = getattr(self, "_ambient_tracks", [])   # #95: corsia ambient dedicata (era in "musica")
            self._ambient_tracks.append({"path": str(f), "start_s": 0, "volume": 0.4})
            self.info.setText(tr("fs.mt.ambient_done", "Ambient pronto."))
        else:
            self.info.setText(tr("fs.mt.ambient_fail", "Ambient non generato."))

    def _gen_sfx(self):
        """#AREA1 SFX puntuali per-azione (§11.2): MMAudio condizionato sul VIDEO di OGNI shot →
        effetti sincroni all'azione, posizionati al tempo dello shot (in sequenza)."""
        pid = self._get_project()
        if not pid:
            return
        self._sfx_list = []   # (video, offset_s)
        off = 0.0
        for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
            for sh in db.query("shot", "scene_id=?", (sc["id"],), order="ord"):
                f = str(db.project_dir(pid) / f"scene_{sc['ord']}" / f"shot_{sh['ord']}.mp4")
                if os.path.exists(f):
                    self._sfx_list.append((f, off))
                off += float(sh["duration_s"] or 4)
        if not self._sfx_list:
            QMessageBox.information(self, tr("fs.mt.sfx", "SFX"), tr("fs.mt.no_clips", "Nessuna clip generata.")); return
        self._sfx_i = 0; self._audio_tracks = getattr(self, "_audio_tracks", [])
        self.btn_sfx.setEnabled(False)
        self._sfx_next()

    def _sfx_next(self):
        from core import music_service as ms
        if self._sfx_i >= len(self._sfx_list):
            self.btn_sfx.setEnabled(True); self._refresh()
            self.info.setText(tr("fs.mt.sfx_done", "SFX per-azione: {n} effetti.").format(n=len(self._sfx_list))); return
        video, off = self._sfx_list[self._sfx_i]
        self.info.setText(tr("fs.mt.sfx_run", "SFX {i}/{n}…").format(i=self._sfx_i + 1, n=len(self._sfx_list)))
        p = ms.SonifyParams(name=f"fs_sfx_{self._get_project()}_{self._sfx_i}", video_path=video,
                            prompt=self._film_mood() + ", sound effects for the on-screen action, foley")
        self._sjob = ms.make_sonify_job(p, on_done=lambda rc, o=off: self._on_sfx(rc, o))
        self._sjob.start() if hasattr(self._sjob, "start") else self._on_sfx(0, off)

    def _on_sfx(self, rc, off):
        from core import music_service as ms
        f = ms.find_latest_audio()
        if f and os.path.exists(str(f)):
            self._audio_tracks.append({"path": str(f), "start_s": off, "volume": 0.5, "dur_s": 4})
        self._sfx_i += 1
        self._sfx_next()

    def _master(self):
        clips = [c["file"] for c in (self._cur_clips or [])]
        if not clips:
            QMessageBox.information(self, tr("fs.mt.master", "Master"), tr("fs.mt.no_clips", "Nessuna clip generata.")); return
        pid = self._get_project()
        out = str(db.project_dir(pid) / "master.mp4")
        tracks = (list(getattr(self, "_audio_tracks", []) or []) + list(getattr(self, "_extra_tracks", []) or [])
                  + list(getattr(self, "_ambient_tracks", []) or []))   # #95: includi l'ambient nel master
        default_t = self.trans.currentData() or ""          # transizione: per-shot (shot.transition) o default
        dur = self.trans_dur.value()
        per, trims = [], []
        for c in (self._cur_clips or []):
            sh = db.get("shot", c.get("shot_id")) or {}
            trims.append({"in_s": sh.get("in_s", 0) or 0, "out_s": sh.get("out_s", 0) or 0})  # trim per-shot
        for c in (self._cur_clips or [])[:-1]:
            sh = db.get("shot", c.get("shot_id")) or {}
            t = sh.get("transition") or default_t
            per.append({"type": t, "dur": dur} if t else {"type": "", "dur": 0})
        trans = per if any(p["type"] for p in per) else None
        trims = trims if any(t["out_s"] > t["in_s"] for t in trims) else None
        try:
            otio_export.render_master(clips, out, audio_tracks=tracks or None,
                                      aspect=self.aspect.currentData() or "",   # B7: aspect ora applicato
                                      transitions=trans, trims=trims)           # transizioni + trim per-shot
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
