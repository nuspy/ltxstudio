"""Tab Scenografia (spec §10): breakdown Scene→Shot via LLM (JSON), generazione keyframe + video
nativi, Continuity Engine. Versione funzionale: breakdown + generazione per scena via Job Scheduler."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
                               QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from core import llm_service as L
from core.film_studio import db, gen_service, continuity as ct, scheduler
from ui.i18n_manager import tr

_SYS = ("You are a film script supervisor. Break the screenplay into SCENES and SHOTS. "
        "Output STRICT JSON: {\"scenes\":[{\"description\":str,\"style\":str,\"duration_s\":number,"
        "\"continuity\":{\"time_of_day\":str,\"weather\":str,\"mood\":str},"
        "\"shots\":[{\"description\":str,\"duration_s\":number,\"camera\":str,"
        "\"dialogues\":[{\"actor\":str,\"text\":str}],\"keyframe\":str}]}]}. "
        "Sum of shot durations = scene duration; sum of scene durations = film duration. "
        "Keep @Name references. No prose outside JSON.")


class StoryboardTab(QWidget):
    def __init__(self, get_project):
        super().__init__()
        self._get_project = get_project
        v = QVBoxLayout(self)
        h = QHBoxLayout()
        self.btn_break = QPushButton(tr("fs.sb.breakdown", "🎬 Breakdown scene/inquadrature (LLM)"))
        self.btn_break.clicked.connect(self._breakdown)
        self.btn_keys = QPushButton(tr("fs.sb.keyframes", "🖼 Genera keyframe")); self.btn_keys.clicked.connect(lambda: self._gen("keyframe"))
        self.btn_video = QPushButton(tr("fs.sb.videos", "🎥 Genera video")); self.btn_video.clicked.connect(lambda: self._gen("video"))
        self.btn_auto = QPushButton(tr("fs.sb.auto", "🎬 Genera film completo")); self.btn_auto.clicked.connect(self._gen_all)
        h.addWidget(self.btn_break); h.addWidget(self.btn_keys); h.addWidget(self.btn_video); h.addWidget(self.btn_auto); h.addStretch(1)
        v.addLayout(h)
        self.tree = QTreeWidget(); self.tree.setHeaderLabels([tr("fs.sb.col", "Scene / Inquadrature"), tr("fs.sb.dur", "Durata")])
        self.tree.currentItemChanged.connect(self._on_tree_sel)
        v.addWidget(self.tree, 1)
        self.dur_warn = QLabel(""); v.addWidget(self.dur_warn)   # validazione durate (§10.1)
        # pannello editing scena/shot selezionato (spec §10.3) con @Reference
        eb = QGroupBox(tr("fs.sb.edit", "Modifica selezione (@Nome per gli asset)")); ev = QVBoxLayout(eb)
        self.edit = QPlainTextEdit(); self.edit.setFixedHeight(80)
        ev.addWidget(self.edit)
        from ui.film_studio.asset_dialog import attach as _attach_at
        _attach_at(self.edit)
        eh = QHBoxLayout()
        self.btn_save_sel = QPushButton(tr("fs.save", "💾 Salva")); self.btn_save_sel.clicked.connect(self._save_sel)
        self.btn_regen_sel = QPushButton(tr("fs.sb.regen_seg", "↻ Rigenera (LLM)")); self.btn_regen_sel.clicked.connect(self._regen_sel)
        self.btn_regen_kf = QPushButton(tr("fs.sb.regen_kf", "🖼 Rigenera keyframe")); self.btn_regen_kf.clicked.connect(lambda: self._regen_shot("keyframe"))
        self.btn_regen_vid = QPushButton(tr("fs.sb.regen_vid", "🎥 Rigenera video")); self.btn_regen_vid.clicked.connect(lambda: self._regen_shot("video"))
        eh.addWidget(self.btn_save_sel); eh.addWidget(self.btn_regen_sel)
        eh.addWidget(self.btn_regen_kf); eh.addWidget(self.btn_regen_vid); eh.addStretch(1)
        ev.addLayout(eh)
        v.addWidget(eb)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(80)
        v.addWidget(self.log)
        self._sel = None
        scheduler.on_event(self._on_job)
        self._refresh()

    def _refresh(self):
        self.tree.clear()
        pid = self._get_project()
        if not pid:
            return
        proj = db.get("project", pid) or {}
        scenes = db.query("scene", "project_id=?", (pid,), order="ord")
        warns = []
        scene_sum = 0.0
        for sc in scenes:
            shots = db.query("shot", "scene_id=?", (sc["id"],), order="ord")
            shot_sum = sum(float(s["duration_s"] or 0) for s in shots)
            scene_sum += float(sc["duration_s"] or 0)
            mark = "" if abs(shot_sum - float(sc["duration_s"] or 0)) < 0.6 else " ⚠"
            it = QTreeWidgetItem([f"S{sc['ord']}: {(sc['description'] or '')[:58]}",
                                  f"{sc['duration_s']:.0f}s{mark}"])
            it.setData(0, Qt.UserRole, ("scene", sc["id"]))
            if mark:
                warns.append(tr("fs.sb.warn_scene", "Scena {n}: Σinquadrature {a:.0f}s ≠ durata scena {b:.0f}s")
                             .format(n=sc["ord"], a=shot_sum, b=float(sc["duration_s"] or 0)))
            for sh in shots:
                ci = QTreeWidgetItem([f"  {sh['ord']}. {(sh['description'] or '')[:58]}", f"{sh['duration_s']:.0f}s"])
                ci.setData(0, Qt.UserRole, ("shot", sh["id"]))
                it.addChild(ci)
            self.tree.addTopLevelItem(it); it.setExpanded(True)
        film = float(proj.get("duration_s") or 0)
        if scenes and film and abs(scene_sum - film) > 1.0:
            warns.append(tr("fs.sb.warn_film", "Σscene {a:.0f}s ≠ durata film {b:.0f}s").format(a=scene_sum, b=film))
        if hasattr(self, "dur_warn"):
            self.dur_warn.setText(("⚠ " + " · ".join(warns)) if warns else
                                  tr("fs.sb.dur_ok", "✓ Durate coerenti"))

    def _breakdown(self):
        pid = self._get_project()
        if not pid:
            QMessageBox.information(self, tr("fs.sb.breakdown", "Breakdown"), tr("fs.no_project", "Crea/seleziona un progetto.")); return
        scripts = db.query("script", "project_id=?", (pid,), order="version")
        if not scripts:
            QMessageBox.information(self, tr("fs.sb.breakdown", "Breakdown"), tr("fs.sb.no_script", "Genera prima il copione.")); return
        proj = db.get("project", pid)
        try:
            data = L.complete_json(_SYS, f"Film duration: {proj.get('duration_s',60)}s.\n\n" + scripts[-1]["text"],
                                   max_tokens=3000)
        except Exception as e:                              # noqa: BLE001
            QMessageBox.critical(self, tr("fs.sb.breakdown", "Breakdown"), str(e)[:600]); return
        # pulisci scene/shot precedenti del progetto
        for sc in db.query("scene", "project_id=?", (pid,)):
            for sh in db.query("shot", "scene_id=?", (sc["id"],)):
                db.delete("shot", sh["id"])
            db.delete("scene", sc["id"])
        prev_state = None
        for i, sc in enumerate(data.get("scenes", []), 1):
            state = ct.propagate(prev_state, sc.get("continuity"))
            prev_state = state
            sid = db.insert("scene", project_id=pid, ord=i, description=sc.get("description", ""),
                            style=sc.get("style", ""), duration_s=float(sc.get("duration_s", 0) or 0),
                            continuity=json.dumps(state, ensure_ascii=False), assets="")
            for j, sh in enumerate(sc.get("shots", []), 1):
                db.insert("shot", scene_id=sid, ord=j, description=sh.get("description", ""),
                          duration_s=float(sh.get("duration_s", 0) or 0),
                          dialogues=json.dumps(sh.get("dialogues", []), ensure_ascii=False),
                          keyframes=json.dumps([sh.get("keyframe", "")], ensure_ascii=False),
                          refs="", camera=sh.get("camera", ""), continuity=json.dumps(state, ensure_ascii=False))
        self._refresh()
        self.log.appendPlainText(tr("fs.sb.done", "Breakdown: {n} scene").format(n=len(data.get('scenes', []))))

    def _gen(self, what: str):
        pid = self._get_project()
        if not pid:
            return
        proj = db.get("project", pid)
        pmode = proj.get("path_mode", "fast")
        n = 0
        for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
            state = json.loads(sc.get("continuity") or "{}")
            for sh in db.query("shot", "scene_id=?", (sc["id"],), order="ord"):
                refs = gen_service.resolve_refs(sh["description"], pmode)
                prompt = refs["clean_text"] + ". " + ct.to_prompt(state)
                outdir = str(db.project_dir(pid) / f"scene_{sc['ord']}")
                Path(outdir).mkdir(parents=True, exist_ok=True)
                if what == "keyframe":
                    out = str(Path(outdir) / f"shot_{sh['ord']}_kf.png")
                    scheduler.enqueue("image", {"prompt": prompt, "out": out,
                                                "face_ref": (refs["ref_images"] or [""])[0],
                                                "loras": refs["loras"]}, owner_kind="shot", owner_id=sh["id"])
                else:
                    out = str(Path(outdir) / f"shot_{sh['ord']}.mp4")
                    kf = str(Path(outdir) / f"shot_{sh['ord']}_kf.png")
                    kfs = [{"path": kf, "frame": 0, "strength": 1.0}] if Path(kf).exists() else []
                    # LIP-SYNC: se lo shot ha un dialogo, sintetizza l'audio con la voce legata e
                    # passalo al video (a2v nativo = labiale sincronizzato sul parlato)
                    audio_path = self._shot_dialogue_audio(sh, outdir)
                    scheduler.enqueue("video", {"prompt": prompt, "out": out, "duration_s": sh["duration_s"] or 4,
                                                "keyframes": kfs, "loras": refs["loras"], "audio_path": audio_path},
                                      owner_kind="shot", owner_id=sh["id"])
                n += 1
        self.log.appendPlainText(tr("fs.sb.queued", "{n} job '{w}' in coda").format(n=n, w=what))

    def _gen_all(self):
        """Modalità Auto (spec §5.6): breakdown (se serve) → keyframe → video in coda. Il master si fa
        nel tab Montaggio (un click dopo che la coda è finita). Stima costo/tempo prima di lanciare."""
        pid = self._get_project()
        if not pid:
            QMessageBox.information(self, tr("fs.sb.auto", "Auto"), tr("fs.no_project", "Crea/seleziona un progetto.")); return
        if not db.query("scene", "project_id=?", (pid,)):
            self._breakdown()
        shots = sum(len(db.query("shot", "scene_id=?", (sc["id"],))) for sc in db.query("scene", "project_id=?", (pid,)))
        est = scheduler.estimate([{"type": "image"}] * shots + [{"type": "video"}] * shots)
        if QMessageBox.question(self, tr("fs.sb.auto", "Genera film completo"),
                                tr("fs.sb.auto_confirm", "Genererò {s} shot (keyframe+video). Stima ~{m} min. Procedere?")
                                .format(s=shots, m=round(est["seconds"] / 60))) != QMessageBox.Yes:
            return
        self._gen("keyframe")
        self._gen("video")
        self.log.appendPlainText(tr("fs.sb.auto_started", "Film completo in coda: keyframe+video. Poi vai a Montaggio → Master."))

    def _regen_shot(self, what: str):
        """1.2 Rigenera SOLO lo shot selezionato (keyframe o video) accodandolo al Job Scheduler."""
        if not self._sel or self._sel[0] != "shot":
            QMessageBox.information(self, tr("fs.sb.regen_kf", "Rigenera"), tr("fs.sb.sel_shot", "Seleziona un'inquadratura.")); return
        sh = db.get("shot", self._sel[1])
        sc = db.get("scene", sh["scene_id"])
        pid = self._get_project()
        proj = db.get("project", pid)
        state = json.loads(sc.get("continuity") or "{}")
        refs = gen_service.resolve_refs(sh["description"], proj.get("path_mode", "fast"))
        prompt = refs["clean_text"] + ". " + ct.to_prompt(state)
        outdir = str(db.project_dir(pid) / f"scene_{sc['ord']}")
        Path(outdir).mkdir(parents=True, exist_ok=True)
        if what == "keyframe":
            out = str(Path(outdir) / f"shot_{sh['ord']}_kf.png")
            scheduler.enqueue("image", {"prompt": prompt, "out": out,
                                        "face_ref": (refs["ref_images"] or [""])[0], "loras": refs["loras"]},
                              owner_kind="shot", owner_id=sh["id"])
        else:
            out = str(Path(outdir) / f"shot_{sh['ord']}.mp4")
            kf = str(Path(outdir) / f"shot_{sh['ord']}_kf.png")
            kfs = [{"path": kf, "frame": 0, "strength": 1.0}] if Path(kf).exists() else []
            scheduler.enqueue("video", {"prompt": prompt, "out": out, "duration_s": sh["duration_s"] or 4,
                                        "keyframes": kfs, "loras": refs["loras"],
                                        "audio_path": self._shot_dialogue_audio(sh, outdir)},
                              owner_kind="shot", owner_id=sh["id"])
        self.log.appendPlainText(tr("fs.sb.regen_queued", "Rigenerazione {w} in coda per lo shot.").format(w=what))

    def _shot_dialogue_audio(self, sh: dict, outdir: str) -> str:
        """Sintetizza il PRIMO dialogo dello shot con la voce legata all'attore parlante → wav per a2v lip-sync."""
        try:
            from core.film_studio import voice_studio as vs
            dlgs = json.loads(sh.get("dialogues") or "[]")
            for d in dlgs:
                actor = (d.get("actor") or "").lstrip("@"); text = d.get("text") or ""
                if not text:
                    continue
                assets = [a for a in db.find_assets(actor, "actor") if a["name"].lower() == actor.lower()]
                vrow = vs.voice_for_actor(assets[0]["id"]) if assets else None
                if not vrow:
                    continue
                wav = str(Path(outdir) / f"shot_{sh['ord']}_dlg.wav")
                vs.synth_dialogue(text, vrow, wav)
                return wav
        except Exception:                                   # noqa: BLE001
            pass
        return ""

    def _on_tree_sel(self, cur, _prev):
        self._sel = cur.data(0, Qt.UserRole) if cur else None
        if not self._sel:
            self.edit.clear(); return
        kind, rid = self._sel
        row = db.get(kind, rid)
        self.edit.setPlainText((row or {}).get("description", "") if row else "")

    def _save_sel(self):
        if not self._sel:
            return
        kind, rid = self._sel
        db.update(kind, rid, description=self.edit.toPlainText().strip())
        # propagazione: se cambio la descrizione di scena, gli shot ereditano i nuovi @ref al refresh gen
        self._refresh()
        self.log.appendPlainText(tr("fs.sb.saved_sel", "Selezione salvata."))

    def _regen_sel(self):
        if not self._sel:
            return
        try:
            kind, rid = self._sel
            sysp = ("Rewrite this film " + ("scene" if kind == "scene" else "shot") +
                    " description, keeping @Name references. Same language. Output only the text.")
            new = L.complete(sysp, self.edit.toPlainText().strip(), max_tokens=400)
            self.edit.setPlainText(new.strip())
        except Exception as e:                              # noqa: BLE001
            QMessageBox.critical(self, tr("fs.sb.edit", "Modifica"), str(e)[:400])

    def _on_job(self, ev: dict):
        if ev.get("event") in ("done", "error"):
            self.log.appendPlainText(f"[{ev['event']}] job {ev.get('job_id')} {ev.get('type')} "
                                     f"{ev.get('error','') or ev.get('output','')}"[:120])
