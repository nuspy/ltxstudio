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

# Schema INTERNO (invisibile all'utente): shot arricchito con camera+lighting per prompt di generazione più
# coerenti e cinematografici. camera = inquadratura+angolo (es. "wide shot, low angle"); lighting = luce/mood.
_SYS = ("You are a film script supervisor. Break the screenplay into SCENES and SHOTS. "
        "Output STRICT JSON: {\"scenes\":[{\"description\":str,\"style\":str,\"duration_s\":number,"
        "\"continuity\":{\"time_of_day\":str,\"weather\":str,\"mood\":str},"
        "\"shots\":[{\"description\":str,\"duration_s\":number,"
        "\"camera\":str,\"lighting\":str,"
        "\"dialogues\":[{\"actor\":str,\"text\":str}],\"keyframe\":str}]}]}. "
        "camera = concrete shot type + angle (e.g. 'wide shot, low angle', 'close-up'). "
        "lighting = light + mood (e.g. 'soft golden hour', 'harsh neon'). "
        "description = the visual content of the shot (subject + action + setting). "
        "Sum of shot durations = scene duration; sum of scene durations = film duration. "
        "Keep @Name references. No prose outside JSON.")


class StoryboardTab(QWidget):
    auto_complete = Signal(int)        # #54: project_id quando l'Auto ha finito i video → montaggio
    _job_event = Signal(dict)          # #95: marshalling eventi scheduler (worker thread) → thread UI

    def __init__(self, get_project):
        super().__init__()
        self._get_project = get_project
        self._auto = None
        v = QVBoxLayout(self)
        h = QHBoxLayout()
        self.btn_break = QPushButton(tr("fs.sb.breakdown", "🎬 Breakdown scene/inquadrature (LLM)"))
        self.btn_break.clicked.connect(self._breakdown)
        self.btn_keys = QPushButton(tr("fs.sb.keyframes", "🖼 Genera keyframe")); self.btn_keys.clicked.connect(lambda: self._gen("keyframe"))
        self.btn_video = QPushButton(tr("fs.sb.videos", "🎥 Genera video")); self.btn_video.clicked.connect(lambda: self._gen("video"))
        self.btn_auto = QPushButton(tr("fs.sb.auto", "🎬 Genera film completo")); self.btn_auto.clicked.connect(self._gen_all)
        self.btn_shotlist = QPushButton(tr("fs.sb.shotlist", "📄 Shot-list")); self.btn_shotlist.clicked.connect(self._export_shotlist)
        h.addWidget(self.btn_break); h.addWidget(self.btn_keys); h.addWidget(self.btn_video)
        h.addWidget(self.btn_auto); h.addWidget(self.btn_shotlist); h.addStretch(1)
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
        from PySide6.QtWidgets import QSpinBox as _SB
        sh_seed = QHBoxLayout()
        sh_seed.addWidget(QLabel(tr("fs.sb.seed", "Seed (0=auto/blocca)")))
        self.shot_seed = _SB(); self.shot_seed.setRange(0, 2**31 - 1)
        self.shot_seed.editingFinished.connect(self._save_seed)
        sh_seed.addWidget(self.shot_seed); sh_seed.addStretch(1)
        ev.addLayout(sh_seed)
        # #56 Continuity State (editabile quando selezioni una SCENA): orario/meteo/luce/mood
        from PySide6.QtWidgets import QComboBox as _CB
        self.ct_box = QGroupBox(tr("fs.sb.continuity", "Continuity scena")); cg = QHBoxLayout(self.ct_box)
        self.ct_time = _CB(); self.ct_time.addItems(["day", "night", "dawn", "dusk", "golden hour"])
        self.ct_weather = _CB(); self.ct_weather.addItems(["clear", "cloudy", "rain", "snow", "fog", "storm"])
        self.ct_light = _CB(); self.ct_light.addItems(["natural soft", "hard sun", "low key", "high key", "candlelight", "neon"])
        self.ct_mood = _CB(); self.ct_mood.addItems(["neutral", "tense", "joyful", "melancholic", "epic", "romantic", "scary"])
        for lbl, w in [("⏱", self.ct_time), ("☁", self.ct_weather), ("💡", self.ct_light), ("🎭", self.ct_mood)]:
            cg.addWidget(QLabel(lbl)); cg.addWidget(w)
        self.btn_ct = QPushButton(tr("fs.sb.ct_apply", "Applica + propaga")); self.btn_ct.clicked.connect(self._save_continuity)
        cg.addWidget(self.btn_ct); cg.addStretch(1)
        self.ct_box.setVisible(False)
        ev.addWidget(self.ct_box)
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
        # lo scheduler emette da un thread worker → marshalla su UI via segnale (connessione queued auto)
        self._job_event.connect(self._on_job)
        scheduler.on_event(lambda ev: self._job_event.emit(dict(ev)))
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
                cam = ", ".join(x for x in (sh.get("camera", ""), sh.get("lighting", "")) if x)  # camera+luce
                db.insert("shot", scene_id=sid, ord=j, description=sh.get("description", ""),
                          duration_s=float(sh.get("duration_s", 0) or 0),
                          dialogues=json.dumps(sh.get("dialogues", []), ensure_ascii=False),
                          keyframes=json.dumps([sh.get("keyframe", "")], ensure_ascii=False),
                          refs="", camera=cam, continuity=json.dumps(state, ensure_ascii=False))
        self._refresh()
        self.log.appendPlainText(tr("fs.sb.done", "Breakdown: {n} scene").format(n=len(data.get('scenes', []))))

    def _gen(self, what: str):
        pid = self._get_project()
        if not pid:
            return
        proj = db.get("project", pid)
        pmode = proj.get("path_mode", "fast")
        guard = json.loads(proj.get("config") or "{}")
        forbidden, required = guard.get("forbidden", ""), guard.get("required", "")
        # B5: modello immagine/video scelto nel progetto (config), non più hardcoded flux
        img_arch = guard.get("img_arch", "flux")
        img_model = guard.get("img_model", "")
        vid_gguf = guard.get("vid_gguf", "")
        n = 0
        jids = []
        for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
            state = json.loads(sc.get("continuity") or "{}")
            for sh in db.query("shot", "scene_id=?", (sc["id"],), order="ord"):
                _flavor = "image" if what == "keyframe" else "video"   # B2: LoRA del flavor giusto
                refs = gen_service.resolve_refs(sh["description"], pmode, flavor=_flavor)
                # B4: seed per-shot — bloccato (>0) o stabile-per-shot (varia tra shot, riproducibile) invece
                # del default fisso 7 per tutti. Congela l'identità/inquadratura di QUESTO shot.
                _seed = int(sh.get("seed", 0) or 0) or (int(sh["id"]) * 2654435761 % (2**31 - 1))
                if len(refs.get("loras") or []) > 3:        # #55 warning composizione multi-LoRA (concept bleed)
                    self.log.appendPlainText(tr("fs.sb.too_many_lora",
                        "⚠ Scena {s} shot {h}: {n} LoRA simultanee (>3) — rischio interferenza, valuta split.")
                        .format(s=sc["ord"], h=sh["ord"], n=len(refs["loras"])))
                cam = (sh.get("camera") or "").strip()      # camera+luce dallo schema shot (prima ignorati)
                prompt = (refs["clean_text"] + ". " + (cam + ". " if cam else "")
                          + ct.to_prompt(state) + ((", " + required) if required else ""))
                outdir = str(db.project_dir(pid) / f"scene_{sc['ord']}")
                Path(outdir).mkdir(parents=True, exist_ok=True)
                if what == "keyframe":
                    out = str(Path(outdir) / f"shot_{sh['ord']}_kf.png")
                    jid = scheduler.enqueue("image", {"prompt": prompt, "out": out,
                                                "face_ref": (refs["ref_images"] or [""])[0],
                                                "ref_images": refs["ref_images"], "loras": refs["loras"],
                                                "arch": img_arch, "model": img_model,
                                                "seed": _seed, "negative": forbidden},
                                      owner_kind="shot", owner_id=sh["id"])
                else:
                    out = str(Path(outdir) / f"shot_{sh['ord']}.mp4")
                    kf = str(Path(outdir) / f"shot_{sh['ord']}_kf.png")
                    kfs = [{"path": kf, "frame": 0, "strength": 1.0}] if Path(kf).exists() else []
                    audio_path = self._shot_dialogue_audio(sh, outdir)
                    jid = scheduler.enqueue("video", {"prompt": prompt, "out": out, "duration_s": sh["duration_s"] or 4,
                                                "keyframes": kfs, "loras": refs["loras"], "audio_path": audio_path,
                                                "model_gguf": vid_gguf, "seed": _seed, "negative": forbidden},
                                      owner_kind="shot", owner_id=sh["id"])
                n += 1
                jids.append(jid)
        self.log.appendPlainText(tr("fs.sb.queued", "{n} job '{w}' in coda").format(n=n, w=what))
        return jids

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
        # ORCHESTRAZIONE A FASI: prima i keyframe; quando TUTTI pronti, i video (così il kf esiste);
        # quando tutti i video pronti, emette auto_complete → la finestra fa audio+master.
        kf_jids = [j for j in self._gen("keyframe") if j]
        self._auto = {"phase": "keyframe", "pending": set(kf_jids), "pid": pid}
        self.log.appendPlainText(tr("fs.sb.auto_started",
            "Film completo (Auto): keyframe → video → montaggio automatici."))

    def _export_shotlist(self):
        """1.6 Esporta la shot-list come documento HTML (scene/shot/durate/dialoghi/keyframe)."""
        pid = self._get_project()
        if not pid:
            return
        proj = db.get("project", pid)
        rows = ["<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;margin:24px}"
                "h2{border-bottom:2px solid #888}table{border-collapse:collapse;width:100%}"
                "td,th{border:1px solid #ccc;padding:6px;vertical-align:top}img{max-height:90px}</style></head><body>"]
        rows.append(f"<h1>Shot-list — {proj.get('name','')}</h1><p>Durata {proj.get('duration_s',0):.0f}s · "
                    f"stile {proj.get('styles','')} · generi {proj.get('genres','')}</p>")
        for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
            rows.append(f"<h2>Scena {sc['ord']} ({sc['duration_s']:.0f}s)</h2><p>{sc.get('description','')}</p>")
            rows.append("<table><tr><th>#</th><th>Durata</th><th>Descrizione</th><th>Dialoghi</th><th>Keyframe</th></tr>")
            for sh in db.query("shot", "scene_id=?", (sc["id"],), order="ord"):
                dlg = "; ".join(f"{d.get('actor','')}: {d.get('text','')}" for d in json.loads(sh.get("dialogues") or "[]"))
                kf = str(db.project_dir(pid) / f"scene_{sc['ord']}" / f"shot_{sh['ord']}_kf.png")
                img = f"<img src='file:///{kf.replace(chr(92),'/')}'>" if Path(kf).exists() else ""
                rows.append(f"<tr><td>{sh['ord']}</td><td>{sh['duration_s']:.0f}s</td>"
                            f"<td>{sh.get('description','')}</td><td>{dlg}</td><td>{img}</td></tr>")
            rows.append("</table>")
        rows.append("</body></html>")
        out = str(db.project_dir(pid) / "shotlist.html")
        open(out, "w", encoding="utf-8").write("\n".join(rows))
        self.log.appendPlainText(tr("fs.sb.shotlist_done", "Shot-list esportata: {f}").format(f=out))
        try:
            import os as _os
            _os.startfile(out)                              # noqa: S606
        except Exception:                                   # noqa: BLE001
            pass

    def _regen_shot(self, what: str):
        """1.2 Rigenera SOLO lo shot selezionato (keyframe o video) accodandolo al Job Scheduler."""
        if not self._sel or self._sel[0] != "shot":
            QMessageBox.information(self, tr("fs.sb.regen_kf", "Rigenera"), tr("fs.sb.sel_shot", "Seleziona un'inquadratura.")); return
        sh = db.get("shot", self._sel[1])
        sc = db.get("scene", sh["scene_id"])
        pid = self._get_project()
        proj = db.get("project", pid)
        state = json.loads(sc.get("continuity") or "{}")
        _cfg = json.loads(proj.get("config") or "{}")                    # B5: modello dal progetto
        _flavor = "image" if what == "keyframe" else "video"             # B2: LoRA del flavor giusto
        refs = gen_service.resolve_refs(sh["description"], proj.get("path_mode", "fast"), flavor=_flavor)
        prompt = refs["clean_text"] + ". " + ct.to_prompt(state)
        outdir = str(db.project_dir(pid) / f"scene_{sc['ord']}")
        Path(outdir).mkdir(parents=True, exist_ok=True)
        import random as _r
        seed = int(sh.get("seed", 0) or 0) or _r.randint(1, 2**31 - 1)   # 0 = auto; >0 = bloccato
        if what == "keyframe":
            out = str(Path(outdir) / f"shot_{sh['ord']}_kf.png")
            scheduler.enqueue("image", {"prompt": prompt, "out": out, "seed": seed,
                                        "face_ref": (refs["ref_images"] or [""])[0], "ref_images": refs["ref_images"],
                                        "loras": refs["loras"], "arch": _cfg.get("img_arch", "flux"),
                                        "model": _cfg.get("img_model", "")},
                              owner_kind="shot", owner_id=sh["id"])
        else:
            out = str(Path(outdir) / f"shot_{sh['ord']}.mp4")
            kf = str(Path(outdir) / f"shot_{sh['ord']}_kf.png")
            kfs = [{"path": kf, "frame": 0, "strength": 1.0}] if Path(kf).exists() else []
            scheduler.enqueue("video", {"prompt": prompt, "out": out, "duration_s": sh["duration_s"] or 4,
                                        "keyframes": kfs, "loras": refs["loras"], "seed": seed,
                                        "model_gguf": _cfg.get("vid_gguf", ""),
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
        if kind == "shot" and row:
            self.shot_seed.setValue(int(row.get("seed", 0) or 0))
        # #56 mostra/popola l'editor Continuity solo per le SCENE
        self.ct_box.setVisible(kind == "scene")
        if kind == "scene" and row:
            import json
            st = json.loads(row.get("continuity") or "{}")
            for w, k, d in [(self.ct_time, "time_of_day", "day"), (self.ct_weather, "weather", "clear"),
                            (self.ct_light, "light", "natural soft"), (self.ct_mood, "mood", "neutral")]:
                i = w.findText(str(st.get(k, d)))
                w.setCurrentIndex(max(0, i))

    def _save_seed(self):
        if self._sel and self._sel[0] == "shot":
            db.update("shot", self._sel[1], seed=int(self.shot_seed.value()))

    def _save_continuity(self):
        """#56 Salva il Continuity State della scena selezionata e lo PROPAGA alle scene successive."""
        if not self._sel or self._sel[0] != "scene":
            return
        import json
        from core.film_studio import continuity as ct
        rid = self._sel[1]
        sc = db.get("scene", rid)
        st = json.loads(sc.get("continuity") or "{}")
        st.update({"time_of_day": self.ct_time.currentText(), "weather": self.ct_weather.currentText(),
                   "light": self.ct_light.currentText(), "mood": self.ct_mood.currentText()})
        db.update("scene", rid, continuity=json.dumps(st, ensure_ascii=False))
        # propaga in avanti: ogni scena successiva eredita lo stato (override mantenuti dove già impostati)
        prev = st
        for nx in db.query("scene", "project_id=? AND ord>?", (sc["project_id"], sc["ord"]), order="ord"):
            nst = ct.propagate(prev, json.loads(nx.get("continuity") or "{}"))
            db.update("scene", nx["id"], continuity=json.dumps(nst, ensure_ascii=False))
            prev = nst
        self.log.appendPlainText(tr("fs.sb.ct_done", "Continuity applicata e propagata alle scene successive."))

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
            # #54 orchestrazione Auto a fasi: keyframe → video → montaggio
            auto = getattr(self, "_auto", None)
            if auto and ev.get("job_id") in auto["pending"]:
                auto["pending"].discard(ev["job_id"])
                if not auto["pending"]:
                    if auto["phase"] == "keyframe":
                        self._refresh()                      # i kf ora esistono → accoda i video
                        vids = [j for j in self._gen("video") if j]
                        auto["phase"] = "video"; auto["pending"] = set(vids)
                        self.log.appendPlainText(tr("fs.sb.auto_vid", "[Auto] keyframe pronti → video in coda."))
                    else:                                    # video finiti → montaggio automatico
                        pid = auto["pid"]; self._auto = None
                        self.log.appendPlainText(tr("fs.sb.auto_done", "[Auto] video pronti → montaggio automatico."))
                        self.auto_complete.emit(pid)
