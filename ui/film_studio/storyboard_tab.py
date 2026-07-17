"""Tab Scenografia (spec §10): breakdown Scene→Shot via LLM (JSON), generazione keyframe + video
nativi, Continuity Engine. Versione funzionale: breakdown + generazione per scena via Job Scheduler."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
                               QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from core import llm_service as L
from core.film_studio import db, gen_service, continuity as ct, scheduler
from core.film_studio.breakdown import SYS as _SYS   # system-prompt breakdown CONDIVISO (core, riuso Lucky Director)
from ui.i18n_manager import tr


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
        self.btn_bible = QPushButton(tr("fs.sb.bible", "🎨 Style Bible")); self.btn_bible.clicked.connect(self._edit_style_bible)
        h.addWidget(self.btn_break); h.addWidget(self.btn_keys); h.addWidget(self.btn_video)
        h.addWidget(self.btn_auto); h.addWidget(self.btn_shotlist); h.addWidget(self.btn_bible); h.addStretch(1)
        v.addLayout(h)
        # Moonshot — agenti autoriali (LLM): segretario di edizione + direttore della fotografia
        hm = QHBoxLayout()
        self.btn_supervisor = QPushButton(tr("fs.sb.supervisor", "🧐 Supervisore continuità"))
        self.btn_supervisor.setToolTip(tr("fs.sb.supervisor_tt", "Rilegge il breakdown e segnala i salti di "
                                          "raccordo (180°/eyeline, oggetti, ora del giorno, condizioni)."))
        self.btn_supervisor.clicked.connect(self._run_supervisor)
        self.btn_dop = QPushButton(tr("fs.sb.dop", "🎥 DoP: suggerisci ripresa"))
        self.btn_dop.setToolTip(tr("fs.sb.dop_tt", "Il direttore della fotografia propone camera+luce per gli "
                                   "shot della SCENA selezionata, dalla Style Bible. Applichi ciò che vuoi."))
        self.btn_dop.clicked.connect(self._run_dop)
        hm.addWidget(self.btn_supervisor); hm.addWidget(self.btn_dop); hm.addStretch(1)
        v.addLayout(hm)
        # B5: selettori modello immagine/video del progetto (salvati in config; niente più hardcode flux)
        mrow = QHBoxLayout()
        self.mdl_img = QComboBox(); self.mdl_vid = QComboBox()
        self.mdl_img.setMinimumWidth(220); self.mdl_vid.setMinimumWidth(220)
        try:
            from core import model_selection as _ms
            self.mdl_img.addItem(tr("fs.sb.mdl_default", "(auto)"), None)
            for lbl, path, arch in _ms.list_image_models():
                self.mdl_img.addItem(lbl, {"path": path, "arch": arch})
            self.mdl_vid.addItem(tr("fs.sb.mdl_default", "(auto)"), None)
            for lbl, path in _ms.list_video_gguf():
                self.mdl_vid.addItem(lbl, {"path": path})
        except Exception:                                    # noqa: BLE001
            pass
        # 2ª architettura IMMAGINE: FLUX (dropdown checkpoint sopra) vs Krea 2 (snapshot diffusers, senza file).
        # Se != FLUX vince sull'arch del checkpoint (img_model resta vuoto → risolto dal worker dedicato).
        self.img_arch = QComboBox(); self.img_arch.setMinimumWidth(120)
        self.img_arch.addItem(tr("fs.sb.arch_flux", "FLUX"), "flux")
        self.img_arch.addItem(tr("fs.sb.arch_krea2", "Krea 2"), "krea2")
        # 2ª architettura video: LTX-2 vs LingBot-Video (salvata nel config progetto → auto_run/handlers)
        self.vid_arch = QComboBox(); self.vid_arch.setMinimumWidth(140)
        self.vid_arch.addItem(tr("fs.sb.arch_ltx", "LTX-2"), "ltx")
        self.vid_arch.addItem(tr("fs.sb.arch_lingbot", "LingBot-Video"), "lingbot")
        self.vid_arch.addItem(tr("fs.sb.arch_krea_rt", "Krea Realtime"), "krea_rt")
        self.vid_arch.addItem(tr("fs.sb.arch_self_forcing", "Self-Forcing (realtime, lungo)"), "self_forcing")
        self.mdl_img.currentIndexChanged.connect(self._save_models)
        self.mdl_vid.currentIndexChanged.connect(self._save_models)
        self.img_arch.currentIndexChanged.connect(self._save_models)
        self.vid_arch.currentIndexChanged.connect(self._save_models)
        mrow.addWidget(QLabel(tr("fs.sb.mdl_img", "Modello immagine:"))); mrow.addWidget(self.mdl_img)
        mrow.addWidget(QLabel(tr("fs.sb.arch_img", "Arch. img:"))); mrow.addWidget(self.img_arch)
        mrow.addWidget(QLabel(tr("fs.sb.mdl_vid", "Modello video:"))); mrow.addWidget(self.mdl_vid)
        mrow.addWidget(QLabel(tr("fs.sb.arch_vid", "Arch. video:"))); mrow.addWidget(self.vid_arch)
        from PySide6.QtWidgets import QCheckBox as _CkB
        self.chk_chain = _CkB(tr("fs.sb.chain", "🔗 Raccordo continuo"))
        self.chk_chain.setToolTip(tr("fs.sb.chain_tt", "Usa l'ultimo frame dello shot precedente come aggancio "
                                     "del successivo (nella stessa scena) → tagli senza scarto brusco."))
        mrow.addWidget(self.chk_chain)
        mrow.addStretch(1)
        v.addLayout(mrow)
        self.tree = QTreeWidget(); self.tree.setHeaderLabels([tr("fs.sb.col", "Scene / Inquadrature"), tr("fs.sb.dur", "Durata")])
        from PySide6.QtCore import QSize as _QSize
        self.tree.setIconSize(_QSize(72, 40))                # thumbnail keyframe visibili (B11)
        self.tree.currentItemChanged.connect(self._on_tree_sel)
        v.addWidget(self.tree, 1)
        self.dur_warn = QLabel(""); v.addWidget(self.dur_warn)   # validazione durate (§10.1)
        # pannello editing scena/shot selezionato (spec §10.3) con @Reference
        eb = QGroupBox(tr("fs.sb.edit", "Modifica selezione (@Nome per gli asset)")); ev = QVBoxLayout(eb)
        self.edit = QPlainTextEdit(); self.edit.setFixedHeight(80)
        ev.addWidget(self.edit)
        from ui.film_studio.asset_dialog import attach as _attach_at
        _attach_at(self.edit)
        from ui.film_studio.regen_menu import attach_regen_menu
        attach_regen_menu(self.edit)                         # tasto destro su selezione → Rigenera segmento
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
        self.ct_lens = _CB(); self.ct_lens.addItems(["", "24mm wide", "35mm", "50mm", "85mm portrait", "anamorphic"])
        self.ct_film = _CB(); self.ct_film.addItems(["", "warm cinematic", "cool teal-orange", "bleach bypass", "vintage film grain", "high contrast noir", "pastel"])
        for lbl, w in [("⏱", self.ct_time), ("☁", self.ct_weather), ("💡", self.ct_light), ("🎭", self.ct_mood),
                       ("🎯", self.ct_lens), ("🎞", self.ct_film)]:
            cg.addWidget(QLabel(lbl)); cg.addWidget(w)
        self.btn_ct = QPushButton(tr("fs.sb.ct_apply", "Applica + propaga")); self.btn_ct.clicked.connect(self._save_continuity)
        self.btn_ct_chars = QPushButton(tr("fs.sb.ct_chars", "👤 Stato personaggi…")); self.btn_ct_chars.clicked.connect(self._edit_char_continuity)
        cg.addWidget(self.btn_ct); cg.addWidget(self.btn_ct_chars); cg.addStretch(1)
        self.ct_box.setVisible(False)
        ev.addWidget(self.ct_box)
        # Grammatica di ripresa per-shot (SHOT_TYPES/CAMERA_MOVES: prima costanti morte) → shot.camera
        from core.film_studio.style_catalog import SHOT_TYPES, CAMERA_MOVES
        self.cam_box = QGroupBox(tr("fs.sb.grammar", "Grammatica ripresa")); cgl = QHBoxLayout(self.cam_box)
        self.sh_type = _CB(); self.sh_type.addItem(tr("fs.sb.auto", "(auto)"), "")
        for x in SHOT_TYPES:
            self.sh_type.addItem(x, x)
        self.sh_move = _CB(); self.sh_move.addItem(tr("fs.sb.auto", "(auto)"), "")
        for x in CAMERA_MOVES:
            self.sh_move.addItem(x, x)
        self.sh_trans = _CB()                                # transizione verso lo shot successivo
        for code, lab in [("", tr("fs.sb.tr_cut", "stacco →")), ("fade", "dissolvenza →"),
                          ("fadeblack", "in nero →"), ("dissolve", "dissolve →"), ("wipeleft", "tendina →")]:
            self.sh_trans.addItem(lab, code)
        from PySide6.QtWidgets import QDoubleSpinBox as _DSB
        self.sh_in = _DSB(); self.sh_in.setRange(0, 3600); self.sh_in.setSingleStep(0.1); self.sh_in.setSuffix(" s")
        self.sh_out = _DSB(); self.sh_out.setRange(0, 3600); self.sh_out.setSingleStep(0.1); self.sh_out.setSuffix(" s")
        for lbl, w in [("🎬", self.sh_type), ("🎥", self.sh_move), ("↦", self.sh_trans),
                       ("⇥in", self.sh_in), ("out⇤", self.sh_out)]:
            cgl.addWidget(QLabel(lbl)); cgl.addWidget(w)
        cgl.addStretch(1)
        self.sh_type.currentIndexChanged.connect(self._save_shot_grammar)
        self.sh_move.currentIndexChanged.connect(self._save_shot_grammar)
        self.sh_trans.currentIndexChanged.connect(self._save_shot_grammar)
        self.sh_in.editingFinished.connect(self._save_shot_grammar)
        self.sh_out.editingFinished.connect(self._save_shot_grammar)
        self.cam_box.setVisible(False)
        ev.addWidget(self.cam_box)
        eh = QHBoxLayout()
        self.btn_save_sel = QPushButton(tr("fs.save", "💾 Salva")); self.btn_save_sel.clicked.connect(self._save_sel)
        self.btn_regen_sel = QPushButton(tr("fs.sb.regen_seg", "↻ Rigenera (LLM)")); self.btn_regen_sel.clicked.connect(self._regen_sel)
        self.btn_regen_kf = QPushButton(tr("fs.sb.regen_kf", "🖼 Rigenera keyframe")); self.btn_regen_kf.clicked.connect(lambda: self._regen_shot("keyframe"))
        self.btn_regen_vid = QPushButton(tr("fs.sb.regen_vid", "🎥 Rigenera video")); self.btn_regen_vid.clicked.connect(lambda: self._regen_shot("video"))
        self.btn_coverage = QPushButton(tr("fs.sb.coverage", "🎬 Espandi copertura dialoghi")); self.btn_coverage.clicked.connect(self._expand_coverage)
        self.btn_takes = QPushButton(tr("fs.sb.takes", "🎞 Storia take…")); self.btn_takes.clicked.connect(self._browse_takes)
        eh.addWidget(self.btn_save_sel); eh.addWidget(self.btn_regen_sel)
        eh.addWidget(self.btn_regen_kf); eh.addWidget(self.btn_regen_vid)
        eh.addWidget(self.btn_coverage); eh.addWidget(self.btn_takes); eh.addStretch(1)
        ev.addLayout(eh)
        # Strip-varianti INLINE: i take del keyframe dello shot selezionato come thumbnail cliccabili
        # (curatela senza dialog: clic = ripristina quel take come corrente, non-distruttivo).
        from PySide6.QtWidgets import QListWidget as _LW
        self.takes_lbl = QLabel(tr("fs.sb.variants", "Varianti (take) — clic per ripristinare:"))
        self.takes_strip = _LW(); self.takes_strip.setViewMode(_LW.IconMode); self.takes_strip.setFlow(_LW.LeftToRight)
        self.takes_strip.setWrapping(False); self.takes_strip.setFixedHeight(92)
        from PySide6.QtCore import QSize as _QSz
        self.takes_strip.setIconSize(_QSz(120, 68)); self.takes_strip.setMovement(_LW.Static)
        self.takes_strip.itemClicked.connect(self._restore_take_inline)
        self.takes_lbl.setVisible(False); self.takes_strip.setVisible(False)
        ev.addWidget(self.takes_lbl); ev.addWidget(self.takes_strip)
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
        try:
            self._load_models()                              # B5: riflette il modello salvato nei dropdown
        except Exception:                                    # noqa: BLE001
            pass
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
                _kf = db.project_dir(pid) / f"scene_{sc['ord']}" / f"shot_{sh['ord']}_kf.png"
                if _kf.exists():                             # B11: il keyframe FINALMENTE visibile nel tree
                    from PySide6.QtGui import QIcon
                    ci.setIcon(0, QIcon(str(_kf)))
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
        copione = db.latest_script(pid, "script") or db.latest_script(pid, "narrative")  # copione, poi Racconto
        if not copione:
            QMessageBox.information(self, tr("fs.sb.breakdown", "Breakdown"), tr("fs.sb.no_script", "Genera prima il copione.")); return
        proj = db.get("project", pid)
        racconto = db.latest_script(pid, "narrative")        # se c'è, il Racconto contestualizza il breakdown
        ctx = ("Cinematic treatment:\n" + racconto["text"] + "\n\n") if racconto else ""
        try:
            data = L.complete_json(_SYS, f"Film duration: {proj.get('duration_s',60)}s.\n\n{ctx}" + copione["text"],
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

    def _style_suffix(self, cfg: dict) -> str:
        """Style Bible → coda del prompt. Delega alla versione CONDIVISA (gen_service.style_suffix)."""
        return gen_service.style_suffix(cfg)

    def _edit_style_bible(self):
        """Editor Style Bible (logline/palette/look/registi) salvato in project.config."""
        pid = self._get_project()
        if not pid:
            QMessageBox.information(self, tr("fs.sb.bible", "Style Bible"), tr("fs.no_project", "Crea/seleziona un progetto.")); return
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLineEdit as _LE
        cfg = json.loads((db.get("project", pid) or {}).get("config") or "{}")
        sb = cfg.get("style_bible") or {}
        dlg = QDialog(self); dlg.setWindowTitle(tr("fs.sb.bible", "🎨 Style Bible")); f = QFormLayout(dlg)
        e_log = _LE(sb.get("logline", "")); e_pal = _LE(sb.get("palette", ""))
        e_look = _LE(sb.get("look", "")); e_dir = _LE(sb.get("directors", ""))
        f.addRow(tr("fs.sb.bible_logline", "Logline emotiva"), e_log)
        f.addRow(tr("fs.sb.bible_palette", "Palette colori"), e_pal)
        f.addRow(tr("fs.sb.bible_look", "Look (grana/ottica/luce)"), e_look)
        f.addRow(tr("fs.sb.bible_dir", "Registi/DoP di riferimento"), e_dir)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); f.addRow(bb)
        if dlg.exec() == QDialog.Accepted:
            cfg["style_bible"] = {"logline": e_log.text().strip(), "palette": e_pal.text().strip(),
                                  "look": e_look.text().strip(), "directors": e_dir.text().strip()}
            db.update("project", pid, config=json.dumps(cfg, ensure_ascii=False))
            self.log.appendPlainText(tr("fs.sb.bible_saved", "Style Bible salvata — iniettata in ogni prompt."))

    def _save_models(self, *_):
        """B5: salva la scelta modello immagine/video nella config del progetto (letta da _gen)."""
        pid = self._get_project()
        if not pid:
            return
        proj = db.get("project", pid)
        cfg = json.loads(proj.get("config") or "{}")
        im = self.mdl_img.currentData()
        vd = self.mdl_vid.currentData()
        img_arch = self.img_arch.currentData() or "flux"
        if img_arch == "flux":                       # FLUX: usa arch+file del checkpoint scelto
            cfg["img_arch"] = (im or {}).get("arch", "flux") if im else "flux"
            cfg["img_model"] = (im or {}).get("path", "") if im else ""
        else:                                        # Krea 2: snapshot diffusers, nessun file
            cfg["img_arch"] = img_arch
            cfg["img_model"] = ""
        cfg["vid_gguf"] = (vd or {}).get("path", "") if vd else ""
        cfg["vid_arch"] = self.vid_arch.currentData() or "ltx"
        db.update("project", pid, config=json.dumps(cfg, ensure_ascii=False))

    def _load_models(self):
        """Riflette la scelta salvata nei dropdown all'apertura del progetto."""
        pid = self._get_project()
        if not pid:
            return
        cfg = json.loads((db.get("project", pid) or {}).get("config") or "{}")
        for combo, key in ((self.mdl_img, "img_model"), (self.mdl_vid, "vid_gguf")):
            want = cfg.get(key, "")
            combo.blockSignals(True)
            idx = 0
            for i in range(combo.count()):
                d = combo.itemData(i)
                if d and d.get("path") == want:
                    idx = i; break
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        # architettura immagine salvata (krea2 = snapshot; qualunque valore non-krea2 → FLUX)
        self.img_arch.blockSignals(True)
        _ii = self.img_arch.findData(cfg.get("img_arch", "flux"))
        self.img_arch.setCurrentIndex(_ii if _ii >= 0 else 0)
        self.img_arch.blockSignals(False)
        # architettura video salvata
        self.vid_arch.blockSignals(True)
        _ia = self.vid_arch.findData(cfg.get("vid_arch", "ltx"))
        self.vid_arch.setCurrentIndex(_ia if _ia >= 0 else 0)
        self.vid_arch.blockSignals(False)

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
        chain = self.chk_chain.isChecked()                   # raccordo visivo tra shot consecutivi (stessa scena)
        for sc in db.query("scene", "project_id=?", (pid,), order="ord"):
            state = json.loads(sc.get("continuity") or "{}")
            _prev_vid = None                                 # ultimo shot-video della scena (per il raccordo)
            for sh in db.query("shot", "scene_id=?", (sc["id"],), order="ord"):
                _flavor = "image" if what == "keyframe" else "video"   # B2: LoRA del flavor giusto
                # prompt-shot CONDIVISO con Lucky Director (gen_service.compose_shot_prompt): @ref + camera +
                # continuity + required + Style Bible. La conversione JSON avviene poi in gen_image/gen_shot_video.
                prompt, refs = gen_service.compose_shot_prompt(sh, state, guard, path_mode=pmode, flavor=_flavor)
                # B4: seed per-shot — bloccato (>0) o stabile-per-shot (varia tra shot, riproducibile) invece
                # del default fisso 7 per tutti. Congela l'identità/inquadratura di QUESTO shot.
                _seed = int(sh.get("seed", 0) or 0) or (int(sh["id"]) * 2654435761 % (2**31 - 1))
                if len(refs.get("loras") or []) > 3:        # #55 warning composizione multi-LoRA (concept bleed)
                    self.log.appendPlainText(tr("fs.sb.too_many_lora",
                        "⚠ Scena {s} shot {h}: {n} LoRA simultanee (>3) — rischio interferenza, valuta split.")
                        .format(s=sc["ord"], h=sh["ord"], n=len(refs["loras"])))
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
                    _job = {"prompt": prompt, "out": out, "duration_s": sh["duration_s"] or 4,
                            "keyframes": kfs, "loras": refs["loras"], "audio_path": audio_path,
                            "model_gguf": vid_gguf, "seed": _seed, "negative": forbidden,
                            "video_arch": guard.get("vid_arch", "ltx")}
                    if chain and _prev_vid and not kfs:       # raccordo: aggancia all'ultimo frame del video prima
                        _job["chain_from"] = _prev_vid        # estratto a runtime (il precedente sarà già pronto)
                    jid = scheduler.enqueue("video", _job, owner_kind="shot", owner_id=sh["id"])
                    _prev_vid = out
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
        prompt = refs["clean_text"] + ". " + ct.to_prompt(state) + self._style_suffix(_cfg)
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
            self.edit.clear(); self._reload_takes_strip(); return
        kind, rid = self._sel
        self._reload_takes_strip()
        row = db.get(kind, rid)
        self.edit.setPlainText((row or {}).get("description", "") if row else "")
        self.cam_box.setVisible(kind == "shot")
        if kind == "shot" and row:
            self.shot_seed.setValue(int(row.get("seed", 0) or 0))
            cam = (row.get("camera") or "").lower()          # popola i combo dai tag già presenti nel camera
            for w in (self.sh_type, self.sh_move):
                w.blockSignals(True); w.setCurrentIndex(0)
                for i in range(1, w.count()):
                    if w.itemData(i) and w.itemData(i).lower() in cam:
                        w.setCurrentIndex(i); break
                w.blockSignals(False)
            self.sh_trans.blockSignals(True)                 # transizione salvata
            self.sh_trans.setCurrentIndex(max(0, self.sh_trans.findData(row.get("transition") or "")))
            self.sh_trans.blockSignals(False)
            for w, k in ((self.sh_in, "in_s"), (self.sh_out, "out_s")):   # trim salvato
                w.blockSignals(True); w.setValue(float(row.get(k, 0) or 0)); w.blockSignals(False)
        # #56 mostra/popola l'editor Continuity solo per le SCENE
        self.ct_box.setVisible(kind == "scene")
        if kind == "scene" and row:
            import json
            st = json.loads(row.get("continuity") or "{}")
            for w, k, d in [(self.ct_time, "time_of_day", "day"), (self.ct_weather, "weather", "clear"),
                            (self.ct_light, "light", "natural soft"), (self.ct_mood, "mood", "neutral"),
                            (self.ct_lens, "lens", ""), (self.ct_film, "film_look", "")]:
                i = w.findText(str(st.get(k, d)))
                w.setCurrentIndex(max(0, i))

    def _save_seed(self):
        if self._sel and self._sel[0] == "shot":
            db.update("shot", self._sel[1], seed=int(self.shot_seed.value()))

    def _reload_takes_strip(self):
        """Popola la strip-varianti inline coi take del keyframe dello shot selezionato (o la nasconde)."""
        if not hasattr(self, "takes_strip"):
            return
        self.takes_strip.clear()
        show = bool(self._sel and self._sel[0] == "shot")
        tk = []
        if show:
            from core.film_studio import takes
            tk = takes.list_takes("shot", self._sel[1], "keyframe")
        show = show and len(tk) > 0
        self.takes_lbl.setVisible(show); self.takes_strip.setVisible(show)
        if not show:
            return
        from PySide6.QtWidgets import QListWidgetItem
        from PySide6.QtGui import QIcon
        import os as _os
        cur = self._current_kf_path()
        cur_size = _os.path.getsize(cur) if (cur and _os.path.exists(cur)) else -1
        for t in tk:
            if not _os.path.exists(t["file"]):
                continue
            it = QListWidgetItem(QIcon(t["file"]), f"v{t['version']}")
            it.setData(Qt.UserRole, t["version"])
            if cur_size >= 0 and _os.path.getsize(t["file"]) == cur_size:   # take corrente (evidenziato)
                it.setText(f"● v{t['version']}")
            self.takes_strip.addItem(it)

    def _current_kf_path(self) -> str:
        if not (self._sel and self._sel[0] == "shot"):
            return ""
        sh = db.get("shot", self._sel[1]); sc = db.get("scene", sh["scene_id"]); pid = self._get_project()
        return str(db.project_dir(pid) / f"scene_{sc['ord']}" / f"shot_{sh['ord']}_kf.png")

    def _restore_take_inline(self, item):
        """Clic su una variante nella strip → la ripristina come keyframe corrente (non-distruttivo)."""
        if not (self._sel and self._sel[0] == "shot"):
            return
        from core.film_studio import takes
        ver = item.data(Qt.UserRole)
        canon = self._current_kf_path()
        if canon and takes.restore_take("shot", self._sel[1], "keyframe", ver, canon):
            self._refresh(); self._reload_takes_strip()
            self.log.appendPlainText(tr("fs.sb.takes_done", "Take v{v} ripristinato come corrente.").format(v=ver))

    def _browse_takes(self):
        """Storia dei take del keyframe dello shot selezionato: galleria + ripristino non-distruttivo."""
        if not self._sel or self._sel[0] != "shot":
            QMessageBox.information(self, tr("fs.sb.takes", "Take"), tr("fs.sb.sel_shot", "Seleziona un'inquadratura.")); return
        from core.film_studio import takes
        sid = self._sel[1]
        tk = takes.list_takes("shot", sid, "keyframe")
        if not tk:
            QMessageBox.information(self, tr("fs.sb.takes", "Take"), tr("fs.sb.takes_none", "Nessun take: genera il keyframe almeno una volta.")); return
        sh = db.get("shot", sid); sc = db.get("scene", sh["scene_id"]); pid = self._get_project()
        canon = str(db.project_dir(pid) / f"scene_{sc['ord']}" / f"shot_{sh['ord']}_kf.png")
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QListWidget, QListWidgetItem, QVBoxLayout
        from PySide6.QtGui import QIcon
        from PySide6.QtCore import QSize as _QS
        dlg = QDialog(self); dlg.setWindowTitle(tr("fs.sb.takes", "🎞 Storia take")); lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(tr("fs.sb.takes_h", "Scegli un take e Ripristina (i tentativi non si perdono):")))
        lst = QListWidget(); lst.setIconSize(_QS(128, 72)); lst.setViewMode(QListWidget.IconMode); lst.setResizeMode(QListWidget.Adjust)
        for t in tk:
            it = QListWidgetItem(QIcon(t["file"]), f"v{t['version']}"); it.setData(Qt.UserRole, t["version"])
            lst.addItem(it)
        lay.addWidget(lst, 1)
        bb = QDialogButtonBox(); btn_r = bb.addButton(tr("fs.sb.takes_restore", "Ripristina"), QDialogButtonBox.AcceptRole)
        bb.addButton(QDialogButtonBox.Close); bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() == QDialog.Accepted and lst.currentItem():
            ver = lst.currentItem().data(Qt.UserRole)
            if takes.restore_take("shot", sid, "keyframe", ver, canon):
                self._refresh()
                self.log.appendPlainText(tr("fs.sb.takes_done", "Take v{v} ripristinato come corrente.").format(v=ver))

    def _expand_coverage(self):
        """Espande lo shot-dialogo selezionato in copertura campo/controcampo (una inquadratura per battuta)."""
        if not self._sel or self._sel[0] != "shot":
            QMessageBox.information(self, tr("fs.sb.coverage", "Copertura"), tr("fs.sb.sel_shot", "Seleziona un'inquadratura.")); return
        from core.film_studio import coverage
        ids = coverage.expand_shot_to_coverage(self._sel[1])
        if not ids:
            QMessageBox.information(self, tr("fs.sb.coverage", "Copertura"),
                                   tr("fs.sb.cov_none", "Servono ≥2 battute di dialogo per la copertura.")); return
        self._refresh()
        self.log.appendPlainText(tr("fs.sb.cov_done", "Copertura: {n} inquadrature (campo/controcampo).").format(n=len(ids)))

    def _run_supervisor(self):
        """Moonshot — segretario di edizione: rilegge il breakdown e segnala i salti di raccordo."""
        pid = self._get_project()
        if not pid or not db.query("scene", "project_id=?", (pid,)):
            QMessageBox.information(self, tr("fs.sb.supervisor", "Supervisore"),
                                   tr("fs.sb.no_break", "Fai prima il breakdown (scene/inquadrature).")); return
        self.btn_supervisor.setEnabled(False); self.log.appendPlainText(tr("fs.working", "Generazione…"))
        from PySide6.QtCore import QThread, Signal

        class _W(QThread):
            done = Signal(dict)

            def run(self):
                from core.film_studio import supervisor
                try:
                    self.done.emit(supervisor.script_supervisor(pid))
                except Exception as e:                       # noqa: BLE001
                    self.done.emit({"deterministic": [], "llm": [], "available": False, "error": str(e)})
        self._sup_w = _W(); self._sup_w.done.connect(self._on_supervisor); self._sup_w.start()

    def _on_supervisor(self, res: dict):
        self.btn_supervisor.setEnabled(True)
        lines = []
        for w in res.get("deterministic", []):
            lines.append("• [continuity] " + w)
        for i in res.get("llm", []):
            sev = {"high": "‼", "med": "⚠", "low": "·"}.get(i.get("severity", "med"), "⚠")
            lines.append(f"{sev} [{i.get('where', '?')}] {i.get('kind', '')}: {i.get('note', '')}")
        if not lines:
            lines = [tr("fs.sb.sup_clean", "✓ Nessun salto di raccordo evidente.")
                     + ("" if res.get("available") else " " + tr("fs.llm.unavailable", "(LLM non disponibile — solo controlli base)"))]
        self.log.appendPlainText(tr("fs.sb.sup_head", "🧐 Supervisore continuità:"))
        for ln in lines:
            self.log.appendPlainText("  " + ln)

    def _run_dop(self):
        """Moonshot — direttore della fotografia: propone camera+luce per gli shot della scena selezionata."""
        if not self._sel or self._sel[0] != "scene":
            QMessageBox.information(self, tr("fs.sb.dop", "DoP"), tr("fs.sb.sel_scene", "Seleziona una scena.")); return
        if not L.available():
            QMessageBox.warning(self, tr("fs.sb.dop", "DoP"), tr("fs.llm.unavailable", "LLM non disponibile.")); return
        scene_id = self._sel[1]
        self.btn_dop.setEnabled(False); self.log.appendPlainText(tr("fs.working", "Generazione…"))
        from PySide6.QtCore import QThread, Signal

        class _W(QThread):
            done = Signal(list)

            def run(self):
                from core.film_studio import supervisor
                try:
                    self.done.emit(supervisor.dop_suggestions(scene_id))
                except Exception:                            # noqa: BLE001
                    self.done.emit([])
        self._dop_w = _W(); self._dop_w.done.connect(self._on_dop); self._dop_w.start()

    def _on_dop(self, sugg: list):
        self.btn_dop.setEnabled(True)
        if not sugg:
            QMessageBox.information(self, tr("fs.sb.dop", "DoP"), tr("fs.sb.dop_none", "Nessuna proposta (LLM o scena vuota).")); return
        from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QVBoxLayout, QTableWidget, QTableWidgetItem,
                                       QHeaderView)
        dlg = QDialog(self); dlg.setWindowTitle(tr("fs.sb.dop", "🎥 DoP: proposta ripresa")); lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(tr("fs.sb.dop_head", "Proposte camera+luce per shot (Applica scrive shot.camera):")))
        t = QTableWidget(len(sugg), 3)
        t.setHorizontalHeaderLabels([tr("fs.sb.dop_shot", "Shot"), tr("fs.sb.dop_cam", "Camera + luce"),
                                     tr("fs.sb.dop_why", "Perché")])
        t.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        t.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        for r, s in enumerate(sugg):
            cam = ", ".join(x for x in (s.get("camera", ""), s.get("lighting", "")) if x)
            t.setItem(r, 0, QTableWidgetItem(str(s.get("ord", ""))))
            t.setItem(r, 1, QTableWidgetItem(cam))
            t.setItem(r, 2, QTableWidgetItem(s.get("rationale", "")))
        t.resizeRowsToContents()
        lay.addWidget(t, 1)
        bb = QDialogButtonBox(); bb.addButton(tr("fs.sb.dop_apply", "Applica a tutti"), QDialogButtonBox.AcceptRole)
        bb.addButton(QDialogButtonBox.Close); bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() == QDialog.Accepted:
            from core.film_studio import supervisor
            n = supervisor.apply_dop(sugg)
            self._refresh()
            self.log.appendPlainText(tr("fs.sb.dop_done", "🎥 DoP: camera+luce applicate a {n} shot.").format(n=n))

    def _save_shot_grammar(self, *_):
        """Compone shot.camera dai combo tipo-inquadratura + movimento (iniettato nel prompt in _gen)."""
        if not self._sel or self._sel[0] != "shot":
            return
        cam = ", ".join(x for x in (self.sh_type.currentData(), self.sh_move.currentData()) if x)
        db.update("shot", self._sel[1], camera=cam, transition=self.sh_trans.currentData() or "",
                  in_s=float(self.sh_in.value()), out_s=float(self.sh_out.value()))

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
                   "light": self.ct_light.currentText(), "mood": self.ct_mood.currentText(),
                   "lens": self.ct_lens.currentText(), "film_look": self.ct_film.currentText()})
        db.update("scene", rid, continuity=json.dumps(st, ensure_ascii=False))
        # propaga in avanti: ogni scena successiva eredita lo stato (override mantenuti dove già impostati)
        prev = st
        _warns = []
        for nx in db.query("scene", "project_id=? AND ord>?", (sc["project_id"], sc["ord"]), order="ord"):
            nst = ct.propagate(prev, json.loads(nx.get("continuity") or "{}"))
            for w in (ct.warnings(prev, nst) or []):         # B8: script-supervisor anti-salto (prima morto)
                _warns.append(f"scena {nx['ord']}: {w}")
            db.update("scene", nx["id"], continuity=json.dumps(nst, ensure_ascii=False))
            prev = nst
        self.log.appendPlainText(tr("fs.sb.ct_done", "Continuity applicata e propagata alle scene successive."))
        for w in _warns:                                     # segnala i salti di raccordo (luce/tempo/condizioni)
            self.log.appendPlainText("⚠ " + w)

    def _edit_char_continuity(self):
        """B9: stato per-personaggio della scena (abbigliamento/condizione/emozione) — la metà del continuity
        engine che la UI non popolava. Gli @Nome si raccolgono dagli shot della scena."""
        if not self._sel or self._sel[0] != "scene":
            QMessageBox.information(self, tr("fs.sb.ct_chars", "Stato personaggi"), tr("fs.sb.sel_scene", "Seleziona una scena.")); return
        import re
        sc = db.get("scene", self._sel[1])
        names = []
        for sh in db.query("shot", "scene_id=?", (sc["id"],)):
            for m in re.finditer(r"@([A-Za-z0-9_]+)", sh.get("description") or ""):
                if m.group(1) not in names:
                    names.append(m.group(1))
        if not names:
            QMessageBox.information(self, tr("fs.sb.ct_chars", "Stato personaggi"), tr("fs.sb.no_chars", "Nessun @personaggio negli shot della scena.")); return
        st = json.loads(sc.get("continuity") or "{}")
        assets = st.get("assets") or {}
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLineEdit as _LE, QGroupBox as _GB, QVBoxLayout as _VB
        dlg = QDialog(self); dlg.setWindowTitle(tr("fs.sb.ct_chars", "👤 Stato personaggi")); outer = _VB(dlg)
        rows = {}
        for n in names:
            a = assets.get(n) or {}
            gb = _GB("@" + n); ff = QFormLayout(gb)
            e_w = _LE(a.get("wardrobe", "")); e_c = _LE(a.get("condition", "")); e_e = _LE(a.get("emotion", ""))
            ff.addRow(tr("fs.sb.ct_wardrobe", "Abbigliamento"), e_w)
            ff.addRow(tr("fs.sb.ct_condition", "Condizione (es. ferito, bagnato)"), e_c)
            ff.addRow(tr("fs.sb.ct_emotion", "Emozione"), e_e)
            outer.addWidget(gb); rows[n] = (e_w, e_c, e_e)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); outer.addWidget(bb)
        if dlg.exec() != QDialog.Accepted:
            return
        new_assets = {}
        for n, (e_w, e_c, e_e) in rows.items():
            d = {k: v for k, v in (("wardrobe", e_w.text().strip()), ("condition", e_c.text().strip()),
                                   ("emotion", e_e.text().strip())) if v}
            if d:
                new_assets[n] = d
        st["assets"] = new_assets
        db.update("scene", sc["id"], continuity=json.dumps(st, ensure_ascii=False))
        # propaga in avanti (le condizioni persistono nelle scene successive)
        prev = st
        for nx in db.query("scene", "project_id=? AND ord>?", (sc["project_id"], sc["ord"]), order="ord"):
            nst = ct.propagate(prev, json.loads(nx.get("continuity") or "{}"))
            db.update("scene", nx["id"], continuity=json.dumps(nst, ensure_ascii=False)); prev = nst
        self.log.appendPlainText(tr("fs.sb.ct_chars_done", "Stato personaggi salvato e propagato."))

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
            if ev.get("event") == "done":                    # snapshot NON-distruttivo del take prodotto
                try:
                    from core.film_studio import takes
                    job = db.get("job", ev.get("job_id")) or {}
                    p = json.loads(job.get("params") or "{}")
                    if p.get("_owner_kind") == "shot" and p.get("out"):
                        kind = "keyframe" if job.get("type") == "image" else "video"
                        takes.record_take("shot", int(p["_owner_id"]), kind, p["out"])
                        if self._sel and self._sel[0] == "shot" and int(p["_owner_id"]) == int(self._sel[1]):
                            self._reload_takes_strip()          # nuovo take → aggiorna la strip inline
                except Exception:                            # noqa: BLE001
                    pass
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
