"""Finestra AI Film Studio — workflow a tab (Input→Copione→Scenografia→Montaggio) + Asset Manager.
Condivide il runtime dell'app madre; backend 100% NATIVO. Registra gli handler del Job Scheduler
mappandoli sul Generation Service nativo."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QTabWidget, QVBoxLayout, QWidget)

from core.film_studio import db, gen_service, scheduler
from ui.i18n_manager import tr
from ui.film_studio.llm_settings import LlmSettings
from ui.film_studio.script_tab import ScriptTab
from ui.film_studio.asset_manager_tab import AssetManagerTab
from ui.film_studio.storyboard_tab import StoryboardTab
from ui.film_studio.montage_tab import MontageTab
from ui.film_studio.voice_tab import VoiceTab
from ui.film_studio.lora_studio_tab import LoraStudioTab

_PATHS = [("fast", "fs.path.fast"), ("lora", "fs.path.lora"), ("mixed", "fs.path.mixed")]
_RUN = [("supervised", "fs.run.supervised"), ("auto", "fs.run.auto")]


def _register_scheduler_handlers():
    """Mappa i tipi di job sul Generation Service nativo (idempotente)."""
    def _img(p):
        return gen_service.gen_image(p["prompt"], p["out"], face_ref=p.get("face_ref", ""),
                                     ref_images=p.get("ref_images"), loras=p.get("loras"),
                                     arch=p.get("arch", "flux"), model=p.get("model", ""),
                                     negative=p.get("negative", ""), seed=int(p.get("seed", 7)))

    def _vid(p):
        return gen_service.gen_shot_video(p["prompt"], p["out"], duration_s=p.get("duration_s", 4),
                                          keyframes=p.get("keyframes"), loras=p.get("loras"),
                                          audio_path=p.get("audio_path", "") or "",
                                          model_gguf=p.get("model_gguf", ""),
                                          negative=p.get("negative", ""), seed=int(p.get("seed", 7)))
    scheduler.register("image", _img)
    scheduler.register("video", _vid)


class FilmStudioWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Finestra top-level INDIPENDENTE: title bar di sistema, spostabile + ridimensionabile,
        # NON modale (non blocca la finestra madre), sfondo pieno (non widget figlio trasparente).
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle(tr("fs.title", "🎬 AI Film Studio"))
        self.setAttribute(Qt.WA_StyledBackground, True)      # rispetta il bg del tema (niente trasparenza)
        self.setAutoFillBackground(True)
        self.setStyleSheet("QWidget#FilmStudioRoot { background: #1e1e1e; }")
        self.setObjectName("FilmStudioRoot")
        self.resize(1100, 760)
        self.setMinimumSize(800, 560)                        # ridimensionabile entro un minimo sensato
        self._project_id = None
        _register_scheduler_handlers()
        try:
            n = scheduler.recover()                          # B10: riprende i job orfani (running post-crash)
            if n:
                print(f"[film-studio] recovery: {n} job ripresi dopo crash", flush=True)
        except Exception:                                    # noqa: BLE001
            pass
        root = QVBoxLayout(self)
        root.addWidget(self._project_bar())
        root.addWidget(LlmSettings())
        _banner = self._flow_banner()                        # guida al flusso (Classica sì, Pro no — riallineamento)
        if _banner is not None:
            root.addWidget(_banner)
        self.tabs = QTabWidget()
        # factory dei tab figli (override in FilmStudioWindowPro per le varianti Pro)
        self.script_tab = self._make_script_tab()
        self.asset_tab = self._make_asset_tab()
        self.sb_tab = self._make_sb_tab()
        self.mt_tab = self._make_mt_tab()
        self.sb_tab.auto_complete.connect(self._on_auto_complete)   # #54: Auto → montaggio automatico
        self.script_tab.script_ready.connect(lambda *_: self.sb_tab._refresh())   # #95: copione salvato → aggiorna Scenografia
        self.voice_tab = self._make_voice_tab()
        self.lora_tab = self._make_lora_tab()
        self.tabs.addTab(self.script_tab, tr("fs.tab.script", "Copione"))
        self.tabs.addTab(self.sb_tab, tr("fs.tab.scenografia", "Scenografia"))
        self.tabs.addTab(self.mt_tab, tr("fs.tab.montage", "Montaggio"))
        self.tabs.addTab(self.asset_tab, tr("fs.tab.assets", "Asset Manager"))
        self.tabs.addTab(self.voice_tab, tr("fs.tab.voice", "Voce"))
        self.tabs.addTab(self.lora_tab, tr("fs.tab.lora", "LoRA Studio"))
        root.addWidget(self.tabs, 1)
        self._refresh_projects()
        self._refresh_series()

    def _get_project(self):
        return self._project_id

    def _flow_banner(self):
        """Banner-guida al flusso di lavoro, per i PRINCIPIANTI (variante Classica). La Pro lo nasconde (override
        → None): riallineamento Classica/Pro (prima la guida stava, invertita, nella Pro)."""
        lbl = QLabel(tr("fs.flow", "Flusso:  ① Copione (idea → Racconto → copione)  →  ② Scenografia "
                                   "(keyframe → video)  →  ③ Montaggio.   Usa @Nome per la coerenza degli asset."))
        lbl.setWordWrap(True)
        lbl.setStyleSheet("QLabel{background:#26333f;color:#d6e6f2;padding:6px 10px;border-radius:5px;}")
        return lbl

    # ---- factory dei tab figli (Classica; override nella variante Pro) ----
    def _make_script_tab(self):
        return ScriptTab(self._get_project)

    def _make_asset_tab(self):
        return AssetManagerTab()

    def _make_sb_tab(self):
        return StoryboardTab(self._get_project)

    def _make_mt_tab(self):
        return MontageTab(self._get_project)

    def _make_voice_tab(self):
        return VoiceTab()

    def _make_lora_tab(self):
        return LoraStudioTab()

    def _project_bar(self) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
        self.proj_combo = QComboBox(); self.proj_combo.setMinimumWidth(220)
        self.proj_combo.currentIndexChanged.connect(self._on_pick_project)
        h.addWidget(QLabel(tr("fs.project", "Progetto"))); h.addWidget(self.proj_combo)
        self.new_name = QLineEdit(); self.new_name.setPlaceholderText(tr("fs.project_name", "Nuovo progetto…"))
        h.addWidget(self.new_name, 1)
        self.dur = QDoubleSpinBox(); self.dur.setRange(5, 7200); self.dur.setValue(60); self.dur.setSuffix(" s")
        h.addWidget(QLabel(tr("fs.duration", "Durata"))); h.addWidget(self.dur)
        self.path_mode = QComboBox()
        for code, key in _PATHS:
            self.path_mode.addItem(tr(key, code), code)
        h.addWidget(QLabel(tr("fs.path_mode", "Coerenza"))); h.addWidget(self.path_mode)
        self.run_mode = QComboBox()
        for code, key in _RUN:
            self.run_mode.addItem(tr(key, code), code)
        h.addWidget(self.run_mode)
        b = QPushButton(tr("fs.project_create", "➕ Crea")); b.clicked.connect(self._create_project)
        h.addWidget(b)
        b_mng = QPushButton(tr("fs.project_manage", "📂 Progetti")); b_mng.clicked.connect(self._manage_projects)
        h.addWidget(b_mng)
        # CRUD diretto sul progetto selezionato (oltre al dialog 📂): rinomina / duplica / elimina
        b_ren = QPushButton("✏️"); b_ren.setToolTip(tr("fs.project_rename", "Rinomina progetto"))
        b_ren.setMaximumWidth(34); b_ren.clicked.connect(self._rename_project); h.addWidget(b_ren)
        b_dup = QPushButton("⧉"); b_dup.setToolTip(tr("fs.project_dup", "Duplica progetto"))
        b_dup.setMaximumWidth(34); b_dup.clicked.connect(self._duplicate_project); h.addWidget(b_dup)
        b_del = QPushButton("🗑️"); b_del.setToolTip(tr("fs.project_del", "Elimina progetto"))
        b_del.setMaximumWidth(34); b_del.clicked.connect(self._delete_project); h.addWidget(b_del)
        # 1.8 Serie: il progetto può appartenere a una Serie (asset/LoRA scope=series condivisi)
        self.series = QComboBox(); self.series.setMinimumWidth(140)
        self.series.currentIndexChanged.connect(self._assign_series)
        h.addWidget(QLabel(tr("fs.series", "Serie"))); h.addWidget(self.series)
        b_ns = QPushButton(tr("fs.series_new", "＋")); b_ns.setMaximumWidth(32); b_ns.clicked.connect(self._new_series)
        h.addWidget(b_ns)
        # 1.7 Continuity: elementi vietati/forzati per progetto
        self.forbidden = QLineEdit(); self.forbidden.setPlaceholderText(tr("fs.forbidden", "vietati (negative)"))
        self.required = QLineEdit(); self.required.setPlaceholderText(tr("fs.required", "must-have (sempre)"))
        self.forbidden.editingFinished.connect(self._save_guardrails)
        self.required.editingFinished.connect(self._save_guardrails)
        h.addWidget(self.forbidden); h.addWidget(self.required)
        return w

    def _refresh_series(self):
        self.series.blockSignals(True); self.series.clear()
        self.series.addItem(tr("fs.series_none", "— nessuna —"), 0)
        self._series = db.query("series", order="id DESC")
        for s in self._series:
            self.series.addItem(s["name"], s["id"])
        if self._project_id:
            sid = (db.get("project", self._project_id) or {}).get("series_id") or 0
            idx = self.series.findData(sid)
            self.series.setCurrentIndex(max(0, idx))
        self.series.blockSignals(False)

    def _new_series(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, tr("fs.series", "Serie"), tr("fs.series_name", "Nome della serie:"))
        if ok and name.strip():
            db.insert("series", name=name.strip(), notes="")
            self._refresh_series()

    def _assign_series(self, _idx: int):
        if self._project_id and hasattr(self, "_series"):
            db.update("project", self._project_id, series_id=int(self.series.currentData() or 0))

    def _save_guardrails(self):
        import json
        if not self._project_id:
            return
        db.update("project", self._project_id,
                  config=json.dumps({"forbidden": self.forbidden.text().strip(),
                                     "required": self.required.text().strip()}, ensure_ascii=False))

    def _on_auto_complete(self, pid: int):
        """#54 Auto end-to-end: i video sono pronti → genera dialoghi+musica+ambient e il master."""
        self.mt_tab._refresh()
        try:
            self.mt_tab._gen_audio()                        # dialoghi (voce legata) — sincrono
            # #95: la musica è ASINCRONA → il master deve partire DOPO (era chiamato subito = master senza musica)
            self.mt_tab._gen_music(then=self.mt_tab._master)
        except Exception:                                   # noqa: BLE001
            self.mt_tab._master()                          # fallback: master comunque
        self.tabs.setCurrentWidget(self.mt_tab)

    def _manage_projects(self):
        from ui.film_studio.projects_dialog import FsProjectsDialog
        dlg = FsProjectsDialog(self)
        if dlg.exec() and dlg.opened_id:
            self._refresh_projects()
            idx = self.proj_combo.findData(dlg.opened_id)
            if idx >= 0:
                self.proj_combo.setCurrentIndex(idx)

    def _refresh_projects(self):
        self.proj_combo.blockSignals(True)
        self.proj_combo.clear()
        self._projects = db.query("project", order="id DESC")
        for p in self._projects:
            self.proj_combo.addItem(f"{p['name']} (#{p['id']})", p["id"])
        self.proj_combo.blockSignals(False)
        if self._projects:
            self._project_id = self._projects[0]["id"]

    def _on_pick_project(self, idx: int):
        if 0 <= idx < len(getattr(self, "_projects", [])):
            self._project_id = self._projects[idx]["id"]
            import json
            guard = json.loads(self._projects[idx].get("config") or "{}")
            self.forbidden.setText(guard.get("forbidden", "")); self.required.setText(guard.get("required", ""))
            self._refresh_series()
            for t in (self.sb_tab, self.mt_tab):
                if hasattr(t, "_refresh"):
                    t._refresh()

    def _create_project(self):
        name = self.new_name.text().strip()
        if not name:
            QMessageBox.information(self, tr("fs.project", "Progetto"), tr("fs.project_need_name", "Dai un nome al progetto.")); return
        pid = db.insert("project", name=name, kind="film", duration_s=float(self.dur.value()),
                        genres="", styles="", path_mode=self.path_mode.currentData(),
                        run_mode=self.run_mode.currentData(), config="{}", updated=0)
        self.new_name.clear()
        self._refresh_projects()
        self._project_id = pid
        idx = self.proj_combo.findData(pid)
        if idx >= 0:
            self.proj_combo.setCurrentIndex(idx)

    def _select_project_in_combo(self, pid):
        idx = self.proj_combo.findData(pid)
        if idx >= 0:
            self.proj_combo.setCurrentIndex(idx)

    def _rename_project(self):
        if not self._project_id:
            QMessageBox.information(self, tr("fs.project", "Progetto"), tr("fs.no_project", "Crea/seleziona un progetto.")); return
        from PySide6.QtWidgets import QInputDialog
        cur = (db.get("project", self._project_id) or {}).get("name", "")
        name, ok = QInputDialog.getText(self, tr("fs.project_rename", "Rinomina progetto"),
                                        tr("fs.project_name", "Nome del progetto:"), text=cur)
        if ok and name.strip():
            keep = self._project_id
            db.update("project", keep, name=name.strip())
            self._refresh_projects(); self._select_project_in_combo(keep)

    def _duplicate_project(self):
        if not self._project_id:
            QMessageBox.information(self, tr("fs.project", "Progetto"), tr("fs.no_project", "Crea/seleziona un progetto.")); return
        new_id = db.duplicate_project(self._project_id)
        self._refresh_projects(); self._select_project_in_combo(new_id)

    def _delete_project(self):
        if not self._project_id:
            QMessageBox.information(self, tr("fs.project", "Progetto"), tr("fs.no_project", "Crea/seleziona un progetto.")); return
        name = (db.get("project", self._project_id) or {}).get("name", "")
        if QMessageBox.question(self, tr("fs.project_del", "Elimina progetto"),
                tr("fs.project_del_confirm", "Eliminare il progetto «{n}»? Operazione irreversibile.").format(n=name)
                ) != QMessageBox.Yes:
            return
        db.delete_project(self._project_id)
        self._project_id = None
        self._refresh_projects()
        if getattr(self, "_projects", []):
            self.proj_combo.setCurrentIndex(0)
        for t in (self.sb_tab, self.mt_tab):
            if hasattr(t, "_refresh"):
                t._refresh()
