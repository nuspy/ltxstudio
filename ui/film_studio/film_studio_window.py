"""Finestra AI Film Studio — workflow a tab (Input→Copione→Scenografia→Montaggio) + Asset Manager.
Condivide il runtime dell'app madre; backend 100% NATIVO. Registra gli handler del Job Scheduler
mappandoli sul Generation Service nativo."""
from __future__ import annotations

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

_PATHS = [("fast", "fs.path.fast"), ("lora", "fs.path.lora"), ("mixed", "fs.path.mixed")]
_RUN = [("supervised", "fs.run.supervised"), ("auto", "fs.run.auto")]


def _register_scheduler_handlers():
    """Mappa i tipi di job sul Generation Service nativo (idempotente)."""
    def _img(p):
        return gen_service.gen_image(p["prompt"], p["out"], face_ref=p.get("face_ref", ""),
                                     loras=p.get("loras"))

    def _vid(p):
        return gen_service.gen_shot_video(p["prompt"], p["out"], duration_s=p.get("duration_s", 4),
                                          keyframes=p.get("keyframes"), loras=p.get("loras"),
                                          audio_path=p.get("audio_path", ""))
    scheduler.register("image", _img)
    scheduler.register("video", _vid)


class FilmStudioWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("fs.title", "🎬 AI Film Studio"))
        self.resize(1100, 760)
        self._project_id = None
        _register_scheduler_handlers()
        root = QVBoxLayout(self)
        root.addWidget(self._project_bar())
        root.addWidget(LlmSettings())
        self.tabs = QTabWidget()
        self.script_tab = ScriptTab(self._get_project)
        self.asset_tab = AssetManagerTab()
        self.sb_tab = StoryboardTab(self._get_project)
        self.mt_tab = MontageTab(self._get_project)
        self.voice_tab = VoiceTab()
        self.tabs.addTab(self.script_tab, tr("fs.tab.script", "Copione"))
        self.tabs.addTab(self.sb_tab, tr("fs.tab.scenografia", "Scenografia"))
        self.tabs.addTab(self.mt_tab, tr("fs.tab.montage", "Montaggio"))
        self.tabs.addTab(self.asset_tab, tr("fs.tab.assets", "Asset Manager"))
        self.tabs.addTab(self.voice_tab, tr("fs.tab.voice", "Voce"))
        root.addWidget(self.tabs, 1)
        self._refresh_projects()

    def _get_project(self):
        return self._project_id

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
        return w

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
