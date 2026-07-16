"""Costruttore di prompt STRUTTURATO (JSON) per immagini/video.

Invece del testo libero, l'utente compila campi (soggetto, azione, stile, camera, luce, mood, dettagli,
negativo). Il widget:
 - COMPONE un prompt in linguaggio naturale dai campi valorizzati (i modelli locali prendono testo);
 - serializza/deserializza in JSON (to_json/from_json) → riproducibilità, preset, versionamento.
Emette composed(prompt, negative) al click su 'Componi'. Riutilizzabile in un dialog o inline nelle tab.
"""
from __future__ import annotations

import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit, QComboBox, QPushButton, QHBoxLayout,
)

from ui.i18n_manager import tr

# opzioni guidate (vuoto = non specificato). Il primo elemento "" = "(qualsiasi)".
_CAMERA = ["", "close-up", "medium shot", "wide shot", "full body", "portrait", "aerial view",
           "low angle", "high angle", "over-the-shoulder", "macro", "bird's eye"]
_LIGHT = ["", "soft light", "dramatic lighting", "golden hour", "blue hour", "neon", "backlight",
          "studio lighting", "natural light", "cinematic", "high key", "low key", "candlelight"]
_STYLE = ["", "photorealistic", "cinematic", "anime", "3d render", "oil painting", "watercolor",
          "cartoon", "cyberpunk", "vintage film", "black and white", "concept art", "hyperrealistic"]


class PromptBuilder(QWidget):
    """Form di campi strutturati → prompt. get_json()/set_json() per preset; compose() → (prompt, negative)."""
    composed = Signal(str, str)                              # (prompt, negative)

    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0)
        f = QFormLayout(); f.setLabelAlignment(Qt.AlignRight)
        self.subject = QLineEdit(); self.subject.setPlaceholderText(tr("pb.subject.ph", "es. una donna con un cappotto rosso"))
        f.addRow(tr("pb.subject", "Soggetto"), self.subject)
        self.action = QLineEdit(); self.action.setPlaceholderText(tr("pb.action.ph", "es. cammina sotto la pioggia"))
        f.addRow(tr("pb.action", "Azione/posa"), self.action)
        self.setting = QLineEdit(); self.setting.setPlaceholderText(tr("pb.setting.ph", "es. strada di città al neon"))
        f.addRow(tr("pb.setting", "Ambientazione"), self.setting)
        self.style = QComboBox(); self.style.setEditable(True); self.style.addItems(_STYLE)
        f.addRow(tr("pb.style", "Stile"), self.style)
        self.camera = QComboBox(); self.camera.addItems(_CAMERA)
        f.addRow(tr("pb.camera", "Camera/inquadratura"), self.camera)
        self.light = QComboBox(); self.light.addItems(_LIGHT)
        f.addRow(tr("pb.light", "Luce"), self.light)
        self.mood = QLineEdit(); self.mood.setPlaceholderText(tr("pb.mood.ph", "es. malinconico, epico, intimo"))
        f.addRow(tr("pb.mood", "Mood"), self.mood)
        self.details = QLineEdit(); self.details.setPlaceholderText(tr("pb.details.ph", "es. dettagli nitidi, 8k, bokeh"))
        f.addRow(tr("pb.details", "Dettagli/qualità"), self.details)
        self.negative = QLineEdit(); self.negative.setPlaceholderText(tr("pb.negative.ph", "es. sfocato, deforme, testo"))
        f.addRow(tr("pb.negative", "Negativo"), self.negative)
        v.addLayout(f)
        row = QHBoxLayout()
        self.btn = QPushButton(tr("pb.compose", "🧱 Componi prompt")); self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.clicked.connect(self._emit)
        row.addStretch(1); row.addWidget(self.btn)
        v.addLayout(row)

    # ── composizione ─────────────────────────────────────────────────────────
    def _fields(self) -> dict:
        """Campi correnti come dizionario nello schema condiviso (core.prompt_struct)."""
        return {"subject": self.subject.text(), "action": self.action.text(), "setting": self.setting.text(),
                "style": self.style.currentText(), "camera": self.camera.currentText(),
                "light": self.light.currentText(), "mood": self.mood.text(),
                "details": self.details.text(), "negative": self.negative.text()}

    def compose(self) -> tuple[str, str]:
        """Costruisce (prompt, negative) dai campi valorizzati. Usa la composizione CONDIVISA con la pipeline."""
        from core import prompt_struct
        return prompt_struct.compose_fields(self._fields())

    def _emit(self):
        p, n = self.compose()
        self.composed.emit(p, n)

    # ── JSON (preset/riproducibilità) ─────────────────────────────────────────
    def get_json(self) -> str:
        return json.dumps(self._fields(), ensure_ascii=False, indent=2)

    def set_json(self, s: str):
        try:
            d = json.loads(s) if isinstance(s, str) else (s or {})
        except Exception:                                    # noqa: BLE001
            return
        self.subject.setText(d.get("subject", "")); self.action.setText(d.get("action", ""))
        self.setting.setText(d.get("setting", "")); self.style.setCurrentText(d.get("style", ""))
        self.camera.setCurrentText(d.get("camera", "")); self.light.setCurrentText(d.get("light", ""))
        self.mood.setText(d.get("mood", "")); self.details.setText(d.get("details", ""))
        self.negative.setText(d.get("negative", ""))


class PromptBuilderDialog(QWidget):
    """Dialog non-modale attorno a PromptBuilder + preset JSON. applied(prompt, negative) al click 'Usa'."""
    applied = Signal(str, str)

    def __init__(self, parent=None):
        from PySide6.QtWidgets import QDialog, QComboBox as _CB
        self._dlg = QDialog(parent); self._dlg.setWindowTitle(tr("pb.title", "🧱 Costruttore di prompt"))
        self._dlg.setModal(False); self._dlg.resize(460, 480)
        super().__init__(self._dlg)
        from pathlib import Path
        self._presets = Path(__file__).resolve().parents[2] / "data" / "prompt_presets.json"
        v = QVBoxLayout(self._dlg)
        self.pb = PromptBuilder(); v.addWidget(self.pb)
        prow = QHBoxLayout()
        self.preset = _CB(); self.preset.setMinimumWidth(140); self.preset.activated.connect(self._load_preset)
        self.btn_save = QPushButton("💾"); self.btn_save.setToolTip(tr("pb.save.tt", "Salva questi campi come preset."))
        self.btn_save.clicked.connect(self._save_preset)
        prow.addWidget(self.preset); prow.addWidget(self.btn_save); prow.addStretch(1)
        self.btn_use = QPushButton(tr("pb.use", "Usa il prompt")); self.btn_use.setCursor(Qt.PointingHandCursor)
        self.btn_use.clicked.connect(self._use)
        prow.addWidget(self.btn_use)
        v.addLayout(prow)
        self._refresh_presets()

    def show(self):
        self._dlg.show(); self._dlg.raise_()

    def _use(self):
        p, n = self.pb.compose(); self.applied.emit(p, n)

    def _load(self) -> dict:
        try:
            return json.loads(self._presets.read_text(encoding="utf-8")) if self._presets.exists() else {}
        except Exception:                                    # noqa: BLE001
            return {}

    def _refresh_presets(self, select: str = ""):
        self.preset.blockSignals(True); self.preset.clear()
        self.preset.addItem(tr("pb.preset.none", "— preset —"), "")
        for name in sorted(self._load()):
            self.preset.addItem(name, name)
        if select:
            i = self.preset.findData(select)
            if i >= 0:
                self.preset.setCurrentIndex(i)
        self.preset.blockSignals(False)

    def _load_preset(self, *_):
        name = self.preset.currentData()
        if name:
            self.pb.set_json(json.dumps(self._load().get(name, {})))

    def _save_preset(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self._dlg, tr("pb.save", "Salva preset"), tr("pb.name", "Nome:"))
        name = (name or "").strip()
        if not ok or not name:
            return
        d = self._load(); d[name] = json.loads(self.pb.get_json())
        self._presets.parent.mkdir(parents=True, exist_ok=True)
        self._presets.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        self._refresh_presets(select=name)
