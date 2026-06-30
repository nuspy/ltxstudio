"""Wizard guidato — porta l'utente da "cosa vuoi creare" alle impostazioni base.

Copre l'intero ventaglio di moduli (non più solo i 4 obiettivi video):
video, immagini, modello 3D, musica/canzone, voce/sintesi, AI Film Studio.

Ogni obiettivo dichiara una *categoria* (`cat`) che la UI principale usa per:
  - aprire la tab giusta (o la finestra Film Studio),
  - sapere se serve un'immagine di partenza,
  - decidere se il prompt e la durata sono pertinenti.

Le scelte vengono restituite da `result_values()`; `MainWindow.open_wizard` le applica
(switch tab + riempimento campi), così la UI principale riflette le selezioni.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QLabel, QComboBox, QPlainTextEdit, QDoubleSpinBox,
)

from ui.i18n_manager import tr


# id stabile → metadati obiettivo.
#   cat        : categoria di instradamento (video|image|3d|music|voice|film)
#   image      : serve un'immagine di partenza
#   audio_mode : (solo video) native|clone|user
#   needs_prompt / needs_length : pertinenza dei campi nella pagina 2
#   i18n       : chiave etichetta tradotta
GOALS = {
    "t2v":       {"cat": "video", "image": False, "audio_mode": "native",
                  "needs_prompt": True,  "needs_length": True,  "i18n": "wizard.goal.t2v"},
    "i2v":       {"cat": "video", "image": True,  "audio_mode": "native",
                  "needs_prompt": True,  "needs_length": True,  "i18n": "wizard.goal.i2v"},
    "i2v_voice": {"cat": "video", "image": True,  "audio_mode": "clone",
                  "needs_prompt": True,  "needs_length": True,  "i18n": "wizard.goal.i2v_voice"},
    "a2v":       {"cat": "video", "image": False, "audio_mode": "user",
                  "needs_prompt": True,  "needs_length": True,  "i18n": "wizard.goal.a2v"},
    "t2i":       {"cat": "image", "image": False, "audio_mode": "native",
                  "needs_prompt": True,  "needs_length": False, "i18n": "wizard.goal.t2i"},
    "img3d":     {"cat": "3d",    "image": True,  "audio_mode": "native",
                  "needs_prompt": False, "needs_length": False, "i18n": "wizard.goal.img3d"},
    "music":     {"cat": "music", "image": False, "audio_mode": "native",
                  "needs_prompt": True,  "needs_length": True,  "i18n": "wizard.goal.music"},
    "voice":     {"cat": "voice", "image": False, "audio_mode": "native",
                  "needs_prompt": True,  "needs_length": False, "i18n": "wizard.goal.voice"},
    "film":      {"cat": "film",  "image": False, "audio_mode": "native",
                  "needs_prompt": True,  "needs_length": False, "i18n": "wizard.goal.film"},
}


class _GoalPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle(tr("wizard.goal_title"))
        self.setSubTitle(tr("wizard.goal_subtitle",
                            "Scegli cosa vuoi creare: il wizard prepara la scheda giusta."))
        lay = QVBoxLayout(self)
        self.goal = QComboBox()
        for gid, meta in GOALS.items():
            self.goal.addItem(tr(meta["i18n"]), gid)        # testo tradotto, dato = id stabile
        lay.addWidget(QLabel(tr("wizard.goal_label"))); lay.addWidget(self.goal)
        self.nsfw = QComboBox(); self.nsfw.addItems(["SFW", "NSFW"])
        lay.addWidget(QLabel(tr("wizard.content_label"))); lay.addWidget(self.nsfw)

        # Descrizione contestuale dell'obiettivo selezionato.
        self.desc = QLabel(); self.desc.setWordWrap(True)
        self.desc.setStyleSheet("color: #888; font-size: 11px; padding: 6px 2px;")
        lay.addWidget(self.desc)
        self.goal.currentIndexChanged.connect(self._update_desc)
        self._update_desc()

    def current_goal_id(self) -> str:
        return self.goal.currentData() or "t2v"

    def _update_desc(self):
        gid = self.current_goal_id()
        self.desc.setText(tr(f"wizard.goal.desc.{gid}", ""))


class _PromptPage(QWizardPage):
    def __init__(self, goal_page: _GoalPage):
        super().__init__()
        self._goal_page = goal_page
        self.setTitle(tr("wizard.prompt_title"))
        lay = QVBoxLayout(self)
        self.hint = QLabel(); self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: #74c0fc; font-size: 11px; padding: 2px;")
        lay.addWidget(self.hint)
        self.prompt = QPlainTextEdit(); self.prompt.setPlaceholderText(tr("wizard.prompt_placeholder"))
        lay.addWidget(self.prompt)
        self.length_lbl = QLabel(tr("wizard.length_label"))
        self.length = QDoubleSpinBox(); self.length.setRange(1, 60); self.length.setValue(8); self.length.setSuffix(" s")
        lay.addWidget(self.length_lbl); lay.addWidget(self.length)

    def initializePage(self):
        """Adatta la pagina all'obiettivo scelto (prompt/durata pertinenti o nascosti)."""
        gid = self._goal_page.current_goal_id()
        meta = GOALS.get(gid, {})
        needs_prompt = meta.get("needs_prompt", True)
        needs_length = meta.get("needs_length", False)
        self.prompt.setVisible(needs_prompt)
        self.length.setVisible(needs_length)
        self.length_lbl.setVisible(needs_length)
        # Suggerimento contestuale su dove andrà a finire la scelta.
        self.hint.setText(tr(f"wizard.prompt.hint.{meta.get('cat', 'video')}", ""))


class SetupWizard(QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("wizard.window_title"))
        self.goal_page = _GoalPage()
        self.prompt_page = _PromptPage(self.goal_page)
        self.addPage(self.goal_page); self.addPage(self.prompt_page)

    def result_values(self) -> dict:
        gid = self.goal_page.current_goal_id()
        meta = dict(GOALS.get(gid, GOALS["t2v"]))
        return {
            "goal": gid,
            "goal_label": self.goal_page.goal.currentText(),
            "category": meta["cat"],
            "nsfw_mode": "nsfw" if self.goal_page.nsfw.currentText() == "NSFW" else "sfw",
            "needs_image": meta["image"],
            "audio_mode": meta["audio_mode"],
            "prompt_positive": self.prompt_page.prompt.toPlainText().strip(),
            "length_s": self.prompt_page.length.value(),
        }
