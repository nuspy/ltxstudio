"""Wizard guidato (Fase 4) — porta l'utente da "cosa vuoi creare" alle impostazioni base.

Le scelte vengono scritte nello stato condiviso; alla fine la UI principale riflette
le selezioni. Usa lo stesso StateManager per la coerenza (conflitti auto-risolti).
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QLabel, QComboBox, QPlainTextEdit, QDoubleSpinBox,
)

from ui.i18n_manager import tr


GOALS = {
    "Text → Video": {"image": False, "audio_mode": "native"},
    "Immagine → Video": {"image": True, "audio_mode": "native"},
    "Immagine → Video + voce (lipsync)": {"image": True, "audio_mode": "clone"},
    "Audio → Video": {"image": False, "audio_mode": "user"},
}


class _GoalPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle(tr("wizard.goal_title"))
        lay = QVBoxLayout(self)
        self.goal = QComboBox(); self.goal.addItems(list(GOALS.keys()))
        lay.addWidget(QLabel(tr("wizard.goal_label"))); lay.addWidget(self.goal)
        self.nsfw = QComboBox(); self.nsfw.addItems(["SFW", "NSFW"])
        lay.addWidget(QLabel(tr("wizard.content_label"))); lay.addWidget(self.nsfw)


class _PromptPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle(tr("wizard.prompt_title"))
        lay = QVBoxLayout(self)
        self.prompt = QPlainTextEdit(); self.prompt.setPlaceholderText(tr("wizard.prompt_placeholder"))
        lay.addWidget(self.prompt)
        self.length = QDoubleSpinBox(); self.length.setRange(1, 60); self.length.setValue(8); self.length.setSuffix(" s")
        lay.addWidget(QLabel(tr("wizard.length_label"))); lay.addWidget(self.length)


class SetupWizard(QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("wizard.window_title"))
        self.goal_page = _GoalPage(); self.prompt_page = _PromptPage()
        self.addPage(self.goal_page); self.addPage(self.prompt_page)

    def result_values(self) -> dict:
        goal = self.goal_page.goal.currentText()
        cfg = dict(GOALS[goal])
        return {
            "goal": goal,
            "nsfw_mode": "nsfw" if self.goal_page.nsfw.currentText() == "NSFW" else "sfw",
            "needs_image": cfg["image"],
            "audio_mode": cfg["audio_mode"],
            "prompt_positive": self.prompt_page.prompt.toPlainText().strip(),
            "length_s": self.prompt_page.length.value(),
        }
