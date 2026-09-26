"""Editor delle battute per MiniMax H3: ogni riga = chi parla + lingua + testo.

H3 genera la voce INSIEME al video: le battute vanno nel prompt come `<d>[Lingua] testo</d>` (guida ufficiale
dei prompt). Qui l'utente le scrive riga per riga; il worker le formatta (core.minimax_h3_spec.dialogue_tag).
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget)

from ui.i18n_manager import tr
from core.minimax_h3_spec import DIALOGUE_LANGS

_LANG_KEYS = {"Italian": "it", "English": "en", "Chinese": "zh", "Spanish": "es", "French": "fr",
              "German": "de", "Portuguese": "pt", "Japanese": "ja", "Korean": "ko", "Russian": "ru",
              "Arabic": "ar"}


class _Row(QWidget):
    removed = Signal(object)

    def __init__(self, speaker: str = "", lang: str = "Italian", text: str = ""):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        self.speaker = QLineEdit(speaker)
        self.speaker.setPlaceholderText(tr("h3.dlg.speaker", "Chi parla"))
        self.speaker.setMaximumWidth(120)
        self.lang = QComboBox()
        for name in DIALOGUE_LANGS:
            self.lang.addItem(tr(f"h3.lang.{_LANG_KEYS[name]}", name), name)
        i = self.lang.findData(lang)
        self.lang.setCurrentIndex(i if i >= 0 else 0)
        self.text = QLineEdit(text)
        self.text.setPlaceholderText(tr("h3.dlg.text", "Battuta (quello che dice)"))
        rm = QPushButton("✕")
        rm.setFixedWidth(32)
        rm.setToolTip(tr("h3.dlg.remove", "Togli questa battuta"))
        rm.clicked.connect(lambda: self.removed.emit(self))
        h.addWidget(self.speaker)
        h.addWidget(self.lang)
        h.addWidget(self.text, 1)
        h.addWidget(rm)

    def value(self) -> dict:
        return {"speaker": self.speaker.text().strip(), "language": self.lang.currentData(),
                "text": self.text.text().strip()}


class H3DialogueEditor(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[_Row] = []
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self._box = QVBoxLayout()
        v.addLayout(self._box)
        self.add_btn = QPushButton(tr("h3.dlg.add", "+ Battuta"))
        self.add_btn.setToolTip(tr("h3.dlg.add.tt",
                                   "Aggiunge una battuta: H3 genera la voce insieme al video, con il labiale "
                                   "sincronizzato. Scegli la lingua di ogni battuta."))
        self.add_btn.clicked.connect(lambda: self.add_row())
        v.addWidget(self.add_btn)

    def add_row(self, speaker: str = "", lang: str = "Italian", text: str = "") -> _Row:
        r = _Row(speaker, lang, text)
        r.removed.connect(self._remove)
        r.text.textChanged.connect(self.changed)
        self._rows.append(r)
        self._box.addWidget(r)
        self.changed.emit()
        return r

    def _remove(self, row: _Row):
        if row in self._rows:
            self._rows.remove(row)
            row.setParent(None)
            row.deleteLater()
            self.changed.emit()

    def clear(self):
        for r in list(self._rows):
            self._remove(r)

    def value(self) -> list[dict]:
        return [r.value() for r in self._rows if r.value()["text"]]

    # persistenza (core/gui_state)
    def gui_get_state(self) -> dict:
        return {"lines": [r.value() for r in self._rows]}

    def gui_set_state(self, st: dict) -> None:
        self.clear()
        for d in (st or {}).get("lines") or []:
            self.add_row(d.get("speaker", ""), d.get("language", "Italian"), d.get("text", ""))
