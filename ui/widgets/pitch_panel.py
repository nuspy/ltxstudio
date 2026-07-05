"""Pannello 'Correzione intonazione' (DiffPitcher) — tab Musica. SCAFFOLD (test live pendente).

Corregge l'intonazione di una voce stonata trasferendo la curva d'intonazione di una voce-guida
(template), via diffusion + WORLD + vocoder BigVGAN. Collassabile. process_requested() al click.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSpinBox, QPushButton, QLabel,
)

from ui.i18n_manager import tr
from ui.widgets.drop_widgets import DropLineEdit


class PitchPanel(QGroupBox):
    process_requested = Signal()
    want_last = Signal()

    def __init__(self):
        super().__init__(tr("pitch.title", "🎤  Correzione intonazione voce (DiffPitcher)"))
        self.setCheckable(True); self.setChecked(False)
        self._build()

    def _build(self):
        v = QVBoxLayout(self)
        note = QLabel(tr("pitch.note",
            "Corregge una voce stonata trasferendo l'intonazione da una voce-guida intonata (stessa frase)."))
        note.setWordWrap(True); note.setStyleSheet("color:#8a93a6;")
        v.addWidget(note)
        srow = QHBoxLayout()
        self.src = DropLineEdit(tr("pitch.src.ph", "voce da correggere (trascina o Sfoglia)"),
                                "Audio (*.wav *.mp3 *.flac *.ogg *.m4a)")
        self.btn_last = QPushButton(tr("pitch.last", "Ultimo brano")); self.btn_last.clicked.connect(self.want_last.emit)
        srow.addWidget(QLabel(tr("pitch.src", "Voce stonata"))); srow.addWidget(self.src, 1); srow.addWidget(self.btn_last)
        v.addLayout(srow)
        rrow = QHBoxLayout()
        self.ref = DropLineEdit(tr("pitch.ref.ph", "voce-guida intonata (stessa frase)"),
                                "Audio (*.wav *.mp3 *.flac *.ogg *.m4a)")
        rrow.addWidget(QLabel(tr("pitch.ref", "Voce-guida"))); rrow.addWidget(self.ref, 1)
        v.addLayout(rrow)
        f = QFormLayout(); f.setLabelAlignment(Qt.AlignRight)
        self.steps = QSpinBox(); self.steps.setRange(10, 100); self.steps.setValue(50)
        self.steps.setToolTip(tr("pitch.steps.tt", "Passi di diffusione. Più passi = correzione più raffinata (più lento)."))
        f.addRow(tr("pitch.steps", "Passi"), self.steps)
        self.shift = QSpinBox(); self.shift.setRange(-12, 12); self.shift.setValue(0); self.shift.setSuffix(tr("pitch.semi", " semitoni"))
        self.shift.setToolTip(tr("pitch.shift.tt", "Trasposizione dell'intonazione di riferimento in semitoni (0 = nessuna)."))
        f.addRow(tr("pitch.shift", "Trasposizione"), self.shift)
        v.addLayout(f)
        self.btn = QPushButton(tr("pitch.run", "🎤 Correggi intonazione"))
        self.btn.setCursor(Qt.PointingHandCursor); self.btn.clicked.connect(self.process_requested.emit)
        v.addWidget(self.btn)

    def set_src(self, path: str):
        if path:
            self.src.setPath(path); self.setChecked(True)

    def get(self) -> dict:
        return {"source": self.src.path(), "pitch_ref": self.ref.path(),
                "steps": self.steps.value(), "shift_semi": self.shift.value()}
