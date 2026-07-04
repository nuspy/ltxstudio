"""Pannello 'Allena LoRA' ACE-Step (#98 F4).

Dataset via drag&drop di file audio singoli e INTERE cartelle (DropList) + fallback Sfoglia. Per i file
senza gemelli <name>_prompt.txt / <name>_lyrics.txt si usano i tag/liriche globali. HeartMuLa non supporta
training (nessuno script) → il pannello mostra una nota e resta disabilitato.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QFormLayout, QWidget, QLabel, QLineEdit, QSpinBox,
    QDoubleSpinBox, QComboBox, QPushButton, QPlainTextEdit,
)

from ui.i18n_manager import tr
from ui.widgets.drop_widgets import DropList


class MusicTrainerPanel(QGroupBox):
    """Allena una LoRA ACE-Step da un dataset audio. Emette train_requested(dict)."""

    train_requested = Signal(dict)

    def __init__(self):
        super().__init__(tr("music.train.title", "🎓  Allena LoRA (ACE-Step)"))
        self.setCheckable(True); self.setChecked(False)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        self.note = QLabel(tr("music.train.hm_note",
            "ⓘ HeartMuLa non supporta l'addestramento di LoRA. Seleziona ACE-Step per allenare."))
        self.note.setWordWrap(True); self.note.setVisible(False)
        self.note.setStyleSheet("color:#e0a458; background:#2a2416; border:1px solid #4a3f1e;"
                                "border-radius:6px; padding:6px 8px;")
        root.addWidget(self.note)

        self._content = QWidget(); v = QVBoxLayout(self._content); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(8)
        root.addWidget(self._content)

        v.addWidget(QLabel(tr("music.train.dataset", "Dataset audio")))
        self.drop = DropList()
        self.drop.setToolTip(tr("music.train.drop.tt",
            "Trascina file audio o cartelle. Se accanto a un audio c'è <nome>_prompt.txt / <nome>_lyrics.txt "
            "vengono usati; altrimenti si usano i tag/liriche globali qui sotto."))
        v.addWidget(self.drop)

        f = QFormLayout(); f.setLabelAlignment(Qt.AlignRight)
        self.exp = QLineEdit("music_lora")
        self.exp.setToolTip(tr("music.train.exp.tt", "Nome della LoRA (cartella di output)."))
        f.addRow(tr("music.train.exp", "Nome LoRA"), self.exp)

        self.gtags = QLineEdit()
        self.gtags.setPlaceholderText(tr("music.train.gtags.ph", "es. melodic techno, male vocal, 124 bpm"))
        self.gtags.setToolTip(tr("music.train.gtags.tt",
            "Tag di stile applicati ai file SENZA <nome>_prompt.txt. Descrivi genere/voce/strumenti/mood/bpm."))
        f.addRow(tr("music.train.gtags", "Tag globali"), self.gtags)
        self.glyr = QPlainTextEdit(); self.glyr.setFixedHeight(56)
        self.glyr.setPlaceholderText(tr("music.train.glyr.ph", "liriche globali opzionali ([Verse]/[Chorus]…)"))
        self.glyr.setToolTip(tr("music.train.glyr.tt", "Liriche applicate ai file senza <nome>_lyrics.txt (opzionale)."))
        f.addRow(tr("music.train.glyr", "Liriche globali"), self.glyr)

        self.steps = QSpinBox(); self.steps.setRange(100, 100000); self.steps.setValue(2000); self.steps.setSingleStep(100)
        self.steps.setToolTip(tr("music.train.steps.tt", "Passi di training. Più passi = più apprendimento (e più tempo)."))
        f.addRow(tr("music.train.steps", "Max steps"), self.steps)
        self.lr = QDoubleSpinBox(); self.lr.setDecimals(6); self.lr.setRange(1e-6, 1e-2)
        self.lr.setSingleStep(1e-5); self.lr.setValue(1e-4)
        self.lr.setToolTip(tr("music.train.lr.tt", "Learning rate. 1e-4 è un buon punto di partenza."))
        f.addRow(tr("music.train.lr", "Learning rate"), self.lr)
        self.repeat = QSpinBox(); self.repeat.setRange(1, 100000); self.repeat.setValue(2000); self.repeat.setSingleStep(100)
        self.repeat.setToolTip(tr("music.train.repeat.tt",
            "Ripetizioni del dataset: per pochi file usa valori alti (es. 2000), per tanti file abbassa."))
        f.addRow(tr("music.train.repeat", "Repeat dataset"), self.repeat)
        self.precision = QComboBox(); self.precision.addItems(["bf16-mixed", "16-mixed", "32"])
        self.precision.setToolTip(tr("music.train.prec.tt", "Precisione. bf16-mixed = veloce e leggera (consigliata su GPU recenti)."))
        f.addRow(tr("music.train.prec", "Precisione"), self.precision)
        self.batch = QSpinBox(); self.batch.setRange(1, 16); self.batch.setValue(1)
        self.batch.setToolTip(tr("music.train.batch.tt", "Accumulo gradiente (batch effettivo). Alza se hai molta VRAM."))
        f.addRow(tr("music.train.batch", "Batch (accum)"), self.batch)
        v.addLayout(f)

        self.btn = QPushButton(tr("music.train.run", "🎓 Allena LoRA"))
        self.btn.setObjectName("generate"); self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.clicked.connect(lambda: self.train_requested.emit(self.get_params()))
        v.addWidget(self.btn)

    def set_engine(self, engine: str):
        is_ace = (engine == "ACE-Step")
        self._content.setEnabled(is_ace); self.note.setVisible(not is_ace)

    def get_params(self) -> dict:
        return {
            "audio_paths": self.drop.paths(),
            "exp_name": self.exp.text().strip() or "music_lora",
            "global_tags": self.gtags.text().strip(),
            "global_lyrics": self.glyr.toPlainText().strip(),
            "max_steps": self.steps.value(),
            "learning_rate": self.lr.value(),
            "repeat_count": self.repeat.value(),
            "precision": self.precision.currentText(),
            "batch": self.batch.value(),
        }
