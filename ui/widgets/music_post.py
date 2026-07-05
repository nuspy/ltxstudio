"""Pannello 'Post-produzione audio' — normalizza loudness, taglia silenzi, fade, converte formato (ffmpeg).

Nessun download: opera su un file audio esistente (o l'ultimo generato) via ffmpeg. Collassabile.
Emette process_requested() al click su 'Elabora'; want_last() per riempire la sorgente con l'ultimo brano.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout, QCheckBox, QDoubleSpinBox,
    QComboBox, QPushButton, QWidget, QLabel,
)

from ui.i18n_manager import tr
from ui.widgets.drop_widgets import DropLineEdit


class MusicPostPanel(QGroupBox):
    process_requested = Signal()
    want_last = Signal()

    def __init__(self):
        super().__init__(tr("post.title", "🎚  Post-produzione audio (normalizza · trim · fade · converti)"))
        self.setCheckable(True); self.setChecked(False)      # collassato di default
        self._build()

    def _build(self):
        v = QVBoxLayout(self)
        # sorgente
        srow = QHBoxLayout()
        self.src = DropLineEdit(tr("post.src.ph", "audio da elaborare (trascina o Sfoglia)"),
                                "Audio (*.wav *.mp3 *.flac *.ogg *.m4a)")
        self.btn_last = QPushButton(tr("post.last", "Ultimo brano"))
        self.btn_last.setToolTip(tr("post.last.tt", "Usa l'ultimo brano generato come sorgente."))
        self.btn_last.clicked.connect(self.want_last.emit)
        srow.addWidget(QLabel(tr("post.src", "Sorgente"))); srow.addWidget(self.src, 1); srow.addWidget(self.btn_last)
        v.addLayout(srow)

        # operazioni
        self.normalize = QCheckBox(tr("post.normalize", "Normalizza loudness (EBU R128, -14 LUFS)"))
        self.normalize.setChecked(True)
        self.normalize.setToolTip(tr("post.normalize.tt",
            "Porta il volume percepito a un livello uniforme 'da streaming' (-14 LUFS), evitando clipping."))
        self.trim = QCheckBox(tr("post.trim", "Taglia silenzi (inizio/fine)"))
        self.trim.setToolTip(tr("post.trim.tt", "Rimuove i silenzi a inizio e fine (e i lunghi vuoti interni)."))
        v.addWidget(self.normalize); v.addWidget(self.trim)

        f = QFormLayout(); f.setLabelAlignment(Qt.AlignRight)
        self.fade_in = QDoubleSpinBox(); self.fade_in.setRange(0.0, 15.0); self.fade_in.setSingleStep(0.5)
        self.fade_in.setSuffix(" s"); self.fade_in.setDecimals(1)
        self.fade_in.setToolTip(tr("post.fadein.tt", "Dissolvenza in entrata (0 = nessuna)."))
        f.addRow(tr("post.fadein", "Fade in"), self.fade_in)
        self.fade_out = QDoubleSpinBox(); self.fade_out.setRange(0.0, 15.0); self.fade_out.setSingleStep(0.5)
        self.fade_out.setSuffix(" s"); self.fade_out.setDecimals(1)
        self.fade_out.setToolTip(tr("post.fadeout.tt", "Dissolvenza in uscita (0 = nessuna)."))
        f.addRow(tr("post.fadeout", "Fade out"), self.fade_out)
        self.fmt = QComboBox(); self.fmt.addItem(tr("post.fmt.keep", "(invariato)"), "")
        for e in ("wav", "mp3", "flac", "ogg"):
            self.fmt.addItem(e, e)
        self.fmt.setToolTip(tr("post.fmt.tt", "Formato di uscita. (invariato) = mantiene quello della sorgente."))
        f.addRow(tr("post.fmt", "Formato"), self.fmt)
        v.addLayout(f)

        self.btn = QPushButton(tr("post.run", "🎚 Elabora"))
        self.btn.setCursor(Qt.PointingHandCursor); self.btn.clicked.connect(self.process_requested.emit)
        v.addWidget(self.btn)

    def set_src(self, path: str):
        if path:
            self.src.setPath(path); self.setChecked(True)

    def get_src(self) -> str:
        return self.src.path()

    def get_opts(self) -> dict:
        return {
            "normalize": self.normalize.isChecked(), "trim_silence": self.trim.isChecked(),
            "fade_in": self.fade_in.value(), "fade_out": self.fade_out.value(),
            "format": self.fmt.currentData(),
        }
