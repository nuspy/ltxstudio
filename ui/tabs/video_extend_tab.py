"""Tab "Prolunga video": estensione LTX-2.3 nativa (multi-segmento).

Carica in automatico l'ultimo video generato; permette di caricarne uno esistente. Genera la
continuazione condizionando sulla coda del video. Vedi core/extend_pipeline.py.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QPushButton, QPlainTextEdit, QGroupBox, QFileDialog, QScrollArea,
)
from PySide6.QtCore import Signal


class VideoExtendTab(QWidget):
    request_extend = Signal()

    def __init__(self):
        super().__init__()
        self._video: str | None = None
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QScrollArea.NoFrame)
        host = QWidget(); root = QVBoxLayout(host); sc.setWidget(host); outer.addWidget(sc)

        srow = QHBoxLayout()
        self.video_lbl = QLabel("(nessun video — genera un video o caricane uno)")
        self.video_lbl.setWordWrap(True)
        pick = QPushButton("📁 Carica video…"); pick.clicked.connect(self._pick)
        srow.addWidget(self.video_lbl, 1); srow.addWidget(pick)
        root.addLayout(srow)

        root.addWidget(QLabel("<b>Prompt di continuazione</b> (cosa succede dopo)"))
        self.prompt = QPlainTextEdit(); self.prompt.setMaximumHeight(70)
        self.prompt.setPlaceholderText("Descrivi come prosegue la scena…")
        root.addWidget(self.prompt)

        box = QGroupBox("Estensione (LTX-2.3 — lunghezza illimitata)"); f = QFormLayout(box)
        self.add_seconds = QDoubleSpinBox(); self.add_seconds.setRange(1, 3600); self.add_seconds.setValue(20.0)
        self.add_seconds.setSuffix(" s da aggiungere")
        self.add_seconds.setToolTip("Durata totale da aggiungere. Nessun limite: viene generata a "
                                    "spezzoni concatenati (memory-bounded → lunghezza illimitata).")
        self.add_seconds.valueChanged.connect(self._update_seg_estimate)
        f.addRow("Durata totale:", self.add_seconds)
        self.seg_seconds = QDoubleSpinBox(); self.seg_seconds.setRange(2, 15); self.seg_seconds.setValue(4.0)
        self.seg_seconds.setSuffix(" s per spezzone")
        self.seg_seconds.setToolTip("Lunghezza di ogni spezzone generato in catena. ~4-5s è il "
                                    "punto dolce per LTX-2.3. Spezzoni più lunghi = meno giunzioni "
                                    "ma più pesanti.")
        self.seg_seconds.valueChanged.connect(self._update_seg_estimate)
        f.addRow("Spezzone:", self.seg_seconds)
        self.seg_estimate = QLabel("")
        self.seg_estimate.setStyleSheet("color:#9aa1b0;")
        f.addRow("", self.seg_estimate)
        self.overlap = QSpinBox(); self.overlap.setRange(0, 48); self.overlap.setValue(8)
        self.overlap.setToolTip("Fotogrammi di sovrapposizione tra spezzoni (continuità di moto).")
        f.addRow("Overlap (frame):", self.overlap)
        self.context = QSpinBox(); self.context.setRange(8, 64); self.context.setValue(16)
        self.context.setToolTip("Fotogrammi della coda usati come contesto per lo spezzone successivo "
                                "(continuità). Più alto = più ancorato al passato.")
        f.addRow("Contesto (frame):", self.context)
        self.steps = QSpinBox(); self.steps.setRange(1, 60); self.steps.setValue(8)
        f.addRow("Steps:", self.steps)
        root.addWidget(box)

        self.run_btn = QPushButton("➕ Prolunga video")
        self.run_btn.setObjectName("generate"); self.run_btn.clicked.connect(lambda: self.request_extend.emit())
        root.addWidget(self.run_btn)
        note = QLabel("Lunghezza <b>illimitata</b>: il video è esteso a spezzoni di ~4-5s concatenati, "
                      "ognuno condizionato sulla coda del precedente (decodifica per spezzone → VRAM "
                      "costante). Più secondi = più spezzoni = più tempo. Tag SSML supportati nel prompt.")
        note.setWordWrap(True); note.setStyleSheet("color:#888;"); note.setTextFormat(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.RichText)
        root.addWidget(note); root.addStretch(1)
        self._update_seg_estimate()

    # ---- API ----
    def set_video(self, path: str):
        self._video = path
        self.video_lbl.setText(f"🎬 {Path(path).name}")

    def _update_seg_estimate(self, *_):
        import math
        n = max(1, math.ceil(self.add_seconds.value() / max(2.0, self.seg_seconds.value())))
        warn = "  ⚠ molti spezzoni: tempo lungo" if n > 30 else ""
        self.seg_estimate.setText(f"≈ {n} spezzoni da {self.seg_seconds.value():.0f}s{warn}")

    def get_config(self) -> dict:
        return {
            "video": self._video,
            "prompt": self.prompt.toPlainText().strip(),
            "add_seconds": self.add_seconds.value(),
            "seg_seconds": self.seg_seconds.value(),
            "overlap_frames": self.overlap.value(),
            "context_frames": self.context.value(),
            "steps": self.steps.value(),
        }

    def _pick(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Video", "", "Video (*.mp4 *.webm *.mkv *.mov)")
        if fn:
            self.set_video(fn)
