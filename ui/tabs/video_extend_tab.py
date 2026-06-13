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

from ui.i18n_manager import tr


class VideoExtendTab(QWidget):
    request_extend = Signal()

    def __init__(self):
        super().__init__()
        self._video: str | None = None
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QScrollArea.NoFrame)
        host = QWidget(); root = QVBoxLayout(host); sc.setWidget(host); outer.addWidget(sc)

        srow = QHBoxLayout()
        self.video_lbl = QLabel(tr("ext.no_video.lbl"))
        self.video_lbl.setWordWrap(True)
        pick = QPushButton(tr("ext.pick.btn")); pick.clicked.connect(self._pick)
        srow.addWidget(self.video_lbl, 1); srow.addWidget(pick)
        root.addLayout(srow)

        root.addWidget(QLabel(tr("ext.prompt.lbl")))
        self.prompt = QPlainTextEdit(); self.prompt.setMaximumHeight(70)
        self.prompt.setPlaceholderText(tr("ext.prompt.ph"))
        root.addWidget(self.prompt)

        box = QGroupBox(tr("ext.group")); f = QFormLayout(box)
        self.add_seconds = QDoubleSpinBox(); self.add_seconds.setRange(1, 3600); self.add_seconds.setValue(20.0)
        self.add_seconds.setSuffix(" s da aggiungere")
        self.add_seconds.setToolTip(tr("ext.duration.tt"))
        self.add_seconds.valueChanged.connect(self._update_seg_estimate)
        f.addRow(tr("ext.duration_lbl"), self.add_seconds)
        self.seg_seconds = QDoubleSpinBox(); self.seg_seconds.setRange(2, 20); self.seg_seconds.setValue(18.0)
        self.seg_seconds.setSuffix(" s per spezzone")
        self.seg_seconds.setToolTip(tr("ext.seg.tt"))
        self.seg_seconds.valueChanged.connect(self._update_seg_estimate)
        f.addRow(tr("ext.seg_lbl"), self.seg_seconds)
        self.seg_estimate = QLabel("")
        self.seg_estimate.setStyleSheet("color:#9aa1b0;")
        f.addRow("", self.seg_estimate)
        self.overlap = QSpinBox(); self.overlap.setRange(0, 48); self.overlap.setValue(8)
        self.overlap.setToolTip(tr("ext.overlap.tt"))
        f.addRow(tr("ext.overlap_lbl"), self.overlap)
        self.context = QSpinBox(); self.context.setRange(8, 64); self.context.setValue(16)
        self.context.setToolTip(tr("ext.context.tt"))
        f.addRow(tr("ext.context_lbl"), self.context)
        self.steps = QSpinBox(); self.steps.setRange(1, 60); self.steps.setValue(8)
        f.addRow(tr("ext.steps_lbl"), self.steps)
        root.addWidget(box)

        self.run_btn = QPushButton(tr("ext.run.btn"))
        self.run_btn.setObjectName("generate"); self.run_btn.clicked.connect(lambda: self.request_extend.emit())
        root.addWidget(self.run_btn)
        note = QLabel(tr("ext.note.lbl"))
        note.setWordWrap(True); note.setStyleSheet("color:#888;"); note.setTextFormat(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.RichText)
        root.addWidget(note); root.addStretch(1)
        self._update_seg_estimate()

    # ---- API ----
    def set_video(self, path: str):
        self._video = path
        self.video_lbl.setText(f"🎬 {Path(path).name}")

    def _update_seg_estimate(self, *_):
        import math
        add = self.add_seconds.value()
        if add <= 20.0:
            self.seg_estimate.setText(f"1 spezzone (passata unica, {add:.0f}s ≤ 20s → niente giunzioni)")
            return
        seg = max(2.0, self.seg_seconds.value())
        n = max(2, math.ceil(add / seg))
        warn = "  ⚠ molti spezzoni: tempo lungo" if n > 30 else ""
        self.seg_estimate.setText(f"≈ {n} spezzoni da {seg:.0f}s con sovrapposizione{warn}")

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
