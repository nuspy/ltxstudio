"""Pannello Storyboard multi-scena per il tab Video di LTX Studio.

Definisce una sequenza ORDINATA di immagini-keyframe ancorate a tempi diversi del clip.
Il motore (template_engine.apply_storyboard) le concatena in una catena di LTXVAddLatentGuide
(come LTXSequencer del workflow di riferimento) → il modello INTERPOLA tra i keyframe producendo
un video multi-scena SENZA montaggio.

API:
  get_storyboard() -> list[{'image': path, 'second': float, 'strength': float}]  (vuota se disattivo)
  validation_errors() -> list[str]

Indipendente: importabile e istanziabile senza il resto dell'app.
"""
from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
    QDoubleSpinBox, QFileDialog, QScrollArea, QFrame,
)

from ui.i18n_manager import tr


class _SceneRow(QWidget):
    """Una scena: immagine + istante (s) + forza di ancoraggio."""

    def __init__(self, on_remove, parent=None):
        super().__init__(parent)
        self.image_path: str | None = None
        h = QHBoxLayout(self); h.setContentsMargins(0, 2, 0, 2); h.setSpacing(6)
        self.pick = QPushButton(tr("vid.sb.pick"))
        self.pick.clicked.connect(self._choose)
        self.lbl = QLabel(tr("vid.sb.no_img"))
        self.lbl.setMinimumWidth(120)
        self.lbl.setStyleSheet("color:#888;")
        self.second = QDoubleSpinBox(); self.second.setRange(0.0, 3600.0)
        self.second.setSingleStep(0.5); self.second.setSuffix(" s")
        self.second.setToolTip(tr("vid.sb.second.tt"))
        self.strength = QDoubleSpinBox(); self.strength.setRange(0.0, 1.0)
        self.strength.setSingleStep(0.05); self.strength.setValue(0.8)
        self.strength.setToolTip(tr("vid.sb.strength.tt"))
        rm = QPushButton("✕"); rm.setFixedWidth(28)
        rm.setToolTip(tr("vid.sb.remove"))
        rm.clicked.connect(lambda: on_remove(self))
        h.addWidget(self.pick); h.addWidget(self.lbl, 1)
        h.addWidget(QLabel(tr("vid.sb.at"))); h.addWidget(self.second)
        h.addWidget(QLabel(tr("vid.sb.force"))); h.addWidget(self.strength)
        h.addWidget(rm)

    def _choose(self):
        f, _ = QFileDialog.getOpenFileName(self, tr("vid.sb.pick"), "",
                                           "Immagini (*.png *.jpg *.jpeg *.webp *.bmp)")
        if f:
            self.image_path = f
            self.lbl.setText(Path(f).name)
            self.lbl.setStyleSheet("color:#ddd;")

    def value(self) -> dict | None:
        if not self.image_path:
            return None
        return {"image": self.image_path, "second": float(self.second.value()),
                "strength": float(self.strength.value())}


class StoryboardPanel(QWidget):
    """Lista di scene-keyframe. Disattivo di default (checkbox)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(4)
        self.enable = QCheckBox(tr("vid.sb.enable"))
        self.enable.setToolTip(tr("vid.sb.enable.tt"))
        self.enable.toggled.connect(self._on_toggle)
        v.addWidget(self.enable)
        self.body = QWidget(); self.body.setVisible(False)
        bv = QVBoxLayout(self.body); bv.setContentsMargins(0, 0, 0, 0); bv.setSpacing(4)
        hint = QLabel(tr("vid.sb.hint"))
        hint.setWordWrap(True); hint.setStyleSheet("color:#999;font-size:11px;")
        bv.addWidget(hint)
        # contenitore scrollabile delle righe scena
        self._rows: list[_SceneRow] = []
        self._rows_box = QVBoxLayout(); self._rows_box.setContentsMargins(0, 0, 0, 0); self._rows_box.setSpacing(0)
        rows_host = QWidget(); rows_host.setLayout(self._rows_box)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMaximumHeight(220); scroll.setWidget(rows_host)
        bv.addWidget(scroll)
        add = QPushButton(tr("vid.sb.add"))
        add.clicked.connect(lambda: self._add_row())
        bv.addWidget(add)
        v.addWidget(self.body)
        # due scene di partenza per comodità
        self._add_row(); self._add_row()

    def _on_toggle(self, on: bool):
        self.body.setVisible(on)

    def _add_row(self):
        row = _SceneRow(self._remove_row)
        self._rows.append(row)
        self._rows_box.addWidget(row)

    def _remove_row(self, row: _SceneRow):
        if row in self._rows:
            self._rows.remove(row)
            row.setParent(None)
            row.deleteLater()

    def get_storyboard(self) -> list[dict]:
        if not self.enable.isChecked():
            return []
        out = []
        for r in self._rows:
            v = r.value()
            if v:
                out.append(v)
        out.sort(key=lambda d: d["second"])   # ordine temporale
        return out

    def validation_errors(self) -> list[str]:
        if not self.enable.isChecked():
            return []
        sb = self.get_storyboard()
        errs = []
        if len(sb) < 2:
            errs.append(tr("vid.sb.err_min"))
        return errs
