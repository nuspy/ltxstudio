"""Tabella dataset da CLIP video esistenti per 'Train LoRA Video' (#99).

Ogni riga = un clip: nome file (sola lettura) · caption editabile · trim inizio/fine (s). Drag&drop di file
singoli E intere cartelle (espanse nei video contenuti) + pulsanti Aggiungi/Rimuovi (gesture-alternative).
UX (ui-ux-pro-max): affordance drop chiara, caption editabile inline, colonne allineate, feedback conteggio.
"""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton,
    QLabel, QDoubleSpinBox, QFileDialog, QAbstractItemView, QHeaderView,
)

from ui.i18n_manager import tr

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


class ClipDatasetWidget(QWidget):
    """Tabella di clip video con caption e trim. get_clips()→[{path, caption, start, end}]."""

    changed = Signal()

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(4)
        self.hint = QLabel(tr("clip.hint", "Trascina qui spezzoni video singoli o INTERE cartelle — poi tagga e (se vuoi) ritaglia ogni clip"))
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setStyleSheet("color:#8a93a6; border:1.5px dashed #3a3f4b; border-radius:8px; padding:12px;")
        v.addWidget(self.hint)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([
            tr("clip.col.file", "Clip"), tr("clip.col.caption", "Caption (tag/descrizione)"),
            tr("clip.col.start", "Inizio (s)"), tr("clip.col.end", "Fine (s)")])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.setMinimumHeight(160)
        v.addWidget(self.table, 1)

        row = QHBoxLayout()
        b1 = QPushButton(tr("clip.add_files", "＋ File…")); b1.clicked.connect(self._add_files)
        b2 = QPushButton(tr("clip.add_dir", "＋ Cartella…")); b2.clicked.connect(self._add_dir)
        b3 = QPushButton(tr("clip.remove", "Rimuovi")); b3.clicked.connect(self._remove_sel)
        b4 = QPushButton(tr("clip.clear", "Svuota")); b4.clicked.connect(self._clear)
        for b in (b1, b2, b3, b4):
            b.setCursor(Qt.PointingHandCursor); row.addWidget(b)
        row.addStretch()
        self.count_lbl = QLabel("0 clip"); row.addWidget(self.count_lbl)
        v.addLayout(row)
        self.setAcceptDrops(True)

    # ---------- dati ----------
    def _paths_in(self):
        return {self.table.item(r, 0).data(Qt.UserRole) for r in range(self.table.rowCount())}

    def _add_row(self, path: str):
        r = self.table.rowCount(); self.table.insertRow(r)
        it = QTableWidgetItem(os.path.basename(path)); it.setData(Qt.UserRole, path)
        it.setFlags(it.flags() & ~Qt.ItemIsEditable); it.setToolTip(path)
        self.table.setItem(r, 0, QTableWidgetItem(it))
        self.table.setItem(r, 1, QTableWidgetItem(""))       # caption editabile
        for col in (2, 3):
            sp = QDoubleSpinBox(); sp.setRange(0.0, 3600.0); sp.setDecimals(1); sp.setSuffix(" s")
            self.table.setCellWidget(r, col, sp)

    def _add_paths(self, paths):
        existing = self._paths_in()
        for p in paths:
            pp = Path(p)
            files = [pp] if pp.is_file() else (sorted(pp.rglob("*")) if pp.is_dir() else [])
            for f in files:
                if f.is_file() and f.suffix.lower() in VIDEO_EXTS and str(f) not in existing:
                    self._add_row(str(f)); existing.add(str(f))
        self.count_lbl.setText(tr("clip.count", "{n} clip").format(n=self.table.rowCount()))
        self.changed.emit()

    def _add_files(self):
        ps, _ = QFileDialog.getOpenFileNames(self, tr("clip.add_files", "＋ File…"), str(Path.home()),
                                             "Video (*.mp4 *.mov *.webm *.mkv *.avi *.m4v)")
        if ps:
            self._add_paths(ps)

    def _add_dir(self):
        d = QFileDialog.getExistingDirectory(self, tr("clip.add_dir", "＋ Cartella…"), str(Path.home()))
        if d:
            self._add_paths([d])

    def _remove_sel(self):
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)
        self.count_lbl.setText(tr("clip.count", "{n} clip").format(n=self.table.rowCount())); self.changed.emit()

    def _clear(self):
        self.table.setRowCount(0); self.count_lbl.setText(tr("clip.count", "{n} clip").format(n=0)); self.changed.emit()

    def get_clips(self) -> list[dict]:
        out = []
        for r in range(self.table.rowCount()):
            path = self.table.item(r, 0).data(Qt.UserRole)
            cap_item = self.table.item(r, 1)
            cap = cap_item.text().strip() if cap_item else ""
            start = self.table.cellWidget(r, 2).value()
            end = self.table.cellWidget(r, 3).value()
            out.append({"path": path, "caption": cap, "start": start, "end": end})
        return out

    # ---------- drag & drop ----------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.hint.setStyleSheet("color:#cdd3df; border:1.5px dashed #3a6ea5; border-radius:8px;"
                                    "padding:12px; background:#1c2430;")

    def dragLeaveEvent(self, e):
        self.hint.setStyleSheet("color:#8a93a6; border:1.5px dashed #3a3f4b; border-radius:8px; padding:12px;")

    def dropEvent(self, e):
        self.dragLeaveEvent(e)
        self._add_paths([u.toLocalFile() for u in e.mimeData().urls()])
