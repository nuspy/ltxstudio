"""Widget con drag&drop di file/cartelle + fallback 'Sfoglia' sempre visibile (#98).

UX (ui-ux-pro-max): il drag&drop non è mai l'UNICO modo — c'è sempre un pulsante Sfoglia (gesture-alternative).
Affordance chiara (bordo tratteggiato + testo), feedback visivo all'hover del drag.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton, QFileDialog,
    QListWidget, QListWidgetItem, QLabel,
)

from ui.i18n_manager import tr


class DropLineEdit(QWidget):
    """Campo path con drag&drop di UN file + pulsante Sfoglia. Emette changed(path)."""

    changed = Signal(str)

    def __init__(self, placeholder: str = "", file_filter: str = "Audio (*.wav *.mp3 *.flac *.ogg)"):
        super().__init__()
        self._filter = file_filter
        h = QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(4)
        self.edit = QLineEdit(); self.edit.setPlaceholderText(placeholder or tr("drop.ph", "trascina qui un file o Sfoglia…"))
        self.edit.setAcceptDrops(False)                      # gestiamo il drop sul contenitore
        self.edit.textChanged.connect(self.changed.emit)
        h.addWidget(self.edit, 1)
        b = QPushButton(tr("drop.browse", "Sfoglia…")); b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(self._browse)
        h.addWidget(b)
        self.setAcceptDrops(True)

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("drop.browse", "Sfoglia…"), str(Path.home()), self._filter)
        if p:
            self.setPath(p)

    def path(self) -> str:
        return self.edit.text().strip()

    def setPath(self, p: str):
        self.edit.setText(p)

    # --- drag & drop ---
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction(); self.edit.setStyleSheet("border:1px solid #3a6ea5;")

    def dragLeaveEvent(self, e):
        self.edit.setStyleSheet("")

    def dropEvent(self, e):
        self.edit.setStyleSheet("")
        urls = e.mimeData().urls()
        if urls:
            self.setPath(urls[0].toLocalFile())


class DropList(QWidget):
    """Lista di path con drag&drop di file SINGOLI e INTERE CARTELLE + pulsanti Aggiungi file / cartella.
    Le cartelle vengono espanse nei file audio contenuti. Emette changed()."""

    changed = Signal()
    AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(4)
        self.hint = QLabel(tr("drop.list_hint", "Trascina qui file audio singoli o INTERE cartelle"))
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setStyleSheet("color:#8a93a6; border:1.5px dashed #3a3f4b; border-radius:8px; padding:14px;")
        v.addWidget(self.hint)
        self.list = QListWidget(); self.list.setMinimumHeight(120)
        v.addWidget(self.list, 1)
        row = QHBoxLayout()
        b1 = QPushButton(tr("drop.add_files", "＋ File…")); b1.clicked.connect(self._add_files)
        b2 = QPushButton(tr("drop.add_dir", "＋ Cartella…")); b2.clicked.connect(self._add_dir)
        b3 = QPushButton(tr("drop.clear", "Svuota")); b3.clicked.connect(self._clear)
        for b in (b1, b2, b3):
            b.setCursor(Qt.PointingHandCursor); row.addWidget(b)
        row.addStretch()
        self.count_lbl = QLabel("0"); row.addWidget(self.count_lbl)
        v.addLayout(row)
        self.setAcceptDrops(True)

    def _add_paths(self, paths):
        existing = set(self.paths())
        for p in paths:
            pp = Path(p)
            files = [pp] if pp.is_file() else (sorted(pp.rglob("*")) if pp.is_dir() else [])
            for f in files:
                if f.is_file() and f.suffix.lower() in self.AUDIO_EXTS and str(f) not in existing:
                    self.list.addItem(QListWidgetItem(str(f))); existing.add(str(f))
        self.count_lbl.setText(str(self.list.count()))
        self.changed.emit()

    def _add_files(self):
        ps, _ = QFileDialog.getOpenFileNames(self, tr("drop.add_files", "＋ File…"), str(Path.home()),
                                             "Audio (*.wav *.mp3 *.flac *.ogg *.m4a)")
        if ps:
            self._add_paths(ps)

    def _add_dir(self):
        d = QFileDialog.getExistingDirectory(self, tr("drop.add_dir", "＋ Cartella…"), str(Path.home()))
        if d:
            self._add_paths([d])

    def _clear(self):
        self.list.clear(); self.count_lbl.setText("0"); self.changed.emit()

    def paths(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction(); self.hint.setStyleSheet(
                "color:#cdd3df; border:1.5px dashed #3a6ea5; border-radius:8px; padding:14px; background:#1c2430;")

    def dragLeaveEvent(self, e):
        self.hint.setStyleSheet("color:#8a93a6; border:1.5px dashed #3a3f4b; border-radius:8px; padding:14px;")

    def dropEvent(self, e):
        self.dragLeaveEvent(e)
        self._add_paths([u.toLocalFile() for u in e.mimeData().urls()])
