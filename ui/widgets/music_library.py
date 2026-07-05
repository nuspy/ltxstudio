"""Libreria / cronologia delle generazioni musicali — tab Musica.

Dialog non-modale che elenca i file audio prodotti (con i metadata del sidecar): data, motore, tag, durata.
Permette di riprodurre (carica nel player), riusare i parametri (applica il sidecar), aprire la cartella o
eliminare. Nessun download: legge solo la cartella output/ e i sidecar.
"""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QLabel, QMessageBox,
)

from ui.i18n_manager import tr

_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


class MusicLibraryDialog(QDialog):
    play_requested = Signal(str)
    reuse_requested = Signal(str)

    def __init__(self, out_dir: Path, parent=None):
        super().__init__(parent)
        self._dir = Path(out_dir)
        self.setWindowTitle(tr("lib.title", "📚 Libreria musicale — cronologia"))
        self.setModal(False); self.resize(760, 440)
        self._build(); self.refresh()

    def _build(self):
        v = QVBoxLayout(self)
        self.info = QLabel(""); self.info.setStyleSheet("color:#8a93a6;")
        v.addWidget(self.info)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            tr("lib.col.name", "Nome"), tr("lib.col.date", "Data"),
            tr("lib.col.engine", "Motore"), tr("lib.col.tags", "Tag/stile"),
            tr("lib.col.dur", "Durata")])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.doubleClicked.connect(self._play)
        v.addWidget(self.table, 1)
        row = QHBoxLayout()
        for label, tip, slot in (
            (tr("lib.play", "▶ Riproduci"), tr("lib.play.tt", "Carica il brano nel player."), self._play),
            (tr("lib.reuse", "♻ Riusa parametri"), tr("lib.reuse.tt", "Applica i parametri del brano al pannello."), self._reuse),
            (tr("lib.folder", "📂 Cartella"), tr("lib.folder.tt", "Apri la cartella del brano."), self._folder),
            (tr("lib.delete", "🗑 Elimina"), tr("lib.delete.tt", "Elimina il brano e il suo sidecar."), self._delete),
            (tr("lib.refresh", "⟳ Aggiorna"), tr("lib.refresh.tt", "Rilegge la cartella output."), self.refresh),
        ):
            b = QPushButton(label); b.setToolTip(tip); b.setCursor(Qt.PointingHandCursor); b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)

    def refresh(self):
        from core.metadata_saver import MetadataSaver
        files = []
        if self._dir.exists():
            files = [f for f in self._dir.iterdir()
                     if f.is_file() and f.suffix.lower() in _AUDIO_EXTS]
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        self.table.setRowCount(0)
        import time as _t
        for f in files:
            sc = {}
            try:
                sc = MetadataSaver.load_sidecar(str(f)) or {}
            except Exception:                                # noqa: BLE001
                sc = {}
            r = self.table.rowCount(); self.table.insertRow(r)
            date = _t.strftime("%Y-%m-%d %H:%M", _t.localtime(f.stat().st_mtime))
            dur = sc.get("duration_s")
            vals = [f.name, date, sc.get("engine", "—"), (sc.get("tags") or "")[:80],
                    f"{dur:.0f} s" if isinstance(dur, (int, float)) else "—"]
            for c, val in enumerate(vals):
                it = QTableWidgetItem(str(val))
                if c == 0:
                    it.setData(Qt.UserRole, str(f))
                self.table.setItem(r, c, it)
        self.info.setText(tr("lib.count", "{n} brani in {d}").format(n=len(files), d=str(self._dir)))

    def _sel_path(self) -> str:
        r = self.table.currentRow()
        if r < 0:
            return ""
        it = self.table.item(r, 0)
        return it.data(Qt.UserRole) if it else ""

    def _play(self, *_):
        p = self._sel_path()
        if p:
            self.play_requested.emit(p)

    def _reuse(self, *_):
        p = self._sel_path()
        if p:
            self.reuse_requested.emit(p)

    def _folder(self, *_):
        if self._dir.exists():
            os.startfile(str(self._dir))

    def _delete(self, *_):
        p = self._sel_path()
        if not p:
            return
        if QMessageBox.question(self, tr("lib.delete", "🗑 Elimina"),
                                tr("lib.delete.confirm", "Eliminare '{f}' (e il suo sidecar)?").format(f=Path(p).name)
                                ) != QMessageBox.Yes:
            return
        try:
            os.remove(p)
            from core.metadata_saver import MetadataSaver
            sc = MetadataSaver.sidecar_path(p) if hasattr(MetadataSaver, "sidecar_path") else None
            if sc and Path(sc).exists():
                os.remove(sc)
        except Exception:                                    # noqa: BLE001
            pass
        self.refresh()
