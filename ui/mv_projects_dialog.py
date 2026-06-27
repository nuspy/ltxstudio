"""Dialog CRUD progetti videomusicali (core/mv_projects): nuovo/apri/duplica/elimina, riapre lo Studio
alla fase salvata."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QListWidget, QMessageBox, QPushButton,
                               QVBoxLayout)

from core import mv_projects as mvp
from ui.i18n_manager import tr

_PHASE_LBL = {"song": "fs.mvp.ph_song", "storyboard": "fs.mvp.ph_sb", "video": "fs.mvp.ph_vid", "done": "fs.mvp.ph_done"}


class MvProjectsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("fs.mvp.title", "Progetti videomusicali"))
        self.resize(560, 420)
        self.opened_id = None
        v = QVBoxLayout(self)
        v.addWidget(QLabel(tr("fs.mvp.list", "Progetti salvati")))
        self.lst = QListWidget(); self.lst.itemDoubleClicked.connect(lambda *_: self._open())
        v.addWidget(self.lst, 1)
        h = QHBoxLayout()
        for txt, fn in ((tr("fs.mvp.open", "📂 Apri"), self._open),
                        (tr("fs.mvp.dup", "📋 Duplica"), self._dup),
                        (tr("fs.mvp.del", "🗑 Elimina"), self._del)):
            b = QPushButton(txt); b.clicked.connect(fn); h.addWidget(b)
        h.addStretch(1)
        v.addLayout(h)
        self._refresh()

    def _refresh(self):
        self.lst.clear()
        self._projects = mvp.list_projects()
        for p in self._projects:
            ph = tr(_PHASE_LBL.get(p.get("phase", "song"), "fs.mvp.ph_song"), p.get("phase", ""))
            nsong = len(mvp.list_songs(p["id"]))
            self.lst.addItem(f"{p['name']}  ·  {ph}  ·  {nsong} canzoni")

    def _sel(self):
        r = self.lst.currentRow()
        return self._projects[r] if 0 <= r < len(self._projects) else None

    def _open(self):
        p = self._sel()
        if p:
            self.opened_id = p["id"]; self.accept()

    def _dup(self):
        p = self._sel()
        if p:
            mvp.duplicate_project(p["id"]); self._refresh()

    def _del(self):
        p = self._sel()
        if p and QMessageBox.question(self, tr("fs.mvp.del", "Elimina"),
                                      tr("fs.mvp.del_confirm", "Eliminare '{n}'?").format(n=p["name"])) == QMessageBox.Yes:
            mvp.delete_project(p["id"]); self._refresh()
