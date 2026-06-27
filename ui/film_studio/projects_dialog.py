"""Dialog CRUD progetti AI Film Studio (core/film_studio/db): apri/duplica/elimina, con stato workflow."""
from __future__ import annotations

from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QListWidget, QMessageBox, QPushButton,
                               QVBoxLayout)

from core.film_studio import db
from ui.i18n_manager import tr


class FsProjectsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("fs.proj.title", "Progetti — AI Film Studio"))
        self.resize(560, 420)
        self.opened_id = None
        v = QVBoxLayout(self)
        v.addWidget(QLabel(tr("fs.proj.list", "Progetti salvati")))
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
        self._projects = db.query("project", order="id DESC")
        for p in self._projects:
            nsc = len(db.query("scene", "project_id=?", (p["id"],)))
            self.lst.addItem(f"{p['name']}  ·  {p.get('path_mode','')}  ·  {nsc} scene  ·  {p.get('duration_s',0):.0f}s")

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
            db.duplicate_project(p["id"]); self._refresh()

    def _del(self):
        p = self._sel()
        if p and QMessageBox.question(self, tr("fs.mvp.del", "Elimina"),
                                      tr("fs.mvp.del_confirm", "Eliminare '{n}'?").format(n=p["name"])) == QMessageBox.Yes:
            db.delete_project(p["id"]); self._refresh()
