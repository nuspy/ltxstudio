"""@Reference + AssetDialog (spec §5.3): digitando @ in un campo testo si apre un popup di ricerca
asset (filtri tipo + LIKE); doppio click inserisce @Nome o @Nome#Variante. Una sola implementazione,
agganciabile a qualsiasi QPlainTextEdit con `attach(editor)`. Zero ComfyUI."""
from __future__ import annotations

from PySide6.QtCore import Qt, QObject, QEvent
from PySide6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QVBoxLayout)

from core.film_studio import db
from ui.i18n_manager import tr

_TYPES = [("", "fs.asset.all"), ("actor", "tlora.type.person"), ("prop", "tlora.type.object"),
          ("location", "tlora.type.place"), ("architecture", "fs.asset.architecture")]


class AssetDialog(QDialog):
    """Popup di selezione asset (ricerca + tipo + variante). Ritorna la stringa @Nome[#Variante]."""
    def __init__(self, parent=None, *, start_variant_of: str = ""):
        super().__init__(parent)
        self.setWindowTitle(tr("fs.assetdlg.title", "Inserisci @riferimento"))
        self.resize(520, 420)
        self.result_ref = ""
        v = QVBoxLayout(self)
        row = QHBoxLayout()
        self.search = QLineEdit(); self.search.setPlaceholderText(tr("fs.asset.search", "Cerca (LIKE)…"))
        self.search.textChanged.connect(self._refresh)
        self.ftype = QComboBox()
        for code, key in _TYPES:
            self.ftype.addItem(tr(key, code), code)
        self.ftype.currentIndexChanged.connect(self._refresh)
        row.addWidget(self.search, 1); row.addWidget(self.ftype)
        v.addLayout(row)
        self.lst = QListWidget(); self.lst.itemDoubleClicked.connect(self._choose_asset)
        v.addWidget(self.lst, 1)
        self.var_label = QLabel(tr("fs.assetdlg.variant", "Doppio click: asset · poi scegli la variante"))
        v.addWidget(self.var_label)
        self.var_lst = QListWidget(); self.var_lst.itemDoubleClicked.connect(self._choose_variant)
        v.addWidget(self.var_lst, 1)
        self._sel_asset = None
        self._refresh()
        if start_variant_of:                                # NOTA A: apri direttamente sulle varianti
            assets = [a for a in db.find_assets(start_variant_of) if a["name"].lower() == start_variant_of.lower()]
            if assets:
                self._sel_asset = assets[0]
                self._load_variants()

    def _refresh(self):
        self.lst.clear()
        self._items = db.find_assets(self.search.text().strip(), self.ftype.currentData())
        for a in self._items:
            it = QListWidgetItem(f"[{a['type']}] {a['name']}")
            self.lst.addItem(it)

    def _choose_asset(self, item):
        row = self.lst.row(item)
        if 0 <= row < len(self._items):
            self._sel_asset = self._items[row]
            self._load_variants()

    def _load_variants(self):
        self.var_lst.clear()
        a = self._sel_asset
        self.var_lst.addItem(tr("fs.assetdlg.base", "(base) — inserisci @{n}").format(n=a["name"]))
        for vrow in db.query("asset_variation", "asset_id=?", (a["id"],)):
            self.var_lst.addItem(f"#{vrow['name']}")

    def _choose_variant(self, item):
        a = self._sel_asset
        if not a:
            return
        txt = item.text()
        if txt.startswith("#"):
            self.result_ref = f"@{a['name']}#{txt[1:]}"
        else:
            self.result_ref = f"@{a['name']}"
        self.accept()


class AtReferenceFilter(QObject):
    """Event filter: intercetta '@' in un QPlainTextEdit e apre AssetDialog, inserendo il riferimento."""
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        editor.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.text() == "@":
            dlg = AssetDialog(self.editor)
            if dlg.exec() and dlg.result_ref:
                self.editor.insertPlainText(dlg.result_ref)
                return True                                 # consuma l'evento (il @ è già nel ref)
        # NOTA A: '#' dopo un @Nome → apre direttamente la sezione VARIANTI di quell'asset
        if event.type() == QEvent.KeyPress and event.text() == "#":
            txt = self.editor.toPlainText() if hasattr(self.editor, "toPlainText") else self.editor.text()
            import re
            m = re.search(r"@([A-Za-z0-9_]+)$", txt)
            if m:
                dlg = AssetDialog(self.editor, start_variant_of=m.group(1))
                if dlg.exec() and dlg.result_ref:
                    # sostituisci l'@Nome appena scritto col riferimento completo @Nome#Variante
                    new = txt[:m.start()] + dlg.result_ref
                    if hasattr(self.editor, "setPlainText"):
                        self.editor.setPlainText(new)
                    else:
                        self.editor.setText(new)
                    return True
        return False


def attach(editor) -> AtReferenceFilter:
    """Aggancia il comportamento @Reference a un QPlainTextEdit/QLineEdit."""
    return AtReferenceFilter(editor)
