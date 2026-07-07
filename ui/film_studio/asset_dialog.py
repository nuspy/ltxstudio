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
        # '@' apre il popup solo se NON c'è il completer inline (QLineEdit); sui QPlainTextEdit il completer
        # inline (_MentionCompleter) gestisce la menzione mentre si digita, non-modale.
        if (event.type() == QEvent.KeyPress and event.text() == "@"
                and not getattr(self.editor, "_mention_completer", None)):
            dlg = AssetDialog(self.editor)
            if dlg.exec() and dlg.result_ref:
                self.editor.insertPlainText(dlg.result_ref)
                return True                                 # consuma l'evento (il @ è già nel ref)
        # NOTA A: '#' dopo un @Nome → apre direttamente la sezione VARIANTI di quell'asset
        if event.type() == QEvent.KeyPress and event.text() == "#":
            is_pte = hasattr(self.editor, "toPlainText")
            if is_pte:                                        # QPlainTextEdit: usa il testo FINO al cursore
                cur = self.editor.textCursor()
                pos = cur.position()
                before = self.editor.toPlainText()[:pos]
            else:                                             # QLineEdit
                before = self.editor.text()
            import re
            m = re.search(r"@([A-Za-z0-9_]+)$", before)
            if m:
                dlg = AssetDialog(self.editor, start_variant_of=m.group(1))
                if dlg.exec() and dlg.result_ref:
                    # B12: sostituisci SOLO l'@Nome col riferimento completo @Nome#Variante — via QTextCursor
                    # (undo/cursore preservati), niente più setPlainText che azzerava tutta la cronologia
                    if is_pte:
                        from PySide6.QtGui import QTextCursor
                        c = self.editor.textCursor()
                        c.setPosition(m.start())
                        c.setPosition(pos, QTextCursor.KeepAnchor)
                        c.insertText(dlg.result_ref)
                    else:
                        self.editor.setText(before[:m.start()] + dlg.result_ref + self.editor.text()[len(before):])
                    return True
        return False


class _MentionCompleter(QObject):
    """Completer INLINE non-modale per le @menzioni in un QPlainTextEdit (stile Notion/Slack): digitando '@'
    e le lettere del nome compare un popup filtrato; Invio/click inserisce l'@Nome. Undo preservato
    (QTextCursor.insertText). Il gesto più frequente diventa fluido invece che un dialog bloccante."""
    def __init__(self, editor):
        super().__init__(editor)
        from PySide6.QtWidgets import QCompleter
        from PySide6.QtCore import QStringListModel
        self.editor = editor
        self.model = QStringListModel(self)
        self.comp = QCompleter(self.model, editor)
        self.comp.setWidget(editor)
        self.comp.setCompletionMode(QCompleter.PopupCompletion)
        self.comp.setCaseSensitivity(Qt.CaseInsensitive)
        self.comp.activated[str].connect(self._insert)
        editor.textChanged.connect(self._on_text)

    def _match(self):
        import re
        cur = self.editor.textCursor()
        before = self.editor.toPlainText()[:cur.position()]
        return re.search(r"@([A-Za-z0-9_]*)$", before)       # @ + prefisso (anche vuoto) fino al cursore

    def _on_text(self):
        m = self._match()
        if m is None:
            self.comp.popup().hide(); return
        self.model.setStringList([a["name"] for a in db.find_assets()])  # asset correnti (aggiornati)
        self.comp.setCompletionPrefix(m.group(1))
        if self.comp.completionCount() == 0:
            self.comp.popup().hide(); return
        cr = self.editor.cursorRect()
        cr.setWidth(self.comp.popup().sizeHintForColumn(0) + self.comp.popup().verticalScrollBar().sizeHint().width() + 20)
        self.comp.complete(cr)

    def _insert(self, name: str):
        from PySide6.QtGui import QTextCursor
        m = self._match()
        if m is None:
            return
        cur = self.editor.textCursor()                       # sostituisci il prefisso digitato col nome pieno
        cur.setPosition(cur.position() - len(m.group(1)), QTextCursor.KeepAnchor)
        cur.removeSelectedText()
        cur.insertText(name)
        self.comp.popup().hide()


def attach(editor):
    """Aggancia il comportamento @Reference a un QPlainTextEdit/QLineEdit.
    QPlainTextEdit → completer INLINE non-modale per @menzioni + '#' apre le varianti; QLineEdit → dialog @."""
    if hasattr(editor, "toPlainText"):
        editor._mention_completer = _MentionCompleter(editor)
    return AtReferenceFilter(editor)
