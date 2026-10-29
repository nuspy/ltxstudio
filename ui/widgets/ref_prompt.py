"""Prompt con riferimenti nominati: completamento automatico digitando `@` + evidenziazione dei token `@image_N`.

`install(edit, names_getter)` collega a una `QPlainTextEdit`:
 - digitando `@` (e le lettere seguenti) compare l'elenco dei nomi esistenti (`@image_1`, `@image_2`…); Invio/Tab/click
   lo inserisce al posto di ciò che è stato digitato;
 - i token evidenziati: validi in azzurro, riferimenti INESISTENTI in rosso sottolineato (sono un errore da
   correggere prima di generare, non vanno ignorati in silenzio).

`names_getter()` → insieme di (kind, n) dei riferimenti esistenti. Nessuno stato proprio: si ricalcola a ogni modifica.
"""
from __future__ import annotations

import re

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QFont, QStandardItem, QStandardItemModel, QSyntaxHighlighter, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QCompleter, QPlainTextEdit

from core import image_refs as R

_AT_WORD = re.compile(r"@\w*$")


class RefHighlighter(QSyntaxHighlighter):
    def __init__(self, document, names_getter):
        super().__init__(document)
        self._names = names_getter
        self.ok_fmt = QTextCharFormat()
        self.ok_fmt.setForeground(QColor("#7fd1ff"))
        self.ok_fmt.setFontWeight(QFont.Bold)
        self.bad_fmt = QTextCharFormat()
        self.bad_fmt.setForeground(QColor("#ff6b6b"))
        self.bad_fmt.setUnderlineStyle(QTextCharFormat.SpellCheckUnderline)
        self.bad_fmt.setUnderlineColor(QColor("#ff6b6b"))

    def highlightBlock(self, text: str) -> None:
        have = set(self._names())
        for t in R.scan(text):
            if t.legacy:                                       # i vecchi <Picture N> non si evidenziano
                continue
            self.setFormat(t.start, t.end - t.start, self.ok_fmt if t.key in have else self.bad_fmt)


class _Helper(QObject):
    """Completer legato alla casella: filtro sul tasto, popup con i nomi, sostituzione del `@…` digitato."""

    def __init__(self, edit: QPlainTextEdit, names_getter, entries_getter=None):
        super().__init__(edit)
        self.edit = edit
        self.names = names_getter
        self.entries = entries_getter                # opz.: [{kind, n, label, icon}] → miniatura e nome nell'elenco
        self.model = QStandardItemModel(self)
        self.completer = QCompleter(self.model, edit)
        self.completer.setCompletionRole(Qt.UserRole)    # si confronta e si inserisce il TOKEN, si MOSTRA «token — file»
        self.completer.setWidget(edit)
        self.completer.setCompletionMode(QCompleter.PopupCompletion)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.activated[str].connect(self._insert)
        edit.installEventFilter(self)
        # NB: contentsChange e NON textChanged: l'applicazione dei formati (evidenziazione) conta come cambio del
        # documento e riattiverebbe il completamento (e, collegando rehighlight, un ciclo infinito)
        edit.document().contentsChange.connect(self._on_change)

    def _on_change(self, _pos: int, removed: int, added: int) -> None:
        if removed == added and added > 1:                          # solo formattazione
            return
        self._on_text()

    def _entries(self) -> list[dict]:
        if self.entries is not None:
            ents = list(self.entries())
        else:
            ents = [{"kind": k, "n": n} for k, n in self.names()]
        return sorted(ents, key=lambda e: (R.KINDS.index(e["kind"]), e["n"]))

    def tokens(self) -> list[str]:
        return [R.token_of(e["kind"], e["n"]) for e in self._entries()]

    def _prefix(self) -> str | None:
        cur = self.edit.textCursor()
        before = self.edit.toPlainText()[:cur.position()]
        m = _AT_WORD.search(before)
        return m.group(0) if m else None

    def _on_text(self) -> None:
        pre = self._prefix()
        ents = self._entries()
        if pre is None or not ents:
            self.completer.popup().hide()
            return
        self.model.clear()
        for e in ents:
            tok = R.token_of(e["kind"], e["n"])
            label = e.get("label") or ""
            it = QStandardItem(f"{tok}  —  {label}" if label else tok)
            it.setData(tok, Qt.UserRole)
            if e.get("icon") is not None:
                it.setIcon(e["icon"])                       # miniatura (immagini)
            if label:
                it.setToolTip(label)
            self.model.appendRow(it)
        self.completer.setCompletionPrefix(pre)
        if self.completer.completionCount() == 0:
            self.completer.popup().hide()
            return
        pop = self.completer.popup()
        pop.setCurrentIndex(self.completer.completionModel().index(0, 0))
        rect = self.edit.cursorRect()
        rect.setWidth(pop.sizeHintForColumn(0) + pop.verticalScrollBar().sizeHint().width() + 12)
        self.completer.complete(rect)

    def _insert(self, text: str) -> None:
        pre = self._prefix()
        cur = self.edit.textCursor()
        if pre:
            cur.movePosition(QTextCursor.Left, QTextCursor.KeepAnchor, len(pre))
        cur.insertText(text + " ")
        self.edit.setTextCursor(cur)

    def eventFilter(self, obj, ev):                                 # noqa: N802
        if obj is self.edit and ev.type() == QEvent.KeyPress and self.completer.popup().isVisible():
            if ev.key() in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Tab, Qt.Key_Backtab):
                idx = self.completer.popup().currentIndex()
                if idx.isValid():
                    self._insert(idx.data(Qt.UserRole))
                    self.completer.popup().hide()
                    return True
            if ev.key() == Qt.Key_Escape:
                self.completer.popup().hide()
                return True
        return False


def install(edit: QPlainTextEdit, names_getter, entries_getter=None) -> _Helper:
    """Attiva completamento + evidenziazione su `edit`. `entries_getter()` → [{kind, n, label, icon}] per mostrare
    miniatura e nome del file nell'elenco (senza, solo i token). Ritorna l'helper (figlio della casella)."""
    hl = RefHighlighter(edit.document(), names_getter)
    helper = _Helper(edit, names_getter, entries_getter)
    helper.highlighter = hl
    return helper
