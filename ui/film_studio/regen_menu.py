"""Context-menu 'Rigenera segmento' per gli editor narrativi del Film Studio (Copione/Scenografia).

Selezione testo + tasto destro → «Rigenera (simile)» / «Rigenera (cosa cambiare…)». Sostituisce SOLO la
selezione via LLM (undo nativo preservato: si usa QTextCursor.insertText, niente setPlainText che azzererebbe
la cronologia). Non blocca la UI (worker in thread). attach_regen_menu(editor) è l'unico aggancio."""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QApplication, QInputDialog

from ui.i18n_manager import tr

_SYS = ("You are a film writing assistant. Rewrite ONLY the given fragment of a screenplay/treatment. Keep the "
        "SAME language, tense and any @Name / @Name#Variant references intact. Return ONLY the rewritten "
        "fragment — no quotes, no preamble, no explanation.")


class _RegenWorker(QThread):
    done = Signal(str)

    def __init__(self, frag: str, direction: str):
        super().__init__()
        self._frag = frag
        self._dir = direction

    def run(self):
        try:
            from core import llm_service as L
            user = f"Fragment:\n{self._frag}\n\n"
            user += (f"Rewrite it: {self._dir}." if self._dir
                     else "Rewrite it with the same meaning but fresh, different wording.")
            out = L.complete(_SYS, user, temperature=0.9, max_tokens=500)
            self.done.emit((out or "").strip())
        except Exception:                                    # noqa: BLE001
            self.done.emit("")


def attach_regen_menu(editor) -> None:
    """Aggancia il context-menu di rigenerazione-segmento a un QPlainTextEdit."""
    editor.setContextMenuPolicy(Qt.CustomContextMenu)

    def _menu(pos):
        menu = editor.createStandardContextMenu()           # mantiene copia/incolla/seleziona-tutto
        cur = editor.textCursor()
        if cur.selectedText().strip():
            menu.addSeparator()
            a1 = menu.addAction(tr("fs.regen.similar", "↻ Rigenera segmento (simile)"))
            a2 = menu.addAction(tr("fs.regen.change", "↻ Rigenera segmento (cosa cambiare…)"))
            a1.triggered.connect(lambda: _run(editor, ""))
            a2.triggered.connect(lambda: _run(editor, _ask(editor)))
        menu.exec(editor.mapToGlobal(pos))

    editor.customContextMenuRequested.connect(_menu)


def _ask(editor) -> str | None:
    txt, ok = QInputDialog.getText(editor, tr("fs.regen.change", "Rigenera segmento"),
                                   tr("fs.regen.prompt", "Cosa cambiare / cosa deve avvenire:"))
    return (txt or "").strip() if ok else None


def _run(editor, direction) -> None:
    if direction is None:                                    # dialog "cosa cambiare" annullato
        return
    cur = editor.textCursor()
    frag = cur.selectedText().replace(" ", "\n")        # QTextEdit usa U+2029 come newline nella selezione
    if not frag.strip():
        return
    editor.setReadOnly(True)
    QApplication.setOverrideCursor(Qt.WaitCursor)
    w = _RegenWorker(frag, direction or "")

    def _done(newtext: str):
        QApplication.restoreOverrideCursor()
        editor.setReadOnly(False)
        if newtext:
            c = editor.textCursor()                          # sostituisce la selezione (undo nativo preservato)
            c.insertText(newtext)
        editor._regen_worker = None                          # rilascia il riferimento

    w.done.connect(_done)
    editor._regen_worker = w                                 # evita GC del thread
    w.start()
