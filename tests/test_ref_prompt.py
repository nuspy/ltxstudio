"""Prompt con riferimenti: completamento digitando «@» ed evidenziazione dei token (offscreen)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QTextCursor  # noqa: E402
from PySide6.QtWidgets import QPlainTextEdit  # noqa: E402

from ui.widgets import ref_prompt  # noqa: E402


def _edit(qtbot, names):
    e = QPlainTextEdit(); qtbot.addWidget(e); e.resize(400, 100); e.show()
    h = ref_prompt.install(e, lambda: set(names))
    return e, h


def test_at_apre_l_elenco_dei_nomi(qtbot):
    e, h = _edit(qtbot, {("image", 1), ("image", 3)})
    qtbot.keyClicks(e, "la tartaruga in @")
    assert h.completer.popup().isVisible(), "digitando @ deve comparire l'elenco"
    assert h.model.stringList() == ["@Foto_1", "@Foto_3"]
    qtbot.keyClicks(e, "Fo")
    assert h.completer.completionCount() == 2
    qtbot.keyClicks(e, "to_3")
    assert h.completer.completionCount() == 1


def test_inserimento_sostituisce_cio_che_e_stato_digitato(qtbot):
    e, h = _edit(qtbot, {("image", 1), ("image", 2)})
    qtbot.keyClicks(e, "ciao @Fo")
    h._insert("@Foto_2")
    assert e.toPlainText() == "ciao @Foto_2 ", e.toPlainText()


def test_invio_sceglie_il_primo(qtbot):
    e, h = _edit(qtbot, {("image", 1), ("image", 2)})
    qtbot.keyClicks(e, "@")
    qtbot.keyClick(e, Qt.Key_Return)
    assert e.toPlainText().startswith("@Foto_1"), e.toPlainText()


def test_nessun_elenco_senza_riferimenti_o_senza_at(qtbot):
    e, h = _edit(qtbot, set())
    qtbot.keyClicks(e, "mail @")
    assert not h.completer.popup().isVisible()
    e2, h2 = _edit(qtbot, {("image", 1)})
    qtbot.keyClicks(e2, "niente chiocciola")
    assert not h2.completer.popup().isVisible()


def _colors(e):
    out = []
    for fr in e.document().firstBlock().layout().formats():
        out.append((fr.start, fr.length, fr.format.foreground().color().name()))
    return out


def test_token_validi_blu_inesistenti_rossi(qtbot):
    e, h = _edit(qtbot, {("image", 1)})
    e.setPlainText("@Foto_1 mangia @Foto_2")
    cols = _colors(e)
    assert len(cols) == 2, cols
    assert cols[0][2] == "#7fd1ff" and cols[1][2] == "#ff6b6b", cols
    h.highlighter.rehighlight()                                   # nessun ciclo infinito
