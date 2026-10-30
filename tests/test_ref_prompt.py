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
    assert h.tokens() == ["@image_1", "@image_3"]
    qtbot.keyClicks(e, "im")
    assert h.completer.completionCount() == 2
    qtbot.keyClicks(e, "age_3")
    assert h.completer.completionCount() == 1


def test_inserimento_sostituisce_cio_che_e_stato_digitato(qtbot):
    e, h = _edit(qtbot, {("image", 1), ("image", 2)})
    qtbot.keyClicks(e, "ciao @im")
    h._insert("@image_2")
    assert e.toPlainText() == "ciao @image_2 ", e.toPlainText()


def test_invio_sceglie_il_primo(qtbot):
    e, h = _edit(qtbot, {("image", 1), ("image", 2)})
    qtbot.keyClicks(e, "@")
    qtbot.keyClick(e, Qt.Key_Return)
    assert e.toPlainText().startswith("@image_1"), e.toPlainText()


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
    e.setPlainText("@image_1 mangia @image_2")
    cols = _colors(e)
    assert len(cols) == 2, cols
    assert cols[0][2] == "#7fd1ff" and cols[1][2] == "#ff6b6b", cols
    h.highlighter.rehighlight()                                   # nessun ciclo infinito


def test_elenco_con_miniatura_nome_e_tutti_i_tipi(qtbot):
    """L'elenco inline mostra «token — file», con miniatura per le immagini; Invio inserisce solo il token. Tutti i tipi
    caricati (immagini e video) sono referenziabili; digitando «@v» restano i soli video."""
    from PySide6.QtGui import QIcon, QPixmap
    pm = QPixmap(8, 8); pm.fill(Qt.red)
    ents = [{"kind": "image", "n": 1, "label": "tartaruga.png", "icon": QIcon(pm)},
            {"kind": "video", "n": 1, "label": "corsa.mp4", "icon": None}]
    e = QPlainTextEdit(); qtbot.addWidget(e); e.resize(400, 100); e.show()
    h = ref_prompt.install(e, lambda: {("image", 1), ("video", 1)}, lambda: ents)
    qtbot.keyClicks(e, "@")
    texts = [h.model.item(i).text() for i in range(h.model.rowCount())]
    assert texts == ["@image_1  —  tartaruga.png", "@video_1  —  corsa.mp4"], texts
    assert not h.model.item(0).icon().isNull() and h.model.item(1).icon().isNull()
    qtbot.keyClicks(e, "v")
    assert h.completer.completionCount() == 1
    qtbot.keyClick(e, Qt.Key_Return)
    assert e.toPlainText() == "@video_1 ", e.toPlainText()


def test_vecchi_nomi_non_si_inseriscono_ma_si_evidenziano_validi(qtbot):
    e, h = _edit(qtbot, {("image", 1)})
    e.setPlainText("@Foto_1 e @image_1 e @Foto_9")
    cols = _colors(e)
    assert [c[2] for c in cols] == ["#7fd1ff", "#7fd1ff", "#ff6b6b"], cols
