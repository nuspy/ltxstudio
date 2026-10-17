"""Widget ui/widgets/track_editor.py offscreen: clic, trascinamento, annulla, nuova traccia, tastiera, salva/carica, anteprima.
Nessuna GPU, nessuna finestra reale (Qt offscreen)."""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.widgets.track_editor import TrackEditor  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def ed(app, tmp_path):
    from PIL import Image
    img = tmp_path / "bg.png"
    Image.new("RGB", (800, 450), (60, 90, 120)).save(img)
    w = TrackEditor()
    w.resize(900, 620)
    w.show()
    assert w.set_reference(str(img), 1024, 576)
    app.processEvents()
    yield w
    w.canvas.set_preview(False)
    w.close()


def _click(ed, x, y, button=Qt.LeftButton):
    """Clic sul punto (x,y) dello spazio dell'immagine (1024x576)."""
    p = ed.canvas.to_screen(x, y)
    QTest.mouseClick(ed.canvas, button, Qt.NoModifier, QPoint(int(round(p.x())), int(round(p.y()))))


def test_click_adds_points_and_tracks(ed):
    assert not ed.has_tracks()
    _click(ed, 200, 200); _click(ed, 500, 300); _click(ed, 800, 250)
    assert ed.model.n_points == 3 and len(ed.model) == 1
    assert abs(ed.model.splines[0][1][0] - 500) < 2.5 and abs(ed.model.splines[0][1][1] - 300) < 2.5
    ed.btn_new.click()
    _click(ed, 300, 450)
    assert len(ed.model) == 2 and ed.model.splines[1][0][1] > 440
    assert "2" in ed.status.text() and "4" in ed.status.text()          # «2 tracce · 4 punti»


def test_click_outside_image_does_nothing(ed):
    p = QPoint(2, 2)                                                    # fuori dal rettangolo dell'immagine
    QTest.mouseClick(ed.canvas, Qt.LeftButton, Qt.NoModifier, p)
    assert not ed.has_tracks()


def test_drag_moves_point(ed):
    _click(ed, 200, 200); _click(ed, 500, 300)
    a = ed.canvas.to_screen(500, 300); b = ed.canvas.to_screen(600, 400)
    QTest.mousePress(ed.canvas, Qt.LeftButton, Qt.NoModifier, QPoint(int(a.x()), int(a.y())))
    QTest.mouseMove(ed.canvas, QPoint(int(b.x()), int(b.y())))
    QTest.mouseRelease(ed.canvas, Qt.LeftButton, Qt.NoModifier, QPoint(int(b.x()), int(b.y())))
    assert ed.model.n_points == 2                                       # trascinare non aggiunge punti
    x, y = ed.model.splines[0][1]
    assert abs(x - 600) < 3 and abs(y - 400) < 3


def test_hit_target_is_24px(ed):
    _click(ed, 400, 300)
    c = ed.canvas.to_screen(400, 300)
    near = QPoint(int(c.x()) + 10, int(c.y()))                          # entro 12 px: prende il punto, non ne crea un altro
    QTest.mousePress(ed.canvas, Qt.LeftButton, Qt.NoModifier, near)
    QTest.mouseRelease(ed.canvas, Qt.LeftButton, Qt.NoModifier, near)
    assert ed.model.n_points == 1 and ed.canvas.sel == (0, 0)


def test_right_click_and_button_undo(ed):
    _click(ed, 200, 200); _click(ed, 500, 300); _click(ed, 700, 100)
    _click(ed, 10, 10, Qt.RightButton)                                  # il tasto destro annulla ovunque
    assert ed.model.n_points == 2
    ed.btn_undo.click()
    assert ed.model.n_points == 1
    ed.btn_undo.click()
    assert not ed.has_tracks() and not ed.btn_undo.isEnabled()
    ed.btn_undo.click()                                                 # a vuoto: nessun errore


def test_keyboard_flow(ed):
    ed.canvas.setFocus()
    QTest.keyClick(ed.canvas, Qt.Key_Return)                            # punto al centro
    assert ed.model.splines[0][0] == [512.0, 288.0]
    QTest.keyClick(ed.canvas, Qt.Key_Right)
    QTest.keyClick(ed.canvas, Qt.Key_Down, Qt.ShiftModifier)
    assert ed.model.splines[0][0] == [513.0, 298.0]
    QTest.keyClick(ed.canvas, Qt.Key_Return); QTest.keyClick(ed.canvas, Qt.Key_Left, Qt.ShiftModifier)
    assert ed.model.n_points == 2 and ed.model.splines[0][1][0] == 502.0
    assert ed.canvas.focusNextPrevChild(False) and ed.canvas.sel == (0, 0)  # Maiusc+Tab: punto precedente
    assert ed.canvas.focusNextPrevChild(True) and ed.canvas.sel == (0, 1)   # Tab: successivo
    assert ed.canvas.focusNextPrevChild(False) and ed.canvas.sel == (0, 0)
    QTest.keyClick(ed.canvas, Qt.Key_Delete)
    assert ed.model.n_points == 1
    QTest.keyClick(ed.canvas, Qt.Key_N)
    QTest.keyClick(ed.canvas, Qt.Key_Return)
    assert len(ed.model) == 2
    QTest.keyClick(ed.canvas, Qt.Key_Z, Qt.ControlModifier)
    assert len(ed.model) == 1


def test_delete_clear_and_save_load(ed, tmp_path):
    _click(ed, 100, 100); _click(ed, 300, 200); ed.btn_new.click(); _click(ed, 700, 400)
    ed.btn_del.click()
    assert len(ed.model) == 1
    ed.btn_clear.click()
    assert not ed.has_tracks()
    _click(ed, 100, 100); _click(ed, 300, 200); _click(ed, 600, 120); ed.btn_new.click(); _click(ed, 700, 400)
    f = str(tmp_path / "tracks.json")
    msgs = []
    ed.message.connect(msgs.append)
    assert ed.save_file(f)
    d = json.load(open(f, encoding="utf-8"))
    assert len(d["points_store"]) == 2 and len(d["points_store"][0]) == 3 and d["size"] == [1024, 576]
    assert len(d["coordinates"][0]) == ed.canvas.frames                   # un punto per frame, come il nodo
    ed.btn_clear.click()
    assert ed.load_file(f) and ed.model.n_points == 4 and len(ed.model) == 2
    assert not ed.load_file(str(tmp_path / "manca.json"))
    bad = tmp_path / "bad.json"; bad.write_text("[]")
    assert not ed.load_file(str(bad)) and any("non lette" in m.lower() or "not" in m.lower() or m for m in msgs)


def test_resize_reference_keeps_tracks_in_proportion(ed, tmp_path):
    _click(ed, 512, 288)
    from PIL import Image
    img = tmp_path / "bg2.png"; Image.new("RGB", (400, 400), (10, 10, 10)).save(img)
    ed.set_reference(str(img), 2048, 1152)                               # risoluzione doppia, stesso aspetto
    x, y = ed.model.splines[0][0]
    assert abs(x - 1024) < 3 and abs(y - 576) < 3


def test_preview_animates_and_set_frames(ed, app):
    _click(ed, 100, 100); _click(ed, 900, 400)
    ed.set_frames(121, 24)
    ed.btn_prev.click()
    assert ed.canvas.preview
    t0 = ed.canvas._pt
    QTest.qWait(250)
    assert ed.canvas._pt != t0 or ed.canvas._timer.isActive()
    img = ed.canvas.grab().toImage()
    assert not img.isNull()
    ed.btn_prev.click()
    assert not ed.canvas.preview and not ed.canvas._timer.isActive()


def test_empty_state_and_no_image(app):
    w = TrackEditor(); w.resize(600, 400); w.show()
    assert not w.has_tracks() and not w.btn_save.isEnabled() and not w.btn_prev.isEnabled()
    assert w.canvas.grab().toImage().width() > 0                         # disegna anche senza immagine
    assert w.set_reference("", 960, 544) and w.model.width == 960        # solo spazio, nessuno sfondo
    assert not w.set_reference("/non/esiste.png", 960, 544)
    w.close()


def test_shift_click_adds_point_on_top_of_existing_one(ed):
    """Un clic su un punto lo afferra; Maiusc+clic ne aggiunge uno nuovo nello stesso posto (chiude un anello)."""
    _click(ed, 300, 200); _click(ed, 600, 300)
    c = ed.canvas.to_screen(300, 200)
    pt = QPoint(int(round(c.x())), int(round(c.y())))
    QTest.mouseClick(ed.canvas, Qt.LeftButton, Qt.NoModifier, pt)
    assert ed.model.n_points == 2
    QTest.mouseClick(ed.canvas, Qt.LeftButton, Qt.ShiftModifier, pt)
    assert ed.model.n_points == 3 and ed.model.splines[0][2] == pytest.approx(ed.model.splines[0][0], abs=2.5)
    assert "Maiusc" in ed.canvas.toolTip() or "Shift" in ed.canvas.toolTip()
