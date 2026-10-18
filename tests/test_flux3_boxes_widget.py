"""Pannello riquadri FLUX.3 (ui/widgets/flux3_boxes.py): disegno con il mouse (= dito), spostamento, ridimensionamento dall'angolo,
modalità di edit, formato consegnato a core.flux3_caption. Offscreen, nessun modello."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from core import flux3_caption as C
from ui.widgets.flux3_boxes import HANDLE, Flux3BoxesPanel

_app = QApplication.instance() or QApplication([])


def _pt(panel, fy, fx):
    f = panel.canvas._frame()
    return QPoint(int(f.left() + fx * f.width()), int(f.top() + fy * f.height()))


def _drag(panel, a, b):
    cv = panel.canvas
    QTest.mousePress(cv, Qt.LeftButton, pos=_pt(panel, *a))
    QTest.mouseMove(cv, _pt(panel, *b))
    QTest.mouseRelease(cv, Qt.LeftButton, pos=_pt(panel, *b))


@pytest.fixture()
def panel(tmp_path):
    p = Flux3BoxesPanel()
    p.resize(900, 600)
    p.show()
    yield p
    p.close()


def _ref(tmp_path, name, color):
    pm = QPixmap(160, 100)
    pm.fill(color)
    path = tmp_path / name
    pm.save(str(path))
    return str(path)


def test_layout_disegna_e_consegna(panel):
    assert not panel.canvas._pix and panel.cb_mode.isHidden() and panel.cb_ref.isHidden()
    panel.add_box()
    assert panel.errors()                                       # zona e descrizione mancano
    _drag(panel, (0.1, 0.1), (0.5, 0.5))
    panel.ed_desc.setPlainText("un palloncino rosso")
    panel.ed_id.setText("balloon_1"); panel.ed_id.textEdited.emit("balloon_1")
    b = panel.get_boxes()
    assert b[0]["id"] == "balloon_1" and b[0]["desc"] == "un palloncino rosso"
    y1, x1, y2, x2 = b[0]["bbox"]
    assert abs(y1 - 0.1) < 0.01 and abs(x2 - 0.5) < 0.01 and y2 > y1 and x2 > x1
    assert panel.errors() == [] and C.validate(b) == []


def test_click_senza_trascinare_non_lascia_riquadri_minuscoli(panel):
    panel.add_box()
    _drag(panel, (0.3, 0.3), (0.3, 0.3))
    assert panel.get_boxes()[0]["bbox"] is None


def test_sposta_e_ridimensiona_dall_angolo(panel):
    panel.add_box()
    _drag(panel, (0.2, 0.2), (0.4, 0.4))
    before = list(panel.canvas.boxes[0]["tgt"])
    _drag(panel, (0.3, 0.3), (0.5, 0.5))                        # dentro il riquadro: sposta di +0.2, +0.2
    moved = panel.canvas.boxes[0]["tgt"]
    assert abs(moved[0] - (before[0] + 0.2)) < 0.02 and abs((moved[2] - moved[0]) - (before[2] - before[0])) < 0.01
    y1, x1, y2, x2 = moved
    _drag(panel, (y2, x2), (y2 + 0.2, x2 + 0.2))                # angolo in basso a destra: ridimensiona
    r = panel.canvas.boxes[0]["tgt"]
    assert abs(r[0] - y1) < 0.01 and abs(r[1] - x1) < 0.01 and r[2] > y2 + 0.15


def test_bersagli_tattili_almeno_44px(panel):
    assert HANDLE >= 44
    for b in (panel.btn_add, panel.btn_del, panel.btn_clear, panel.ed_id, panel.cb_mode, panel.cb_ref, panel.cb_role):
        assert b.minimumHeight() >= 44, b


def test_edit_quattro_modalita_e_regole_dei_ruoli(panel, tmp_path):
    a, b = _ref(tmp_path, "a.png", Qt.red), _ref(tmp_path, "b.png", Qt.blue)
    panel.set_context([("Foto_1", a), ("Foto_2", b)])
    assert not panel.cb_mode.isHidden() and panel.cb_ref.count() == 2 and panel.canvas._pix is not None
    panel.add_box()                                             # nuovo: solo destinazione
    assert panel.canvas.role == "tgt" and not panel.cb_role.isEnabled()
    _drag(panel, (0.0, 0.3), (0.3, 0.6))
    panel.ed_desc.setPlainText("a straw hat")
    panel.add_box()                                             # sposta: origine poi destinazione
    panel.cb_mode.setCurrentIndex(panel.cb_mode.findData("move"))
    panel.cb_ref.setCurrentIndex(1)
    panel.cb_role.setCurrentIndex(panel.cb_role.findData("src")); panel._role_changed()
    _drag(panel, (0.5, 0.0), (0.9, 0.4))
    panel.cb_role.setCurrentIndex(panel.cb_role.findData("tgt")); panel._role_changed()
    _drag(panel, (0.5, 0.6), (0.9, 1.0))
    panel.ed_desc.setPlainText("the vase")
    panel.add_box()                                             # rimuovi: solo origine
    panel.cb_mode.setCurrentIndex(panel.cb_mode.findData("remove"))
    assert panel.canvas.role == "src" and not panel.cb_role.isEnabled()
    _drag(panel, (0.1, 0.1), (0.3, 0.3))
    panel.add_box()                                             # ancora: tracciando l'origine, la destinazione la segue
    panel.cb_mode.setCurrentIndex(panel.cb_mode.findData("anchor"))
    panel.cb_role.setCurrentIndex(panel.cb_role.findData("src")); panel._role_changed()
    _drag(panel, (0.05, 0.7), (0.3, 0.95))
    panel.ed_desc.setPlainText("the tree")
    boxes = panel.get_boxes()
    assert [x["mode"] for x in boxes] == ["new", "move", "remove", "anchor"]
    assert boxes[0]["tgt_bbox"] and boxes[0]["src_bbox"] is None
    assert boxes[1]["ref"] == 2 and boxes[1]["src_bbox"] and boxes[1]["tgt_bbox"] and boxes[1]["src_bbox"] != boxes[1]["tgt_bbox"]
    assert boxes[2]["src_bbox"] and boxes[2]["tgt_bbox"] is None
    assert boxes[3]["src_bbox"] == boxes[3]["tgt_bbox"]
    errs = C.validate(boxes, n_refs=2)
    assert errs == [], errs
    rows = {r["id"]: r for r in C.to_rows(boxes)}
    assert rows["obj_3"]["tgt_bbox"] is None and rows["obj_2"]["from"] == "ref_image_1"
    # cambiare modalità ripulisce il rettangolo che non serve
    panel.canvas.select(1)
    panel._sync_fields()
    panel.cb_mode.setCurrentIndex(panel.cb_mode.findData("new"))
    assert panel.canvas.boxes[1]["src"] is None


def test_set_boxes_ricarica_stato_e_rimuovi(panel):
    panel.set_context([], 1.0)
    panel.set_boxes([{"id": "a_1", "desc": "x", "bbox": [0.1, 0.1, 0.4, 0.4]}, {"id": "b_1", "desc": "y", "bbox": [0.5, 0.5, 0.9, 0.9]}])
    assert len(panel.get_boxes()) == 2 and panel.listw.count() == 2
    panel.canvas.select(0)
    panel.remove_selected()
    assert [b["id"] for b in panel.get_boxes()] == ["b_1"]
    panel.clear()
    assert panel.get_boxes() == [] and panel.listw.count() == 0
    panel.add_box(); panel.add_box()
    assert [b["id"] for b in panel.get_boxes()] == ["obj_1", "obj_2"]                # id univoci
