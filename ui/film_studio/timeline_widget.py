"""#53 Timeline visuale del montaggio: QGraphicsScene con clip VIDEO trascinabili (riordino) +
tracce AUDIO (dialoghi/musica/ambient) come barre proporzionali ai tempi. Emette `reordered` con il
nuovo ordine degli shot_id al drop, e `clip_clicked` alla selezione."""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem,
                               QGraphicsView, QVBoxLayout, QWidget)

from ui.i18n_manager import tr

_PPS = 24.0            # pixel per secondo
_ROW_H = 40
_TRACKS = ["video", "dialoghi", "musica", "ambient"]
_COL = {"video": "#3a6ea5", "dialoghi": "#4a8", "musica": "#a58", "ambient": "#876"}


class _Clip(QGraphicsRectItem):
    def __init__(self, x, y, w, h, label, shot_id, track, owner):
        super().__init__(0, 0, w, h)
        self.setPos(x, y)
        self.setBrush(QBrush(QColor(_COL.get(track, "#666"))))
        self.setPen(QPen(QColor("#111")))
        self._shot_id = shot_id
        self._track = track
        self._owner = owner
        self._y = y
        if track == "video":
            self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        t = QGraphicsSimpleTextItem(label, self)
        t.setBrush(QBrush(QColor("white"))); t.setPos(4, 4)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if self._track == "video":
            self.setY(self._y)                # blocca lo spostamento verticale
            self._owner._commit_order()
        self._owner.clip_clicked.emit(self._shot_id)


class TimelineWidget(QWidget):
    reordered = Signal(list)                   # [shot_id,...] nel nuovo ordine
    clip_clicked = Signal(int)

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0)
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene); self.view.setMinimumHeight(_ROW_H * len(_TRACKS) + 30)
        v.addWidget(self.view)
        self._clips = []

    def set_data(self, shots: list[dict], audio: list[dict] | None = None):
        """shots: [{id, label, duration_s}], audio: [{track, start_s, dur_s, label}]."""
        self.scene.clear(); self._clips = []
        # etichette tracce
        for i, name in enumerate(_TRACKS):
            lbl = QGraphicsSimpleTextItem(tr(f"fs.tl.{name}", name)); lbl.setPos(2, i * _ROW_H + 2)
            lbl.setBrush(QBrush(QColor("#aaa"))); self.scene.addItem(lbl)
        # traccia video: clip in sequenza
        x = 70.0
        for sh in shots:
            w = max(20.0, float(sh.get("duration_s") or 3) * _PPS)
            c = _Clip(x, 0 * _ROW_H + 14, w, _ROW_H - 18, sh.get("label", ""), sh["id"], "video", self)
            self.scene.addItem(c); self._clips.append(c)
            x += w + 2
        total_w = x
        # tracce audio: barre ai loro start
        for a in (audio or []):
            ti = _TRACKS.index(a["track"]) if a.get("track") in _TRACKS else 1
            ax = 70.0 + float(a.get("start_s", 0)) * _PPS
            aw = max(10.0, float(a.get("dur_s", 1)) * _PPS)
            bar = _Clip(ax, ti * _ROW_H + 14, aw, _ROW_H - 18, a.get("label", ""), a.get("shot_id", 0), a["track"], self)
            bar.setFlag(QGraphicsRectItem.ItemIsMovable, False)
            self.scene.addItem(bar)
        self.scene.setSceneRect(QRectF(0, 0, max(total_w, 400), _ROW_H * len(_TRACKS)))

    def _commit_order(self):
        ordered = sorted([c for c in self._clips], key=lambda c: c.scenePos().x())
        self.reordered.emit([c._shot_id for c in ordered])
