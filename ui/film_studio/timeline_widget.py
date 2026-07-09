"""#53 Timeline visuale del montaggio: QGraphicsScene con clip VIDEO trascinabili (riordino) +
tracce AUDIO (dialoghi/musica/ambient) come barre proporzionali ai tempi.

Timeline PRO:
- **Marker**: doppio-click sul righello aggiunge un marker (linea + etichetta); tasto destro → rimuovi.
  Emette `markers_changed([t,...])` per la persistenza a monte (project.config).
- **Ripple delete**: tasto destro su una clip video → «Elimina (ripple)»: la clip esce dalla sequenza e
  le successive scorrono a sinistra (chiude il buco). Emette `ripple_deleted(shot_id)` (NON distrugge lo shot,
  è un'esclusione dal master, reversibile a monte).
- **Waveform**: le barre audio disegnano l'inviluppo dei picchi del wav (lettura leggera, downsampled).

Emette `reordered` con il nuovo ordine degli shot_id al drop, e `clip_clicked` alla selezione.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen, QPolygonF
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import (QGraphicsLineItem, QGraphicsRectItem, QGraphicsScene,
                               QGraphicsSimpleTextItem, QGraphicsView, QMenu, QVBoxLayout, QWidget)

from ui.i18n_manager import tr

_PPS = 24.0            # pixel per secondo
_ROW_H = 40
_X0 = 70.0            # offset x della prima clip (dopo le etichette traccia)
_TRACKS = ["video", "dialoghi", "musica", "ambient"]
_COL = {"video": "#3a6ea5", "dialoghi": "#4a8", "musica": "#a58", "ambient": "#876"}


def _wav_peaks(path: str, n: int = 80) -> list[float]:
    """Inviluppo di picchi normalizzato (0..1) del wav, downsampled a ~n punti. [] se illeggibile.
    Lettura leggera (un canale, mono-max per blocco) senza dipendenze audio esterne."""
    try:
        import array
        import wave
        with wave.open(path, "rb") as w:
            ch, sw, fr, nf = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            if nf <= 0 or sw != 2:                            # supporto solo PCM 16-bit (il caso dei nostri wav)
                return []
            raw = w.readframes(min(nf, fr * 120))            # al più 120s per non caricare interi film
        a = array.array("h")
        a.frombytes(raw)
        if ch > 1:
            a = a[::ch]                                      # tieni il primo canale
        if not len(a):
            return []
        step = max(1, len(a) // n)
        peaks = []
        for i in range(0, len(a), step):
            chunk = a[i:i + step]
            peaks.append(max(abs(x) for x in chunk) if chunk else 0)
        m = max(peaks) or 1
        return [p / m for p in peaks]
    except Exception:                                        # noqa: BLE001
        return []


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

    def contextMenuEvent(self, ev):           # noqa: N802 — ripple delete sulle clip video
        if self._track != "video":
            return
        m = QMenu()
        act = m.addAction(tr("fs.tl.ripple", "Elimina (ripple)"))
        if m.exec(ev.screenPos()) is act:
            self._owner.ripple_deleted.emit(self._shot_id)


class _TimelineView(QGraphicsView):
    """View che intercetta il doppio-click sul righello per aggiungere un marker."""

    def __init__(self, scene, owner):
        super().__init__(scene)
        self._owner = owner

    def mouseDoubleClickEvent(self, ev):      # noqa: N802
        p = self.mapToScene(ev.pos())
        if p.y() <= 12:                        # doppio-click nella fascia righello (in alto) = nuovo marker
            t = max(0.0, (p.x() - _X0) / _PPS)
            self._owner.add_marker(round(t, 2))
        else:
            super().mouseDoubleClickEvent(ev)


class TimelineWidget(QWidget):
    reordered = Signal(list)                   # [shot_id,...] nel nuovo ordine
    clip_clicked = Signal(int)
    ripple_deleted = Signal(int)               # shot_id escluso via ripple delete
    markers_changed = Signal(list)             # [t_seconds,...]

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0)
        self.scene = QGraphicsScene()
        self.view = _TimelineView(self.scene, self)
        self.view.setMinimumHeight(_ROW_H * len(_TRACKS) + 42)
        v.addWidget(self.view)
        self._clips = []
        self._markers = []                     # [t_seconds]
        self._shots = []
        self._audio = []

    def set_data(self, shots: list[dict], audio: list[dict] | None = None, markers: list[float] | None = None):
        """shots: [{id, label, duration_s}], audio: [{track, start_s, dur_s, label, path?}], markers:[t]."""
        self._shots = shots or []
        self._audio = audio or []
        if markers is not None:
            self._markers = [float(m) for m in markers]
        self._redraw()

    def add_marker(self, t: float):
        self._markers.append(float(t)); self._markers.sort()
        self._redraw(); self.markers_changed.emit(list(self._markers))

    def _remove_marker(self, t: float):
        self._markers = [m for m in self._markers if abs(m - t) > 1e-3]
        self._redraw(); self.markers_changed.emit(list(self._markers))

    def _redraw(self):
        self.scene.clear(); self._clips = []
        # etichette tracce
        for i, name in enumerate(_TRACKS):
            lbl = QGraphicsSimpleTextItem(tr(f"fs.tl.{name}", name)); lbl.setPos(2, 12 + i * _ROW_H + 2)
            lbl.setBrush(QBrush(QColor("#aaa"))); self.scene.addItem(lbl)
        # traccia video: clip in sequenza
        x = _X0
        for sh in self._shots:
            w = max(20.0, float(sh.get("duration_s") or 3) * _PPS)
            c = _Clip(x, 12 + 0 * _ROW_H + 2, w, _ROW_H - 6, sh.get("label", ""), sh["id"], "video", self)
            self.scene.addItem(c); self._clips.append(c)
            x += w + 2
        total_w = x
        # tracce audio: barre ai loro start + waveform
        for a in self._audio:
            ti = _TRACKS.index(a["track"]) if a.get("track") in _TRACKS else 1
            ax = _X0 + float(a.get("start_s", 0)) * _PPS
            aw = max(10.0, float(a.get("dur_s", 1)) * _PPS)
            ay = 12 + ti * _ROW_H + 2
            ah = _ROW_H - 6
            bar = _Clip(ax, ay, aw, ah, a.get("label", ""), a.get("shot_id", 0), a["track"], self)
            bar.setFlag(QGraphicsRectItem.ItemIsMovable, False)
            self.scene.addItem(bar)
            self._draw_waveform(a.get("path", ""), ax, ay, aw, ah)
            total_w = max(total_w, ax + aw)
        # marker: linee verticali con etichetta tempo, cliccabili (doppio-click sul righello per aggiungere)
        self._draw_markers(total_w)
        self.scene.setSceneRect(QRectF(0, 0, max(total_w, 400), 12 + _ROW_H * len(_TRACKS)))

    def _draw_waveform(self, path: str, x: float, y: float, w: float, h: float):
        if not (path and os.path.exists(path)):
            return
        peaks = _wav_peaks(path, n=max(8, int(w / 3)))
        if not peaks:
            return
        mid = y + h / 2.0
        poly = QPolygonF()
        for i, p in enumerate(peaks):
            px = x + (i / max(1, len(peaks) - 1)) * w
            poly.append(QPointF(px, mid - p * (h / 2.0 - 2)))
        for i in range(len(peaks) - 1, -1, -1):
            px = x + (i / max(1, len(peaks) - 1)) * w
            poly.append(QPointF(px, mid + peaks[i] * (h / 2.0 - 2)))
        item = self.scene.addPolygon(poly, QPen(Qt.NoPen), QBrush(QColor(255, 255, 255, 90)))
        item.setZValue(1)

    def _draw_markers(self, total_w: float):
        h = 12 + _ROW_H * len(_TRACKS)
        for t in self._markers:
            mx = _X0 + t * _PPS
            ln = QGraphicsLineItem(mx, 0, mx, h)
            ln.setPen(QPen(QColor("#e0b23a"), 1, Qt.DashLine))
            ln.setZValue(2); self.scene.addItem(ln)
            lab = QGraphicsSimpleTextItem(f"▼{t:.1f}s"); lab.setBrush(QBrush(QColor("#e0b23a")))
            lab.setPos(mx + 1, 0); lab.setZValue(3); self.scene.addItem(lab)
            # click sull'etichetta = rimuovi il marker
            lab.setFlag(QGraphicsSimpleTextItem.ItemIsSelectable, True)
            lab.setData(0, t)
        # (rimozione via doppio-click sull'etichetta gestita in mouseDoubleClick della view? no: usiamo menu)

    def contextMenuEvent(self, ev):            # noqa: N802 — menu rimozione marker vicino al click
        p = self.view.mapToScene(self.view.mapFromGlobal(ev.globalPos()))
        near = [t for t in self._markers if abs((_X0 + t * _PPS) - p.x()) < 6 and p.y() < 20]
        if near:
            m = QMenu()
            act = m.addAction(tr("fs.tl.marker_del", "Rimuovi marker {t:.1f}s").format(t=near[0]))
            if m.exec(ev.globalPos()) is act:
                self._remove_marker(near[0])

    def _commit_order(self):
        ordered = sorted([c for c in self._clips], key=lambda c: c.scenePos().x())
        self.reordered.emit([c._shot_id for c in ordered])
