"""Editor di TRACCE di movimento (motion-track IC-LoRA di LTX-2.3) — widget Qt riusabile.

Su un'immagine di sfondo l'utente disegna tracce = successioni di punti. L'ORDINE dei punti e' il TEMPO: il primo punto e' dove sta
il soggetto al primo frame, l'ultimo dove arriva; una traccia con un solo punto resta ferma. Le tracce diventano un video di controllo
(core/motion_tracks.py, porta di LTXVDrawTracks) letto dalla IC-LoRA. Il modello dati (punti, annulla, JSON nel formato del nodo
ufficiale) e' in core/motion_tracks.TrackEditorModel: qui c'e' solo la parte grafica.

Interazione (tutto raggiungibile anche da tastiera; bersagli dei punti 24 px):
- clic sull'immagine: aggiunge un punto alla traccia corrente (Maiusc+clic anche sopra un altro punto) · trascina un punto: lo sposta · tasto destro / «Annulla»: toglie
  l'ultimo punto · «Nuova traccia»: chiude la corrente · «Cancella tutto»
- tastiera (canvas con il fuoco): frecce = sposta il punto selezionato (Maiusc = 10 px), Tab/Maiusc+Tab = punto successivo/precedente (finiti i punti il fuoco esce),
  Canc = elimina il punto, N = nuova traccia, Ctrl+Z = annulla, Invio = aggiunge un punto al centro
- «Anteprima del movimento»: anima la scia come la vedra' il modello (cerchi per eta' blu -> rosso)

API: set_reference(immagine, w, h) · set_frames(n, fps) · has_tracks() · tracks_json() · load_file/save_file · clear_all()
Segnale `changed` ad ogni modifica; `message(str)` per gli avvisi (nessun dialog bloccante).
"""
from __future__ import annotations

import json

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QImage, QKeyEvent, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from core import motion_tracks as mt
from ui.i18n_manager import tr

HIT_PX = 12.0                 # raggio del bersaglio di un punto sullo schermo (24 px di diametro)
POINT_R = 7.0


def _tip() -> str:
    return tr("vedit.track.tip",
              "Clic sull'immagine: aggiunge un punto alla traccia corrente. Trascina un punto per spostarlo. Tasto destro o "
              "«Annulla»: toglie l'ultimo punto. L'ORDINE DEI PUNTI È IL TEMPO: il primo punto è dove si trova il soggetto "
              "all'inizio, l'ultimo dove arriva; una traccia con un solo punto resta ferma (utile per tenere fermo uno sfondo). "
              "Tastiera: frecce = sposta il punto selezionato (Maiusc = 10 px), Tab = punto successivo, Canc = elimina il punto, "
              "N = nuova traccia, Ctrl+Z = annulla, Invio = aggiunge un punto al centro.") + " " + tr(
        "vedit.track.tip_shift", "Maiusc+clic: aggiunge un punto anche sopra un altro punto (per chiudere un anello).")


class _Canvas(QWidget):
    changed = Signal()
    selection_changed = Signal()

    def __init__(self, model: mt.TrackEditorModel):
        super().__init__()
        self.model = model
        self.bg: QImage | None = None
        self.sel: tuple[int, int] | None = None
        self._drag = False
        self._hover: tuple[int, int] | None = None
        self.preview = False
        self._pt = 0                                 # frame corrente dell'anteprima
        self._dense: list[list[dict]] = []
        self.frames = mt.DEFAULT_FRAMES
        self.fps = mt.DEFAULT_FPS
        self.setMinimumSize(320, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setToolTip(_tip())
        self.setAccessibleName(tr("vedit.track.canvas_name", "Editor delle tracce di movimento"))
        self.setAccessibleDescription(_tip())
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    # -- geometria
    def _rect(self) -> QRectF:
        m = 8.0
        W, H = max(1, self.width() - 2 * m), max(1, self.height() - 2 * m)
        s = min(W / self.model.width, H / self.model.height)
        w, h = self.model.width * s, self.model.height * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def scale(self) -> float:
        return self._rect().width() / float(self.model.width)

    def to_screen(self, x: float, y: float) -> QPointF:
        r = self._rect(); s = self.scale()
        return QPointF(r.x() + x * s, r.y() + y * s)

    def to_ref(self, p: QPointF) -> tuple[float, float]:
        r = self._rect(); s = self.scale()
        return (p.x() - r.x()) / s, (p.y() - r.y()) / s

    def _hit(self, p: QPointF):
        x, y = self.to_ref(p)
        return self.model.hit_test(x, y, HIT_PX / self.scale())

    # -- stato
    def _emit(self):
        self._dense = []
        self.update()
        self.changed.emit()

    def set_selection(self, sel):
        self.sel = sel
        if sel:
            self.model.select(sel[0])
        self.selection_changed.emit()
        self.update()

    def dense(self) -> list[list[dict]]:
        if not self._dense:
            self._dense = self.model.dense(max(2, self.frames))
        return self._dense

    # -- mouse
    def mousePressEvent(self, e):                      # noqa: N802
        self.setFocus()
        if e.button() == Qt.RightButton:
            self.undo()
            return
        if e.button() != Qt.LeftButton:
            return
        hit = None if (e.modifiers() & Qt.ShiftModifier) else self._hit(e.position())   # Maiusc+clic: aggiunge anche sopra un punto
        if hit:
            self.set_selection(hit)
            self._drag = True
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            return
        if not self._rect().contains(e.position()):
            return
        x, y = self.to_ref(e.position())
        si, pi = self.model.add_point(x, y)
        self.sel = (si, pi)
        self.selection_changed.emit()
        self._emit()

    def mouseMoveEvent(self, e):                       # noqa: N802
        if self._drag and self.sel:
            x, y = self.to_ref(e.position())
            self.model.move_point(self.sel[0], self.sel[1], x, y)
            self._emit()
            return
        h = self._hit(e.position())
        if h != self._hover:
            self._hover = h
            self.setCursor(QCursor(Qt.OpenHandCursor if h else Qt.CrossCursor))
            self.update()

    def mouseReleaseEvent(self, e):                    # noqa: N802
        if self._drag:
            self._drag = False
            self.setCursor(QCursor(Qt.CrossCursor))
            self._emit()

    # -- tastiera
    def keyPressEvent(self, e: QKeyEvent):             # noqa: N802
        k, mods = e.key(), e.modifiers()
        step = 10 if mods & Qt.ShiftModifier else 1
        if k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down) and self.sel:
            dx = {Qt.Key_Left: -step, Qt.Key_Right: step}.get(k, 0)
            dy = {Qt.Key_Up: -step, Qt.Key_Down: step}.get(k, 0)
            self.model.nudge_point(self.sel[0], self.sel[1], dx, dy)
            self._emit()
        elif k in (Qt.Key_Delete, Qt.Key_Backspace) and self.sel:
            self.model.delete_point(*self.sel)
            self.sel = None
            self._emit()
        elif k == Qt.Key_N:
            self.model.new_track(); self.sel = None; self._emit()
        elif k == Qt.Key_Z and mods & Qt.ControlModifier:
            self.undo()
        elif k in (Qt.Key_Return, Qt.Key_Enter):
            si, pi = self.model.add_point(self.model.width / 2, self.model.height / 2)
            self.sel = (si, pi); self.selection_changed.emit(); self._emit()
        else:
            super().keyPressEvent(e)

    def focusNextPrevChild(self, nxt: bool) -> bool:    # noqa: N802
        """Tab / Maiusc+Tab scorrono i punti; finiti i punti il fuoco esce dal canvas (niente trappola da tastiera)."""
        flat = [(si, pi) for si, s in enumerate(self.model.splines) for pi in range(len(s))]
        if flat:
            if self.sel not in flat:
                self.set_selection(flat[0] if nxt else flat[-1])
                return True
            j = flat.index(self.sel) + (1 if nxt else -1)
            if 0 <= j < len(flat):
                self.set_selection(flat[j])
                return True
            self.sel = None
            self.update()
        return super().focusNextPrevChild(nxt)

    def undo(self) -> bool:
        ok = self.model.undo_last_point()
        self.sel = None
        self._emit()
        return ok

    # -- anteprima animata
    def set_preview(self, on: bool):
        self.preview = bool(on)
        if on:
            self._pt = 0
            self._timer.start(int(1000 / max(8, min(30, self.fps))))
        else:
            self._timer.stop()
        self.update()

    def _tick(self):
        self._pt = (self._pt + 1) % max(2, self.frames)
        self.update()

    # -- disegno
    def paintEvent(self, e):                           # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#16181d"))
        r = self._rect()
        if self.bg is not None:
            p.setOpacity(0.9)
            p.drawImage(r, self.bg)
            p.setOpacity(1.0)
        else:
            p.fillRect(r, QColor("#23262e"))
            p.setPen(QColor("#c9ccd4"))
            p.drawText(r, Qt.AlignCenter | Qt.TextWordWrap,
                       tr("vedit.track.no_image", "Scegli un'immagine di partenza per disegnare le tracce."))
        p.setPen(QPen(QColor("#ffffff"), 1, Qt.DashLine)); p.setBrush(Qt.NoBrush)
        p.drawRect(r)
        if self.preview:
            self._paint_preview(p)
        else:
            self._paint_tracks(p)
        if self.hasFocus():
            p.setPen(QPen(QColor("#4da3ff"), 2)); p.setBrush(Qt.NoBrush)
            p.drawRect(self.rect().adjusted(1, 1, -1, -1))
        if self.model.is_empty() and self.bg is not None:
            p.setPen(QColor("#ffffff"))
            p.fillRect(QRectF(r.x() + 8, r.y() + 8, min(r.width() - 16, 460), 26), QColor(0, 0, 0, 150))
            p.drawText(QRectF(r.x() + 14, r.y() + 8, min(r.width() - 20, 450), 26), Qt.AlignVCenter | Qt.TextWordWrap,
                       tr("vedit.track.empty", "Clicca sull'immagine per mettere il primo punto della traccia."))
        p.end()

    def _paint_tracks(self, p: QPainter):
        m = self.model
        for si, sp in enumerate(m.splines):
            if not sp:
                continue
            active = si == m.active
            col = QColor(m.color(si))
            if len(sp) >= 2:
                curve = mt.interpolate_spline([{"x": q[0], "y": q[1]} for q in sp], 80)
                path = QPainterPath(self.to_screen(curve[0]["x"], curve[0]["y"]))
                for c in curve[1:]:
                    path.lineTo(self.to_screen(c["x"], c["y"]))
                p.setPen(QPen(col, 4 if active else 2.5)); p.setBrush(Qt.NoBrush)
                p.drawPath(path)
                self._arrow(p, curve[-8] if len(curve) > 8 else curve[0], curve[-1], col)
            for pi, q in enumerate(sp):
                c = self.to_screen(q[0], q[1])
                sel = self.sel == (si, pi) or self._hover == (si, pi)
                rr = POINT_R + (2 if sel else 0)
                p.setPen(QPen(QColor("#ffffff" if active else "#cfd3da"), 2))
                p.setBrush(QBrush(col))
                if pi == 0:                                        # primo punto = inizio del tempo: quadrato
                    p.drawRect(QRectF(c.x() - rr, c.y() - rr, 2 * rr, 2 * rr))
                else:
                    p.drawEllipse(c, rr, rr)
                if sel:
                    p.setPen(QPen(QColor("#ffffff"), 2)); p.setBrush(Qt.NoBrush)
                    p.drawEllipse(c, rr + 4, rr + 4)
                if active or sel:
                    p.setPen(QColor("#ffffff"))
                    p.drawText(QPointF(c.x() + rr + 3, c.y() - rr - 2), str(pi + 1))

    def _arrow(self, p: QPainter, a: dict, b: dict, col: QColor):
        pa, pb = self.to_screen(a["x"], a["y"]), self.to_screen(b["x"], b["y"])
        dx, dy = pb.x() - pa.x(), pb.y() - pa.y()
        n = (dx * dx + dy * dy) ** 0.5
        if n < 3:
            return
        ux, uy = dx / n, dy / n
        tip = QPointF(pb.x() + ux * 6, pb.y() + uy * 6)
        poly = QPolygonF([tip, QPointF(tip.x() - ux * 12 - uy * 6, tip.y() - uy * 12 + ux * 6),
                          QPointF(tip.x() - ux * 12 + uy * 6, tip.y() - uy * 12 - ux * 6)])
        p.setPen(Qt.NoPen); p.setBrush(QBrush(col))
        p.drawPolygon(poly)

    def _paint_preview(self, p: QPainter):
        """Scia come la disegna il nodo (cerchi per eta' blu->verde->giallo->rosso, raggio 2..8 a 1080 p di riferimento)."""
        dense = self.dense()
        if not dense:
            return
        t = self._pt
        lo = max(0, t - mt.MAX_TRAIL)
        s = self.scale()
        k_ref = self.model.height / 1080.0 * s          # raggio di riferimento -> pixel sullo schermo
        p.setPen(Qt.NoPen)
        for k in range(lo, t + 1):
            age = t - k
            ratio = 1.0 - age / mt.MAX_TRAIL
            r = (mt.MIN_RADIUS + (mt.MAX_RADIUS - mt.MIN_RADIUS) * ratio) * k_ref * 1.6
            col = mt.age_colors(np.array([ratio], np.float32))[0]
            p.setBrush(QBrush(QColor(int(col[0] * 255), int(col[1] * 255), int(col[2] * 255))))
            for trk in dense:
                if k < len(trk):
                    c = self.to_screen(trk[k]["x"], trk[k]["y"])
                    p.drawEllipse(c, max(1.5, r), max(1.5, r))
        p.setPen(QColor("#ffffff"))
        p.drawText(QPointF(self._rect().x() + 8, self._rect().bottom() - 8),
                   tr("vedit.track.preview_t", "frame {t}/{n}").format(t=t + 1, n=self.frames))


class TrackEditor(QWidget):
    """Barra dei comandi + canvas."""
    changed = Signal()
    message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = mt.TrackEditorModel(1024, 576)
        self.canvas = _Canvas(self.model)
        self.canvas.changed.connect(self._on_changed)
        self.canvas.selection_changed.connect(self._refresh)
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.btn_new = self._btn(tr("vedit.track.new", "Nuova traccia"), self.new_track,
                                 tr("vedit.track.new_tt", "Chiude la traccia corrente: i prossimi clic ne iniziano un'altra (un soggetto per traccia)."))
        self.btn_undo = self._btn(tr("vedit.track.undo", "Annulla"), self.canvas.undo,
                                  tr("vedit.track.undo_tt", "Toglie l'ultimo punto aggiunto (anche: tasto destro sull'immagine, Ctrl+Z)."))
        self.btn_del = self._btn(tr("vedit.track.delete", "Elimina traccia"), self.delete_track,
                                 tr("vedit.track.delete_tt", "Elimina la traccia corrente per intero."))
        self.btn_clear = self._btn(tr("vedit.track.clear", "Cancella tutto"), self.clear_all,
                                   tr("vedit.track.clear_tt", "Toglie tutte le tracce."))
        self.btn_prev = self._btn(tr("vedit.track.preview", "▶ Anteprima del movimento"), None,
                                  tr("vedit.track.preview_tt", "Anima la scia delle tracce come la vedrà il modello."))
        self.btn_prev.setCheckable(True)
        self.btn_prev.toggled.connect(self.canvas.set_preview)
        self.btn_load = self._btn(tr("vedit.track.load", "Carica tracce…"), self._pick_load,
                                  tr("vedit.track.load_tt", "Carica un JSON di tracce (anche quello di un workflow ComfyUI di Lightricks)."))
        self.btn_save = self._btn(tr("vedit.track.save", "Salva tracce…"), self._pick_save,
                                  tr("vedit.track.save_tt", "Salva le tracce in JSON (formato del nodo LTXVSparseTrackEditor)."))
        for b in (self.btn_new, self.btn_undo, self.btn_del, self.btn_clear, self.btn_prev, self.btn_load, self.btn_save):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(self.canvas, 1)
        self.status = QLabel(""); self.status.setWordWrap(True)
        v.addWidget(self.status)
        self._refresh()

    def _btn(self, text, slot, tip):
        b = QPushButton(text)
        b.setToolTip(tip); b.setAccessibleName(text)
        b.setMinimumHeight(32)
        if slot:
            b.clicked.connect(lambda *_: slot())
        return b

    # -- API
    def set_reference(self, image_path: str, width: int, height: int) -> bool:
        """Imposta immagine di sfondo (ritaglio «cover» alla risoluzione di lavoro, come il frame 0) e spazio dell'editor;
        le tracce gia' disegnate seguono l'immagine (riscalate). False se l'immagine non si legge."""
        width, height = int(width), int(height)
        img = None
        if image_path:
            try:
                from core import video_ic_prep as prep
                arr = np.ascontiguousarray(prep._load_image_cover(image_path, width, height))
                img = QImage(arr.data, width, height, width * 3, QImage.Format_RGB888).copy()
            except Exception:                                    # noqa: BLE001
                img = None
        self.model.set_size(width, height, rescale=True)
        self.canvas.bg = img
        self.canvas._emit()
        self._refresh()
        return img is not None or not image_path

    def set_frames(self, frames: int, fps: float):
        self.canvas.frames, self.canvas.fps = int(frames), float(fps)
        self.canvas._dense = []
        if self.canvas.preview:
            self.canvas.set_preview(True)
        self.canvas.update()

    def has_tracks(self) -> bool:
        return not self.model.is_empty()

    def tracks_json(self, frames: int | None = None) -> str:
        return self.model.to_json(int(frames or self.canvas.frames))

    def new_track(self):
        self.model.new_track(); self.canvas.sel = None; self.canvas._emit()

    def delete_track(self):
        self.model.delete_active(); self.canvas.sel = None; self.canvas._emit()

    def clear_all(self):
        self.model.clear(); self.canvas.sel = None; self.canvas._emit()

    def load_file(self, path: str) -> bool:
        try:
            with open(path, encoding="utf-8") as f:
                n = self.model.load(f.read(), self.canvas.frames)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as e:
            self.message.emit(tr("vedit.track.load_err", "Tracce non lette: {e}").format(e=e))
            return False
        self.canvas.sel = None
        self.canvas._emit()
        self.message.emit(tr("vedit.track.loaded", "{n} tracce caricate da {p}").format(n=n, p=path))
        return True

    def save_file(self, path: str) -> bool:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.tracks_json())
        except OSError as e:
            self.message.emit(tr("vedit.track.save_err", "Tracce non salvate: {e}").format(e=e))
            return False
        self.message.emit(tr("vedit.track.saved", "Tracce salvate: {p}").format(p=path))
        return True

    # -- interni
    def _pick_load(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("vedit.track.load", "Carica tracce…"), "", "JSON (*.json)")
        if p:
            self.load_file(p)

    def _pick_save(self):
        p, _ = QFileDialog.getSaveFileName(self, tr("vedit.track.save", "Salva tracce…"), "tracce.json", "JSON (*.json)")
        if p:
            self.save_file(p)

    def _on_changed(self):
        self._refresh()
        self.changed.emit()

    def _refresh(self):
        m = self.model
        self.status.setText(tr("vedit.track.status", "{t} tracce · {p} punti").format(
            t=len([s for s in m.splines if s]), p=m.n_points))
        self.btn_undo.setEnabled(m.n_points > 0)
        self.btn_del.setEnabled(0 <= m.active < len(m.splines))
        self.btn_clear.setEnabled(m.n_points > 0)
        self.btn_prev.setEnabled(m.n_points > 0)
        self.btn_save.setEnabled(m.n_points > 0)
