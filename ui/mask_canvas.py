"""Editor immagine stile MagicQuill (PySide6): cornice formato video (no deformazione),
foto base spostabile/zoomabile/croppabile, layer immagini importate con ordine (porta su/giù),
maschera inpaint con pennello ridimensionabile, gomma, fill, undo/redo (100 passi).

La GENERAZIONE inpaint/outpaint (FLUX Fill/Kontext via ComfyUI) e il lazzo SAM verranno
collegati separatamente; qui c'è tutta l'editing UI + export (immagine composita + maschera).
"""
from __future__ import annotations

import math
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None

from PySide6.QtCore import (
    Qt, QRectF, QPointF, QPoint, Signal, QPropertyAnimation, QEasingCurve,
    QParallelAnimationGroup,
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush, QWheelEvent, QMouseEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsLineItem,
    QPushButton, QSlider, QLabel, QFileDialog,
    QListWidget, QListWidgetItem, QComboBox, QFrame, QGraphicsOpacityEffect,
)

from core.mask_store import MaskStore
from ui.i18n_manager import tr

MAX_UNDO = 100

# strumenti: (modo, icona, chiave_titolo, chiave_descrizione)
TOOLS = [
    ("move",      "✥",  "canvas.tool_move_title",   "canvas.tool_move_desc"),
    ("brush",     "✎",  "canvas.tool_brush_title",  "canvas.tool_brush_desc"),
    ("erase",     "⌫",  "canvas.tool_erase_title",  "canvas.tool_erase_desc"),
    ("fill",      "▣",  "canvas.tool_fill_title",   "canvas.tool_fill_desc"),
    ("lasso",     "❍",  "canvas.tool_lasso_title",  "canvas.tool_lasso_desc"),
    ("sam_point", "✛",  "canvas.tool_sam_title",    "canvas.tool_sam_desc"),
]


class _ToolButton(QPushButton):
    """Bottone strumento solo-icona; all'hover mostra il tooltip grafico animato del canvas."""
    def __init__(self, canvas, tool):
        super().__init__(tool[1])
        self._canvas = canvas
        self._tool = tool
        self.setObjectName("tool")
        self.setCheckable(True)
        self.setFixedSize(46, 46)
        self.setCursor(Qt.PointingHandCursor)

    def enterEvent(self, e):
        self._canvas._show_hint(self)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._canvas._hide_hint()
        super().leaveEvent(e)


class _ToolHint(QFrame):
    """Tooltip grafico: pannello che scorre/sfuma verso destra all'hover dell'icona."""
    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("toolhint")
        self.setStyleSheet(
            "#toolhint{background:rgba(26,30,40,238);border:1px solid #5a8dee;border-radius:12px;}")
        h = QHBoxLayout(self); h.setContentsMargins(12, 9, 14, 9); h.setSpacing(11)
        self.icon = QLabel(); self.icon.setStyleSheet("font-size:22px;color:#cfe0ff;")
        col = QVBoxLayout(); col.setSpacing(1)
        self.title = QLabel(); self.title.setStyleSheet("color:#fff;font-weight:700;font-size:13px;")
        self.desc = QLabel(); self.desc.setStyleSheet("color:#aab0bd;font-size:11px;"); self.desc.setWordWrap(True)
        col.addWidget(self.title); col.addWidget(self.desc)
        h.addWidget(self.icon); h.addLayout(col, 1)
        self.setMaximumWidth(250)
        self._eff = QGraphicsOpacityEffect(self); self.setGraphicsEffect(self._eff)
        self._anim = QParallelAnimationGroup(self)
        self.hide()

    def popup(self, x: int, y: int, icon: str, title: str, desc: str):
        self.icon.setText(icon); self.title.setText(title); self.desc.setText(desc)
        self.adjustSize()
        fy = int(y - self.height() / 2)
        self.move(x - 16, fy); self.show(); self.raise_()
        self._anim.stop(); self._anim.clear()
        pa = QPropertyAnimation(self, b"pos")
        pa.setDuration(160); pa.setStartValue(QPoint(x - 16, fy)); pa.setEndValue(QPoint(x, fy))
        pa.setEasingCurve(QEasingCurve.OutCubic)
        oa = QPropertyAnimation(self._eff, b"opacity")
        oa.setDuration(160); oa.setStartValue(0.0); oa.setEndValue(1.0)
        self._anim.addAnimation(pa); self._anim.addAnimation(oa); self._anim.start()


class _View(QGraphicsView):
    def __init__(self, canvas: "MaskCanvas"):
        super().__init__(canvas.scene)
        self.canvas = canvas
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)        # serve per il cursore-pennello in hover
        self._painting = False
        self._lassoing = False
        self._transforming = False

    def wheelEvent(self, e: QWheelEvent):
        up = e.angleDelta().y() > 0
        if self.canvas.mode in ("brush", "erase"):
            self.canvas.set_brush(self.canvas.brush_size + (5 if up else -5))  # rotella = dim. pennello
            self.canvas.update_brush_cursor(self.mapToScene(e.position().toPoint()))
            return
        f = 1.15 if up else 1/1.15        # rotella = ZOOM VISTA (frame + immagine insieme, coerente)
        self.scale(f, f)
        self.canvas._update_handles()

    def mousePressEvent(self, e: QMouseEvent):
        m = self.canvas.mode
        if m in ("brush", "erase") and e.button() == Qt.LeftButton:
            self.canvas.push_undo()
            self._painting = True
            self.canvas.paint_at(self.mapToScene(e.pos()))
        elif m == "fill" and e.button() == Qt.LeftButton:
            self.canvas.push_undo(); self.canvas.fill_at(self.mapToScene(e.pos()))
        elif m == "lasso" and e.button() == Qt.LeftButton:
            self._lassoing = True
            self.canvas.lasso_begin(self.mapToScene(e.pos()))
        elif m == "sam_point" and e.button() == Qt.LeftButton:
            self.canvas.sam_point_at(self.mapToScene(e.pos()))
        elif m == "move" and e.button() == Qt.LeftButton:
            role = self.canvas.handle_at(self.mapToScene(e.pos()))
            if role:                                  # maniglia: ridimensiona o ruota
                self.canvas.begin_transform(role, self.mapToScene(e.pos()))
                self._transforming = True
            else:
                super().mousePressEvent(e)            # corpo immagine: sposta (ItemIsMovable)
        else:
            super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QMouseEvent):
        sp = self.mapToScene(e.pos())
        self.canvas.update_brush_cursor(sp)           # cursore-cerchio del pennello
        if self._painting:
            self.canvas.paint_at(sp)
        elif self._lassoing:
            self.canvas.lasso_drag(sp)
        elif self._transforming:
            self.canvas.update_transform(sp)
        else:
            super().mouseMoveEvent(e)
            if self.canvas.mode == "move":
                self.canvas._update_handles()         # le maniglie seguono l'immagine spostata

    def mouseReleaseEvent(self, e: QMouseEvent):
        self._painting = False
        if self._lassoing:
            self._lassoing = False
            self.canvas.lasso_finish()
        if self._transforming:
            self._transforming = False
            self.canvas.end_transform()
        super().mouseReleaseEvent(e)
        if self.canvas.mode == "move":
            self.canvas._update_handles()

    def leaveEvent(self, e):
        self.canvas._hide_brush_cursor()
        super().leaveEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.canvas._position_overlay()


class MaskCanvas(QWidget):
    request_inpaint = Signal()
    request_outpaint = Signal()
    request_integrate = Signal()        # integra le immagini posizionate (composito + maschere)
    request_segment_region = Signal()   # selezione oggetto via click/riquadro (SAM)

    def __init__(self, store: MaskStore | None = None):
        super().__init__()
        self.store = store or MaskStore()
        self.scene = QGraphicsScene()
        self.view = _View(self)
        self.aspect_w, self.aspect_h = 720, 1280
        self.mode = "move"      # move | brush | erase | fill | lasso | sam_point | sam_box
        self.brush_size = 40
        self._image_path: str | None = None
        self._undo: list[QImage] = []
        self._redo: list[QImage] = []
        self._lasso_pts: list[QPointF] = []
        self._lasso_item = None
        self._edge_map = None          # mappa bordi per il lazzo magnetico
        self._brush_cursor = None      # cerchio cursore pennello
        self._seg_region = None        # ("point", QPointF) | ("box", QRectF)
        self._seg_marker = None        # item visivo della selezione
        self._seg_start = QPointF()
        self._handles: list = []       # maniglie ridimensiona/ruota (in modalità Sposta)
        self._handle_rects: dict = {}  # role -> QRectF (hit test)
        self._tf = None                # stato trasformazione attiva

        self.frame_item = QGraphicsRectItem(QRectF(0, 0, self.aspect_w, self.aspect_h))
        self.frame_item.setPen(QPen(QColor("#46a"), 4, Qt.DashLine))
        self.frame_item.setBrush(QBrush(QColor(20, 20, 28)))
        self.frame_item.setZValue(-100)
        self.scene.addItem(self.frame_item)

        # immagine base + layer importati: OGNI immagine ha la PROPRIA maschera.
        self.img_item = QGraphicsPixmapItem()
        self.img_item.setFlag(QGraphicsPixmapItem.ItemIsMovable, True)
        self.img_item.setZValue(0)
        self.scene.addItem(self.img_item)
        self._entries: list[dict] = [
            {"item": self.img_item, "mask": self._new_mask(), "name": tr("canvas.base_photo_name"), "base": True}
        ]
        self._active = 0               # voce selezionata (0 = base)

        self.mask_item = QGraphicsPixmapItem(); self.mask_item.setZValue(500)
        self.scene.addItem(self.mask_item)

        self._build_ui(); self._refresh_mask(); self._rebuild_layer_list(); self._fit()

    # ---------- modello "entries": ogni immagine ha la sua maschera ----------
    def _new_mask(self) -> QImage:
        m = QImage(self.aspect_w, self.aspect_h, QImage.Format_ARGB32)
        m.fill(Qt.transparent)
        return m

    @property
    def mask_image(self) -> QImage:
        return self._entries[self._active]["mask"]

    @mask_image.setter
    def mask_image(self, v: QImage):
        self._entries[self._active]["mask"] = v

    @property
    def _layers(self) -> list:
        return [e["item"] for e in self._entries if not e["base"]]

    def _rebuild_layer_list(self):
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for e in self._entries:
            self.layer_list.addItem(QListWidgetItem(e["name"]))
        self.layer_list.setCurrentRow(self._active)
        self.layer_list.blockSignals(False)

    def _build_ui(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(4, 4, 4, 4)
        # ---- top bar: azioni (non strumenti) ----
        bar = QHBoxLayout()
        def b(txt, fn, w=None):
            x = QPushButton(txt); x.clicked.connect(fn)
            if w:
                x.setMaximumWidth(w)
            bar.addWidget(x); return x
        b("📁", self._load_dialog, 40); b("⤢", self._fit, 40)
        b("↶", self.undo, 40); b("↷", self.redo, 40)
        b(tr("canvas.clear_btn"), self.clear_mask); b(tr("canvas.save_btn"), self.save_mask); b(tr("canvas.reset_btn"), self.reset_saved)
        # Bottone «Integra immagini» spostato nel tab «Inpaint / Outpaint», sezione Integra.
        bar.addStretch(1)
        bar.addWidget(QLabel(tr("canvas.brush_label")))
        self.brush_sl = QSlider(Qt.Horizontal); self.brush_sl.setRange(3, 200); self.brush_sl.setValue(self.brush_size)
        self.brush_sl.setMaximumWidth(140); self.brush_sl.valueChanged.connect(self.set_brush)
        bar.addWidget(self.brush_sl)
        self.brush_lbl = QLabel(f"{self.brush_size}px"); bar.addWidget(self.brush_lbl)
        lay.addLayout(bar)

        mid = QHBoxLayout(); mid.setSpacing(6)
        # ---- colonna strumenti VERTICALE (solo icone) a sinistra ----
        toolcol = QVBoxLayout(); toolcol.setSpacing(6); toolcol.setContentsMargins(0, 2, 0, 2)
        self._mode_btns = {}
        for tool in TOOLS:
            btn = _ToolButton(self, tool)
            btn.clicked.connect(lambda _=False, m=tool[0]: self.set_mode(m))
            toolcol.addWidget(btn); self._mode_btns[tool[0]] = btn
        toolcol.addStretch(1)
        self._mode_btns["move"].setChecked(True)
        mid.addLayout(toolcol)
        mid.addWidget(self.view, 1)
        # ---- pannello layer (immagini importate) a destra ----
        side = QVBoxLayout()
        side.addWidget(QLabel(tr("canvas.images_panel_label")))
        self.layer_list = QListWidget(); self.layer_list.setMaximumWidth(170)
        self.layer_list.currentRowChanged.connect(self._on_layer_sel)
        side.addWidget(self.layer_list)
        lb = QHBoxLayout()
        for t, fn in [(tr("canvas.add_img_btn"), self._import_layer), ("↑", self._layer_up),
                      ("↓", self._layer_down), ("🗑", self._layer_del)]:
            x = QPushButton(t); x.setMaximumWidth(48); x.clicked.connect(fn); lb.addWidget(x)
        side.addLayout(lb)
        mid.addLayout(side)
        lay.addLayout(mid, 1)
        self._hint = _ToolHint(self)        # tooltip grafico animato (fluttua sul canvas)
        self.status = QLabel(tr("canvas.status_hint"))
        self.status.setStyleSheet("color:#888;")
        lay.addWidget(self.status)

        # icona di caricamento centrata sul frame (al posto del bottone), visibile se non c'è foto
        self.upload_overlay = QPushButton(tr("canvas.upload_overlay"), self.view)
        self.upload_overlay.setCursor(Qt.PointingHandCursor)
        self.upload_overlay.setStyleSheet(
            "QPushButton{background:rgba(30,34,46,210);color:#cdd;border:2px dashed #5a86c0;"
            "border-radius:16px;font-size:22px;font-weight:bold;padding:22px 30px;}"
            "QPushButton:hover{background:rgba(46,86,140,230);color:#fff;border-color:#7fb0ff;}")
        self.upload_overlay.clicked.connect(self._load_dialog)
        self.upload_overlay.adjustSize()
        self._position_overlay()

    def _position_overlay(self):
        ov = getattr(self, "upload_overlay", None)
        if ov is None:
            return
        vp = self.view.viewport()
        ov.move((vp.width() - ov.width()) // 2, (vp.height() - ov.height()) // 2)
        ov.setVisible(self._image_path is None)

    # ---------- tooltip grafico degli strumenti ----------
    def _show_hint(self, btn: "_ToolButton"):
        p = btn.mapTo(self, QPoint(btn.width() + 8, btn.height() // 2))
        self._hint.popup(p.x(), p.y(), btn._tool[1], tr(btn._tool[2]), tr(btn._tool[3]))

    def _hide_hint(self):
        if getattr(self, "_hint", None) is not None:
            self._hint.hide()

    # ---------- mode / brush ----------
    def set_mode(self, m: str):
        self.mode = m
        for name, btn in self._mode_btns.items():
            btn.setChecked(name == m)
        movable = m == "move"
        self.img_item.setFlag(QGraphicsPixmapItem.ItemIsMovable, movable)
        for L in self._layers:
            L.setFlag(QGraphicsPixmapItem.ItemIsMovable, movable)
        if m != "lasso":
            self._lasso_reset()
        if m != "sam_point":
            self._clear_seg_marker()
        if m not in ("brush", "erase"):
            self._hide_brush_cursor()
        if m == "sam_point":
            self.status.setText(tr("canvas.status_sam"))
        elif m == "lasso":
            self.status.setText(tr("canvas.status_lasso"))
        self._update_handles()      # maniglie visibili solo in modalità Sposta

    # ---------- cursore-cerchio del pennello (mostra il diametro reale) ----------
    def _ensure_brush_cursor(self):
        if getattr(self, "_brush_cursor", None) is None:
            self._brush_cursor = self.scene.addEllipse(QRectF(), QPen(QColor("#ffcc00"), 1.5))
            self._brush_cursor.setZValue(800); self._brush_cursor.setVisible(False)

    def update_brush_cursor(self, sp: QPointF):
        if self.mode not in ("brush", "erase"):
            self._hide_brush_cursor(); return
        self._ensure_brush_cursor()
        inside = 0 <= sp.x() <= self.aspect_w and 0 <= sp.y() <= self.aspect_h
        if not inside:
            self._brush_cursor.setVisible(False); return
        r = self.brush_size / 2.0
        self._brush_cursor.setRect(QRectF(sp.x()-r, sp.y()-r, 2*r, 2*r))
        self._brush_cursor.setVisible(True)

    def _hide_brush_cursor(self):
        if getattr(self, "_brush_cursor", None) is not None:
            self._brush_cursor.setVisible(False)

    # ---------- lazzo continuo + magnetico (segue i bordi degli oggetti vicini) ----------
    def lasso_begin(self, sp: QPointF):
        self._edge_map = self._compute_edge_map()
        self._lasso_pts = [self._snap(sp)]
        self._draw_lasso()

    def lasso_drag(self, sp: QPointF):
        p = self._snap(sp)
        if not self._lasso_pts or (abs(p.x()-self._lasso_pts[-1].x()) + abs(p.y()-self._lasso_pts[-1].y())) >= 2:
            self._lasso_pts.append(p); self._draw_lasso()

    def lasso_finish(self):
        from PySide6.QtGui import QPolygonF
        if len(self._lasso_pts) >= 3:
            self.push_undo()
            p = QPainter(self.mask_image)
            p.setBrush(QColor(255, 40, 40, 150)); p.setPen(Qt.NoPen)
            p.drawPolygon(QPolygonF(self._lasso_pts)); p.end()
            self._refresh_mask()
            self.status.setText(tr("canvas.status_lasso_applied"))
        self._lasso_reset()

    def _draw_lasso(self):
        from PySide6.QtGui import QPolygonF, QPainterPath
        if self._lasso_item is not None:
            self.scene.removeItem(self._lasso_item); self._lasso_item = None
        if len(self._lasso_pts) < 2:
            return
        path = QPainterPath(); path.moveTo(self._lasso_pts[0])
        for pt in self._lasso_pts[1:]:
            path.lineTo(pt)
        self._lasso_item = self.scene.addPath(path, QPen(QColor("#ffcc00"), 2))
        self._lasso_item.setZValue(600)

    def _compute_edge_map(self):
        """Mappa dei bordi (gradiente) del composito → per il magnetismo del lazzo."""
        if np is None:
            return None
        img = QImage(self.aspect_w, self.aspect_h, QImage.Format_ARGB32); img.fill(Qt.black)
        p = QPainter(img)
        self._render_scene(p, QRectF(0, 0, self.aspect_w, self.aspect_h), QRectF(0, 0, self.aspect_w, self.aspect_h))
        p.end()
        g = img.convertToFormat(QImage.Format_Grayscale8)
        w, h, bpl = g.width(), g.height(), g.bytesPerLine()
        ptr = g.constBits()
        try:
            arr = np.frombuffer(ptr, dtype=np.uint8, count=bpl*h).reshape((h, bpl))[:, :w].astype(np.float32)
        except Exception:
            return None
        gy, gx = np.gradient(arr)
        return np.hypot(gx, gy)

    def _snap(self, sp: QPointF, r: int = 12) -> QPointF:
        """Aggancia il punto al bordo più forte entro r px (se significativo)."""
        if self._edge_map is None:
            return QPointF(sp)
        H, W = self._edge_map.shape
        xi, yi = int(sp.x()), int(sp.y())
        x0, x1 = max(0, xi-r), min(W, xi+r+1)
        y0, y1 = max(0, yi-r), min(H, yi+r+1)
        if x1 <= x0 or y1 <= y0:
            return QPointF(sp)
        win = self._edge_map[y0:y1, x0:x1]
        if float(win.max()) < 24.0:        # nessun bordo significativo → non agganciare
            return QPointF(sp)
        dy, dx = np.unravel_index(int(np.argmax(win)), win.shape)
        return QPointF(float(x0+dx), float(y0+dy))

    def _lasso_reset(self):
        self._lasso_pts = []
        self._edge_map = None
        if self._lasso_item is not None:
            self.scene.removeItem(self._lasso_item); self._lasso_item = None

    # ---------- selezione oggetto (SAM): punto / riquadro ----------
    def _clamp(self, sp: QPointF) -> QPointF:
        return QPointF(max(0, min(self.aspect_w, sp.x())), max(0, min(self.aspect_h, sp.y())))

    def _clear_seg_marker(self):
        if self._seg_marker is not None:
            self.scene.removeItem(self._seg_marker); self._seg_marker = None
        self._seg_region = None

    def sam_point_at(self, sp: QPointF):
        sp = self._clamp(sp)
        self._clear_seg_marker()
        self._seg_region = ("point", sp)
        r = max(6, self.aspect_w / 80)
        self._seg_marker = self.scene.addEllipse(sp.x()-r, sp.y()-r, 2*r, 2*r,
                                                 QPen(QColor("#36f"), 3), QBrush(QColor(60, 120, 255, 90)))
        self._seg_marker.setZValue(650)
        self.status.setText(tr("canvas.status_point_selected"))
        self.request_segment_region.emit()

    def export_region_mask(self, out_dir: Path) -> dict:
        """Composita + maschera della regione indicata (cerchio per punto, rettangolo per box)."""
        out_dir = Path(out_dir)
        image = self._composite(out_dir)
        mask = QImage(self.aspect_w, self.aspect_h, QImage.Format_Grayscale8); mask.fill(0)
        p = QPainter(mask); p.setPen(Qt.NoPen); p.setBrush(QColor(255, 255, 255))
        kind = self._seg_region[0] if self._seg_region else None
        if kind == "point":
            sp = self._seg_region[1]; r = max(8, self.aspect_w / 60)
            p.drawEllipse(sp, r, r)
        elif kind == "box":
            p.drawRect(self._seg_region[1])
        p.end()
        mpath = out_dir / "seg_region.png"; mask.save(str(mpath))
        return {"image": image, "mask": str(mpath), "hint": "center-1"}

    def load_mask_result(self, path: str):
        """Carica una maschera (bianco=selezione) generata da SAM nel layer maschera, modificabile."""
        res = QImage(path)
        if res.isNull():
            return
        self.push_undo()
        w, h = self.aspect_w, self.aspect_h
        res8 = res.convertToFormat(QImage.Format_Grayscale8).scaled(
            w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        res8 = res8.convertToFormat(QImage.Format_Grayscale8)   # riassicura il formato dopo lo scaled
        # NB: NON usare `tinted.setAlphaChannel(res.convertToFormat(Format_Alpha8))`: convertire una
        # Grayscale8 (opaca) ad Alpha8 produce alpha=255 OVUNQUE → la maschera copriva TUTTO il
        # frame qualunque fosse la selezione SAM ("seleziona tutto"). Costruiamo l'ARGB32 a mano:
        # ogni pixel = rosso (255,40,40) con alpha = 0.6 * luminanza_maschera.
        bpl = res8.bytesPerLine()
        buf = np.frombuffer(res8.constBits(), np.uint8).reshape(h, bpl)[:, :w]   # luminanza (bianco=selez)
        arr = np.zeros((h, w, 4), np.uint8)        # ARGB32 little-endian in memoria = B,G,R,A
        arr[..., 0] = 40; arr[..., 1] = 40; arr[..., 2] = 255
        arr[..., 3] = (buf.astype(np.uint16) * 153 // 255).astype(np.uint8)   # alpha ∝ maschera (0.6)
        out = QImage(arr.data, w, h, QImage.Format_ARGB32).copy()
        self.mask_image = out; self._refresh_mask()
        self._clear_seg_marker(); self.set_mode("brush")
        self.status.setText(tr("canvas.status_mask_loaded"))

    def set_brush(self, v: int):
        self.brush_size = max(3, min(200, int(v)))
        if self.brush_sl.value() != self.brush_size:
            self.brush_sl.setValue(self.brush_size)
        self.brush_lbl.setText(f"{self.brush_size}px")

    # ---------- aspetto / foto ----------
    def set_aspect(self, w: int, h: int):
        self.aspect_w, self.aspect_h = max(1, w), max(1, h)
        self.frame_item.setRect(QRectF(0, 0, self.aspect_w, self.aspect_h))
        for e in self._entries:        # ridimensiona la maschera di ogni immagine
            e["mask"] = e["mask"].scaled(self.aspect_w, self.aspect_h) if not e["mask"].isNull() else self._new_mask()
        self._refresh_mask(); self._fit()

    def _load_dialog(self):
        fn, _ = QFileDialog.getOpenFileName(self, tr("canvas.load_photo_title"), "", tr("canvas.image_filter"))
        if fn:
            self.load_image(fn)

    def load_image(self, path: str):
        pix = QPixmap(path)
        if pix.isNull():
            return
        self._active = 0; self._rebuild_layer_list()   # la nuova foto è la BASE
        self._image_path = path
        self.img_item.setPixmap(pix)
        s = min(self.aspect_w / pix.width(), self.aspect_h / pix.height())
        self.img_item.setScale(s)
        self.img_item.setPos((self.aspect_w - pix.width()*s)/2, (self.aspect_h - pix.height()*s)/2)
        self.clear_mask(silent=True)
        saved = self.store.load(path)
        if saved:
            m = QImage(saved["mask_png"])
            if not m.isNull():
                self.mask_image = m.convertToFormat(QImage.Format_ARGB32).scaled(self.aspect_w, self.aspect_h)
                self._refresh_mask()
            tr = saved.get("transform", {})
            if "scale" in tr: self.img_item.setScale(tr["scale"])
            if "x" in tr and "y" in tr: self.img_item.setPos(tr["x"], tr["y"])
            self.status.setText(tr("canvas.status_reloaded"))
        self._undo.clear(); self._redo.clear(); self._fit()

    # ---------- layer immagini importate (ognuno con la sua maschera) ----------
    def _import_layer(self):
        fn, _ = QFileDialog.getOpenFileName(self, tr("canvas.import_layer_title"), "", tr("canvas.image_filter"))
        if not fn:
            return
        pix = QPixmap(fn)
        if pix.isNull():
            return
        it = QGraphicsPixmapItem(pix)
        it.setFlag(QGraphicsPixmapItem.ItemIsMovable, self.mode == "move")
        s = min(self.aspect_w / pix.width(), self.aspect_h / pix.height()) * 0.6
        it.setScale(s); it.setPos(self.aspect_w*0.2, self.aspect_h*0.2)
        self.scene.addItem(it)
        self._entries.append({"item": it, "mask": self._new_mask(), "name": Path(fn).name, "base": False})
        self._reindex_z()
        self._rebuild_layer_list()
        self.layer_list.setCurrentRow(len(self._entries) - 1)   # seleziona la nuova (mostra la sua maschera)

    def selected_layer(self):
        e = self._entries[self._active]
        return None if e["base"] else e["item"]

    def active_item(self):
        return self._entries[self._active]["item"]

    # ---------- maniglie ridimensiona / ruota (modalità Sposta) ----------
    def _clear_handles(self):
        for it in self._handles:
            self.scene.removeItem(it)
        self._handles = []; self._handle_rects = {}

    def _update_handles(self):
        self._clear_handles()
        if self.mode != "move":
            return
        it = self.active_item()
        if it is None or it.pixmap().isNull():
            return
        r = it.sceneBoundingRect()
        hr = max(6.0, self.aspect_w / 90)
        pen = QPen(QColor("#36f"), 2); brush = QBrush(QColor(255, 255, 255, 235))
        for role, pt in {"tl": r.topLeft(), "tr": r.topRight(),
                         "bl": r.bottomLeft(), "br": r.bottomRight()}.items():
            rect = QRectF(pt.x() - hr, pt.y() - hr, 2 * hr, 2 * hr)
            item = self.scene.addRect(rect, pen, brush); item.setZValue(720)
            self._handles.append(item); self._handle_rects[role] = rect
        top_c = QPointF(r.center().x(), r.top())
        rot_c = QPointF(top_c.x(), top_c.y() - max(24, self.aspect_w / 28))
        line = self.scene.addLine(top_c.x(), top_c.y(), rot_c.x(), rot_c.y(), pen); line.setZValue(719)
        self._handles.append(line)
        rrect = QRectF(rot_c.x() - hr, rot_c.y() - hr, 2 * hr, 2 * hr)
        rot = self.scene.addEllipse(rrect, pen, QBrush(QColor(120, 200, 120, 235))); rot.setZValue(720)
        self._handles.append(rot); self._handle_rects["rotate"] = rrect

    def handle_at(self, sp: QPointF):
        for role, rect in self._handle_rects.items():
            if rect.contains(sp):
                return role
        return None

    def begin_transform(self, role: str, sp: QPointF):
        it = self.active_item()
        if it is None:
            return
        bc = it.boundingRect().center()
        before = it.mapToScene(bc)
        it.setTransformOriginPoint(bc)                # trasforma attorno al centro
        after = it.mapToScene(bc)
        it.setPos(it.pos() + (before - after))        # compensa: niente salto
        c = it.mapToScene(it.transformOriginPoint())
        self._tf = {"role": role, "item": it, "center": c, "start": sp,
                    "scale0": it.scale(), "rot0": it.rotation()}

    def update_transform(self, sp: QPointF):
        tf = self._tf
        if not tf:
            return
        it = tf["item"]; c = tf["center"]
        if tf["role"] == "rotate":
            a0 = math.atan2(tf["start"].y() - c.y(), tf["start"].x() - c.x())
            a1 = math.atan2(sp.y() - c.y(), sp.x() - c.x())
            it.setRotation(tf["rot0"] + math.degrees(a1 - a0))
        else:
            d0 = math.hypot(tf["start"].x() - c.x(), tf["start"].y() - c.y()) or 1.0
            d1 = math.hypot(sp.x() - c.x(), sp.y() - c.y())
            it.setScale(max(0.05, tf["scale0"] * (d1 / d0)))
        self._update_handles()

    def end_transform(self):
        self._tf = None
        self._update_handles()

    def _on_layer_sel(self, row: int):
        if not (0 <= row < len(self._entries)) or row == self._active:
            return
        self._active = row
        self._undo.clear(); self._redo.clear()      # undo è per-maschera
        self._refresh_mask()                          # mostra SOLO la maschera dell'immagine selezionata
        self._update_handles()                        # maniglie sull'immagine selezionata
        self.status.setText(tr("canvas.status_selected").format(name=self._entries[row]['name']))

    def _reindex_z(self):
        for i, e in enumerate(self._entries):
            if not e["base"]:
                e["item"].setZValue(10 + i)

    def _swap_entries(self, a: int, b: int):
        self._entries[a], self._entries[b] = self._entries[b], self._entries[a]
        self._reindex_z(); self._active = b
        self._rebuild_layer_list()

    def _layer_up(self):     # su = verso l'alto nella lista (indice minore), mai sotto la base (0)
        i = self._active
        if i > 1:
            self._swap_entries(i, i - 1)

    def _layer_down(self):
        i = self._active
        if 0 < i < len(self._entries) - 1:
            self._swap_entries(i, i + 1)

    def _layer_del(self):
        i = self._active
        if i <= 0 or i >= len(self._entries):   # non eliminare la base
            return
        self.scene.removeItem(self._entries.pop(i)["item"])
        self._active = min(i, len(self._entries) - 1)
        self._reindex_z(); self._rebuild_layer_list(); self._refresh_mask()

    # ---------- maschera: paint / fill / undo ----------
    def push_undo(self):
        self._undo.append(self.mask_image.copy())
        if len(self._undo) > MAX_UNDO:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self):
        if self._undo:
            self._redo.append(self.mask_image.copy())
            self.mask_image = self._undo.pop(); self._refresh_mask()

    def redo(self):
        if self._redo:
            self._undo.append(self.mask_image.copy())
            self.mask_image = self._redo.pop(); self._refresh_mask()

    def paint_at(self, sp: QPointF):
        p = QPainter(self.mask_image)
        if self.mode == "erase":
            p.setCompositionMode(QPainter.CompositionMode_Clear); p.setBrush(Qt.transparent)
        else:
            p.setBrush(QColor(255, 40, 40, 150))
        p.setPen(Qt.NoPen)
        p.drawEllipse(sp, self.brush_size/2, self.brush_size/2); p.end()
        self._refresh_mask()

    def fill_at(self, sp: QPointF):
        # flood fill semplice sull'alpha della maschera, partendo dal punto
        x, y = int(sp.x()), int(sp.y())
        if not (0 <= x < self.aspect_w and 0 <= y < self.aspect_h):
            return
        img = self.mask_image
        target_on = (img.pixel(x, y) >> 24) & 0xFF > 0
        fillcol = QColor(255, 40, 40, 150).rgba()
        clear = (self.mode == "erase")
        stack = [(x, y)]; seen = set()
        W, H = self.aspect_w, self.aspect_h
        cnt = 0
        while stack and cnt < W*H:
            cx, cy = stack.pop()
            if (cx, cy) in seen or not (0 <= cx < W and 0 <= cy < H):
                continue
            seen.add((cx, cy)); cnt += 1
            on = (img.pixel(cx, cy) >> 24) & 0xFF > 0
            if on != target_on:
                continue
            img.setPixel(cx, cy, 0 if clear else fillcol)
            stack += [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]
        self._refresh_mask()

    def _refresh_mask(self):
        self.mask_item.setPixmap(QPixmap.fromImage(self.mask_image))

    def clear_mask(self, silent: bool = False):
        if not silent:
            self.push_undo()
        self.mask_image.fill(Qt.transparent); self._refresh_mask()

    def _mask_white(self) -> QImage:
        tmp = QImage(self.mask_image.size(), QImage.Format_ARGB32); tmp.fill(Qt.white)
        tmp.setAlphaChannel(self.mask_image.convertToFormat(QImage.Format_Alpha8))
        out = QImage(self.mask_image.size(), QImage.Format_ARGB32); out.fill(Qt.black)
        p = QPainter(out); p.drawImage(0, 0, tmp); p.end()
        return out

    def save_mask(self):
        if not self._image_path:
            return
        tmp = Path(self.store.dir) / "_tmp_mask.png"
        self._mask_white().save(str(tmp))
        self.store.save(self._image_path, str(tmp),
                        {"scale": self.img_item.scale(), "x": self.img_item.x(), "y": self.img_item.y()})
        self.status.setText(tr("canvas.status_saved"))

    def reset_saved(self):
        if self._image_path:
            self.store.reset(self._image_path); self.clear_mask(); self.status.setText(tr("canvas.status_reset"))

    def _fit(self):
        self.view.fitInView(self.frame_item, Qt.KeepAspectRatio)
        self._position_overlay()
        self._update_handles()

    def has_outpaint_area(self) -> bool:
        """True se la foto NON copre tutto il frame (c'è bordo vuoto da estendere)."""
        frame = QRectF(0, 0, self.aspect_w, self.aspect_h)
        covered = self.img_item.sceneBoundingRect().intersected(frame)
        return (covered.width() * covered.height()) < (frame.width() * frame.height()) * 0.995

    def has_mask(self) -> bool:
        img = self.mask_image; step = max(1, img.width()//64)
        for y in range(0, img.height(), step):
            for x in range(0, img.width(), step):
                if (img.pixel(x, y) >> 24) & 0xFF:
                    return True
        return False

    def _render_scene(self, painter, target: QRectF, source: QRectF):
        """Renderizza la scena nascondendo maschera/maniglie/marker (non devono finire nell'output)."""
        hidden = [self.mask_item] + list(self._handles)
        if self._seg_marker is not None:
            hidden.append(self._seg_marker)
        if self._lasso_item is not None:
            hidden.append(self._lasso_item)
        if self._brush_cursor is not None:
            hidden.append(self._brush_cursor)
        for it in hidden:
            it.setVisible(False)
        self.scene.render(painter, target, source)
        for it in hidden:
            it.setVisible(True)

    def export(self, out_dir: Path) -> dict:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        img = QImage(self.aspect_w, self.aspect_h, QImage.Format_ARGB32); img.fill(Qt.black)
        p = QPainter(img)
        self._render_scene(p, QRectF(0, 0, self.aspect_w, self.aspect_h), QRectF(0, 0, self.aspect_w, self.aspect_h))
        p.end()
        ipath = out_dir / "frame1.png"; img.save(str(ipath))
        res = {"image": str(ipath), "mask": None, "has_mask": False}
        if self.has_mask():
            mpath = out_dir / "frame1_mask.png"; self._mask_white().save(str(mpath))
            res["mask"] = str(mpath); res["has_mask"] = True
        return res

    def export_subject_rgba(self, out_dir: Path) -> dict:
        """Soggetto isolato dal canvas con canale ALPHA = maschera (sfondo trasparente).

        Usato per alimentare la generazione 3D (TRELLIS) col SOLO soggetto
        selezionato a maschera (pennello o SAM), escludendo sfondo e oggetti
        indesiderati. Ritorna {"image": <png rgba>, "has_mask": bool}. Se non
        c'è maschera ritorna has_mask=False (il chiamante userà l'immagine piena).
        """
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        if not self.has_mask():
            return {"image": None, "has_mask": False}
        comp = QImage(self.aspect_w, self.aspect_h, QImage.Format_ARGB32)
        comp.fill(Qt.transparent)
        p = QPainter(comp)
        self._render_scene(p, QRectF(0, 0, self.aspect_w, self.aspect_h),
                           QRectF(0, 0, self.aspect_w, self.aspect_h))
        p.end()
        # alpha = maschera disegnata (opaco dove tenere il soggetto)
        comp.setAlphaChannel(self.mask_image.convertToFormat(QImage.Format_Alpha8))
        path = out_dir / "frame1_subject.png"
        comp.save(str(path))
        return {"image": str(path), "has_mask": True}

    def _composite(self, out_dir: Path) -> str:
        """Renderizza cornice + foto + layer (senza maschera) a WxH → PNG."""
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        img = QImage(self.aspect_w, self.aspect_h, QImage.Format_ARGB32); img.fill(Qt.black)
        p = QPainter(img)
        self._render_scene(p, QRectF(0, 0, self.aspect_w, self.aspect_h), QRectF(0, 0, self.aspect_w, self.aspect_h))
        p.end()
        path = out_dir / "frame1.png"; img.save(str(path))
        return str(path)

    def _border_mask(self) -> QImage:
        """Maschera bianca dove la cornice NON è coperta dalla foto (area da outpaint)."""
        out = QImage(self.aspect_w, self.aspect_h, QImage.Format_Grayscale8); out.fill(255)
        covered = self.img_item.sceneBoundingRect().intersected(QRectF(0, 0, self.aspect_w, self.aspect_h))
        p = QPainter(out); p.fillRect(covered, QColor(0, 0, 0)); p.end()
        return out

    def export_inpaint(self, out_dir: Path, mode: str) -> dict:
        """Outpaint: composito intero frame + maschera-bordo.
        Inpaint: RITAGLIA sull'area coperta dalla foto (niente bordo → l'inpaint non
        tocca/genera mai la zona vuota), con maschera ritagliata. Ritorna anche il bbox
        per rimettere il risultato esattamente al suo posto."""
        out_dir = Path(out_dir)
        if mode == "outpaint":
            image = self._composite(out_dir)
            mpath = out_dir / "frame1_mask.png"; self._border_mask().save(str(mpath))
            return {"image": image, "mask": str(mpath), "mode": mode, "bbox": None}
        # --- inpaint: solo il riquadro della foto ---
        frame = QRectF(0, 0, self.aspect_w, self.aspect_h)
        bbox = self.img_item.sceneBoundingRect().intersected(frame)
        if bbox.width() < 4 or bbox.height() < 4:        # nessuna foto valida → frame intero
            image = self._composite(out_dir)
            mpath = out_dir / "frame1_mask.png"; self._mask_white().save(str(mpath))
            return {"image": image, "mask": str(mpath), "mode": mode, "bbox": None}
        x, y, w, h = int(bbox.x()), int(bbox.y()), int(bbox.width()), int(bbox.height())
        img = QImage(w, h, QImage.Format_ARGB32); img.fill(Qt.black)
        p = QPainter(img)
        self._render_scene(p, QRectF(0, 0, w, h), QRectF(x, y, w, h))   # solo la regione foto
        p.end()
        ipath = out_dir / "frame1.png"; img.save(str(ipath))
        mpath = out_dir / "frame1_mask.png"
        self._mask_white().copy(x, y, w, h).save(str(mpath))           # maschera ritagliata
        return {"image": str(ipath), "mask": str(mpath), "mode": mode, "bbox": (x, y, w, h)}

    def has_layers(self) -> bool:
        return len(self._entries) > 1

    def _mask_filled(self, m: QImage) -> bool:
        step = max(1, m.width() // 64)
        for y in range(0, m.height(), step):
            for x in range(0, m.width(), step):
                if (m.pixel(x, y) >> 24) & 0xFF:
                    return True
        return False

    def export_integrate(self, out_dir: Path) -> dict:
        """Input stile MagicQuill `foreground_edit`: NON incolla il rettangolo intero.
        L'OGGETTO (maschera disegnata del layer, o in fallback il suo bbox) resta come seme;
        la ZONA DI EDIT attorno (bbox cresciuto) viene messa a BIANCO; il resto = base.
        Kontext (+ puzzle_lora) RIGENERA quella zona integrando il contenuto, senza copiarlo."""
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        W, H = self.aspect_w, self.aspect_h
        frame = QRectF(0, 0, W, H)
        grow = max(24, int(0.05 * max(W, H)))
        comp = QImage(self._composite(out_dir)).convertToFormat(QImage.Format_ARGB32)
        obj = QImage(W, H, QImage.Format_ARGB32); obj.fill(Qt.transparent)        # dove c'è l'oggetto
        region = QImage(W, H, QImage.Format_ARGB32); region.fill(Qt.transparent)  # zona di edit
        po = QPainter(obj); pr = QPainter(region); po.setPen(Qt.NoPen); pr.setPen(Qt.NoPen)
        any_region = False
        for e in self._entries[1:]:                        # solo immagini importate
            bbox = e["item"].sceneBoundingRect().intersected(frame)
            if bbox.width() < 2 or bbox.height() < 2:
                continue
            if self._mask_filled(e["mask"]):               # oggetto = maschera disegnata
                wm = QImage(W, H, QImage.Format_ARGB32); wm.fill(QColor(255, 255, 255))
                wm.setAlphaChannel(e["mask"].convertToFormat(QImage.Format_Alpha8))
                po.drawImage(0, 0, wm)
            else:                                           # fallback: tutto il rettangolo
                po.fillRect(bbox, QColor(255, 255, 255))
            pr.fillRect(bbox.adjusted(-grow, -grow, grow, grow).intersected(frame), QColor(255, 255, 255))
            any_region = True
        po.end(); pr.end()
        # alone bianco = zona di edit MENO l'oggetto
        whiten = region.copy()
        pw = QPainter(whiten); pw.setCompositionMode(QPainter.CompositionMode_DestinationOut)
        pw.drawImage(0, 0, obj); pw.end()
        prepared = comp.copy()
        pp = QPainter(prepared); pp.drawImage(0, 0, whiten); pp.end()
        ipath = out_dir / "integrate_input.png"; prepared.save(str(ipath))
        msk = QImage(W, H, QImage.Format_ARGB32); msk.fill(Qt.black)
        pm = QPainter(msk); pm.drawImage(0, 0, region); pm.end()
        mpath = out_dir / "integrate_mask.png"; msk.save(str(mpath))
        return {"image": str(ipath), "mask": str(mpath), "has_region": any_region and self.has_layers()}

    def _drop_layers(self):
        for e in self._entries[1:]:           # rimuove i layer importati (tiene la base)
            self.scene.removeItem(e["item"])
        del self._entries[1:]
        self._active = 0
        self._rebuild_layer_list()

    def load_result(self, path: str):
        """Sostituisce la foto base col risultato generato (riempie esattamente la cornice)."""
        pix = QPixmap(path)
        if pix.isNull():
            return
        self._image_path = path
        self._drop_layers()                  # i layer sono già inglobati nel risultato
        self.img_item.setPixmap(pix)
        self.img_item.setScale(min(self.aspect_w / pix.width(), self.aspect_h / pix.height()))
        self.img_item.setPos(0, 0)
        self.clear_mask(silent=True); self._undo.clear(); self._redo.clear(); self._fit()

    def load_inpaint_result(self, path: str, bbox: tuple[int, int, int, int]):
        """Rimette il risultato dell'inpaint esattamente dove stava la foto (stesso bbox),
        senza introdurre il bordo. Mantiene l'inquadratura nel frame."""
        pix = QPixmap(path)
        if pix.isNull():
            return
        x, y, w, h = bbox
        self._image_path = path
        self._drop_layers()
        self.img_item.setPixmap(pix)
        self.img_item.setScale(1.0)          # il ritaglio è già a risoluzione di scena
        self.img_item.setPos(x, y)
        self.clear_mask(silent=True); self._undo.clear(); self._redo.clear()
        self._position_overlay(); self._fit()

    @property
    def image_path(self):
        return self._image_path
