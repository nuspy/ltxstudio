"""Editor immagine stile MagicQuill (PySide6): cornice formato video (no deformazione),
foto base spostabile/zoomabile/croppabile, layer immagini importate con ordine (porta su/giù),
maschera inpaint con pennello ridimensionabile, gomma, fill, undo/redo (100 passi).

La GENERAZIONE inpaint/outpaint (FLUX Fill/Kontext via ComfyUI) e il lazzo SAM verranno
collegati separatamente; qui c'è tutta l'editing UI + export (immagine composita + maschera).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None

from PySide6.QtCore import (
    Qt, QRectF, QPointF, QPoint, Signal, QPropertyAnimation, QEasingCurve,
    QParallelAnimationGroup, QTimer,
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush, QWheelEvent, QMouseEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsLineItem,
    QPushButton, QSlider, QLabel, QFileDialog,
    QListWidget, QListWidgetItem, QComboBox, QFrame, QGraphicsOpacityEffect,
    QToolButton, QMenu, QDialog, QSpinBox, QDialogButtonBox, QFormLayout, QCheckBox,
    QMessageBox, QApplication,
)

from core.mask_store import MaskStore
from ui.i18n_manager import tr

MAX_UNDO = 100

# Integra, due maschere: POSIZIONE (dove sta il soggetto) in ROSSO, CONTESTO (zona che il modello puo rigenerare) in GIALLO.
POS_RGB = (255, 40, 40)
CTX_RGB = (255, 204, 0)
BANNER_POS_FG = "#ff6b6b"          # rosso leggibile su fondo scuro (contrasto > 4,5:1 su #14141c)
BANNER_CTX_FG = "#ffd23f"          # giallo (contrasto > 10:1 su #14141c)
BANNER_BG = "#14141c"
OVL_TRANSP_GAP = 5                 # px fra i due cursori di trasparenza (richiesta utente 03/11)
OVL_TRANSP_COL_W = 16              # larghezza di una colonna cursore (px): la sola maniglia, cosi i due cursori restano a OVL_TRANSP_GAP px
# Contesto automatico: margine in PIXEL attorno alla sagoma vera (decisione dell'utente 03/11). 87 px = stessa area della zona di prima (riquadro + 5 % della
# scena) sui casi di prova (scena 1152x832 + ritratto largo 389 px, scontornato: 87-88 px), cosi il comportamento predefinito resta vicino a quello di sempre.
DEFAULT_CTX_PX = 87

# strumenti: (modo, icona, chiave_titolo, chiave_descrizione)
TOOLS = [
    ("move",      "✥",  "canvas.tool_move_title",   "canvas.tool_move_desc"),
    ("brush",     "✎",  "canvas.tool_brush_title",  "canvas.tool_brush_desc"),
    ("erase",     "⌫",  "canvas.tool_erase_title",  "canvas.tool_erase_desc"),
    ("fill",      "▣",  "canvas.tool_fill_title",   "canvas.tool_fill_desc"),
    ("lasso",     "❍",  "canvas.tool_lasso_title",  "canvas.tool_lasso_desc"),
    ("sam_point", "✛",  "canvas.tool_sam_title",    "canvas.tool_sam_desc"),
]

# Proporzioni standard → sottomenù per ratio, ogni voce è una dimensione concreta (multipli di 32,
# model-friendly). Lista estesa: quadrato, widescreen, verticale social, foto, cinema, ultrawide.
ASPECT_GROUPS = [
    ("1:1 ▪ Quadrato",          [(512, 512), (768, 768), (1024, 1024), (1280, 1280)]),
    ("16:9 ▪ Widescreen",       [(1024, 576), (1280, 704), (1536, 864), (1920, 1088)]),
    ("9:16 ▪ Verticale",        [(576, 1024), (704, 1280), (864, 1536), (1088, 1920)]),
    ("4:3 ▪ Classico",          [(1024, 768), (1280, 960), (1408, 1056)]),
    ("3:4 ▪ Ritratto",          [(768, 1024), (960, 1280), (1056, 1408)]),
    ("3:2 ▪ Foto",              [(1152, 768), (1536, 1024)]),
    ("2:3 ▪ Foto vert.",        [(768, 1152), (1024, 1536)]),
    ("5:4",                     [(1280, 1024)]),
    ("4:5 ▪ Instagram",         [(1024, 1280)]),
    ("2:1 ▪ Panorama",          [(1024, 512), (1280, 640)]),
    ("21:9 ▪ Ultrawide",        [(1344, 576), (1536, 640)]),
    ("2.39:1 ▪ Cinemascope",    [(1216, 512), (1536, 640)]),
    ("1.85:1 ▪ Flat",           [(1024, 544), (1408, 768)]),
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
    resolution_chosen = Signal(int, int)   # proporzioni/dimensioni scelte dalla barra canvas (→ video_tab)
    request_inpaint = Signal()
    request_outpaint = Signal()
    request_integrate = Signal()        # integra le immagini posizionate (composito + maschere)
    request_segment_region = Signal()   # selezione oggetto via click/riquadro (SAM)
    image_changed = Signal(str)         # nuova foto base nel canvas (path) → abilita i bottoni "usa canvas"
    integrate_range_changed = Signal(int)   # massimo del cursore del margine di contesto = meta del lato lungo dell'immagine (px)
    ctx_geometry_changed = Signal()     # sagoma ridisegnata (larghezza cambiata): il pannello aggiorna la % equivalente

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
        # ── Integra: due maschere (POSIZIONE rossa, CONTESTO gialla), sempre visibili ──
        self.integrate_ctx_mode = "auto"           # "auto" (dilatazione della sagoma, integrate_context_pct) | "manual" (pennello)
        self.integrate_draw_target = "position"    # in manuale: quale maschera si sta disegnando ("position" | "context")
        self.integrate_ctx_enabled = True          # False = il motore non usa la maschera di contesto (es. Krea 2 Edit)
        self.integrate_mode_active = True          # False = non si e nella modalita Integra (pagina diversa da Modifica): decide frame1_tab
        self._ctx_manual = None                    # QImage ARGB (giallo) della zona di contesto disegnata a mano
        self._ctx_touched = False                  # il contesto manuale e gia stato precompilato/disegnato
        self._ctx_undo: list[QImage] = []
        self._ctx_redo: list[QImage] = []
        self._ovl_sig = None
        self._ovl_base = None
        self._ovl_pos = None
        self._ovl_ctx = None

        self.frame_item = QGraphicsRectItem(QRectF(0, 0, self.aspect_w, self.aspect_h))
        self.frame_item.setPen(QPen(Qt.NoPen))                 # sfondo del frame (sotto la foto)
        self.frame_item.setBrush(QBrush(QColor(20, 20, 28)))
        self.frame_item.setZValue(-100)
        self.scene.addItem(self.frame_item)
        # Contorno tratteggiato blu: elemento SEPARATO disegnato SOPRA la foto (punto 12 dell'utente: prima stava
        # nello stesso item dello sfondo e la foto lo copriva). Sotto la maschera (500) e sotto le maniglie (720).
        self.frame_border = QGraphicsRectItem(QRectF(0, 0, self.aspect_w, self.aspect_h))
        self.frame_border.setPen(QPen(QColor("#46a"), 4, Qt.DashLine))
        self.frame_border.setBrush(QBrush(Qt.NoBrush))
        self.frame_border.setZValue(450)
        self.frame_border.setAcceptedMouseButtons(Qt.NoButton)
        self.scene.addItem(self.frame_border)

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
        # Sovrapposizioni di Integra: item SEPARATI (come il contorno blu del punto 12), mai nell'export (_render_scene li nasconde).
        # Contesto (giallo) sotto, posizione (rosso) sopra; sotto il pennello (500) e le maniglie (720).
        self._ovl_ctx = QGraphicsPixmapItem(); self._ovl_ctx.setZValue(460); self._ovl_ctx.setAcceptedMouseButtons(Qt.NoButton)
        self._ovl_pos = QGraphicsPixmapItem(); self._ovl_pos.setZValue(470); self._ovl_pos.setAcceptedMouseButtons(Qt.NoButton)
        self.scene.addItem(self._ovl_ctx); self.scene.addItem(self._ovl_pos)
        self._ovl_transp = {"position": 50, "context": 50}      # trasparenza % di OGNI sovrapposizione (solo visualizzazione)
        self._ovl_timer = QTimer(self); self._ovl_timer.setSingleShot(True); self._ovl_timer.setInterval(90)
        self._ovl_timer.timeout.connect(self.refresh_integrate_overlay)

        self._build_ui(); self._refresh_mask(); self._rebuild_layer_list(); self._fit()
        self.scene.changed.connect(self._ovl_schedule)   # sposta/ridimensiona/aggiungi layer: l'anteprima si aggiorna da sola
        self._update_banner(); self.refresh_integrate_overlay()

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
        self._refresh_cutout_buttons()
        if hasattr(self, "ovl_legend"):
            self._apply_masks_visibility()

    # ───────────────────────── Proporzioni / dimensioni ─────────────────────────
    def _build_aspect_menu(self) -> QMenu:
        m = QMenu(self)
        for label, sizes in ASPECT_GROUPS:
            sub = m.addMenu(label)
            for w, h in sizes:
                act = sub.addAction(f"{w} × {h}")
                act.triggered.connect(lambda _=False, ww=w, hh=h: self._set_resolution(ww, hh))
        m.addSeparator()
        act_custom = m.addAction(tr("canvas.aspect_custom", "Personalizzato…"))
        act_custom.triggered.connect(self._open_custom_size)
        return m

    def _set_resolution(self, w: int, h: int) -> None:
        """Applica una risoluzione: aggiorna l'aspetto del canvas (che aggiorna l'etichetta) ed emette
        il segnale (main_window lo propaga a video_tab = sorgente di verità per la generazione)."""
        w, h = int(w), int(h)
        self.set_aspect(w, h)
        self.resolution_chosen.emit(w, h)

    def set_resolution_label(self, w: int, h: int) -> None:
        if hasattr(self, "aspect_btn"):
            self.aspect_btn.setText(f"{int(w)}×{int(h)} ▾")

    def _open_custom_size(self) -> None:
        """Dialog dimensioni personalizzate: modificando una misura, l'altra si adatta per mantenere
        la proporzione corrente (se il lucchetto è attivo)."""
        cur_w, cur_h = getattr(self, "aspect_w", 768), getattr(self, "aspect_h", 768)
        dlg = QDialog(self); dlg.setWindowTitle(tr("canvas.aspect_custom", "Dimensioni personalizzate"))
        form = QFormLayout(dlg)
        sp_w = QSpinBox(); sp_w.setRange(256, 4096); sp_w.setSingleStep(32); sp_w.setValue(int(cur_w))
        sp_h = QSpinBox(); sp_h.setRange(256, 4096); sp_h.setSingleStep(32); sp_h.setValue(int(cur_h))
        lock = QCheckBox(tr("canvas.aspect_lock", "Mantieni proporzione")); lock.setChecked(True)
        ratio = {"v": (cur_w / cur_h) if cur_h else 1.0}
        guard = {"on": False}

        def _snap(v):
            return max(256, int(round(v / 32.0)) * 32)

        def _on_w(val):
            if guard["on"] or not lock.isChecked():
                return
            guard["on"] = True
            sp_h.setValue(_snap(val / ratio["v"])); guard["on"] = False

        def _on_h(val):
            if guard["on"] or not lock.isChecked():
                return
            guard["on"] = True
            sp_w.setValue(_snap(val * ratio["v"])); guard["on"] = False

        def _on_lock(state):
            if state:                                        # ri-aggancia la proporzione corrente
                ratio["v"] = sp_w.value() / max(1, sp_h.value())

        sp_w.valueChanged.connect(_on_w); sp_h.valueChanged.connect(_on_h); lock.toggled.connect(_on_lock)
        form.addRow(tr("vid.w_lbl", "Larghezza"), sp_w)
        form.addRow(tr("vid.h_lbl", "Altezza"), sp_h)
        form.addRow(lock)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() == QDialog.Accepted:
            self._set_resolution(sp_w.value(), sp_h.value())

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
        _bx = b("🗑", self.clear_image, 40)
        _bx.setToolTip(tr("canvas.clear_image.tt", "Svuota il canvas: nessuna immagine iniziale (il video diventa text-to-video)"))
        b("↶", self.undo, 40); b("↷", self.redo, 40)
        b(tr("canvas.clear_btn"), self.clear_mask); b(tr("canvas.save_btn"), self.save_mask); b(tr("canvas.reset_btn"), self.reset_saved)
        # ── Menu PROPORZIONI/dimensioni (spostato qui dal tab Video). Sottomenù per ratio + Personalizzato.
        self.aspect_btn = QToolButton()
        self.aspect_btn.setText(tr("canvas.aspect_btn", "Proporzioni ▾"))
        self.aspect_btn.setPopupMode(QToolButton.InstantPopup)
        self.aspect_btn.setMenu(self._build_aspect_menu())
        bar.addWidget(self.aspect_btn)
        # Bottone «Integra immagini» spostato nel tab «Inpaint / Outpaint», sezione Integra.
        bar.addStretch(1)
        # «Stai disegnando»: in Integra manuale pennello e gomma agiscono SOLO sulla maschera scelta qui
        self.draw_target_lbl = QLabel(tr("canvas.draw_target_lbl", "Stai disegnando:"))
        self._draw_target_cb = QComboBox()
        self._draw_target_cb.addItem(tr("canvas.draw_target_pos", "Posizione"), "position")
        self._draw_target_cb.addItem(tr("canvas.draw_target_ctx", "Contesto"), "context")
        self._draw_target_cb.setToolTip(tr("canvas.draw_target_tt", "Quale maschera di Integra disegnano pennello e gomma: Posizione (rosso) = dove sta il soggetto; Contesto (giallo) = quanto il modello puo ridisegnare attorno."))
        self._draw_target_cb.currentIndexChanged.connect(self._on_draw_target_changed)
        bar.addWidget(self.draw_target_lbl); bar.addWidget(self._draw_target_cb)
        self.draw_target_lbl.setVisible(False); self._draw_target_cb.setVisible(False)    # solo con «Manuale (pennello)»
        bar.addWidget(QLabel(tr("canvas.brush_label")))
        self.brush_sl = QSlider(Qt.Horizontal); self.brush_sl.setRange(3, 200); self.brush_sl.setValue(self.brush_size)
        self.brush_sl.setMaximumWidth(140); self.brush_sl.valueChanged.connect(self.set_brush)
        bar.addWidget(self.brush_sl)
        self.brush_lbl = QLabel(f"{self.brush_size}px"); bar.addWidget(self.brush_lbl)
        lay.addLayout(bar)
        # Scritta ben leggibile della maschera in disegno (colore della maschera, fondo scuro: contrasto > 4,5:1 anche su foto chiare)
        self.draw_banner = QLabel(""); self.draw_banner.setObjectName("draw_banner")
        self.draw_banner.setAlignment(Qt.AlignCenter)
        self.draw_banner_color = BANNER_CTX_FG
        lay.addWidget(self.draw_banner)

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
        # DUE cursori VERTICALI di trasparenza, uno per sovrapposizione (posizione rossa / contesto gialla): solo visualizzazione,
        # non cambiano le maschere inviate. Ognuno vale solo per la sua sovrapposizione. Stanno in UN contenitore: cursori quasi attaccati
        # (OVL_TRANSP_GAP px) e margini a zero (richiesta utente 03/11); il contenitore si nasconde intero quando il motore non usa le maschere.
        self.ovl_transp_sl = {}
        self.ovl_transp_val = {}
        self.ovl_transp_box = QWidget(); self.ovl_transp_box.setObjectName("ovl_transp_box")
        _tb = QVBoxLayout(self.ovl_transp_box); _tb.setContentsMargins(0, 2, 0, 2); _tb.setSpacing(2)
        _cap = QLabel(tr("canvas.ovl_transp_title", "Trasparenza")); _cap.setAlignment(Qt.AlignHCenter); _cap.setWordWrap(True)
        _cap.setToolTip(tr("canvas.ovl_transp_tt", "Trasparenza di questa sovrapposizione: 0 = piena, 100 = invisibile. Solo visualizzazione, non cambia le maschere inviate."))
        _cap.setStyleSheet("QLabel{color:#e8e8f0;font-size:10px;font-weight:bold;}")
        _tb.addWidget(_cap)
        _row = QHBoxLayout(); _row.setContentsMargins(0, 0, 0, 0); _row.setSpacing(OVL_TRANSP_GAP)
        _row.addStretch(1)                                    # i due cursori restano al centro, attaccati fra loro
        _vals = QVBoxLayout(); _vals.setContentsMargins(0, 0, 0, 0); _vals.setSpacing(0)     # i due valori in %, sotto i cursori
        for _key, _fg, _txt, _def in (("position", BANNER_POS_FG, tr("canvas.ovl_transp_pos", "Trasparenza posizione"), self._ovl_transp["position"]),
                                      ("context", BANNER_CTX_FG, tr("canvas.ovl_transp_ctx", "Trasparenza contesto"), self._ovl_transp["context"])):
            tcol = QVBoxLayout(); tcol.setSpacing(1); tcol.setContentsMargins(0, 0, 0, 0)
            # l'iniziale (da "Posizione"/"Contesto", gia tradotti) affianca il colore: il colore non e l'unico veicolo d'informazione
            _word = tr("canvas.draw_target_pos", "Posizione") if _key == "position" else tr("canvas.draw_target_ctx", "Contesto")
            _lab = QLabel((_word[:1] or "?").upper()); _lab.setAlignment(Qt.AlignHCenter); _lab.setFixedWidth(OVL_TRANSP_COL_W)
            _lab.setStyleSheet("QLabel{color:%s;background:%s;border-radius:4px;padding:0;font-size:11px;font-weight:bold;}" % (_fg, BANNER_BG))
            _lab.setToolTip(_txt)
            sl = QSlider(Qt.Vertical); sl.setRange(0, 100); sl.setValue(int(_def)); sl.setMinimumHeight(110); sl.setFixedWidth(OVL_TRANSP_COL_W)
            sl.setToolTip(_txt + ": " + tr("canvas.ovl_transp_tt", "Trasparenza di questa sovrapposizione: 0 = piena, 100 = invisibile. Solo visualizzazione, non cambia le maschere inviate."))
            val = QLabel(f"{_word[:1].upper() or '?'} {int(_def)}%"); val.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            val.setStyleSheet("QLabel{color:%s;font-size:10px;font-weight:bold;}" % _fg); val.setToolTip(_txt)
            sl.valueChanged.connect(lambda v, k=_key: self._on_ovl_transp(k, v))
            tcol.addWidget(_lab); tcol.addWidget(sl, 1, Qt.AlignHCenter)
            _row.addLayout(tcol)
            _vals.addWidget(val)
            self.ovl_transp_sl[_key] = sl; self.ovl_transp_val[_key] = val
        _row.addStretch(1)
        _tb.addLayout(_row, 1)
        _tb.addLayout(_vals)
        self.ovl_transp_box.setFixedWidth(64)
        mid.addWidget(self.ovl_transp_box)
        self.ovl_transp_pos_sl = self.ovl_transp_sl["position"]; self.ovl_transp_ctx_sl = self.ovl_transp_sl["context"]   # attributi pubblici: gui_state li ricorda
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
        cut = QHBoxLayout()
        b_cut = QPushButton(tr("canvas.cutout_btn", "✂ Scontorna"))
        b_cut.setToolTip(tr("canvas.cutout.tt", "Toglie lo sfondo alla foto selezionata: senza questo, "
                                                "lo sfondo (il bianco) viene incollato sulla scena e non "
                                                "viene integrato."))
        b_cut.clicked.connect(self._cutout_selected)
        b_rst = QPushButton(tr("canvas.cutout_undo_btn", "↺ Sfondo"))
        b_rst.setToolTip(tr("canvas.cutout_undo.tt", "Rimette la foto originale, con il suo sfondo."))
        b_rst.clicked.connect(self._restore_selected)
        cut.addWidget(b_cut); cut.addWidget(b_rst)
        self._cutout_btns = (b_cut, b_rst)
        side.addLayout(cut)
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
        # legenda piccola + interruttore mostra/nascondi delle due sovrapposizioni (widget sopra la vista: non e nella scena, non va nell'export)
        self.ovl_legend = QWidget(self.view)
        self.ovl_legend.setObjectName("ovl_legend")
        self.ovl_legend.setStyleSheet("QWidget#ovl_legend{background:%s;border-radius:6px;} QLabel{background:transparent;}" % BANNER_BG)
        _ll = QHBoxLayout(self.ovl_legend); _ll.setContentsMargins(6, 2, 6, 2); _ll.setSpacing(6)
        self.ovl_show_cb = QCheckBox(tr("canvas.ovl_show", "Mostra zone"))
        self.ovl_show_cb.setChecked(True)
        self.ovl_show_cb.setStyleSheet("color:#e8e8f0;")
        self.ovl_show_cb.setToolTip(tr("canvas.ovl_show_tt", "Mostra o nasconde le sovrapposizioni di posizione (rosso) e contesto (giallo)."))
        self.ovl_show_cb.toggled.connect(lambda _v: self.refresh_integrate_overlay(force=True))
        self.ovl_leg_pos = QLabel("<span style='color:%s'>&#9632;</span> <span style='color:#e8e8f0'>%s</span>" % (BANNER_POS_FG, tr("canvas.draw_target_pos", "Posizione")))
        self.ovl_leg_ctx = QLabel("<span style='color:%s'>&#9632;</span> <span style='color:#e8e8f0'>%s</span>" % (BANNER_CTX_FG, tr("canvas.draw_target_ctx", "Contesto")))
        _ll.addWidget(self.ovl_show_cb); _ll.addWidget(self.ovl_leg_pos); _ll.addWidget(self.ovl_leg_ctx)
        self.ovl_legend.adjustSize()
        self._position_overlay()

    def _position_overlay(self):
        ov = getattr(self, "upload_overlay", None)
        if ov is None:
            return
        vp = self.view.viewport()
        ov.move((vp.width() - ov.width()) // 2, (vp.height() - ov.height()) // 2)
        ov.setVisible(self._image_path is None)
        lg = getattr(self, "ovl_legend", None)
        if lg is not None:
            lg.adjustSize(); lg.move(6, max(0, vp.height() - lg.height() - 6)); lg.raise_()

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
        self._update_banner()

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
        self._brush_cursor.setPen(QPen(QColor(CTX_RGB[0], CTX_RGB[1], CTX_RGB[2]) if self._ctx_active() else QColor(POS_RGB[0], POS_RGB[1], POS_RGB[2]), 1.5))
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
        if len(self._lasso_pts) >= 3 and getattr(self, "lasso_ai", True):
            # Punto 22: il LAZO non disegna piu il poligono grezzo: l'oggetto CONTENUTO nell'area lo seleziona l'AI
            # (SAM2 con il riquadro del lazo come prompt). Senza SAM (errore) il lazo resta il poligono (ripiego sotto).
            xs = [pt.x() for pt in self._lasso_pts]
            ys = [pt.y() for pt in self._lasso_pts]
            self._lasso_fallback = list(self._lasso_pts)
            self._clear_seg_marker()
            self._seg_region = ("box", QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))
            self._seg_from_lasso = True
            self.status.setText(tr("canvas.status_lasso_ai", "Lazo: l'AI seleziona l'oggetto nell'area…"))
            self._lasso_reset()
            self.request_segment_region.emit()
            return
        if len(self._lasso_pts) >= 3:
            self.push_undo()
            p = QPainter(self.mask_image)
            p.setBrush(QColor(255, 40, 40, 150)); p.setPen(Qt.NoPen)
            p.drawPolygon(QPolygonF(self._lasso_pts)); p.end()
            self._refresh_mask()
            self.status.setText(tr("canvas.status_lasso_applied"))
        self._lasso_reset()

    def apply_lasso_fallback(self) -> None:
        """Se la selezione AI del lazo non e riuscita: si applica il poligono disegnato (comportamento di prima)."""
        pts = getattr(self, "_lasso_fallback", None)
        if pts and len(pts) >= 3:
            from PySide6.QtGui import QPolygonF
            self.push_undo()
            p = QPainter(self.mask_image)
            p.setBrush(QColor(255, 40, 40, 150)); p.setPen(Qt.NoPen)
            p.drawPolygon(QPolygonF(pts)); p.end()
            self._refresh_mask()
        self._lasso_fallback = None

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
        self._seg_from_lasso = False
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
        out = {"image": image, "mask": str(mpath), "hint": "center-1", "kind": kind}
        if kind == "box" and getattr(self, "_seg_from_lasso", False):
            out["lasso"] = True                                # il lazo chiede «l'oggetto contenuto nell'area»: maschera piu grande che non invade lo sfondo
        if kind == "point":                                   # coordinate nello spazio del composito (= quello inviato a SAM)
            out["point"] = [float(self._seg_region[1].x()), float(self._seg_region[1].y())]
        return out

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
        if hasattr(self, "frame_border"):
            self.frame_border.setRect(QRectF(0, 0, self.aspect_w, self.aspect_h))
        for e in self._entries:        # ridimensiona la maschera di ogni immagine
            e["mask"] = e["mask"].scaled(self.aspect_w, self.aspect_h) if not e["mask"].isNull() else self._new_mask()
        if self._ctx_manual is not None:
            self._ctx_manual = self._ctx_manual.scaled(self.aspect_w, self.aspect_h)
        self._refresh_mask(); self._fit()
        self.integrate_range_changed.emit(self.integrate_range_max())
        self.set_resolution_label(self.aspect_w, self.aspect_h)   # sincronizza etichetta menu Proporzioni

    def _load_dialog(self):
        fn, _ = QFileDialog.getOpenFileName(self, tr("canvas.load_photo_title"), "", tr("canvas.image_filter"))
        if fn:
            self.load_image(fn)

    def _ask_fit_mode(self) -> str:
        """Chiede come inserire la foto nel canvas (richiesta utente): 'canvas' = il canvas prende le
        dimensioni della foto (default, ideale inpaint/outpaint); 'native' = mantiene il canvas, foto a
        dimensione REALE (pixel 1:1, può debordare); 'contain' = mantiene il canvas, adatta la foto (un asse
        100%, l'altro ≤). Preferenza ricordabile in user_prefs 'canvas_fit_mode'."""
        try:
            from core import user_prefs as _up
            pol = str(_up.get("canvas_fit_mode", "ask"))
        except Exception:                                    # noqa: BLE001
            _up, pol = None, "ask"
        if pol in ("canvas", "native", "contain"):
            return pol
        from PySide6.QtWidgets import QMessageBox, QCheckBox
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(tr("canvas.fit.title", "Inserimento foto"))
        box.setText(tr("canvas.fit.msg", "Come inserire la foto nel canvas ({cw}×{ch})?").format(
            cw=int(self.aspect_w), ch=int(self.aspect_h)))
        b_canvas = box.addButton(tr("canvas.fit.canvas", "Canvas = dimensioni foto"), QMessageBox.AcceptRole)
        box.addButton(tr("canvas.fit.contain", "Adatta la foto (contain)"), QMessageBox.ActionRole)
        box.addButton(tr("canvas.fit.native", "Foto a dimensione reale"), QMessageBox.ActionRole)
        cb = QCheckBox(tr("canvas.fit.remember", "Ricorda la scelta (ripristinabile in Impostazioni)"))
        box.setCheckBox(cb)
        box.exec()
        clicked = box.clickedButton()
        if clicked is b_canvas:
            mode = "canvas"
        elif clicked is not None and clicked.text() == tr("canvas.fit.native", "Foto a dimensione reale"):
            mode = "native"
        else:
            mode = "contain"
        if cb.isChecked() and _up is not None:
            _up.set("canvas_fit_mode", mode)
        return mode

    def clear_image(self) -> None:
        """Svuota il canvas: nessuna foto di base, nessun layer, nessuna maschera. Comando VERO della GUI (pulsante 🗑):
        senza di esso, una foto caricata una volta restava per sempre l'immagine iniziale di OGNI video (nel codice reale
        `use_image = bool(frame1_tab.image_path)`), e non c'era modo di fare un text-to-video senza riavviare l'app."""
        for e in list(self._entries)[1:]:                          # layer importati: via dalla scena
            try:
                self.scene.removeItem(e["item"])
            except Exception:                                      # noqa: BLE001
                pass
        self._entries = self._entries[:1]
        self._entries[0]["mask"] = self._new_mask()
        self._active = 0
        self.img_item.setPixmap(QPixmap())
        self.img_item.setScale(1.0)
        self.img_item.setPos(0, 0)
        self._image_path = None
        self._clear_seg_marker()
        self._undo.clear(); self._redo.clear()
        self._ctx_reset()
        self._rebuild_layer_list()
        self._refresh_mask()
        self.status.setText(tr("canvas.status_cleared", "Canvas svuotato: nessuna immagine iniziale."))
        self.image_changed.emit("")

    def load_image(self, path: str, fit_mode: str | None = None):
        pix = QPixmap(path)
        if pix.isNull():
            return
        pw, ph = max(1, pix.width()), max(1, pix.height())
        # Modalità inserimento: se il canvas ha già una dimensione utile, CHIEDI (canvas/native/contain);
        # al primo caricamento (nessun canvas) usa 'canvas' (adotta la foto) senza chiedere.
        has_canvas = int(getattr(self, "aspect_w", 0)) >= 64 and int(getattr(self, "aspect_h", 0)) >= 64
        if fit_mode is None:
            fit_mode = self._ask_fit_mode() if has_canvas else "canvas"
        if fit_mode == "canvas" or not has_canvas:
            # Il canvas adotta le dimensioni della foto (cap lato lungo 1536, multipli di 16) — inpaint/
            # outpaint sulla foto REALE. Le modalità native/contain NON toccano aspect_w/h (canvas fisso).
            _sc = min(1.0, 1536.0 / max(pw, ph))
            tw = max(256, int(round(pw * _sc / 16)) * 16)
            th = max(256, int(round(ph * _sc / 16)) * 16)
            self.set_aspect(tw, th)
        self._active = 0; self._rebuild_layer_list()   # la nuova foto è la BASE
        self._image_path = path
        self.image_changed.emit(path)
        self.img_item.setPixmap(pix)
        if fit_mode == "native":
            s = 1.0                                     # dimensione REALE (pixel 1:1), può debordare dal canvas
        else:                                           # 'canvas'/'contain' → contain (un asse 100%, l'altro ≤)
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
        if fn:
            self.add_layer(fn)

    def add_layer(self, path: str, mask_path: str | None = None, *, index: int = 0) -> bool:
        """Aggiunge una foto come LAYER sul canvas. API pubblica: la usa anche «Integra» per portare
        sul canvas le foto della lista laterale (che altrimenti non entrerebbero mai nel composito).

        `mask_path` è la maschera DELLA FOTO (spazio-immagine): la riproiettiamo nello spazio-scena con
        la stessa trasformazione del layer, altrimenti finirebbe fuori posto. `index` sfalsa le foto
        aggiunte in blocco, così non si sovrappongono tutte nello stesso punto.
        """
        pix = QPixmap(path)
        if pix.isNull():
            return False
        it = QGraphicsPixmapItem(pix)
        it.setFlag(QGraphicsPixmapItem.ItemIsMovable, self.mode == "move")
        s = min(self.aspect_w / pix.width(), self.aspect_h / pix.height()) * 0.6
        it.setScale(s)
        off = (index % 4) * 0.08                                 # a cascata: restano tutte afferrabili
        it.setPos(self.aspect_w * (0.2 + off), self.aspect_h * (0.2 + off))
        self.scene.addItem(it)
        mask = self._new_mask()
        if mask_path and os.path.exists(mask_path):
            m = QImage(mask_path)
            if not m.isNull():
                m = m.scaled(pix.width(), pix.height(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                pm = QPainter(mask)
                pm.setTransform(it.sceneTransform())             # spazio-immagine → spazio-scena
                pm.drawImage(0, 0, m.convertToFormat(QImage.Format_Alpha8))
                pm.end()
        self._entries.append({"item": it, "mask": mask, "name": Path(path).name,
                              "base": False, "path": path})
        # SCONTORNO AUTOMATICO (nessuna azione separata da fare): lo sfondo di una foto stock è quasi
        # sempre un RETTANGOLO BIANCO, e per il modello un rettangolo bianco su una foto "è" uno
        # screenshot — a denoise alto ci disegna dentro finestre di dialogo e paesaggi (visto il 13/07).
        # Tolto lo sfondo, resta il solo soggetto e la scena sotto è quella vera.
        # OPT-IN (default OFF): lo scontorno u2net gira SINCRONO sul thread GUI → all'aggiunta foto BLOCCAVA
        # l'app (specie al 1° load del modello ONNX). Ora l'aggiunta è istantanea; lo scontorno si fa a mano
        # (azione dedicata) o attivando `_auto_cutout_on_add`. TODO: spostarlo in un thread di sfondo (l'apply
        # alla scene deve restare sul thread GUI via signal).
        if getattr(self, "_auto_cutout_on_add", False) \
                and not pix.hasAlphaChannel() and (Path.home() / ".u2net" / "u2net.onnx").exists():
            self.cutout_layer(len(self._entries) - 1, ask=False)   # ask=False → nessun download a sorpresa
        self._reindex_z()
        self._rebuild_layer_list()
        self.layer_list.setCurrentRow(len(self._entries) - 1)   # seleziona la nuova (mostra la sua maschera)
        return True

    # ---------- scontorno (rimozione sfondo) ----------
    def _cutout_selected(self):
        e = self._entries[self._active]
        if e["base"]:
            QMessageBox.information(self, tr("canvas.cutout_btn", "✂ Scontorna"),
                                    tr("canvas.cutout.pick_layer",
                                       "Seleziona una FOTO AGGIUNTA (non lo sfondo di base).")); return
        self.cutout_layer(self._active)

    def _restore_selected(self):
        e = self._entries[self._active]
        if e["base"] or not e.get("path"):
            return
        pix = QPixmap(e["path"])
        if not pix.isNull():
            e["item"].setPixmap(pix)                    # torna la foto originale, sfondo compreso
            e.pop("cutout", None)
            self.status.setText(tr("canvas.cutout_restored", "Sfondo originale ripristinato."))

    def cutout_layer(self, idx: int, *, ask: bool = True) -> bool:
        """Toglie lo SFONDO alla foto-layer (rembg/u2net, su CPU).

        È il motivo per cui «Integra» lasciava il rettangolo bianco attorno al soggetto: una foto opaca
        porta con sé il proprio sfondo, che finisce nel composito e — peggio — diventa parte della
        silhouette «da non toccare». Scontornata, resta il solo soggetto e tutto ciò che gli sta intorno
        è zona che il modello rigenera e integra.

        CPU forzata: onnxruntime-gpu creerebbe un contesto CUDA NEL PROCESSO GUI, che manda in crash i
        worker GPU isolati. Il modello (~176MB) si scarica al primo uso, previa conferma.
        """
        e = self._entries[idx]
        src = e.get("path")
        if e["base"] or not src or not os.path.exists(src):
            return False
        home_model = Path.home() / ".u2net" / "u2net.onnx"
        if not home_model.exists() and ask:
            if QMessageBox.question(
                    self, tr("canvas.cutout_dl_title", "Scarico il modello di scontorno?"),
                    tr("canvas.cutout_dl_msg",
                       "Per togliere lo sfondo serve il modello u2net (~176 MB), che non è ancora sul "
                       "disco. Lo scarico ora? (una sola volta)"),
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) != QMessageBox.Yes:
                return False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            from PIL import Image as _PILImage
            from rembg import new_session, remove
            if getattr(self, "_rembg", None) is None:
                self._rembg = new_session("u2net", providers=["CPUExecutionProvider"])
            out = remove(_PILImage.open(src).convert("RGB"), session=self._rembg,
                         post_process_mask=True).convert("RGBA")
            # PULIZIA DELL'ALPHA: rembg lascia un alone di alpha intermedia sullo sfondo (nel test:
            # solo il 7% dei pixel era pienamente opaco, l'83% trasparente, il resto un VELO). Quel velo
            # è esattamente il rettangolo chiaro che si vedeva attorno all'oggetto integrato. Sotto 60 →
            # sfondo (via del tutto), sopra 190 → soggetto pieno, in mezzo una rampa per non seghettare.
            if np is not None:
                arr = np.array(out)
                a = arr[..., 3].astype(np.float32)
                arr[..., 3] = np.clip((a - 60.0) * (255.0 / 130.0), 0, 255).astype(np.uint8)
                out = _PILImage.fromarray(arr, "RGBA")
            qi = QImage(out.tobytes("raw", "RGBA"), out.width, out.height, QImage.Format_RGBA8888).copy()
            e["item"].setPixmap(QPixmap.fromImage(qi))   # stesse dimensioni → scala/posizione invariate
            e["cutout"] = True
            self.status.setText(tr("canvas.cutout_done", "Sfondo rimosso: ora il soggetto si integra."))
            return True
        except Exception as ex:                          # noqa: BLE001
            QMessageBox.warning(self, tr("canvas.cutout_btn", "✂ Scontorna"),
                                tr("canvas.cutout_failed", "Scontorno non riuscito: {e}").format(e=ex))
            return False
        finally:
            QApplication.restoreOverrideCursor()

    def _alpha_silhouette(self, e) -> QImage | None:
        """Silhouette del layer ricavata dalla sua TRASPARENZA (PNG con alpha o foto scontornata),
        proiettata nello spazio-scena. None se la foto è opaca (là non sappiamo dov'è il soggetto)."""
        pm = e["item"].pixmap()
        if pm.isNull() or not pm.hasAlphaChannel() or np is None:
            return None
        img = pm.toImage().convertToFormat(QImage.Format_ARGB32)
        if self._alpha_array(img).min() >= 250:          # opaca: niente da cui dedurre il contorno
            return None
        W, H = self.aspect_w, self.aspect_h
        sil = QImage(W, H, QImage.Format_ARGB32); sil.fill(Qt.transparent)
        wm = QImage(img.size(), QImage.Format_ARGB32); wm.fill(QColor(255, 255, 255))
        wm.setAlphaChannel(img.convertToFormat(QImage.Format_Alpha8))
        p = QPainter(sil)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.setTransform(e["item"].sceneTransform())
        p.drawImage(0, 0, wm)
        p.end()
        return sil

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
    def _ctx_active(self) -> bool:
        """True se pennello/gomma/annulla agiscono sulla maschera di CONTESTO (Integra manuale + selettore «Contesto»)."""
        return (self.integrate_ctx_mode == "manual" and self.integrate_ctx_enabled
                and self.integrate_draw_target == "context" and self._ctx_manual is not None)

    def push_undo(self):
        if self._ctx_active():
            self._ctx_undo.append(self._ctx_manual.copy())
            if len(self._ctx_undo) > MAX_UNDO:
                self._ctx_undo.pop(0)
            self._ctx_redo.clear()
            return
        self._undo.append(self.mask_image.copy())
        if len(self._undo) > MAX_UNDO:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self):
        if self._ctx_active():
            if self._ctx_undo:
                self._ctx_redo.append(self._ctx_manual.copy())
                self._ctx_manual = self._ctx_undo.pop(); self.refresh_integrate_overlay(force=True)
            return
        if self._undo:
            self._redo.append(self.mask_image.copy())
            self.mask_image = self._undo.pop(); self._refresh_mask()

    def redo(self):
        if self._ctx_active():
            if self._ctx_redo:
                self._ctx_undo.append(self._ctx_manual.copy())
                self._ctx_manual = self._ctx_redo.pop(); self.refresh_integrate_overlay(force=True)
            return
        if self._redo:
            self._undo.append(self.mask_image.copy())
            self.mask_image = self._redo.pop(); self._refresh_mask()

    def paint_at(self, sp: QPointF):
        ctx = self._ctx_active()                      # Integra manuale, «Stai disegnando: Contesto» → pennello GIALLO sulla maschera di contesto
        p = QPainter(self._ctx_manual if ctx else self.mask_image)
        if self.mode == "erase":
            p.setCompositionMode(QPainter.CompositionMode_Clear); p.setBrush(Qt.transparent)
        else:
            p.setBrush(QColor(CTX_RGB[0], CTX_RGB[1], CTX_RGB[2], 150) if ctx else QColor(255, 40, 40, 150))
        p.setPen(Qt.NoPen)
        p.drawEllipse(sp, self.brush_size/2, self.brush_size/2); p.end()
        if ctx:
            self._ctx_touched = True
            self.refresh_integrate_overlay(force=True)
        else:
            self._refresh_mask()

    def fill_at(self, sp: QPointF):
        # flood fill semplice sull'alpha della maschera, partendo dal punto
        x, y = int(sp.x()), int(sp.y())
        if not (0 <= x < self.aspect_w and 0 <= y < self.aspect_h):
            return
        ctx = self._ctx_active()
        img = self._ctx_manual if ctx else self.mask_image
        target_on = (img.pixel(x, y) >> 24) & 0xFF > 0
        fillcol = QColor(CTX_RGB[0], CTX_RGB[1], CTX_RGB[2], 150).rgba() if ctx else QColor(255, 40, 40, 150).rgba()
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
        if ctx:
            self._ctx_touched = True
            self.refresh_integrate_overlay(force=True)
        else:
            self._refresh_mask()

    def _refresh_mask(self):
        self.mask_item.setPixmap(QPixmap.fromImage(self.mask_image))

    def clear_mask(self, silent: bool = False):
        if not silent and self._ctx_active():
            self.clear_context()                                   # «Pulisci» con «Stai disegnando: Contesto» svuota il contesto, non la posizione
            return
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
        hidden = [self.mask_item] + list(self._handles) + [i for i in (self._ovl_pos, self._ovl_ctx) if i is not None]
        if self._seg_marker is not None:
            hidden.append(self._seg_marker)
        if self._lasso_item is not None:
            hidden.append(self._lasso_item)
        if self._brush_cursor is not None:
            hidden.append(self._brush_cursor)
        for it in hidden:
            it.setVisible(False)
        # Il contorno tratteggiato blu del frame (#46a, 4 px: mezzo dentro il frame) è solo una guida a video: nell'export
        # restava come riga blu di 2 px sul bordo dell'immagine, e Klein/Qwen la leggevano come parte della scena (task 16a).
        _border_vis = self.frame_border.isVisible()
        self.frame_border.setVisible(False)                     # guida solo a video: mai nell'export
        try:
            self.scene.render(painter, target, source)
        finally:
            self.frame_border.setVisible(_border_vis)
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
            # OUTPAINT ignora SEMPRE la maschera disegnata dall'utente (anche se presente): l'area da generare
            # è per definizione SOLO l'estensione (la cornice non coperta dalla foto) = _border_mask (geometrica),
            # mai il disegno interno alla foto.
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

    def _morph(self, sil: QImage, r: int, *, grow: bool = True) -> QImage:
        """Ingrossa (grow) o assottiglia (erode) una sagoma di r pixel — max/min filter separabile.

        Serve per due cose opposte e complementari:
          • dilate → ZONA DI EDIT: il modello lavora attorno all'oggetto (mano, ombra, contatto).
          • erode  → ZONA PROTETTA: l'oggetto che HAI SCELTO non deve essere rigenerato, o il modello
                     te ne inventa un altro (letteralmente: tre calici al posto del tuo).
        """
        if np is None or r <= 0:
            return sil
        a = self._alpha_array(sil).astype(np.uint8)
        for axis in (0, 1):
            pad = [(0, 0), (0, 0)]; pad[axis] = (r, r)
            p = np.pad(a, pad, mode="constant", constant_values=0 if grow else 255)
            st = np.stack([np.roll(p, s, axis=axis) for s in range(-r, r + 1)], 0)
            a = st.max(0) if grow else st.min(0)
            a = a[r:-r, :] if axis == 0 else a[:, r:-r]
        out = QImage(sil.width(), sil.height(), QImage.Format_ARGB32)
        out.fill(QColor(255, 255, 255))
        alpha = QImage(np.ascontiguousarray(a).data, sil.width(), sil.height(),
                       sil.width(), QImage.Format_Alpha8).copy()
        out.setAlphaChannel(alpha)
        return out

    def _binarize(self, sil: QImage, thr: int) -> QImage:
        """Sagoma piena: alpha ≥ thr → 255, altrimenti 0. Serve per «è oggetto o non lo è» (vetro, capelli)."""
        if np is None:
            return sil
        a = (self._alpha_array(sil) >= thr).astype(np.uint8) * 255
        out = QImage(sil.width(), sil.height(), QImage.Format_ARGB32)
        out.fill(QColor(255, 255, 255))
        out.setAlphaChannel(QImage(np.ascontiguousarray(a).data, sil.width(), sil.height(),
                                   sil.width(), QImage.Format_Alpha8).copy())
        return out

    def _alpha_array(self, m: QImage):
        """Canale alpha come array numpy (H×W, uint8). La COPIA è obbligatoria: `a` è un QImage locale e
        una vista sul suo buffer diventerebbe memoria liberata appena la funzione ritorna (letture
        casuali — erode e dilate producevano lo stesso risultato)."""
        a = m.convertToFormat(QImage.Format_Alpha8)
        buf = np.frombuffer(a.constBits(), dtype=np.uint8, count=a.sizeInBytes())
        return np.array(buf.reshape(a.height(), a.bytesPerLine())[:, :a.width()])

    def _mask_bounds(self, m: QImage) -> QRectF:
        """Riquadro dei pixel dipinti in una maschera (vuota → rettangolo nullo)."""
        if np is not None:
            a = self._alpha_array(m)
            ys, xs = np.nonzero(a > 0)
            if not len(xs):
                return QRectF()
            return QRectF(float(xs.min()), float(ys.min()),
                          float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
        x0 = y0 = 10**9; x1 = y1 = -1                      # senza numpy: scansione a campione
        step = max(1, m.width() // 256)
        for y in range(0, m.height(), step):
            for x in range(0, m.width(), step):
                if (m.pixel(x, y) >> 24) & 0xFF:
                    x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
        if x1 < 0:
            return QRectF()
        return QRectF(x0, y0, x1 - x0 + step, y1 - y0 + step)

    def _auto_cutout_opaque_layers(self) -> list[int]:
        """Prima di Integra, ogni foto-layer OPACA e senza maschera disegnata viene scontornata (u2net, CPU), se il modello e gia sul disco.
        Senza, la foto opaca e un rettangolo: finisce intero nel composito, il ripiego «alone bianco» le disegna attorno una cornice bianca e il
        modello (a) la lascia li, incollata (aspetto da stitch) o (b) la ridisegna come cornice (commenti dell'utente sulle revisioni del 31/10 e
        1/11, 12 casi modifica_integra_* e componi_*). Non scarica nulla (niente sorprese): senza il modello resta il ripiego di prima.
        Ritorna gli indici scontornati."""
        done: list[int] = []
        if not (Path.home() / ".u2net" / "u2net.onnx").exists():
            return done
        for i, e in enumerate(self._entries):
            if i == 0 or e.get("base") or e.get("cutout") or self._mask_filled(e["mask"]):
                continue
            pm = e["item"].pixmap()
            if pm.isNull() or pm.hasAlphaChannel() and self._alpha_silhouette(e) is not None:
                continue                                  # gia scontornata / PNG con trasparenza vera
            if self.cutout_layer(i, ask=False):
                done.append(i)
        return done

    # ── due maschere di Integra: POSIZIONE (dove sta il soggetto) e CONTESTO (zona che il modello puo rigenerare) ─────────────────────────────────────
    # Prima c'era un solo tracciato: la zona da rigenerare era il riquadro della foto incollata + il 5 % del lato lungo. Il giudizio dell'utente del 03/11
    # (Kontext: «la maschera deve essere piu grande della posizione della ragazza, perche il modello tiene conto solo di cio che c'e sotto la maschera»)
    # chiede di separarle. `integrate_context_pct` (frazione del lato lungo del riquadro del soggetto) o `integrate_context_px` allargano il CONTESTO,
    # dilatandolo con la stessa forma; None = comportamento di sempre. Per ora e un parametro interno (casi di prova): diventa comando in GUI solo
    # dopo l'ok dell'utente.
    integrate_context_pct: float | None = None
    integrate_context_px: int | None = None
    # «Rigenera»: invece di dare al modello il composito con la foto incollata (che lui riproduce quasi identico = «solo incollata», giudizio 03/11),
    # export_integrate scrive anche la SCENA SENZA le foto (`base`) e i soggetti scontornati su fondo neutro (`layer_refs`): il modello li usa come
    # riferimenti e RIDISEGNA la persona nella zona della maschera. Parametro interno dei casi di prova (`_rigenera`), non ancora un comando in GUI.
    integrate_regenerate: bool = False

    @staticmethod
    def _dilate_alpha(img: QImage, px: int) -> QImage:
        """Dilatazione di `px` pixel di una maschera (alpha) con kernel ellittico: la zona si allarga attorno alla stessa forma."""
        if px <= 0 or np is None:
            return img
        try:
            import cv2 as _cv2
        except Exception:                                    # noqa: BLE001
            return img
        a = img.convertToFormat(QImage.Format_Alpha8)
        buf = np.frombuffer(a.constBits(), dtype=np.uint8, count=a.sizeInBytes())
        arr = np.array(buf.reshape(a.height(), a.bytesPerLine())[:, :a.width()])
        src = arr > 0
        if not src.any():
            return img
        # Dilatazione = tutti i pixel entro `px` dalla forma (distanza euclidea esatta): equivale al kernel ellittico, ma costa come un
        # solo passaggio anche per raggi enormi (oltre il 100 % del soggetto un kernel (2px+1)^2 sarebbe lentissimo). Il limite e il bordo.
        dist = _cv2.distanceTransform((~src).astype(np.uint8), _cv2.DIST_L2, _cv2.DIST_MASK_PRECISE)
        d = np.ascontiguousarray(((dist <= float(px)) | src).astype(np.uint8) * 255)
        out = QImage(d.data, d.shape[1], d.shape[0], d.strides[0], QImage.Format_Alpha8).copy()
        res = QImage(img.width(), img.height(), QImage.Format_ARGB32)
        res.fill(QColor(255, 255, 255))
        res.setAlphaChannel(out)
        return res

    # ── Integra: parametri dal pannello, sovrapposizioni, banner ─────────────────────────────────────────────────────────
    def set_integrate_context(self, mode: str | None = None, px: int | None = None, enabled: bool | None = None, pct: float | None = None) -> None:
        """Il pannello Integra comanda il canvas. `mode` = "auto" | "manual"; `px` = margine in PIXEL attorno alla sagoma (0 = contesto = sagoma);
        `pct` = alias: frazione della LARGHEZZA della sagoma (casi vecchi); `enabled` = il motore usa la maschera di contesto (False per Krea 2 Edit:
        niente sovrapposizione gialla, selettore spento)."""
        if enabled is not None:
            self.integrate_ctx_enabled = bool(enabled)
            self._apply_masks_visibility()
        if px is not None:
            self.integrate_context_px = max(0, int(px)); self.integrate_context_pct = None
        elif pct is not None:
            self.integrate_context_pct = max(0.0, float(pct)); self.integrate_context_px = None
        if mode in ("auto", "manual") and mode != self.integrate_ctx_mode:
            self.integrate_ctx_mode = mode
            if mode == "manual":
                if self._ctx_manual is None or not self._ctx_touched:
                    self.context_from_auto()                          # parte PRECOMPILATO con la zona automatica del momento
            else:
                self.integrate_draw_target = "position"
                self._draw_target_cb.blockSignals(True); self._draw_target_cb.setCurrentIndex(0); self._draw_target_cb.blockSignals(False)
        manual = self.integrate_ctx_mode == "manual" and self.integrate_ctx_enabled
        self.draw_target_lbl.setVisible(manual); self._draw_target_cb.setVisible(manual)
        if not manual and self.integrate_draw_target != "position":
            self.integrate_draw_target = "position"
            self._draw_target_cb.blockSignals(True); self._draw_target_cb.setCurrentIndex(0); self._draw_target_cb.blockSignals(False)
        self._update_banner()
        self.refresh_integrate_overlay(force=True)

    def _ovl_schedule(self, *_a) -> None:
        try:
            self._ovl_timer.start()
        except RuntimeError:                                     # canvas gia distrutto (chiusura): niente da aggiornare
            pass

    def _on_draw_target_changed(self, _i: int) -> None:
        self.integrate_draw_target = self._draw_target_cb.currentData() or "position"
        self._update_banner()
        if self.brush_cursor_visible():
            self._hide_brush_cursor()

    def brush_cursor_visible(self) -> bool:
        return getattr(self, "_brush_cursor", None) is not None and self._brush_cursor.isVisible()

    def _on_ovl_transp(self, key: str, v: int) -> None:
        """Trasparenza di UNA sovrapposizione (solo visualizzazione: le maschere esportate non cambiano)."""
        self._ovl_transp[key] = max(0, min(100, int(v)))
        _w = self.ovl_transp_val[key]; _w.setText(_w.text().split(" ")[0] + f" {self._ovl_transp[key]}%")
        self._apply_overlay_opacity()

    def masks_visible(self) -> bool:
        """Le due maschere e i loro controlli si vedono SOLO in modalita Integra (pagina Modifica del pannello), con un motore che le usa e con almeno un
        layer importato da integrare. Altrimenti sono NASCOSTI (non disattivati)."""
        return bool(self.integrate_ctx_enabled and self.integrate_mode_active and self.has_layers())

    def set_integrate_mode_active(self, active: bool) -> None:
        self.integrate_mode_active = bool(active)
        self._apply_masks_visibility()
        self.refresh_integrate_overlay(force=True)

    def _apply_masks_visibility(self) -> None:
        """Scritta «Stai disegnando», legenda «Mostra zone», cursori di trasparenza, selettore della maschera da disegnare e sovrapposizioni rossa/gialla:
        visibili solo se `masks_visible()` (modalita Integra + motore che usa le maschere + layer presenti)."""
        on = self.masks_visible()
        for w in (getattr(self, "draw_banner", None), getattr(self, "ovl_legend", None), getattr(self, "ovl_transp_box", None)):
            if w is not None:
                w.setVisible(on)
        if not on:
            for w in (getattr(self, "draw_target_lbl", None), getattr(self, "_draw_target_cb", None)):
                if w is not None:
                    w.setVisible(False)
            if self.integrate_draw_target != "position":
                self.integrate_draw_target = "position"
        else:
            manual = self.integrate_ctx_mode == "manual"
            self.draw_target_lbl.setVisible(manual); self._draw_target_cb.setVisible(manual)
        self._position_overlay()

    def set_layer_tools_allowed(self, allowed: bool) -> None:
        """«✂ Scontorna» e «↺ Sfondo» servono solo dove i layer importati vengono composti sulla scena (decide frame1_tab): qui si nascondono, mai disattivano."""
        self._cutout_allowed = bool(allowed)
        self._refresh_cutout_buttons()

    def _refresh_cutout_buttons(self) -> None:
        on = getattr(self, "_cutout_allowed", True) and self.has_layers()
        for b in getattr(self, "_cutout_btns", ()):
            b.setVisible(bool(on))

    def _apply_overlay_opacity(self) -> None:
        self._ovl_pos.setOpacity(1.0 - self._ovl_transp["position"] / 100.0)
        self._ovl_ctx.setOpacity(1.0 - self._ovl_transp["context"] / 100.0)

    def silhouette_width(self) -> int:
        """Larghezza (px) della sagoma vera dei soggetti incollati (estensione orizzontale), 0 se non c'e nulla: serve alla % equivalente del pannello."""
        base = getattr(self, "_ovl_base", None)
        if base is None:
            try:
                obj, region, _w, anyr = self._build_integrate_masks()
            except Exception:                                    # noqa: BLE001
                return 0
        else:
            obj, region, anyr = base[0], base[1], base[2]
        if not anyr or np is None:
            return 0
        b = self._mask_bounds(obj)
        if b.isNull():
            b = self._mask_bounds(region)
        return 0 if b.isNull() else int(round(b.width()))

    def integrate_range_max(self) -> int:
        """Massimo del cursore del margine: meta del lato lungo dell'immagine (il campo numerico accetta di piu)."""
        return max(1, int(max(self.aspect_w, self.aspect_h) // 2))

    def _ctx_pct_label(self) -> int:
        if self.integrate_context_px is not None:
            return int(self.integrate_context_px)
        if self.integrate_context_pct is not None:
            return int(round(self.integrate_context_pct * (self.silhouette_width() or 0)))
        return DEFAULT_CTX_PX

    def _update_banner(self) -> None:
        """Scritta sopra il canvas: quale maschera si sta disegnando, nel colore di quella maschera, su fondo scuro (leggibile anche su foto chiare)."""
        lab = getattr(self, "draw_banner", None)
        if lab is None:
            return
        if self.integrate_ctx_mode == "manual" and self.integrate_ctx_enabled:
            if self.integrate_draw_target == "context":
                txt, col = tr("canvas.banner_ctx", "Stai disegnando: CONTESTO"), BANNER_CTX_FG
            else:
                txt, col = tr("canvas.banner_pos", "Stai disegnando: POSIZIONE"), BANNER_POS_FG
        elif self.integrate_ctx_enabled:
            txt, col = tr("canvas.banner_auto", "Contesto automatico: {n} px").format(n=self._ctx_pct_label()), BANNER_CTX_FG
        else:
            txt, col = tr("canvas.banner_pos", "Stai disegnando: POSIZIONE"), BANNER_POS_FG
        lab.setVisible(self.masks_visible())                   # fuori da Integra / motore senza maschere / niente layer: scritta nascosta
        self.draw_banner_color = col
        lab.setText(txt)
        lab.setStyleSheet("QLabel#draw_banner{color:%s;background:%s;font-size:15px;font-weight:bold;border-radius:6px;padding:3px 10px;}" % (col, BANNER_BG))

    def _ctx_reset(self) -> None:
        self._ctx_manual = None; self._ctx_touched = False
        self._ctx_undo.clear(); self._ctx_redo.clear()
        if self.integrate_ctx_mode == "manual":
            self.context_from_auto(); self._ctx_touched = False

    def _new_ctx_image(self) -> QImage:
        m = QImage(self.aspect_w, self.aspect_h, QImage.Format_ARGB32); m.fill(Qt.transparent)
        return m

    def auto_context_alpha(self):
        """Zona di contesto AUTOMATICA al valore corrente (= quella che verrebbe esportata in modalita automatica), come QImage alpha."""
        obj, region, _w, any_region = self._build_integrate_masks()
        if not any_region:
            return None
        sav = self.integrate_ctx_mode
        self.integrate_ctx_mode = "auto"
        try:
            out, _px = self._integrate_context(obj, region, any_region)
        finally:
            self.integrate_ctx_mode = sav
        return out

    def context_from_auto(self) -> None:
        """Precompila (o ripristina) la maschera di contesto manuale con la zona automatica del momento."""
        if self._ctx_manual is not None:
            self._ctx_undo.append(self._ctx_manual.copy())
        auto = self.auto_context_alpha()
        img = self._new_ctx_image()
        if auto is not None:
            tint = QImage(self.aspect_w, self.aspect_h, QImage.Format_ARGB32); tint.fill(QColor(CTX_RGB[0], CTX_RGB[1], CTX_RGB[2], 150))
            tint.setAlphaChannel(auto.convertToFormat(QImage.Format_Alpha8).convertToFormat(QImage.Format_Alpha8))
            img = tint
        self._ctx_manual = img; self._ctx_touched = True
        self._ctx_redo.clear()
        self.refresh_integrate_overlay(force=True)

    def clear_context(self) -> None:
        """«Svuota»: toglie tutta la maschera di contesto manuale (resta la sagoma: contesto = posizione)."""
        if self._ctx_manual is not None:
            self._ctx_undo.append(self._ctx_manual.copy())
        self._ctx_manual = self._new_ctx_image(); self._ctx_touched = True
        self._ctx_redo.clear()
        self.refresh_integrate_overlay(force=True)

    @staticmethod
    def _tint(alpha_src: QImage, rgb) -> QImage:
        out = QImage(alpha_src.width(), alpha_src.height(), QImage.Format_ARGB32); out.fill(QColor(rgb[0], rgb[1], rgb[2]))
        out.setAlphaChannel(alpha_src.convertToFormat(QImage.Format_Alpha8))
        return out

    def _ovl_signature(self):
        """(firma della scena, firma delle impostazioni): la prima cambia se muovi/ridimensioni/aggiungi layer o disegni la posizione, la seconda se
        cambi modo/percentuale/contesto manuale. Cosi un tratto di pennello CONTESTO non rifa il calcolo della sagoma."""
        ents = []
        for e in self._entries:
            it = e["item"]
            pm = it.pixmap()
            ents.append((id(it), round(it.x(), 2), round(it.y(), 2), round(it.scale(), 4), round(it.rotation(), 3),
                         pm.cacheKey() if not pm.isNull() else 0, e["mask"].cacheKey(), it.isVisible()))
        scene_sig = (self._active, self.aspect_w, self.aspect_h, tuple(ents))
        set_sig = (self.integrate_ctx_mode, self.integrate_ctx_enabled, self.integrate_context_pct, self.integrate_context_px,
                   self.ovl_show_cb.isChecked() if hasattr(self, "ovl_show_cb") else True,
                   self._ctx_manual.cacheKey() if self._ctx_manual is not None else 0)
        return scene_sig, set_sig

    def refresh_integrate_overlay(self, force: bool = False) -> None:
        """Ridisegna in tempo reale le due sovrapposizioni: POSIZIONE (rosso) e CONTESTO (giallo), calcolate con lo STESSO codice dell'export.
        Sono item separati dalla foto: non finiscono mai nell'immagine esportata."""
        if self._ovl_pos is None:
            return
        sig = self._ovl_signature()
        if not force and sig == self._ovl_sig:
            return
        scene_changed = self._ovl_sig is None or sig[0] != self._ovl_sig[0]
        self._ovl_sig = sig
        self._apply_overlay_opacity()
        show = self.ovl_show_cb.isChecked() if hasattr(self, "ovl_show_cb") else True
        empty = QPixmap()
        try:
            if scene_changed or getattr(self, "_ovl_base", None) is None:
                obj, region, _w, any_region = self._build_integrate_masks()
                # posizione = sagoma; dove la maschera del layer selezionato e gia mostrata dal pennello (mask_item) non la ridisegno (niente rosso doppio)
                pos_alpha = obj.copy()
                act = self._entries[self._active] if 0 <= self._active < len(self._entries) else None
                if act is not None and self._mask_filled(act["mask"]):
                    pp = QPainter(pos_alpha); pp.setCompositionMode(QPainter.CompositionMode_DestinationOut)
                    pp.drawImage(0, 0, act["mask"].convertToFormat(QImage.Format_ARGB32)); pp.end()
                self._ovl_base = (obj, region, any_region, QPixmap.fromImage(self._tint(pos_alpha, POS_RGB)) if any_region else empty)
            obj, region, any_region, pos_pm = self._ovl_base
        except Exception:                                        # noqa: BLE001  (l'anteprima non deve mai rompere il canvas)
            self._ovl_pos.setPixmap(empty); self._ovl_ctx.setPixmap(empty); return
        self.ctx_geometry_changed.emit()
        if not any_region or not show:
            self._ovl_pos.setPixmap(empty); self._ovl_ctx.setPixmap(empty); return
        if not self.masks_visible():                          # fuori da Integra o motore senza maschere: nemmeno il rosso della posizione
            self._ovl_pos.setPixmap(empty); self._ovl_ctx.setPixmap(empty); return
        self._ovl_pos.setPixmap(pos_pm)
        if self.integrate_ctx_mode == "manual" and self._ctx_manual is not None:
            ctx_show = self._ctx_manual                            # in manuale si vede il TRATTO (giallo) cosi come e disegnato
        else:
            ctx_show, _px = self._integrate_context(obj, region, any_region)
        self._ovl_ctx.setPixmap(QPixmap.fromImage(self._tint(ctx_show, CTX_RGB)))
        self.ctx_geometry_changed.emit()


    def _build_integrate_masks(self):
        """Costruisce le tre maschere di base di Integra (senza contesto allargato): oggetto/sagoma, zona di edit e alone da sbiancare.
        Usata da export_integrate e dall'anteprima sul canvas, cosi quello che vedi e quello che parte."""
        W, H = self.aspect_w, self.aspect_h
        frame = QRectF(0, 0, W, H)
        grow = max(24, int(0.05 * max(W, H)))
        obj = QImage(W, H, QImage.Format_ARGB32); obj.fill(Qt.transparent)        # dove c'è l'oggetto
        region = QImage(W, H, QImage.Format_ARGB32); region.fill(Qt.transparent)  # zona di edit
        wreg = QImage(W, H, QImage.Format_ARGB32); wreg.fill(Qt.transparent)      # zona da SBIANCARE
        po = QPainter(obj); pr = QPainter(region); pv = QPainter(wreg)
        for _p in (po, pr, pv):
            _p.setPen(Qt.NoPen)
        any_region = False
        for e in self._entries[1:]:                        # solo immagini importate
            bbox = e["item"].sceneBoundingRect().intersected(frame)
            if bbox.width() < 2 or bbox.height() < 2:
                continue
            grown = bbox.adjusted(-grow, -grow, grow, grow).intersected(frame)
            if self._mask_filled(e["mask"]):               # oggetto = maschera disegnata
                wm = QImage(W, H, QImage.Format_ARGB32); wm.fill(QColor(255, 255, 255))
                wm.setAlphaChannel(e["mask"].convertToFormat(QImage.Format_Alpha8))
                po.drawImage(0, 0, wm)
            else:
                sil = self._alpha_silhouette(e)            # foto scontornata/PNG: l'oggetto è la sagoma…
                if sil is not None:
                    po.drawImage(0, 0, sil)
                else:
                    # …foto OPACA (scontorno non disponibile): l'oggetto è tutto il suo rettangolo e lo
                    # sfondo va coperto — l'alone bianco serve solo QUI, ed è un ripiego: il modello lo
                    # legge come "spazio vuoto da riempire" e a volte ci mette dentro qualsiasi cosa.
                    po.fillRect(bbox, QColor(255, 255, 255))
                    pv.fillRect(grown, QColor(255, 255, 255))
            # La ZONA DI EDIT è il rettangolo cresciuto in OGNI caso: è lo spazio in cui il modello
            # ridisegna la mano che afferra e piega il braccio. Stringerla è ciò che rompe l'integrazione
            # (con l'anello sottile l'oggetto resta lì appoggiato, e nessuno lo prende in mano).
            pr.fillRect(grown, QColor(255, 255, 255))
            any_region = True
        if not any_region and self._mask_filled(self._entries[0]["mask"]):
            base_m = self._entries[0]["mask"].convertToFormat(QImage.Format_Alpha8)
            wm = QImage(W, H, QImage.Format_ARGB32); wm.fill(QColor(255, 255, 255))
            wm.setAlphaChannel(base_m)
            po.drawImage(0, 0, wm)                          # oggetto = l'area che hai mascherato
            bounds = self._mask_bounds(base_m)
            if not bounds.isNull():
                pr.fillRect(bounds.adjusted(-grow, -grow, grow, grow).intersected(frame),
                            QColor(255, 255, 255))
                any_region = True
        po.end(); pr.end(); pv.end()
        return obj, region, wreg, any_region

    def _integrate_context(self, obj: QImage, region: QImage, any_region: bool):
        """Maschera di CONTESTO finale (zona che il modello puo rigenerare) = contesto UNITO alla sagoma, e i pixel di dilatazione usati.
        Automatica (decisione dell'utente 03/11): si espande la maschera VERA della persona (la sagoma), non il riquadro che la contiene, di
        `integrate_context_px` PIXEL tutto intorno, con la sua forma (distance transform), senza tetto: il limite e il bordo dell'immagine. 0 px = il
        contesto coincide con la sagoma. Senza valore: DEFAULT_CTX_PX. `integrate_context_pct` (frazione della LARGHEZZA della sagoma) resta come alias
        per i casi vecchi. Manuale: la zona disegnata col pennello giallo."""
        if not any_region:
            return region, 0
        if self.integrate_ctx_mode == "manual" and self.integrate_ctx_enabled and self._ctx_manual is not None:
            return self._alpha_union_alpha(self._binarize(self._ctx_manual, 1), obj), 0
        base = obj
        _b = self._mask_bounds(obj) if np is not None else QRectF()
        if _b.isNull():
            base = region
            _b = self._mask_bounds(region)
        if self.integrate_context_px is not None:
            ctx_px = int(self.integrate_context_px)
        elif self.integrate_context_pct is not None:
            ctx_px = int(round(max(0.0, float(self.integrate_context_pct)) * _b.width()))
        else:
            ctx_px = DEFAULT_CTX_PX
        ctx_px = max(0, ctx_px)
        ctx = self._dilate_alpha(base, ctx_px) if ctx_px > 0 else base.copy()          # stessa forma della sagoma, allargata di ctx_px
        return self._alpha_union_alpha(ctx, obj), ctx_px

    @staticmethod
    def _alpha_union_alpha(a: QImage, b: QImage) -> QImage:
        """Unione di due maschere (alpha): il risultato e bianco dove una delle due ha alpha."""
        out = QImage(a.width(), a.height(), QImage.Format_ARGB32); out.fill(Qt.transparent)
        p = QPainter(out); p.drawImage(0, 0, a); p.drawImage(0, 0, b); p.end()
        return out

    def export_integrate(self, out_dir: Path) -> dict:
        """Input stile MagicQuill `foreground_edit` — la ricetta che INTEGRA DAVVERO (validata il 3/06).

        L'OGGETTO (maschera disegnata del layer, o in fallback il suo bbox) resta come seme; la ZONA DI
        EDIT attorno (bbox cresciuto) va a BIANCO; il resto = base. Il modello rigenera quella zona: ed è
        LARGA di proposito, perché lì dentro ci finiscono anche mano e braccio — così può PIEGARE IL
        BRACCIO e far CHIUDERE LE DITA attorno all'oggetto, invece di limitarsi a rifinirne i bordi.

        Due strade sbagliate già percorse (13/07, entrambe verificate a occhio sul risultato):
          · ring-mask sottile attorno alla sagoma → non c'è spazio per la mano: l'oggetto resta appoggiato;
          · sagoma protetta + niente alone → l'oggetto è intatto ma la scena non lo accoglie.
        La cucitura del riquadro NON si cura restringendo la maschera: si cura sfumandola (feather ampio
        passato al backend da generate_integrate).

        Se non c'è NESSUN layer ma hai mascherato un'area della foto del canvas, quella è la regione da
        integrare: stessa logica, l'oggetto è ciò che hai mascherato.
        """
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        W, H = self.aspect_w, self.aspect_h
        frame = QRectF(0, 0, W, H)
        self._auto_cutout_opaque_layers()
        comp = QImage(self._composite(out_dir)).convertToFormat(QImage.Format_ARGB32)
        obj, region, wreg, any_region = self._build_integrate_masks()
        region, ctx_px = self._integrate_context(obj, region, any_region)         # CONTESTO (auto o manuale) UNITO alla sagoma
        # alone bianco = zona da sbiancare (solo foto opache) MENO l'oggetto
        whiten = wreg.copy()
        pw = QPainter(whiten); pw.setCompositionMode(QPainter.CompositionMode_DestinationOut)
        pw.drawImage(0, 0, obj); pw.end()
        prepared = comp.copy()
        pp = QPainter(prepared); pp.drawImage(0, 0, whiten); pp.end()
        ipath = out_dir / "integrate_input.png"; prepared.save(str(ipath))
        msk = QImage(W, H, QImage.Format_ARGB32); msk.fill(Qt.black)
        pm = QPainter(msk); pm.drawImage(0, 0, region); pm.end()
        mpath = out_dir / "integrate_mask.png"; msk.save(str(mpath))
        # maschera di POSIZIONE (dove sta il soggetto: la sagoma) accanto a quella di CONTESTO (integrate_mask.png = quanto il modello puo rigenerare)
        posm = QImage(W, H, QImage.Format_ARGB32); posm.fill(Qt.black)
        ppm = QPainter(posm); ppm.drawImage(0, 0, obj); ppm.end()
        pospath = out_dir / "integrate_position_mask.png"; posm.save(str(pospath))
        base_path, layer_refs = "", []
        if any_region and len(self._entries) > 1:
            try:
                vis = [(e["item"], e["item"].isVisible()) for e in self._entries[1:]]
                for it, _v in vis:
                    it.setVisible(False)
                try:
                    bimg = QImage(self._composite(out_dir)).convertToFormat(QImage.Format_RGB32)      # la sola scena (nessuna foto incollata)
                    base_path = str(out_dir / "integrate_base.png"); bimg.save(base_path)
                finally:
                    for it, _v in vis:
                        it.setVisible(_v)
                for k, e in enumerate(self._entries[1:], 1):                                          # soggetti scontornati su fondo neutro, per i riferimenti
                    pm = e["item"].pixmap()
                    if pm.isNull():
                        continue
                    im = pm.toImage().convertToFormat(QImage.Format_ARGB32)
                    bg = QImage(im.size(), QImage.Format_RGB32); bg.fill(QColor(128, 128, 128))
                    pq = QPainter(bg); pq.drawImage(0, 0, im); pq.end()
                    lp = out_dir / f"integrate_layer_{k}.png"; bg.save(str(lp)); layer_refs.append(str(lp))
            except Exception:                                # noqa: BLE001
                base_path, layer_refs = "", []
        # has_region NON dipende da has_layers(): senza layer ma con una maschera sulla foto del canvas
        # c'è eccome qualcosa da integrare (era il blocco «aggiungi una o più immagini»).
        return {"image": str(ipath), "mask": str(mpath), "position_mask": str(pospath), "context_px": int(ctx_px), "base": base_path, "layer_refs": layer_refs,
                "layers": [e.get("path") for e in self._entries[1:] if e.get("path")], "has_region": bool(any_region)}

    def _drop_layers(self):
        for e in self._entries[1:]:           # rimuove i layer importati (tiene la base)
            self.scene.removeItem(e["item"])
        del self._entries[1:]
        self._active = 0
        self._ctx_reset()
        self._rebuild_layer_list()

    def load_result(self, path: str):
        """Sostituisce la foto base col risultato generato (riempie esattamente la cornice)."""
        pix = QPixmap(path)
        if pix.isNull():
            return
        self._image_path = path
        self.image_changed.emit(path)
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
        self.image_changed.emit(path)
        self._drop_layers()
        self.img_item.setPixmap(pix)
        self.img_item.setScale(1.0)          # il ritaglio è già a risoluzione di scena
        self.img_item.setPos(x, y)
        self.clear_mask(silent=True); self._undo.clear(); self._redo.clear()
        self._position_overlay(); self._fit()

    @property
    def image_path(self):
        return self._image_path
