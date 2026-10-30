"""Pannello RIQUADRI di FLUX.3 Image Klein: disegna riquadri con id e descrizione su una tela e li consegna nel formato di
`core/flux3_caption` (coordinate 0–1, [y_min, x_min, y_max, x_max]).

Due usi, scelti da `set_context`:
  · LAYOUT (nessun riferimento, «Crea»): ogni riquadro = una zona dell'immagine con la sua descrizione → {"id","desc","bbox"};
  · EDIT (con riferimenti): ogni riquadro ha una modalità — NUOVO (solo destinazione) · SPOSTA (origine + destinazione) ·
    ANCORA (resta dov'è: origine = destinazione) · RIMUOVI (solo origine) — e il riferimento di origine (@Foto_N):
    {"id","desc","mode","ref","src_bbox","tgt_bbox"}. I riquadri fissano posizione e scala, NON sono una maschera rigida.

La tela mostra il riferimento di origine (o un riquadro vuoto del rapporto richiesto per il layout). Disegno con mouse o dito:
trascina in un punto libero per tracciare, dentro il riquadro per spostarlo, dall'angolo per ridimensionarlo (maniglie da 44 px).
Modulo Qt; la logica di formato sta in `core.flux3_caption` (testabile senza GUI).
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QListWidget,
                               QPlainTextEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget, QLineEdit)

from core import flux3_caption as C
from core import flux3_spec as S
from ui.i18n_manager import tr

_PALETTE = ("#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080", "#9a6324", "#800000", "#808000", "#000075")
HANDLE = 44            # px: bersaglio tattile minimo (Apple HIG 44 pt / Material 48 dp ≈) per le maniglie d'angolo
MIN_SPAN = 0.02        # un riquadro più piccolo del 2 % del lato non è un riquadro


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


class BoxCanvas(QWidget):
    """Tela di disegno: sfondo (pixmap o riquadro vuoto), riquadri di destinazione (pieni) e di origine (tratteggiati)."""
    changed = Signal()
    selected = Signal(int)                       # indice del riquadro selezionato (-1 = nessuno)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(False)
        self._pix: QPixmap | None = None
        self._ratio = 1.0                        # larghezza / altezza quando non c'è lo sfondo
        self.boxes: list[dict] = []              # {"id","src","tgt"} con src/tgt = [y1,x1,y2,x2] 0–1 | None
        self.sel = -1
        self.role = "tgt"                        # quale rettangolo del riquadro selezionato si disegna: tgt | src
        self._drag = None                        # (azione, rettangolo di partenza, punto di partenza normalizzato)

    # -- contesto --------------------------------------------------------------------------------------
    def set_background(self, pixmap: QPixmap | None, ratio: float | None = None):
        self._pix = pixmap if (pixmap is not None and not pixmap.isNull()) else None
        if ratio:
            self._ratio = float(ratio)
        self.update()

    def _frame(self) -> QRectF:
        """Rettangolo (in px del widget) dove vive l'immagine: massimo possibile col rapporto giusto, centrato."""
        w, h = self.width() - 8, self.height() - 8
        ratio = (self._pix.width() / self._pix.height()) if self._pix else self._ratio
        if w / h > ratio:
            fw, fh = h * ratio, h
        else:
            fw, fh = w, w / ratio
        return QRectF((self.width() - fw) / 2, (self.height() - fh) / 2, fw, fh)

    def _to_norm(self, p: QPointF) -> tuple[float, float]:
        f = self._frame()
        return _clamp01((p.y() - f.top()) / f.height()), _clamp01((p.x() - f.left()) / f.width())

    def _rect_px(self, box) -> QRectF | None:
        if not box:
            return None
        f = self._frame()
        y1, x1, y2, x2 = box
        return QRectF(f.left() + x1 * f.width(), f.top() + y1 * f.height(), (x2 - x1) * f.width(), (y2 - y1) * f.height())

    def _color(self, i: int) -> QColor:
        return QColor(_PALETTE[i % len(_PALETTE)])

    # -- disegno ---------------------------------------------------------------------------------------
    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#1b1b1f"))
        f = self._frame()
        if self._pix:
            p.drawPixmap(f.toRect(), self._pix)
        else:
            p.fillRect(f, QColor("#2b2b33"))
            p.setPen(QPen(QColor("#555"), 1, Qt.DashLine))
            p.drawRect(f)
        p.setPen(QPen(QColor("#888"), 1))
        p.drawRect(f)
        font = QFont(self.font()); font.setBold(True); p.setFont(font)
        for i, b in enumerate(self.boxes):
            col = self._color(i)
            for role in ("src", "tgt"):
                r = self._rect_px(b.get(role))
                if r is None:
                    continue
                pen = QPen(col, 3 if i == self.sel else 2, Qt.DashLine if role == "src" else Qt.SolidLine)
                p.setPen(pen)
                fill = QColor(col); fill.setAlpha(60 if (i == self.sel and role == self.role) else 28)
                p.setBrush(fill)
                p.drawRect(r)
                tag = ("⟲ " if role == "src" else "") + str(b.get("id") or f"obj_{i + 1}")
                p.setPen(QPen(col.lighter(130)))
                p.setBrush(Qt.NoBrush)
                p.drawText(r.adjusted(4, 2, -2, -2), Qt.AlignTop | Qt.AlignLeft, tag)
                if i == self.sel and role == self.role:      # maniglie d'angolo (visuali piccole, bersaglio da 44 px)
                    p.setPen(QPen(Qt.white, 1)); p.setBrush(col)
                    for c in self._corners(r):
                        p.drawEllipse(c, 6, 6)
        p.end()

    @staticmethod
    def _corners(r: QRectF):
        return [r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()]

    # -- interazione -----------------------------------------------------------------------------------
    def _hit(self, pos: QPointF):
        """(indice, ruolo, azione) di ciò che sta sotto il puntatore, preferendo il riquadro selezionato."""
        order = list(range(len(self.boxes)))
        if 0 <= self.sel < len(order):
            order.remove(self.sel); order.insert(0, self.sel)
        for i in order:
            for role in ((self.role, "src" if self.role == "tgt" else "tgt") if i == self.sel else ("tgt", "src")):
                r = self._rect_px(self.boxes[i].get(role))
                if r is None:
                    continue
                if i == self.sel and role == self.role:
                    for k, c in enumerate(self._corners(r)):
                        if abs(c.x() - pos.x()) <= HANDLE / 2 and abs(c.y() - pos.y()) <= HANDLE / 2:
                            return i, role, ("resize", k)
                if r.contains(pos):
                    return i, role, ("move", None)
        return -1, "", ("new", None)

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        pos = e.position()
        i, role, (act, k) = self._hit(pos)
        if i >= 0 and i != self.sel:
            self.select(i)
        if i >= 0 and role != self.role and role:
            self.role = role
            self.selected.emit(self.sel)
        if act == "new" or i < 0:
            if self.sel < 0:
                return                                   # serve prima un riquadro («Aggiungi»)
            y, x = self._to_norm(pos)
            self._drag = ("new", [y, x, y, x], (y, x))
            self.boxes[self.sel][self.role] = [y, x, y + 0.0001, x + 0.0001]
        elif act == "move":
            self._drag = ("move", list(self.boxes[i][self.role]), self._to_norm(pos))
        else:                                            # resize dall'angolo k: l'angolo opposto resta fermo
            y1, x1, y2, x2 = self.boxes[i][self.role]
            ay, ax = (y2 if k in (0, 1) else y1), (x2 if k in (0, 2) else x1)
            self._drag = ("resize", [ay, ax], (ay, ax))
        self.update()

    def mouseMoveEvent(self, e):
        if not self._drag or self.sel < 0:
            return
        act, start, p0 = self._drag
        y, x = self._to_norm(e.position())
        b = self.boxes[self.sel]
        if act in ("new", "resize"):
            ay, ax = (p0 if act == "new" else start)
            b[self.role] = [min(ay, y), min(ax, x), max(ay, y), max(ax, x)]
        else:
            dy, dx = y - p0[0], x - p0[1]
            y1, x1, y2, x2 = start
            dy = max(-y1, min(1 - y2, dy)); dx = max(-x1, min(1 - x2, dx))
            b[self.role] = [y1 + dy, x1 + dx, y2 + dy, x2 + dx]
        self.update()

    def mouseReleaseEvent(self, _e):
        if self._drag and self.sel >= 0:
            b = self.boxes[self.sel]
            r = b.get(self.role)
            if r and (r[2] - r[0] < MIN_SPAN or r[3] - r[1] < MIN_SPAN):
                b[self.role] = None                      # un click senza trascinare non lascia un riquadro minuscolo
            self.changed.emit()
        self._drag = None
        self.update()

    def select(self, i: int):
        self.sel = i
        self.selected.emit(i)
        self.update()


class Flux3BoxesPanel(QWidget):
    """Elenco riquadri + campi (id, descrizione, modalità, riferimento) + tela di disegno."""
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._refs: list[tuple[int, str, str]] = []      # (posizione 1-based, nome @Foto_N, path)
        self._edit = False
        self._busy = False
        self.canvas = BoxCanvas()
        self.canvas.changed.connect(self._canvas_changed)
        self.canvas.selected.connect(self._on_canvas_selected)
        lay = QHBoxLayout(self)

        left = QVBoxLayout()
        left.addWidget(self.canvas, 1)
        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#aaa; font-size:11px;")
        left.addWidget(self.hint)
        lay.addLayout(left, 3)

        right = QVBoxLayout()
        self.listw = QListWidget()
        self.listw.setMinimumWidth(180)
        self.listw.currentRowChanged.connect(self._on_row)
        right.addWidget(self.listw, 1)
        btns = QHBoxLayout()
        self.btn_add = QPushButton(tr("f3.box.add", "＋ Aggiungi"))
        self.btn_del = QPushButton(tr("f3.box.del", "Elimina"))
        self.btn_clear = QPushButton(tr("f3.box.clear", "Svuota"))
        for b in (self.btn_add, self.btn_del, self.btn_clear):
            b.setMinimumHeight(44)
            btns.addWidget(b)
        self.btn_add.clicked.connect(self.add_box)
        self.btn_del.clicked.connect(self.remove_selected)
        self.btn_clear.clicked.connect(self.clear)
        right.addLayout(btns)
        form = QFormLayout()
        self.ed_id = QLineEdit(); self.ed_id.setMinimumHeight(44)
        self.ed_id.setToolTip(tr("f3.box.id.tt", "Nome del riquadro (lettera iniziale, poi lettere/cifre/_). Il testo lo cita come <nome>."))
        self.ed_id.textEdited.connect(self._fields_changed)
        form.addRow(tr("f3.box.id", "Nome"), self.ed_id)
        self.ed_desc = QPlainTextEdit(); self.ed_desc.setMaximumHeight(84); self.ed_desc.setMinimumHeight(64)
        self.ed_desc.setPlaceholderText(tr("f3.box.desc.ph", "Cosa deve esserci in questa zona"))
        self.ed_desc.textChanged.connect(self._fields_changed)
        form.addRow(tr("f3.box.desc", "Descrizione"), self.ed_desc)
        self.cb_mode = QComboBox(); self.cb_mode.setMinimumHeight(44)
        for key, fb, data in (("f3.box.mode.new", "Nuovo (aggiungi qui)", "new"), ("f3.box.mode.move", "Sposta (da → a)", "move"),
                              ("f3.box.mode.anchor", "Ancora (resta dov'è)", "anchor"),
                              ("f3.box.mode.remove", "Rimuovi", "remove")):
            self.cb_mode.addItem(tr(key, fb), data)
        self.cb_mode.currentIndexChanged.connect(self._fields_changed)
        self.lb_mode = QLabel(tr("f3.box.mode", "Azione"))
        form.addRow(self.lb_mode, self.cb_mode)
        self.cb_ref = QComboBox(); self.cb_ref.setMinimumHeight(44)
        self.cb_ref.currentIndexChanged.connect(self._ref_changed)
        self.lb_ref = QLabel(tr("f3.box.ref", "Riferimento di origine"))
        form.addRow(self.lb_ref, self.cb_ref)
        self.cb_role = QComboBox(); self.cb_role.setMinimumHeight(44)
        self.cb_role.addItem(tr("f3.box.role.tgt", "Disegno la destinazione"), "tgt")
        self.cb_role.addItem(tr("f3.box.role.src", "Disegno l'origine"), "src")
        self.cb_role.currentIndexChanged.connect(self._role_changed)
        self.lb_role = QLabel(tr("f3.box.role", "Sulla tela"))
        form.addRow(self.lb_role, self.cb_role)
        right.addLayout(form)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        right.addWidget(self.status)
        lay.addLayout(right, 2)
        self.set_context([], 1.0)

    # -- contesto --------------------------------------------------------------------------------------
    def set_context(self, refs, ratio: float = 1.0):
        """refs = [(nome @Foto_N, path)] nell'ordine in cui vanno al modello (vuoto = LAYOUT). `ratio` = larghezza/altezza
        dell'immagine da creare (solo layout)."""
        self._refs = [(i + 1, n, p) for i, (n, p) in enumerate(refs or [])]
        self._edit = bool(self._refs)
        self.cb_ref.blockSignals(True)
        self.cb_ref.clear()
        for k, n, _p in self._refs:
            self.cb_ref.addItem(f"@{n}", k)
        self.cb_ref.blockSignals(False)
        for w in (self.lb_mode, self.cb_mode, self.lb_ref, self.cb_ref, self.lb_role, self.cb_role):
            w.setVisible(self._edit)
        self.hint.setText(tr("f3.box.hint.edit",
                             "Modifica: scegli azione e riferimento, poi traccia sulla tela l'origine (tratteggio) e/o la "
                             "destinazione (pieno). I riquadri fissano posizione e scala: non sono una maschera rigida.")
                          if self._edit else
                          tr("f3.box.hint.layout",
                             "Layout: traccia una zona per ogni elemento e descrivi cosa deve contenere; il testo del prompt "
                             "può citare ogni zona come <nome>. Posizione e scala sono vincoli, non una maschera rigida."))
        self._ratio = float(ratio) if ratio else 1.0
        self._refresh_background()
        self._sync_fields()
        self._validate()

    def _refresh_background(self):
        pm = None
        if self._edit:
            k = self.cb_ref.currentData() or 1
            path = next((p for kk, _n, p in self._refs if kk == k), "")
            pm = QPixmap(path) if path else None
        self.canvas.set_background(pm, self._ratio if hasattr(self, "_ratio") else 1.0)

    # -- dati ------------------------------------------------------------------------------------------
    def _next_id(self) -> str:
        used = {str(b.get("id")) for b in self.canvas.boxes}
        n = 1
        while f"obj_{n}" in used:
            n += 1
        return f"obj_{n}"

    def add_box(self):
        b = {"id": self._next_id(), "desc": "", "mode": "new" if self._edit else "layout",
             "ref": (self.cb_ref.currentData() if self._edit else None), "src": None, "tgt": None}
        self.canvas.boxes.append(b)
        self._rebuild_list(select=len(self.canvas.boxes) - 1)
        self.canvas.role = "tgt"
        self.changed.emit()

    def remove_selected(self):
        i = self.canvas.sel
        if 0 <= i < len(self.canvas.boxes):
            del self.canvas.boxes[i]
            self._rebuild_list(select=min(i, len(self.canvas.boxes) - 1))
            self.changed.emit()

    def clear(self):
        self.canvas.boxes.clear()
        self._rebuild_list(select=-1)
        self.changed.emit()

    def _rebuild_list(self, select: int = -1):
        self._busy = True
        self.listw.clear()
        for i, b in enumerate(self.canvas.boxes):
            self.listw.addItem(self._label(i, b))
        self._busy = False
        self.canvas.select(select)
        self.listw.setCurrentRow(select)
        self._sync_fields()
        self._validate()

    def _label(self, i: int, b: dict) -> str:
        done = (b.get("tgt") or b.get("src")) is not None
        return f"{'●' if done else '○'} {b.get('id')}" + (f"  [{b.get('mode')}]" if self._edit else "")

    def _on_row(self, row: int):
        if self._busy:
            return
        self.canvas.select(row)
        self._sync_fields()

    def _on_canvas_selected(self, i: int):
        if self._busy:
            return
        self._busy = True
        self.listw.setCurrentRow(i)
        self._busy = False
        self._sync_fields()

    def _canvas_changed(self):
        i = self.canvas.sel
        if 0 <= i < len(self.canvas.boxes):
            b = self.canvas.boxes[i]
            if self._edit and b.get("mode") == "anchor" and b.get("src") and self.canvas.role == "src":
                b["tgt"] = list(b["src"])                  # ancora: origine = destinazione
            if self._edit and b.get("mode") == "anchor" and b.get("tgt") and self.canvas.role == "tgt":
                b["src"] = list(b["tgt"])
            item = self.listw.item(i)
            if item:
                item.setText(self._label(i, b))
        self._validate()
        self.changed.emit()

    def _sync_fields(self):
        i = self.canvas.sel
        ok = 0 <= i < len(self.canvas.boxes)
        for w in (self.ed_id, self.ed_desc, self.cb_mode, self.cb_ref, self.cb_role, self.btn_del):
            w.setEnabled(ok)
        self._busy = True
        if ok:
            b = self.canvas.boxes[i]
            self.ed_id.setText(str(b.get("id") or ""))
            if self.ed_desc.toPlainText() != str(b.get("desc") or ""):
                self.ed_desc.setPlainText(str(b.get("desc") or ""))
            if self._edit:
                self.cb_mode.setCurrentIndex(max(0, self.cb_mode.findData(b.get("mode") or "new")))
                self.cb_ref.setCurrentIndex(max(0, self.cb_ref.findData(b.get("ref") or 1)))
                self.cb_role.setCurrentIndex(max(0, self.cb_role.findData(self.canvas.role)))
                self._mode_rules(b)
        else:
            self.ed_id.clear(); self.ed_desc.clear()
        self._busy = False

    def _mode_rules(self, b: dict):
        """Quale rettangolo ha senso disegnare per la modalità: nuovo → solo destinazione; rimuovi → solo origine."""
        mode = b.get("mode")
        roles = {"new": ("tgt",), "remove": ("src",)}.get(mode, ("src", "tgt"))
        self.cb_role.setEnabled(len(roles) > 1)
        if self.canvas.role not in roles:
            self.canvas.role = roles[0]
            self.cb_role.setCurrentIndex(max(0, self.cb_role.findData(self.canvas.role)))
        self.cb_ref.setEnabled(mode != "new")
        self.lb_ref.setEnabled(mode != "new")

    def _fields_changed(self, *_):
        if self._busy:
            return
        i = self.canvas.sel
        if not (0 <= i < len(self.canvas.boxes)):
            return
        b = self.canvas.boxes[i]
        b["id"] = self.ed_id.text().strip()
        b["desc"] = self.ed_desc.toPlainText().strip()
        if self._edit:
            new_mode = self.cb_mode.currentData() or "new"
            if new_mode != b.get("mode"):
                b["mode"] = new_mode
                if new_mode == "new":
                    b["src"] = None
                elif new_mode == "remove":
                    b["tgt"] = None
                elif new_mode == "anchor":
                    r = b.get("tgt") or b.get("src")
                    b["src"] = list(r) if r else None
                    b["tgt"] = list(r) if r else None
                self._mode_rules(b)
        item = self.listw.item(i)
        if item:
            item.setText(self._label(i, b))
        self.canvas.update()
        self._validate()
        self.changed.emit()

    def _ref_changed(self, *_):
        if self._busy:
            return
        i = self.canvas.sel
        if 0 <= i < len(self.canvas.boxes) and self._edit:
            self.canvas.boxes[i]["ref"] = self.cb_ref.currentData()
        self._refresh_background()
        self.changed.emit()

    def _role_changed(self, *_):
        if self._busy:
            return
        self.canvas.role = self.cb_role.currentData() or "tgt"
        self.canvas.update()

    # -- consegna --------------------------------------------------------------------------------------
    def get_boxes(self) -> list[dict]:
        """Riquadri nel formato di core.flux3_caption (coordinate 0–1). Quelli senza zona disegnata si omettono qui e li
        segnala `validate()` quando la zona manca."""
        out = []
        for b in self.canvas.boxes:
            if self._edit:
                out.append({"id": b.get("id"), "desc": b.get("desc", ""), "mode": b.get("mode") or "new",
                            "ref": b.get("ref"), "src_bbox": b.get("src"), "tgt_bbox": b.get("tgt")})
            else:
                out.append({"id": b.get("id"), "desc": b.get("desc", ""), "bbox": b.get("tgt")})
        return out

    def set_boxes(self, boxes: list[dict]):
        """Ricarica riquadri già nel formato di get_boxes (es. stato salvato)."""
        self.canvas.boxes = []
        for b in boxes or []:
            self.canvas.boxes.append({"id": b.get("id"), "desc": b.get("desc", ""),
                                      "mode": b.get("mode") or ("new" if self._edit else "layout"), "ref": b.get("ref"),
                                      "src": b.get("src_bbox"), "tgt": b.get("tgt_bbox", b.get("bbox"))})
        self._rebuild_list(select=0 if self.canvas.boxes else -1)

    def errors(self) -> list[str]:
        return C.validate(self.get_boxes(), len(self._refs))

    def _validate(self):
        errs = self.errors() if self.canvas.boxes else []
        self.status.setText("\n".join(f"⚠ {e}" for e in errs[:4]) if errs else
                            (tr("f3.box.ok", "{n} riquadri pronti").format(n=len(self.canvas.boxes)) if self.canvas.boxes else ""))
        self.status.setStyleSheet("color:#e8b339;" if errs else "color:#7bd88f;")


class Flux3BoxesDialog(QDialog):
    """Finestra del pannello (OK = tieni i riquadri, Annulla = scarta le modifiche)."""

    def __init__(self, parent, refs, ratio: float, boxes: list[dict] | None = None):
        super().__init__(parent)
        self.setWindowTitle(tr("f3.box.title", "Riquadri FLUX.3"))
        self.resize(940, 640)
        lay = QVBoxLayout(self)
        self.panel = Flux3BoxesPanel()
        self.panel.set_context(refs, ratio)
        if boxes:
            self.panel.set_boxes(boxes)
        lay.addWidget(self.panel, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setMinimumHeight(44)
        bb.button(QDialogButtonBox.Cancel).setMinimumHeight(44)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _ok(self):
        if self.panel.errors():
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, self.windowTitle(), "\n".join(self.panel.errors()))
            return
        self.accept()

    def boxes(self) -> list[dict]:
        return self.panel.get_boxes()
