"""Tab TRELLIS.2 — generazione di un modello 3D da una singola immagine.

Da una foto/immagine generata produce una mesh 3D con materiali PBR (GLB),
mostrata in un visore 3D interattivo integrato.

L'operazione viene inviata a ComfyUI tramite core/trellis_pipeline.py.
Il risultato (.glb) viene caricato nel visore e salvato in output/.
"""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QUrl
from ui.i18n_manager import tr
from ui.widgets.lazy_viewer import LazyViewer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton,
    QLineEdit, QCheckBox, QFileDialog, QScrollArea, QSplitter, QMessageBox,
)

# Visore 3D opzionale (QtWebEngine). Se assente, si degrada a soli pulsanti.
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEngineSettings
    _HAS_WEBENGINE = True
except Exception:  # pragma: no cover - dipende dall'installazione PySide6
    _HAS_WEBENGINE = False


def _no_webgl() -> bool:
    """True se il visore WebEngine va EVITATO (LTX_NO_WEBGL=1). Su alcune GPU/driver (es. RTX 5090/Blackwell)
    il processo GPU di Chromium fallisce (kFatalFailure) e **crasha l'intera app** durante la generazione 3D —
    mentre il subprocess che genera il GLB sta benissimo. In quel caso si usa il visore semplice (pulsanti
    Apri/Salva), niente WebEngine → l'app non crasha e il modello viene generato lo stesso."""
    return os.environ.get("LTX_NO_WEBGL") == "1"


class _Model3DViewer(QWidget):
    """Visore 3D GLB basato su <model-viewer> (Google) dentro QWebEngineView."""

    def __init__(self):
        super().__init__()
        self._glb_path: str = ""
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0)

        if _HAS_WEBENGINE and not _no_webgl():
            self.view = QWebEngineView()
            s = self.view.settings()
            # consenti al contenuto locale di raggiungere il file .glb e la CDN del viewer
            s.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
            s.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
            s.setAttribute(QWebEngineSettings.WebGLEnabled, True)
            v.addWidget(self.view, 1)
            self._show_placeholder()
        else:
            self.view = None
            # Distingui i due casi: WebEngine davvero assente vs disattivato a mano (LTX_NO_WEBGL=1) — così non
            # sembra un errore quando è una scelta/residuo di sessione.
            if _HAS_WEBENGINE and _no_webgl():
                msg = tr("trellis.viewer.webgl_off",
                         "Visore WebGL disattivato (LTX_NO_WEBGL=1). Riavvia SENZA quella variabile "
                         "d'ambiente per il visore 3D interattivo. Intanto usa i pulsanti qui sotto.")
            else:
                msg = tr("trellis.viewer.no_webengine")
            lbl = QLabel(msg)
            lbl.setWordWrap(True); lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color:#9aa1b0; padding:24px;")
            v.addWidget(lbl, 1)

        # barra pulsanti (sempre disponibile)
        bar = QHBoxLayout()
        self.btn_folder = QPushButton(tr("trellis.viewer.open_folder"))
        self.btn_folder.clicked.connect(self._open_folder)
        self.btn_system = QPushButton(tr("trellis.viewer.open_system"))
        self.btn_system.clicked.connect(self._open_system)
        self.btn_saveas = QPushButton(tr("trellis.viewer.save_as"))
        self.btn_saveas.clicked.connect(self._save_as)
        for b in (self.btn_folder, self.btn_system, self.btn_saveas):
            b.setEnabled(False); bar.addWidget(b)
        bar.addStretch()
        v.addLayout(bar)

    # ---------- API ----------
    def load_model(self, glb_path: str):
        """Carica un .glb nel visore."""
        self._glb_path = glb_path
        for b in (self.btn_folder, self.btn_system, self.btn_saveas):
            b.setEnabled(bool(glb_path))
        if not _HAS_WEBENGINE or self.view is None:
            return
        p = Path(glb_path)
        if not p.exists():
            self._show_placeholder(tr("trellis.viewer.not_found"))
            return
        # src relativo + baseUrl sulla cartella → evita problemi di file access
        src = p.name
        html = _VIEWER_HTML.replace("__SRC__", src)
        base = QUrl.fromLocalFile(str(p.parent) + os.sep)
        self.view.setHtml(html, base)

    def _show_placeholder(self, msg: str = ""):
        if not _HAS_WEBENGINE or self.view is None:
            return
        text = msg or tr("trellis.viewer.placeholder")
        self.view.setHtml(_PLACEHOLDER_HTML.replace("__MSG__", text))

    # ---------- pulsanti ----------
    def _open_folder(self):
        if self._glb_path:
            folder = str(Path(self._glb_path).parent)
            try:
                os.startfile(folder)  # Windows
            except Exception:
                pass

    def _open_system(self):
        if self._glb_path and Path(self._glb_path).exists():
            try:
                os.startfile(self._glb_path)
            except Exception:
                pass

    def _save_as(self):
        if not self._glb_path or not Path(self._glb_path).exists():
            return
        ext = Path(self._glb_path).suffix.lstrip(".") or "glb"
        dest, _ = QFileDialog.getSaveFileName(
            self, tr("trellis.viewer.save_as"),
            str(Path.home() / Path(self._glb_path).name),
            f"3D model (*.{ext})")
        if dest:
            try:
                import shutil
                shutil.copyfile(self._glb_path, dest)
            except Exception as e:
                QMessageBox.warning(self, tr("trellis.viewer.save_as"), str(e))


def make_viewer():
    """Restituisce il visore 3D interattivo (selezione/cancellazione/separazione)
    se disponibile, altrimenti il visore storico di sola visualizzazione.
    Paracadute: LTX_NO_WEBGL=1 forza il visore semplice se WebGL/Chromium crasha su una certa GPU/driver."""
    import os as _os
    if _os.environ.get("LTX_NO_WEBGL") == "1":
        return _Model3DViewer()
    try:
        from ui.widgets.model3d_viewer import InteractiveModel3DViewer
        return InteractiveModel3DViewer()
    except Exception:
        return _Model3DViewer()


class TrellisTab(QWidget):
    """Tab per generare un modello 3D (TRELLIS.2) da un'immagine."""

    request_trellis = Signal(dict)   # config → MainWindow

    def __init__(self):
        super().__init__()
        self._input_path: str = ""
        self._build()

    # ---------- build ----------
    def _build(self):
        root = QVBoxLayout(self)
        split = QSplitter(Qt.Horizontal)

        # ═══ Sinistra: impostazioni (scroll) ═══════════════════════════════
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        v = QVBoxLayout(container); v.setSpacing(8)

        # ── Immagine di input ──
        inp_grp = QGroupBox(tr("trellis.input.group"))
        il = QHBoxLayout(inp_grp)
        self.path_edit = QLineEdit(); self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText(tr("trellis.input.ph"))
        il.addWidget(self.path_edit, 1)
        browse_btn = QPushButton(tr("trellis.input.browse"))
        browse_btn.clicked.connect(self._browse_input)
        il.addWidget(browse_btn)
        canvas_btn = QPushButton(tr("trellis.input.canvas"))
        canvas_btn.setToolTip(tr("trellis.input.canvas.tt"))
        canvas_btn.clicked.connect(self._use_canvas_image)
        il.addWidget(canvas_btn)
        masked_btn = QPushButton(tr("trellis.input.canvas_masked"))
        masked_btn.setToolTip(tr("trellis.input.canvas_masked.tt"))
        masked_btn.clicked.connect(self._use_canvas_masked)
        il.addWidget(masked_btn)
        v.addWidget(inp_grp)

        # ── Viste aggiuntive (multi-immagine): più foto dello stesso oggetto da angoli diversi
        #    migliorano la geometria (i token delle viste vengono concatenati nel condizionatore). ──
        mv_grp = QGroupBox(tr("trellis.multi.group", "Viste aggiuntive (opzionale)"))
        mv = QVBoxLayout(mv_grp)
        mv_info = QLabel(tr("trellis.multi.info",
                            "Aggiungi altre foto dello STESSO oggetto da angoli diversi (fronte/retro/lato): "
                            "la geometria 3D migliora. Max 3 viste extra."))
        mv_info.setWordWrap(True); mv_info.setStyleSheet("color:#9aa1b0; font-size:11px;")
        mv.addWidget(mv_info)
        from PySide6.QtWidgets import QListWidget
        from PySide6.QtCore import QSize
        self.views_list = QListWidget()
        self.views_list.setViewMode(QListWidget.IconMode)
        self.views_list.setIconSize(QSize(72, 72))
        self.views_list.setFixedHeight(96)
        self.views_list.setToolTip(tr("trellis.multi.list.tt",
                                      "Anteprime delle viste extra: selezionane una e usa 'Rimuovi' per toglierla."))
        mv.addWidget(self.views_list)
        mvb = QHBoxLayout()
        self.mv_add_btn = QPushButton(tr("trellis.multi.add", "＋ Aggiungi vista"))
        self.mv_add_btn.setToolTip(tr("trellis.multi.add.tt", "Scegli un'altra foto dell'oggetto da un angolo diverso."))
        self.mv_add_btn.clicked.connect(self._add_view)
        self.mv_del_btn = QPushButton(tr("trellis.multi.remove", "− Rimuovi"))
        self.mv_del_btn.setToolTip(tr("trellis.multi.remove.tt", "Rimuovi la vista selezionata dalla lista."))
        self.mv_del_btn.clicked.connect(self._remove_view)
        mvb.addWidget(self.mv_add_btn); mvb.addWidget(self.mv_del_btn); mvb.addStretch()
        mv.addLayout(mvb)
        v.addWidget(mv_grp)

        # ── Qualità / modello ──
        q_grp = QGroupBox(tr("trellis.quality.group"))
        qf = QFormLayout(q_grp); qf.setLabelAlignment(Qt.AlignRight)
        self.resolution = QComboBox()
        self.resolution.addItems(["512", "1024", "1024_cascade", "1536_cascade"])
        self.resolution.setCurrentText("1024_cascade")
        self.resolution.setToolTip(tr("trellis.quality.res.tt"))
        qf.addRow(tr("trellis.quality.res"), self.resolution)
        self.file_format = QComboBox()
        self.file_format.addItems(["glb", "obj", "ply", "stl"])
        qf.addRow(tr("trellis.quality.format"), self.file_format)
        self.remove_bg = QCheckBox(tr("trellis.quality.rmbg"))
        self.remove_bg.setChecked(True)
        self.remove_bg.setToolTip(tr("trellis.quality.rmbg.tt"))
        qf.addRow(tr("trellis.quality.rmbg_lbl"), self.remove_bg)
        v.addWidget(q_grp)

        # ── Geometria ──
        g_grp = QGroupBox(tr("trellis.geo.group"))
        gf = QFormLayout(g_grp); gf.setLabelAlignment(Qt.AlignRight)
        self.detail = QSpinBox()
        self.detail.setRange(16384, 262144); self.detail.setSingleStep(4096)
        self.detail.setValue(49152)
        self.detail.setToolTip(tr("trellis.geo.detail.tt"))
        gf.addRow(tr("trellis.geo.detail"), self.detail)
        self.guidance = QDoubleSpinBox()
        self.guidance.setRange(0.0, 20.0); self.guidance.setSingleStep(0.5)
        self.guidance.setValue(6.5)
        self.guidance.setToolTip(tr("trellis.geo.guidance.tt"))
        gf.addRow(tr("trellis.geo.guidance"), self.guidance)
        self.steps = QSpinBox()
        self.steps.setRange(1, 50); self.steps.setValue(12)
        gf.addRow(tr("trellis.geo.steps"), self.steps)
        self.faces = QSpinBox()
        self.faces.setRange(1000, 5000000); self.faces.setSingleStep(50000)
        self.faces.setValue(500000)
        self.faces.setToolTip(tr("trellis.geo.faces.tt"))
        gf.addRow(tr("trellis.geo.faces"), self.faces)
        v.addWidget(g_grp)

        # ── Texture PBR ──
        t_grp = QGroupBox(tr("trellis.tex.group"))
        tf = QFormLayout(t_grp); tf.setLabelAlignment(Qt.AlignRight)
        self.texture = QCheckBox(tr("trellis.tex.enable"))
        self.texture.setChecked(True)
        self.texture.setToolTip(tr("trellis.tex.enable.tt"))
        self.texture.toggled.connect(self._on_texture_toggled)
        tf.addRow(tr("trellis.tex.enable_lbl"), self.texture)
        self.tex_size = QComboBox()
        self.tex_size.addItems(["1024", "2048", "4096"])
        self.tex_size.setCurrentText("2048")
        tf.addRow(tr("trellis.tex.size"), self.tex_size)
        self.tex_guidance = QDoubleSpinBox()
        self.tex_guidance.setRange(0.0, 20.0); self.tex_guidance.setSingleStep(0.5)
        self.tex_guidance.setValue(3.0)
        tf.addRow(tr("trellis.tex.guidance"), self.tex_guidance)
        self.tex_steps = QSpinBox()
        self.tex_steps.setRange(1, 50); self.tex_steps.setValue(12)
        tf.addRow(tr("trellis.tex.steps"), self.tex_steps)
        v.addWidget(t_grp)

        # ── Seed ──
        s_grp = QGroupBox(tr("trellis.seed.group"))
        sf = QHBoxLayout(s_grp)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1); self.seed.setValue(0)
        sf.addWidget(self.seed, 1)
        self.seed_random = QCheckBox(tr("trellis.seed.random"))
        self.seed_random.setChecked(True)
        self.seed_random.toggled.connect(lambda c: self.seed.setEnabled(not c))
        self.seed.setEnabled(False)
        sf.addWidget(self.seed_random)
        v.addWidget(s_grp)

        # ── Avanzate ──
        a_grp = QGroupBox(tr("trellis.adv.group"))
        a_grp.setCheckable(True); a_grp.setChecked(False)
        af = QFormLayout(a_grp); af.setLabelAlignment(Qt.AlignRight)
        self.precision = QComboBox()
        self.precision.addItems(["auto", "bf16", "fp16", "fp32"])
        # default bf16+sdpa+gray = parametri della generazione di riferimento approvata
        # (con "auto" precisione/attention aggressive → meno dettagli, cuciture texture)
        self.precision.setCurrentText("bf16")
        af.addRow(tr("trellis.adv.precision"), self.precision)
        self.attn = QComboBox()
        self.attn.addItems(["auto", "flash_attn", "xformers", "sdpa", "sageattn"])
        self.attn.setCurrentText("sdpa")
        af.addRow(tr("trellis.adv.attn"), self.attn)
        self.bg_color = QComboBox()
        self.bg_color.addItems(["black", "gray", "white"])
        self.bg_color.setCurrentText("gray")
        af.addRow(tr("trellis.adv.bg"), self.bg_color)
        v.addWidget(a_grp)

        # nota
        note = QLabel(tr("trellis.note"))
        note.setWordWrap(True); note.setStyleSheet("color:#74c0fc; padding:6px;")
        v.addWidget(note)
        v.addStretch()
        scroll.setWidget(container)

        # ═══ Destra: visore 3D (interattivo se disponibile) ════════════════
        self.viewer = LazyViewer(make_viewer)         # il visore Chromium nasce alla prima apertura del tab

        split.addWidget(scroll)
        split.addWidget(self.viewer)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([420, 600])
        split.setChildrenCollapsible(True)                  # #96.2: separatore libero, senza magnet né limiti
        split.setHandleWidth(8)
        scroll.setMinimumWidth(0); self.viewer.setMinimumWidth(0)
        root.addWidget(split, 1)

        # pulsante genera
        self.gen_btn = QPushButton(tr("trellis.run.btn"))
        self.gen_btn.setObjectName("generate")
        self.gen_btn.clicked.connect(self._on_generate)
        root.addWidget(self.gen_btn)

    # ---------- slot ----------
    def _on_texture_toggled(self, on: bool):
        for w in (self.tex_size, self.tex_guidance, self.tex_steps):
            w.setEnabled(on)

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("trellis.input.browse"), str(Path.home()),
            tr("trellis.input.filter"))
        if path:
            self._input_path = path
            self.path_edit.setText(path)

    def _use_canvas_image(self):
        # main_window risolve "<from_canvas>" con frame1_tab.image_path
        self._input_path = "<from_canvas>"
        self.path_edit.setText(tr("trellis.input.canvas.selected"))

    def _use_canvas_masked(self):
        # main_window risolve "<from_canvas_masked>" esportando il soggetto
        # isolato dalla maschera del canvas (sfondo trasparente).
        self._input_path = "<from_canvas_masked>"
        self.path_edit.setText(tr("trellis.input.canvas_masked.selected"))

    # ---------- viste aggiuntive (multi-immagine) ----------
    def _add_view(self):
        if self.views_list.count() >= 3:
            QMessageBox.information(self, tr("trellis.multi.group", "Viste aggiuntive (opzionale)"),
                                    tr("trellis.multi.max_msg", "Massimo 3 viste extra (4 immagini totali)."))
            return
        path, _ = QFileDialog.getOpenFileName(
            self, tr("trellis.multi.add", "＋ Aggiungi vista"), str(Path.home()),
            tr("trellis.input.filter"))
        if not path:
            return
        from PySide6.QtWidgets import QListWidgetItem
        from PySide6.QtGui import QIcon, QPixmap
        pix = QPixmap(path)
        item = QListWidgetItem(Path(path).name)
        if not pix.isNull():
            item.setIcon(QIcon(pix.scaled(72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        item.setToolTip(path)
        item.setData(Qt.UserRole, path)
        self.views_list.addItem(item)

    def _remove_view(self):
        for it in self.views_list.selectedItems():
            self.views_list.takeItem(self.views_list.row(it))

    def extra_views(self) -> list[str]:
        """Path delle viste aggiuntive (esistenti su disco), in ordine di lista."""
        out = []
        for i in range(self.views_list.count()):
            p = self.views_list.item(i).data(Qt.UserRole)
            if p and os.path.exists(str(p)):
                out.append(str(p))
        return out

    def _on_generate(self):
        # FASE F weights-unification: primo-uso CON CONSENSO — se i pesi TRELLIS mancano, il
        # dialog li offre QUI (prima: auto-download silenzioso da 8GB, ora il runner li rifiuta).
        try:
            from core import ensure_weights as ew
            if ew.missing(["trellis2-4b"]) and not ew.offer_download_gui(
                    self, ["trellis2-4b"], tr("tab.trellis", "Modello 3D")):
                return
        except Exception:                                # noqa: BLE001 — il preflight del service protegge comunque
            pass
        self.request_trellis.emit(self.get_config())

    # ---------- API ----------
    def get_config(self) -> dict:
        return {
            "image_path": self._input_path,
            # multi-vista: input primario + viste extra (il backend concatena i token delle viste)
            "image_paths": ([self._input_path] if self._input_path else []) + self.extra_views(),
            "resolution": self.resolution.currentText(),
            "precision": self.precision.currentText(),
            "attn_backend": self.attn.currentText(),
            "remove_background": self.remove_bg.isChecked(),
            "background_color": self.bg_color.currentText(),
            "max_tokens": self.detail.value(),
            "guidance": self.guidance.value(),
            "steps": self.steps.value(),
            "target_face_count": self.faces.value(),
            "texture": self.texture.isChecked(),
            "texture_size": int(self.tex_size.currentText()),
            "tex_guidance": self.tex_guidance.value(),
            "tex_steps": self.tex_steps.value(),
            "seed": self.seed.value(),
            "seed_random": self.seed_random.isChecked(),
            "file_format": self.file_format.currentText(),
        }

    def set_input_path(self, path: str):
        self._input_path = path
        self.path_edit.setText(path)

    def show_result(self, glb_path: str):
        self.viewer.load_model(glb_path)


# ── HTML del visore (model-viewer da CDN, fallback a messaggio) ───────────────
_VIEWER_HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>html,body{margin:0;height:100%;background:#1e1f26;overflow:hidden}
model-viewer{width:100%;height:100vh;--poster-color:#1e1f26}
#err{color:#9aa1b0;font-family:sans-serif;text-align:center;padding:40px}</style>
<script type="module" src="https://unpkg.com/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>
</head><body>
<model-viewer src="__SRC__" camera-controls auto-rotate shadow-intensity="1"
  exposure="1.0" environment-image="neutral" ar
  style="background:#1e1f26"></model-viewer>
</body></html>"""

_PLACEHOLDER_HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>html,body{margin:0;height:100%;background:#1e1f26;display:flex;
align-items:center;justify-content:center}
#m{color:#9aa1b0;font-family:sans-serif;text-align:center;padding:40px;max-width:80%}</style>
</head><body><div id="m">__MSG__</div></body></html>"""
