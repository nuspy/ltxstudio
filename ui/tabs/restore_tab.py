"""Tab Restauro — ripristino qualità e colorazione di fotografie.

Operazioni:
- Restauro qualità: ESRGAN super-resolution (aumento definizione)
- Colorazione B&W: converte foto in bianco e nero a colori (FLUX.2)
- Restauro + Colorazione: pipeline combinata

L'operazione viene inviata a ComfyUI tramite restore_pipeline.py.
Il risultato appare come immagine nel canvas del tab Prompt/Frame1.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from ui.i18n_manager import tr
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QComboBox, QSlider, QDoubleSpinBox, QSpinBox,
    QPushButton, QLineEdit, QCheckBox, QRadioButton, QButtonGroup,
    QFileDialog, QScrollArea,
)


class RestoreTab(QWidget):
    """Tab per restauro fotografico e colorazione."""

    request_restore = Signal(dict)       # emesso con la config per MainWindow

    def __init__(self):
        super().__init__()
        self._upscale_models: list[str] = []
        self._flux2_models: list[str] = []
        self._input_path: str = ""
        self._build()

    # ---------- build ----------
    def _build(self):
        root = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        v = QVBoxLayout(container)
        v.setSpacing(8)

        # ── Immagine di input ──────────────────────────────────────────────
        inp_grp = QGroupBox(tr("res.input.group"))
        il = QHBoxLayout(inp_grp)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(tr("res.input.ph"))
        self.path_edit.setReadOnly(True)
        il.addWidget(self.path_edit, 1)
        browse_btn = QPushButton(tr("res.input.browse.btn"))
        browse_btn.clicked.connect(self._browse_input)
        il.addWidget(browse_btn)
        use_canvas_btn = QPushButton(tr("res.input.canvas.btn"))
        use_canvas_btn.setToolTip(tr("res.input.canvas.tt"))
        use_canvas_btn.clicked.connect(self._use_canvas_image)
        il.addWidget(use_canvas_btn)
        v.addWidget(inp_grp)

        # ── Operazione ────────────────────────────────────────────────────
        op_grp = QGroupBox(tr("res.op.group"))
        ov = QVBoxLayout(op_grp)
        self._op_group = QButtonGroup(self)
        self.radio_restore = QRadioButton(tr("res.op.restore"))
        self.radio_colorize = QRadioButton(tr("res.op.colorize"))
        self.radio_both = QRadioButton(tr("res.op.both"))
        self.radio_faceswap = QRadioButton(tr("res.op.faceswap"))   # T5: face-swap su foto
        self.radio_restore.setChecked(True)
        for rb in (self.radio_restore, self.radio_colorize, self.radio_both, self.radio_faceswap):
            self._op_group.addButton(rb)
            ov.addWidget(rb)
        self.radio_colorize.toggled.connect(self._on_op_changed)
        self.radio_both.toggled.connect(self._on_op_changed)
        self.radio_faceswap.toggled.connect(self._on_op_changed)
        v.addWidget(op_grp)

        # ── Impostazioni restauro ─────────────────────────────────────────
        self.restore_grp = QGroupBox(tr("res.restore.group"))
        rf = QFormLayout(self.restore_grp)
        rf.setLabelAlignment(Qt.AlignRight)
        self.upscale_model = QComboBox()
        self.upscale_model.setMaxVisibleItems(20)
        self.upscale_model.view().setMinimumWidth(380)
        rf.addRow(tr("res.restore.model_lbl"), self.upscale_model)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(1.0, 4.0)
        self.scale_spin.setSingleStep(0.5)
        self.scale_spin.setValue(2.0)
        self.scale_spin.setSuffix("×")
        rf.addRow(tr("res.restore.scale_lbl"), self.scale_spin)
        self.denoise_chk = QCheckBox(tr("res.restore.denoise"))
        self.denoise_chk.setChecked(False)
        self.denoise_chk.setToolTip(tr("res.restore.denoise.tt"))
        rf.addRow(tr("res.restore.denoise_lbl"), self.denoise_chk)
        self.denoise_str = QDoubleSpinBox()
        self.denoise_str.setRange(0.05, 0.50)
        self.denoise_str.setSingleStep(0.05)
        self.denoise_str.setValue(0.20)
        self.denoise_str.setEnabled(False)
        self.denoise_chk.toggled.connect(self.denoise_str.setEnabled)
        rf.addRow(tr("res.restore.denoise_str_lbl"), self.denoise_str)
        v.addWidget(self.restore_grp)

        # ── Impostazioni colorizzazione ───────────────────────────────────
        self.color_grp = QGroupBox(tr("res.color.group"))
        self.color_grp.setEnabled(False)
        cf = QFormLayout(self.color_grp)
        cf.setLabelAlignment(Qt.AlignRight)
        self.flux2_model = QComboBox()
        self.flux2_model.setMaxVisibleItems(15)
        self.flux2_model.view().setMinimumWidth(380)
        cf.addRow(tr("res.color.model_lbl"), self.flux2_model)
        self.subject_edit = QLineEdit()
        self.subject_edit.setPlaceholderText(tr("res.color.subject.ph"))
        cf.addRow(tr("res.color.subject_lbl"), self.subject_edit)
        self.style_edit = QLineEdit()
        self.style_edit.setText("natural realistic colors, period-accurate")
        cf.addRow(tr("res.color.style_lbl"), self.style_edit)
        self.color_steps = QSpinBox()
        self.color_steps.setRange(5, 50)
        self.color_steps.setValue(20)
        cf.addRow(tr("res.color.steps_lbl"), self.color_steps)
        v.addWidget(self.color_grp)

        # ── Face-swap (T5): impone un volto sorgente sulla foto target ─────
        self.faceswap_grp = QGroupBox(tr("res.faceswap.group"))
        self.faceswap_grp.setEnabled(False)
        ff = QFormLayout(self.faceswap_grp)
        ff.setLabelAlignment(Qt.AlignRight)
        frow = QHBoxLayout()
        self.fs_source = QLineEdit(); self.fs_source.setReadOnly(True)
        self.fs_source.setPlaceholderText(tr("res.faceswap.source.ph"))
        fs_btn = QPushButton(tr("res.faceswap.source.btn"))
        fs_btn.clicked.connect(self._pick_fs_source)
        frow.addWidget(self.fs_source, 1); frow.addWidget(fs_btn)
        _w = QWidget(); _w.setLayout(frow)
        ff.addRow(tr("res.faceswap.source_lbl"), _w)
        v.addWidget(self.faceswap_grp)

        # ── Note / Avvisi ─────────────────────────────────────────────────
        self.note_lbl = QLabel(tr("res.note.lbl"))
        self.note_lbl.setWordWrap(True)
        self.note_lbl.setStyleSheet("color: #74c0fc; padding: 6px;")
        v.addWidget(self.note_lbl)

        v.addStretch()
        scroll.setWidget(container)
        root.addWidget(scroll, 1)

        # ── Pulsante Processa ─────────────────────────────────────────────
        proc_btn = QPushButton(tr("res.run.btn"))
        proc_btn.setObjectName("generate")
        proc_btn.clicked.connect(self._on_process)
        root.addWidget(proc_btn)

    # ---------- slot ----------
    def _on_op_changed(self, _checked: bool):
        is_fs = self.radio_faceswap.isChecked()
        needs_color = self.radio_colorize.isChecked() or self.radio_both.isChecked()
        self.color_grp.setEnabled(needs_color and not is_fs)
        self.restore_grp.setEnabled(not self.radio_colorize.isChecked() and not is_fs)
        self.faceswap_grp.setEnabled(is_fs)

    def _pick_fs_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("res.faceswap.source.btn"), str(Path.home()), tr("res.input.image_filter"))
        if path:
            self.fs_source.setText(path)

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("res.input.browse.dlg"),
            str(Path.home()),
            tr("res.input.image_filter"))
        if path:
            self._input_path = path
            self.path_edit.setText(path)

    def _use_canvas_image(self):
        """Recupera il path dell'immagine dal tab Frame1 tramite segnale (cablato in main_window)."""
        # Emette request_restore con image_path="<from_canvas>" come placeholder;
        # main_window intercetta e usa frame1_tab.image_path
        self._input_path = "<from_canvas>"
        self.path_edit.setText(tr("res.input.canvas.selected"))

    def _on_process(self):
        if self.radio_restore.isChecked():
            op = "restore"
        elif self.radio_colorize.isChecked():
            op = "colorize"
        elif self.radio_faceswap.isChecked():
            op = "faceswap"
        else:
            op = "restore_and_colorize"

        cfg = {
            "operation": op,
            "image_path": self._input_path,
            "source_face": self.fs_source.text().strip(),
            # restauro
            "upscale_model": self.upscale_model.currentText() or "",
            "target_scale": self.scale_spin.value(),
            "denoise": self.denoise_chk.isChecked(),
            "denoise_strength": self.denoise_str.value(),
            # colorizzazione
            "flux2_model": self.flux2_model.currentText() or "",
            "subject": self.subject_edit.text().strip(),
            "style": self.style_edit.text().strip() or "natural realistic colors",
            "color_steps": self.color_steps.value(),
        }
        self.request_restore.emit(cfg)

    # ---------- API inventory ----------
    def set_upscale_models(self, models: list[str]):
        """Popola il combo modelli upscale (chiamato da main_window al caricamento inventario)."""
        prev = self.upscale_model.currentText()
        self.upscale_model.clear()
        preferred = [m for m in models if "ultra" in m.lower() or "real" in m.lower() or "nomos" in m.lower()]
        rest = [m for m in models if m not in preferred]
        for m in preferred + rest:
            self.upscale_model.addItem(m)
        idx = self.upscale_model.findText(prev)
        if idx >= 0:
            self.upscale_model.setCurrentIndex(idx)

    def set_flux2_models(self, models: list[str]):
        """Popola il combo modelli FLUX.2 per colorizzazione."""
        prev = self.flux2_model.currentText()
        self.flux2_model.clear()
        flux2 = [m for m in models if "flux2" in m.lower() or "flux_2" in m.lower() or "flux 2" in m.lower()]
        if not flux2:
            self.flux2_model.addItem(tr("res.color.no_model"))
        else:
            for m in flux2:
                self.flux2_model.addItem(m)
        idx = self.flux2_model.findText(prev)
        if idx >= 0:
            self.flux2_model.setCurrentIndex(idx)

    def get_config(self) -> dict:
        """Ritorna la configurazione corrente (per lettura esterna senza cliccare il pulsante)."""
        op = ("restore" if self.radio_restore.isChecked()
              else "colorize" if self.radio_colorize.isChecked()
              else "faceswap" if self.radio_faceswap.isChecked()
              else "restore_and_colorize")
        return {
            "operation": op,
            "image_path": self._input_path,
            "source_face": self.fs_source.text().strip(),
            "upscale_model": self.upscale_model.currentText() or "",
            "target_scale": self.scale_spin.value(),
            "denoise": self.denoise_chk.isChecked(),
            "denoise_strength": self.denoise_str.value(),
            "flux2_model": self.flux2_model.currentText() or "",
            "subject": self.subject_edit.text().strip(),
            "style": self.style_edit.text().strip() or "natural realistic colors",
            "color_steps": self.color_steps.value(),
        }
