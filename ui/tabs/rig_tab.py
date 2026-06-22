"""Tab Rig 3D — rigging automatico di una mesh (SkinTokens / TokenRig).

Da un modello 3D (.glb), tipicamente quello generato da TRELLIS.2, predice
automaticamente scheletro + pesi di skinning ed esporta un GLB/FBX rigged,
mostrato nel visore 3D integrato.

L'operazione viene inviata a ComfyUI tramite core/rig_pipeline.py.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from ui.i18n_manager import tr
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton,
    QLineEdit, QCheckBox, QFileDialog, QScrollArea, QSplitter,
)

# Riusa il visore 3D del tab TRELLIS (interattivo se disponibile)
from ui.tabs.trellis_tab import make_viewer


class RigTab(QWidget):
    """Tab per il rigging automatico di una mesh 3D."""

    request_rig = Signal(dict)   # config → MainWindow

    def __init__(self):
        super().__init__()
        self._input_path: str = ""
        self._last_mesh: str = ""   # ultimo .glb prodotto da TRELLIS
        self._build()

    # ---------- build ----------
    def _build(self):
        root = QVBoxLayout(self)
        split = QSplitter(Qt.Horizontal)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        v = QVBoxLayout(container); v.setSpacing(8)

        # ── Mesh di input ──
        inp_grp = QGroupBox(tr("rig.input.group"))
        il = QHBoxLayout(inp_grp)
        self.path_edit = QLineEdit(); self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText(tr("rig.input.ph"))
        il.addWidget(self.path_edit, 1)
        browse_btn = QPushButton(tr("rig.input.browse"))
        browse_btn.clicked.connect(self._browse_input)
        il.addWidget(browse_btn)
        self.last_btn = QPushButton(tr("rig.input.last"))
        self.last_btn.setToolTip(tr("rig.input.last.tt"))
        self.last_btn.clicked.connect(self._use_last_mesh)
        self.last_btn.setEnabled(False)
        il.addWidget(self.last_btn)
        v.addWidget(inp_grp)

        # ── Scheletro / export ──
        s_grp = QGroupBox(tr("rig.skel.group"))
        sf = QFormLayout(s_grp); sf.setLabelAlignment(Qt.AlignRight)
        self.skeleton = QComboBox()
        self.skeleton.addItems(["Keep model names", "Mixamo", "Unreal Engine 5"])
        self.skeleton.setToolTip(tr("rig.skel.template.tt"))
        sf.addRow(tr("rig.skel.template"), self.skeleton)
        self.file_format = QComboBox()
        self.file_format.addItems(["glb", "fbx"])
        self.file_format.setToolTip(tr("rig.skel.format.tt"))
        sf.addRow(tr("rig.skel.format"), self.file_format)
        self.bottom_origin = QCheckBox(tr("rig.skel.bottom"))
        self.bottom_origin.setToolTip(tr("rig.skel.bottom.tt"))
        sf.addRow(tr("rig.skel.bottom_lbl"), self.bottom_origin)
        v.addWidget(s_grp)

        # ── Skinning ──
        k_grp = QGroupBox(tr("rig.skin.group"))
        kf = QFormLayout(k_grp); kf.setLabelAlignment(Qt.AlignRight)
        self.transfer = QCheckBox(tr("rig.skin.transfer"))
        self.transfer.setChecked(True)
        self.transfer.setToolTip(tr("rig.skin.transfer.tt"))
        kf.addRow(tr("rig.skin.transfer_lbl"), self.transfer)
        self.postprocess = QCheckBox(tr("rig.skin.postprocess"))
        self.postprocess.setToolTip(tr("rig.skin.postprocess.tt"))
        kf.addRow(tr("rig.skin.postprocess_lbl"), self.postprocess)
        self.gpv = QSpinBox(); self.gpv.setRange(1, 32); self.gpv.setValue(4)
        self.gpv.setToolTip(tr("rig.skin.gpv.tt"))
        kf.addRow(tr("rig.skin.gpv"), self.gpv)
        self.smooth = QDoubleSpinBox(); self.smooth.setRange(0.0, 180.0)
        self.smooth.setSingleStep(1.0); self.smooth.setValue(55.0); self.smooth.setSuffix("°")
        kf.addRow(tr("rig.skin.smooth"), self.smooth)
        v.addWidget(k_grp)

        # ── Avanzate (campionamento) ──
        a_grp = QGroupBox(tr("rig.adv.group"))
        a_grp.setCheckable(True); a_grp.setChecked(False)
        af = QFormLayout(a_grp); af.setLabelAlignment(Qt.AlignRight)
        self.num_beams = QSpinBox(); self.num_beams.setRange(1, 20); self.num_beams.setValue(10)
        self.num_beams.setToolTip(tr("rig.adv.beams.tt"))
        af.addRow(tr("rig.adv.beams"), self.num_beams)
        self.temperature = QDoubleSpinBox(); self.temperature.setRange(0.1, 2.0)
        self.temperature.setSingleStep(0.1); self.temperature.setValue(1.0)
        af.addRow(tr("rig.adv.temp"), self.temperature)
        self.top_k = QSpinBox(); self.top_k.setRange(1, 200); self.top_k.setValue(5)
        af.addRow(tr("rig.adv.topk"), self.top_k)
        self.top_p = QDoubleSpinBox(); self.top_p.setRange(0.1, 1.0)
        self.top_p.setSingleStep(0.01); self.top_p.setValue(0.95)
        af.addRow(tr("rig.adv.topp"), self.top_p)
        self.rep_pen = QDoubleSpinBox(); self.rep_pen.setRange(0.5, 3.0)
        self.rep_pen.setSingleStep(0.1); self.rep_pen.setValue(2.0)
        af.addRow(tr("rig.adv.reppen"), self.rep_pen)
        v.addWidget(a_grp)

        note = QLabel(tr("rig.note"))
        note.setWordWrap(True); note.setStyleSheet("color:#74c0fc; padding:6px;")
        v.addWidget(note)
        v.addStretch()
        scroll.setWidget(container)

        self.viewer = make_viewer()
        split.addWidget(scroll)
        split.addWidget(self.viewer)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([420, 600])
        root.addWidget(split, 1)

        self.gen_btn = QPushButton(tr("rig.run.btn"))
        self.gen_btn.setObjectName("generate")
        self.gen_btn.clicked.connect(self._on_generate)
        root.addWidget(self.gen_btn)

    # ---------- slot ----------
    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("rig.input.browse"), str(Path.home()),
            tr("rig.input.filter"))
        if path:
            self._input_path = path
            self.path_edit.setText(path)

    def _use_last_mesh(self):
        if self._last_mesh:
            self._input_path = self._last_mesh
            self.path_edit.setText(self._last_mesh)

    def _on_generate(self):
        self.request_rig.emit(self.get_config())

    # ---------- API ----------
    def set_last_mesh(self, path: str):
        """Registra l'ultimo modello 3D generato (TRELLIS) per il pulsante 'Usa ultimo'."""
        self._last_mesh = path
        self.last_btn.setEnabled(bool(path))

    def get_config(self) -> dict:
        return {
            "mesh_path": self._input_path,
            "skeleton_template": self.skeleton.currentText(),
            "file_format": self.file_format.currentText(),
            "bottom_center_origin": self.bottom_origin.isChecked(),
            "use_transfer": self.transfer.isChecked(),
            "use_postprocess": self.postprocess.isChecked(),
            "group_per_vertex": self.gpv.value(),
            "smooth_angle": self.smooth.value(),
            "num_beams": self.num_beams.value(),
            "temperature": self.temperature.value(),
            "top_k": self.top_k.value(),
            "top_p": self.top_p.value(),
            "repetition_penalty": self.rep_pen.value(),
        }

    def show_result(self, path: str):
        self.viewer.load_model(path)
