"""Tab Convertitore — import/export universale di formati 3D via Blender headless.

Carica un file 3D (qualsiasi formato supportato) e lo riesporta nel formato
scelto. Non usa ComfyUI: delega a Blender (scripts/blender/convert.py) tramite
core/blender_service.BlenderWorker. Output in output/converted/.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QComboBox, QLineEdit, QPushButton, QFileDialog, QCheckBox, QDoubleSpinBox,
    QPlainTextEdit,
)

from ui.i18n_manager import tr

# Formati di export offerti (allineati a EXPORT_OPS in scripts/blender/bl_common.py).
EXPORT_FORMATS = [
    ("glb", "glTF Binary (.glb) — Unity, web, generale"),
    ("gltf", "glTF separato (.gltf)"),
    ("fbx", "FBX (.fbx) — Unity, Unreal, Maya"),
    ("usd", "USD (.usd) — Unreal, Omniverse"),
    ("usdz", "USDZ (.usdz) — Apple/AR"),
    ("obj", "OBJ (.obj) — generale"),
    ("dae", "Collada (.dae)"),
    ("stl", "STL (.stl) — stampa 3D"),
    ("ply", "PLY (.ply)"),
    ("abc", "Alembic (.abc) — VFX"),
    ("bvh", "BVH (.bvh) — solo animazione/scheletro"),
    ("x3d", "X3D (.x3d)"),
    ("dxf", "DXF (.dxf) — AutoCAD (geometria, richiede add-on)"),
]

INPUT_FILTER = (
    "Modelli 3D (*.glb *.gltf *.fbx *.obj *.dae *.stl *.ply "
    "*.usd *.usda *.usdc *.usdz *.abc *.bvh *.x3d *.dxf)"
)


class ConvertTab(QWidget):
    """Convertitore di formati 3D."""

    request_convert = Signal(dict)   # config → MainWindow

    def __init__(self):
        super().__init__()
        self._input_path: str = ""
        self._build()

    def _build(self):
        root = QVBoxLayout(self)

        # ── Input ──
        inp_grp = QGroupBox(tr("convert.input.group"))
        il = QHBoxLayout(inp_grp)
        self.path_edit = QLineEdit(); self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText(tr("convert.input.ph"))
        il.addWidget(self.path_edit, 1)
        browse = QPushButton(tr("convert.input.browse"))
        browse.clicked.connect(self._browse)
        il.addWidget(browse)
        root.addWidget(inp_grp)

        # ── Output / opzioni ──
        o_grp = QGroupBox(tr("convert.out.group"))
        of = QFormLayout(o_grp); of.setLabelAlignment(Qt.AlignRight)
        self.fmt = QComboBox()
        for code, label in EXPORT_FORMATS:
            self.fmt.addItem(label, code)
        of.addRow(tr("convert.out.format"), self.fmt)
        self.scale = QDoubleSpinBox(); self.scale.setRange(0.001, 1000.0)
        self.scale.setValue(1.0); self.scale.setSingleStep(0.1)
        of.addRow(tr("convert.out.scale"), self.scale)
        self.embed = QCheckBox(tr("convert.out.embed"))
        self.embed.setChecked(True)
        of.addRow(tr("convert.out.embed_lbl"), self.embed)
        self.animations = QCheckBox(tr("convert.out.anim"))
        self.animations.setChecked(True)
        of.addRow(tr("convert.out.anim_lbl"), self.animations)
        root.addWidget(o_grp)

        # ── Azione ──
        self.run_btn = QPushButton(tr("convert.run"))
        self.run_btn.setObjectName("generate")
        self.run_btn.clicked.connect(self._on_run)
        root.addWidget(self.run_btn)

        note = QLabel(tr("convert.note"))
        note.setWordWrap(True); note.setStyleSheet("color:#74c0fc; padding:6px;")
        root.addWidget(note)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        root.addWidget(self.log, 1)

    # ---------- slot ----------
    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("convert.input.browse"), str(Path.home()), INPUT_FILTER)
        if path:
            self.set_input_path(path)

    def _on_run(self):
        if not self._input_path:
            self.log_msg(tr("convert.need_input"))
            return
        self.request_convert.emit(self.get_config())

    # ---------- API ----------
    def set_input_path(self, path: str):
        self._input_path = path
        self.path_edit.setText(path)

    def set_format(self, code: str):
        idx = self.fmt.findData(code)
        if idx >= 0:
            self.fmt.setCurrentIndex(idx)

    def get_config(self) -> dict:
        return {
            "input": self._input_path,
            "format": self.fmt.currentData(),
            "options": {
                "scale": self.scale.value(),
                "embed_textures": self.embed.isChecked(),
                "animations": self.animations.isChecked(),
            },
        }

    def log_msg(self, msg: str):
        self.log.appendPlainText(msg)
