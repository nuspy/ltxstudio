"""Tab Animazione / Mocap — applica un'animazione mocap al modello 3D riggato.

Due sorgenti (entrambe richieste dall'utente):
  - Cattura webcam in-app (MediaPipe single-camera → BVH, venv mocap_env);
  - Import di una sessione FreeMoCap / file di animazione (.bvh/.fbx).
Il retarget (Blender) applica l'animazione allo SCHELETRO ESISTENTE del rig; se il
modello non ha rig, ricade sull'armatura del BVH/plugin FreeMoCap. Risultato
esportabile in FBX/GLB/BVH e mostrato nel visore 3D interattivo.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QComboBox, QSpinBox, QPushButton, QLineEdit, QFileDialog, QScrollArea,
    QSplitter, QTabWidget,
)

from ui.i18n_manager import tr
from ui.tabs.trellis_tab import make_viewer


class AnimTab(QWidget):
    """Tab per catturare/importare un'animazione e applicarla al rig."""

    request_mocap_start = Signal(dict)   # {"camera": int}
    request_mocap_stop = Signal()
    request_retarget = Signal(dict)      # {"model","anim","format","template"}

    def __init__(self):
        super().__init__()
        self._rig_path: str = ""
        self._last_rig: str = ""
        self._anim_path: str = ""        # bvh/fbx importato o catturato
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        split = QSplitter(Qt.Horizontal)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget(); v = QVBoxLayout(container); v.setSpacing(8)

        # ── Modello riggato di input ──
        r_grp = QGroupBox(tr("anim.rig.group"))
        rl = QHBoxLayout(r_grp)
        self.rig_edit = QLineEdit(); self.rig_edit.setReadOnly(True)
        self.rig_edit.setPlaceholderText(tr("anim.rig.ph"))
        rl.addWidget(self.rig_edit, 1)
        b = QPushButton(tr("anim.rig.browse")); b.clicked.connect(self._browse_rig)
        rl.addWidget(b)
        self.last_btn = QPushButton(tr("anim.rig.last")); self.last_btn.setEnabled(False)
        self.last_btn.clicked.connect(self._use_last_rig)
        rl.addWidget(self.last_btn)
        v.addWidget(r_grp)

        # ── Sorgente animazione (webcam | import) ──
        self.src_tabs = QTabWidget()
        # webcam
        cam = QWidget(); cl = QVBoxLayout(cam)
        cf = QFormLayout(); cf.setLabelAlignment(Qt.AlignRight)
        self.camera = QSpinBox(); self.camera.setRange(0, 8); self.camera.setValue(0)
        cf.addRow(tr("anim.cam.index"), self.camera)
        cl.addLayout(cf)
        cam_btns = QHBoxLayout()
        self.start_btn = QPushButton(tr("anim.cam.start")); self.start_btn.clicked.connect(self._on_start)
        self.stop_btn = QPushButton(tr("anim.cam.stop")); self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        cam_btns.addWidget(self.start_btn); cam_btns.addWidget(self.stop_btn)
        cl.addLayout(cam_btns)
        self.preview = QLabel(tr("anim.cam.preview_off"))
        self.preview.setMinimumHeight(180); self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background:#15161b; color:#6b7280; border-radius:6px;")
        cl.addWidget(self.preview, 1)
        self.src_tabs.addTab(cam, tr("anim.src.webcam"))
        # import
        imp = QWidget(); il = QVBoxLayout(imp)
        ig = QHBoxLayout()
        self.anim_edit = QLineEdit(); self.anim_edit.setReadOnly(True)
        self.anim_edit.setPlaceholderText(tr("anim.import.ph"))
        ig.addWidget(self.anim_edit, 1)
        ib = QPushButton(tr("anim.import.browse")); ib.clicked.connect(self._browse_anim)
        ig.addWidget(ib)
        il.addLayout(ig)
        il.addWidget(QLabel(tr("anim.import.note")))
        il.addStretch()
        self.src_tabs.addTab(imp, tr("anim.src.import"))
        v.addWidget(self.src_tabs)

        # ── Retarget / export ──
        o_grp = QGroupBox(tr("anim.out.group"))
        of = QFormLayout(o_grp); of.setLabelAlignment(Qt.AlignRight)
        self.template = QComboBox()
        self.template.addItems(["auto", "Mixamo", "Unreal Engine 5"])
        self.template.setToolTip(tr("anim.out.template.tt"))
        of.addRow(tr("anim.out.template"), self.template)
        self.fmt = QComboBox(); self.fmt.addItems(["fbx", "glb", "bvh"])
        of.addRow(tr("anim.out.format"), self.fmt)
        v.addWidget(o_grp)

        note = QLabel(tr("anim.note")); note.setWordWrap(True)
        note.setStyleSheet("color:#74c0fc; padding:6px;")
        v.addWidget(note); v.addStretch()
        scroll.setWidget(container)

        self.viewer = make_viewer()
        split.addWidget(scroll); split.addWidget(self.viewer)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 1)
        split.setSizes([440, 600])
        split.setChildrenCollapsible(True)                  # #96.2: separatore libero, senza magnet né limiti
        split.setHandleWidth(8)
        scroll.setMinimumWidth(0); self.viewer.setMinimumWidth(0)
        root.addWidget(split, 1)

        self.apply_btn = QPushButton(tr("anim.apply"))
        self.apply_btn.setObjectName("generate")
        self.apply_btn.clicked.connect(self._on_apply)
        root.addWidget(self.apply_btn)

    # ---------- slot ----------
    def _browse_rig(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("anim.rig.browse"), str(Path.home()),
            "Modelli riggati (*.glb *.gltf *.fbx)")
        if path:
            self._rig_path = path; self.rig_edit.setText(path)

    def _use_last_rig(self):
        if self._last_rig:
            self._rig_path = self._last_rig; self.rig_edit.setText(self._last_rig)

    def _browse_anim(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("anim.import.browse"), str(Path.home()),
            "Animazione (*.bvh *.fbx)")
        if path:
            self._anim_path = path; self.anim_edit.setText(path)

    def _on_start(self):
        self.request_mocap_start.emit({"camera": self.camera.value()})

    def _on_stop(self):
        self.request_mocap_stop.emit()

    def _on_apply(self):
        self.request_retarget.emit({
            "model": self._rig_path,
            "anim": self._anim_path,
            "format": self.fmt.currentText(),
            "template": self.template.currentText(),
        })

    # ---------- API per MainWindow ----------
    def set_last_rig(self, path: str):
        self._last_rig = path
        self.last_btn.setEnabled(bool(path))

    def set_capturing(self, on: bool):
        self.start_btn.setEnabled(not on)
        self.stop_btn.setEnabled(on)

    def set_preview(self, png_path: str, n: int):
        pm = QPixmap(png_path)
        if not pm.isNull():
            self.preview.setPixmap(pm.scaledToHeight(200, Qt.SmoothTransformation))
        self.preview.setToolTip(tr("anim.cam.frames").format(n=n))

    def set_anim(self, path: str):
        """Imposta il BVH catturato/prodotto come sorgente animazione."""
        self._anim_path = path
        self.anim_edit.setText(path)

    def show_result(self, path: str):
        self.viewer.load_model(path)
