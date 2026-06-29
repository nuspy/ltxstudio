"""Tab "Definizione video": restore + super-risoluzione + (opz.) aumento FPS.

Carica in automatico l'ultimo video generato; permette di sceglierne un altro. Motori:
SeedVR2 (consigliato, AI video), ESRGAN per-fotogramma (veloce), FlashVSR (se installato).
Opzione RIFE per aumentare gli FPS. Vedi core/video_enhance_pipeline.py.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox,
    QDoubleSpinBox, QPushButton, QCheckBox, QGroupBox, QFileDialog, QScrollArea,
    QSlider, QMessageBox, QApplication,
)
from PySide6.QtCore import Signal, Qt

from core.video_enhance_pipeline import SEEDVR2_MODELS
from core import face_restore_pipeline as facerestore
from ui.i18n_manager import tr

_ENGINE_KEY = {0: "seedvr2", 1: "esrgan", 2: "flashvsr"}


class VideoDefTab(QWidget):
    request_enhance = Signal()
    request_face_restore = Signal()

    def __init__(self):
        super().__init__()
        self._video: str | None = None
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QScrollArea.NoFrame)
        host = QWidget(); root = QVBoxLayout(host); sc.setWidget(host); outer.addWidget(sc)

        # sorgente
        srow = QHBoxLayout()
        self.video_lbl = QLabel(tr("def.no_video.lbl"))
        self.video_lbl.setWordWrap(True)
        pick = QPushButton(tr("def.pick.btn")); pick.clicked.connect(self._pick)
        srow.addWidget(self.video_lbl, 1); srow.addWidget(pick)
        root.addLayout(srow)

        # motore + parametri
        box = QGroupBox(tr("def.restore.group")); f = QFormLayout(box)
        self.engine = QComboBox()
        self.engine.addItems([tr("def.engine.seedvr2"), tr("def.engine.esrgan"), tr("def.engine.flashvsr")])
        self.engine.currentIndexChanged.connect(self._on_engine)
        f.addRow(tr("def.engine_lbl"), self.engine)
        self.seedvr2_model = QComboBox(); self.seedvr2_model.addItems(list(SEEDVR2_MODELS.keys()))
        f.addRow(tr("def.seedvr2_model_lbl"), self.seedvr2_model)
        self.resolution = QSpinBox(); self.resolution.setRange(256, 8192); self.resolution.setSingleStep(64)
        self.resolution.setValue(1080); self.resolution.setSuffix(tr("def.resolution.suffix"))
        f.addRow(tr("def.resolution_lbl"), self.resolution)
        self.blocks = QSpinBox(); self.blocks.setRange(0, 36); self.blocks.setValue(0)
        self.blocks.setToolTip(tr("def.blocks.tt"))
        f.addRow(tr("def.blocks_lbl"), self.blocks)
        self.batch = QSpinBox(); self.batch.setRange(1, 121); self.batch.setSingleStep(4); self.batch.setValue(5)
        self.batch.setToolTip(tr("def.batch.tt"))
        f.addRow(tr("def.batch_lbl"), self.batch)
        # esrgan
        self.esrgan_model = QComboBox()
        self.esrgan_model.setMaxVisibleItems(20)   # fix: mostra tutti i modelli nel popup
        self.esrgan_model.view().setMinimumWidth(420)
        f.addRow(tr("def.esrgan_model_lbl"), self.esrgan_model)
        self.scale = QDoubleSpinBox(); self.scale.setRange(1.0, 8.0); self.scale.setSingleStep(0.5); self.scale.setValue(2.0)
        f.addRow(tr("def.scale_lbl"), self.scale)
        root.addWidget(box)

        # RIFE / FPS
        rbox = QGroupBox(tr("def.rife.group")); rf = QFormLayout(rbox)
        self.rife = QCheckBox(tr("def.rife.check")); rf.addRow(self.rife)
        self.rife_mult = QComboBox(); self.rife_mult.addItems(["2×", "3×", "4×"])
        rf.addRow(tr("def.rife_mult_lbl"), self.rife_mult)
        self.rife_ckpt = QComboBox(); self.rife_ckpt.addItems(["rife47.pth", "rife48.pth", "rife46.pth"])
        rf.addRow(tr("def.rife_ckpt_lbl"), self.rife_ckpt)
        root.addWidget(rbox)

        self.run_btn = QPushButton(tr("def.run.btn"))
        self.run_btn.setObjectName("generate"); self.run_btn.clicked.connect(lambda: self.request_enhance.emit())
        root.addWidget(self.run_btn)
        note = QLabel(tr("def.note.lbl"))
        note.setWordWrap(True); note.setStyleSheet("color:#888;")
        root.addWidget(note)

        # ── Restauro volti (CodeFormer/GFPGAN) ──
        fbox = QGroupBox(tr("def.face.group"))
        ff = QFormLayout(fbox)
        info = QLabel(tr("def.face.info.lbl"))
        info.setWordWrap(True); info.setStyleSheet("color:#9aa1b0;"); info.setTextFormat(Qt.RichText)
        ff.addRow(info)
        self.fr_model = QComboBox(); self.fr_model.addItems(facerestore.FACERESTORE_MODELS)
        ff.addRow(tr("def.face.model_lbl"), self.fr_model)
        self.fr_detection = QComboBox(); self.fr_detection.addItems(facerestore.FACE_DETECTORS)
        ff.addRow(tr("def.face.detection_lbl"), self.fr_detection)
        frow = QHBoxLayout()
        self.fr_fidelity = QSlider(Qt.Horizontal); self.fr_fidelity.setRange(0, 100); self.fr_fidelity.setValue(70)
        self.fr_fidelity_lbl = QLabel("0.70")
        self.fr_fidelity.valueChanged.connect(lambda v: self.fr_fidelity_lbl.setText(f"{v/100:.2f}"))
        self.fr_fidelity.setToolTip(tr("def.face.fidelity.tt"))
        frow.addWidget(self.fr_fidelity, 1); frow.addWidget(self.fr_fidelity_lbl)
        ff.addRow(tr("def.face.fidelity_lbl"), self._wrap(frow))
        brow = QHBoxLayout()
        self.fr_run_btn = QPushButton(tr("def.face.run.btn"))
        self.fr_run_btn.clicked.connect(self._on_face_restore_clicked)
        self.fr_install_btn = QPushButton(tr("def.face.install.btn"))
        self.fr_install_btn.clicked.connect(self._install_facerestore)
        brow.addWidget(self.fr_run_btn); brow.addWidget(self.fr_install_btn)
        ff.addRow(self._wrap(brow))
        self.fr_status = QLabel(""); self.fr_status.setStyleSheet("color:#9aa1b0; font-size:11px;")
        self.fr_status.setWordWrap(True)
        ff.addRow(self.fr_status)
        root.addWidget(fbox)
        root.addStretch(1)
        self._on_engine()
        self._refresh_facerestore_status()

    # ---- API ----
    def set_video(self, path: str):
        self._video = path
        self.video_lbl.setText(f"🎬 {Path(path).name}")

    def set_upscale_models(self, names: list[str]):
        self.esrgan_model.clear()
        self.esrgan_model.addItems(names or [tr("def.no_upscale_model")])
        # colori VRAM (header-aware) — path risolto sotto Models\Upscale
        try:
            from ui.widgets.precision_combo import color_existing
            from pathlib import Path as _P
            _root = _P(r"E:\AI\Models\Upscale")
            color_existing(self.esrgan_model,
                           name_to_path=lambda n: str(_root / n) if not _P(n).is_file() else n)
        except Exception:                                    # noqa: BLE001
            pass

    def get_config(self) -> dict:
        return {
            "video": self._video,
            "engine": _ENGINE_KEY[self.engine.currentIndex()],
            "seedvr2_model": SEEDVR2_MODELS[self.seedvr2_model.currentText()],
            "resolution": self.resolution.value(),
            "blocks_to_swap": self.blocks.value(),
            "batch_size": self.batch.value(),
            "esrgan_model": self.esrgan_model.currentText() if not self.esrgan_model.currentText().startswith("(") else "",
            "target_scale": self.scale.value(),
            "rife": self.rife.isChecked(),
            "rife_multiplier": self.rife_mult.currentIndex() + 2,
            "rife_ckpt": self.rife_ckpt.currentText(),
        }

    def _wrap(self, layout):
        w = QWidget(); w.setLayout(layout); return w

    def get_face_restore_config(self) -> dict:
        return {
            "video": self._video,
            "model": self.fr_model.currentText(),
            "detection": self.fr_detection.currentText(),
            "fidelity": self.fr_fidelity.value() / 100.0,
        }

    def _refresh_facerestore_status(self):
        try:
            installed = facerestore.node_installed()
        except Exception:
            installed = False
        self.fr_install_btn.setVisible(not installed)
        self.fr_run_btn.setEnabled(installed)
        self.fr_status.setText(tr("def.face.status_ok") if installed else tr("def.face.status_nok"))

    def _on_face_restore_clicked(self):
        if not self._video:
            QMessageBox.warning(self, tr("def.face.no_video.title"), tr("def.face.no_video.msg"))
            return
        self.request_face_restore.emit()

    def _install_facerestore(self):
        if QMessageBox.question(self, tr("def.face.install.title"),
                                tr("def.face.install.confirm").format(
                                    url=facerestore.FACERESTORE_REPO['url'],
                                    note=facerestore.FACERESTORE_REPO['note'])) != QMessageBox.Yes:
            return
        self.fr_install_btn.setEnabled(False); self.fr_install_btn.setText(tr("def.face.install_update.btn"))
        QApplication.processEvents()
        try:
            ok, msg = facerestore.install_node()
        except Exception as e:
            ok, msg = False, str(e)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information if ok else QMessageBox.Warning)
        box.setWindowTitle(tr("def.face.install.title"))
        box.setText(msg if len(msg) < 400 else msg[:400] + "…"); box.setDetailedText(msg); box.exec()
        self.fr_install_btn.setEnabled(True); self.fr_install_btn.setText(tr("def.face.install.btn"))
        self._refresh_facerestore_status()

    # ---- interno ----
    def _on_engine(self, *_):
        key = _ENGINE_KEY[self.engine.currentIndex()]
        sv = key == "seedvr2"; es = key == "esrgan"
        self.seedvr2_model.setEnabled(sv); self.blocks.setEnabled(sv); self.batch.setEnabled(sv)
        self.resolution.setEnabled(sv)
        self.esrgan_model.setEnabled(es); self.scale.setEnabled(es)

    def _pick(self):
        fn, _ = QFileDialog.getOpenFileName(self, tr("def.pick.dialog.title"), "", tr("def.pick.dialog.filter"))
        if fn:
            self.set_video(fn)
