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

ENGINES = ["SeedVR2 (AI, consigliato)", "ESRGAN per-fotogramma (veloce)", "FlashVSR (richiede nodo)"]
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
        self.video_lbl = QLabel("(nessun video — genera un video o caricane uno)")
        self.video_lbl.setWordWrap(True)
        pick = QPushButton("📁 Carica video…"); pick.clicked.connect(self._pick)
        srow.addWidget(self.video_lbl, 1); srow.addWidget(pick)
        root.addLayout(srow)

        # motore + parametri
        box = QGroupBox("Restauro / super-risoluzione"); f = QFormLayout(box)
        self.engine = QComboBox(); self.engine.addItems(ENGINES)
        self.engine.currentIndexChanged.connect(self._on_engine)
        f.addRow("Motore:", self.engine)
        self.seedvr2_model = QComboBox(); self.seedvr2_model.addItems(list(SEEDVR2_MODELS.keys()))
        f.addRow("Modello SeedVR2:", self.seedvr2_model)
        self.resolution = QSpinBox(); self.resolution.setRange(256, 8192); self.resolution.setSingleStep(64)
        self.resolution.setValue(1080); self.resolution.setSuffix(" px (lato corto)")
        f.addRow("Risoluzione target:", self.resolution)
        self.blocks = QSpinBox(); self.blocks.setRange(0, 36); self.blocks.setValue(0)
        self.blocks.setToolTip("BlockSwap: >0 riduce la VRAM (più lento). 0 = tutto in GPU (32GB ok).")
        f.addRow("BlockSwap (VRAM):", self.blocks)
        self.batch = QSpinBox(); self.batch.setRange(1, 121); self.batch.setSingleStep(4); self.batch.setValue(5)
        self.batch.setToolTip("Fotogrammi per batch (formula 4n+1: 1,5,9,…). Più alto = più coerenza, più VRAM.")
        f.addRow("Batch (4n+1):", self.batch)
        # esrgan
        self.esrgan_model = QComboBox()
        self.esrgan_model.setMaxVisibleItems(20)   # fix: mostra tutti i modelli nel popup
        self.esrgan_model.view().setMinimumWidth(420)
        f.addRow("Modello ESRGAN:", self.esrgan_model)
        self.scale = QDoubleSpinBox(); self.scale.setRange(1.0, 8.0); self.scale.setSingleStep(0.5); self.scale.setValue(2.0)
        f.addRow("Scala (ESRGAN):", self.scale)
        root.addWidget(box)

        # RIFE / FPS
        rbox = QGroupBox("Aumento FPS (RIFE, opzionale)"); rf = QFormLayout(rbox)
        self.rife = QCheckBox("Interpola fotogrammi (più fluidità)"); rf.addRow(self.rife)
        self.rife_mult = QComboBox(); self.rife_mult.addItems(["2×", "3×", "4×"])
        rf.addRow("Moltiplicatore FPS:", self.rife_mult)
        self.rife_ckpt = QComboBox(); self.rife_ckpt.addItems(["rife47.pth", "rife48.pth", "rife46.pth"])
        rf.addRow("Modello RIFE:", self.rife_ckpt)
        root.addWidget(rbox)

        self.run_btn = QPushButton("✨ Migliora video")
        self.run_btn.setObjectName("generate"); self.run_btn.clicked.connect(lambda: self.request_enhance.emit())
        root.addWidget(self.run_btn)
        note = QLabel("SeedVR2 scarica il modello al primo uso (in ComfyUI/models/SEEDVR2). "
                      "Mantiene l'audio originale.")
        note.setWordWrap(True); note.setStyleSheet("color:#888;")
        root.addWidget(note)

        # ── Restauro volti (CodeFormer/GFPGAN) ──
        fbox = QGroupBox("🙂 Restauro volti (CodeFormer / GFPGAN)")
        ff = QFormLayout(fbox)
        info = QLabel("Ricostruisce i volti frame-by-frame correggendo l'aspetto «plastico» e la "
                      "deriva d'identità di LTX, <b>preservando l'identità</b> (non ri-genera col "
                      "diffusion). Applicalo come passaggio finale sul video.")
        info.setWordWrap(True); info.setStyleSheet("color:#9aa1b0;"); info.setTextFormat(Qt.RichText)
        ff.addRow(info)
        self.fr_model = QComboBox(); self.fr_model.addItems(facerestore.FACERESTORE_MODELS)
        ff.addRow("Modello:", self.fr_model)
        self.fr_detection = QComboBox(); self.fr_detection.addItems(facerestore.FACE_DETECTORS)
        ff.addRow("Rilevatore volti:", self.fr_detection)
        frow = QHBoxLayout()
        self.fr_fidelity = QSlider(Qt.Horizontal); self.fr_fidelity.setRange(0, 100); self.fr_fidelity.setValue(70)
        self.fr_fidelity_lbl = QLabel("0.70")
        self.fr_fidelity.valueChanged.connect(lambda v: self.fr_fidelity_lbl.setText(f"{v/100:.2f}"))
        self.fr_fidelity.setToolTip("Fedeltà all'input: 1.0 preserva di più l'originale; "
                                    "0.0 ricostruisce di più (più nitido, può scostarsi). ~0.7 consigliato.")
        frow.addWidget(self.fr_fidelity, 1); frow.addWidget(self.fr_fidelity_lbl)
        ff.addRow("Fedeltà (fidelity):", self._wrap(frow))
        brow = QHBoxLayout()
        self.fr_run_btn = QPushButton("🙂 Restaura volti del video")
        self.fr_run_btn.clicked.connect(self._on_face_restore_clicked)
        self.fr_install_btn = QPushButton("📥 Installa face-restore")
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
        self.esrgan_model.addItems(names or ["(nessun modello upscale)"])

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
        self.fr_status.setText("✅ facerestore_cf installato." if installed else
                               "⬇ Nodo non installato: premi «Installa face-restore» (poi riavvia ComfyUI).")

    def _on_face_restore_clicked(self):
        if not self._video:
            QMessageBox.warning(self, "Nessun video", "Carica o genera un video prima.")
            return
        self.request_face_restore.emit()

    def _install_facerestore(self):
        if QMessageBox.question(self, "Installa face-restore",
                                f"Clonare {facerestore.FACERESTORE_REPO['url']} in custom_nodes e "
                                "installarne le dipendenze?\n\n"
                                f"{facerestore.FACERESTORE_REPO['note']}") != QMessageBox.Yes:
            return
        self.fr_install_btn.setEnabled(False); self.fr_install_btn.setText("⏳ Installazione…")
        QApplication.processEvents()
        try:
            ok, msg = facerestore.install_node()
        except Exception as e:
            ok, msg = False, str(e)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information if ok else QMessageBox.Warning)
        box.setWindowTitle("Installazione " + ("ok" if ok else "fallita"))
        box.setText(msg if len(msg) < 400 else msg[:400] + "…"); box.setDetailedText(msg); box.exec()
        self.fr_install_btn.setEnabled(True); self.fr_install_btn.setText("📥 Installa face-restore")
        self._refresh_facerestore_status()

    # ---- interno ----
    def _on_engine(self, *_):
        key = _ENGINE_KEY[self.engine.currentIndex()]
        sv = key == "seedvr2"; es = key == "esrgan"
        self.seedvr2_model.setEnabled(sv); self.blocks.setEnabled(sv); self.batch.setEnabled(sv)
        self.resolution.setEnabled(sv)
        self.esrgan_model.setEnabled(es); self.scale.setEnabled(es)

    def _pick(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Video", "", "Video (*.mp4 *.webm *.mkv *.mov)")
        if fn:
            self.set_video(fn)
