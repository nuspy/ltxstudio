"""Tab ControlNet — copia il movimento da un video di riferimento al video generato.

I 3 tipi disponibili (tutti implementati nel workflow `pluto_superset`):
- **Canny** — copia i contorni del soggetto (forma generale).
- **Depth** — copia la geometria 3D / profondità (struttura della scena).
- **Pose** — copia la posa dello scheletro umano (movimento di una persona).

Componenti del workflow:
- `VHS_LoadVideo` (5069)             → carica il video di riferimento
- `CannyEdgePreprocessor` (4991)    → estrazione canny
- `AIO_Preprocessor` (5068)         → DepthAnythingV2 per la profondità
- `DWPreprocessor` (4986)            → DWPose per lo scheletro
- `ImpactSwitch` (5564, binding `controlnet_type`) → seleziona quale ramo usare
- `LTXAddVideoICLoRAGuide` (5012)   → applica il guidance al modello

La generazione si lancia dal pulsante «Genera» del pannello principale (in basso a sinistra),
gli stessi che generano il video normale. Il tab ControlNet ne attiva il ramo del workflow.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QComboBox, QPushButton, QLabel, QFileDialog,
    QGroupBox,
)

from ui.i18n_manager import tr


class ControlNetTab(QWidget):
    # Mantengo l'ordine UI: 0 Canny, 1 Depth, 2 Pose
    # (la mappatura a ImpactSwitch.select avviene in main_window: 0→2, 1→1, 2→3)
    TYPES = [
        ("Canny — contorni del soggetto",      "Adatto a oggetti, prodotti, forme rigide"),
        ("Depth — geometria 3D / profondità",  "Adatto a scene complete (ambienti, paesaggi)"),
        ("Pose — scheletro umano (DWPose)",    "Adatto al movimento del corpo (danza, sport, gesti)"),
    ]

    def __init__(self):
        super().__init__()
        self._video: str | None = None
        lay = QVBoxLayout(self)

        intro = QLabel(tr("cn.intro.lbl"))
        intro.setWordWrap(True); intro.setStyleSheet("color:#ccd; padding:4px;")
        lay.addWidget(intro)

        self.enable = QCheckBox(tr("cn.enable"))
        lay.addWidget(self.enable)

        type_box = QGroupBox(tr("cn.type.group"))
        tv = QVBoxLayout(type_box)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("cn.type_lbl")))
        self.type = QComboBox()
        for label, _ in self.TYPES:
            self.type.addItem(label)
        self.type.currentIndexChanged.connect(self._on_type_changed)
        row.addWidget(self.type); row.addStretch(1)
        tv.addLayout(row)
        self.type_hint = QLabel(self.TYPES[0][1])
        self.type_hint.setStyleSheet("color:#888; padding:2px 6px;")
        tv.addWidget(self.type_hint)
        lay.addWidget(type_box)

        video_box = QGroupBox(tr("cn.video.group"))
        vb = QVBoxLayout(video_box)
        row2 = QHBoxLayout()
        self.pick = QPushButton(tr("cn.video.pick.btn")); self.pick.clicked.connect(self._pick)
        self.lbl = QLabel(tr("cn.video.none.lbl"))
        row2.addWidget(self.pick); row2.addWidget(self.lbl, 1)
        vb.addLayout(row2)
        vhint = QLabel(tr("cn.video.hint.lbl"))
        vhint.setWordWrap(True); vhint.setStyleSheet("color:#888;")
        vb.addWidget(vhint)
        lay.addWidget(video_box)

        usage_box = QGroupBox(tr("cn.usage.group"))
        uv = QVBoxLayout(usage_box)
        usage = QLabel(tr("cn.usage.lbl"))
        usage.setTextFormat(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.RichText)
        usage.setWordWrap(True)
        uv.addWidget(usage)
        lay.addWidget(usage_box)

        lay.addStretch(1)
        self._on_type_changed(0)

    def _on_type_changed(self, idx: int):
        if 0 <= idx < len(self.TYPES):
            self.type_hint.setText(self.TYPES[idx][1])

    def _pick(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Video riferimento", "", "Video (*.mp4 *.mov *.webm *.mkv)")
        if fn:
            self._video = fn; self.lbl.setText(Path(fn).name)

    def get_values(self) -> dict:
        type_idx = self.type.currentIndex()  # 0 canny, 1 depth, 2 pose
        return {"controlnet_enabled": self.enable.isChecked(),
                "controlnet_ref_video": self._video,
                "controlnet_type": type_idx}
