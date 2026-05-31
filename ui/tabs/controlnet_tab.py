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

        intro = QLabel(
            "<b>ControlNet</b> copia il <i>movimento/forma</i> da un video di riferimento al video generato. "
            "Il contenuto rimane definito dal prompt nel tab <b>Video</b>; il movimento è dettato dal video qui caricato. "
            "Tutti e tre i tipi sono completamente implementati nel workflow."
        )
        intro.setWordWrap(True); intro.setStyleSheet("color:#ccd; padding:4px;")
        lay.addWidget(intro)

        self.enable = QCheckBox("✅ Abilita ControlNet (copia il movimento dal video qui sotto)")
        lay.addWidget(self.enable)

        type_box = QGroupBox("Tipo di guida")
        tv = QVBoxLayout(type_box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Tipo:"))
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

        video_box = QGroupBox("Video di riferimento")
        vb = QVBoxLayout(video_box)
        row2 = QHBoxLayout()
        self.pick = QPushButton("📁 Carica video di riferimento…"); self.pick.clicked.connect(self._pick)
        self.lbl = QLabel("(nessun video)")
        row2.addWidget(self.pick); row2.addWidget(self.lbl, 1)
        vb.addLayout(row2)
        vhint = QLabel(
            "Il video deve avere movimento chiaramente leggibile (es. per Pose: il soggetto deve "
            "essere ben visibile). Risoluzione e durata: indifferenti — il workflow ricampiona "
            "alla risoluzione del tab Video e tronca alla durata configurata."
        )
        vhint.setWordWrap(True); vhint.setStyleSheet("color:#888;")
        vb.addWidget(vhint)
        lay.addWidget(video_box)

        usage_box = QGroupBox("Come si usa")
        uv = QVBoxLayout(usage_box)
        usage = QLabel(
            "<b>1. Configura</b>: abilita la checkbox sopra, scegli il tipo, carica il video di riferimento.<br>"
            "<b>2. Prompt</b>: nel tab <b>Video</b> scrivi il prompt che descrive il <i>contenuto</i> del video "
            "che vuoi generare (soggetto, ambiente, stile). Il prompt è obbligatorio — descrive la scena.<br>"
            "<b>3. Lancia</b>: dal pannello principale, bottone <b>«🎬 Genera»</b> (in basso a sinistra, "
            "lo stesso del flusso normale).<br><br>"
            "<b>Esempio</b>: video reference = persona che balla → prompt = "
            "<i>«una ballerina con abito rosso danza in una sala elegante, illuminazione soffusa»</i> → "
            "il modello produce una ballerina con quel movimento."
        )
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
