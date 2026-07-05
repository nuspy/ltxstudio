"""Selettore captioner riutilizzabile per i wizard LoRA (immagini/video).

Motore (Procedurale / JoyCaption / Florence-2 — solo quelli con venv+pesi presenti) + modalità JoyCaption.
get() → {"caption_engine", "caption_mode"} da passare ai servizi lora_*_service.
"""
from __future__ import annotations

from PySide6.QtWidgets import QWidget, QHBoxLayout, QComboBox

from ui.i18n_manager import tr


class CaptionPicker(QWidget):
    def __init__(self):
        super().__init__()
        from core import caption_service as cs
        h = QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0)
        self.engine = QComboBox()
        self.engine.addItem(tr("ilora.cap.proc", "Procedurale (trigger+tag)"), "procedural")
        _lbl = {"joycaption": tr("train.cap.joy", "JoyCaption"), "florence2": tr("train.cap.flo", "Florence-2")}
        for e in cs.available_engines():
            self.engine.addItem(_lbl.get(e, e), e)
        self.engine.setToolTip(tr("ilora.cap.tt",
            "Come descrivere il dataset. JoyCaption = caption VLM realistiche (migliori per il training, "
            "non censurato); Procedurale = trigger + tag generati (nessun modello). Per i video: 1 frame/clip."))
        self.mode = QComboBox()
        for m, lab in (("descriptive", tr("train.cap.m.desc", "Descrittiva")),
                       ("tags", tr("train.cap.m.tags", "Tag booru")), ("sd-prompt", tr("train.cap.m.sd", "Prompt SD"))):
            self.mode.addItem(lab, m)
        self.mode.setToolTip(tr("train.cap.mode.tt", "Stile della caption JoyCaption (ignorato da Florence-2)."))
        self.engine.currentIndexChanged.connect(self._sync)
        self._sync()
        h.addWidget(self.engine, 1); h.addWidget(self.mode)

    def _sync(self):
        self.mode.setEnabled(self.engine.currentData() == "joycaption")

    def get(self) -> dict:
        return {"caption_engine": self.engine.currentData() or "procedural",
                "caption_mode": self.mode.currentData() or "descriptive"}
