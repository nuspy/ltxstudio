"""Tab «Train LoRA Video» — allena una LoRA video LTX-2.3 da SPEZZONI VIDEO ESISTENTI (#99).

A differenza di «LoRA Wizard Vid.» (che auto-genera i clip dai soggetti), qui importi tuoi clip, li tagghi
(caption) ed eventualmente li ritagli, poi si allena la LoRA sul tuo materiale reale. Riusa preprocess +
trainer ufficiale LTX-2 di core/lora_video_service. UX: ui-ux-pro-max (drag&drop+fallback, tooltip,
progressive disclosure, conferma, un solo CTA primario).
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QLabel, QLineEdit, QSpinBox,
    QComboBox, QPushButton, QPlainTextEdit, QMessageBox, QCheckBox,
)

from ui.i18n_manager import tr
from ui.widgets.clip_dataset import ClipDatasetWidget

OUT_DIR = r"E:\AI\Models\LoRA\LTX2"


class _ClipTrainWorker(QThread):
    progress = Signal(str)
    finished_lora = Signal(dict)

    def __init__(self, clips, name, opts):
        super().__init__()
        self._a = (clips, name, opts)

    def run(self):
        clips, name, opts = self._a
        try:
            from core import lora_video_service as svc
            res = svc.make_lora_from_clips(clips, name, OUT_DIR, progress=self.progress.emit, **opts)
            self.finished_lora.emit(res)
        except Exception as e:                              # noqa: BLE001
            self.finished_lora.emit({"error": str(e)})


class TrainLoraVideoTab(QWidget):
    """Import di clip → tag/trim → training LoRA video LTX-2.3."""

    def __init__(self):
        super().__init__()
        self._worker: _ClipTrainWorker | None = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        intro = QLabel(tr("tclip.intro",
            "Allena una LoRA video sui TUOI spezzoni: importa i clip, scrivi una caption per ognuno "
            "(o una globale) e, se vuoi, ritagliali. Poi avvia il training."))
        intro.setWordWrap(True); intro.setStyleSheet("color:#74c0fc; padding:2px 2px 6px;")
        root.addWidget(intro)

        # dataset clip (drag&drop)
        gd = QGroupBox(tr("tclip.dataset", "Dataset — spezzoni video"))
        dv = QVBoxLayout(gd)
        self.clips = ClipDatasetWidget()
        dv.addWidget(self.clips)
        self.global_cap = QLineEdit()
        self.global_cap.setPlaceholderText(tr("tclip.gcap.ph", "caption globale per i clip senza caption (es. mytoken, cinematic, slow motion)"))
        self.global_cap.setToolTip(tr("tclip.gcap.tt",
            "Applicata ai clip che non hanno una caption propria né un file .txt gemello."))
        dv.addWidget(self.global_cap)
        root.addWidget(gd)

        # configurazione (progressive disclosure: gruppo dedicato)
        gc = QGroupBox(tr("tclip.config", "Configurazione"))
        f = QFormLayout(gc); f.setLabelAlignment(Qt.AlignRight)
        self.name = QLineEdit("my_video_lora")
        self.name.setToolTip(tr("tclip.name.tt", "Nome della LoRA (cartella di output)."))
        f.addRow(tr("tclip.name", "Nome LoRA"), self.name)
        self.trigger = QLineEdit()
        self.trigger.setPlaceholderText(tr("tclip.trigger.ph", "es. mytoken (parola-innesco opzionale)"))
        self.trigger.setToolTip(tr("tclip.trigger.tt",
            "Parola-innesco: sarà associata al concetto e va inserita nel prompt per richiamare la LoRA."))
        f.addRow(tr("tclip.trigger", "Trigger"), self.trigger)
        self.steps = QSpinBox(); self.steps.setRange(100, 100000); self.steps.setValue(2000); self.steps.setSingleStep(100)
        self.steps.setToolTip(tr("tclip.steps.tt", "Passi di training. Più passi = più apprendimento (e più tempo)."))
        f.addRow(tr("tclip.steps", "Max steps"), self.steps)
        self.rank = QSpinBox(); self.rank.setRange(4, 128); self.rank.setValue(16)
        self.rank.setToolTip(tr("tclip.rank.tt", "Rango della LoRA: più alto = più capacità (e file più grande)."))
        f.addRow(tr("tclip.rank", "Rank"), self.rank)
        self.res = QComboBox(); self.res.addItems(["768x512", "512x320", "1024x576"])
        self.res.setToolTip(tr("tclip.res.tt", "Risoluzione di training. Più bassa = meno VRAM e più veloce."))
        f.addRow(tr("tclip.res", "Risoluzione"), self.res)
        self.secs = QSpinBox(); self.secs.setRange(1, 10); self.secs.setValue(3); self.secs.setSuffix(" s")
        self.secs.setToolTip(tr("tclip.secs.tt", "Durata dei frammenti usati per il training (numero di frame)."))
        f.addRow(tr("tclip.secs", "Durata clip"), self.secs)
        self.low_vram = QCheckBox(tr("tclip.lowvram", "Low-VRAM")); self.low_vram.setChecked(True)
        self.low_vram.setToolTip(tr("tclip.lowvram.tt", "Config a basso consumo di VRAM (consigliata)."))
        f.addRow("", self.low_vram)
        root.addWidget(gc)

        self.btn = QPushButton(tr("tclip.run", "🎓 Allena LoRA Video"))
        self.btn.setObjectName("generate"); self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.clicked.connect(self._start)
        root.addWidget(self.btn)

        self.status = QLabel(""); self.status.setStyleSheet("color:#8a93a6;")
        root.addWidget(self.status)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(150)
        root.addWidget(self.log)

    def _start(self):
        if self._worker and self._worker.isRunning():
            return
        from core import lora_video_service as svc
        if not svc.trainer_installed():
            QMessageBox.warning(self, tr("tclip.err", "Impossibile allenare"),
                                tr("tclip.notrainer", "Il trainer LTX-2 non è installato.")); return
        clips = self.clips.get_clips()
        if not clips:
            QMessageBox.warning(self, tr("tclip.err", "Impossibile allenare"),
                                tr("tclip.noclips", "Aggiungi almeno uno spezzone video (trascina file o cartelle).")); return
        name = self.name.text().strip() or "my_video_lora"
        if QMessageBox.question(
                self, tr("tclip.confirm.t", "Avviare il training?"),
                tr("tclip.confirm.m", "Alleno la LoRA '{name}' su {n} clip per {s} passi. Può richiedere molto "
                   "tempo. Continuare?").format(name=name, n=len(clips), s=self.steps.value())
                ) != QMessageBox.Yes:
            return
        w, h = (int(x) for x in self.res.currentText().split("x"))
        opts = dict(global_caption=self.global_cap.text().strip(), trigger=self.trigger.text().strip(),
                    rank=self.rank.value(), steps=self.steps.value(), width=w, height=h,
                    clip_seconds=float(self.secs.value()), low_vram=self.low_vram.isChecked())
        self.log.clear(); self.status.setText(tr("tclip.running", "Training in corso…")); self.btn.setEnabled(False)
        self._worker = _ClipTrainWorker(clips, name, opts)
        self._worker.progress.connect(lambda m: self.log.appendPlainText(m))
        self._worker.finished_lora.connect(self._done)
        self._worker.start()

    def _done(self, res: dict):
        self.btn.setEnabled(True)
        if res.get("error"):
            self.status.setText(tr("tclip.failed", "Training fallito"))
            self.log.appendPlainText("ERRORE: " + res["error"]); return
        self.status.setText(tr("tclip.done", "LoRA pronta ✓ ({p})").format(p=res.get("lora", "")))
        self.log.appendPlainText(f"OK: {res.get('lora', '')} ({res.get('clips', 0)} clip)")
        try:                                                 # #97 sidecar
            from core.metadata_saver import MetadataSaver
            MetadataSaver.save_sidecar(res.get("lora", ""), {"label": "LoRA Video", "clips": res.get("clips", 0),
                                       "steps": self.steps.value(), "rank": self.rank.value(),
                                       "trigger": self.trigger.text().strip()})
        except Exception:                                    # noqa: BLE001
            pass
