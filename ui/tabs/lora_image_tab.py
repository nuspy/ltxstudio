"""Tab «LoRA Foto (FLUX.2)» — addestra LoRA IMMAGINI con dataset auto-generato.

Riusa l'editor soggetti del LoRA Video (stessa logica: tipi, foto multiple con angolo, copertura
best-practice, situazioni custom, materiale reale, focus-volto) ma genera IMMAGINI (FLUX.1+PuLID per
i volti, Kontext per oggetti/luoghi) e allena via ai-toolkit (FLUX.2). Zero ComfyUI.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from core import lora_coverage
from ui.i18n_manager import tr
from ui.tabs.lora_video_tab import LoraVideoTab

_BASES = [("flux2-dev", "FLUX.2 [dev]"), ("flux2-klein-4b", "FLUX.2 Klein 4B"),
          ("flux2-klein-9b", "FLUX.2 Klein 9B"), ("flux1-dev", "FLUX.1 [dev]"),
          ("krea2", "Krea 2 [Raw]")]


class _LoraImageWorker(QThread):
    progress = Signal(str)
    finished_lora = Signal(dict)

    def __init__(self, subjects, name, out_dir, opts: dict, full: bool):
        super().__init__()
        self._a = (subjects, name, out_dir, opts, full)

    def run(self):
        subjects, name, out_dir, opts, full = self._a
        try:
            from core import lora_image_service as svc
            if full:
                res = svc.make_lora(subjects, name, out_dir, progress=self.progress.emit, **opts)
            else:
                import os
                ds = os.path.join(out_dir, name, "dataset")
                svc.generate_dataset(subjects, ds, width=opts.get("width", 1024),
                                     height=opts.get("height", 1024),
                                     caption_engine=opts.get("caption_engine", "procedural"),
                                     caption_mode=opts.get("caption_mode", "descriptive"),
                                     coherence_engine=opts.get("coherence_engine", "flux"),
                                     progress=self.progress.emit)
                res = {"dataset": ds}
            self.finished_lora.emit(res)
        except Exception as e:                          # noqa: BLE001
            self.finished_lora.emit({"error": str(e)})


class LoraImageTab(LoraVideoTab):
    IMAGE_MODE = True
    _CAP = "cap.lora_image_train"                    # gating feature_matrix (override della base video)

    def _default_cases(self, stype: str) -> list:
        return lora_coverage.default_case_ids(stype, image=True)   # esclude i casi solo-video

    # impostazioni specifiche FLUX.2 (sostituisce il box video)
    def _train_box(self) -> QGroupBox:
        box = QGroupBox(tr("ilora.train", "Addestramento LoRA (FLUX.2)"))
        v = QVBoxLayout(box)
        g = QGridLayout()
        self.lora_name = QLineEdit("my_flux_lora")
        self.base_model = QComboBox()
        for code, label in _BASES:
            self.base_model.addItem(label, code)
        self.lora_rank = QSpinBox(); self.lora_rank.setRange(4, 128); self.lora_rank.setValue(16)
        self.lora_steps = QSpinBox(); self.lora_steps.setRange(200, 20000); self.lora_steps.setValue(2000); self.lora_steps.setSingleStep(200)
        self.img_res = QComboBox(); self.img_res.addItems(["768", "1024", "1280"]); self.img_res.setCurrentText("1024")
        # Captioner del dataset: procedurale (trigger+tag) o VLM (JoyCaption consigliato / Florence-2)
        from core import caption_service as cs
        self.cap_engine = QComboBox()
        self.cap_engine.addItem(tr("ilora.cap.proc", "Procedurale (trigger+tag)"), "procedural")
        _lbl = {"joycaption": tr("train.cap.joy", "JoyCaption"), "florence2": tr("train.cap.flo", "Florence-2")}
        for e in cs.available_engines():
            self.cap_engine.addItem(_lbl.get(e, e), e)
        self.cap_engine.setToolTip(tr("ilora.cap.tt",
            "Come descrivere le immagini del dataset. JoyCaption = caption VLM realistiche (migliori per il "
            "training, non censurato); Procedurale = trigger + tag generati (nessun modello)."))
        self.cap_mode = QComboBox()
        for m, lab in (("descriptive", tr("train.cap.m.desc", "Descrittiva")),
                       ("tags", tr("train.cap.m.tags", "Tag booru")), ("sd-prompt", tr("train.cap.m.sd", "Prompt SD"))):
            self.cap_mode.addItem(lab, m)
        # coerenza volto per gli scatti CON foto di riferimento (soggetto): FLUX (PuLID/Kontext) o Krea 2 Edit
        # (in-context, identità più fedele nel dataset). NON sostituisce FLUX.
        self.coherence = QComboBox()
        self.coherence.addItem(tr("ilora.coh_flux", "FLUX (PuLID/Kontext)"), "flux")
        self.coherence.addItem(tr("ilora.coh_krea2edit", "Krea 2 Edit (coerenza volto)"), "krea2_edit")
        self.coherence.setToolTip(tr("ilora.coh.tt",
            "Motore per la coerenza del soggetto negli scatti con foto: Krea 2 Edit preserva meglio l'identità."))
        g.addWidget(QLabel(tr("tlora.lora_name", "Nome LoRA")), 0, 0); g.addWidget(self.lora_name, 0, 1)
        g.addWidget(QLabel(tr("ilora.base", "Modello base")), 0, 2); g.addWidget(self.base_model, 0, 3)
        g.addWidget(QLabel(tr("tlora.rank", "Rank")), 1, 0); g.addWidget(self.lora_rank, 1, 1)
        g.addWidget(QLabel(tr("tlora.steps", "Step")), 1, 2); g.addWidget(self.lora_steps, 1, 3)
        g.addWidget(QLabel(tr("ilora.res", "Risoluzione")), 2, 0); g.addWidget(self.img_res, 2, 1)
        g.addWidget(QLabel(tr("ilora.cap", "Captioner")), 2, 2)
        _cw = QHBoxLayout(); _cw.setContentsMargins(0, 0, 0, 0); _cw.addWidget(self.cap_engine, 1); _cw.addWidget(self.cap_mode)
        _cwrap = QWidget(); _cwrap.setLayout(_cw); g.addWidget(_cwrap, 2, 3)
        g.addWidget(QLabel(tr("ilora.coh_lbl", "Coerenza volto")), 3, 0); g.addWidget(self.coherence, 3, 1)
        v.addLayout(g)
        h = QHBoxLayout()
        self.btn_install = QPushButton(tr("ilora.install", "Installa ai-toolkit")); self.btn_install.clicked.connect(self._install)
        self.btn_dataset = QPushButton(tr("tlora.gen_dataset", "🎬 Genera dataset")); self.btn_dataset.clicked.connect(lambda: self._start(False))
        self.btn_full = QPushButton(tr("ilora.gen_train", "🎓 Genera + Allena LoRA FLUX.2")); self.btn_full.clicked.connect(lambda: self._start(True))
        h.addWidget(self.btn_install); h.addStretch(1); h.addWidget(self.btn_dataset); h.addWidget(self.btn_full)
        v.addLayout(h)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(110)
        v.addWidget(self.log)
        self._build_sample_preview(v)                    # sample del training visibili anche qui
        return box

    def _install(self):
        import subprocess
        import sys
        self._log("Installazione ai-toolkit…")
        try:
            subprocess.Popen([sys.executable, str(Path(__file__).resolve().parents[2] / "scripts" / "install_training.py")])
        except Exception as e:                          # noqa: BLE001
            QMessageBox.critical(self, tr("ilora.train", "LoRA"), str(e))

    def _start(self, full: bool):
        if self._worker and self._worker.isRunning():
            return
        self._save_cur()
        subjects = self._to_subjects()
        if not subjects:
            QMessageBox.warning(self, tr("ilora.train", "LoRA"),
                               tr("tlora.need_subject", "Aggiungi almeno un soggetto con un nome.")); return
        if not any(s.all_photos() or s.real_media for s in subjects):
            QMessageBox.warning(self, tr("ilora.train", "LoRA"),
                               tr("tlora.need_media", "Aggiungi almeno una foto o un'immagine reale per i soggetti.")); return
        import os
        out_dir = str(Path(__file__).resolve().parents[2] / "output" / "loras")
        os.makedirs(out_dir, exist_ok=True)
        res = int(self.img_res.currentText())
        opts = {"base": self.base_model.currentData(), "rank": self.lora_rank.value(),
                "steps": self.lora_steps.value(), "width": res, "height": res,
                "caption_engine": self.cap_engine.currentData() or "procedural",
                "caption_mode": self.cap_mode.currentData() or "descriptive",
                "coherence_engine": self.coherence.currentData() or "flux"}
        self.btn_dataset.setEnabled(False); self.btn_full.setEnabled(False)
        self._log(tr("ilora.started", "Avvio LoRA FLUX.2 (può richiedere molto tempo)…"))
        _name = self.lora_name.text().strip() or "my_flux_lora"
        self._worker = _LoraImageWorker(subjects, _name, out_dir, opts, full)
        self._worker.progress.connect(self._log)
        self._worker.finished_lora.connect(self._done)
        if full:
            self._watch_samples(out_dir, _name)          # sample preview durante il training
        self._worker.start()
