"""Tab «Allena LoRA» — addestramento di LoRA per FLUX.1 e FLUX.2 con ai-toolkit.

Tutto in-app: selezione dataset, modello base, parametri, avvio/stop, progresso e log.
Il training gira come SUBPROCESS isolato (core/training_service.py); i callback del
processo (eseguiti in un thread secondario) vengono inoltrati alla UI via Qt Signal.

L'installazione del modulo (venv + torch cu128 + ai-toolkit) è lanciata da qui con QProcess.
Al termine, il LoRA finale viene copiato in E:\\AI\\Models\\LoRA\\Flux\\ e segnalato (lora_ready)
così il catalogo lo espone nei tab di generazione.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QObject, Signal, QProcess
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QGridLayout,
    QLabel, QComboBox, QSpinBox, QPushButton, QLineEdit, QCheckBox,
    QFileDialog, QScrollArea, QPlainTextEdit, QProgressBar, QMessageBox,
)

from ui.i18n_manager import tr
from core import training_service as ts


class _Bridge(QObject):
    """Inoltra i callback del thread di training ai segnali Qt (thread-safe)."""
    log = Signal(str)
    progress = Signal(int, int)
    loss = Signal(float)
    done = Signal(int)


class TrainTab(QWidget):
    lora_ready = Signal(str)            # path del LoRA copiato nel catalogo

    def __init__(self):
        super().__init__()
        self._run: ts.TrainingRun | None = None
        self._bridge = _Bridge()
        self._install_proc: QProcess | None = None
        self._caption_proc: QProcess | None = None
        self._bridge.log.connect(self._append_log)
        self._bridge.progress.connect(self._on_progress)
        self._bridge.loss.connect(self._on_loss)
        self._bridge.done.connect(self._on_done)

        root = QVBoxLayout(self)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); form = QVBoxLayout(inner)
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        form.addWidget(self._dataset_box())
        form.addWidget(self._model_box())
        form.addWidget(self._params_box())
        form.addWidget(self._sample_box())
        form.addWidget(self._actions_box())
        form.addWidget(self._progress_box())
        form.addStretch(1)

        self._on_model_changed()
        self._refresh_install_state()

    # ── Sezioni UI ───────────────────────────────────────────────────────────
    def _dataset_box(self) -> QGroupBox:
        box = QGroupBox(tr("train.dataset.title", "Dataset"))
        g = QFormLayout(box)
        row = QHBoxLayout()
        self.ds_path = QLineEdit(); self.ds_path.setPlaceholderText(
            tr("train.dataset.hint", "Cartella con immagini + file .txt di caption omonimi"))
        self.ds_path.textChanged.connect(self._update_ds_info)
        btn = QPushButton(tr("train.browse", "Sfoglia…")); btn.clicked.connect(self._browse_dataset)
        row.addWidget(self.ds_path, 1); row.addWidget(btn)
        w = QWidget(); w.setLayout(row)
        g.addRow(tr("train.dataset.folder", "Cartella"), w)
        self.ds_info = QLabel("—"); self.ds_info.setStyleSheet("color:#9aa1b0;")
        g.addRow("", self.ds_info)
        self.trigger = QLineEdit(); self.trigger.setPlaceholderText(
            tr("train.trigger.hint", "es. ohwx — token attivatore opzionale"))
        g.addRow(tr("train.trigger", "Trigger word"), self.trigger)
        # Auto-captioning (motore selezionabile: JoyCaption consigliato, Florence-2 fallback)
        crow = QHBoxLayout()
        from PySide6.QtWidgets import QComboBox
        from core import caption_service as cs
        self.cap_engine = QComboBox()
        _eng_labels = {"joycaption": tr("train.cap.joy", "JoyCaption"), "florence2": tr("train.cap.flo", "Florence-2")}
        for e in (cs.available_engines() or ["florence2"]):   # solo motori con venv+pesi presenti
            self.cap_engine.addItem(_eng_labels.get(e, e), e)
        self.cap_engine.setToolTip(tr("train.cap.engine.tt",
            "Modello di captioning. JoyCaption = migliore per training (non censurato, tag/descrittivo); "
            "Florence-2 = leggero e veloce."))
        self.cap_mode = QComboBox()
        for m, lab in (("descriptive", tr("train.cap.m.desc", "Descrittiva")),
                       ("tags", tr("train.cap.m.tags", "Tag booru")),
                       ("sd-prompt", tr("train.cap.m.sd", "Prompt SD"))):
            self.cap_mode.addItem(lab, m)
        self.cap_mode.setToolTip(tr("train.cap.mode.tt", "Stile della caption JoyCaption (ignorato da Florence-2)."))
        self.cap_engine.currentIndexChanged.connect(
            lambda: self.cap_mode.setEnabled(self.cap_engine.currentData() == "joycaption"))
        self.cap_mode.setEnabled(self.cap_engine.currentData() == "joycaption")
        self.btn_caption = QPushButton(tr("train.caption", "Genera caption"))
        self.btn_caption.clicked.connect(self._caption_dataset)
        self.cap_overwrite = QCheckBox(tr("train.caption.overwrite", "Sovrascrivi esistenti"))
        crow.addWidget(self.cap_engine); crow.addWidget(self.cap_mode)
        self.btn_review = QPushButton(tr("train.review", "📝 Review caption"))
        self.btn_review.setToolTip(tr("train.review.tt", "Rivedi/sovrascrivi la caption di ogni immagine (finestra non bloccante, auto-salvataggio)."))
        self.btn_review.clicked.connect(self._review_captions)
        crow.addWidget(self.btn_caption); crow.addWidget(self.cap_overwrite)
        crow.addWidget(self.btn_review); crow.addStretch(1)
        cw = QWidget(); cw.setLayout(crow)
        g.addRow(tr("train.caption.row", "Caption automatiche"), cw)
        return box

    def _review_captions(self):
        """Dialog non-modale: rivede le caption .txt omonime delle immagini nella cartella dataset."""
        import os
        from PySide6.QtWidgets import QMessageBox
        folder = self.ds_path.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.information(self, tr("train.review", "📝 Review caption"),
                                    tr("train.review.nofolder", "Seleziona prima una cartella dataset valida.")); return
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        imgs = sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts)
        if not imgs:
            QMessageBox.information(self, tr("train.review", "📝 Review caption"),
                                    tr("train.review.noimg", "Nessuna immagine nella cartella dataset.")); return
        items = []
        for f in imgs:
            p = os.path.join(folder, f)
            txt = os.path.splitext(p)[0] + ".txt"
            cap = open(txt, encoding="utf-8").read().strip() if os.path.exists(txt) else ""
            items.append({"path": p, "caption": cap})

        def _save(_i, path, cap):                            # scrive la caption nel .txt omonimo
            open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8").write(cap)
        from ui.widgets.caption_review import CaptionReviewDialog
        self._cap_dialog = CaptionReviewDialog(items, _save, is_video=False, parent=self)
        self._cap_dialog.show()

    def _model_box(self) -> QGroupBox:
        box = QGroupBox(tr("train.model.title", "Modello base"))
        g = QFormLayout(box)
        self.model = QComboBox()
        for spec in ts.BASE_MODELS.values():
            self.model.addItem(spec.label, spec.key)
        self.model.currentIndexChanged.connect(self._on_model_changed)
        g.addRow(tr("train.model.base", "Architettura"), self.model)
        self.model_path = QLineEdit()
        g.addRow(tr("train.model.path", "Repo HF / cartella"), self.model_path)
        self.model_note = QLabel(""); self.model_note.setWordWrap(True)
        self.model_note.setStyleSheet("color:#c9a227;")
        g.addRow("", self.model_note)
        self.hf_token = QLineEdit(); self.hf_token.setEchoMode(QLineEdit.Password)
        self.hf_token.setPlaceholderText(tr("train.hf.hint", "necessario per i modelli gated (FLUX.1/2-dev)"))
        g.addRow(tr("train.hf.token", "HF token"), self.hf_token)
        return box

    def _params_box(self) -> QGroupBox:
        box = QGroupBox(tr("train.params.title", "Parametri"))
        g = QGridLayout(box)
        self.rank = QSpinBox(); self.rank.setRange(2, 256); self.rank.setValue(16)
        self.alpha = QSpinBox(); self.alpha.setRange(1, 256); self.alpha.setValue(16)
        self.steps = QSpinBox(); self.steps.setRange(100, 100000); self.steps.setSingleStep(100); self.steps.setValue(2000)
        self.batch = QSpinBox(); self.batch.setRange(1, 16); self.batch.setValue(1)
        self.save_every = QSpinBox(); self.save_every.setRange(50, 10000); self.save_every.setSingleStep(50); self.save_every.setValue(250)
        self.sample_every = QSpinBox(); self.sample_every.setRange(50, 10000); self.sample_every.setSingleStep(50); self.sample_every.setValue(250)
        self.lr = QComboBox(); self.lr.setEditable(True)
        self.lr.addItems(["1e-4", "2e-4", "5e-5", "1e-3", "8e-5"]); self.lr.setCurrentText("1e-4")
        self.seed = QSpinBox(); self.seed.setRange(0, 2_000_000_000); self.seed.setValue(42)

        g.addWidget(QLabel(tr("train.p.rank", "Rank (dim)")), 0, 0); g.addWidget(self.rank, 0, 1)
        g.addWidget(QLabel(tr("train.p.alpha", "Alpha")), 0, 2); g.addWidget(self.alpha, 0, 3)
        g.addWidget(QLabel(tr("train.p.steps", "Step")), 1, 0); g.addWidget(self.steps, 1, 1)
        g.addWidget(QLabel(tr("train.p.lr", "Learning rate")), 1, 2); g.addWidget(self.lr, 1, 3)
        g.addWidget(QLabel(tr("train.p.batch", "Batch size")), 2, 0); g.addWidget(self.batch, 2, 1)
        g.addWidget(QLabel(tr("train.p.seed", "Seed")), 2, 2); g.addWidget(self.seed, 2, 3)
        g.addWidget(QLabel(tr("train.p.save", "Salva ogni")), 3, 0); g.addWidget(self.save_every, 3, 1)
        g.addWidget(QLabel(tr("train.p.sample", "Anteprima ogni")), 3, 2); g.addWidget(self.sample_every, 3, 3)

        # risoluzioni (multi-scala)
        resw = QHBoxLayout()
        self.res = {}
        for r in (512, 768, 1024):
            cb = QCheckBox(str(r)); cb.setChecked(True); self.res[r] = cb; resw.addWidget(cb)
        resbox = QWidget(); resbox.setLayout(resw)
        g.addWidget(QLabel(tr("train.p.res", "Risoluzioni")), 4, 0); g.addWidget(resbox, 4, 1, 1, 3)

        self.quantize = QCheckBox(tr("train.p.quantize", "Quantizza (meno VRAM)")); self.quantize.setChecked(True)
        self.low_vram = QCheckBox(tr("train.p.lowvram", "Low VRAM (offload CPU)"))
        self.train_te = QCheckBox(tr("train.p.train_te", "Allena anche il text encoder"))
        g.addWidget(self.quantize, 5, 0, 1, 2); g.addWidget(self.low_vram, 5, 2, 1, 2)
        g.addWidget(self.train_te, 6, 0, 1, 2)
        return box

    def _sample_box(self) -> QGroupBox:
        box = QGroupBox(tr("train.sample.title", "Prompt di anteprima (uno per riga)"))
        v = QVBoxLayout(box)
        self.prompts = QPlainTextEdit()
        self.prompts.setPlainText("\n".join(ts.DEFAULT_SAMPLE_PROMPTS))
        self.prompts.setFixedHeight(90)
        v.addWidget(self.prompts)
        return box

    def _actions_box(self) -> QGroupBox:
        box = QGroupBox(tr("train.actions.title", "Azioni"))
        h = QHBoxLayout(box)
        self.btn_install = QPushButton(tr("train.install", "Installa modulo training"))
        self.btn_install.clicked.connect(self._install_module)
        self.btn_start = QPushButton(tr("train.start", "Avvia training"))
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QPushButton(tr("train.stop", "Stop")); self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        h.addWidget(self.btn_install); h.addStretch(1)
        h.addWidget(self.btn_start); h.addWidget(self.btn_stop)
        return box

    def _progress_box(self) -> QGroupBox:
        box = QGroupBox(tr("train.progress.title", "Progresso"))
        v = QVBoxLayout(box)
        self.bar = QProgressBar(); self.bar.setRange(0, 100)
        self.status = QLabel("—"); self.status.setStyleSheet("color:#9aa1b0;")
        self.logview = QPlainTextEdit(); self.logview.setReadOnly(True); self.logview.setFixedHeight(180)
        self.logview.setStyleSheet("font-family:Consolas,monospace;font-size:11px;")
        v.addWidget(self.bar); v.addWidget(self.status); v.addWidget(self.logview)
        return box

    # ── Slot / logica ─────────────────────────────────────────────────────────
    def _browse_dataset(self):
        d = QFileDialog.getExistingDirectory(self, tr("train.dataset.pick", "Scegli la cartella dataset"))
        if d:
            self.ds_path.setText(d)

    def _update_ds_info(self):
        d = Path(self.ds_path.text().strip() or "")
        if not d.is_dir():
            self.ds_info.setText(tr("train.ds.missing", "Cartella non valida")); return
        imgs = [f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
        caps = [f for f in d.iterdir() if f.suffix.lower() == ".txt"]
        self.ds_info.setText(
            tr("train.ds.count", "{n} immagini · {c} caption").format(n=len(imgs), c=len(caps)))

    def _on_model_changed(self):
        spec = ts.BASE_MODELS[self.model.currentData()]
        self.model_path.setText(spec.default_path)
        self.model_note.setText("⚠ " + spec.note)
        # default sensati: FLUX.2 dev → suggerisci low_vram
        if spec.key == "flux2-dev":
            self.low_vram.setChecked(True)

    def _collect_params(self) -> ts.TrainParams:
        res = [r for r, cb in self.res.items() if cb.isChecked()] or [1024]
        try:
            lr = float(self.lr.currentText().strip())
        except ValueError:
            lr = 1e-4
        prompts = [l.strip() for l in self.prompts.toPlainText().splitlines() if l.strip()]
        import time as _time                                # #95: suffisso timestamp → niente collisioni/sovrascritture tra run
        return ts.TrainParams(
            name=((self.trigger.text().strip() or "lora") + "_"
                  + Path(self.ds_path.text().strip() or "dataset").name + "_" + _time.strftime("%m%d_%H%M%S")),
            dataset_path=self.ds_path.text().strip(),
            base_model=self.model.currentData(),
            model_path=self.model_path.text().strip(),
            rank=self.rank.value(), alpha=self.alpha.value(),
            lr=lr, steps=self.steps.value(),
            resolutions=res, batch_size=self.batch.value(),
            save_every=self.save_every.value(), sample_every=self.sample_every.value(),
            quantize=self.quantize.isChecked(), low_vram=self.low_vram.isChecked(),
            train_text_encoder=self.train_te.isChecked(),
            trigger_word=self.trigger.text().strip(),
            sample_prompts=prompts or list(ts.DEFAULT_SAMPLE_PROMPTS),
            seed=self.seed.value(),
        )

    def _refresh_install_state(self):
        ok = ts.is_installed()
        self.btn_install.setEnabled(not ok)
        self.btn_install.setText(
            tr("train.installed", "Modulo training installato ✓") if ok
            else tr("train.install", "Installa modulo training"))
        self.btn_start.setEnabled(ok)

    # ── Installazione modulo ───────────────────────────────────────────────────
    def _install_module(self):
        if self._install_proc is not None:
            return
        if QMessageBox.question(
            self, tr("train.install.title", "Installa modulo training"),
            tr("train.install.confirm",
               "Verranno scaricati ai-toolkit + PyTorch (cu128) in un ambiente isolato "
               "(diversi GB). Procedo?")) != QMessageBox.Yes:
            return
        script = str(Path(__file__).resolve().parents[2] / "scripts" / "install_training.py")
        self._append_log("[install] avvio installazione…")
        self.btn_install.setEnabled(False)
        p = QProcess(self)
        p.setProcessChannelMode(QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(
            lambda: self._append_log(bytes(p.readAllStandardOutput()).decode("utf-8", "replace").rstrip()))
        p.finished.connect(self._install_finished)
        self._install_proc = p
        p.start(sys.executable, ["-u", script])

    def _install_finished(self, code, _status):
        self._append_log(f"[install] terminato (codice {code}).")
        self._install_proc = None
        self._refresh_install_state()

    # ── Auto-captioning (Florence-2 nel venv di training) ───────────────────────
    def _caption_dataset(self):
        if self._caption_proc is not None:
            return
        ds = Path(self.ds_path.text().strip() or "")
        if not ds.is_dir():
            QMessageBox.warning(self, tr("train.caption", "Genera caption"),
                                tr("train.ds.missing", "Cartella non valida")); return
        from core import caption_service as cs
        engine = self.cap_engine.currentData() or "florence2"
        mode = self.cap_mode.currentData() or "descriptive"
        py, cmd = cs.python_and_args(str(ds), engine=engine, mode=mode,
                                     trigger=self.trigger.text().strip(), overwrite=self.cap_overwrite.isChecked())
        if py is None:                                       # venv del motore mancante
            msg = (tr("train.caption.needjoy", "JoyCaption non pronto: manca il venv15 o i pesi (E:\\AI\\Models\\JoyCaption).")
                   if engine == "joycaption"
                   else tr("train.caption.needinstall", "Installa prima il modulo training (serve il suo ambiente)."))
            QMessageBox.warning(self, tr("train.caption", "Genera caption"), msg); return
        self._append_log(f"[caption] avvio ({engine}/{mode})…")
        self.btn_caption.setEnabled(False)
        p = QProcess(self)
        p.setProcessChannelMode(QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(
            lambda: self._append_log(bytes(p.readAllStandardOutput()).decode("utf-8", "replace").rstrip()))
        p.finished.connect(self._caption_finished)
        self._caption_proc = p
        p.start(py, cmd)

    def _caption_finished(self, code, _status):
        self._append_log(f"[caption] terminato (codice {code}).")
        self._caption_proc = None
        self.btn_caption.setEnabled(True)
        self._update_ds_info()

    # ── Training ────────────────────────────────────────────────────────────────
    def _start(self):
        p = self._collect_params()
        errs = ts.preflight(p)
        if errs:
            QMessageBox.warning(self, tr("train.preflight.title", "Controlli falliti"), "\n".join(errs))
            return
        spec = p.spec()
        if spec.key == "flux2-dev" and QMessageBox.question(
            self, tr("train.heavy.title", "Download pesante"),
            tr("train.heavy.msg",
               "FLUX.2 [dev] scarica ~110GB di pesi e l'addestramento è molto pesante. "
               "Consigliato FLUX.2 Klein per risultati più rapidi. Procedo comunque?")) != QMessageBox.Yes:
            return
        self.logview.clear()
        self.bar.setValue(0)
        self._run = ts.TrainingRun(
            p,
            on_log=self._bridge.log.emit,
            on_progress=self._bridge.progress.emit,
            on_loss=self._bridge.loss.emit,
            on_done=self._bridge.done.emit,
            hf_token=self.hf_token.text().strip() or None,
        )
        try:
            cfg = self._run.start()
        except Exception as e:                          # noqa: BLE001
            QMessageBox.critical(self, tr("train.error", "Errore"), str(e)); self._run = None; return
        self._append_log(f"[train] config: {cfg}")
        self._append_log(f"[train] avviato: {spec.label} · {p.steps} step")
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self.status.setText(tr("train.running", "In addestramento…"))

    def _stop(self):
        if self._run:
            self._run.stop()
            self._append_log("[train] interruzione richiesta…")
            self.btn_stop.setEnabled(False)

    def _append_log(self, line: str):
        if not line:
            return
        self.logview.appendPlainText(line)
        sb = self.logview.verticalScrollBar(); sb.setValue(sb.maximum())

    def _on_progress(self, step: int, total: int):
        self.bar.setRange(0, total); self.bar.setValue(step)
        self.status.setText(tr("train.step", "Step {s}/{t}").format(s=step, t=total))

    def _on_loss(self, loss: float):
        cur = self.status.text().split("  |")[0]
        self.status.setText(f"{cur}  |  loss {loss:.4f}")

    def _on_done(self, code: int):
        self.btn_start.setEnabled(ts.is_installed()); self.btn_stop.setEnabled(False)
        if code != 0:
            self.status.setText(tr("train.failed", "Training terminato con errori (codice {c})").format(c=code))
            self._append_log(f"[train] FINE con codice {code}")
            self._run = None
            return
        self.status.setText(tr("train.complete", "Training completato ✓"))
        # copia il LoRA finale nel catalogo
        try:
            src = ts.find_final_lora(self._run.params) if self._run else None
            if src and src.exists():
                ts.LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                dest = ts.LORA_OUTPUT_DIR / src.name
                shutil.copy2(src, dest)
                self._append_log(f"[train] LoRA copiata in: {dest}")
                self.lora_ready.emit(str(dest))
            else:
                self._append_log("[train] LoRA finale non trovata nella cartella output.")
        except Exception as e:                          # noqa: BLE001
            self._append_log(f"[train] copia LoRA fallita: {e}")
        self._run = None
