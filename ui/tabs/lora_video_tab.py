"""Tab «LoRA Video (LTX-2.3)» — addestra LoRA video con dataset auto-generato.

Flusso: elenca SOGGETTI (personaggio/oggetto/luogo/concetto) con nome=trigger, foto multiple
(con angolo), copertura automatica best-practice (espressioni/viste/luci…), situazioni custom
illimitate e materiale reale opzionale → genera il dataset video (PuLID per i volti, focus-volto,
rotazione multi-keyframe) → preprocess + training col trainer ufficiale LTX-2. Zero ComfyUI.
"""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QDoubleSpinBox,
                               QVBoxLayout, QWidget)

from core import lora_coverage
from ui.i18n_manager import tr

_TYPE_LABELS = [("person", "tlora.type.person"), ("place", "tlora.type.place"),
                ("object", "tlora.type.object"), ("concept", "tlora.type.concept")]
_ANGLES = ["front", "three_quarter", "profile", "back", "any"]


class _LoraWorker(QThread):
    progress = Signal(str)
    finished_lora = Signal(dict)

    def __init__(self, subjects, name, out_dir, opts: dict, full: bool):
        super().__init__()
        self._a = (subjects, name, out_dir, opts, full)

    def run(self):
        subjects, name, out_dir, opts, full = self._a
        try:
            from core import lora_video_service as svc
            if full:
                res = svc.make_lora(subjects, name, out_dir, progress=self.progress.emit, **opts)
            else:
                import os
                ds = os.path.join(out_dir, name, "dataset")
                svc.generate_dataset(subjects, ds, clip_seconds=opts.get("clip_seconds", 3.0),
                                     use_pulid=opts.get("use_pulid", True),
                                     caption_engine=opts.get("caption_engine", "procedural"),
                                     caption_mode=opts.get("caption_mode", "descriptive"),
                                     progress=self.progress.emit)
                res = {"dataset": ds}
            self.finished_lora.emit(res)
        except Exception as e:                          # noqa: BLE001
            self.finished_lora.emit({"error": str(e)})


class LoraVideoTab(QWidget):
    def __init__(self):
        super().__init__()
        self._subjects: list[dict] = []
        self._cur = -1
        self._worker = None
        root = QVBoxLayout(self)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); form = QVBoxLayout(inner); scroll.setWidget(inner)
        root.addWidget(scroll, 1)
        form.addWidget(self._subjects_box())
        form.addWidget(self._editor_box())
        form.addWidget(self._train_box())
        form.addStretch(1)
        self._refresh_editor()

    # ── elenco soggetti ──
    def _subjects_box(self) -> QGroupBox:
        box = QGroupBox(tr("tlora.subjects", "Soggetti della LoRA"))
        v = QVBoxLayout(box)
        self.subj_list = QListWidget()
        self.subj_list.currentRowChanged.connect(self._on_select)
        v.addWidget(self.subj_list)
        h = QHBoxLayout()
        b_add = QPushButton(tr("tlora.add", "➕ Aggiungi soggetto")); b_add.clicked.connect(self._add_subject)
        b_del = QPushButton(tr("tlora.del", "🗑 Rimuovi")); b_del.clicked.connect(self._del_subject)
        h.addWidget(b_add); h.addWidget(b_del); h.addStretch(1)
        v.addLayout(h)
        return box

    # ── editor del soggetto selezionato ──
    def _editor_box(self) -> QGroupBox:
        box = QGroupBox(tr("tlora.editor", "Dettagli soggetto"))
        v = QVBoxLayout(box)
        f = QFormLayout()
        self.ed_type = QComboBox()
        for code, key in _TYPE_LABELS:
            self.ed_type.addItem(tr(key, code), code)
        self.ed_type.currentIndexChanged.connect(self._on_type_change)
        self.ed_name = QLineEdit(); self.ed_name.setPlaceholderText(tr("tlora.name_ph", "Nome = trigger/tag (es. Martina)"))
        self.ed_name.textChanged.connect(self._save_cur)
        self.ed_face = QCheckBox(tr("tlora.face_focus", "Focus sul volto (impara il viso, non abiti/sfondo)"))
        self.ed_face.setChecked(True); self.ed_face.toggled.connect(self._save_cur)
        f.addRow(tr("tlora.type", "Tipo"), self.ed_type)
        f.addRow(tr("tlora.name", "Nome/Trigger"), self.ed_name)
        f.addRow("", self.ed_face)
        v.addLayout(f)
        # foto multiple con angolo
        v.addWidget(QLabel(tr("tlora.photos", "Foto di riferimento (più angoli = video più coerenti)")))
        self.photo_list = QListWidget(); self.photo_list.setMaximumHeight(90)
        v.addWidget(self.photo_list)
        hp = QHBoxLayout()
        self.photo_angle = QComboBox(); self.photo_angle.addItems(_ANGLES)
        b_ap = QPushButton(tr("tlora.add_photo", "Aggiungi foto…")); b_ap.clicked.connect(self._add_photo)
        b_rp = QPushButton(tr("tlora.del_photo", "Rimuovi foto")); b_rp.clicked.connect(self._del_photo)
        hp.addWidget(QLabel(tr("tlora.angle", "Angolo"))); hp.addWidget(self.photo_angle); hp.addWidget(b_ap); hp.addWidget(b_rp); hp.addStretch(1)
        v.addLayout(hp)
        # copertura automatica (caselle dinamiche per tipo)
        v.addWidget(QLabel(tr("tlora.coverage", "Copertura automatica (casi da generare)")))
        self.cov_host = QWidget(); self.cov_lay = QGridLayout(self.cov_host); self.cov_lay.setContentsMargins(0, 0, 0, 0)
        self._cov_checks: dict[str, QCheckBox] = {}
        v.addWidget(self.cov_host)
        # situazioni custom illimitate
        v.addWidget(QLabel(tr("tlora.customs", "Situazioni custom (una per riga, illimitate)")))
        self.ed_customs = QPlainTextEdit(); self.ed_customs.setFixedHeight(80)
        self.ed_customs.setPlaceholderText(tr("tlora.customs_ph", "es. corre sulla spiaggia\nballa felice\nbeve un caffè"))
        self.ed_customs.textChanged.connect(self._save_cur)
        v.addWidget(self.ed_customs)
        # materiale reale opzionale
        v.addWidget(QLabel(tr("tlora.real", "Materiale reale opzionale (video/foto del soggetto)")))
        self.real_list = QListWidget(); self.real_list.setMaximumHeight(70)
        v.addWidget(self.real_list)
        hr = QHBoxLayout()
        b_ar = QPushButton(tr("tlora.add_real", "Aggiungi reale…")); b_ar.clicked.connect(self._add_real)
        b_rr = QPushButton(tr("tlora.del_real", "Rimuovi")); b_rr.clicked.connect(self._del_real)
        hr.addWidget(b_ar); hr.addWidget(b_rr); hr.addStretch(1)
        v.addLayout(hr)
        return box

    # ── impostazioni LoRA + azioni ──
    def _train_box(self) -> QGroupBox:
        box = QGroupBox(tr("tlora.train", "Addestramento LoRA"))
        v = QVBoxLayout(box)
        g = QGridLayout()
        self.lora_name = QLineEdit("my_lora")
        self.lora_rank = QSpinBox(); self.lora_rank.setRange(4, 128); self.lora_rank.setValue(16)
        self.lora_steps = QSpinBox(); self.lora_steps.setRange(200, 20000); self.lora_steps.setValue(2000); self.lora_steps.setSingleStep(200)
        self.lora_clip = QDoubleSpinBox(); self.lora_clip.setRange(2.0, 6.0); self.lora_clip.setSingleStep(0.5); self.lora_clip.setValue(3.0); self.lora_clip.setSuffix(" s")
        self.lora_pulid = QCheckBox(tr("tlora.pulid", "PuLID (somiglianza volti forte)")); self.lora_pulid.setChecked(True)
        self.lora_lowvram = QCheckBox(tr("tlora.lowvram", "Low VRAM")); self.lora_lowvram.setChecked(True)
        g.addWidget(QLabel(tr("tlora.lora_name", "Nome LoRA")), 0, 0); g.addWidget(self.lora_name, 0, 1)
        g.addWidget(QLabel(tr("tlora.rank", "Rank")), 0, 2); g.addWidget(self.lora_rank, 0, 3)
        g.addWidget(QLabel(tr("tlora.steps", "Step")), 1, 0); g.addWidget(self.lora_steps, 1, 1)
        g.addWidget(QLabel(tr("tlora.cliplen", "Durata clip")), 1, 2); g.addWidget(self.lora_clip, 1, 3)
        g.addWidget(self.lora_pulid, 2, 0, 1, 2); g.addWidget(self.lora_lowvram, 2, 2, 1, 2)
        from ui.widgets.caption_picker import CaptionPicker
        self.cap_picker = CaptionPicker()                    # captioner VLM (frame/clip) opzionale
        g.addWidget(QLabel(tr("ilora.cap", "Captioner")), 3, 0); g.addWidget(self.cap_picker, 3, 1, 1, 3)
        v.addLayout(g)
        h = QHBoxLayout()
        self.btn_install = QPushButton(tr("tlora.install", "Installa trainer")); self.btn_install.clicked.connect(self._install)
        self.btn_dataset = QPushButton(tr("tlora.gen_dataset", "🎬 Genera dataset")); self.btn_dataset.clicked.connect(lambda: self._start(False))
        self.btn_full = QPushButton(tr("tlora.gen_train", "🎓 Genera + Allena LoRA")); self.btn_full.clicked.connect(lambda: self._start(True))
        h.addWidget(self.btn_install); h.addStretch(1); h.addWidget(self.btn_dataset); h.addWidget(self.btn_full)
        v.addLayout(h)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(110)
        v.addWidget(self.log)
        return box

    # ── stato/selezione ──
    def _default_cases(self, stype: str) -> list:
        return lora_coverage.default_case_ids(stype)   # video: tutti i casi; sovrascritto in immagini

    def _blank(self) -> dict:
        return {"name": "", "type": "person", "photos": [], "case_ids": self._default_cases("person"),
                "customs": "", "real_media": [], "face_focus": True}

    def _add_subject(self):
        self._subjects.append(self._blank())
        self.subj_list.addItem("(nuovo)")
        self.subj_list.setCurrentRow(len(self._subjects) - 1)

    def _del_subject(self):
        if self._cur < 0:
            return
        self._subjects.pop(self._cur)
        self.subj_list.takeItem(self._cur)
        self._cur = self.subj_list.currentRow()
        self._refresh_editor()

    def _on_select(self, row: int):
        self._cur = row
        self._refresh_editor()

    def _rebuild_coverage(self, stype: str, active: list[str]):
        for i in reversed(range(self.cov_lay.count())):
            w = self.cov_lay.itemAt(i).widget()
            if w:
                w.setParent(None)
        self._cov_checks = {}
        cases = lora_coverage.cases_for(stype)
        for idx, (cid, _t, _tags) in enumerate(cases):
            cb = QCheckBox(cid.replace("_", " "))
            cb.setChecked(cid in active)
            cb.toggled.connect(self._save_cur)
            self._cov_checks[cid] = cb
            self.cov_lay.addWidget(cb, idx // 3, idx % 3)

    def _refresh_editor(self):
        en = 0 <= self._cur < len(self._subjects)
        for w in (self.ed_type, self.ed_name, self.ed_face, self.ed_customs):
            w.setEnabled(en)
        if not en:
            self._rebuild_coverage("person", [])
            return
        s = self._subjects[self._cur]
        self.ed_type.blockSignals(True); self.ed_name.blockSignals(True); self.ed_customs.blockSignals(True); self.ed_face.blockSignals(True)
        self.ed_type.setCurrentIndex([c for c, _ in _TYPE_LABELS].index(s["type"]))
        self.ed_name.setText(s["name"]); self.ed_customs.setPlainText(s["customs"]); self.ed_face.setChecked(s["face_focus"])
        self.ed_type.blockSignals(False); self.ed_name.blockSignals(False); self.ed_customs.blockSignals(False); self.ed_face.blockSignals(False)
        self.photo_list.clear()
        for p in s["photos"]:
            self.photo_list.addItem(f"[{p['angle']}] {p['path']}")
        self.real_list.clear()
        for r in s["real_media"]:
            self.real_list.addItem(r)
        self._rebuild_coverage(s["type"], s["case_ids"])

    def _on_type_change(self):
        if not (0 <= self._cur < len(self._subjects)):
            return
        s = self._subjects[self._cur]
        s["type"] = self.ed_type.currentData()
        s["case_ids"] = self._default_cases(s["type"])
        self._rebuild_coverage(s["type"], s["case_ids"])
        self._save_cur()

    def _save_cur(self):
        if not (0 <= self._cur < len(self._subjects)):
            return
        s = self._subjects[self._cur]
        s["name"] = self.ed_name.text().strip()
        s["type"] = self.ed_type.currentData()
        s["face_focus"] = self.ed_face.isChecked()
        s["customs"] = self.ed_customs.toPlainText()
        s["case_ids"] = [cid for cid, cb in self._cov_checks.items() if cb.isChecked()]
        self.subj_list.item(self._cur).setText(f"[{s['type']}] {s['name'] or '(senza nome)'}")

    def _add_photo(self):
        if not (0 <= self._cur < len(self._subjects)):
            return
        p, _ = QFileDialog.getOpenFileName(self, tr("tlora.add_photo", "Aggiungi foto"), "", "Immagini (*.png *.jpg *.jpeg *.webp)")
        if p:
            self._subjects[self._cur]["photos"].append({"path": p, "angle": self.photo_angle.currentText()})
            self._refresh_editor()

    def _del_photo(self):
        r = self.photo_list.currentRow()
        if 0 <= self._cur < len(self._subjects) and 0 <= r < len(self._subjects[self._cur]["photos"]):
            self._subjects[self._cur]["photos"].pop(r); self._refresh_editor()

    def _add_real(self):
        if not (0 <= self._cur < len(self._subjects)):
            return
        ps, _ = QFileDialog.getOpenFileNames(self, tr("tlora.add_real", "Aggiungi reale"), "",
                                             "Media (*.mp4 *.mov *.webm *.mkv *.png *.jpg *.jpeg)")
        if ps:
            self._subjects[self._cur]["real_media"].extend(ps); self._refresh_editor()

    def _del_real(self):
        r = self.real_list.currentRow()
        if 0 <= self._cur < len(self._subjects) and 0 <= r < len(self._subjects[self._cur]["real_media"]):
            self._subjects[self._cur]["real_media"].pop(r); self._refresh_editor()

    # ── azioni ──
    def _to_subjects(self):
        from core.lora_video_service import Subject
        out = []
        for s in self._subjects:
            if not s["name"].strip():
                continue
            out.append(Subject(name=s["name"].strip(), type=s["type"], photos=list(s["photos"]),
                               case_ids=list(s["case_ids"]),
                               customs=[ln for ln in s["customs"].splitlines() if ln.strip()],
                               real_media=list(s["real_media"]), face_focus=s["face_focus"]))
        return out

    def _install(self):
        import subprocess
        import sys
        self._log("Installazione trainer LTX-2…")
        try:
            subprocess.Popen([sys.executable, str(Path(__file__).resolve().parents[2] / "scripts" / "install_ltx_trainer.py")])
        except Exception as e:                          # noqa: BLE001
            QMessageBox.critical(self, tr("tlora.train", "LoRA"), str(e))

    def _start(self, full: bool):
        if self._worker and self._worker.isRunning():
            return
        self._save_cur()
        subjects = self._to_subjects()
        if not subjects:
            QMessageBox.warning(self, tr("tlora.train", "LoRA"),
                               tr("tlora.need_subject", "Aggiungi almeno un soggetto con un nome.")); return
        if not any(s.all_photos() or s.real_media for s in subjects):
            QMessageBox.warning(self, tr("tlora.train", "LoRA"),
                               tr("tlora.need_media", "Aggiungi almeno una foto o un video reale per i soggetti.")); return
        import os
        out_dir = str(Path(__file__).resolve().parents[2] / "output" / "loras")
        os.makedirs(out_dir, exist_ok=True)
        opts = {"rank": self.lora_rank.value(), "steps": self.lora_steps.value(),
                "clip_seconds": self.lora_clip.value(), "use_pulid": self.lora_pulid.isChecked(),
                "low_vram": self.lora_lowvram.isChecked(), **self.cap_picker.get()}
        self.btn_dataset.setEnabled(False); self.btn_full.setEnabled(False)
        self._log(tr("tlora.started", "Avvio (può richiedere molto tempo)…"))
        self._worker = _LoraWorker(subjects, self.lora_name.text().strip() or "my_lora", out_dir, opts, full)
        self._worker.progress.connect(self._log)
        self._worker.finished_lora.connect(self._done)
        self._worker.start()

    def _done(self, res: dict):
        self.btn_dataset.setEnabled(True); self.btn_full.setEnabled(True)
        if res.get("error"):
            self._log("ERRORE: " + str(res["error"])[:300])
            QMessageBox.critical(self, tr("tlora.train", "LoRA"), str(res["error"])[:1000]); return
        if res.get("lora"):
            self._log(tr("tlora.done_lora", "LoRA pronta: {f}").format(f=res["lora"]))
        else:
            self._log(tr("tlora.done_ds", "Dataset pronto: {f}").format(f=res.get("dataset", "")))

    def _log(self, m: str):
        self.log.appendPlainText(time.strftime("%H:%M:%S ") + str(m))
