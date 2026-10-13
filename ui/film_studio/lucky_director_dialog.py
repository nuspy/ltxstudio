"""Lucky Director — maschera unica + esecuzione headless con SOLO barra di progresso.

L'utente inserisce un prompt (o carica un file), sceglie formato/risoluzione/durata, il tipo di montaggio
TRA SCENE e TRA RIPRESE, e la modalità di coerenza. Poi tutto avviene sotto il cofano
(core.film_studio.lucky_director.make_film): copione → scene → riprese → immagini → video → montaggio.
Visibile solo l'avanzamento stimato + la fase corrente con il prompt dell'artefatto in generazione.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QVBoxLayout,
                               QWidget, QCheckBox)

from core.film_studio import lucky_director as ld
from ui.film_studio.montage_tab import _TRANS
from ui.i18n_manager import tr

_CONSIST = [("fast", tr("fs.path.fast", "Rapida (riferimenti)")),
            ("lora", tr("fs.path.lora", "LoRA per personaggio")),
            ("mixed", tr("fs.path.mixed", "Mista"))]


def _phase_text(phase: str, idx: int, tot: int, prompt: str) -> str:
    p = (prompt or "").strip().replace("\n", " ")
    if len(p) > 140:
        p = p[:140] + "…"
    if phase == "script":
        return tr("ld.ph.script", "Phase 1: writing the screenplay…")
    if phase == "breakdown":
        return tr("ld.ph.breakdown", "Phase 2: breaking into scenes and shots…")
    if phase == "keyframe":
        return tr("ld.ph.keyframe", "Phase 3: generating images. Image {i} of {n}: {p}").format(i=idx, n=tot, p=p)
    if phase == "video":
        return tr("ld.ph.video", "Phase 4: generating video. Shot {i} of {n}: {p}").format(i=idx, n=tot, p=p)
    if phase == "audio":
        return tr("ld.ph.audio", "Phase 5: synthesizing dialogue…")
    if phase == "music":
        return tr("ld.ph.music", "Phase 6: scoring the soundtrack…")
    if phase == "master":
        return tr("ld.ph.master", "Phase 7: final edit…")
    if phase == "done":
        return tr("ld.ph.done", "Done.")
    return phase


class _Worker(QThread):
    progress = Signal(str, int, int, str, float)
    ok = Signal(dict)
    fail = Signal(str)

    def __init__(self, spec: dict):
        super().__init__()
        self._spec = spec
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            res = ld.make_film(self._spec,
                               on_progress=lambda ph, i, n, pr, pc: self.progress.emit(ph, i, n, pr, pc),
                               cancel=lambda: self._cancel)
            self.ok.emit(res)
        except Exception as e:                               # noqa: BLE001
            self.fail.emit(str(e))


class LuckyDirectorDialog(QDialog):
    """Maschera + progresso in un unico dialog. Emette project_ready(project_id) a fine generazione."""
    project_ready = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("ld.title", "🍀 Lucky Director"))
        self.setModal(True)
        self.resize(620, 560)
        self._worker = None
        v = QVBoxLayout(self)

        # ── pannello INPUT ──────────────────────────────────────────────
        self._panel = QWidget(); pv = QVBoxLayout(self._panel); pv.setContentsMargins(0, 0, 0, 0)
        pv.addWidget(QLabel(tr("ld.intro", "Describe the story. Lucky Director does everything automatically "
                                           "and saves a normal Film Studio project you can reopen and edit.")))
        self.prompt = QPlainTextEdit(); self.prompt.setPlaceholderText(
            tr("ld.prompt.ph", "e.g. A lonely lighthouse keeper befriends a stranded astronaut during a storm…"))
        self.prompt.setMinimumHeight(120)
        pv.addWidget(self.prompt)
        frow = QHBoxLayout()
        self.file_lbl = QLabel(tr("ld.file.none", "(no file)"))
        btn_file = QPushButton(tr("ld.file.load", "📄 Load file…")); btn_file.clicked.connect(self._pick_file)
        btn_clear = QPushButton(tr("ld.file.clear", "✖")); btn_clear.setMaximumWidth(34)
        btn_clear.clicked.connect(lambda: self._set_file(""))
        frow.addWidget(btn_file); frow.addWidget(self.file_lbl, 1); frow.addWidget(btn_clear)
        pv.addLayout(frow)
        self._file = ""

        form = QFormLayout()
        self.aspect = QComboBox()
        for a in ld.ASPECTS:
            self.aspect.addItem(a, a)
        form.addRow(tr("ld.aspect", "Aspect ratio"), self.aspect)
        self.res = QComboBox()
        for lab, px in ld.RESOLUTIONS:
            self.res.addItem(lab, px)
        self.res.setCurrentIndex(1)                          # 480p default
        form.addRow(tr("ld.resolution", "Resolution"), self.res)
        # durata h/m/s
        drow = QHBoxLayout()
        self.h = QSpinBox(); self.h.setRange(0, 12); self.h.setSuffix(" h")
        self.m = QSpinBox(); self.m.setRange(0, 59); self.m.setValue(1); self.m.setSuffix(" m")
        self.s = QSpinBox(); self.s.setRange(0, 59); self.s.setSuffix(" s")
        drow.addWidget(self.h); drow.addWidget(self.m); drow.addWidget(self.s); drow.addStretch(1)
        dwrap = QWidget(); dwrap.setLayout(drow)
        form.addRow(tr("ld.duration", "Output duration"), dwrap)
        # montaggi
        self.scene_trans = QComboBox(); self.shot_trans = QComboBox()
        for code, lab in _TRANS:
            self.scene_trans.addItem(lab, code); self.shot_trans.addItem(lab, code)
        self.scene_trans.setCurrentIndex(1)                  # dissolvenza tra scene
        form.addRow(tr("ld.trans_scene", "Between scenes"), self.scene_trans)
        form.addRow(tr("ld.trans_shot", "Between shots"), self.shot_trans)
        self.consist = QComboBox()
        for code, lab in _CONSIST:
            self.consist.addItem(lab, code)
        form.addRow(tr("ld.consistency", "Character consistency"), self.consist)
        # 2ª architettura IMMAGINE: FLUX vs Krea 2 (per i keyframe)
        self.img_arch = QComboBox()
        self.img_arch.addItem(tr("ld.arch_flux", "FLUX"), "flux")
        self.img_arch.addItem(tr("ld.arch_krea2", "Krea 2"), "krea2")
        self.img_arch.addItem(tr("f1.engine.qwen21", "Qwen-Image 2.1"), "qwen_image21")
        form.addRow(tr("ld.image_arch", "Image architecture"), self.img_arch)
        # coerenza volto negli shot con personaggi @Nome (NON sostituisce FLUX: switch)
        self.coherence = QComboBox()
        self.coherence.addItem(tr("ld.coh_flux", "FLUX (PuLID/Kontext)"), "flux")
        self.coherence.addItem(tr("ld.coh_krea2edit", "Krea 2 Edit (coerenza volto)"), "krea2_edit")
        self.coherence.addItem(tr("fs.sb.coh_qwen21", "Qwen-Image 2.1 (fino a 10 personaggi)"), "qwen_image21")
        self.coherence.setToolTip(tr("ld.coh.tt", "Ogni scena usa i riferimenti dei suoi personaggi."))
        form.addRow(tr("ld.coherence", "Face coherence"), self.coherence)
        from ui.film_studio.qwen21_combo import make_qwen21_dit_combo
        self.qwen_dit = make_qwen21_dit_combo()
        form.addRow(tr("fs.sb.qwen21_dit", "Qwen:"), self.qwen_dit)
        # 2ª architettura video: LTX-2 vs LingBot-Video
        self.vid_arch = QComboBox()
        self.vid_arch.addItem(tr("ld.arch_ltx", "LTX-2"), "ltx")
        self.vid_arch.addItem(tr("ld.arch_lingbot", "LingBot-Video"), "lingbot")
        self.vid_arch.addItem(tr("ld.arch_krea_rt", "Krea Realtime"), "krea_rt")
        self.vid_arch.addItem(tr("ld.arch_self_forcing", "Self-Forcing (realtime, lungo)"), "self_forcing")
        form.addRow(tr("ld.video_arch", "Video architecture"), self.vid_arch)
        self.music = QCheckBox(tr("ld.music", "Generate a soundtrack")); self.music.setChecked(True)
        form.addRow("", self.music)
        pv.addLayout(form)
        v.addWidget(self._panel)

        # ── pannello PROGRESSO (nascosto finché non parte) ──────────────
        self._prog_panel = QWidget(); gv = QVBoxLayout(self._prog_panel); gv.setContentsMargins(0, 0, 0, 0)
        self.status = QLabel(tr("ld.ready", "Ready.")); self.status.setWordWrap(True)
        self.status.setStyleSheet("QLabel{color:#d6e6f2;}")
        self.bar = QProgressBar(); self.bar.setRange(0, 1000); self.bar.setValue(0)
        gv.addWidget(self.status); gv.addWidget(self.bar)
        self._prog_panel.setVisible(False)
        v.addWidget(self._prog_panel)

        # ── pulsanti ────────────────────────────────────────────────────
        self.bb = QDialogButtonBox()
        self.btn_go = self.bb.addButton(tr("ld.start", "🍀 Generate film"), QDialogButtonBox.AcceptRole)
        self.btn_cancel = self.bb.addButton(QDialogButtonBox.Cancel)
        self.btn_go.clicked.connect(self._start)
        self.btn_cancel.clicked.connect(self._on_cancel)
        v.addWidget(self.bb)

    # ── file ────────────────────────────────────────────────────────────
    def _pick_file(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, tr("ld.file.load", "Load file"), "",
            tr("ld.file.filter", "Story files (*.txt *.md *.fountain *.pdf *.docx)"))
        if fn:
            self._set_file(fn)

    def _set_file(self, path: str):
        self._file = path or ""
        import os
        self.file_lbl.setText(os.path.basename(path) if path else tr("ld.file.none", "(no file)"))

    # ── spec ────────────────────────────────────────────────────────────
    def _build_spec(self) -> dict:
        return {
            "prompt": self.prompt.toPlainText().strip(),
            "file": self._file,
            "duration_s": self.h.value() * 3600 + self.m.value() * 60 + self.s.value(),
            "aspect": self.aspect.currentData(),
            "resolution": int(self.res.currentData()),
            "consistency": self.consist.currentData(),
            "scene_trans": self.scene_trans.currentData() or "",
            "shot_trans": self.shot_trans.currentData() or "",
            "make_music": self.music.isChecked(),
            "img_arch": self.img_arch.currentData() or "flux",
            "vid_arch": self.vid_arch.currentData() or "ltx",
            "coherence_engine": self.coherence.currentData() or "flux",
            "qwen21_dit": self.qwen_dit.currentData() or "",
        }

    # ── esecuzione ──────────────────────────────────────────────────────
    def _start(self):
        spec = self._build_spec()
        if not spec["prompt"] and not spec["file"]:
            self.status.setText(tr("ld.need_input", "Enter a prompt or load a file first."))
            self._prog_panel.setVisible(True)
            return
        if spec["duration_s"] < 5:
            spec["duration_s"] = 60
        self._panel.setEnabled(False)
        self._prog_panel.setVisible(True)
        self.btn_go.setEnabled(False)
        self.status.setText(tr("ld.starting", "Starting…"))
        self._worker = _Worker(spec)
        self._worker.progress.connect(self._on_prog)
        self._worker.ok.connect(self._on_ok)
        self._worker.fail.connect(self._on_fail)
        self._worker.start()

    def _on_prog(self, phase, idx, tot, prompt, pct):
        self.bar.setValue(int(max(0.0, min(1.0, pct)) * 1000))
        self.status.setText(_phase_text(phase, idx, tot, prompt))

    def _on_ok(self, res: dict):
        self.bar.setValue(1000)
        self.status.setText(tr("ld.finished", "🍀 Film ready: {f}").format(f=res.get("master", "")))
        pid = int(res.get("project_id") or 0)
        if pid:
            self.project_ready.emit(pid)
        self.btn_cancel.setText(tr("ld.close", "Close"))

    def _on_fail(self, msg: str):
        self.status.setText(tr("ld.error", "Error: {e}").format(e=msg[:300]))
        self._panel.setEnabled(True)
        self.btn_go.setEnabled(True)

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.status.setText(tr("ld.cancelling", "Cancelling…"))
        else:
            self.reject()
