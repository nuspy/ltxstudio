"""Tab «Musica» — generazione musicale standalone (HeartMuLa + MMAudio), niente ComfyUI.

Due sezioni:
- Canzone / Strumentale (HeartMuLa): liriche + tag di stile → MP3 (voce o strumentale).
- Sonorizza video (MMAudio): da un video genera l'audio sincronizzato; senza video, text-to-audio.

I motori girano come subprocess isolati (core/music_service.py); i callback (thread secondario)
sono inoltrati alla UI via Qt Signal. L'installazione dei moduli è lanciata da qui con QProcess.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, Signal, QProcess
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QGridLayout,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QLineEdit, QCheckBox,
    QFileDialog, QScrollArea, QPlainTextEdit, QProgressBar, QMessageBox,
)

from ui.i18n_manager import tr
from core import music_service as ms


class _Bridge(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    song_done = Signal(int)
    sonify_done = Signal(int)


class MusicTab(QWidget):
    music_ready = Signal(str)               # path del file audio prodotto

    def __init__(self):
        super().__init__()
        self._song_job: ms.Job | None = None
        self._sonify_job: ms.Job | None = None
        self._install_proc: QProcess | None = None
        self._last_video = ""
        self._t0 = 0.0
        self._b = _Bridge()
        self._b.log.connect(self._log)
        self._b.progress.connect(self._on_progress)
        self._b.song_done.connect(lambda rc: self._on_done(rc, "song"))
        self._b.sonify_done.connect(lambda rc: self._on_done(rc, "sonify"))

        root = QVBoxLayout(self)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); form = QVBoxLayout(inner); scroll.setWidget(inner)
        root.addWidget(scroll, 1)
        form.addWidget(self._song_box())
        form.addWidget(self._sonify_box())
        form.addWidget(self._progress_box())
        form.addStretch(1)
        self._refresh_install_state()

    # ── Canzone / Strumentale (HeartMuLa) ──────────────────────────────────────
    def _song_box(self) -> QGroupBox:
        box = QGroupBox(tr("music.song.title", "Canzone / Strumentale (HeartMuLa)"))
        v = QVBoxLayout(box)
        f = QFormLayout()
        self.song_name = QLineEdit("song")
        f.addRow(tr("music.name", "Nome"), self.song_name)
        self.song_tags = QLineEdit(ms.DEFAULT_TAGS)
        self.song_tags.setPlaceholderText(tr("music.tags.hint", "stile, separati da virgola: pop, allegro, voce femminile…"))
        f.addRow(tr("music.tags", "Tag di stile"), self.song_tags)
        v.addLayout(f)
        v.addWidget(QLabel(tr("music.lyrics", "Liriche (usa sezioni [Verse], [Chorus]…)")))
        self.song_lyrics = QPlainTextEdit()
        self.song_lyrics.setPlaceholderText("[Verse]\n…\n[Chorus]\n…")
        self.song_lyrics.setFixedHeight(140)
        v.addWidget(self.song_lyrics)
        g = QGridLayout()
        self.song_instr = QCheckBox(tr("music.instrumental", "Strumentale (senza voce)"))
        self.song_instr.toggled.connect(lambda on: self.song_lyrics.setDisabled(on))
        self.song_dur = QSpinBox(); self.song_dur.setRange(5, 360); self.song_dur.setValue(120); self.song_dur.setSuffix(" s")
        self.song_ver = QComboBox(); self.song_ver.addItems(["3B", "7B"])
        self.song_temp = QDoubleSpinBox(); self.song_temp.setRange(0.1, 2.0); self.song_temp.setSingleStep(0.1); self.song_temp.setValue(1.0)
        self.song_cfg = QDoubleSpinBox(); self.song_cfg.setRange(1.0, 10.0); self.song_cfg.setSingleStep(0.5); self.song_cfg.setValue(1.5)
        self.song_lazy = QCheckBox(tr("music.lazy", "Low VRAM (lazy load)"))
        g.addWidget(self.song_instr, 0, 0, 1, 2)
        g.addWidget(QLabel(tr("music.duration", "Durata")), 1, 0); g.addWidget(self.song_dur, 1, 1)
        g.addWidget(QLabel(tr("music.version", "Modello")), 1, 2); g.addWidget(self.song_ver, 1, 3)
        g.addWidget(QLabel(tr("music.temp", "Temperatura")), 2, 0); g.addWidget(self.song_temp, 2, 1)
        g.addWidget(QLabel(tr("music.cfg", "CFG")), 2, 2); g.addWidget(self.song_cfg, 2, 3)
        g.addWidget(self.song_lazy, 3, 0, 1, 2)
        v.addLayout(g)
        h = QHBoxLayout()
        self.btn_song_install = QPushButton(tr("music.install.heart", "Installa HeartMuLa"))
        self.btn_song_install.clicked.connect(lambda: self._install("--heartmula"))
        self.btn_song = QPushButton(tr("music.song.gen", "Genera musica"))
        self.btn_song.clicked.connect(self._gen_song)
        h.addWidget(self.btn_song_install); h.addStretch(1); h.addWidget(self.btn_song)
        v.addLayout(h)
        return box

    # ── Sonorizza video (MMAudio) ──────────────────────────────────────────────
    def _sonify_box(self) -> QGroupBox:
        box = QGroupBox(tr("music.sonify.title", "Sonorizza video (MMAudio)"))
        f = QFormLayout(box)
        row = QHBoxLayout()
        self.son_video = QLineEdit(); self.son_video.setPlaceholderText(tr("music.sonify.vhint", "video da sonorizzare (vuoto = solo prompt)"))
        b = QPushButton(tr("music.browse", "Sfoglia…")); b.clicked.connect(self._pick_video)
        self.btn_last_video = QPushButton(tr("music.lastvideo", "Ultimo video")); self.btn_last_video.setEnabled(False)
        self.btn_last_video.setToolTip(tr("music.lastvideo.tt", "Usa l'ultimo video generato nell'app"))
        self.btn_last_video.clicked.connect(self._use_last_video)
        row.addWidget(self.son_video, 1); row.addWidget(b); row.addWidget(self.btn_last_video)
        w = QWidget(); w.setLayout(row); f.addRow(tr("music.video", "Video"), w)
        self.son_prompt = QLineEdit(); self.son_prompt.setPlaceholderText(tr("music.sonify.phint", "es. pioggia e tuoni, passi sulla ghiaia…"))
        f.addRow(tr("music.prompt", "Prompt"), self.son_prompt)
        self.son_neg = QLineEdit()
        f.addRow(tr("music.neg", "Prompt negativo"), self.son_neg)
        g = QHBoxLayout()
        self.son_dur = QDoubleSpinBox(); self.son_dur.setRange(1, 120); self.son_dur.setValue(8); self.son_dur.setSuffix(" s")
        self.son_steps = QSpinBox(); self.son_steps.setRange(10, 100); self.son_steps.setValue(25)
        self.son_cfg = QDoubleSpinBox(); self.son_cfg.setRange(1.0, 10.0); self.son_cfg.setSingleStep(0.5); self.son_cfg.setValue(4.5)
        self.son_seed = QSpinBox(); self.son_seed.setRange(0, 2_000_000_000); self.son_seed.setValue(42)
        for lbl, wdg in ((tr("music.duration", "Durata"), self.son_dur), (tr("music.steps", "Step"), self.son_steps),
                         (tr("music.cfg", "CFG"), self.son_cfg), (tr("music.seed", "Seed"), self.son_seed)):
            g.addWidget(QLabel(lbl)); g.addWidget(wdg)
        gw = QWidget(); gw.setLayout(g); f.addRow("", gw)
        h = QHBoxLayout()
        self.btn_son_install = QPushButton(tr("music.install.mm", "Installa MMAudio"))
        self.btn_son_install.clicked.connect(lambda: self._install("--mmaudio"))
        self.btn_son = QPushButton(tr("music.sonify.gen", "Genera audio"))
        self.btn_son.clicked.connect(self._gen_sonify)
        h.addWidget(self.btn_son_install); h.addStretch(1); h.addWidget(self.btn_son)
        hw = QWidget(); hw.setLayout(h); f.addRow("", hw)
        return box

    def _progress_box(self) -> QGroupBox:
        box = QGroupBox(tr("music.progress", "Progresso"))
        v = QVBoxLayout(box)
        self.bar = QProgressBar(); self.bar.setRange(0, 100)
        self.status = QLabel("—"); self.status.setStyleSheet("color:#9aa1b0;")
        bar2 = QHBoxLayout()
        self.btn_play = QPushButton(tr("music.play", "Riproduci")); self.btn_play.clicked.connect(self._play); self.btn_play.setEnabled(False)
        self.btn_folder = QPushButton(tr("music.folder", "Apri cartella")); self.btn_folder.clicked.connect(self._open_folder)
        self.btn_stop = QPushButton(tr("music.stop", "Stop")); self.btn_stop.clicked.connect(self._stop); self.btn_stop.setEnabled(False)
        bar2.addWidget(self.btn_play); bar2.addWidget(self.btn_folder); bar2.addStretch(1); bar2.addWidget(self.btn_stop)
        self.logview = QPlainTextEdit(); self.logview.setReadOnly(True); self.logview.setFixedHeight(150)
        self.logview.setStyleSheet("font-family:Consolas,monospace;font-size:11px;")
        v.addWidget(self.bar); v.addWidget(self.status); v.addLayout(bar2); v.addWidget(self.logview)
        self._last_audio = ""
        return box

    # ── Helpers / slot ──────────────────────────────────────────────────────────
    def _pick_video(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("music.video", "Video"), "", "Video (*.mp4 *.mov *.mkv *.webm *.avi)")
        if p:
            self.son_video.setText(p)

    def set_last_video(self, path: str):
        """Chiamato da MainWindow quando l'app genera un video: abilita «Ultimo video»."""
        self._last_video = path or ""
        self.btn_last_video.setEnabled(bool(self._last_video))

    def _use_last_video(self):
        if self._last_video:
            self.son_video.setText(self._last_video)

    def _refresh_install_state(self):
        h = ms.is_installed_heartmula(); m = ms.is_installed_mmaudio()
        self.btn_song_install.setEnabled(not h)
        self.btn_song_install.setText(tr("music.installed.heart", "HeartMuLa installato ✓") if h else tr("music.install.heart", "Installa HeartMuLa"))
        self.btn_son_install.setEnabled(not m)
        self.btn_son_install.setText(tr("music.installed.mm", "MMAudio installato ✓") if m else tr("music.install.mm", "Installa MMAudio"))

    def _busy(self) -> bool:
        return bool((self._song_job and self._song_job.running) or (self._sonify_job and self._sonify_job.running)
                    or self._install_proc is not None)

    def _gen_song(self):
        if self._busy():
            return
        p = ms.SongParams(
            name=self.song_name.text().strip() or "song",
            lyrics=self.song_lyrics.toPlainText(), tags=self.song_tags.text().strip(),
            duration_s=self.song_dur.value(), instrumental=self.song_instr.isChecked(),
            version=self.song_ver.currentText(), temperature=self.song_temp.value(),
            cfg_scale=self.song_cfg.value(), lazy_load=self.song_lazy.isChecked())
        errs = ms.preflight_song(p)
        if errs:
            QMessageBox.warning(self, tr("music.err", "Impossibile generare"), "\n".join(errs)); return
        self._start_job(ms.make_song_job(p, on_log=self._b.log.emit, on_progress=self._b.progress.emit,
                                         on_done=self._b.song_done.emit), "song", p.out_mp3())

    def _gen_sonify(self):
        if self._busy():
            return
        p = ms.SonifyParams(
            name=self.son_name() , video_path=self.son_video.text().strip(),
            prompt=self.son_prompt.text().strip(), negative_prompt=self.son_neg.text().strip(),
            duration_s=self.son_dur.value(), cfg_strength=self.son_cfg.value(),
            num_steps=self.son_steps.value(), seed=self.son_seed.value())
        errs = ms.preflight_sonify(p)
        if errs:
            QMessageBox.warning(self, tr("music.err", "Impossibile generare"), "\n".join(errs)); return
        self._start_job(ms.make_sonify_job(p, on_log=self._b.log.emit, on_progress=self._b.progress.emit,
                                           on_done=self._b.sonify_done.emit), "sonify", None)

    def son_name(self) -> str:
        v = self.son_video.text().strip()
        return Path(v).stem if v else "sfx"

    def _start_job(self, job: ms.Job, kind: str, expected_out):
        self.logview.clear(); self.bar.setValue(0); self._t0 = time.time()
        self._expected_out = str(expected_out) if expected_out else ""
        if kind == "song":
            self._song_job = job
        else:
            self._sonify_job = job
        try:
            job.start()
        except Exception as e:                          # noqa: BLE001
            QMessageBox.critical(self, tr("music.err", "Errore"), str(e)); return
        self.btn_song.setEnabled(False); self.btn_son.setEnabled(False); self.btn_stop.setEnabled(True)
        self.btn_play.setEnabled(False)
        self.status.setText(tr("music.generating", "Generazione in corso…"))
        self._log(f"[{kind}] avviato")

    def _stop(self):
        for j in (self._song_job, self._sonify_job):
            if j and j.running:
                j.stop()
        self.btn_stop.setEnabled(False)
        self._log("[music] stop richiesto")

    def _install(self, which: str):
        if self._install_proc is not None:
            return
        if QMessageBox.question(self, tr("music.install.title", "Installa modulo musica"),
                                tr("music.install.confirm", "Verranno scaricati il motore + PyTorch (cu128) e i pesi (diversi GB). Procedo?")) != QMessageBox.Yes:
            return
        script = str(Path(__file__).resolve().parents[2] / "scripts" / "install_music.py")
        self._log(f"[install] {which} …")
        self.btn_song_install.setEnabled(False); self.btn_son_install.setEnabled(False)
        p = QProcess(self); p.setProcessChannelMode(QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(
            lambda: self._log(bytes(p.readAllStandardOutput()).decode("utf-8", "replace").rstrip()))
        p.finished.connect(self._install_finished)
        self._install_proc = p
        p.start(sys.executable, ["-u", script, which])

    def _install_finished(self, code, _s):
        self._log(f"[install] terminato (codice {code})")
        self._install_proc = None
        self._refresh_install_state()

    def _log(self, line: str):
        if not line:
            return
        self.logview.appendPlainText(line)
        sb = self.logview.verticalScrollBar(); sb.setValue(sb.maximum())

    def _on_progress(self, cur: int, total: int):
        self.bar.setRange(0, total); self.bar.setValue(cur)
        self.status.setText(tr("music.step", "{c}/{t}").format(c=cur, t=total))

    def _on_done(self, code: int, kind: str):
        self._song_job = None if kind == "song" else self._song_job
        self._sonify_job = None if kind == "sonify" else self._sonify_job
        self.btn_song.setEnabled(True); self.btn_son.setEnabled(True); self.btn_stop.setEnabled(False)
        if code != 0:
            self.status.setText(tr("music.failed", "Generazione fallita (codice {c})").format(c=code)); return
        out = self._expected_out if (self._expected_out and Path(self._expected_out).exists()) else None
        if not out:
            found = ms.find_latest_audio(after=self._t0 - 1)
            out = str(found) if found else ""
        if out:
            self._last_audio = out
            self.btn_play.setEnabled(True)
            self.status.setText(tr("music.done", "Pronto: {f}").format(f=Path(out).name))
            self._log(f"[music] output: {out}")
            self.music_ready.emit(out)
        else:
            self.status.setText(tr("music.done.nofile", "Completato (file non individuato)"))

    def _play(self):
        if self._last_audio and os.path.exists(self._last_audio):
            os.startfile(self._last_audio)             # player di sistema

    def _open_folder(self):
        ms.MUSIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(str(ms.MUSIC_OUTPUT_DIR))
