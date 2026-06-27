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

from PySide6.QtCore import QObject, Qt, Signal, QProcess, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QGridLayout,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QLineEdit, QCheckBox,
    QFileDialog, QScrollArea, QPlainTextEdit, QProgressBar, QMessageBox,
)

from ui.i18n_manager import tr

# lingue per la generazione del testo (codice → nome nativo), allineate all'i18n dell'app
_LYRIC_LANGS = [
    ("it", "Italiano"), ("en", "English"), ("es", "Español"), ("fr", "Français"),
    ("de", "Deutsch"), ("pt", "Português"), ("ru", "Русский"), ("uk", "Українська"),
    ("zh", "中文"), ("ko", "한국어"), ("ar", "العربية"), ("he", "עברית"),
    ("hi", "हिन्दी"), ("el", "Ελληνικά"), ("hu", "Magyar"),
]


class _LyricsWorker(QThread):
    """Genera le liriche con l'LLM locale (Gemma-3) fuori dal thread UI."""
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, prompt: str, language: str, tags: str):
        super().__init__()
        self._prompt, self._lang, self._tags = prompt, language, tags

    def run(self):
        try:
            from core import lyrics_llm
            self.done.emit(lyrics_llm.generate(self._prompt, self._lang, self._tags))
        except Exception as e:                          # noqa: BLE001
            self.failed.emit(str(e))


class _MusicVideoWorker(QThread):
    """Genera SOLO la canzone (poi si apre il Music Video Studio interattivo) fuori dal thread UI."""
    progress = Signal(str)
    finished_mv = Signal(dict)

    def __init__(self, lyrics: str, tags: str, image: str, out_video: str, duration: float,
                 scene_seconds: float = 6.0, video_style: str = ""):
        super().__init__()
        self._args = (lyrics, tags, image, out_video, duration, scene_seconds, video_style)

    def run(self):
        lyrics, tags, image, out_video, duration, scene_seconds, video_style = self._args
        try:
            import os as _os
            from core import music_video_service as mvs
            song = mvs.generate_song(lyrics=lyrics, tags=tags, out_mp3=_os.path.splitext(out_video)[0] + ".mp3",
                                     duration_s=duration, progress=self.progress.emit)
            self.finished_mv.emit({"song": song, "lyrics": lyrics, "image": image,
                                   "scene_seconds": scene_seconds, "video_style": video_style})
        except Exception as e:                          # noqa: BLE001
            self.finished_mv.emit({"error": str(e)})


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
        # --- Genera il testo con l'AI (LLM locale Gemma-3) ---
        ai = QHBoxLayout()
        self.lyr_prompt = QLineEdit()
        self.lyr_prompt.setPlaceholderText(
            tr("music.lyr_ai.prompt_ph", "Tema della canzone (es. amore d'estate, nostalgia…)"))
        self.lyr_lang = QComboBox()
        for _code, _name in _LYRIC_LANGS:
            self.lyr_lang.addItem(_name, _code)
        self.lyr_lang.setToolTip(tr("music.lyr_ai.lang.tt", "Lingua del testo della canzone"))
        self.btn_lyr_ai = QPushButton(tr("music.lyr_ai.btn", "✨ Genera testo con AI"))
        self.btn_lyr_ai.setToolTip(tr("music.lyr_ai.btn.tt",
            "Scrive il testo con l'AI dal tema e lo inserisce qui sopra (usa Gemma-3 locale)"))
        self.btn_lyr_ai.clicked.connect(self._gen_lyrics_ai)
        ai.addWidget(self.lyr_prompt, 1)
        ai.addWidget(self.lyr_lang)
        ai.addWidget(self.btn_lyr_ai)
        v.addLayout(ai)
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
        # --- Videomusicale: immagine iniziale/sfondo ---
        mv = QHBoxLayout()
        self.mv_image = QLineEdit()
        self.mv_image.setPlaceholderText(tr("music.mv.image_ph", "Immagine iniziale/sfondo per il videomusicale"))
        self.btn_mv_browse = QPushButton(tr("music.mv.browse", "Sfoglia…"))
        self.btn_mv_browse.clicked.connect(self._pick_mv_image)
        mv.addWidget(QLabel(tr("music.mv.image", "Immagine video")))
        mv.addWidget(self.mv_image, 1)
        mv.addWidget(self.btn_mv_browse)
        v.addLayout(mv)
        # --- Videomusicale: stile video + ritmo dei cambi scena ---
        mv2 = QHBoxLayout()
        self.mv_style = QLineEdit()
        self.mv_style.setPlaceholderText(tr("music.mv.style_ph",
            "Stile/resa del video: realistico, cartone animato, bianco e nero, romantico triste…"))
        mv2.addWidget(QLabel(tr("music.mv.style", "Stile video")))
        mv2.addWidget(self.mv_style, 1)
        self.mv_pace = QComboBox()
        # (etichetta, secondi/scena) — più piccolo = stacchi più rapidi
        for _lab, _sec in (("music.mv.pace.slow", 12.0), ("music.mv.pace.medium", 7.0),
                           ("music.mv.pace.fast", 4.0), ("music.mv.pace.vfast", 2.5)):
            self.mv_pace.addItem(tr(_lab, {"music.mv.pace.slow": "Lento (~12s)",
                "music.mv.pace.medium": "Medio (~7s)", "music.mv.pace.fast": "Veloce (~4s)",
                "music.mv.pace.vfast": "Molto veloce (~2.5s)"}[_lab]), _sec)
        self.mv_pace.setCurrentIndex(2)                   # default Veloce
        mv2.addWidget(QLabel(tr("music.mv.pace", "Ritmo cambi scena")))
        mv2.addWidget(self.mv_pace)
        v.addLayout(mv2)
        h = QHBoxLayout()
        self.btn_song_install = QPushButton(tr("music.install.heart", "Installa HeartMuLa"))
        self.btn_song_install.clicked.connect(lambda: self._install("--heartmula"))
        self.btn_song = QPushButton(tr("music.song.gen", "Genera musica"))
        self.btn_song.clicked.connect(self._gen_song)
        # Videomusicale automatico: canzone → storyboard LLM → video nativo coerente col testo
        self.btn_music_video = QPushButton(tr("music.mv.gen", "🎬 Genera videomusicale"))
        self.btn_music_video.setToolTip(tr("music.mv.gen.tt",
            "Crea canzone + video coerente col testo (storyboard LLM + LTX-2.3 nativo). Richiede un'immagine iniziale."))
        self.btn_music_video.clicked.connect(self._gen_music_video)
        h.addWidget(self.btn_song_install); h.addStretch(1)
        h.addWidget(self.btn_song); h.addWidget(self.btn_music_video)
        v.addLayout(h)
        return box

    def _pick_mv_image(self):
        from PySide6.QtWidgets import QFileDialog
        p, _ = QFileDialog.getOpenFileName(self, tr("music.mv.image", "Immagine video"), "",
                                           "Immagini (*.png *.jpg *.jpeg *.webp)")
        if p:
            self.mv_image.setText(p)

    # ── Generazione testo canzone con AI (LLM locale Gemma-3) ──
    def _gen_lyrics_ai(self):
        prompt = self.lyr_prompt.text().strip()
        if not prompt:
            QMessageBox.information(self, tr("music.lyr_ai.btn", "Genera testo"),
                                    tr("music.lyr_ai.need_prompt", "Scrivi prima un tema per la canzone."))
            return
        self.btn_lyr_ai.setEnabled(False)
        self.btn_lyr_ai.setText(tr("music.lyr_ai.working", "Generazione…"))
        self._lyr_worker = _LyricsWorker(prompt, self.lyr_lang.currentData(), self.song_tags.text().strip())
        self._lyr_worker.done.connect(self._on_lyrics_done)
        self._lyr_worker.failed.connect(self._on_lyrics_failed)
        self._lyr_worker.start()

    def _reset_lyr_btn(self):
        self.btn_lyr_ai.setEnabled(True)
        self.btn_lyr_ai.setText(tr("music.lyr_ai.btn", "✨ Genera testo con AI"))

    def _on_lyrics_done(self, text: str):
        self.song_lyrics.setPlainText(text)             # inserimento automatico nel campo liriche
        self._reset_lyr_btn()

    def _on_lyrics_failed(self, err: str):
        self._reset_lyr_btn()
        QMessageBox.warning(self, tr("music.lyr_ai.btn", "Genera testo"),
                            tr("music.lyr_ai.error", "Generazione testo fallita:") + "\n" + err)

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

    def _gen_music_video(self):
        if getattr(self, "_mv_worker", None) and self._mv_worker.isRunning():
            return
        if self._busy():
            QMessageBox.information(self, tr("music.mv.gen", "Videomusicale"),
                                   tr("music.busy", "Un'altra generazione è in corso.")); return
        lyrics = self.song_lyrics.toPlainText().strip()
        tags = self.song_tags.text().strip() or ms.DEFAULT_TAGS
        image = self.mv_image.text().strip()
        if not lyrics:
            QMessageBox.warning(self, tr("music.mv.gen", "Videomusicale"),
                               tr("music.mv.need_lyrics", "Inserisci le liriche (con sezioni [Verse]/[Chorus]).")); return
        if not image or not Path(image).exists():
            QMessageBox.warning(self, tr("music.mv.gen", "Videomusicale"),
                               tr("music.mv.need_image", "Scegli un'immagine iniziale/sfondo per il video.")); return
        ms.MUSIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = str(ms.MUSIC_OUTPUT_DIR / ((self.song_name.text().strip() or "song") + "_video.mp4"))
        self.btn_music_video.setEnabled(False)
        self._t0 = time.time()
        self._log(tr("music.mv.started", "Videomusicale: canzone + storyboard + video (può richiedere parecchi minuti)…"))
        self._mv_worker = _MusicVideoWorker(lyrics, tags, image, out, float(self.song_dur.value()),
                                            scene_seconds=float(self.mv_pace.currentData()),
                                            video_style=self.mv_style.text().strip())
        self._mv_worker.progress.connect(self._log)
        self._mv_worker.finished_mv.connect(self._on_mv_done)
        self._mv_worker.start()

    def _on_mv_done(self, res: dict):
        self.btn_music_video.setEnabled(True)
        if res.get("error"):
            self._log(tr("music.mv.fail", "Videomusicale fallito: ") + str(res["error"])[:300])
            QMessageBox.critical(self, tr("music.mv.gen", "Videomusicale"), str(res["error"])[:1000]); return
        # canzone pronta → apre il Music Video Studio interattivo (storyboard editabile → immagini → video)
        song = res.get("song")
        self._log(tr("music.mv.song_done", "Canzone pronta → apro lo Studio interattivo."))
        from ui.music_video_studio import MusicVideoStudio
        self._mv_studio = MusicVideoStudio(
            song_path=song, lyrics=res.get("lyrics", ""), image_hint="the provided starting photo",
            scene_seconds=res.get("scene_seconds", 4.0), video_style=res.get("video_style", ""),
            parent=self.window())
        self._mv_studio.setWindowFlag(Qt.Window, True)
        self._mv_studio.show(); self._mv_studio.raise_(); self._mv_studio.activateWindow()

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
