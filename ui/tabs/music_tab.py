"""Tab «Musica» — generazione musicale standalone (HeartMuLa + MMAudio), niente ComfyUI.

Due sezioni:
- Canzone / Strumentale (HeartMuLa): liriche + tag di stile → MP3 (voce o strumentale).
- Sonorizza video (MMAudio): da un video genera l'audio sincronizzato; senza video, text-to-audio.

I motori girano come subprocess isolati (core/music_service.py); i callback (thread secondario)
sono inoltrati alla UI via Qt Signal. L'installazione dei moduli è lanciata da qui con QProcess.
"""
from __future__ import annotations

import os
import random
import re
import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal, QProcess, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QGridLayout,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QLineEdit, QCheckBox,
    QFileDialog, QScrollArea, QPlainTextEdit, QProgressBar, QMessageBox, QToolButton,
)

from ui.i18n_manager import tr

# lingue per la generazione del testo (codice → nome nativo), allineate all'i18n dell'app
_LYRIC_LANGS = [
    ("it", "Italiano"), ("en", "English"), ("es", "Español"), ("fr", "Français"),
    ("de", "Deutsch"), ("pt", "Português"), ("ru", "Русский"), ("uk", "Українська"),
    ("zh", "中文"), ("ko", "한국어"), ("ar", "العربية"), ("he", "עברית"),
    ("hi", "हिन्दी"), ("el", "Ελληνικά"), ("hu", "Magyar"),
]


# Sezioni struttura per i "chip" liriche (inseriscono il tag al cursore).
_LYRIC_SECTIONS = ["Intro", "Verse", "Pre-Chorus", "Chorus", "Bridge", "Instrumental", "Guitar Solo", "Outro"]

# Esempi creativi per il pulsante 🎲 (stili sempre diversi: mai banali). tags/lyrics/bpm/tonalità.
_SONG_EXAMPLES = [
    {"tags": "dark cinematic orchestral, epic choir, war drums, ominous, powerful",
     "lyrics": "[Intro]\n[Verse]\nAsh on the wind, the towers fall\n[Chorus]\nWe rise, we rise beyond the wall\n[Outro]",
     "bpm": 90, "key": "D minor"},
    {"tags": "80s synthwave, neon, driving bassline, retro, nostalgic, female vocals",
     "lyrics": "[Verse]\nMidnight chrome on the empty road\n[Chorus]\nDrive into the electric glow\n[Bridge]\n[Outro]",
     "bpm": 118, "key": "A minor"},
    {"tags": "gregorian chant, sacred, cathedral reverb, male choir, slow, solemn",
     "lyrics": "[Intro]\n[Verse]\nLux aeterna in the stone\n[Chorus]\nVoices carry us alone\n[Outro]",
     "bpm": 60, "key": "E minor"},
    {"tags": "upbeat reggaeton, tropical, dembow, catchy, party, spanish",
     "lyrics": "[Verse]\nBajo el sol la fiesta empieza\n[Chorus]\nBaila, baila, sin pereza\n[Verse 2]\n[Outro]",
     "bpm": 96, "key": "C minor"},
    {"tags": "lo-fi hip hop, dusty vinyl, mellow piano, rainy, chill, instrumental",
     "lyrics": "", "bpm": 84, "key": "F major"},
    {"tags": "aggressive industrial metal, distorted guitars, glitch, pounding, intense",
     "lyrics": "[Intro]\n[Verse]\nSteel and static in my veins\n[Chorus]\nBreak the circuit, break the chains\n[Outro]",
     "bpm": 150, "key": "E minor"},
    {"tags": "warm acoustic folk, fingerpicked guitar, intimate, storytelling, male vocals",
     "lyrics": "[Verse]\nOld river knows my father's name\n[Chorus]\nCarry me home again the same\n[Bridge]\n[Outro]",
     "bpm": 76, "key": "G major"},
]


_TOTAL_VRAM_GB = None                                     # cache: VRAM totale (query una-tantum via nvidia-smi)


def _total_vram_gb() -> float:
    """VRAM totale della GPU in GB (nvidia-smi, cache). 0 se non rilevabile → badge senza colore."""
    global _TOTAL_VRAM_GB
    if _TOTAL_VRAM_GB is None:
        try:
            import subprocess
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=3)
            _TOTAL_VRAM_GB = float(out.stdout.strip().splitlines()[0]) / 1024.0
        except Exception:                                  # noqa: BLE001
            _TOTAL_VRAM_GB = 0.0
    return _TOTAL_VRAM_GB


def _expand_wildcards(text: str) -> str:
    """Risolve i wildcard {a|b|c} scegliendo un'opzione a caso (single-level). Preserva [tag] struttura."""
    if not text or "{" not in text:
        return text
    def _pick(m):
        opts = [o.strip() for o in m.group(1).split("|")]
        return random.choice(opts) if opts else ""
    # più passate per gestire wildcard multipli sulla stessa riga
    for _ in range(8):
        new = re.sub(r"\{([^{}]*)\}", _pick, text)
        if new == text:
            break
        text = new
    return text


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


# (rimosso _MusicVideoWorker: ora il videomusicale è un PROGETTO persistente gestito dal Music Video
#  Studio — la canzone si genera/rigenera nella fase Canzone dello Studio stesso.)


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
        # trainer LoRA musica creato PRIMA (referenziato da _on_song_engine dentro _song_box)
        from ui.widgets.music_trainer import MusicTrainerPanel
        self.song_trainer = MusicTrainerPanel()
        self.song_trainer.train_requested.connect(self._train_lora)
        form.addWidget(self._song_box())
        form.addWidget(self._sonify_box())
        # Post-produzione audio (ffmpeg): normalizza/trim/fade/converti — tra Sonorizza e Trainer
        from ui.widgets.music_post import MusicPostPanel
        self.song_post = MusicPostPanel()
        self.song_post.want_last.connect(lambda: self.song_post.set_src(self._last_audio))
        self.song_post.process_requested.connect(self._process_audio)
        form.addWidget(self.song_post)
        # Correzione intonazione voce (DiffPitcher) — scaffold, test live pendente
        from ui.widgets.pitch_panel import PitchPanel
        self.pitch_panel = PitchPanel()
        self.pitch_panel.want_last.connect(lambda: self.pitch_panel.set_src(self._last_audio))
        self.pitch_panel.process_requested.connect(self._process_pitch)
        form.addWidget(self.pitch_panel)
        form.addWidget(self.song_trainer)          # standalone: tra Sonorizza Video e Progresso
        form.addWidget(self._progress_box())
        form.addStretch(1)
        self._refresh_install_state()

    # ── Canzone / Strumentale (HeartMuLa) ──────────────────────────────────────
    def _song_box(self) -> QGroupBox:
        box = QGroupBox(tr("music.song.title", "Canzone / Strumentale"))
        v = QVBoxLayout(box)
        f = QFormLayout()
        self.song_name = QLineEdit("song")
        f.addRow(tr("music.name", "Nome"), self.song_name)
        self.song_tags = QLineEdit(ms.DEFAULT_TAGS)
        self.song_tags.setPlaceholderText(tr("music.tags.hint", "stile, separati da virgola: pop, allegro, voce femminile…"))
        _tags_row = QWidget(); _trl = QHBoxLayout(_tags_row); _trl.setContentsMargins(0, 0, 0, 0); _trl.setSpacing(4)
        _trl.addWidget(self.song_tags, 1)
        self.btn_tagpick = QToolButton(); self.btn_tagpick.setText("🎨")
        self.btn_tagpick.setToolTip(tr("music.tagpicker.btn.tt", "Scegli i tag di stile dalla palette"))
        self.btn_tagpick.setCursor(Qt.PointingHandCursor); self.btn_tagpick.clicked.connect(self._pick_tags)
        _trl.addWidget(self.btn_tagpick)
        self.btn_example = QToolButton(); self.btn_example.setText("🎲")
        self.btn_example.setToolTip(tr("music.example.tt",
            "Riempie tag, liriche e metadata con un esempio di stile casuale (sempre diverso)."))
        self.btn_example.setCursor(Qt.PointingHandCursor); self.btn_example.clicked.connect(self._random_example)
        _trl.addWidget(self.btn_example)
        self.song_enhance = QCheckBox(tr("music.enhance", "✨ Arricchisci stile (AI)"))
        self.song_enhance.setToolTip(tr("music.enhance.tt",
            "Prima di generare, un LLM locale trasforma i tag in una descrizione di stile ricca (genere, "
            "strumenti, voce, mood, produzione) — come la caption di ACE-Step 1.5. Invisibile, opzionale."))
        _trl.addWidget(self.song_enhance)
        f.addRow(tr("music.tags", "Tag di stile"), _tags_row)
        v.addLayout(f)
        v.addWidget(QLabel(tr("music.lyrics", "Liriche (usa sezioni [Verse], [Chorus]…)")))
        self.song_lyrics = QPlainTextEdit()
        self.song_lyrics.setPlaceholderText("[Verse]\n…\n[Chorus]\n…")
        self.song_lyrics.setFixedHeight(140)
        v.addWidget(self.song_lyrics)
        # Chip sezioni: inseriscono il tag struttura al cursore (niente sintassi a mano)
        chips = QHBoxLayout(); chips.setSpacing(3)
        chips.addWidget(QLabel(tr("music.sections", "Sezioni:")))
        for _sec in _LYRIC_SECTIONS:
            b = QToolButton(); b.setText(_sec); b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(tr("music.sections.tt", "Inserisce [{s}] al punto del cursore nelle liriche.").format(s=_sec))
            b.clicked.connect(lambda _=False, s=_sec: self._insert_section(s))
            chips.addWidget(b)
        chips.addStretch(1)
        v.addLayout(chips)
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
        self.song_ver = QComboBox(); self.song_ver.addItems(ms.available_heartmula_versions() or ["3B"])
        # Motore di generazione: HeartMuLa (default) · ACE-Step v1 · ACE-Step 1.5 (DiT 4B + planner 5Hz)
        self.song_engine = QComboBox(); self.song_engine.addItems(["HeartMuLa", "ACE-Step", "ACE-Step 1.5"])
        self.song_temp = QDoubleSpinBox(); self.song_temp.setRange(0.1, 2.0); self.song_temp.setSingleStep(0.1); self.song_temp.setValue(1.0)
        self.song_cfg = QDoubleSpinBox(); self.song_cfg.setRange(1.0, 10.0); self.song_cfg.setSingleStep(0.5); self.song_cfg.setValue(1.5)
        self.song_lazy = QCheckBox(tr("music.lazy", "Low VRAM (lazy load)"))
        # ACE-Step: aderenza stile via Double Condition Guidance (guidance_scale_text). 0 = off (usa solo CFG).
        self.song_adherence = QDoubleSpinBox(); self.song_adherence.setRange(0.0, 10.0)
        self.song_adherence.setSingleStep(0.5); self.song_adherence.setValue(3.0)
        self.song_adherence.setToolTip(tr("music.adherence.tt",
            "Aderenza ai tag di stile (ACE-Step, Double Condition Guidance). Si attiva solo se >1.0; "
            "3-5 = i tag di stile pesano di più. 0/≤1 = off (solo CFG combinata)."))
        g.addWidget(self.song_instr, 0, 0, 1, 2)
        g.addWidget(QLabel(tr("music.engine", "Motore")), 0, 2); g.addWidget(self.song_engine, 0, 3)
        g.addWidget(QLabel(tr("music.duration", "Durata")), 1, 0); g.addWidget(self.song_dur, 1, 1)
        g.addWidget(QLabel(tr("music.version", "Modello")), 1, 2); g.addWidget(self.song_ver, 1, 3)
        g.addWidget(QLabel(tr("music.temp", "Temperatura")), 2, 0); g.addWidget(self.song_temp, 2, 1)
        g.addWidget(QLabel(tr("music.cfg", "CFG")), 2, 2); g.addWidget(self.song_cfg, 2, 3)
        self.lbl_adherence = QLabel(tr("music.adherence", "Aderenza stile"))
        g.addWidget(self.lbl_adherence, 3, 2); g.addWidget(self.song_adherence, 3, 3)
        g.addWidget(self.song_lazy, 3, 0, 1, 2)
        # ACE-Step: aderenza alle LIRICHE (guidance_scale_lyric). La Double Condition Guidance si attiva
        # solo se SIA 'Aderenza stile' SIA 'Aderenza parole' sono >1.0 → con 0 lo stile-slider non ha effetto.
        self.song_gs_lyric = QDoubleSpinBox(); self.song_gs_lyric.setRange(0.0, 10.0)
        self.song_gs_lyric.setSingleStep(0.5); self.song_gs_lyric.setValue(3.0)
        self.song_gs_lyric.setToolTip(tr("music.gs_lyric.tt",
            "Aderenza alle LIRICHE (ACE-Step, Double Condition Guidance). Insieme a 'Aderenza stile' attiva "
            "la doppia guida: ENTRAMBE devono essere >1.0 perché faccia effetto. 3-5 = testo più fedele. "
            "0 = disattiva la doppia guida (torna alla sola CFG)."))
        self.lbl_gs_lyric = QLabel(tr("music.gs_lyric", "Aderenza parole"))
        g.addWidget(self.lbl_gs_lyric, 4, 2); g.addWidget(self.song_gs_lyric, 4, 3)
        # ACE-Step 1.5: planner 5Hz LM (Chain-of-Thought) che pianifica metadata/struttura prima del DiT.
        self.song_think = QCheckBox(tr("music.think", "Planner 5Hz (think)")); self.song_think.setChecked(True)
        self.song_think.setToolTip(tr("music.think.tt",
            "ACE-Step 1.5: attiva il language-model planner che ragiona su metadata e struttura prima di generare "
            "(migliora coerenza/aderenza). Disattivalo per generare direttamente dalla caption."))
        g.addWidget(self.song_think, 4, 0, 1, 2)
        v.addLayout(g)
        # Preset (salva/carica configurazione) + ↺ Reset
        _rr = QHBoxLayout()
        _rr.addWidget(QLabel(tr("music.preset", "Preset:")))
        self.preset_combo = QComboBox(); self.preset_combo.setMinimumWidth(150)
        self.preset_combo.setToolTip(tr("music.preset.tt", "Configurazioni di stile salvate. Seleziona per caricare."))
        self.preset_combo.activated.connect(self._load_preset)
        _rr.addWidget(self.preset_combo)
        self.btn_preset_save = QToolButton(); self.btn_preset_save.setText("💾")
        self.btn_preset_save.setToolTip(tr("music.preset.save.tt", "Salva i parametri correnti come nuovo preset."))
        self.btn_preset_save.clicked.connect(self._save_preset); _rr.addWidget(self.btn_preset_save)
        self.btn_preset_del = QToolButton(); self.btn_preset_del.setText("🗑")
        self.btn_preset_del.setToolTip(tr("music.preset.del.tt", "Elimina il preset selezionato."))
        self.btn_preset_del.clicked.connect(self._del_preset); _rr.addWidget(self.btn_preset_del)
        self.btn_preset_song = QToolButton(); self.btn_preset_song.setText(tr("music.preset.song", "Da brano…"))
        self.btn_preset_song.setToolTip(tr("music.preset.song.tt",
            "Recupera i parametri da un brano generato in precedenza (legge il suo sidecar)."))
        self.btn_preset_song.clicked.connect(self._load_from_song); _rr.addWidget(self.btn_preset_song)
        _rr.addStretch(1)
        self.btn_reset = QPushButton(tr("music.reset", "↺ Reset default"))
        self.btn_reset.setToolTip(tr("music.reset.tt", "Reimposta tutti i parametri ai valori ottimali del modello selezionato."))
        self.btn_reset.setCursor(Qt.PointingHandCursor); self.btn_reset.clicked.connect(self._reset_defaults)
        _rr.addWidget(self.btn_reset)
        v.addLayout(_rr)
        # F-A: Metadata musicali strutturati (BPM/tonalità/metro/lingua) — accodati ai tag come fa ACE-Step 1.5.
        from ui.widgets.music_metadata import MusicMetadata
        self.song_meta = MusicMetadata()
        v.addWidget(self.song_meta)
        # #98 F2: sezione Avanzate ACE-Step (collassabile, disabilitata con HeartMuLa)
        from ui.widgets.music_advanced import MusicAdvanced
        self.song_adv = MusicAdvanced()
        v.addWidget(self.song_adv)
        # #98 F3: pannello Operazione (retake/repaint/edit/extend/audio2audio) — solo ACE-Step
        from ui.widgets.music_tasks import MusicTaskPanel
        self.song_task = MusicTaskPanel()
        v.addWidget(self.song_task)
        # NB: 'Allena LoRA Musica' NON è qui dentro — è una sezione standalone (creata in _build, tra
        # Sonorizza Video e Progresso), perché è un'operazione a sé (non legata alla generazione canzone).
        # #96/4: default dei parametri per MODELLO selezionato (apertura + cambio) — le due scale di CFG
        # sono DIVERSE: HeartMuLa ~1.5 (1-10), ACE-Step ~15 (apg, 1-30). Applica anche i controlli HM-only.
        self.song_engine.currentTextChanged.connect(self._on_song_engine)
        self._on_song_engine(self.song_engine.currentText())   # applica subito per l'engine già selezionato
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
        # Badge stima carico VRAM (verde/giallo/rosso) + indicatore coda
        self.est_badge = QLabel(""); self.est_badge.setTextFormat(Qt.RichText)
        self.est_badge.setToolTip(tr("music.est.tt",
            "Stima del carico VRAM in base a durata × batch × step × quantizzazione (verde ok, giallo al limite, "
            "rosso rischio offload/OOM). La stima del tempo è indicativa e dipende dalla GPU."))
        v.addWidget(self.est_badge)
        h = QHBoxLayout()
        self.btn_song_install = QPushButton(tr("music.install.heart", "Installa HeartMuLa"))
        self.btn_song_install.clicked.connect(lambda: self._install("--heartmula"))
        self.btn_acestep_install = QPushButton(tr("music.install.acestep", "Installa ACE-Step"))
        self.btn_acestep_install.clicked.connect(lambda: self._install("--acestep"))
        self.btn_song = QPushButton(tr("music.song.gen", "Genera musica"))
        self.btn_song.clicked.connect(self._gen_song)
        # Videomusicale automatico: canzone → storyboard LLM → video nativo coerente col testo
        self.btn_music_video = QPushButton(tr("music.mv.gen", "🎬 Genera videomusicale"))
        self.btn_music_video.setToolTip(tr("music.mv.gen.tt",
            "Crea canzone + video coerente col testo (storyboard LLM + LTX-2.3 nativo). Un'immagine iniziale è opzionale."))
        self.btn_music_video.clicked.connect(self._gen_music_video)
        self.btn_mv_projects = QPushButton(tr("music.mv.projects", "📂 Progetti videomusicali"))
        self.btn_mv_projects.clicked.connect(self._open_mv_projects)
        h.addWidget(self.btn_song_install); h.addWidget(self.btn_acestep_install); h.addStretch(1)
        h.addWidget(self.btn_song); h.addWidget(self.btn_music_video); h.addWidget(self.btn_mv_projects)
        v.addLayout(h)
        # aggiorna il badge quando cambiano i parametri che influenzano il carico
        for _sig in (self.song_dur.valueChanged, self.song_adv.batch.valueChanged,
                     self.song_adv.steps.valueChanged, self.song_lazy.toggled,
                     self.song_adv.quantized.toggled, self.song_engine.currentTextChanged,
                     self.song_ver.currentTextChanged):
            _sig.connect(self._update_estimate)
        self._update_estimate()
        self._refresh_presets()
        return box

    def _update_estimate(self, *_):
        """Aggiorna il badge di stima carico VRAM/tempo in base ai parametri correnti."""
        eng = self.song_engine.currentText()
        is_hm = eng == "HeartMuLa"
        is_15 = eng == "ACE-Step 1.5"
        dur = self.song_dur.value()
        batch = 1 if (is_hm or is_15) else self.song_adv.batch.value()
        steps = 8 if is_15 else (60 if is_hm else self.song_adv.steps.value())
        quant = self.song_adv.quantized.isChecked()
        lazy = self.song_lazy.isChecked()
        if is_hm:                                            # HeartMuLa: modello 3B/RL3B più grande
            base = 20.0 if "RL" in self.song_ver.currentText() else 15.0
        elif is_15:                                          # 1.5: DiT turbo 4B + planner 5Hz LM 1.7B
            base = 12.0
        else:
            base = 5.0 if quant else 8.0
        est = base + 0.03 * dur + 1.5 * max(0, batch - 1)
        if lazy:
            est = min(est, 6.0 + 0.02 * dur)                 # offload su CPU: molta meno VRAM (più lento)
        total = _total_vram_gb()
        # colore in base al rapporto stima/totale (o soglie assolute se totale ignoto)
        ref = total if total > 0 else 24.0
        frac = est / ref
        color = "#2f9e44" if frac < 0.6 else ("#e0a458" if frac < 0.9 else "#e03131")
        load = (tr("music.est.light", "leggero") if steps * dur * batch < 4000 else
                tr("music.est.heavy", "pesante") if steps * dur * batch > 15000 else
                tr("music.est.medium", "medio"))
        tot_txt = f" / {total:.0f}" if total > 0 else ""
        self.est_badge.setText(
            f"<span style='color:{color};'>●</span> " +
            tr("music.est", "VRAM stimata ~{gb:.0f}{tot} GB · carico {load}").format(
                gb=est, tot=tot_txt, load=load) +
            (f" · <span style='color:#e0a458;'>{tr('music.est.lazy', 'offload CPU attivo')}</span>" if lazy else ""))

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
        self.btn_spectro = QPushButton(tr("music.spectro", "📊 Spettro / A-B"))
        self.btn_spectro.setToolTip(tr("music.spectro.tt",
            "Apre lo spettrogramma dell'ultimo brano e permette di confrontarlo con un altro (centroide spettrale)."))
        self.btn_spectro.clicked.connect(self._open_spectro)
        self.btn_library = QPushButton(tr("music.library", "📚 Libreria"))
        self.btn_library.setToolTip(tr("music.library.tt", "Cronologia delle generazioni: riproduci, riusa i parametri, elimina."))
        self.btn_library.clicked.connect(self._open_library)
        self.btn_stop = QPushButton(tr("music.stop", "Stop")); self.btn_stop.clicked.connect(self._stop); self.btn_stop.setEnabled(False)
        bar2.addWidget(self.btn_play); bar2.addWidget(self.btn_folder); bar2.addWidget(self.btn_spectro)
        bar2.addWidget(self.btn_library); bar2.addStretch(1); bar2.addWidget(self.btn_stop)
        self.logview = QPlainTextEdit(); self.logview.setReadOnly(True); self.logview.setFixedHeight(150)
        self.logview.setStyleSheet("font-family:Consolas,monospace;font-size:11px;")
        v.addWidget(self.bar); v.addWidget(self.status); v.addLayout(bar2)
        # Player con forma d'onda + selezione regione (trim) — caricato a fine generazione
        self.wave_player = None
        try:
            from ui.widgets.waveform import WaveformPlayer
            self.wave_player = WaveformPlayer()
            self.wave_player.use_region.connect(self._use_region_as_task)
            v.addWidget(self.wave_player)
        except Exception:                                    # noqa: BLE001 — QtMultimedia assente: player opzionale
            self.wave_player = None
        # Sezione Risultati ricca: riepilogo metadata + qualità (centroide) + LRC/karaoke
        rrow = QHBoxLayout()
        self.result_info = QLabel(""); self.result_info.setTextFormat(Qt.RichText); self.result_info.setWordWrap(True)
        self.result_info.setStyleSheet("color:#9aa1b0;")
        self.btn_lrc = QPushButton(tr("music.lrc", "📝 Genera LRC"))
        self.btn_lrc.setToolTip(tr("music.lrc.tt",
            "Genera un file .lrc con le liriche sincronizzate (karaoke) trascrivendo l'audio (Whisper, CPU)."))
        self.btn_lrc.setEnabled(False); self.btn_lrc.clicked.connect(self._gen_lrc)
        rrow.addWidget(self.result_info, 1); rrow.addWidget(self.btn_lrc)
        v.addLayout(rrow)
        v.addWidget(self.logview)
        self._last_audio = ""; self._lrc_worker = None
        return box

    def _update_result_info(self, path: str):
        """Aggiorna il riepilogo Risultati: metadata + qualità (centroide spettrale)."""
        m = getattr(self, "_song_meta", {}) or {}
        parts = []
        if m.get("engine"):
            parts.append(str(m["engine"]))
        if m.get("duration_s"):
            parts.append(f"{float(m['duration_s']):.0f}s")
        for k, lab in (("bpm", ""), ("keyscale", ""), ("gs_text", "gs_txt "), ("gs_lyric", "gs_lyr ")):
            if m.get(k):
                parts.append(f"{lab}{m[k]}")
        cent = 0.0
        try:
            from ui.widgets.spectrogram import compute_spectrogram
            _img, _sr, cent = compute_spectrogram(path)
        except Exception:                                    # noqa: BLE001
            cent = 0.0
        qual = ""
        if cent > 0:
            tag = (tr("music.q.clean", "pulito") if cent < 2500 else
                   tr("music.q.bright", "brillante") if cent < 3500 else
                   tr("music.q.noisy", "molto brillante/rumoroso"))
            qual = f" · <span style='color:#74c0fc;'>{tr('music.q.centroid', 'centroide')} {cent:.0f} Hz ({tag})</span>"
        self.result_info.setText(" · ".join(parts) + qual)
        self.btn_lrc.setEnabled(bool(path) and not m.get("instrumental"))

    def _gen_lrc(self):
        """Genera il file LRC (karaoke) dall'ultimo brano, in un worker (Whisper CPU)."""
        if self._busy() or not self._last_audio:
            return
        if self._lrc_worker and self._lrc_worker.isRunning():
            return
        self.btn_lrc.setEnabled(False)
        self.status.setText(tr("music.lrc.run", "Genero le liriche sincronizzate (LRC)…"))

        class _LrcWorker(QThread):
            done = Signal(str); failed = Signal(str)

            def __init__(self, path):
                super().__init__(); self._p = path

            def run(self):
                try:
                    from core import lrc_service
                    self.done.emit(lrc_service.transcribe_to_lrc(self._p))
                except Exception as e:                        # noqa: BLE001
                    self.failed.emit(str(e))

        self._lrc_worker = _LrcWorker(self._last_audio)
        self._lrc_worker.done.connect(lambda p: (self.status.setText(
            tr("music.lrc.done", "LRC salvato: {f}").format(f=Path(p).name)), self.btn_lrc.setEnabled(True)))
        self._lrc_worker.failed.connect(lambda e: (self.status.setText(
            tr("music.lrc.err", "LRC non riuscito: {e}").format(e=e)), self.btn_lrc.setEnabled(True)))
        self._lrc_worker.start()

    def _use_region_as_task(self, start: float, end: float):
        """Regione selezionata sul waveform → task Repaint (ACE-Step) su quell'intervallo."""
        if self.song_engine.currentText() != "ACE-Step":
            self.song_engine.setCurrentText("ACE-Step")
        self.song_task.set_repaint(self._last_audio, start, end)
        self.status.setText(tr("music.region.set",
            "Regione [{a:.1f}–{b:.1f}s] impostata su Repaint (rigenera solo quella parte).").format(a=start, b=end))

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
        try:
            from core import acestep_service as ax
            a = ax.is_installed()
        except Exception:
            a = False
        self.btn_acestep_install.setEnabled(not a)
        self.btn_acestep_install.setText(tr("music.installed.acestep", "ACE-Step installato ✓") if a
                                         else tr("music.install.acestep", "Installa ACE-Step"))

    def _busy(self) -> bool:
        return bool((self._song_job and self._song_job.running) or (self._sonify_job and self._sonify_job.running)
                    or self._install_proc is not None)

    def _gen_song(self):
        if self._busy():
            return
        eng = self.song_engine.currentText()
        if eng == "ACE-Step":
            return self._gen_song_acestep()
        if eng == "ACE-Step 1.5":
            return self._gen_song_ace15()
        p = ms.SongParams(
            name=self.song_name.text().strip() or "song",
            lyrics=_expand_wildcards(self.song_lyrics.toPlainText()),      # wildcards {a|b|c}
            tags=self.song_meta.compose(self._style_tags(self.song_tags.text())),   # F-A: metadata + wildcards + enhance
            duration_s=self.song_dur.value(), instrumental=self.song_instr.isChecked(),
            version=self.song_ver.currentText(), temperature=self.song_temp.value(),
            cfg_scale=self.song_cfg.value(), lazy_load=self.song_lazy.isChecked())
        errs = ms.preflight_song(p)
        if errs:
            QMessageBox.warning(self, tr("music.err", "Impossibile generare"), "\n".join(errs)); return
        self._song_meta = {"label": "Musica", "engine": "HeartMuLa", "tags": p.tags, "lyrics": p.lyrics,
                           "duration_s": p.duration_s, "instrumental": p.instrumental, "version": p.version,
                           "temperature": p.temperature, "cfg": p.cfg_scale}   # #97 sidecar
        self._start_job(ms.make_song_job(p, on_log=self._b.log.emit, on_progress=self._b.progress.emit,
                                         on_done=self._b.song_done.emit), "song", p.out_mp3())

    def _pick_tags(self):
        """Apre la palette grafica dei tag di stile e applica la selezione al campo (punto 3)."""
        from ui.widgets.music_tags import MusicTagDialog
        dlg = MusicTagDialog(self.song_tags.text(), engine=self.song_engine.currentText(), parent=self)
        if dlg.exec():
            self.song_tags.setText(dlg.result_tags())

    def _style_tags(self, raw: str) -> str:
        """Espande i wildcard e, se 'Arricchisci stile' è attivo, trasforma i tag in caption ricca via LLM."""
        t = _expand_wildcards((raw or "").strip())
        if getattr(self, "song_enhance", None) and self.song_enhance.isChecked() and t:
            try:
                from core import prompt_enhance as pe
                if pe.available():
                    self.status.setText(tr("music.enhance.run", "Arricchimento stile (AI)…"))
                    t = pe.enhance(t, kind="music") or t
            except Exception:                                # noqa: BLE001
                pass
        return t

    def _insert_section(self, sec: str):
        """Inserisce il tag di sezione [sec] al cursore delle liriche (con a-capo)."""
        if self.song_lyrics.isEnabled():
            self.song_lyrics.insertPlainText(f"[{sec}]\n")
            self.song_lyrics.setFocus()

    def _random_example(self):
        """Riempie tag/liriche/metadata con un esempio di stile casuale (sempre diverso)."""
        ex = random.choice(_SONG_EXAMPLES)
        self.song_tags.setText(ex["tags"])
        instr = not ex.get("lyrics")
        self.song_instr.setChecked(instr)
        if not instr:
            self.song_lyrics.setPlainText(ex["lyrics"])
        self.song_meta.bpm.setValue(int(ex.get("bpm", 0)))
        if ex.get("key"):
            self.song_meta.key.setCurrentText(ex["key"])
        self.song_meta.setChecked(True)                      # apri il pannello metadata per mostrarli
        self.status.setText(tr("music.example.done", "Esempio caricato — modifica pure e genera."))

    def _train_lora(self, params: dict):
        """#98 F4: avvia il training di una LoRA ACE-Step dal dataset trascinato."""
        if self._busy():
            return
        from core import acestep_train_service as tx
        p = tx.TrainParams(**params)
        errs = tx.preflight(p)
        if errs:
            QMessageBox.warning(self, tr("music.train.err", "Impossibile allenare"), "\n".join(errs)); return
        n = len([a for a in p.audio_paths if Path(a).suffix.lower() in tx.AUDIO_EXTS])
        if QMessageBox.question(
                self, tr("music.train.confirm.t", "Avviare il training?"),
                tr("music.train.confirm.m", "Alleno la LoRA '{name}' su {n} file per {s} passi. "
                   "Può richiedere molto tempo. Continuare?").format(name=p.exp_name, n=n, s=p.max_steps)
                ) != QMessageBox.Yes:
            return
        job = tx.make_job(p, on_log=self._b.log.emit, on_progress=self._b.progress.emit,
                          on_done=self._b.song_done.emit)
        self._start_job(job, "train", str(tx.WORK_ROOT / tx._safe(p.exp_name)))

    def _reset_defaults(self):
        """Reimposta tutti i parametri del tab ai valori ottimali del modello selezionato."""
        e = self.song_engine.currentText()
        self.song_dur.setValue(120); self.song_instr.setChecked(False)
        self.song_temp.setValue(1.0); self.song_lazy.setChecked(False)
        self.song_adherence.setValue(3.0); self.song_gs_lyric.setValue(3.0)
        self.song_meta.reset_defaults()                      # F-A: metadata musicali → (auto)
        self._on_song_engine(e)                              # cfg/adherence/ranges + gating pannelli per engine
        self.song_adv.reset_defaults(); self.song_task.reset_defaults()
        self.song_adv.set_engine(e); self.song_task.set_engine(e)   # ri-applica il gating dopo il reset
        self.status.setText(tr("music.reset.done", "Parametri reimpostati ai default di {e}").format(e=e))

    # ── Preset (salva/carica configurazione di stile) ────────────────────────────
    def _presets_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / "data" / "music_presets.json"

    def _presets_load(self) -> dict:
        import json
        p = self._presets_path()
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except Exception:                                    # noqa: BLE001
            return {}

    def _presets_save(self, d: dict):
        import json
        p = self._presets_path(); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    def _refresh_presets(self, select: str = ""):
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem(tr("music.preset.none", "— preset —"), "")
        for name in sorted(self._presets_load()):
            self.preset_combo.addItem(name, name)
        if select:
            i = self.preset_combo.findData(select)
            if i >= 0:
                self.preset_combo.setCurrentIndex(i)
        self.preset_combo.blockSignals(False)

    def _get_config(self) -> dict:
        """Snapshot dei parametri di stile correnti (per preset/sidecar)."""
        return {
            "engine": self.song_engine.currentText(),
            "tags": self.song_tags.text(), "lyrics": self.song_lyrics.toPlainText(),
            "instrumental": self.song_instr.isChecked(), "duration": self.song_dur.value(),
            "version": self.song_ver.currentText(), "temp": self.song_temp.value(),
            "cfg": self.song_cfg.value(), "adherence": self.song_adherence.value(),
            "gs_lyric": self.song_gs_lyric.value(),
            "meta": self.song_meta.get_meta(), "adv": self.song_adv.get_adv(),
        }

    def _apply_config(self, cfg: dict):
        """Applica una configurazione ai controlli (da preset o sidecar). Robusto a chiavi mancanti."""
        if not cfg:
            return
        if cfg.get("engine"):
            self.song_engine.setCurrentText(cfg["engine"])   # _on_song_engine reimposta cfg/adherence/gs_lyric…
        self.song_tags.setText(cfg.get("tags", "") or "")
        self.song_lyrics.setPlainText(cfg.get("lyrics", "") or "")
        self.song_instr.setChecked(bool(cfg.get("instrumental")))
        if cfg.get("duration"):
            self.song_dur.setValue(int(cfg["duration"]))
        if cfg.get("version"):
            self.song_ver.setCurrentText(str(cfg["version"]))
        self.song_temp.setValue(float(cfg.get("temp", 1.0)))
        for key, w in (("cfg", self.song_cfg), ("adherence", self.song_adherence), ("gs_lyric", self.song_gs_lyric)):
            if cfg.get(key) is not None:
                w.setValue(float(cfg[key]))
        self.song_meta.set_meta(cfg.get("meta", {}))
        self.song_adv.set_config(cfg.get("adv", {}))
        self._update_estimate()

    def _save_preset(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, tr("music.preset.save", "Salva preset"),
                                        tr("music.preset.name", "Nome del preset:"))
        name = (name or "").strip()
        if not ok or not name:
            return
        d = self._presets_load(); d[name] = self._get_config(); self._presets_save(d)
        self._refresh_presets(select=name)
        self.status.setText(tr("music.preset.saved", "Preset '{n}' salvato.").format(n=name))

    def _load_preset(self, *_):
        name = self.preset_combo.currentData()
        if not name:
            return
        self._apply_config(self._presets_load().get(name, {}))
        self.status.setText(tr("music.preset.loaded", "Preset '{n}' caricato.").format(n=name))

    def _del_preset(self):
        name = self.preset_combo.currentData()
        if not name:
            return
        d = self._presets_load(); d.pop(name, None); self._presets_save(d)
        self._refresh_presets()
        self.status.setText(tr("music.preset.deleted", "Preset '{n}' eliminato.").format(n=name))

    def _apply_sidecar(self, p: str) -> bool:
        """Legge il sidecar del brano e applica i parametri ai controlli. True se qualcosa è stato applicato."""
        try:
            from core.metadata_saver import MetadataSaver
            sc = MetadataSaver.load_sidecar(p) or {}
        except Exception:                                    # noqa: BLE001
            sc = {}
        if not sc:
            return False
        _adv_keys = ("steps", "scheduler", "cfg_type", "omega", "guidance_interval",
                     "guidance_interval_decay", "min_guidance", "format", "seeds", "batch",
                     "quantized", "overlapped_decode", "torch_compile",
                     "no_erg_tag", "no_erg_lyric", "no_erg_diffusion")
        cfg = {
            "engine": sc.get("engine"), "tags": sc.get("tags"), "lyrics": sc.get("lyrics"),
            "instrumental": sc.get("instrumental"), "duration": sc.get("duration_s"),
            "cfg": sc.get("cfg"), "adherence": sc.get("gs_text"), "gs_lyric": sc.get("gs_lyric"),
            "meta": {k: sc[k] for k in ("bpm", "keyscale", "timesignature", "vocal_language") if k in sc},
            "adv": {k: sc[k] for k in _adv_keys if k in sc},
        }
        self._apply_config(cfg)
        return True

    def _load_from_song(self):
        """Recupera i parametri dal sidecar di un brano già generato (dialog Sfoglia)."""
        p, _ = QFileDialog.getOpenFileName(self, tr("music.preset.song", "Da brano…"), str(ms.MUSIC_OUTPUT_DIR),
                                           "Audio (*.wav *.mp3 *.flac *.ogg)")
        if not p:
            return
        if self._apply_sidecar(p):
            self.status.setText(tr("music.preset.fromsong", "Parametri caricati da {f}.").format(f=Path(p).name))
        else:
            QMessageBox.information(self, tr("music.preset.song", "Da brano…"),
                                    tr("music.preset.nosc", "Nessun parametro trovato per questo brano."))

    def _open_library(self):
        """Apre la libreria/cronologia; collega play → player e riuso → applica sidecar."""
        from ui.widgets.music_library import MusicLibraryDialog
        self._lib_dlg = MusicLibraryDialog(ms.MUSIC_OUTPUT_DIR, parent=self)   # non-modale
        self._lib_dlg.play_requested.connect(self._library_play)
        self._lib_dlg.reuse_requested.connect(self._apply_sidecar)
        self._lib_dlg.show()

    def _library_play(self, path: str):
        """Carica un brano scelto dalla libreria nel player della tab."""
        self._last_audio = path
        self.btn_play.setEnabled(True)
        if self.wave_player and Path(path).suffix.lower() in (".wav", ".mp3", ".flac", ".ogg"):
            self.wave_player.load(path)
        try:                                                 # info-only: NON applica i parametri ai controlli
            from core.metadata_saver import MetadataSaver
            self._song_meta = MetadataSaver.load_sidecar(path) or {"label": "Musica"}
        except Exception:                                    # noqa: BLE001
            self._song_meta = {"label": "Musica"}
        self._update_result_info(path)
        self.status.setText(tr("music.done", "Pronto: {f}").format(f=Path(path).name))

    def _on_song_engine(self, e: str):
        """Applica i default dei parametri per il MODELLO selezionato (apertura + cambio selezione)."""
        is_hm = (e == "HeartMuLa")
        is_v1 = (e == "ACE-Step")
        is_15 = (e == "ACE-Step 1.5")
        self.song_ver.setEnabled(is_hm)                      # 3B/7B solo HeartMuLa
        self.song_temp.setEnabled(is_hm)                     # temperatura solo HeartMuLa
        # Doppia guida (gs_text/gs_lyric): solo ACE-Step v1. 1.5 usa dcw + planner (think), non queste leve.
        self.song_adherence.setVisible(is_v1); self.lbl_adherence.setVisible(is_v1)
        if hasattr(self, "song_gs_lyric"):
            self.song_gs_lyric.setVisible(is_v1); self.lbl_gs_lyric.setVisible(is_v1)
        if hasattr(self, "song_think"):                      # planner 5Hz (think): solo 1.5
            self.song_think.setVisible(is_15)
        if hasattr(self, "song_adv"):
            self.song_adv.set_engine(e)                      # avanzate: abilitate solo per ACE-Step
        if hasattr(self, "song_task"):
            self.song_task.set_engine(e)                     # operazione/task: solo ACE-Step
        if hasattr(self, "song_trainer"):
            self.song_trainer.set_engine(e)                  # trainer LoRA: solo ACE-Step
        # CFG = aderenza al prompt. HeartMuLa (LM-CFG): 2.5 default, scala 1-8 (oltre → artefatti/ripetizioni).
        # ACE-Step (apg): ~15, scala 1-30. È l'UNICA leva di aderenza per HeartMuLa (non ha guidance_scale_text).
        if is_hm:
            self.song_cfg.setRange(1.0, 8.0); self.song_cfg.setValue(2.5)
            self.song_cfg.setToolTip(tr("music.cfg.hm.tt",
                "CFG = aderenza ai tag/liriche (unica leva di HeartMuLa). Più alto = più aderente; oltre ~4-5 "
                "rischia artefatti/ripetizioni. Per l'aderenza migliore usa il checkpoint RL."))
        elif is_15:
            self.song_cfg.setRange(1.0, 10.0); self.song_cfg.setValue(1.0)
            self.song_cfg.setToolTip(tr("music.cfg.15.tt",
                "ACE-Step 1.5 (turbo): guidance ~1.0 (il modello turbo lavora quasi senza CFG). "
                "L'aderenza viene soprattutto dal planner 5Hz (think) e da caption+metadata strutturati."))
        else:
            self.song_cfg.setRange(1.0, 30.0); self.song_cfg.setValue(15.0)
            self.song_cfg.setToolTip(tr("music.cfg.ace.tt",
                "CFG (apg) di ACE-Step. ~15 è lo sweet-spot. Per l'aderenza ai TAG usa anche 'Aderenza stile'."))

    def _gen_song_acestep(self):
        """Generazione canzone/strumentale con ACE-Step XL 1.5 (alternativa a HeartMuLa)."""
        from core import acestep_service as ax
        raw_tags = self._style_tags(self.song_tags.text())        # wildcards {a|b|c} + enhance opzionale
        instrumental = self.song_instr.isChecked()
        lyrics = "" if instrumental else _expand_wildcards(self.song_lyrics.toPlainText().strip())
        name = self.song_name.text().strip() or "song"
        errs = ax.preflight(raw_tags)
        if errs:
            QMessageBox.warning(self, tr("music.err", "Impossibile generare"), "\n".join(errs)); return
        # F-A: accoda i metadata strutturati (BPM/tonalità/metro/lingua) ai tag di stile, come ACE-Step 1.5.
        tags = self.song_meta.compose(raw_tags)
        adv = self.song_adv.get_adv()                        # #98: tutti i param avanzati ACE-Step (incl. steps)
        adv.update(self.song_task.get_task())                # #98 F3: task editing (retake/repaint/edit/…)
        steps = int(adv.get("steps", 60))                    # F-A: step ora regolabile (era hardcoded 60)
        # F-A: aderenza alle LIRICHE (Double Condition Guidance). Attiva SOLO se sia gs_text sia gs_lyric >1.
        # Con liriche vuote (strumentale) la disattiviamo (nessuna lirica da seguire).
        gs_lyric = 0.0 if (instrumental or not lyrics) else float(self.song_gs_lyric.value())
        ext = adv.get("format") or "wav"
        out = ax.OUT_DIR / f"{ax._safe(name)}.{ext}"
        # NB: ACE-Step usa una scala di guidance DIVERSA da HeartMuLa. Il selettore engine imposta
        # song_cfg al default corretto (~15) quando ACE-Step è attivo (_on_song_engine); passare 1.5
        # (default HeartMuLa) sotto-condizionerebbe il modello → RUMORE grigio.
        self._song_meta = {"label": "Musica", "engine": "ACE-Step", "tags": tags, "lyrics": lyrics,
                           "duration_s": float(self.song_dur.value()), "steps": steps,
                           "cfg": float(self.song_cfg.value()), "gs_text": float(self.song_adherence.value()),
                           "gs_lyric": gs_lyric, "instrumental": instrumental,
                           **self.song_meta.get_meta(), **adv}   # #97 sidecar
        job = ax.make_job(tags, lyrics, name, duration=float(self.song_dur.value()),
                          steps=steps, guidance=float(self.song_cfg.value()),
                          gs_text=float(self.song_adherence.value()), gs_lyric=gs_lyric,
                          lazy=self.song_lazy.isChecked(),
                          adv=adv, on_log=self._b.log.emit, on_progress=self._b.progress.emit,
                          on_done=self._b.song_done.emit)
        self._start_job(job, "song", str(out))

    def _gen_song_ace15(self):
        """Generazione con ACE-Step 1.5 (DiT turbo 4B + planner 5Hz). Metadata come parametri STRUTTURATI
        (non accodati alla caption, come nel v1) — è la separazione caption/lyrics/metadata della 1.5."""
        from core import acestep15_service as ax15
        caption = self._style_tags(self.song_tags.text())
        instrumental = self.song_instr.isChecked()
        lyrics = "" if instrumental else _expand_wildcards(self.song_lyrics.toPlainText().strip())
        name = self.song_name.text().strip() or "song"
        errs = ax15.preflight(caption)
        if errs:
            QMessageBox.warning(self, tr("music.err", "Impossibile generare"), "\n".join(errs)); return
        meta = self.song_meta.get_meta()                     # bpm/keyscale/timesignature/vocal_language separati
        task = self.song_task.get_task()                     # modalità editing 1.5 (cover/repaint/lego/extract/complete)
        fmt = self.song_adv.fmt.currentText() or "wav"
        job = ax15.make_job(
            caption, lyrics, name, instrumental=instrumental,
            bpm=int(meta.get("bpm", 0) or 0), keyscale=meta.get("keyscale", ""),
            timesignature=meta.get("timesignature", ""), language=meta.get("vocal_language", "unknown"),
            duration=float(self.song_dur.value()), steps=8, guidance=float(self.song_cfg.value()),
            seed=-1, think=self.song_think.isChecked(), fmt=fmt, batch=1, task=task,
            on_log=self._b.log.emit, on_progress=self._b.progress.emit, on_done=self._b.song_done.emit)
        self._song_meta = {"label": "Musica", "engine": "ACE-Step 1.5", "tags": caption, "lyrics": lyrics,
                           "duration_s": float(self.song_dur.value()), "steps": 8,
                           "cfg": float(self.song_cfg.value()), "think": self.song_think.isChecked(),
                           "instrumental": instrumental, **meta, **task}   # #97 sidecar
        out = ax15.OUT_DIR / f"{ax15._safe(name)}.{fmt}"
        self._start_job(job, "song", str(out))

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
        """Crea un PROGETTO videomusicale (persistente) dai campi correnti e apre il Music Video Studio
        alla fase Canzone (genera/rigenera, lista canzoni, poi storyboard editabile)."""
        lyrics = self.song_lyrics.toPlainText().strip()
        if not lyrics:
            QMessageBox.warning(self, tr("music.mv.gen", "Videomusicale"),
                               tr("music.mv.need_lyrics", "Inserisci le liriche (con sezioni [Verse]/[Chorus]).")); return
        from core import mv_projects as mvp
        image = self.mv_image.text().strip()
        pid = mvp.create_project(
            self.song_name.text().strip() or "videomusicale",
            lyrics=lyrics, tags=self.song_tags.text().strip() or ms.DEFAULT_TAGS,
            image_hint=("coerente con l'immagine: " + image) if image else "",
            scene_seconds=float(self.mv_pace.currentData()), video_style=self.mv_style.text().strip(),
            song_duration=float(self.song_dur.value()), base_image=image if image and Path(image).exists() else "")
        self._open_mv_studio(pid)

    def _open_mv_projects(self):
        """Apre il dialog CRUD dei progetti videomusicali; all'apertura di uno, lancia lo Studio."""
        from ui.mv_projects_dialog import MvProjectsDialog
        dlg = MvProjectsDialog(self.window())
        if dlg.exec() and dlg.opened_id:
            self._open_mv_studio(dlg.opened_id)

    def _open_mv_studio(self, pid: int):
        from ui.music_video_studio import MusicVideoStudio
        self._mv_studio = MusicVideoStudio(pid, parent=self.window())
        self._mv_studio.setWindowFlag(Qt.Window, True)
        self._mv_studio.show(); self._mv_studio.raise_(); self._mv_studio.activateWindow()

    def son_name(self) -> str:
        v = self.son_video.text().strip()
        if v:
            return Path(v).stem
        # text-to-audio (nessun video): nome derivato dal prompt per non sovrascrivere "sfx" ad ogni run
        import re
        slug = re.sub(r"[^\w]+", "_", (self.son_prompt.text() or "").strip()).strip("_")[:40]
        return slug or "sfx"

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
        scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
        # ACE-Step ha un installer dedicato; HeartMuLa/MMAudio passano dal flag di install_music.py
        if which == "--acestep":
            args = ["-u", str(scripts_dir / "install_acestep.py")]
        else:
            args = ["-u", str(scripts_dir / "install_music.py"), which]
        self._log(f"[install] {which} …")
        self.btn_song_install.setEnabled(False); self.btn_son_install.setEnabled(False)
        self.btn_acestep_install.setEnabled(False)
        p = QProcess(self); p.setProcessChannelMode(QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(
            lambda: self._log(bytes(p.readAllStandardOutput()).decode("utf-8", "replace").rstrip()))
        p.finished.connect(self._install_finished)
        self._install_proc = p
        p.start(sys.executable, args)

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
        if kind == "train":                                  # #98 F4: il training non produce audio
            self.status.setText(tr("music.train.done", "LoRA allenata ✓ ({p})").format(p=self._expected_out))
            self._log(f"[train] LoRA in: {self._expected_out}"); return
        out = self._expected_out if (self._expected_out and Path(self._expected_out).exists()) else None
        if not out:
            found = ms.find_latest_audio(after=self._t0 - 1)
            out = str(found) if found else ""
        if out:
            self._last_audio = out
            self.btn_play.setEnabled(True)
            if self.wave_player and Path(out).suffix.lower() in (".wav", ".mp3", ".flac", ".ogg"):
                self.wave_player.load(out)                   # forma d'onda + player in-app
            self._update_result_info(out)                    # riepilogo + qualità + abilita LRC
            self.status.setText(tr("music.done", "Pronto: {f}").format(f=Path(out).name))
            self._log(f"[music] output: {out}")
            try:                                             # #97: sidecar (params riutilizzabili nel Navigator)
                from core.metadata_saver import MetadataSaver
                MetadataSaver.save_sidecar(out, getattr(self, "_song_meta", {"label": "Musica"}))
            except Exception:                                # noqa: BLE001
                pass
            self.music_ready.emit(out)
        else:
            self.status.setText(tr("music.done.nofile", "Completato (file non individuato)"))

    def _play(self):
        if self._last_audio and os.path.exists(self._last_audio):
            os.startfile(self._last_audio)             # player di sistema

    def _open_folder(self):
        ms.MUSIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(str(ms.MUSIC_OUTPUT_DIR))

    def _process_audio(self):
        """Post-produzione audio (ffmpeg) sull'ultimo brano o su un file scelto."""
        if self._busy():
            return
        from core import audio_post as ap
        src = self.song_post.get_src() or self._last_audio
        opts = self.song_post.get_opts()
        errs = ap.preflight(src, opts)
        if errs:
            QMessageBox.warning(self, tr("post.err", "Impossibile elaborare"), "\n".join(errs)); return
        name = self.song_name.text().strip() or "song"
        fmt = opts.get("format") or Path(src).suffix.lstrip(".")
        out = ap.out_path(name, fmt)
        self._song_meta = {"label": "Audio post", "source": Path(src).name, **opts}   # #97 sidecar
        job = ap.make_job(src, name, opts, on_log=self._b.log.emit,
                          on_progress=self._b.progress.emit, on_done=self._b.song_done.emit)
        self._start_job(job, "song", str(out))

    def _process_pitch(self):
        """Correzione intonazione voce (DiffPitcher). Scaffold: da validare live."""
        if self._busy():
            return
        from core import diffpitcher_service as dp
        p = self.pitch_panel.get()
        errs = dp.preflight(p["source"], p["pitch_ref"])
        if errs:
            QMessageBox.warning(self, tr("pitch.err", "Impossibile correggere"), "\n".join(errs)); return
        name = self.song_name.text().strip() or "vocal"
        out = dp.out_path(name)
        self._song_meta = {"label": "Intonazione", "source": Path(p["source"]).name,
                           "pitch_ref": Path(p["pitch_ref"]).name, "steps": p["steps"], "shift_semi": p["shift_semi"]}
        job = dp.make_job(p["source"], p["pitch_ref"], name, steps=p["steps"], shift_semi=p["shift_semi"],
                          on_log=self._b.log.emit, on_progress=self._b.progress.emit, on_done=self._b.song_done.emit)
        self._start_job(job, "song", str(out))

    def _open_spectro(self):
        """Apre il confronto spettrogramma A/B (A = ultimo brano generato, se presente)."""
        from ui.widgets.spectrogram import SpectrogramCompare
        a = self._last_audio if (self._last_audio and os.path.exists(self._last_audio)) else ""
        self._spectro_dlg = SpectrogramCompare(a, parent=self)     # non-modale
        self._spectro_dlg.show()
