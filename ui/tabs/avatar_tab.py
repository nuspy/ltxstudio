"""Tab Avatar — foto (o solo testo) + voce → video di una persona che parla.

Motore: LongCat-Video-Avatar 1.5 (core/longcat_service, venv dedicato). La DURATA la decide l'audio:
il tab mostra quanti segmenti servono e il video viene tagliato sulla voce.

Immagine iniziale: «Usa immagine da canvas» (attivo solo se il canvas ha una foto) oppure Sfoglia,
che carica il file ANCHE nel canvas. Audio: «Usa ultimo audio generato» (TTS del tab Audio/Voce, poi
musica, poi il file più recente in output/tts|audio) oppure Sfoglia; si ascolta col player
(⏪ ▶/⏸ ⏹ ⏩) e si può selezionare la regione da usare. Il pannello «Genera voce» crea la voce qui
dentro (1 o 2 persone) con la stessa catena usata dall'API (core/avatar/audio_chain).
"""
from __future__ import annotations

import os
import random
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QThread, QUrl, QRect, QPoint
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel, QComboBox, QSpinBox,
    QDoubleSpinBox, QPushButton, QLineEdit, QCheckBox, QFileDialog, QScrollArea, QSplitter,
    QPlainTextEdit, QRadioButton, QButtonGroup, QSlider, QDialog, QDialogButtonBox, QMessageBox,
)

from ui.i18n_manager import tr

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "output"
IMG_FILTER_FALLBACK = "Immagini (*.png *.jpg *.jpeg *.webp *.bmp)"
AUDIO_FILTER_FALLBACK = "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.aac)"
LANGS = ["it-IT", "en-US", "es-ES", "fr-FR", "de-DE", "pt-BR", "ru-RU", "zh-CN", "ja-JP", "ko-KR", "ar-EG"]
EMOTIONS = ["neutral", "happy", "sad", "angry", "whisper", "calm"]
SUGGESTED_VOICES = {
    "edge": ["it-IT-ElsaNeural", "it-IT-IsabellaNeural", "it-IT-DiegoNeural", "en-US-AriaNeural",
             "en-US-GuyNeural"],
    "kokoro": ["if_sara", "im_nicola"],
}


def _hint(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color:#9aa1b0; font-size:11px;")
    return lbl


def _collapsible(group: QGroupBox) -> QWidget:
    """Gruppo spuntabile che NASCONDE il contenuto quando è spento (invece di lasciarlo grigio)."""
    inner = QWidget()
    lay = QVBoxLayout(group); lay.setContentsMargins(6, 4, 6, 4); lay.addWidget(inner)
    group.toggled.connect(inner.setVisible)
    inner.setVisible(group.isChecked())
    return inner


def _latest_audio_file() -> str:
    """File audio più recente fra output/tts e output/audio (ripiego di «ultimo audio generato»)."""
    best, best_t = "", 0.0
    for d in (OUTPUT / "tts", OUTPUT / "audio"):
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a") and f.stat().st_mtime > best_t:
                best, best_t = str(f), f.stat().st_mtime
    return best


# ── riquadri delle due persone ────────────────────────────────────────────────────────────────────
class _BoxCanvas(QLabel):
    """Foto scalata dove si trascinano due rettangoli (persona 1 = verde, persona 2 = azzurro)."""

    COLORS = (QColor("#3ecf8e"), QColor("#74c0fc"))

    def __init__(self, image_path: str, boxes: list | None = None):
        super().__init__()
        self._pix = QPixmap(image_path)
        self._scaled = self._pix.scaled(720, 520, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(self._scaled)
        self.setFixedSize(self._scaled.size())
        self.boxes: list[QRect | None] = [None, None]
        self.active = 0
        self._start: QPoint | None = None
        for i, b in enumerate(boxes or []):
            if b and i < 2:
                y0, x0, y1, x1 = b
                w, h = self._scaled.width(), self._scaled.height()
                self.boxes[i] = QRect(QPoint(int(x0 * w), int(y0 * h)), QPoint(int(x1 * w), int(y1 * h)))

    def mousePressEvent(self, e):
        self._start = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self._start is not None:
            self.boxes[self.active] = QRect(self._start, e.position().toPoint()).normalized()
            self.update()

    def mouseReleaseEvent(self, e):
        self._start = None
        if self.boxes[self.active] is not None and self.active == 0:
            self.active = 1

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        for i, r in enumerate(self.boxes):
            if r is not None:
                p.setPen(QPen(self.COLORS[i], 3))
                p.drawRect(r)
                p.drawText(r.topLeft() + QPoint(6, 16), f"{i + 1}")
        p.end()

    def fractions(self) -> list | None:
        if not all(self.boxes):
            return None
        w, h = self._scaled.width(), self._scaled.height()
        out = []
        for r in self.boxes:
            out.append([max(0.0, r.top() / h), max(0.0, r.left() / w),
                        min(1.0, r.bottom() / h), min(1.0, r.right() / w)])
        return out


class _BoxDialog(QDialog):
    def __init__(self, parent, image_path: str, boxes: list | None):
        super().__init__(parent)
        self.setWindowTitle(tr("avatar.boxes.title", "Riquadri delle due persone"))
        v = QVBoxLayout(self)
        v.addWidget(_hint(tr("avatar.boxes.hint",
                             "Trascina un rettangolo sul volto/busto della PERSONA 1 (verde), poi uno sulla "
                             "PERSONA 2 (azzurro). La persona 1 parla con l'audio 1, la 2 con l'audio 2.")))
        self.canvas = _BoxCanvas(image_path, boxes)
        v.addWidget(self.canvas, 0, Qt.AlignCenter)
        row = QHBoxLayout()
        b1 = QPushButton(tr("avatar.boxes.redo1", "Ridisegna persona 1"))
        b1.clicked.connect(lambda: self._redo(0))
        b2 = QPushButton(tr("avatar.boxes.redo2", "Ridisegna persona 2"))
        b2.clicked.connect(lambda: self._redo(1))
        row.addWidget(b1); row.addWidget(b2); row.addStretch()
        v.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _redo(self, i: int):
        self.canvas.boxes[i] = None
        self.canvas.active = i
        self.canvas.update()


# ── pannello voce di una persona ──────────────────────────────────────────────────────────────────
class _SpeakerBox(QGroupBox):
    def __init__(self, title: str, name: str):
        super().__init__(title)
        f = QFormLayout(self); f.setLabelAlignment(Qt.AlignRight)
        self.name = QLineEdit(name)
        self.name.setToolTip(tr("avatar.voice.name.tt",
                                "Nome usato nel copione: righe «Nome: battuta» (vanno bene anche A: e B:)."))
        f.addRow(tr("avatar.voice.name", "Nome nel copione"), self.name)
        self.engine = QComboBox()
        try:
            from core import tts_engine as T
            for eid, disp, *_ in T.ENGINES_META:
                self.engine.addItem(disp, eid)
        except Exception:                                            # noqa: BLE001
            for eid in ("edge", "kokoro"):
                self.engine.addItem(eid, eid)
        self.engine.setToolTip(tr("avatar.voice.engine.tt",
                                  "Motore di sintesi. Edge e Kokoro non clonano; F5/XTTS/Fish/VibeVoice/"
                                  "CosyVoice/Handy-clone clonano una voce da un campione."))
        self.engine.currentIndexChanged.connect(self._refresh_voices)
        f.addRow(tr("avatar.voice.engine", "Motore voce"), self.engine)
        self.voice = QComboBox(); self.voice.setEditable(True)
        self.voice.setToolTip(tr("avatar.voice.voice.tt",
                                 "Voce della libreria (tab Audio/Voce) oppure id del motore "
                                 "(es. it-IT-ElsaNeural, if_sara). Vuoto = voce predefinita."))
        f.addRow(tr("avatar.voice.voice", "Voce"), self.voice)
        self.description = QLineEdit()
        self.description.setPlaceholderText(tr("avatar.voice.desc.ph", "es. uomo anziano, voce calda e bassa"))
        self.description.setToolTip(tr("avatar.voice.desc.tt",
                                       "Se la voce è vuota, l'LLM sceglie una voce adatta a questa descrizione."))
        f.addRow(tr("avatar.voice.desc", "Voce da descrizione"), self.description)
        crow = QHBoxLayout()
        self.clone = QLineEdit(); self.clone.setPlaceholderText(tr("avatar.voice.clone.ph", "campione WAV (clone)"))
        cb = QPushButton(tr("avatar.browse", "Sfoglia…")); cb.clicked.connect(self._pick_clone)
        crow.addWidget(self.clone, 1); crow.addWidget(cb)
        f.addRow(tr("avatar.voice.clone", "Clona da campione"), crow)
        self.transcript = QLineEdit()
        self.transcript.setToolTip(tr("avatar.voice.transcript.tt",
                                      "Trascrizione esatta del campione: migliora molto la clonazione."))
        f.addRow(tr("avatar.voice.transcript", "Trascrizione campione"), self.transcript)
        self.language = QComboBox(); self.language.addItems(LANGS)
        f.addRow(tr("avatar.voice.language", "Lingua"), self.language)
        self.speed = QDoubleSpinBox(); self.speed.setRange(0.5, 2.0); self.speed.setSingleStep(0.05)
        self.speed.setValue(1.0)
        f.addRow(tr("avatar.voice.speed", "Velocità"), self.speed)
        self.pitch = QDoubleSpinBox(); self.pitch.setRange(-12.0, 12.0); self.pitch.setSingleStep(0.5)
        self.pitch.setSuffix(" st")
        self.pitch.setToolTip(tr("avatar.voice.pitch.tt", "Tono in semitoni (0 = invariato)."))
        f.addRow(tr("avatar.voice.pitch", "Tono"), self.pitch)
        self.volume = QDoubleSpinBox(); self.volume.setRange(-20.0, 12.0); self.volume.setSuffix(" dB")
        f.addRow(tr("avatar.voice.volume", "Volume"), self.volume)
        self.emotion = QComboBox()
        for e in EMOTIONS:
            self.emotion.addItem(tr(f"avatar.emotion.{e}", e), e)
        self.emotion.setToolTip(tr("avatar.voice.emotion.tt",
                                   "Emozione: nativa con Edge; con gli altri motori è un'indicazione di stile."))
        f.addRow(tr("avatar.voice.emotion", "Emozione"), self.emotion)
        self._refresh_voices()

    def _refresh_voices(self):
        eid = self.engine.currentData()
        cur = self.voice.currentText()
        self.voice.clear()
        self.voice.addItem("")
        try:
            from core import tts_engine as T
            for v in T.list_user_voices():
                if not eid or v.engine == eid:
                    self.voice.addItem(v.name)
        except Exception:                                            # noqa: BLE001
            pass
        for s in SUGGESTED_VOICES.get(eid, []):
            self.voice.addItem(s)
        if cur:
            self.voice.setCurrentText(cur)

    def _pick_clone(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("avatar.voice.clone", "Clona da campione"), "",
                                           tr("avatar.audio.filter", AUDIO_FILTER_FALLBACK))
        if p:
            self.clone.setText(p)

    def get(self) -> dict:
        return {"name": self.name.text().strip() or "A", "engine": self.engine.currentData() or "edge",
                "voice_id": self.voice.currentText().strip(),
                "voice_description": self.description.text().strip(),
                "clone": ({"sample_path": self.clone.text().strip(), "transcript": self.transcript.text().strip()}
                          if self.clone.text().strip() else None),
                "language": self.language.currentText(), "speed": self.speed.value(),
                "pitch": self.pitch.value(), "volume_db": self.volume.value(),
                "emotion": self.emotion.currentData() or ""}


class _VoiceWorker(QThread):
    done = Signal(dict)

    def __init__(self, text: str, speakers: list[dict], gap_ms: int, pad_start: int, pad_end: int):
        super().__init__()
        self.args = (text, speakers, gap_ms, pad_start, pad_end)

    def run(self):
        try:
            from core.avatar import audio_chain as A
            text, speakers, gap, ps, pe = self.args
            res = A.build_speech(text, [A.Speaker.from_dict(s) for s in speakers],
                                 gap_ms=gap, pad_start_ms=ps, pad_end_ms=pe)
            self.done.emit({"status": "success", "path": res.path, "tracks": res.tracks,
                            "duration": res.duration_s})
        except Exception as e:                                       # noqa: BLE001
            self.done.emit({"status": "error", "error": str(e)})


# ── audio di una persona (percorso + player) ──────────────────────────────────────────────────────
class _AudioBox(QGroupBox):
    changed = Signal()

    def __init__(self, title: str, last_audio_getter):
        super().__init__(title)
        self._last = last_audio_getter
        v = QVBoxLayout(self)
        row = QHBoxLayout()
        self.path = QLineEdit(); self.path.setReadOnly(True)
        self.path.setPlaceholderText(tr("avatar.audio.ph", "Nessun audio scelto"))
        row.addWidget(self.path, 1)
        self.btn_last = QPushButton(tr("avatar.audio.last", "Usa ultimo audio generato"))
        self.btn_last.setToolTip(tr("avatar.audio.last.tt",
                                    "Prende l'ultima voce del tab Audio/Voce (o l'ultima musica, o il file "
                                    "più recente in output/tts e output/audio)."))
        self.btn_last.clicked.connect(self._use_last)
        self.btn_browse = QPushButton(tr("avatar.browse", "Sfoglia…"))
        self.btn_browse.clicked.connect(self._browse)
        row.addWidget(self.btn_last); row.addWidget(self.btn_browse)
        v.addLayout(row)
        self.player = None
        try:
            from ui.widgets.waveform import WaveformPlayer
            self.player = WaveformPlayer(region_button=False)
            self.player.setMinimumHeight(110)
            self.player.region_changed.connect(lambda *_: self.changed.emit())
            v.addWidget(self.player)
        except Exception:                                            # noqa: BLE001 (QtMultimedia assente)
            v.addWidget(_hint(tr("avatar.audio.no_player", "Player audio non disponibile (QtMultimedia).")))

    def set_path(self, p: str):
        self.path.setText(p or "")
        if self.player is not None and p and os.path.isfile(p):
            self.player.load(p)
        self.changed.emit()

    def value(self) -> str:
        return self.path.text().strip()

    # gui_state: salva/ripristina il file e lo ricarica nel player (non solo il testo del campo)
    def gui_get_state(self) -> dict:
        return {"path": self.value()}

    def gui_set_state(self, st):
        p = (st or {}).get("path", "") if isinstance(st, dict) else ""
        if p and os.path.isfile(p):
            self.set_path(p)

    def region(self):
        return self.player.region() if self.player is not None else None

    def _use_last(self):
        p = self._last() if self._last else ""
        if not p:
            QMessageBox.information(self, tr("avatar.audio.group", "Audio"),
                                    tr("avatar.audio.none_generated",
                                       "Nessun audio generato trovato: genera una voce nel tab Audio/Voce o "
                                       "nel pannello «Genera voce», oppure scegli un file."))
            return
        self.set_path(p)

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("avatar.audio.pick", "Scegli audio"), "",
                                           tr("avatar.audio.filter", AUDIO_FILTER_FALLBACK))
        if p:
            self.set_path(p)


class AvatarTab(QWidget):
    request_avatar = Signal(dict)
    load_into_canvas = Signal(str)       # MainWindow → canvas.load_image(path, fit_mode="canvas")

    def __init__(self):
        super().__init__()
        self.get_canvas_image = lambda: ""     # iniettato da MainWindow
        self._last_generated_audio = ""
        self._bboxes: list | None = None
        self._voice_worker = None
        self._build()
        self.on_canvas_image_changed(self.get_canvas_image())

    # ── UI ──────────────────────────────────────────────────────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        box = QWidget(); v = QVBoxLayout(box); v.setSpacing(8)

        # motore / modalità
        g = QGroupBox(tr("avatar.engine.group", "Motore e modalità")); f = QFormLayout(g)
        self.engine = QComboBox()
        self.engine.addItem(tr("avatar.engine.longcat", "LongCat Avatar 1.5 (audio dato, durata dall'audio)"),
                            "longcat")
        f.addRow(tr("avatar.engine", "Motore"), self.engine)
        mrow = QHBoxLayout()
        self.mode_img = QRadioButton(tr("avatar.mode.ai2v", "Da immagine"))
        self.mode_txt = QRadioButton(tr("avatar.mode.at2v", "Solo testo"))
        self.mode_img.setToolTip(tr("avatar.mode.ai2v.tt", "Anima la foto iniziale sulla voce."))
        self.mode_txt.setToolTip(tr("avatar.mode.at2v.tt",
                                    "Nessuna foto: la persona e la scena nascono dal prompt video."))
        self.mode_img.setChecked(True)
        self._mode_grp = QButtonGroup(self)
        self._mode_grp.addButton(self.mode_img); self._mode_grp.addButton(self.mode_txt)
        mrow.addWidget(self.mode_img); mrow.addWidget(self.mode_txt); mrow.addStretch()
        f.addRow(tr("avatar.mode", "Partenza"), mrow)
        self.people = QComboBox()
        self.people.addItem(tr("avatar.people.one", "Una persona"), 1)
        self.people.addItem(tr("avatar.people.two", "Due persone (dialogo)"), 2)
        f.addRow(tr("avatar.people", "Persone"), self.people)
        v.addWidget(g)

        # immagine
        self.img_group = QGroupBox(tr("avatar.image.group", "Immagine iniziale"))
        iv = QVBoxLayout(self.img_group)
        irow = QHBoxLayout()
        self.img_thumb = QLabel(); self.img_thumb.setFixedSize(128, 128)
        self.img_thumb.setAlignment(Qt.AlignCenter)
        self.img_thumb.setStyleSheet("border:1px solid #2f3441; border-radius:6px; color:#6b7280;")
        self.img_thumb.setText(tr("avatar.image.none", "nessuna"))
        irow.addWidget(self.img_thumb)
        ib = QVBoxLayout()
        self.img_path = QLineEdit(); self.img_path.setReadOnly(True)
        self.img_path.setPlaceholderText(tr("avatar.image.ph", "Nessuna immagine scelta"))
        self.img_path.textChanged.connect(self._refresh_thumb)
        ib.addWidget(self.img_path)
        self.btn_canvas = QPushButton(tr("avatar.image.canvas", "Usa immagine da canvas"))
        self.btn_canvas.setToolTip(tr("avatar.image.canvas.tt",
                                      "Usa la foto caricata nel canvas del tab Immagini. Attivo solo se il "
                                      "canvas contiene un'immagine."))
        self.btn_canvas.clicked.connect(self._use_canvas)
        self.btn_img_browse = QPushButton(tr("avatar.browse", "Sfoglia…"))
        self.btn_img_browse.setToolTip(tr("avatar.image.browse.tt",
                                          "Scegli un file: viene caricato anche nel canvas."))
        self.btn_img_browse.clicked.connect(self._browse_image)
        brow = QHBoxLayout(); brow.addWidget(self.btn_canvas); brow.addWidget(self.btn_img_browse)
        ib.addLayout(brow)
        self.btn_boxes = QPushButton(tr("avatar.boxes.btn", "Disegna riquadri delle persone"))
        self.btn_boxes.setToolTip(tr("avatar.boxes.btn.tt",
                                     "Due persone: indica chi è la persona 1 e chi la 2. Senza riquadri si "
                                     "usano metà sinistra (1) e metà destra (2)."))
        self.btn_boxes.clicked.connect(self._draw_boxes)
        self.lbl_boxes = QLabel("")
        self.lbl_boxes.setStyleSheet("color:#74c0fc; font-size:11px;")
        ib.addWidget(self.btn_boxes); ib.addWidget(self.lbl_boxes)
        irow.addLayout(ib, 1)
        iv.addLayout(irow)
        v.addWidget(self.img_group)

        # audio
        self.audio1 = _AudioBox(tr("avatar.audio.p1", "Audio — persona 1"), self._last_audio)
        self.audio2 = _AudioBox(tr("avatar.audio.p2", "Audio — persona 2"), self._last_audio)
        self.audio1.changed.connect(self._update_duration)
        self.audio2.changed.connect(self._update_duration)
        v.addWidget(self.audio1)
        v.addWidget(self.audio2)
        trow = QHBoxLayout()
        self.turns = QComboBox()
        self.turns.addItem(tr("avatar.turns.add", "A turno (prima 1, poi 2)"), "add")
        self.turns.addItem(tr("avatar.turns.para", "Insieme / tracce allineate"), "para")
        self.turns.setToolTip(tr("avatar.turns.tt",
                                 "A turno: gli audio vengono messi uno dopo l'altro. Insieme: le due tracce "
                                 "suonano in parallelo (usalo per i dialoghi creati da «Genera voce», dove "
                                 "ognuno tace mentre parla l'altro)."))
        self.lbl_turns = QLabel(tr("avatar.turns", "Come parlano"))
        trow.addWidget(self.lbl_turns); trow.addWidget(self.turns, 1)
        v.addLayout(trow)
        self.use_region = QCheckBox(tr("avatar.audio.use_region", "Usa solo la regione selezionata sulla forma d'onda"))
        v.addWidget(self.use_region)
        self.lbl_duration = QLabel(tr("avatar.duration.none", "Durata: scegli un audio"))
        self.lbl_duration.setStyleSheet("color:#3ecf8e; font-weight:bold;")
        v.addWidget(self.lbl_duration)

        # genera voce
        self.voice_group = QGroupBox(tr("avatar.voice.group", "Genera voce (TTS / clone / dialogo)"))
        self.voice_group.setCheckable(True); self.voice_group.setChecked(False)
        vv = QVBoxLayout(_collapsible(self.voice_group)); vv.setContentsMargins(0, 0, 0, 0)
        self.voice_text = QPlainTextEdit()
        self.voice_text.setPlaceholderText(tr(
            "avatar.voice.text.ph",
            "Una persona: scrivi il testo.\nDue persone: un copione, es.\nA: Ciao! Hai visto che luna stasera?\n"
            "B: Enorme. Sembra un formaggio appeso al cielo.\nPause: [pause:500ms]"))
        self.voice_text.setFixedHeight(110)
        vv.addWidget(self.voice_text)
        self.spk1 = _SpeakerBox(tr("avatar.voice.spk1", "Voce persona 1"), "A")
        self.spk2 = _SpeakerBox(tr("avatar.voice.spk2", "Voce persona 2"), "B")
        vv.addWidget(self.spk1); vv.addWidget(self.spk2)
        pf = QFormLayout()
        self.gap_ms = QSpinBox(); self.gap_ms.setRange(0, 3000); self.gap_ms.setSingleStep(50)
        self.gap_ms.setValue(350); self.gap_ms.setSuffix(" ms")
        pf.addRow(tr("avatar.voice.gap", "Pausa fra le battute"), self.gap_ms)
        self.pad_start = QSpinBox(); self.pad_start.setRange(0, 5000); self.pad_start.setSuffix(" ms")
        self.pad_start.setToolTip(tr("avatar.voice.pad.tt",
                                     "Silenzio prima/dopo: la persona resta viva ma in silenzio."))
        pf.addRow(tr("avatar.voice.pad_start", "Silenzio iniziale"), self.pad_start)
        self.pad_end = QSpinBox(); self.pad_end.setRange(0, 5000); self.pad_end.setSuffix(" ms")
        self.pad_end.setToolTip(self.pad_start.toolTip())
        pf.addRow(tr("avatar.voice.pad_end", "Silenzio finale"), self.pad_end)
        vv.addLayout(pf)
        self.btn_voice = QPushButton(tr("avatar.voice.run", "🎙 Genera voce"))
        self.btn_voice.clicked.connect(self._generate_voice)
        vv.addWidget(self.btn_voice)
        v.addWidget(self.voice_group)

        # video
        g = QGroupBox(tr("avatar.video.group", "Video")); f = QFormLayout(g); f.setLabelAlignment(Qt.AlignRight)
        prow = QVBoxLayout()
        self.prompt = QPlainTextEdit(); self.prompt.setFixedHeight(90)
        self.prompt.setPlaceholderText(tr(
            "avatar.prompt.ph",
            "Descrivi persona, azione e scena: più dettagli = più naturale. Es.: una donna con i capelli "
            "ricci rossi parla sorridendo, maglione verde, seduta in una libreria piena di luce calda."))
        prow.addWidget(self.prompt)
        self.btn_enhance = QPushButton(tr("avatar.prompt.enhance", "✨ Migliora prompt"))
        self.btn_enhance.setToolTip(tr("avatar.prompt.enhance.tt",
                                       "Arricchisce il prompt con l'LLM locale (il modello rende meglio con "
                                       "descrizioni lunghe)."))
        self.btn_enhance.clicked.connect(self._enhance_prompt)
        prow.addWidget(self.btn_enhance, 0, Qt.AlignRight)
        f.addRow(tr("avatar.prompt", "Prompt video"), prow)
        self.resolution = QComboBox()
        self.resolution.addItem("480p (832×480)", "480p")
        self.resolution.addItem("720p (1280×768)", "720p")
        self.resolution.setToolTip(tr("avatar.resolution.tt",
                                      "Con una foto, le proporzioni seguono la foto (anche verticale)."))
        f.addRow(tr("avatar.resolution", "Risoluzione"), self.resolution)
        self.aspect = QComboBox(); self.aspect.addItems(["16:9", "9:16", "1:1", "4:3", "3:4"])
        self.lbl_aspect = QLabel(tr("avatar.aspect", "Formato"))
        f.addRow(self.lbl_aspect, self.aspect)
        self.resize_mode = QComboBox()
        self.resize_mode.addItem(tr("avatar.resize.crop", "Ritaglia"), "crop")
        self.resize_mode.addItem(tr("avatar.resize.fit", "Adatta (senza ritaglio)"), "default")
        f.addRow(tr("avatar.resize", "Adattamento foto"), self.resize_mode)
        self.max_dur = QDoubleSpinBox(); self.max_dur.setRange(0.0, 3600.0); self.max_dur.setSuffix(" s")
        self.max_dur.setToolTip(tr("avatar.maxdur.tt", "0 = usa tutto l'audio."))
        self.max_dur.valueChanged.connect(self._update_duration)
        f.addRow(tr("avatar.maxdur", "Limita a"), self.max_dur)
        srow = QHBoxLayout()
        self.seed = QSpinBox(); self.seed.setRange(0, 2**31 - 1); self.seed.setValue(42)
        self.seed_random = QCheckBox(tr("avatar.seed.random", "Casuale")); self.seed_random.setChecked(True)
        self.seed_random.toggled.connect(lambda c: self.seed.setEnabled(not c)); self.seed.setEnabled(False)
        srow.addWidget(self.seed, 1); srow.addWidget(self.seed_random)
        f.addRow(tr("avatar.seed", "Seed"), srow)
        v.addWidget(g)

        # avanzate
        self.adv = QGroupBox(tr("avatar.adv.group", "Avanzate")); self.adv.setCheckable(True)
        self.adv.setChecked(False)
        af = QFormLayout(_collapsible(self.adv)); af.setLabelAlignment(Qt.AlignRight)
        self.precision = QComboBox()
        for pid, label in (("int8", tr("avatar.precision.int8", "INT8 ufficiale (15.9GB, ogni GPU)")),
                           ("fp8", tr("avatar.precision.fp8", "FP8 e4m3 (15.9GB, RTX 40xx/50xx)")),
                           ("bf16", tr("avatar.precision.bf16", "bf16 (31.7GB, solo confronti)"))):
            self.precision.addItem(label, pid)
        af.addRow(tr("avatar.precision", "Precisione modello"), self.precision)
        self.negative = QPlainTextEdit(); self.negative.setFixedHeight(70)
        try:
            from workers.longcat_avatar_infer import NEGATIVE_DEFAULT
        except Exception:                                            # noqa: BLE001
            NEGATIVE_DEFAULT = ""
        self.negative.setPlainText(NEGATIVE_DEFAULT)
        af.addRow(tr("avatar.negative", "Prompt negativo"), self.negative)
        self.ref_idx = QSlider(Qt.Horizontal); self.ref_idx.setRange(0, 30); self.ref_idx.setValue(10)
        self.lbl_ref = QLabel("10")
        self.ref_idx.valueChanged.connect(lambda x: self.lbl_ref.setText(str(x)))
        self.ref_idx.setToolTip(tr("avatar.refidx.tt",
                                   "Video lunghi: 0–24 = più coerenza col volto iniziale; 30 = meno gesti "
                                   "ripetuti."))
        rr = QHBoxLayout(); rr.addWidget(QLabel(tr("avatar.refidx.left", "coerenza")))
        rr.addWidget(self.ref_idx, 1); rr.addWidget(QLabel(tr("avatar.refidx.right", "varietà"))); rr.addWidget(self.lbl_ref)
        af.addRow(tr("avatar.refidx", "Riferimento continuazioni"), rr)
        self.mask_range = QSpinBox(); self.mask_range.setRange(1, 8); self.mask_range.setValue(3)
        self.mask_range.setToolTip(tr("avatar.maskrange.tt",
                                      "Più alto = meno gesti ripetuti, ma valori troppo alti creano artefatti."))
        af.addRow(tr("avatar.maskrange", "Anti-ripetizione"), self.mask_range)
        self.vocal_sep = QCheckBox(tr("avatar.vocalsep", "Pulisci voce da musica di sottofondo"))
        self.vocal_sep.setChecked(True)
        self.vocal_sep.setToolTip(tr("avatar.vocalsep.tt",
                                     "Il labiale segue solo la voce isolata; nel video resta l'audio originale."))
        af.addRow("", self.vocal_sep)
        self.offload_kv = QCheckBox(tr("avatar.offloadkv", "Risparmia VRAM (video lunghi / 720p)"))
        af.addRow("", self.offload_kv)
        self.lbl_steps = QLabel(tr("avatar.steps.info", "Passi: 8 (distillazione DMD obbligatoria per la v1.5)"))
        af.addRow(tr("avatar.steps", "Passi"), self.lbl_steps)
        cfg_tt = tr("avatar.cfg.tt", "Con la distillazione DMD (obbligatoria nella v1.5) le guidance sono "
                                     "fissate a 1.0 dal modello: i valori sono mostrati solo per completezza.")
        self.text_cfg = QDoubleSpinBox(); self.text_cfg.setValue(1.0); self.text_cfg.setEnabled(False)
        self.text_cfg.setToolTip(cfg_tt)
        self.audio_cfg = QDoubleSpinBox(); self.audio_cfg.setValue(1.0); self.audio_cfg.setEnabled(False)
        self.audio_cfg.setToolTip(cfg_tt)
        af.addRow(tr("avatar.cfg.text", "Guidance testo"), self.text_cfg)
        af.addRow(tr("avatar.cfg.audio", "Guidance audio"), self.audio_cfg)
        v.addWidget(self.adv)

        # post
        g = QGroupBox(tr("avatar.post.group", "Post-produzione")); f = QFormLayout(g)
        self.face_restore = QComboBox()
        self.face_restore.addItem(tr("avatar.face.none", "Nessuno"), "")
        self.face_restore.addItem("CodeFormer", "codeformer-v0.1.0.pth")
        self.face_restore.addItem("GFPGAN 1.4", "GFPGANv1.4.pth")
        f.addRow(tr("avatar.face", "Restauro volto"), self.face_restore)
        self.rife = QCheckBox(tr("avatar.rife", "Fluidità 50 fps (RIFE)"))
        f.addRow("", self.rife)
        v.addWidget(g)
        v.addStretch()
        scroll.setWidget(box)

        # destra: anteprima + log
        right = QWidget(); rv = QVBoxLayout(right)
        self.video_widget = None
        self._vplayer = None
        try:
            from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
            from PySide6.QtMultimediaWidgets import QVideoWidget
            self.video_widget = QVideoWidget(); self.video_widget.setMinimumSize(320, 240)
            self._vplayer = QMediaPlayer(self); self._vaudio = QAudioOutput(self)
            self._vplayer.setAudioOutput(self._vaudio); self._vplayer.setVideoOutput(self.video_widget)
            rv.addWidget(self.video_widget, 1)
        except Exception:                                            # noqa: BLE001
            rv.addWidget(_hint(tr("avatar.preview.none", "Anteprima video non disponibile (QtMultimedia).")), 1)
        pr = QHBoxLayout()
        self.btn_vplay = QPushButton(tr("wave.play", "▶")); self.btn_vplay.clicked.connect(self._vtoggle)
        self.btn_vstop = QPushButton(tr("wave.stop", "⏹")); self.btn_vstop.clicked.connect(self._vstop)
        self.btn_vfolder = QPushButton(tr("avatar.preview.folder", "Apri cartella"))
        self.btn_vfolder.clicked.connect(self._open_folder)
        for b in (self.btn_vplay, self.btn_vstop, self.btn_vfolder):
            b.setEnabled(False); pr.addWidget(b)
        pr.addStretch()
        rv.addLayout(pr)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(2000)
        self.log.setFixedHeight(150)
        rv.addWidget(self.log)
        self._last_video = ""

        split.addWidget(scroll); split.addWidget(right)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 1)
        split.setSizes([480, 560]); split.setChildrenCollapsible(True); split.setHandleWidth(8)
        root.addWidget(split, 1)
        self.gen_btn = QPushButton(tr("avatar.run", "🗣 Genera avatar parlante"))
        self.gen_btn.setObjectName("generate")
        try:
            from ui import theme
            theme.style_generate(self.gen_btn)
        except Exception:                                            # noqa: BLE001
            pass
        self.gen_btn.clicked.connect(self._on_generate)
        root.addWidget(self.gen_btn)

        self.mode_img.toggled.connect(self._sync_visibility)
        self.people.currentIndexChanged.connect(self._sync_visibility)
        self._sync_visibility()

    # ── stato ───────────────────────────────────────────────────────────────────────────────────
    def _two(self) -> bool:
        return self.people.currentData() == 2

    def _sync_visibility(self, *_):
        img = self.mode_img.isChecked()
        two = self._two()
        if two and not img:                              # due persone = serve la foto
            self.mode_img.setChecked(True); img = True
        self.img_group.setVisible(img)
        self.btn_boxes.setVisible(two); self.lbl_boxes.setVisible(two)
        self.audio2.setVisible(two); self.turns.setVisible(two); self.lbl_turns.setVisible(two)
        self.spk2.setVisible(two)
        self.aspect.setVisible(not img); self.lbl_aspect.setVisible(not img)
        self.resize_mode.setEnabled(img)
        self._update_duration()

    def on_canvas_image_changed(self, path: str = ""):
        ok = bool(path) and os.path.isfile(str(path))
        self.btn_canvas.setEnabled(ok)
        self.btn_canvas.setToolTip(tr("avatar.image.canvas.tt",
                                      "Usa la foto caricata nel canvas del tab Immagini. Attivo solo se il "
                                      "canvas contiene un'immagine.") if ok else
                                   tr("avatar.image.canvas.off", "Il canvas è vuoto: carica o genera una foto."))

    def set_last_audio(self, path: str):
        """Slot per audio_tab.tts_generated / music_tab.music_ready."""
        if path and os.path.isfile(path):
            self._last_generated_audio = path

    def _last_audio(self) -> str:
        p = self._last_generated_audio
        return p if p and os.path.isfile(p) else _latest_audio_file()

    @property
    def _image_path(self) -> str:
        return self.img_path.text().strip()

    def _refresh_thumb(self, *_):
        p = self._image_path
        pix = QPixmap(p) if p and os.path.isfile(p) else QPixmap()
        if pix.isNull():
            self.img_thumb.setPixmap(QPixmap()); self.img_thumb.setText(tr("avatar.image.none", "nessuna"))
        else:
            self.img_thumb.setPixmap(pix.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def set_image(self, path: str, *, to_canvas: bool = False):
        self.img_path.setText(path or "")
        self._bboxes = None; self.lbl_boxes.setText("")
        if to_canvas and self._image_path:
            self.load_into_canvas.emit(self._image_path)

    def _use_canvas(self):
        p = self.get_canvas_image() or ""
        if not p or not os.path.isfile(p):
            self.on_canvas_image_changed("")
            return
        self.set_image(p)

    def _browse_image(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("avatar.image.pick", "Scegli immagine"), "",
                                           tr("avatar.image.filter", IMG_FILTER_FALLBACK))
        if p:
            self.set_image(p, to_canvas=True)

    def _draw_boxes(self):
        if not self._image_path or not os.path.isfile(self._image_path):
            QMessageBox.information(self, tr("avatar.boxes.title", "Riquadri delle due persone"),
                                    tr("avatar.boxes.need_image", "Scegli prima l'immagine iniziale."))
            return
        dlg = _BoxDialog(self, self._image_path, self._bboxes)
        if dlg.exec() == QDialog.Accepted:
            fr = dlg.canvas.fractions()
            self._bboxes = fr
            self.lbl_boxes.setText(tr("avatar.boxes.set", "Riquadri impostati") if fr else
                                   tr("avatar.boxes.auto", "Riquadri automatici (sinistra / destra)"))

    def _audio_duration(self, path: str) -> float:
        try:
            from core.avatar import audio_chain as A
            return A.duration(path)
        except Exception:                                            # noqa: BLE001
            return 0.0

    def _update_duration(self, *_):
        from core.avatar import timing
        p1 = self.audio1.value()
        if not p1 or not os.path.isfile(p1):
            self.lbl_duration.setText(tr("avatar.duration.none", "Durata: scegli un audio"))
            return
        d = self._audio_duration(p1)
        start = end = 0.0
        if self.use_region.isChecked() and self.audio1.region():
            start, end = self.audio1.region()
        d = timing.effective_duration(d, trim_start=start, trim_end=end, max_s=self.max_dur.value())
        if self._two() and self.audio2.value() and self.turns.currentData() == "add":
            d += self._audio_duration(self.audio2.value())
        n = timing.segments_for(d)
        self.lbl_duration.setText(tr("avatar.duration", "Durata: {d:.1f} s → {n} segmenti (~{c:.1f} s generati, "
                                                        "poi tagliati sulla voce)").format(
            d=d, n=n, c=timing.covered_seconds(n)))

    # ── azioni ──────────────────────────────────────────────────────────────────────────────────
    def _enhance_prompt(self):
        txt = self.prompt.toPlainText().strip()
        if not txt:
            return
        self.btn_enhance.setEnabled(False)
        self.log.appendPlainText(tr("avatar.prompt.enhancing", "Miglioro il prompt…"))

        class _W(QThread):
            done = Signal(str)

            def run(inner):
                try:
                    from core import prompt_enhance
                    inner.done.emit(prompt_enhance.enhance(txt, kind="video") or txt)
                except Exception:                                    # noqa: BLE001
                    inner.done.emit(txt)
        self._enh = _W()
        self._enh.done.connect(lambda s: (self.prompt.setPlainText(s), self.btn_enhance.setEnabled(True)))
        self._enh.start()

    def _generate_voice(self):
        text = self.voice_text.toPlainText().strip()
        if not text:
            QMessageBox.information(self, tr("avatar.voice.group", "Genera voce"),
                                    tr("avatar.voice.empty", "Scrivi il testo (o il copione) da far dire."))
            return
        speakers = [self.spk1.get()] + ([self.spk2.get()] if self._two() else [])
        self.btn_voice.setEnabled(False)
        self.log.appendPlainText(tr("avatar.voice.running", "Genero la voce…"))
        self._voice_worker = _VoiceWorker(text, speakers, self.gap_ms.value(), self.pad_start.value(),
                                          self.pad_end.value())
        self._voice_worker.done.connect(self._on_voice_done)
        self._voice_worker.start()

    def _on_voice_done(self, res: dict):
        self.btn_voice.setEnabled(True)
        if res.get("status") != "success":
            self.log.appendPlainText("✗ " + res.get("error", ""))
            QMessageBox.warning(self, tr("avatar.voice.group", "Genera voce"), res.get("error", "")[:800])
            return
        tracks = res.get("tracks") or {}
        self._last_generated_audio = res["path"]
        if self._two() and "person2" in tracks:
            self.audio1.set_path(tracks["person1"]); self.audio2.set_path(tracks["person2"])
            self.turns.setCurrentIndex(self.turns.findData("para"))
        else:
            self.audio1.set_path(res["path"])
        self.log.appendPlainText(tr("avatar.voice.done", "Voce pronta ({d:.1f} s).").format(d=res.get("duration", 0)))

    def get_config(self) -> dict:
        mode = "ai2v" if self.mode_img.isChecked() else "at2v"
        start = end = 0.0
        if self.use_region.isChecked() and self.audio1.region():
            start, end = self.audio1.region()
        seed = random.randint(0, 2**31 - 1) if self.seed_random.isChecked() else self.seed.value()
        cfg = {
            "engine": self.engine.currentData(), "mode": mode,
            "image_path": self._image_path if mode == "ai2v" else "",
            "prompt": self.prompt.toPlainText().strip(),
            "negative": self.negative.toPlainText().strip(),
            "resolution": self.resolution.currentData(), "aspect": self.aspect.currentText(),
            "resize_mode": self.resize_mode.currentData(),
            "audio_path": self.audio1.value(), "audio2_path": self.audio2.value() if self._two() else "",
            "audio_type": self.turns.currentData() if self._two() else "add",
            "bbox": ({"person1": self._bboxes[0], "person2": self._bboxes[1]}
                     if self._two() and self._bboxes else None),
            "trim_start": start, "trim_end": end, "max_duration": self.max_dur.value(),
            "precision": self.precision.currentData(), "seed": seed,
            "ref_img_index": self.ref_idx.value(), "mask_frame_range": self.mask_range.value(),
            "vocal_separation": self.vocal_sep.isChecked(), "offload_kv_cache": self.offload_kv.isChecked(),
            "face_restore": self.face_restore.currentData() or "", "rife": self.rife.isChecked(),
        }
        return cfg

    def _on_generate(self):
        cfg = self.get_config()
        if cfg["mode"] == "ai2v" and not (cfg["image_path"] and os.path.isfile(cfg["image_path"])):
            QMessageBox.information(self, tr("tab.avatar", "Avatar"),
                                    tr("avatar.err.image", "Scegli l'immagine iniziale (canvas o file)."))
            return
        if not (cfg["audio_path"] and os.path.isfile(cfg["audio_path"])):
            QMessageBox.information(self, tr("tab.avatar", "Avatar"),
                                    tr("avatar.err.audio", "Scegli l'audio o genera la voce."))
            return
        if cfg["audio2_path"] == "" and self._two():
            QMessageBox.information(self, tr("tab.avatar", "Avatar"),
                                    tr("avatar.err.audio2", "Due persone: serve anche l'audio della persona 2."))
            return
        self.request_avatar.emit(cfg)

    # ── risultato ───────────────────────────────────────────────────────────────────────────────
    def append_log(self, line: str):
        self.log.appendPlainText(line)

    def set_busy(self, busy: bool):
        self.gen_btn.setEnabled(not busy)

    def show_result(self, path: str):
        self._last_video = path or ""
        ok = bool(path) and os.path.isfile(path)
        for b in (self.btn_vplay, self.btn_vstop, self.btn_vfolder):
            b.setEnabled(ok)
        if ok and self._vplayer is not None:
            self._vplayer.setSource(QUrl.fromLocalFile(path))
            self._vplayer.play()

    def _vtoggle(self):
        if self._vplayer is None:
            return
        from PySide6.QtMultimedia import QMediaPlayer
        if self._vplayer.playbackState() == QMediaPlayer.PlayingState:
            self._vplayer.pause()
        else:
            self._vplayer.play()

    def _vstop(self):
        if self._vplayer is not None:
            self._vplayer.stop()

    def _open_folder(self):
        if self._last_video:
            try:
                os.startfile(str(Path(self._last_video).parent))
            except Exception:                                        # noqa: BLE001
                pass
