"""Tab «Modifica Video AI» (#24/#59/#60 + Fase 1 IC-LoRA) — editing video nativo con LoRA LTX + tracking SAM2.

UI a DUE livelli (famiglia → modalità), derivata da `core/video_edit_modes.py` (fonte unica):
- **Modifica**: edit-anything (V2V da istruzione), inpaint oggetto (SAM2, sperimentale), sostituisci personaggio.
- **Stile**: anime↔real, ReStyle (immagine di stile).
- **Ripara video**: refocus / uncompress / deinterlace / motion deblur / detailer (guida = video sorgente).
- **Outpaint video**: estende il quadro (formato o margini) con IC-LoRA outpaint.
- **Camera / Movimento**: Cameraman (video di riferimento del movimento).
- **Lip dub**: il video (con la sua voce) è il riferimento; una NUOVA battuta di testo genera voce e labbra nuove
  (ricetta ufficiale Lightricks JUST-DUB-IT). Non si fornisce un file audio esterno.
- **HDR** (in «Ripara video»): video SDR → HDR (IC-LoRA HDR, ricetta ufficiale Lightricks, nessun prompt). Uscita:
  anteprima SDR tonemappata (quella che si apre nel player) + video HDR10 (HEVC 10 bit PQ/BT.2020) e/o sequenza EXR
  lineare (core/hdr_export.py), a scelta; clip lunghi a segmenti nel worker.

Il checkpoint è quello scelto nel tab Video: lo si MOSTRA in una riga di stato e, per le modalità che lo
raccomandano/richiedono distilled, si avvisa/blocca. MAI scelto o cambiato al posto dell'utente.

Emette `request_edit` con la config; l'orchestrazione (tracking + worker) è in main_window.
"""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from ui import theme  # noqa: E402
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from core import video_dub_translate as vdt
from core import motion_tracks as mt
from core import video_edit_ic as vic
from core import video_edit_modes as vem
from core import video_ic_prep as prep_mod
from ui.i18n_manager import tr
from ui.widgets.track_editor import TrackEditor

_LORA = "LTX-2/"
# Le 5 modalità storiche derivano da core/video_edit_modes (fonte unica): stessi key/i18n/edit_lora di prima.
MODES = vem.legacy_modes()

# default italiani (i18n: chiavi vedit.*, 15 lingue)
_MODE_DEFAULT = {
    "edit": "Modifica globale (edit-anything)", "inpaint": "Inpaint oggetto (sperimentale)",
    "anime2real": "Anime → reale", "real2anime": "Reale → anime", "replace_char": "Sostituisci personaggio",
    "restyle": "ReStyle con immagine di stile", "refocus": "Rimetti a fuoco (sfocatura da obiettivo)",
    "uncompress": "Togli artefatti di compressione", "deinterlace": "Deinterlaccia",
    "motiondeblur": "Riduci il mosso (motion blur)", "detailer": "Aggiungi dettaglio (detailer)",
    "outpaint": "Outpaint video", "cameraman": "Cameraman (movimento di camera da riferimento)",
    "lipdub": "Doppiaggio da testo", "lipdub_audio": "Doppiaggio da audio (sperimentale)",
    "hdr": "SDR → HDR", "motion_track": "Movimento da tracce (motion-track)",
}
_FAMILY_DEFAULT = {"edit": "Modifica", "style": "Stile", "repair": "Ripara video", "outpaint": "Outpaint video",
                   "lipdub": "Doppiaggio labiale", "camera": "Camera / Movimento"}
_NOTE_DEFAULT = {
    "vedit.note_edit": "Modifica globale del video da istruzione testuale.",
    "vedit.note_style": "Trasferimento di stile sull'intero video.",
    "vedit.note_refocus": "Corregge SOLO la sfocatura da obiettivo (fuori fuoco). Per il mosso usa «Riduci il mosso».",
    "vedit.note_uncompress": "Toglie gli artefatti di compressione (blocchi, banding, mosquito, ringing). "
                             "Il video sorgente è la guida: nessun altro campo.",
    "vedit.note_deinterlace": "Toglie il pettine dell'interlacciato e il frame-blending. Anteprima: poco testato.",
    "vedit.note_motiondeblur": "Attenua il mosso da movimento (un blur forte viene attenuato, non eliminato).",
    "vedit.note_detailer": "Arricchisce dettagli e texture. LoRA di LTX-2 (19B) usata su LTX 2.3: sperimentale.",
    "vedit.note_outpaint": "Estende il quadro: l'originale resta al centro, il resto (nero) viene generato. "
                           "Descrivi la scena completa nel prompt. Oltre ~60% di area nuova il modello inventa molto.",
    "vedit.note_restyle": "Scegli un'immagine di stile e descrivi lo stile («Make it … style.»). "
                          "Funziona bene con stili semplici; texture complesse sono deboli (v0.1).",
    "vedit.note_lipdub": "Doppiaggio labiale (JUST-DUB-IT, Lightricks): il video con la sua voce è il riferimento; scrivi la "
                         "NUOVA battuta e il modello genera voce e labbra nuove, per la stessa durata del video. Un solo "
                         "parlante; scrivi nell'alfabeto della lingua parlata (non traduce). Battuta di lunghezza simile "
                         "all'originale: troppo lunga = parole saltate, troppo corta = voce lenta. Non si fornisce un file "
                         "audio qui: per usare un tuo audio (canzone, TTS, file) usa i bottoni Talk/Canta sotto, oppure la variante "
                         "sperimentale «Doppiaggio da audio».",
    "vedit.note_lipdub_audio": "SPERIMENTALE (non è nella ricetta ufficiale): il TUO audio viene tenuto fisso e si rigenera "
                               "solo il video, con la LoRA del doppiaggio. Qualità non garantita; per canzoni e voci la via "
                               "collaudata è Talk/Canta del tab Video. L'audio comanda la durata: se è più corto il video "
                               "viene tagliato, se è più lungo l'audio viene tagliato.",
    "vedit.note_hdr": "Converte un video SDR (Rec.709) in HDR (IC-LoRA HDR di Lightricks): nessun prompt, il testo è un file "
                      "pre-calcolato. Nel player si apre un'anteprima SDR tonemappata; il file HDR vero (HEVC 10 bit PQ, "
                      "BT.2020) e/o la sequenza EXR lineare stanno accanto, con lo stesso nome. Dai un sorgente SDR, non già "
                      "HDR. Col checkpoint dev viene aggiunta la LoRA distilled ufficiale.",
    "vedit.note_motion_track": "Disegna sull'immagine le tracce del movimento: ogni traccia è un soggetto (o un punto dello "
                               "sfondo) e l'ORDINE dei punti è il tempo. L'immagine è il primo fotogramma; descrivi nel prompt chi è "
                               "e cosa fa. Col checkpoint dev viene aggiunta la LoRA distilled ufficiale. Il video dura quanto scegli "
                               "(24 fps ufficiali, 121 frame = 5 s); l'audio lo genera il modello.",
    "vedit.note_cameraman": "Il movimento di camera viene copiato dal video di riferimento. Se il risultato "
                            "ricalca troppo il riferimento, abbassa la forza dell'immagine o cambia seme.",
}
_RESTORE_PROMPT_DEFAULT = "Sharp, clean, high-quality footage, natural colors."


class _FnThread(QThread):
    """Esegue `fn()` fuori dal thread UI (STT/LLM locali: lenti e su GPU)."""
    finished_ok = Signal(object)
    failed = Signal(object)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            self.finished_ok.emit(self._fn())
        except Exception as e:                               # noqa: BLE001
            self.failed.emit(e)


class VideoEditAITab(QWidget):
    request_edit = Signal(dict)
    request_open_video_tab = Signal()       # «Apri tab Video» dal dialog sul checkpoint
    request_open_weights_tab = Signal()     # «Apri tab Pesi modelli» (distilled-LoRA mancante)
    request_open_v2v = Signal(str, str, str)  # (modalita' 'talk'|'sing', video, audio) → tab Video, pannello V2V
    # «Non ripetere» dell'avviso dev: SOLO per la sessione (non persistente)
    _dev_ack_session = False

    def __init__(self):
        super().__init__()
        self._video = ""
        self._vinfo = None                  # VideoInfo del sorgente (ffprobe veloce) o None
        self._model_getter = None
        self._last_audio_getter = None      # fn() -> path dell'ultimo audio/TTS/canzone generato (main_window)
        self._dub_thread = None
        self._dub_src_video = ""            # video da cui viene il «testo originale» mostrato (riuso STT)
        self._auto_prompt = ""              # ultimo testo di prompt precompilato dalla modalità
        self._canvas_image_getter = None    # fn() -> path dell'immagine nel canvas del Frame 1 (main_window)
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn_pick = QPushButton(tr("vedit.pick", "🎬 Scegli video…")); self.btn_pick.clicked.connect(self._pick)
        self.video_lbl = QLabel(tr("vedit.no_video", "nessun video"))
        top.addWidget(self.btn_pick); top.addWidget(self.video_lbl, 1)
        v.addLayout(top)

        f = QFormLayout()
        self.family = QComboBox()
        for fam in vem.families():
            self.family.addItem(tr(vem.FAMILY_I18N[fam], _FAMILY_DEFAULT.get(fam, fam)), fam)
        self.family.currentIndexChanged.connect(self._on_family)
        f.addRow(tr("vedit.family_lbl", "Famiglia"), self.family)
        self.mode = QComboBox()
        self.mode.currentIndexChanged.connect(self._on_mode)
        f.addRow(tr("vedit.mode_lbl", "Modalità"), self.mode)
        v.addLayout(f)

        # riga di stato del checkpoint (icona + testo: il colore non è l'unico veicolo)
        self.ckpt_lbl = QLabel(""); self.ckpt_lbl.setWordWrap(True)
        v.addWidget(self.ckpt_lbl)

        f = QFormLayout()
        self.obj = QLineEdit(); self.obj.setPlaceholderText(tr("vedit.obj_ph", "oggetto da selezionare (es. car, person) — per Inpaint"))
        f.addRow(tr("vedit.obj_lbl", "Oggetto (a parole)"), self.obj)

        # Sostituisci personaggio / ReStyle: immagine di riferimento (replace_char) o di stile (restyle).
        rrow = QHBoxLayout()
        self.ref_img = QLineEdit(); self.ref_img.setPlaceholderText(
            tr("vedit.ref.ph", "immagine sostitutiva: = primo frame del video ma con l'altro personaggio"))
        self.ref_btn = QPushButton(tr("vedit.ref.browse", "Sfoglia…")); self.ref_btn.clicked.connect(self._pick_ref)
        rrow.addWidget(self.ref_img, 1); rrow.addWidget(self.ref_btn)
        self.ref_row_lbl = QLabel(tr("vedit.ref_lbl", "Immagine sostitutiva (personaggio)"))
        f.addRow(self.ref_row_lbl, self._wrap_ref(rrow))
        # #102 forza dell'IDENTITÀ del nuovo personaggio (ref) vs movimento/struttura (canny). Prima hardcoded 1.0.
        self.ref_strength = QDoubleSpinBox(); self.ref_strength.setRange(0.1, 1.0); self.ref_strength.setSingleStep(0.05)
        self.ref_strength.setValue(0.8)
        self.ref_strength.setToolTip(tr("vedit.ref_strength.tt", "Quanto conta l'immagine di riferimento "
                                        "(identità del nuovo personaggio) rispetto al movimento del video: "
                                        "alto = più fedele alla foto, basso = più libertà."))
        self.ref_strength_lbl = QLabel(tr("vedit.ref_strength_lbl", "Forza identità (riferimento)"))
        f.addRow(self.ref_strength_lbl, self.ref_strength)

        # ReStyle: stile a parole → prompt «Make it {stile} style.»
        self.style_text = QLineEdit()
        self.style_text.setPlaceholderText(tr("vedit.style.text_ph", "es. Disney 2D Animation"))
        self.style_text_lbl = QLabel(tr("vedit.style.text_lbl", "Stile (a parole)"))
        f.addRow(self.style_text_lbl, self.style_text)

        # Camera: video di riferimento del movimento + immagine iniziale + dimensioni
        crow = QHBoxLayout()
        self.cam_ref = QLineEdit(); self.cam_ref.setPlaceholderText(tr("vedit.cam.ref_ph", "video con il movimento di camera da copiare"))
        self.cam_ref_btn = QPushButton(tr("vedit.cam.ref_pick", "🎬 Scegli riferimento…")); self.cam_ref_btn.clicked.connect(self._pick_cam_ref)
        crow.addWidget(self.cam_ref, 1); crow.addWidget(self.cam_ref_btn)
        self.cam_ref_lbl = QLabel(tr("vedit.cam.ref_lbl", "Video di riferimento del movimento"))
        f.addRow(self.cam_ref_lbl, self._wrap_ref(crow))
        irow = QHBoxLayout()
        self.cam_img = QLineEdit(); self.cam_img.setPlaceholderText(tr("vedit.cam.img_ph", "opzionale: immagine iniziale (i2v); vuoto = solo prompt"))
        self.cam_img_btn = QPushButton(tr("vedit.ref.browse", "Sfoglia…")); self.cam_img_btn.clicked.connect(self._pick_cam_img)
        irow.addWidget(self.cam_img, 1); irow.addWidget(self.cam_img_btn)
        self.cam_img_lbl = QLabel(tr("vedit.cam.img_lbl", "Immagine iniziale (opz.)"))
        f.addRow(self.cam_img_lbl, self._wrap_ref(irow))
        self.cam_img_strength = QDoubleSpinBox(); self.cam_img_strength.setRange(0.3, 1.0)
        self.cam_img_strength.setSingleStep(0.05); self.cam_img_strength.setValue(0.7)
        self.cam_img_strength.setToolTip(tr("vedit.cam.strong_ref_tip",
            "Più bassa = più movimento dal riferimento (0.5-0.7 consigliato). Se il risultato ricalca il riferimento, abbassa."))
        self.cam_img_strength_lbl = QLabel(tr("vedit.cam.img_strength_lbl", "Forza immagine"))
        f.addRow(self.cam_img_strength_lbl, self.cam_img_strength)
        self.cam_w = QSpinBox(); self.cam_w.setRange(256, 1280); self.cam_w.setSingleStep(64); self.cam_w.setValue(960)
        self.cam_h = QSpinBox(); self.cam_h.setRange(256, 1280); self.cam_h.setSingleStep(64); self.cam_h.setValue(576)
        srow = QHBoxLayout(); srow.addWidget(self.cam_w); srow.addWidget(QLabel("×")); srow.addWidget(self.cam_h)
        self.cam_size_lbl = QLabel(tr("vedit.cam.size_lbl", "Larghezza × altezza (≥ 960×512)"))
        f.addRow(self.cam_size_lbl, self._wrap_ref(srow))
        self.cam_seconds = QDoubleSpinBox(); self.cam_seconds.setRange(0.0, 60.0); self.cam_seconds.setSingleStep(0.5)
        self.cam_seconds.setSpecialValueText(tr("vedit.cam.all", "tutto il riferimento"))
        self.cam_seconds_lbl = QLabel(tr("vedit.cam.dur_lbl", "Durata (s)"))
        f.addRow(self.cam_seconds_lbl, self.cam_seconds)

        # Motion-track: immagine di partenza (frame 0 + sfondo dell'editor) + editor di tracce + dimensioni/durata/seme
        mrow = QHBoxLayout()
        self.mt_img = QLineEdit(); self.mt_img.setPlaceholderText(
            tr("vedit.mt.img_ph", "immagine di partenza: è il primo fotogramma e lo sfondo su cui disegni le tracce"))
        self.mt_img.editingFinished.connect(self._mt_image_changed)
        self.mt_img_btn = QPushButton(tr("vedit.ref.browse", "Sfoglia…")); self.mt_img_btn.clicked.connect(self._pick_mt_img)
        self.mt_img_canvas = QPushButton(tr("vedit.mt.from_canvas", "Dal canvas (Frame 1)"))
        self.mt_img_canvas.clicked.connect(self._mt_from_canvas)
        self.mt_img_frame = QPushButton(tr("vedit.mt.from_video", "Primo fotogramma del video"))
        self.mt_img_frame.clicked.connect(self._mt_from_video)
        for _w in (self.mt_img_btn, self.mt_img_canvas, self.mt_img_frame):
            _w.setMinimumHeight(32)
        mrow.addWidget(self.mt_img, 1); mrow.addWidget(self.mt_img_btn); mrow.addWidget(self.mt_img_canvas)
        mrow.addWidget(self.mt_img_frame)
        self.mt_img_lbl = QLabel(tr("vedit.mt.img_lbl", "Immagine di partenza"))
        f.addRow(self.mt_img_lbl, self._wrap_ref(mrow))
        self.mt_editor = TrackEditor()
        self.mt_editor.canvas.setMinimumHeight(260)
        self.mt_editor.changed.connect(self._mt_refresh_info)
        self.mt_tracks_lbl = QLabel(tr("vedit.mt.tracks_lbl", "Tracce di movimento"))
        f.addRow(self.mt_tracks_lbl, self.mt_editor)
        self.mt_w = QSpinBox(); self.mt_w.setRange(256, 1280); self.mt_w.setSingleStep(64); self.mt_w.setValue(1024)
        self.mt_h = QSpinBox(); self.mt_h.setRange(256, 1280); self.mt_h.setSingleStep(64); self.mt_h.setValue(576)
        self.mt_w.valueChanged.connect(self._mt_size_changed); self.mt_h.valueChanged.connect(self._mt_size_changed)
        msrow = QHBoxLayout(); msrow.addWidget(self.mt_w); msrow.addWidget(QLabel("×")); msrow.addWidget(self.mt_h)
        self.mt_size_lbl = QLabel(tr("vedit.mt.size_lbl", "Larghezza × altezza"))
        f.addRow(self.mt_size_lbl, self._wrap_ref(msrow))
        self.mt_seconds = QDoubleSpinBox(); self.mt_seconds.setRange(0.5, 10.7); self.mt_seconds.setSingleStep(0.5)
        self.mt_seconds.setDecimals(1); self.mt_seconds.setValue(5.0); self.mt_seconds.setSuffix(" s")
        self.mt_fps = QComboBox()
        for _fps in (24, 25, 30):
            self.mt_fps.addItem(f"{_fps} fps", _fps)
        self.mt_seconds.valueChanged.connect(self._mt_timing_changed); self.mt_fps.currentIndexChanged.connect(self._mt_timing_changed)
        mdrow = QHBoxLayout(); mdrow.addWidget(self.mt_seconds); mdrow.addWidget(self.mt_fps)
        self.mt_dur_lbl = QLabel(tr("vedit.mt.dur_lbl", "Durata e fps"))
        f.addRow(self.mt_dur_lbl, self._wrap_ref(mdrow))
        self.mt_info = QLabel(""); self.mt_info.setWordWrap(True)
        self.mt_info_lbl = QLabel("")
        f.addRow(self.mt_info_lbl, self.mt_info)
        self.mt_img_strength = QDoubleSpinBox(); self.mt_img_strength.setRange(0.3, 1.0); self.mt_img_strength.setSingleStep(0.05)
        self.mt_img_strength.setValue(1.0)
        self.mt_img_strength.setToolTip(tr("vedit.mt.img_strength_tt",
            "Quanto il primo fotogramma resta fedele all'immagine (1.0 = come nella ricetta ufficiale)."))
        self.mt_img_strength_lbl = QLabel(tr("vedit.cam.img_strength_lbl", "Forza immagine"))
        f.addRow(self.mt_img_strength_lbl, self.mt_img_strength)
        self.mt_audio = QCheckBox(tr("vedit.mt.audio", "Genera anche l'audio (parlato/suoni dal prompt)"))
        self.mt_audio.setChecked(True)
        self.mt_audio.setToolTip(tr("vedit.mt.audio_tt",
            "Come nel workflow ufficiale il modello genera l'audio insieme al video: voce e suoni vengono dal prompt. "
            "Spegnilo se il prompt non parla (altrimenti l'audio puo' essere rumore)."))
        f.addRow("", self.mt_audio)
        self.mt_seed = QSpinBox(); self.mt_seed.setRange(0, 2147483647); self.mt_seed.setValue(42)
        self.mt_seed_lbl = QLabel(tr("vedit.mt.seed_lbl", "Seme"))
        f.addRow(self.mt_seed_lbl, self.mt_seed)

        # Doppiaggio (lipdub): scena (opz.) + battuta + lingua. Da testo: la battuta NUOVA genera voce+labbra
        # («<scena>, speaking in <Lingua> saying: "<battuta>"»); da audio (sperimentale): audio dell'utente fisso.
        self.lip_scene = QLineEdit()
        self.lip_scene.setPlaceholderText(tr("vedit.lipdub.scene_ph",
            "opzionale, in inglese: es. A woman in a red jacket, close-up in a kitchen"))
        self.lip_scene_lbl = QLabel(tr("vedit.lipdub.scene_lbl", "Scena / chi parla (opz.)"))
        f.addRow(self.lip_scene_lbl, self.lip_scene)
        # B: audio dell'utente (file o ultimo audio generato) + riferimento voce
        arow = QHBoxLayout()
        self.lip_audio = QLineEdit(); self.lip_audio.setPlaceholderText(
            tr("vedit.lipdub.audio_ph", "file audio (canzone, voce, TTS)"))
        self.lip_audio.textChanged.connect(self._refresh_lipdub_info)
        self.lip_audio_btn = QPushButton(tr("vedit.ref.browse", "Sfoglia…")); self.lip_audio_btn.clicked.connect(self._pick_lip_audio)
        self.lip_audio_last = QPushButton(tr("vedit.lipdub.audio_last", "Ultimo audio generato"))
        self.lip_audio_last.clicked.connect(self._use_last_audio)
        arow.addWidget(self.lip_audio, 1); arow.addWidget(self.lip_audio_btn); arow.addWidget(self.lip_audio_last)
        self.lip_audio_lbl = QLabel(tr("vedit.lipdub.audio_lbl", "Audio da usare"))
        f.addRow(self.lip_audio_lbl, self._wrap_ref(arow))
        self.lip_audio_ref = QComboBox()
        self.lip_audio_ref.addItem(tr("vedit.lipdub.audio_ref_ext", "l'audio scelto"), "external")
        self.lip_audio_ref.addItem(tr("vedit.lipdub.audio_ref_orig", "la voce del video originale"), "original")
        self.lip_audio_ref.setToolTip(tr("vedit.lipdub.audio_ref_tt",
            "Quale audio fa da riferimento di voce per la LoRA. Sperimentale: «l'audio scelto» imita lo stage 2 ufficiale; "
            "«la voce del video» lo stage 1 (serve una traccia audio nel video)."))
        self.lip_audio_ref_lbl = QLabel(tr("vedit.lipdub.audio_ref_lbl", "Riferimento voce"))
        f.addRow(self.lip_audio_ref_lbl, self.lip_audio_ref)
        self.lip_exp = QLabel(tr("vedit.lipdub.exp_banner",
            "⚠ SPERIMENTALE: qualità non garantita. Per canzoni e voci la via collaudata è Talk/Canta del tab Video."))
        self.lip_exp.setWordWrap(True)
        self.lip_exp_lbl = QLabel("")
        f.addRow(self.lip_exp_lbl, self.lip_exp)
        self.lip_line = QPlainTextEdit(); self.lip_line.setFixedHeight(70)
        self.lip_line.setPlaceholderText(tr("vedit.lipdub.line_ph",
            "la NUOVA battuta, nell'alfabeto della lingua parlata (il modello non traduce)"))
        self.lip_line.textChanged.connect(self._refresh_lipdub_info)
        self.lip_line_lbl = QLabel(tr("vedit.lipdub.line_lbl", "Nuova battuta (testo)"))
        f.addRow(self.lip_line_lbl, self.lip_line)
        self.lip_lang = QComboBox()                       # SOLO le lingue parlabili (core/video_edit_ic.LIPDUB_SPOKEN_LANGS)
        for code, name in vic.LIPDUB_LANGUAGES:
            self.lip_lang.addItem(name if code else tr("vedit.lipdub.lang_none", "(non specificata)"), code)
        self.lip_lang.setToolTip(tr("vedit.lipdub.lang_tt",
            "Lingua che il modello parla nella nuova battuta (solo lingue con prove di supporto)."))
        self.lip_lang_lbl = QLabel(tr("vedit.lipdub.lang_dest_lbl", "Lingua di destinazione"))
        f.addRow(self.lip_lang_lbl, self.lip_lang)
        # C: estrai il parlato dal video e traducilo (STT + LLM locali) nel campo battuta, modificabile
        self.lip_translate = QPushButton(tr("vedit.dub.translate_btn", "Estrai e traduci"))
        self.lip_translate.setToolTip(tr("vedit.dub.translate_tt",
            "Estrae il parlato del video (STT locale), lo traduce nella lingua di destinazione con l'LLM locale e mette "
            "il testo nel campo battuta, modificabile. Chiede una lunghezza simile all'originale."))
        self.lip_translate.clicked.connect(self._translate_clicked)
        self.lip_translate_lbl = QLabel("")
        f.addRow(self.lip_translate_lbl, self.lip_translate)
        self.lip_src_lang = QLabel(""); self.lip_src_lang.setWordWrap(True)
        self.lip_src_lang_lbl = QLabel("")
        f.addRow(self.lip_src_lang_lbl, self.lip_src_lang)
        self.lip_src_text = QPlainTextEdit(); self.lip_src_text.setFixedHeight(54); self.lip_src_text.setReadOnly(False)
        self.lip_src_text.setPlaceholderText(tr("vedit.dub.src_ph", "testo originale riconosciuto (modificabile: se lo correggi, la traduzione parte da qui)"))
        self.lip_src_text_lbl = QLabel(tr("vedit.dub.src_text_lbl", "Testo originale"))
        f.addRow(self.lip_src_text_lbl, self.lip_src_text)
        # A: l'utente vuole un SUO audio → rimando a Talk/Canta (esistono gia', collaudate)
        orow = QHBoxLayout()
        self.lip_open_talk = QPushButton(tr("vedit.lipdub.open_talk", "Apri Talk (video + audio)"))
        self.lip_open_talk.clicked.connect(lambda: self._open_v2v("talk"))
        self.lip_open_sing = QPushButton(tr("vedit.lipdub.open_sing", "Apri Canta (foto + canzone)"))
        self.lip_open_sing.clicked.connect(lambda: self._open_v2v("sing"))
        orow.addWidget(self.lip_open_talk); orow.addWidget(self.lip_open_sing); orow.addStretch(1)
        self.lip_own_lbl = QLabel(tr("vedit.lipdub.own_audio_lbl", "Vuoi usare un TUO audio (canzone, TTS, file)?"))
        f.addRow(self.lip_own_lbl, self._wrap_ref(orow))
        self.lip_info = QLabel(""); self.lip_info.setWordWrap(True)
        self.lip_info_lbl = QLabel("")
        f.addRow(self.lip_info_lbl, self.lip_info)

        # Outpaint: formato / margini / ancora / opzioni
        self.out_format = QComboBox()
        for key, i18n, dflt in (("16:9", "vedit.out.fmt.169", "16:9"), ("9:16", "vedit.out.fmt.916", "9:16"),
                                ("1:1", "vedit.out.fmt.11", "1:1"), ("4:3", "vedit.out.fmt.43", "4:3"),
                                ("3:4", "vedit.out.fmt.34", "3:4"), ("21:9", "vedit.out.fmt.219", "21:9"),
                                ("custom", "vedit.out.fmt.custom", "Margini personalizzati")):
            self.out_format.addItem(tr(i18n, dflt), key)
        self.out_format.currentIndexChanged.connect(self._on_outpaint_change)
        self.out_format_lbl = QLabel(tr("vedit.out.canvas_lbl", "Formato del canvas"))
        f.addRow(self.out_format_lbl, self.out_format)
        self.out_anchor = QComboBox()
        # «Estendi verso»: il dato resta l'ancora del sorgente (center|left|right|top|bottom), ma l'etichetta dice
        # DOVE si aggiunge area nuova (video a sinistra = si estende a destra).
        for key, i18n, dflt in (("center", "vedit.out.dir.center", "Tutti i lati (video al centro)"),
                                ("left", "vedit.out.dir.right", "Solo a destra (video a sinistra)"),
                                ("right", "vedit.out.dir.left", "Solo a sinistra (video a destra)"),
                                ("top", "vedit.out.dir.down", "Solo in basso (video in alto)"),
                                ("bottom", "vedit.out.dir.up", "Solo in alto (video in basso)")):
            self.out_anchor.addItem(tr(i18n, dflt), key)
        self.out_anchor.setToolTip(tr("vedit.out.dir.tt",
            "Dove aggiungere l'area nuova. Con un formato più largo del video si estende ai lati, con uno più alto "
            "sopra/sotto; «solo in alto/in basso» (o a destra/sinistra) la mette tutta da una parte."))
        self.out_anchor.currentIndexChanged.connect(self._on_outpaint_change)
        self.out_anchor_lbl = QLabel(tr("vedit.out.dir_lbl", "Estendi verso"))
        f.addRow(self.out_anchor_lbl, self.out_anchor)
        self.out_m = {}
        mrow = QHBoxLayout()
        for key, i18n, dflt in (("top", "vedit.out.margin.top", "Sopra"), ("right", "vedit.out.margin.right", "Destra"),
                                ("bottom", "vedit.out.margin.bottom", "Sotto"), ("left", "vedit.out.margin.left", "Sinistra")):
            sb = QSpinBox(); sb.setRange(0, 4096); sb.setSingleStep(16); sb.setPrefix(tr(i18n, dflt) + " ")
            sb.valueChanged.connect(self._on_outpaint_change)
            self.out_m[key] = sb; mrow.addWidget(sb)
        self.out_margins_lbl = QLabel(tr("vedit.out.margins_lbl", "Margini (px del sorgente)"))
        f.addRow(self.out_margins_lbl, self._wrap_ref(mrow))
        self.out_dark = QCheckBox(tr("vedit.out.dark_lbl", "Sorgente scura (gamma 2.0 / 0.5)"))
        self.out_dark.setToolTip(tr("vedit.out.dark.tt",
            "Schiarisce il solo sorgente prima dell'outpaint e scurisce il risultato dopo: evita che le zone scure "
            "si confondano con le barre nere da generare. Suggerito in automatico se il video è scuro."))
        f.addRow("", self.out_dark)
        self.out_paste = QCheckBox(tr("vedit.out.paste_lbl", "Rimetti l'originale al centro"))
        self.out_paste.setChecked(True)
        self.out_paste.setToolTip(tr("vedit.out.paste.tt",
            "Dopo la generazione incolla i pixel originali al centro (bordo di 2 px escluso): la parte originale resta identica."))
        f.addRow("", self.out_paste)
        self.out_paste.toggled.connect(self._on_outpaint_change)
        self.out_feather = QSpinBox(); self.out_feather.setRange(0, prep_mod.MAX_FEATHER)
        self.out_feather.setSingleStep(4); self.out_feather.setSuffix(" px")
        self.out_feather.setValue(prep_mod.DEFAULT_FEATHER)
        self.out_feather.setSpecialValueText(tr("vedit.out.feather.off", "0 (taglio netto)"))
        self.out_feather.setToolTip(tr("vedit.out.feather.tt",
            "Larghezza della sfumatura fra l'originale incollato e la parte generata (dentro il bordo del video, solo "
            "dove confina con area nuova). Nasconde la giunta quando il centro generato differisce dall'originale; "
            "uguale per ogni fotogramma, quindi senza sfarfallio. 0 = taglio netto."))
        self.out_feather_lbl = QLabel(tr("vedit.out.feather_lbl", "Sfumatura del bordo"))
        f.addRow(self.out_feather_lbl, self.out_feather)
        self.out_minbar = QCheckBox(tr("vedit.out.minbar_lbl", "Barre larghe almeno 15% (rimpicciolisce il video)"))
        self.out_minbar.setToolTip(tr("vedit.out.minbar.tt",
            "Sperimentale. Il modello non riempie le barre nuove sotto il ~15% del canvas: con questa opzione il video "
            "viene rimpicciolito dentro lo stesso canvas finché ogni barra è larga almeno il 15%. Costa risoluzione "
            "dell'originale e aumenta l'area da generare."))
        self.out_minbar.toggled.connect(self._on_outpaint_change)
        f.addRow("", self.out_minbar)
        # «Garantisci riempimento» (live 17/10): due barre simmetriche sullo stesso asse sembrano un letterbox e il modello
        # le lascia NERE (area nuova 33-45%); con l'area nuova >= 60% le riempie. ATTIVO di default, disattivabile.
        self.out_fill = QCheckBox(tr("vedit.out.fill_lbl", "Garantisci riempimento (rimpicciolisce il video)"))
        self.out_fill.setChecked(True)
        self.out_fill.setToolTip(tr("vedit.out.fill.tt",
            "Con due barre opposte (sopra+sotto o sinistra+destra) il modello le scambia per un letterbox e le lascia "
            "nere. Con questa opzione, in quel caso, il video originale viene rimpicciolito al centro finché l'area "
            "nuova è almeno il 60%: le barre vengono riempite, ma l'originale incollato risulta più piccolo (meno "
            "risoluzione). Disattivala per tenere l'originale alla sua scala (le barre possono restare nere)."))
        self.out_fill.toggled.connect(self._on_outpaint_change)
        f.addRow("", self.out_fill)
        self.out_info = QLabel(""); self.out_info.setWordWrap(True)
        self.out_info_lbl = QLabel("")
        f.addRow(self.out_info_lbl, self.out_info)
        self.out_preview = QLabel(); self.out_preview.setFixedSize(220, 124)
        self.out_preview.setAccessibleName(tr("vedit.out.preview", "Anteprima area nuova"))
        self.out_preview_lbl = QLabel(tr("vedit.out.preview", "Anteprima area nuova"))
        f.addRow(self.out_preview_lbl, self.out_preview)

        self.prompt = QPlainTextEdit(); self.prompt.setFixedHeight(70)
        self._prompt_ph = tr("vedit.prompt_ph", "cosa generare/cambiare (es. a blue motorcycle / snowy winter)")
        self.prompt.setPlaceholderText(self._prompt_ph)
        f.addRow(tr("vedit.prompt_lbl", "Prompt"), self.prompt)
        self.vedit_enhance = QCheckBox(tr("vedit.enhance", "✨ Arricchisci prompt (AI)"))
        self.vedit_enhance.setToolTip(tr("vedit.enhance.tt",
            "Un LLM locale arricchisce il prompt di modifica prima di elaborare. Invisibile, opzionale."))
        f.addRow("", self.vedit_enhance)

        self.strength = QComboBox()
        for s in ("0.6", "0.8", "1.0"):
            self.strength.addItem(s, float(s))
        self.strength.setCurrentText("1.0")
        self.strength_lbl = QLabel(tr("vedit.strength_lbl", "Forza editing"))
        f.addRow(self.strength_lbl, self.strength)

        # IC-LoRA: attenzione della guida, risoluzione, audio
        self.attn = QDoubleSpinBox(); self.attn.setRange(0.1, 1.0); self.attn.setSingleStep(0.1); self.attn.setValue(1.0)
        self.attn.setToolTip(tr("vedit.attn.tt", "Quanto la guida (il video) vincola il risultato: 1.0 = massimo."))
        self.attn_lbl = QLabel(tr("vedit.attn_lbl", "Attenzione guida"))
        f.addRow(self.attn_lbl, self.attn)
        self.res_mode = QComboBox()
        self.res_mode.addItem(tr("vedit.res.source", "Come sorgente (entro il tetto)"), "source")
        self.res_mode.addItem(tr("vedit.res.half", "Ridotta (metà)"), "half")
        self.res_mode.addItem(tr("vedit.res.custom", "Personalizzata"), "custom")
        self.res_mode.currentIndexChanged.connect(self._on_mode)
        self.res_long = QSpinBox(); self.res_long.setRange(256, 1280); self.res_long.setSingleStep(64); self.res_long.setValue(1024)
        rrow2 = QHBoxLayout(); rrow2.addWidget(self.res_mode, 1); rrow2.addWidget(self.res_long)
        self.res_lbl = QLabel(tr("vedit.res_lbl", "Risoluzione (lato lungo)"))
        f.addRow(self.res_lbl, self._wrap_ref(rrow2))
        self.keep_audio = QCheckBox(tr("vedit.keep_audio", "Mantieni l'audio originale")); self.keep_audio.setChecked(True)
        self.keep_audio.setToolTip(tr("vedit.keep_audio.tt", "Rimette l'audio del video sorgente sul risultato."))
        f.addRow("", self.keep_audio)
        self.seg_info = QLabel(""); self.seg_info.setWordWrap(True)
        f.addRow("", self.seg_info)

        # HDR (SDR → HDR): formato d'uscita, esposizione dell'anteprima SDR, EXR half, alta qualità
        self.hdr_format = QComboBox()
        for key, i18n, dflt in (("pq", "vedit.hdr.fmt_pq", "Video HDR10 (HEVC 10 bit, PQ)"),
                                ("both", "vedit.hdr.fmt_both", "Video HDR10 + sequenza EXR"),
                                ("exr", "vedit.hdr.fmt_exr", "Solo sequenza EXR (compositing)")):
            self.hdr_format.addItem(tr(i18n, dflt), key)
        self.hdr_format.setToolTip(tr("vedit.hdr.format_tt",
            "Video HDR10: HEVC 10 bit, PQ, BT.2020 (TV/telefoni HDR, Resolve, YouTube). EXR: un file float lineare per "
            "frame, per il compositing (DJV, Nuke, Blender): pesa molto. L'anteprima SDR c'è sempre."))
        self.hdr_format.currentIndexChanged.connect(self._refresh_hdr_info)
        self.hdr_format_lbl = QLabel(tr("vedit.hdr.format_lbl", "Formato HDR"))
        f.addRow(self.hdr_format_lbl, self.hdr_format)
        self.hdr_exposure = QDoubleSpinBox(); self.hdr_exposure.setRange(-6.0, 6.0); self.hdr_exposure.setSingleStep(0.5)
        self.hdr_exposure.setDecimals(1); self.hdr_exposure.setSuffix(" EV"); self.hdr_exposure.setValue(0.0)
        self.hdr_exposure.setToolTip(tr("vedit.hdr.exposure_tt",
            "Esposizione dell'ANTEPRIMA SDR in stop (+1 = il doppio più chiaro). Non cambia il video HDR né gli EXR."))
        self.hdr_exposure_lbl = QLabel(tr("vedit.hdr.exposure_lbl", "Esposizione anteprima SDR"))
        f.addRow(self.hdr_exposure_lbl, self.hdr_exposure)
        self.hdr_exr_half = QCheckBox(tr("vedit.hdr.exr_half", "EXR a 16 bit (half): metà dello spazio"))
        self.hdr_exr_half.setChecked(True)
        f.addRow("", self.hdr_exr_half)
        self.hdr_hq = QCheckBox(tr("vedit.hdr.hq", "Alta qualità (circa 2× più lento)"))
        self.hdr_hq.setToolTip(tr("vedit.hdr.hq_tt",
            "Genera il doppio dei fotogrammi e tiene uno su due: meno artefatti temporali, tempo doppio."))
        f.addRow("", self.hdr_hq)
        self.hdr_info = QLabel(""); self.hdr_info.setWordWrap(True)
        self.hdr_info_lbl = QLabel("")
        f.addRow(self.hdr_info_lbl, self.hdr_info)

        # #59 parametri inpaint (regolabili): CFG (aderenza prompt), guide (ancoraggio all'originale),
        # dilatazione maschera (spazio al nuovo oggetto). Valori equilibrati di default.
        self.cfg = QDoubleSpinBox(); self.cfg.setRange(1.0, 10.0); self.cfg.setSingleStep(0.5); self.cfg.setValue(6.0)
        self.cfg_lbl = QLabel(tr("vedit.cfg_lbl", "CFG (aderenza prompt) — inpaint"))
        f.addRow(self.cfg_lbl, self.cfg)
        self.guide = QDoubleSpinBox(); self.guide.setRange(0.1, 1.0); self.guide.setSingleStep(0.1); self.guide.setValue(0.4)
        self.guide_lbl = QLabel(tr("vedit.guide_lbl", "Forza guide (ancoraggio originale)"))
        f.addRow(self.guide_lbl, self.guide)
        self.dilate = QSpinBox(); self.dilate.setRange(0, 64); self.dilate.setValue(24)
        self.dilate.setToolTip(tr("vedit.dilate.tt",
            "Quanto ALLARGARE la maschera tracciata (px): più spazio per il nuovo oggetto, bordi meno stretti."))
        self.dilate_lbl = QLabel(tr("vedit.dilate_lbl", "Dilatazione maschera (px)"))
        f.addRow(self.dilate_lbl, self.dilate)
        # blur maschera: era supportato dal worker (mask_blur) ma NON esposto → hardcoded 0.
        self.blur = QDoubleSpinBox(); self.blur.setRange(0.0, 30.0); self.blur.setSingleStep(1.0); self.blur.setValue(6.0)
        self.blur.setToolTip(tr("vedit.blur.tt",
            "Sfumatura dei bordi della maschera (gaussian, px): 0 = bordo netto, 6-12 = transizione morbida."))
        self.blur_lbl = QLabel(tr("vedit.blur_lbl", "Sfumatura maschera (px)"))
        f.addRow(self.blur_lbl, self.blur)
        v.addLayout(f)
        self._form = f

        self.note = QLabel(""); self.note.setWordWrap(True); v.addWidget(self.note)
        self.btn_go = QPushButton(tr("vedit.go", "✨ Genera modifica")); self.btn_go.clicked.connect(self._go); theme.style_generate(self.btn_go)
        v.addWidget(self.btn_go)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); v.addWidget(self.log, 1)
        self.mt_editor.message.connect(self.log.appendPlainText)
        self._on_family()

    # ── utilità ──────────────────────────────────────────────────────────────────────────────
    def _wrap_ref(self, layout):
        w = QWidget(); w.setLayout(layout); w._vedit_layout = layout; return w

    def _show_row(self, field, visible: bool):
        """Mostra/nasconde una riga del form dal widget campo (e la sua etichetta)."""
        lab = self._form.labelForField(field)
        field.setVisible(visible)
        if lab is not None:
            lab.setVisible(visible)

    def select_mode(self, key: str):
        """Seleziona famiglia e modalità per chiave (API pubblica per test/menu). KeyError se sconosciuta."""
        fam = vem.family_of(key)
        i = self.family.findData(fam)
        if i < 0:
            raise KeyError(f"famiglia «{fam}» non disponibile per la modalità {key}")
        if self.family.currentIndex() != i:
            self.family.setCurrentIndex(i)
        j = self.mode.findData(key)
        if j < 0:
            raise KeyError(f"modalità «{key}» non presente (famiglia {fam})")
        self.mode.setCurrentIndex(j)

    def current_spec(self) -> vem.ModeSpec:
        return vem.get_spec(self.mode.currentData())

    def _on_family(self, *_):
        fam = self.family.currentData()
        prev = self.mode.currentData()
        self.mode.blockSignals(True)
        self.mode.clear()
        for sp in vem.modes_for_family(fam):
            self.mode.addItem(tr(sp.i18n_key, _MODE_DEFAULT.get(sp.key, sp.key)), sp.key)
        j = self.mode.findData(prev)
        self.mode.setCurrentIndex(j if j >= 0 else 0)
        self.mode.blockSignals(False)
        self._on_mode()

    def _pick_ref(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("vedit.ref.browse", "Sfoglia…"), "",
                                           "Immagini (*.png *.jpg *.jpeg *.webp *.bmp)")
        if p:
            self.ref_img.setText(p)

    def _pick_cam_img(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("vedit.ref.browse", "Sfoglia…"), "",
                                           "Immagini (*.png *.jpg *.jpeg *.webp *.bmp)")
        if p:
            self.cam_img.setText(p)

    def _pick_cam_ref(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("vedit.cam.ref_pick", "Scegli riferimento"), "",
                                           "Video (*.mp4 *.mov *.avi *.mkv *.webm)")
        if p:
            self.cam_ref.setText(p)

    # ── motion-track: immagine, dimensioni, durata ─────────────────────────────────────────────
    def set_canvas_image_getter(self, fn):
        """`fn()` -> path dell'immagine nel canvas del Frame 1 o ''. Collegato da main_window."""
        self._canvas_image_getter = fn

    def _pick_mt_img(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("vedit.mt.img_lbl", "Immagine di partenza"), "",
                                           "Immagini (*.png *.jpg *.jpeg *.webp *.bmp)")
        if p:
            self.set_track_image(p)

    def _mt_from_canvas(self):
        p = ""
        if self._canvas_image_getter is not None:
            try:
                p = self._canvas_image_getter() or ""
            except Exception:                                # noqa: BLE001
                p = ""
        if p and os.path.isfile(p):
            self.set_track_image(p)
        else:
            self._block("vedit.mt.no_canvas", "Nessuna immagine nel canvas del Frame 1: scegli un file.")

    def _mt_from_video(self):
        if not self._video or not os.path.isfile(self._video):
            self._block("vedit.err_no_video", "Scegli prima un video."); return
        import tempfile
        out = os.path.join(tempfile.gettempdir(), "ltxstudio_motion", f"{Path(self._video).stem}_frame0.png")
        try:
            self.set_track_image(mt.extract_first_frame(self._video, out))
        except Exception as e:                               # noqa: BLE001
            self._block("vedit.mt.frame_err", "Primo fotogramma non estratto: {e}", e=e)

    def set_track_image(self, path: str):
        """Imposta l'immagine di partenza (sfondo dell'editor + frame 0), ricalcola le dimensioni dall'aspetto e riallinea
        l'editor. API pubblica anche per il rimando dal tab Video."""
        self.mt_img.setText(path or "")
        self._mt_image_changed()

    def _mt_image_changed(self, *_):
        p = self.mt_img.text().strip()
        if p and os.path.isfile(p):
            try:
                from PIL import Image
                with Image.open(p) as im:
                    iw, ih = im.size
                w, h = mt.default_size(iw, ih, max_long=vem.get_spec("motion_track").max_long_edge)
                for sb, v in ((self.mt_w, w), (self.mt_h, h)):
                    sb.blockSignals(True); sb.setValue(v); sb.blockSignals(False)
            except Exception:                                # noqa: BLE001
                pass
        self._mt_sync_editor()

    def _mt_size_changed(self, *_):
        self._mt_sync_editor()

    def _mt_timing_changed(self, *_):
        sp = vem.get_spec("motion_track")
        fps = int(self.mt_fps.currentData() or mt.DEFAULT_FPS)
        self.mt_seconds.blockSignals(True)
        self.mt_seconds.setMaximum(round(sp.max_frames_per_segment / float(fps), 1) - 0.1)
        self.mt_seconds.blockSignals(False)
        self._mt_sync_editor()

    def _mt_frames(self) -> int:
        """Frame 8k+1 per durata e fps scelti, mai oltre il tetto della modalità."""
        sp = vem.get_spec("motion_track")
        fps = int(self.mt_fps.currentData() or mt.DEFAULT_FPS)
        n = mt.frames_for_seconds(float(self.mt_seconds.value()), fps)
        while n > sp.max_frames_per_segment:
            n -= 8
        return n

    def _mt_sync_editor(self):
        """Riallinea l'editor (sfondo + spazio di lavoro + frame/fps dell'anteprima) ai campi della modalità."""
        if not hasattr(self, "mt_editor"):
            return
        w, h = int(self.mt_w.value()), int(self.mt_h.value())
        img = self.mt_img.text().strip()
        ok = self.mt_editor.set_reference(img if os.path.isfile(img) else "", w, h)
        if img and not ok:
            self._block("vedit.mt.img_unreadable", "Immagine di partenza illeggibile.")
        self.mt_editor.set_frames(self._mt_frames(), int(self.mt_fps.currentData() or mt.DEFAULT_FPS))
        self._mt_refresh_info()

    def _mt_refresh_info(self, *_):
        if not hasattr(self, "mt_info"):
            return
        n, fps = self._mt_frames(), int(self.mt_fps.currentData() or mt.DEFAULT_FPS)
        parts = [tr("vedit.mt.info_frames", "{n} frame a {f} fps = {s} s").format(n=n, f=fps, s=f"{n / fps:.1f}")]
        if self.mode.currentData() == "motion_track" and not self.mt_editor.has_tracks():
            parts.append(tr("vedit.mt.info_notracks", "disegna almeno una traccia"))
        self.mt_info.setText("  ·  ".join(parts))

    def _pick(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("vedit.pick", "Scegli video"), "",
                                           "Video (*.mp4 *.mov *.avi *.mkv *.webm)")
        if p:
            self.set_video(p)

    def set_video(self, path: str):
        if path:
            self._video = path
            self.video_lbl.setText(f"🎬 {Path(path).name}")
            self._probe_source()
            self._on_mode()

    def _lipdub_line_text(self) -> str:
        return self.lip_line.toPlainText().strip()

    def _lipdub_lang_value(self) -> str:
        """Codice breve della lingua scelta ('' = non specificata)."""
        return str(self.lip_lang.currentData() or "")

    def set_last_audio_getter(self, fn):
        """`fn()` -> path dell'ultimo audio generato (TTS/canzone) o ''. Collegato da main_window."""
        self._last_audio_getter = fn

    def _pick_lip_audio(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("vedit.lipdub.audio_lbl", "Audio da usare"), "",
                                           "Audio (*.wav *.mp3 *.flac *.m4a *.ogg *.aac)")
        if p:
            self.lip_audio.setText(p)

    def _use_last_audio(self):
        p = ""
        if self._last_audio_getter is not None:
            try:
                p = self._last_audio_getter() or ""
            except Exception:                                # noqa: BLE001
                p = ""
        if p and os.path.isfile(p):
            self.lip_audio.setText(p)
        else:
            self._block("vedit.lipdub.no_last_audio", "Nessun audio generato di recente: scegli un file.")

    def _open_v2v(self, mode: str):
        """Rimando a Talk/Canta del tab Video (esistono gia': non vengono toccate). Passa video e audio se noti."""
        audio = self.lip_audio.text().strip()
        if not audio and self._last_audio_getter is not None:
            try:
                audio = self._last_audio_getter() or ""
            except Exception:                                # noqa: BLE001
                audio = ""
        self.request_open_v2v.emit(mode, self._video, audio if audio and os.path.isfile(audio) else "")

    def _refresh_lipdub_info(self, *_):
        """Riga informativa del doppiaggio: durata del video, stima del parlato / durata dell'audio, avvisi (mai blocchi)."""
        if not hasattr(self, "lip_info"):
            return
        vi = self._vinfo
        key = self.mode.currentData()
        if key not in ("lipdub", "lipdub_audio") or vi is None or not getattr(vi, "duration", 0):
            self.lip_info.setText(""); return
        if key == "lipdub_audio":
            parts = [tr("vedit.lipdub.audio_dur_video", "Video: {v} s").format(v=f"{vi.duration:.1f}")]
            ap = self.lip_audio.text().strip()
            if ap and os.path.isfile(ap):
                from core import video_ic_prep as _prep
                ad = _prep.audio_duration(ap)
                if ad:
                    parts.append(tr("vedit.lipdub.audio_dur_audio", "audio: {a} s").format(a=f"{ad:.1f}"))
                    if ad + 0.3 < vi.duration:
                        parts.append("⚠ " + tr("vedit.lipdub.audio_shorter",
                                               "audio più corto: il video verrà tagliato (l'audio comanda la durata)."))
                    elif ad > vi.duration + 0.3:
                        parts.append("⚠ " + tr("vedit.lipdub.audio_longer",
                                               "audio più lungo: verrà tagliato alla durata del video."))
            self.lip_info.setText("  ·  ".join(parts)); return
        parts = [tr("vedit.lipdub.info_dur", "Video: {v} s (la voce nuova dura quanto il video)").format(
            v=f"{vi.duration:.1f}")]
        if not vi.has_audio:
            parts.append("⛔ " + tr("vedit.err_lipdub_no_audio",
                "Lip dub: il video non ha una traccia audio (serve come riferimento della voce)."))
        line = self._lipdub_line_text()
        if line:
            fit, est = vic.lipdub_text_fit(line, float(vi.duration))
            parts.append(tr("vedit.lipdub.info_est", "battuta ≈ {e} s di parlato (stima)").format(e=f"{est:.1f}"))
            if fit == "long":
                parts.append("⚠ " + tr("vedit.lipdub.fit_long",
                    "battuta lunga: il modello potrebbe saltare delle parole. Accorciala."))
            elif fit == "short":
                parts.append("⚠ " + tr("vedit.lipdub.fit_short",
                    "battuta corta: la voce potrebbe risultare lenta e innaturale. Allungala."))
        self.lip_info.setText("  ·  ".join(parts))

    # ── C: estrai e traduci (STT + LLM locali, in un thread) ─────────────────────────────────
    def _translate_clicked(self):
        self._translate_start(sync=False)

    def _translate_start(self, sync: bool = False):
        if not self._video:
            self._block("vedit.err_no_video", "Scegli prima un video."); return
        target = self._lipdub_lang_value()
        if not target:
            self._block("vedit.dub.need_target", "Scegli la lingua di destinazione."); return
        if self._dub_thread is not None and self._dub_thread.isRunning():
            return
        src_text = self.lip_src_text.toPlainText().strip() if self._dub_src_video == self._video else ""
        dur = float(getattr(self._vinfo, "duration", 0.0) or 0.0)
        video = self._video
        self.lip_translate.setEnabled(False)
        self.log.appendPlainText(tr("vedit.dub.busy", "Estrazione del parlato e traduzione in corso (STT e LLM locali)…"))

        def work():
            return vdt.run(video, target, source_text=src_text, video_seconds=dur,
                           log=lambda m: None)
        if sync:                                             # test/script: nessun thread
            try:
                res, err = work(), None
            except Exception as e:                           # noqa: BLE001
                res, err = None, e
            self._translate_done(res, err, video, target)
            return
        th = _FnThread(work)
        th.finished_ok.connect(lambda r: self._translate_done(r, None, video, target))
        th.failed.connect(lambda e: self._translate_done(None, e, video, target))
        self._dub_thread = th
        th.start()

    def _translate_done(self, res, err, video: str, target: str):
        self.lip_translate.setEnabled(True)
        if err is not None:
            if isinstance(err, vdt.DubTranslateError):
                self.log.appendPlainText("⚠ " + tr(err.key, str(err)))
            else:
                self.log.appendPlainText("⚠ " + tr("vedit.dub.err_generic", "Estrai e traduci non riuscito: {e}").format(e=err))
            return
        self._dub_src_video = video
        self.lip_line.setPlainText(res["text"])
        self.lip_src_text.setPlainText(res.get("original_text", ""))
        ol = (res.get("original_language") or "").lower()
        name = dict(vic.LIPDUB_LANGUAGES).get(ol) or ol or tr("vedit.dub.lang_unknown", "non rilevata")
        self.lip_src_lang.setText(tr("vedit.dub.src_lang", "Lingua rilevata dell'originale: {l}").format(l=name))
        self.log.appendPlainText(tr("vedit.dub.done",
            "Traduzione pronta: controlla e correggi il testo prima di generare.") + f" [{res.get('backend', '')}]")
        if res.get("warning") == "long":
            self.log.appendPlainText("⚠ " + tr("vedit.lipdub.fit_long",
                "battuta lunga: il modello potrebbe saltare delle parole. Accorciala."))
        self._refresh_lipdub_info()

    def _probe_source(self):
        """ffprobe veloce del sorgente (dimensioni/fps/frame): serve ad anteprima outpaint e info segmenti."""
        self._vinfo = None
        try:
            from core import video_ic_prep as prep
            self._vinfo = prep.probe(self._video)
        except Exception:                                    # noqa: BLE001 — file non video/ffprobe assente: solo niente anteprima
            self._vinfo = None

    # ── checkpoint ───────────────────────────────────────────────────────────────────────────
    def set_model_getter(self, fn):
        """`fn()` -> `video_tab.get_values()` (dict) o None. Collegato da main_window."""
        self._model_getter = fn
        self.refresh_checkpoint_hint()

    def showEvent(self, e):                                  # noqa: N802
        super().showEvent(e)
        self.refresh_checkpoint_hint()

    def _checkpoint(self) -> dict:
        vals = None
        if self._model_getter is not None:
            try:
                vals = self._model_getter()
            except Exception:                                # noqa: BLE001
                vals = None
        return vic.checkpoint_info(vals)

    def checkpoint_state_text(self, spec: vem.ModeSpec, info: dict) -> tuple[str, str]:
        """(stato, testo) della riga checkpoint per la modalità. Stati: ok | warn | blocked | info | none."""
        st = info["state"]
        name = info["name"] or tr("vedit.ckpt.auto_name", "automatico")
        head = tr("vedit.ckpt.label", "Checkpoint (tab Video): {m}").format(m=name)
        if st == "unknown":
            return "none", ""
        if st == "release25" and spec.uses_ic_registry and spec.cap:
            return "blocked", head + " — " + tr("vedit.ckpt.release25",
                "⛔ release 2.5: le IC-LoRA locali sono per LTX 2.3. Scegli un checkpoint 2.3 nel tab Video.")
        if st == "distilled":
            return "ok", head + " — " + tr("vedit.ckpt.distilled", "✔ distilled")
        if not info.get("dlora_ok", True):
            return "warn", head + " — " + tr("vedit.ckpt.dlora_missing",
                "⚠ dev: serve la LoRA distilled ({f}) e non è su disco. Scaricala dal tab «Pesi modelli» "
                "(nessun download automatico): senza, il video esce «dematerializzato».").format(
                    f=os.path.basename(str(info.get("dlora_path", ""))))
        if st == "auto":
            return "warn", head + " — " + tr("vedit.ckpt.auto2",
                "ℹ automatico: il worker userà il checkpoint di default (dev) e col dev verrà aggiunta la LoRA "
                "distilled ufficiale. Il checkpoint resta quello scelto.")
        if st == "other_arch":
            return "warn", head + " — " + tr("vedit.ckpt.other_arch2",
                "ℹ il modello scelto nel tab Video non è LTX: qui si userà il checkpoint di default (dev) "
                "con la LoRA distilled ufficiale aggiunta.")
        return "warn", head + " — " + tr("vedit.ckpt.dev2",
            "ℹ dev: col dev verrà aggiunta la LoRA distilled ufficiale (forza 0,5, passi distilled). "
            "Il checkpoint resta quello scelto.")

    def refresh_checkpoint_hint(self, *_):
        spec = self.current_spec() if self.mode.count() else None
        if spec is None:
            return
        st, text = self.checkpoint_state_text(spec, self._checkpoint())
        self.ckpt_lbl.setText(text)
        self.ckpt_lbl.setVisible(bool(text))
        self.ckpt_lbl.setProperty("state", st)

    def _ask_checkpoint(self, spec: vem.ModeSpec, info: dict, blocking: bool = False) -> str:
        """Dialog sul checkpoint dev: AVVISA, non blocca mai (`blocking` è ignorato, resta per compatibilità).
        Ritorna 'continue' | 'cancel' | 'open' (tab Video) | 'weights' (tab Pesi modelli).
        Sovrascrivibile nei test."""
        box = QMessageBox(self)
        missing = not info.get("dlora_ok", True)
        box.setIcon(QMessageBox.Warning if missing else QMessageBox.Information)
        box.setWindowTitle(tr("vedit.ckpt.dlg_title2", "Checkpoint dev"))
        _, text = self.checkpoint_state_text(spec, info)
        cb = None
        if missing:
            box.setText(tr("vedit.ckpt.dlg_missing",
                "Col checkpoint dev serve la LoRA distilled ufficiale e non è su disco: senza, le IC-LoRA danno "
                "video sotto-denoisati. Scaricala dal tab «Pesi modelli» (non scarico nulla io).") + "\n\n" + text)
        else:
            box.setText(tr("vedit.ckpt.dlg_body2",
                "Il checkpoint è un dev: verrà aggiunta la LoRA distilled ufficiale (forza 0,5, sia stage 1 che "
                "stage 2, passi distilled) come nei workflow Lightricks. Il checkpoint resta quello che hai scelto.")
                + "\n\n" + text)
            cb = QCheckBox(tr("vedit.ckpt.dont_repeat", "Non ripetere in questa sessione"))
            box.setCheckBox(cb)
        b_go = box.addButton(tr("vedit.ckpt.btn_continue", "Procedi comunque"), QMessageBox.AcceptRole)
        b_w = (box.addButton(tr("vedit.ckpt.btn_open_weights_tab", "Apri tab Pesi modelli"), QMessageBox.ActionRole)
               if missing else None)
        b_open = box.addButton(tr("vedit.ckpt.btn_open_video_tab", "Apri tab Video"), QMessageBox.ActionRole)
        box.addButton(tr("vedit.ckpt.btn_cancel", "Annulla"), QMessageBox.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is b_go:
            if cb is not None and cb.isChecked():
                VideoEditAITab._dev_ack_session = True
            return "continue"
        if b_w is not None and clicked is b_w:
            return "weights"
        return "open" if clicked is b_open else "cancel"

    def _ask_missing_weight(self, spec: vem.ModeSpec, msg: str) -> str:
        """Peso IC-LoRA della modalità assente su disco: avviso con «Apri tab Pesi modelli» (nessun download
        automatico). Ritorna 'weights' | 'cancel'. Sovrascrivibile nei test."""
        self.log.appendPlainText("⚠ " + msg)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(tr("vedit.weight.missing_title", "Peso mancante"))
        box.setText(tr("vedit.weight.missing_body",
                       "Manca il peso di questa modalità (IC-LoRA «{k}»). Non scarico nulla io: aprilo dal tab "
                       "«Pesi modelli».").format(k=spec.kind) + "\n\n" + msg)
        b_w = box.addButton(tr("vedit.ckpt.btn_open_weights_tab", "Apri tab Pesi modelli"), QMessageBox.ActionRole)
        box.addButton(tr("vedit.ckpt.btn_cancel", "Annulla"), QMessageBox.RejectRole)
        box.exec()
        return "weights" if box.clickedButton() is b_w else "cancel"

    # ── UI per modalità ──────────────────────────────────────────────────────────────────────
    def _on_mode(self, *_):
        if not self.mode.count():
            return
        sp = self.current_spec()
        key = sp.key
        is_rc = (key == "replace_char")
        is_restyle = (key == "restyle")
        is_cam = (key == "cameraman")
        is_mt = (key == "motion_track")                     # movimento da tracce disegnate
        is_out = (key == "outpaint")
        is_lipt = (key == "lipdub")                          # doppiaggio da TESTO (ricetta ufficiale)
        is_lipa = (key == "lipdub_audio")                    # doppiaggio da AUDIO (sperimentale)
        is_lip = is_lipt or is_lipa
        is_hdr = (key == "hdr")                              # SDR → HDR: nessun prompt, forza/attenzione fisse
        is_repair = (sp.family == vem.FAMILY_REPAIR)
        is_ic_new = bool(sp.cap)                              # modalità IC-LoRA a registro (non storiche)
        # #102: il campo «oggetto» serve all'inpaint E a replace_char (= personaggio da sostituire, per la
        # maschera SAM2 di precisione — opzionale: vuoto = sostituzione su tutto il frame).
        self.obj.setEnabled(key == "inpaint" or is_rc)
        self.obj.setPlaceholderText(
            tr("vedit.obj_ph_rc", "personaggio da sostituire (es. person) — opz., limita al soggetto") if is_rc
            else tr("vedit.obj_ph", "oggetto da selezionare (es. car, person) — per Inpaint"))
        for w in (self.ref_img, self.ref_btn, self.ref_row_lbl):
            w.setEnabled(is_rc or is_restyle)
        for w in (self.ref_strength, self.ref_strength_lbl):
            w.setEnabled(is_rc)
        self.ref_row_lbl.setText(tr("vedit.style.image_lbl", "Immagine di stile") if is_restyle
                                 else tr("vedit.ref_lbl", "Immagine sostitutiva (personaggio)"))
        self.ref_img.setPlaceholderText(
            tr("vedit.style.image_ph", "immagine che definisce lo stile (diventa il frame 0 della guida)") if is_restyle
            else tr("vedit.ref.ph", "immagine sostitutiva: = primo frame del video ma con l'altro personaggio"))
        # visibilità righe
        self._show_row(self.obj, not is_ic_new)
        self._set_ref_row_visible(is_rc or is_restyle)
        self._show_row(self.ref_strength, is_rc)
        self._show_row(self.style_text, is_restyle)
        self._set_row_visible_for(self.cam_ref, is_cam)
        self._set_row_visible_for(self.cam_img, is_cam)
        self._set_row_visible_for(self.cam_w, is_cam)
        for w in (self.cam_img_strength, self.cam_seconds):
            self._show_row(w, is_cam)
        self._set_row_visible_for(self.mt_img, is_mt)
        for w in (self.mt_editor, self.mt_info, self.mt_img_strength, self.mt_seed, self.mt_audio):
            self._show_row(w, is_mt)
        for w in (self.mt_w, self.mt_seconds):
            self._set_row_visible_for(w, is_mt)
        for w in (self.out_format, self.out_anchor, self.out_dark, self.out_paste, self.out_feather, self.out_minbar, self.out_fill,
                  self.out_info, self.out_preview):
            self._show_row(w, is_out)
        self._set_row_visible_for(self.out_m["top"], is_out and self.out_format.currentData() == "custom")
        self.out_anchor.setEnabled(is_out and self.out_format.currentData() != "custom")
        for w in (self.lip_scene, self.lip_line, self.lip_lang, self.lip_info):
            self._show_row(w, is_lip)
        for w in (self.lip_translate, self.lip_src_lang, self.lip_src_text, self.lip_exp):
            self._show_row(w, is_lipt if w is not self.lip_exp else is_lipa)
        self._set_row_visible_for(self.lip_audio, is_lipa)
        self._show_row(self.lip_audio_ref, is_lipa)
        self._set_row_visible_for(self.lip_open_talk, is_lipt)
        self.lip_line_lbl.setText(tr("vedit.lipdub.line_opt_lbl", "Testo detto nell'audio (opz.)") if is_lipa
                                  else tr("vedit.lipdub.line_lbl", "Nuova battuta (testo)"))
        self.lip_lang_lbl.setText(tr("vedit.lipdub.lang_lbl", "Lingua parlata (opz.)") if is_lipa
                                  else tr("vedit.lipdub.lang_dest_lbl", "Lingua di destinazione"))
        self.prompt.setPlaceholderText(
            tr("vedit.lipdub.prompt_ph", "avanzato (opzionale): prompt completo; se lo scrivi, scena/battuta/lingua "
                                         "sopra vengono ignorate") if is_lip else self._prompt_ph)
        self._show_row(self.prompt, sp.needs_prompt != vem.NEEDS_NONE)   # hdr: il testo e' il file scene-emb
        self._show_row(self.vedit_enhance, not is_lip and not is_hdr)    # l'LLM potrebbe riscrivere la battuta
        self._show_row(self.attn, is_ic_new and not is_lip and not is_hdr)
        for w in (self.hdr_format, self.hdr_exposure, self.hdr_exr_half, self.hdr_hq, self.hdr_info):
            self._show_row(w, is_hdr)
        self._set_row_visible_for(self.res_mode, is_repair or is_restyle or is_lip)
        self.res_long.setEnabled(self.res_mode.currentData() == "custom")
        self._show_row(self.keep_audio, is_ic_new and sp.keep_audio)
        for w in (self.cfg, self.guide):
            self._show_row(w, key == "inpaint")
        for w in (self.dilate, self.blur):
            self._show_row(w, key in ("inpaint", "replace_char"))
        self._show_row(self.strength, not is_cam and not is_mt and not is_lip and not is_hdr)   # lip dub/hdr: forza 1.0 fissa (ufficiale)
        self.strength.setEnabled(not is_restyle)          # ReStyle: forza 1.0 fissa (card)
        self.btn_pick.setEnabled(True)
        self.btn_pick.setText(tr("vedit.pick", "🎬 Scegli video…") if not (is_cam or is_mt)
                              else tr("vedit.pick_opt", "🎬 Scegli video (opzionale)…"))
        self.mt_img_frame.setEnabled(bool(self._video))
        if is_mt:
            self._mt_timing_changed()
        # prompt precompilato (modalità di riparazione)
        cur = self.prompt.toPlainText().strip()
        if sp.default_prompt_key:
            dflt = tr(sp.default_prompt_key, _RESTORE_PROMPT_DEFAULT)
            if not cur or cur == self._auto_prompt:
                self.prompt.setPlainText(dflt); self._auto_prompt = dflt
        elif cur and cur == self._auto_prompt:
            self.prompt.clear(); self._auto_prompt = ""
        # note
        nd = _NOTE_DEFAULT.get(sp.note_key, "")
        if key == "inpaint":
            self.note.setText(tr("vedit.note_inpaint",
                "⚠ Inpaint sperimentale: SAM2 traccia l'oggetto su tutto il clip e ne rigenera l'area."))
        elif is_rc:
            self.note.setText(tr("vedit.note_replace_char",
                "Sostituisci personaggio: carica un'immagine = primo frame del video ma con SOLO il personaggio "
                "cambiato. Il canny preserva scena/movimento; l'audio è mantenuto. Indica il «personaggio da "
                "sostituire» (a parole) per limitare la sostituzione al soggetto (SAM2, sfondo intatto); "
                "vuoto = tutto il frame. Video lunghi a segmenti."))
        else:
            self.note.setText(tr(sp.note_key, nd) if sp.note_key else "")
        if is_out:
            if self._vinfo is not None and self._vinfo.width and not self.out_dark.isChecked():
                self._suggest_dark()
            self._on_outpaint_change()
        self._refresh_segments_info(sp)
        self._refresh_lipdub_info()
        self._refresh_hdr_info()
        self.refresh_checkpoint_hint()

    def _set_ref_row_visible(self, vis: bool):
        self.ref_row_lbl.setVisible(vis)
        self.ref_img.parentWidget().setVisible(vis)

    def _set_row_visible_for(self, inner, vis: bool):
        """Riga il cui campo è un contenitore: nasconde il contenitore e la sua etichetta."""
        w = inner.parentWidget()
        if w is None or w is self:
            w = inner
        self._show_row(w, vis)

    def _suggest_dark(self):
        """Suggerisce (mai forza senza dirlo) il gamma per sorgenti scure: luma media del primo secondo."""
        try:
            from core import video_ic_prep as prep
            if prep.looks_dark(self._video):
                self.out_dark.setChecked(True)
                self.log.appendPlainText(tr("vedit.out.dark_auto",
                    "[outpaint] sorgente scura: attivato «Sorgente scura (gamma 2.0 / 0.5)». Puoi disattivarlo."))
        except Exception:                                    # noqa: BLE001
            pass

    def _outpaint_layout(self):
        """Layout dell'outpaint dal sorgente noto (o None)."""
        if self._vinfo is None or not self._vinfo.width:
            return None
        from core import video_ic_prep as prep
        sp = vem.get_spec("outpaint")
        fmt = self.out_format.currentData()
        kw = {"anchor": self.out_anchor.currentData(), "max_long_edge": sp.max_long_edge}
        if fmt == "custom":
            kw["margins"] = (self.out_m["top"].value(), self.out_m["right"].value(),
                             self.out_m["bottom"].value(), self.out_m["left"].value())
        else:
            kw["aspect"] = fmt
            if self.out_minbar.isChecked():
                kw["min_bar"] = prep.MIN_BAR_FRACTION
        if self.out_fill.isChecked():
            kw["fill_area"] = prep.FILL_AREA
        try:
            return prep.compute_outpaint_layout(self._vinfo.width, self._vinfo.height, **kw)
        except ValueError:
            return None

    @staticmethod
    def _adds_text(lay) -> str:
        """«  ·  Aggiunge: sopra 96 px, sotto 96 px»: dice con chiarezza in quale direzione si estende il video."""
        sides = (("top", "vedit.out.side.top", "sopra {n} px", lay.y),
                 ("bottom", "vedit.out.side.bottom", "sotto {n} px", lay.canvas_h - lay.y - lay.src_h),
                 ("left", "vedit.out.side.left", "a sinistra {n} px", lay.x),
                 ("right", "vedit.out.side.right", "a destra {n} px", lay.canvas_w - lay.x - lay.src_w))
        parts = [tr(k, d).format(n=n) for _s, k, d, n in sides if n > 4]
        if not parts:
            return ""
        return "  ·  " + tr("vedit.out.adds", "Aggiunge: {where}").format(where=", ".join(parts))

    @staticmethod
    def _narrow_bar_text() -> str:
        return tr("vedit.out.narrow_bar", "barra nuova troppo stretta (< 15% del canvas): il modello potrebbe non riempirla. Usa margini più larghi.")

    def _on_outpaint_change(self, *_):
        custom = self.out_format.currentData() == "custom"
        self._set_row_visible_for(self.out_m["top"], self.mode.currentData() == "outpaint" and custom)
        self.out_anchor.setEnabled(not custom)
        self.out_minbar.setEnabled(not custom)
        self.out_feather.setEnabled(self.out_paste.isChecked())
        self.out_feather_lbl.setEnabled(self.out_paste.isChecked())
        lay = self._outpaint_layout()
        pm = QPixmap(self.out_preview.size()); pm.fill(QColor("#101010"))
        if lay is None:
            self.out_info.setText(tr("vedit.out.no_info", "Scegli un video per vedere canvas e area nuova."))
            self.out_preview.setPixmap(pm); return
        warn = ("  ⚠ " + tr("vedit.out.too_big", "area nuova molto grande (> 60%): il modello inventerà molto.")) \
            if lay.too_much_new_area else ""
        if lay.narrow_bar and not (self.out_fill.isChecked() and lay.scale < 0.999):
            warn += "  ⚠ " + self._narrow_bar_text()
        if self.out_fill.isChecked() and lay.scale < 0.999 and not lay.clamped:
            warn += "  ℹ " + tr("vedit.out.fill_scaled", "originale ridotto al {p}% per far riempire le barre").format(
                p=int(round(lay.scale * 100)))
        self.out_info.setText(tr("vedit.out.canvas_info", "Canvas: {w}x{h} (area nuova {p}%)").format(
            w=lay.canvas_w, h=lay.canvas_h, p=int(round(lay.new_area * 100))) + self._adds_text(lay) + warn)
        # anteprima: canvas scalato nel box, sorgente grigio chiaro, area nuova nera tratteggiata
        W, H = pm.width(), pm.height()
        s = min((W - 8) / lay.canvas_w, (H - 8) / lay.canvas_h)
        cw, ch = lay.canvas_w * s, lay.canvas_h * s
        ox, oy = (W - cw) / 2, (H - ch) / 2
        pt = QPainter(pm)
        pt.fillRect(int(ox), int(oy), int(cw), int(ch), QColor("#000000"))
        pen = QPen(QColor("#ffffff")); pen.setStyle(Qt.DashLine); pt.setPen(pen)
        pt.drawRect(int(ox), int(oy), int(cw), int(ch))
        pt.fillRect(int(ox + lay.x * s), int(oy + lay.y * s), int(lay.src_w * s), int(lay.src_h * s), QColor("#c8c8c8"))
        pt.end()
        self.out_preview.setPixmap(pm)

    def _refresh_segments_info(self, sp: vem.ModeSpec | None = None):
        sp = sp or self.current_spec()
        n = int(getattr(self._vinfo, "nb_frames", 0) or 0) if self._vinfo else 0
        if not (sp.cap and (sp.chunkable or sp.own_segments) and n):
            self.seg_info.setText(""); return
        k = vem.segments_needed(vem.snap_frames(n), sp)
        self.seg_info.setText(tr("vedit.segments_info", "{n} segmenti da {f} frame").format(
            n=k, f=sp.max_frames_per_segment) if k > 1 else "")

    def _refresh_hdr_info(self, *_):
        """Riga informativa della modalità HDR: dove stanno i file d'uscita e avviso se il sorgente è già a 10 bit."""
        if not self.mode.count():
            return
        is_hdr = (self.mode.currentData() == "hdr")
        fmt = self.hdr_format.currentData()
        self._show_row(self.hdr_exr_half, is_hdr and fmt in ("exr", "both"))
        if not is_hdr:
            self.hdr_info.setText(""); return
        parts = [tr("vedit.hdr.info_out",
                    "Il player mostra l'anteprima SDR; il file HDR vero (_HDR_PQ.mp4) e/o la cartella _EXR si trovano "
                    "accanto, con lo stesso nome.")]
        vi = self._vinfo
        if vi is not None and any(t in str(getattr(vi, "pix_fmt", "")) for t in ("10le", "10be", "12le", "12be", "p010")):
            parts.append("⚠ " + tr("vedit.hdr.info_src10",
                "Il sorgente è a {p}: la LoRA HDR vuole un SDR Rec.709 a 8 bit; un video già HDR (PQ/HLG) non va dato "
                "in ingresso.").format(p=vi.pix_fmt))
        self.hdr_info.setText("  ".join(parts))

    # ── azione ───────────────────────────────────────────────────────────────────────────────
    def _block(self, key: str, default: str, **fmt):
        txt = tr(key, default)
        self.log.appendPlainText(txt.format(**fmt) if fmt else txt)

    def _go(self):
        sp = self.current_spec()
        key = sp.key
        is_cam = (key == "cameraman")
        if is_cam:
            if not self.cam_ref.text().strip():
                self._block("vedit.err_no_cam_ref", "Cameraman: scegli il video di riferimento del movimento."); return
        elif key == "motion_track":
            pass                                             # niente video: immagine + tracce (controlli sotto)
        elif not self._video:
            self._block("vedit.err_no_video", "Scegli prima un video."); return
        lora = next((l for k, _i, l in MODES if k == key), "")
        prompt = self.prompt.toPlainText().strip()
        if key == "restyle" and not prompt and self.style_text.text().strip():
            prompt = tr("vedit.style.tpl", "Make it {s} style.").format(s=self.style_text.text().strip())
        cfg = {"video": self.cam_ref.text().strip() if is_cam else self._video, "mode": key, "edit_lora": lora,
               "object": self.obj.text().strip(), "prompt": prompt,
               "strength": float(self.strength.currentData()),
               "cfg_scale": float(self.cfg.value()), "guide": float(self.guide.value()),
               "mask_dilate": int(self.dilate.value()), "mask_blur": float(self.blur.value()),
               "enhance": self.vedit_enhance.isChecked(),
               "ref_image": self.ref_img.text().strip(),
               "ref_strength": float(self.ref_strength.value())}   # #102 forza identità del riferimento
        if key == "inpaint" and not cfg["object"]:
            self._block("vedit.err_no_obj", "Inpaint: indica l'oggetto da selezionare (a parole)."); return
        if key == "replace_char" and not cfg["ref_image"]:
            self._block("vedit.err_no_ref", "Sostituisci personaggio: scegli l'immagine sostitutiva del personaggio."); return
        if key == "restyle":
            if not cfg["ref_image"]:
                self._block("vedit.err_no_style", "ReStyle: scegli l'immagine di stile."); return
            if not prompt:
                self._block("vedit.err_no_style_text", "ReStyle: scrivi lo stile (es. «Disney 2D Animation») o un prompt."); return
            cfg["style_image"] = cfg["ref_image"]
        if key in ("lipdub", "lipdub_audio"):
            vi = self._vinfo
            if key == "lipdub":
                if not cfg["prompt"] and not self._lipdub_line_text():
                    self._block("vedit.err_no_line", "Lip dub: scrivi la nuova battuta (o un prompt completo)."); return
                if vi is not None and not vi.has_audio:
                    self._block("vedit.err_lipdub_no_audio",
                                "Lip dub: il video non ha una traccia audio (serve come riferimento della voce)."); return
            else:
                ap = self.lip_audio.text().strip()
                if not ap or not os.path.isfile(ap):
                    self._block("vedit.err_lipdub_no_audio_file",
                                "Doppiaggio da audio: scegli il file audio da usare."); return
                if self.lip_audio_ref.currentData() == "original" and vi is not None and not vi.has_audio:
                    self._block("vedit.err_lipdub_no_audio",
                                "Lip dub: il video non ha una traccia audio (serve come riferimento della voce)."); return
                cfg.update(lipdub_audio=ap, lipdub_audio_ref=self.lip_audio_ref.currentData())
            n = int(getattr(vi, "nb_frames", 0) or 0) if vi is not None else 0
            if n and vem.snap_frames(n) > sp.max_frames_per_segment:
                self._block("vedit.err_lipdub_too_long",
                            "Lip dub: il video e' troppo lungo ({n} frame, massimo {m}): la modalita' non si spezza "
                            "in segmenti. Accorcia il clip.", n=n, m=sp.max_frames_per_segment); return
            cfg.update(lipdub_line=self._lipdub_line_text(), lipdub_lang=self._lipdub_lang_value(),
                       lipdub_scene=self.lip_scene.text().strip(), enhance=False)
        elif sp.needs_prompt == vem.NEEDS_REQUIRED and not cfg["prompt"] and key != "replace_char":
            self._block("vedit.err_no_prompt", "Inserisci un prompt."); return
        # campi delle nuove modalità IC
        if sp.cap:
            cfg["conditioning_attention"] = float(self.attn.value())
            cfg["keep_audio"] = bool(self.keep_audio.isChecked()) and sp.keep_audio
        if key == "hdr":
            cfg.update(hdr_format=self.hdr_format.currentData(), hdr_exposure_ev=float(self.hdr_exposure.value()),
                       hdr_exr_half=self.hdr_exr_half.isChecked(), hdr_hq=self.hdr_hq.isChecked())
        if sp.family == vem.FAMILY_REPAIR or key == "restyle":
            cfg["res_mode"] = self.res_mode.currentData()
            cfg["res_long"] = int(self.res_long.value())
        if key == "outpaint":
            cfg.update(out_format=self.out_format.currentData(), out_anchor=self.out_anchor.currentData(),
                       out_margins=(self.out_m["top"].value(), self.out_m["right"].value(),
                                    self.out_m["bottom"].value(), self.out_m["left"].value()),
                       out_dark=self.out_dark.isChecked(), out_paste=self.out_paste.isChecked(),
                       out_feather=int(self.out_feather.value()), out_fill=self.out_fill.isChecked(),
                       out_min_bar=(prep_mod.MIN_BAR_FRACTION if self.out_minbar.isChecked()
                                    and self.out_format.currentData() != "custom" else 0.0))
            if cfg["out_format"] == "custom" and not any(cfg["out_margins"]):
                self._block("vedit.err_no_margins", "Outpaint: indica almeno un margine (o scegli un formato)."); return
            lay = self._outpaint_layout()
            if lay is not None and lay.new_area <= 0.001:
                self._block("vedit.err_no_new_area", "Outpaint: nessuna area nuova da generare con questo formato."); return
            if lay is not None and lay.too_much_new_area:
                self.log.appendPlainText(tr("vedit.out.too_big", "area nuova molto grande (> 60%): il modello inventerà molto."))
            if lay is not None and lay.narrow_bar:
                self.log.appendPlainText("⚠ " + self._narrow_bar_text())
        if is_cam:
            w, h = int(self.cam_w.value()), int(self.cam_h.value())
            cfg.update(width=w, height=h, cam_ref=cfg["video"], cam_img=self.cam_img.text().strip(),
                       cam_img_strength=float(self.cam_img_strength.value()),
                       cam_seconds=float(self.cam_seconds.value()))
            if max(w, h) < sp.min_long_edge or min(w, h) < 512:
                self.log.appendPlainText(tr("vedit.cam.too_small",
                    "⚠ Risoluzione sotto 960×512: il riferimento viene seguito peggio."))
        if key == "motion_track":
            img = self.mt_img.text().strip()
            if not img or not os.path.isfile(img):
                self._block("vedit.err_no_track_image", "Motion-track: scegli l'immagine di partenza."); return
            if not self.mt_editor.has_tracks():
                self._block("vedit.err_no_tracks", "Motion-track: disegna almeno una traccia sull'immagine."); return
            w, h = int(self.mt_w.value()), int(self.mt_h.value())
            if (self.mt_editor.model.width, self.mt_editor.model.height) != (w, h):
                self.mt_editor.set_reference(img, w, h)      # le tracce seguono la risoluzione scelta
            n, fps = self._mt_frames(), int(self.mt_fps.currentData() or mt.DEFAULT_FPS)
            cfg.update(video=img, track_image=img, tracks=self.mt_editor.tracks_json(n), tracks_size=[w, h],
                       width=w, height=h, fps=fps, num_frames=n, seed=int(self.mt_seed.value()),
                       image_strength=float(self.mt_img_strength.value()), strength=1.0,
                       native_audio=bool(self.mt_audio.isChecked()))
        if sp.kind and sp.cap:                               # peso IC-LoRA assente: avviso + «Apri tab Pesi», nessun download
            miss = vic.missing_weight(sp)
            if miss:
                if self._ask_missing_weight(sp, miss) == "weights":
                    self.request_open_weights_tab.emit()
                return
        # checkpoint: mostrato, avvisato, MAI cambiato
        info = self._checkpoint()
        st = info["state"]
        if st == "release25" and sp.cap:
            self._block("vedit.ckpt.release25",
                        "⛔ release 2.5: le IC-LoRA locali sono per LTX 2.3. Scegli un checkpoint 2.3 nel tab Video.")
            return
        if st != "unknown" and not info["distilled"] and st != "release25":
            # dev/auto/non-LTX: AVVISO, mai blocco (il worker aggiunge la distilled-LoRA). Se la LoRA manca su
            # disco l'avviso compare sempre (anche con «non ripetere»); altrimenti una volta per sessione.
            if not info.get("dlora_ok", True) or not VideoEditAITab._dev_ack_session:
                ans = self._ask_checkpoint(sp, info, False)
                if ans == "open":
                    self.request_open_video_tab.emit(); return
                if ans == "weights":
                    self.request_open_weights_tab.emit(); return
                if ans != "continue":
                    return
        cfg.update(info["params"])
        cfg["ckpt_state"] = st
        self.log.appendPlainText(tr("vedit.queued", "In elaborazione: {m}…").format(m=key))
        self.request_edit.emit(cfg)
