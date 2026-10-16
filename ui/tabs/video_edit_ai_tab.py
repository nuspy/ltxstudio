"""Tab «Modifica Video AI» (#24/#59/#60 + Fase 1 IC-LoRA) — editing video nativo con LoRA LTX + tracking SAM2.

UI a DUE livelli (famiglia → modalità), derivata da `core/video_edit_modes.py` (fonte unica):
- **Modifica**: edit-anything (V2V da istruzione), inpaint oggetto (SAM2, sperimentale), sostituisci personaggio.
- **Stile**: anime↔real, ReStyle (immagine di stile).
- **Ripara video**: refocus / uncompress / deinterlace / motion deblur / detailer (guida = video sorgente).
- **Outpaint video**: estende il quadro (formato o margini) con IC-LoRA outpaint.
- **Camera / Movimento**: Cameraman (video di riferimento del movimento).
(Lip dub e HDR: fasi 1b/1c, `enabled=False` nelle spec → non compaiono.)

Il checkpoint è quello scelto nel tab Video: lo si MOSTRA in una riga di stato e, per le modalità che lo
raccomandano/richiedono distilled, si avvisa/blocca. MAI scelto o cambiato al posto dell'utente.

Emette `request_edit` con la config; l'orchestrazione (tracking + worker) è in main_window.
"""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from ui import theme  # noqa: E402
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from core import video_edit_ic as vic
from core import video_edit_modes as vem
from core import video_ic_prep as prep_mod
from ui.i18n_manager import tr

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
}
_FAMILY_DEFAULT = {"edit": "Modifica", "style": "Stile", "repair": "Ripara video", "outpaint": "Outpaint video",
                   "lipdub": "Lip dub", "camera": "Camera / Movimento"}
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
    "vedit.note_cameraman": "Il movimento di camera viene copiato dal video di riferimento. Se il risultato "
                            "ricalca troppo il riferimento, abbassa la forza dell'immagine o cambia seme.",
}
_RESTORE_PROMPT_DEFAULT = "Sharp, clean, high-quality footage, natural colors."


class VideoEditAITab(QWidget):
    request_edit = Signal(dict)
    request_open_video_tab = Signal()       # «Apri tab Video» dal dialog sul checkpoint
    request_open_weights_tab = Signal()     # «Apri tab Pesi modelli» (distilled-LoRA mancante)
    # «Non ripetere» dell'avviso dev: SOLO per la sessione (non persistente)
    _dev_ack_session = False

    def __init__(self):
        super().__init__()
        self._video = ""
        self._vinfo = None                  # VideoInfo del sorgente (ffprobe veloce) o None
        self._model_getter = None
        self._auto_prompt = ""              # ultimo testo di prompt precompilato dalla modalità
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
        self.out_info = QLabel(""); self.out_info.setWordWrap(True)
        self.out_info_lbl = QLabel("")
        f.addRow(self.out_info_lbl, self.out_info)
        self.out_preview = QLabel(); self.out_preview.setFixedSize(220, 124)
        self.out_preview.setAccessibleName(tr("vedit.out.preview", "Anteprima area nuova"))
        self.out_preview_lbl = QLabel(tr("vedit.out.preview", "Anteprima area nuova"))
        f.addRow(self.out_preview_lbl, self.out_preview)

        self.prompt = QPlainTextEdit(); self.prompt.setFixedHeight(70)
        self.prompt.setPlaceholderText(tr("vedit.prompt_ph", "cosa generare/cambiare (es. a blue motorcycle / snowy winter)"))
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

    # ── UI per modalità ──────────────────────────────────────────────────────────────────────
    def _on_mode(self, *_):
        if not self.mode.count():
            return
        sp = self.current_spec()
        key = sp.key
        is_rc = (key == "replace_char")
        is_restyle = (key == "restyle")
        is_cam = (key == "cameraman")
        is_out = (key == "outpaint")
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
        for w in (self.out_format, self.out_anchor, self.out_dark, self.out_paste, self.out_feather, self.out_minbar,
                  self.out_info, self.out_preview):
            self._show_row(w, is_out)
        self._set_row_visible_for(self.out_m["top"], is_out and self.out_format.currentData() == "custom")
        self.out_anchor.setEnabled(is_out and self.out_format.currentData() != "custom")
        self._show_row(self.attn, is_ic_new)
        self._set_row_visible_for(self.res_mode, is_repair or is_restyle)
        self.res_long.setEnabled(self.res_mode.currentData() == "custom")
        self._show_row(self.keep_audio, is_ic_new and sp.keep_audio)
        for w in (self.cfg, self.guide):
            self._show_row(w, key == "inpaint")
        for w in (self.dilate, self.blur):
            self._show_row(w, key in ("inpaint", "replace_char"))
        self._show_row(self.strength, not is_cam)
        self.strength.setEnabled(not is_restyle)          # ReStyle: forza 1.0 fissa (card)
        self.btn_pick.setEnabled(True)
        self.btn_pick.setText(tr("vedit.pick", "🎬 Scegli video…") if not is_cam
                              else tr("vedit.pick_opt", "🎬 Scegli video (opzionale)…"))
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
        if lay.narrow_bar:
            warn += "  ⚠ " + self._narrow_bar_text()
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
        if not (sp.cap and sp.chunkable and n):
            self.seg_info.setText(""); return
        k = vem.segments_needed(vem.snap_frames(n), sp)
        self.seg_info.setText(tr("vedit.segments_info", "{n} segmenti da {f} frame").format(
            n=k, f=sp.max_frames_per_segment) if k > 1 else "")

    # ── azione ───────────────────────────────────────────────────────────────────────────────
    def _block(self, key: str, default: str):
        self.log.appendPlainText(tr(key, default))

    def _go(self):
        sp = self.current_spec()
        key = sp.key
        is_cam = (key == "cameraman")
        if is_cam:
            if not self.cam_ref.text().strip():
                self._block("vedit.err_no_cam_ref", "Cameraman: scegli il video di riferimento del movimento."); return
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
        if sp.needs_prompt == vem.NEEDS_REQUIRED and not cfg["prompt"] and key != "replace_char":
            self._block("vedit.err_no_prompt", "Inserisci un prompt."); return
        if not cfg["prompt"] and key not in ("replace_char",) and sp.needs_prompt == vem.NEEDS_REQUIRED:
            self._block("vedit.err_no_prompt", "Inserisci un prompt."); return
        # campi delle nuove modalità IC
        if sp.cap:
            cfg["conditioning_attention"] = float(self.attn.value())
            cfg["keep_audio"] = bool(self.keep_audio.isChecked()) and sp.keep_audio
        if sp.family == vem.FAMILY_REPAIR or key == "restyle":
            cfg["res_mode"] = self.res_mode.currentData()
            cfg["res_long"] = int(self.res_long.value())
        if key == "outpaint":
            cfg.update(out_format=self.out_format.currentData(), out_anchor=self.out_anchor.currentData(),
                       out_margins=(self.out_m["top"].value(), self.out_m["right"].value(),
                                    self.out_m["bottom"].value(), self.out_m["left"].value()),
                       out_dark=self.out_dark.isChecked(), out_paste=self.out_paste.isChecked(),
                       out_feather=int(self.out_feather.value()),
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
