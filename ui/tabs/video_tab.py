"""Tab Video: caratteristiche video (fps, risoluzione, durata, NSFW/SFW, modello),
+ Sampler/Seed (spostati da Avanzate), + LoRA LTX-2 (integrato dal tab LoRA).
Il selettore NSFW/SFW del video re-filtra istantaneamente modelli e LoRA."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
    QHBoxLayout, QLabel, QGroupBox, QSlider, QCheckBox, QPushButton,
    QPlainTextEdit, QToolButton, QInputDialog, QMessageBox, QScrollArea, QLineEdit, QTabWidget,
)

from ui.tabs.advanced_tab import SamplerSeedGroup
from ui.tabs.lora_tab import LoraTab
from core.model_catalog import ModelCatalog
from core.presets import VIDEO_PRESETS
from core import user_prefs
from ui.widgets.tag_palette import install_tag_palette
from ui.i18n_manager import tr

# Default per il prompt negativo (spostato qui dal tab Frame 1)
DEFAULT_NEG = ("pc game, console game, video game, cartoon, childish, ugly, deformed, "
               "low quality, blurry, watermark, text")

# Fallback hardcoded mostrato quando l'inventario live non è ancora disponibile.
# La distinzione tra "modalità qualità" Distilled/Base è stata rimossa: tutti i modelli
# LTX-2.3 vengono esposti uniformemente, nessun filtro/predefinito.
MODEL_FILES = {
    "STANDARD 1.1 (distilled fp8)": "LTXVideo\\v2\\ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors",
    "BASE dev fp8 (qualità)":        "LTX\\ltx-2.3-22b-dev-fp8.safetensors",
    "10EROS v1.5 (NSFW)":             "LTX\\10Eros_v1.5_fp8mixed_experimental_learned.safetensors",
}
_MODEL_NSFW = {"10EROS v1.5 (NSFW)"}      # voci modello considerate NSFW (nascoste in SFW)
GGUF_FALLBACK_FILE = "LTXVideo\\LTX-2\\unsloth\\distilled-1.1\\ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf"

RES_PRESETS = {
    "Verticale 9:16 (704x1280)":          (704, 1280),
    "Orizzontale 16:9 (1280x704)":        (1280, 704),
    "Quadrato 1:1 (768x768)":             (768, 768),
    "Classico 4:3 (960x704)":             (960, 704),
    "Ritratto 3:4 (704x960)":             (704, 960),
    "Ultrawide 21:9 (1344x576)":          (1344, 576),
    "Anamorphic 2.39:1 (1280x544)":       (1280, 544),
    "Cinemascope 2.35:1 (1216x512)":      (1216, 512),
    "Flat 1.85:1 (1024x544)":             (1024, 544),
    "Verticale piccolo (480x832)":         (480, 832),
    "Orizzontale piccolo (832x480)":       (832, 480),
    "Personalizzato":                       None,
}


def round32(v: int) -> int:
    return max(32, int(round(v / 32.0)) * 32)


class VideoTab(QWidget):
    request_pulid_keyframe = Signal(str, str)   # (foto|foto|…, prompt) → genera il primo frame col volto (PuLID)

    def __init__(self):
        super().__init__()
        self._mcat = ModelCatalog()
        self._diffusion_models: list[str] = []
        self._checkpoint_models: list[str] = []
        self._gguf_models: list[str] = []
        self._vram_gb: float | None = None
        self._auto_selected: bool = False
        # L'auto-select del modello migliore scatta solo dopo che main_window ha consegnato
        # l'inventario reale (e non sul fallback hardcoded MODEL_FILES dell'init).
        self._inventory_ready: bool = False
        # Contenuto in un'area scrollabile: la sezione impostazioni è lunga e senza scroll
        # i campi si sovrapponevano su finestre basse.
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QScrollArea.NoFrame)
        host = QWidget(); root = QVBoxLayout(host); sc.setWidget(host); outer.addWidget(sc, 1)
        # FOOTER ANCORATO (fuori dallo scroll → sempre visibile anche scorrendo il tab):
        # «Genera Video» + «Interrompi», spostati qui da sotto il canvas (richiesta utente).
        # I click li cabla main_window (on_generate / driver.interrupt) — qui solo i widget.
        _footer = QWidget(); _fl = QHBoxLayout(_footer)
        _fl.setContentsMargins(8, 6, 8, 8)
        self.generate_btn = QPushButton(tr("vid.generate.btn", "🎬 Genera Video"))
        self.generate_btn.setObjectName("generate")
        self.generate_btn.setToolTip(tr("vid.generate.btn.tt", "Genera il video con i parametri impostati."))
        self.stop_btn = QPushButton(tr("btn.stop"))
        self.stop_btn.setToolTip(tr("vid.stop.tt", "Interrompi la generazione in corso."))
        _fl.addWidget(self.generate_btn, 2); _fl.addWidget(self.stop_btn, 1)
        outer.addWidget(_footer)

        # --- Prompt (spostato qui dal tab Frame 1) ---
        prompt_group = QGroupBox(tr("vid.prompt.group"))
        prompt_vl = QVBoxLayout(prompt_group)
        prompt_vl.addWidget(QLabel(tr("vid.prompt.lbl")))
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText(tr("vid.prompt.ph"))
        self.prompt.setMaximumHeight(100)
        self.prompt.setToolTip(tr("vid.prompt.tt", "Descrivi il video. Digita \\\\ per la palette tag (pause, emozioni, velocità)."))
        prompt_vl.addWidget(self.prompt)
        prompt_vl.addWidget(QLabel(tr("vid.neg.lbl")))
        self.neg = QPlainTextEdit(DEFAULT_NEG)
        self.neg.setMaximumHeight(70)
        self.neg.setToolTip(tr("vid.neg.tt", "Ciò che il modello deve evitare nel video (artefatti, stilemi indesiderati)."))
        prompt_vl.addWidget(self.neg)
        root.addWidget(prompt_group)
        # Palette di tag SSML/markup riusabile (stessa di tts_text in AudioTab).
        # Lo stesso set di tag funziona qui: il modello LTX-2.3 tende a rispettare
        # marker testuali come [pause:Xms], <emotion>, <prosody> nel parlato generato.
        install_tag_palette(self.prompt)
        install_tag_palette(self.neg)

        # --- Preset selector ---
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel(tr("vid.preset_lbl")))
        self.preset_combo = QComboBox()
        self.preset_combo.setMaxVisibleItems(20)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_apply)
        self.preset_combo.setToolTip(tr("vid.preset_combo.tt", "Applica un preset di impostazioni video: risoluzione, FPS, steps, LoRA."))
        preset_row.addWidget(self.preset_combo, 1)
        self.preset_save_btn = QPushButton(tr("vid.preset.save.btn"))
        self.preset_save_btn.setToolTip(tr("vid.preset.save.tt"))
        self.preset_save_btn.clicked.connect(self._save_current_preset)
        preset_row.addWidget(self.preset_save_btn)
        self.preset_del_btn = QPushButton("🗑")
        self.preset_del_btn.setMaximumWidth(36)
        self.preset_del_btn.setToolTip(tr("vid.preset.del.tt"))
        self.preset_del_btn.clicked.connect(self._delete_selected_preset)
        preset_row.addWidget(self.preset_del_btn)
        root.addLayout(preset_row)
        self._reload_presets()

        # --- controlli video ---
        vbox = QGroupBox(tr("vid.video.group")); form = QFormLayout(vbox)
        self.nsfw = QComboBox(); self.nsfw.addItems(["SFW", "NSFW"])
        self.nsfw.currentTextChanged.connect(self._on_nsfw)
        self.nsfw.setToolTip(tr("vid.nsfw.tt", "SFW: solo modelli sicuri. NSFW: aggiunge modelli per contenuto adulto."))
        form.addRow(tr("vid.content_lbl"), self.nsfw)
        # --- Architettura video: LTX-2 vs LingBot-Video (2ª architettura) ---
        # Cambiare architettura ripopola il dropdown modello e abilita/disabilita i controlli non
        # supportati (v2v/controlnet/audio per LingBot). Vedi core.model_catalog.video_caps.
        self.video_arch = QComboBox()
        self.video_arch.addItem(tr("vid.arch.ltx", "LTX-2 (tutte le funzioni)"), "ltx")
        self.video_arch.addItem(tr("vid.arch.lingbot", "LingBot-Video (MoE, qualità SOTA)"), "lingbot")
        self.video_arch.addItem(tr("vid.arch.krea_rt", "Krea Realtime (autoregressivo, long/extend)"), "krea_rt")
        self.video_arch.addItem(tr("vid.arch.minimax_h3",
                                   "MiniMax H3 (audio nativo + riferimenti multipli)"), "minimax_h3")
        self.video_arch.setToolTip(tr("vid.arch.tt",
            "LTX-2: audio nativo, controlnet, v2v, storyboard. LingBot-Video: T2V/I2V/TI2V + refiner, "
            "prompt JSON strutturato; niente audio/controlnet/v2v. MiniMax H3: video e audio stereo "
            "generati insieme + riferimenti multipli (9 immagini, 3 video, 3 audio); niente "
            "controlnet/v2v/storyboard."))
        try:
            _ia = self.video_arch.findData(user_prefs.get("video_arch", "ltx"))
            self.video_arch.setCurrentIndex(_ia if _ia >= 0 else 0)
        except Exception:                                       # noqa: BLE001
            pass
        self.video_arch.currentIndexChanged.connect(self._on_arch_changed)
        form.addRow(tr("vid.arch_lbl", "Architettura"), self.video_arch)

        # VIDEO LUNGO: Continuo (Self-Forcing rolling, una sola ripresa senza tagli, durata illimitata a memoria
        # costante) vs A blocchi (l'architettura selezionata sopra genera a segmenti/estensione: qualità per
        # segmento più alta ma con possibili cuciture). "Continuo" instrada al motore Self-Forcing.
        self.long_mode = QComboBox()
        self.long_mode.addItem(tr("vid.long.blocks", "A blocchi (arch selezionata)"), "blocks")
        self.long_mode.addItem(tr("vid.long.continuous", "Continuo (Self-Forcing, senza tagli)"), "continuous")
        self.long_mode.setToolTip(tr("vid.long.tt",
            "Continuo: una sola ripresa senza cuciture, durata illimitata a memoria costante (motore "
            "Self-Forcing 1.3B). A blocchi: l'architettura scelta genera a segmenti (qualità per segmento più "
            "alta ma possibili stacchi)."))
        form.addRow(tr("vid.long_lbl", "Video lungo"), self.long_mode)

        # Decode a basso consumo (LTX): tile VAE più piccoli → meno VRAM di picco per film lunghi / alta
        # risoluzione (più lento ma niente OOM). Lo streaming decode/su-disco di LTX è già attivo di suo.
        from PySide6.QtWidgets import QCheckBox as _CkBox
        self.decode_lowmem = _CkBox(tr("vid.decode_lowmem", "Decode a basso consumo (film lunghi / 4K)"))
        self.decode_lowmem.setToolTip(tr("vid.decode_lowmem.tt",
            "Riduce la VRAM di picco del decode VAE (tile più piccoli). Utile per video molto lunghi o ad alta "
            "risoluzione; leggermente più lento. Vale per LTX-2."))
        form.addRow("", self.decode_lowmem)
        # AUDIO NATIVO LTX-2: LTX-2.3 genera video+audio dal prompt (parlato/suoni). Di default l'audio non
        # richiesto viene scartato (evita rumore sui prompt non-parlato). Attiva per TENERLO quando il prompt
        # descrive parlato/suoni (es. «says "hello world"») → voce + lip-sync. Solo LTX-2.
        self.native_audio = _CkBox(tr("vid.native_audio", "🔊 Audio nativo (parlato/suoni dal prompt) — LTX-2"))
        self.native_audio.setToolTip(tr("vid.native_audio.tt",
            "LTX-2.3 può generare l'audio dal prompt (voce, suoni) e sincronizzare le labbra. Attiva quando il "
            "prompt descrive parlato/suoni. Da spento il video è muto (default, evita rumore casuale)."))
        form.addRow("", self.native_audio)

        self.model = QComboBox()
        self.model.setMaxVisibleItems(20)       # fix: mostra tutti i modelli nel popup
        self.model.currentIndexChanged.connect(self._on_model_changed)
        self.model.currentIndexChanged.connect(self._update_model_star)
        self.model.activated.connect(self._on_model_activated)   # offerta download LingBot solo su click utente
        mrow = QHBoxLayout(); mrow.setContentsMargins(0, 0, 0, 0)
        mrow.addWidget(self.model, 1)
        # filtro precisione (fp16/fp8/fp4/gguf) + colori VRAM sul dropdown modelli video
        from ui.widgets.precision_combo import make_precision_selector
        self.precision = make_precision_selector(); self.precision.setMaximumWidth(110)
        self.precision.currentIndexChanged.connect(lambda *_: self._refilter_models())
        mrow.addWidget(self.precision)
        self.model_star = QToolButton(); self.model_star.setText("☆")
        self.model_star.setStyleSheet("QToolButton{border:none;font-size:16px;color:#777;}")
        self.model_star.setToolTip(tr("vid.model_star.tt"))
        self.model_star.clicked.connect(self._toggle_fav_model_current)
        mrow.addWidget(self.model_star)
        form.addRow(tr("vid.model_lbl"), self._wrap(mrow))
        # Text encoder LTX-2.5: l'unica alternativa all'ufficiale è 18+ (Gemma 4 uncensored) → riga visibile solo con
        # un transformer 2.5 scelto e voci 18+ ammesse; altrove (o tornando in SFW) resta l'ufficiale.
        self.ltx_te = QComboBox()
        self.ltx_te.addItem(tr("vid.ltx_te.official", "Ufficiale (Gemma 4)"), "")
        self.ltx_te.addItem(tr("vid.ltx_te.uncensored", "Gemma 4 Uncensored v1.1 (18+)"), "ltx25-te-uncensored")
        self.ltx_te.setToolTip(tr("vid.ltx_te.tt",
            "Encoder che legge il prompt per LTX 2.5. «Uncensored» descrive i contenuti espliciti senza "
            "ammorbidirli; vale con ogni transformer 2.5."))
        self.ltx_te_lbl = QLabel(tr("vid.ltx_te.lbl", "Text encoder"))
        form.addRow(self.ltx_te_lbl, self.ltx_te)
        self.model.currentIndexChanged.connect(self._sync_ltx_te)
        self.nsfw.currentTextChanged.connect(self._sync_ltx_te)
        self._sync_ltx_te()
        self.res_preset = QComboBox(); self.res_preset.addItems(list(RES_PRESETS.keys()))
        self.res_preset.currentTextChanged.connect(self._on_preset)
        self.res_preset.setToolTip(tr("vid.res_preset.tt", "Aspect ratio e risoluzione del video. 'Personalizzato' abilita W/H manuali."))
        form.addRow(tr("vid.res_lbl"), self.res_preset)
        wh = QHBoxLayout()
        self.width = QSpinBox(); self.width.setRange(256, 2048); self.width.setSingleStep(32); self.width.setValue(704)
        self.width.setToolTip(tr("vid.width.tt", "Larghezza del video in pixel (multiplo di 32)."))
        self.height = QSpinBox(); self.height.setRange(256, 2048); self.height.setSingleStep(32); self.height.setValue(1280)
        self.height.setToolTip(tr("vid.height.tt", "Altezza del video in pixel (multiplo di 32)."))
        wh.addWidget(QLabel(tr("vid.w_lbl"))); wh.addWidget(self.width)
        wh.addWidget(QLabel(tr("vid.h_lbl"))); wh.addWidget(self.height)
        form.addRow("", self._wrap(wh))
        # FPS e Durata sulla stessa riga (compattazione richiesta)
        self.fps = QSpinBox(); self.fps.setRange(1, 60); self.fps.setValue(24)
        self.fps.setToolTip(tr("vid.fps.tt", "Frame al secondo. 24 = cinematografico, 30 = TV, 60 = fluid."))
        # DURATA ILLIMITATA (13/07): il cap a 60s era un limite della sola GUI — il motore sa già fare
        # video di qualunque lunghezza con la catena autoregressiva a chunk (memory-bounded: genera,
        # decodifica e cuce un segmento alla volta, quindi la VRAM non cresce con la durata).
        # Oltre i 20s per passata LTX-2 non regge: sopra quella soglia si spezza in chunk in automatico.
        self.length = QDoubleSpinBox(); self.length.setRange(1, 86400)   # fino a 24h: di fatto nessun tetto
        self.length.setDecimals(1); self.length.setValue(8.0); self.length.setSuffix(" s")
        self.length.setToolTip(tr("vid.length.tt",
                                  "Durata del video in secondi. Nessun limite: oltre i 20s il video "
                                  "viene generato a segmenti concatenati (la VRAM resta costante)."))
        fpsdur = QHBoxLayout()
        fpsdur.addWidget(QLabel(tr("vid.fps_lbl"))); fpsdur.addWidget(self.fps)
        fpsdur.addSpacing(12)
        fpsdur.addWidget(QLabel(tr("vid.dur_lbl"))); fpsdur.addWidget(self.length, 1)
        form.addRow(tr("vid.fps_dur_lbl"), self._wrap(fpsdur))
        # Forza immagine: quanto la foto di partenza è rispettata nel VIDEO (nodo 5647 → LTXVAddGuide.strength).
        # Default 0.50 ("soft reference" come il workflow di riferimento): valori alti (0.8+) ancorano
        # troppo al fotogramma → movimento rigido/innaturale; 0.45-0.6 lasciano al modello libertà di
        # movimento mantenendo l'identità. Alza verso 0.8 solo se vuoi aderenza stretta all'immagine.
        # Queste righe (immagine iniziale / I2V / PuLID) stanno nel gruppo «Immagine iniziale», nel sotto-tab «Genera»:
        # sono ingressi della GENERAZIONE, non della modifica. Il resto del gruppo «Video» (architettura, modello,
        # risoluzione, VRAM…) è condiviso e sta SOPRA la barra dei sotto-tab.
        i2v_box = QGroupBox(tr("vid.i2v.group", "Immagine iniziale (I2V)")); i2f = QFormLayout(i2v_box)
        self._i2v_box = i2v_box
        isr = QHBoxLayout()
        self.img_strength = QSlider(Qt.Horizontal); self.img_strength.setRange(0, 100); self.img_strength.setValue(50)
        self.img_strength_lbl = QLabel("0.50")
        self.img_strength.valueChanged.connect(lambda v: self.img_strength_lbl.setText(f"{v/100:.2f}"))
        self.img_strength.setToolTip(tr("vid.img_strength.tt"))
        isr.addWidget(self.img_strength, 1); isr.addWidget(self.img_strength_lbl)
        i2f.addRow(tr("vid.img_strength_lbl"), self._wrap(isr))
        # Metodo di condizionamento immagine→video (solo quando si parte da un'immagine)
        self.i2v_method = QComboBox()
        self.i2v_method.addItem(tr("vid.i2v_method.addguide"), "addguide")
        self.i2v_method.addItem(tr("vid.i2v_method.inplace"), "inplace")
        self.i2v_method.setToolTip(tr("vid.i2v_method.tt"))
        i2f.addRow(tr("vid.i2v_method_lbl"), self.i2v_method)
        # One-click: foto volto (PuLID) → genera il PRIMO FRAME con l'identità → poi l'i2v lo anima.
        # PuLID è FLUX-only, il video è LTX: keyframe→i2v è il modo corretto per l'identità sul video.
        # CHECKBOX di attivazione (richiesta utente): PuLID è OPT-IN esplicito; da spento i campi/bottone sono
        # disabilitati e il volto NON viene imposto.
        self.pulid_kf_enable = QCheckBox(tr("vid.pulid_kf.enable", "Attiva PuLID (identità volto da foto)"))
        self.pulid_kf_enable.setToolTip(tr("vid.pulid_kf.enable.tt",
            "Se attivo: genera il primo frame con l'identità del volto (PuLID-FLUX) → i2v. Se spento: i2v normale, "
            "nessun volto imposto."))
        i2f.addRow("", self.pulid_kf_enable)
        prow = QHBoxLayout()
        self.pulid_kf_face = QLineEdit(); self.pulid_kf_face.setPlaceholderText(
            tr("vid.pulid_kf.ph", "foto volto (una o più, separate) → primo frame con l'identità"))
        self.pulid_kf_browse = QPushButton(tr("vid.pulid_kf.browse", "Sfoglia…"))
        self.pulid_kf_browse.clicked.connect(self._pick_pulid_kf)
        self.pulid_kf_gen = QPushButton(tr("vid.pulid_kf.gen", "🧑 Primo frame dal volto"))
        self.pulid_kf_gen.setToolTip(tr("vid.pulid_kf.gen.tt",
            "Genera il primo frame con l'identità del volto (PuLID-FLUX), poi usalo come start per l'i2v."))
        self.pulid_kf_gen.clicked.connect(
            lambda: self.request_pulid_keyframe.emit(self.pulid_kf_face.text().strip(), "")
            if self.pulid_kf_enable.isChecked() else None)
        # abilita/disabilita i campi PuLID secondo la checkbox
        self.pulid_kf_enable.toggled.connect(lambda on: (
            self.pulid_kf_face.setEnabled(on), self.pulid_kf_browse.setEnabled(on), self.pulid_kf_gen.setEnabled(on)))
        for _w in (self.pulid_kf_face, self.pulid_kf_browse, self.pulid_kf_gen):
            _w.setEnabled(False)                             # default: PuLID spento
        prow.addWidget(self.pulid_kf_face, 1); prow.addWidget(self.pulid_kf_browse); prow.addWidget(self.pulid_kf_gen)
        i2f.addRow(tr("vid.pulid_kf_lbl", "Volto → primo frame (PuLID)"), self._wrap(prow))
        # Selettore MOTORE video: ltx2_infer (default, tutte le feature) vs comfy-native (VRAM dinamica,
        # no-OOM, ma solo t2v/i2v semplici). Persistito in user_prefs; il routing è in _generate_video_native.
        self.video_engine_sel = QComboBox()
        self.video_engine_sel.addItem(tr("vid.engine.ltx2", "LTX-2 nativo (tutte le funzioni)"), "ltx2_infer")
        self.video_engine_sel.addItem(tr("vid.engine.comfy", "Comfy (VRAM dinamica, t2v/i2v)"), "comfy")
        try:
            from core import user_prefs as _up
            _i = self.video_engine_sel.findData(_up.get_engine("video", "ltx2_infer"))
            self.video_engine_sel.setCurrentIndex(_i if _i >= 0 else 0)
            self.video_engine_sel.currentIndexChanged.connect(
                lambda *_: _up.set_engine("video", self.video_engine_sel.currentData()))
        except Exception:                                       # noqa: BLE001
            pass
        self.video_engine_sel.setToolTip(tr("vid.engine.tt",
            "LTX-2 nativo: tutte le funzioni (audio/keyframe/controlnet). Comfy: VRAM dinamica, solo t2v/i2v."))
        self.engine_row_lbl = QLabel(tr("vid.engine_lbl", "Motore video"))
        form.addRow(self.engine_row_lbl, self.video_engine_sel)
        # UN SOLO MOTORE → NIENTE DROPDOWN (richiesta utente): con ComfyUI rimosso resta solo il
        # nativo LTX-2 → la riga sparisce e la pref viene normalizzata (via i residui 'comfy').
        try:
            from core.ltx2_service import comfy_available as _ca
            if not _ca():
                _ci = self.video_engine_sel.findData("comfy")
                if _ci >= 0:
                    self.video_engine_sel.removeItem(_ci)
                if self.video_engine_sel.count() <= 1:
                    self.video_engine_sel.setVisible(False)
                    self.engine_row_lbl.setVisible(False)
                    from core import user_prefs as _up2
                    _up2.set_engine("video", "ltx2_infer")
        except Exception:                                    # noqa: BLE001
            pass
        # Backend di ATTENZIONE (LTX): flash/SDPA (esatto, veloce su GPU recenti) vs Sage (fp8, GPU piccole).
        # Sostituiscono lo stesso operatore; se ne usa uno. Default flash. Vedi core/attention_backend.
        self.attn_backend = QComboBox()
        self.attn_backend.addItem(tr("vid.attn.flash", "Flash / SDPA (esatto, veloce)"), "flash")
        self.attn_backend.addItem(tr("vid.attn.sage", "SageAttention (fp8, GPU piccole)"), "sage")
        self.attn_backend.addItem(tr("vid.attn.auto", "Auto (flash su GPU recenti)"), "auto")
        self.attn_backend.setToolTip(tr("vid.attn.tt",
            "Backend di attenzione LTX. Flash/SDPA = esatto e più veloce su GPU recenti (es. RTX 5090). "
            "Sage = fp8 quantizzato (approssimato), può aiutare su GPU piccole. Ne gira uno solo."))
        try:
            from core import attention_backend as _ab
            _ia = self.attn_backend.findData(_ab.get())
            self.attn_backend.setCurrentIndex(_ia if _ia >= 0 else 0)
            self.attn_backend.currentIndexChanged.connect(
                lambda *_: _ab.set(self.attn_backend.currentData()))
        except Exception:                                       # noqa: BLE001
            pass
        self.attn_row_lbl = QLabel(tr("vid.attn_lbl", "Attenzione"))
        form.addRow(self.attn_row_lbl, self.attn_backend)
        # --- Opzioni LingBot-Video (progressive disclosure: visibili solo con arch=LingBot) ---
        self.lingbot_box = QGroupBox(tr("vid.lingbot.group", "Opzioni LingBot-Video"))
        _lbf = QFormLayout(self.lingbot_box)
        self.lingbot_refiner = QCheckBox(tr("vid.lingbot.refiner", "Refiner a 2 stadi (solo MoE, +qualità)"))
        self.lingbot_refiner.setChecked(True)
        self.lingbot_refiner.setToolTip(tr("vid.lingbot.refiner.tt",
            "Applica il 2° DiT di raffinamento (~8 step) dopo il video base. Solo modello MoE 30B."))
        _lbf.addRow(self.lingbot_refiner)
        self.lingbot_rewriter = QCheckBox(tr("vid.lingbot.rewriter", "Prompt Rewriter (prosa → JSON strutturato)"))
        self.lingbot_rewriter.setChecked(True)
        self.lingbot_rewriter.setToolTip(tr("vid.lingbot.rewriter.tt",
            "LingBot preferisce prompt JSON strutturati: riscrive il tuo prompt con il rewriter proprietario "
            "(Qwen3.6-27B + LoRA) e genera anche il negative automatico. Richiede i pesi del rewriter."))
        _lbf.addRow(self.lingbot_rewriter)
        _lb_note = QLabel(tr("vid.lingbot.note",
            "LingBot non supporta audio/lip-sync, ControlNet, V2V e storyboard a keyframe: quei pannelli "
            "sono disattivati. Per il lip-sync usa MultiTalk sul video generato."))
        _lb_note.setWordWrap(True); _lb_note.setStyleSheet("color:#74c0fc;")
        _lbf.addRow(_lb_note)
        self.lingbot_box.setVisible(False)
        root.addWidget(self.lingbot_box)
        self._build_h3_box(root)
        # #104 TeaCache: cache che salta gli step di denoising quasi-invariati (velocità). 0=off.
        # Conviene con MOLTI step (HQ/Pro); soglia più alta = più veloce ma più perdita qualità.
        self.teacache = QDoubleSpinBox()
        self.teacache.setRange(0.0, 0.15); self.teacache.setSingleStep(0.01); self.teacache.setValue(0.0)
        self.teacache.setDecimals(2)
        self.teacache.setToolTip(tr("vid.teacache.tt",
            "TeaCache: accelera saltando gli step di denoising quasi identici (0 = off). Prova 0.03 (~1.6×) o "
            "0.05 (~2.1×). Utile con molti step (HQ/Pro); troppo alto degrada la qualità. Da tarare."))
        form.addRow(tr("vid.teacache_lbl", "Cache veloce (TeaCache)"), self.teacache)
        # HR upscale/refine RIMOSSI dalla GUI (T3): erano un ramo legacy ComfyUI rotto e ridondante.
        # Nel flusso nativo la risoluzione si imposta direttamente nei preset; la super-risoluzione e il
        # dettaglio si fanno nel tab «Definizione Video» (SeedVR2/FlashVSR nativi). Gli attributi restano
        # (nascosti, default False) per compatibilità con get_values()/gui_state, senza rompere nulla.
        self.downscale_half = QCheckBox(); self.downscale_half.setChecked(False); self.downscale_half.hide()
        self.upscale_x2 = QCheckBox(); self.upscale_x2.setChecked(False); self.upscale_x2.hide()
        self.refine_hr = QCheckBox(); self.refine_hr.setChecked(False); self.refine_hr.hide()
        hr_note = QLabel(tr("vid.hr_moved.note"))
        hr_note.setWordWrap(True); hr_note.setStyleSheet("color:#74c0fc;")
        form.addRow("", hr_note)
        note = QLabel(tr("vid.res.note"))
        note.setWordWrap(True); note.setStyleSheet("color:#888;")
        form.addRow(note)
        root.addWidget(vbox)

        # --- Sotto-tab «Genera» | «Modifica» ---
        # SOPRA la barra (condiviso): prompt, preset, gruppo «Video» (architettura/checkpoint/risoluzione/VRAM), opzioni
        # dell'architettura, Sampler/Seed e LoRA. «Genera»: immagine iniziale, storyboard, riferimenti, post-processo.
        # «Modifica»: V2V (Normale / Talk / Canta / Ascolta / ReTake). Stessi widget di prima, stessa cfg emessa.
        self.mode_tabs = QTabWidget()
        _gen_w = QWidget(); gen_l = QVBoxLayout(_gen_w); gen_l.setContentsMargins(0, 6, 0, 0)
        _edit_w = QWidget(); edit_l = QVBoxLayout(_edit_w); edit_l.setContentsMargins(0, 6, 0, 0)
        gen_l.addWidget(self._i2v_box)
        self.mode_tabs.addTab(_gen_w, tr("vid.tab.generate", "Genera"))
        self.mode_tabs.addTab(_edit_w, tr("vid.tab.edit", "Modifica"))
        self.mode_tabs.tabBar().setToolTip(tr("vid.tab.tt", "Genera: crea un video nuovo. Modifica: Talk, Canta, "
                                              "Ascolta, ReTake e V2V su un video o una foto."))

        # --- V2V: editing video (Normal / Talk / ReTake) ---
        from ui.widgets.v2v_panel import V2VPanel
        v2v_box = QGroupBox(tr("vid.v2v.group"))
        v2v_l = QVBoxLayout(v2v_box)
        self.v2v_panel = V2VPanel(nsfw_getter=lambda: self.nsfw.currentText() == "NSFW")
        v2v_l.addWidget(self.v2v_panel)
        edit_l.addWidget(v2v_box)
        edit_l.addStretch(1)
        # Tab e modalità V2V si tengono d'accordo: in «Genera» si genera (modalità Normale); scegliere Talk/Canta/…
        # porta in «Modifica»; tornare in «Genera» rimette Normale. Così ciò che vedi è ciò che parte.
        self._mode_sync = False
        self.mode_tabs.currentChanged.connect(self._on_mode_tab)
        self.v2v_panel._mode_group.buttonToggled.connect(self._on_panel_mode)

        # --- Storyboard multi-scena (immagini-keyframe a tempo → video senza montaggio) ---
        from ui.widgets.storyboard_panel import StoryboardPanel
        sb_box = QGroupBox(tr("vid.sb.group"))
        sb_l = QVBoxLayout(sb_box)
        self.storyboard_panel = StoryboardPanel()
        sb_l.addWidget(self.storyboard_panel)
        gen_l.addWidget(sb_box)

        # --- Riferimenti multipli (identità/stile/movimento/voce, SENZA posizione temporale) ---
        # Complementare allo storyboard: lì le immagini sono keyframe a un istante, qui sono
        # riferimenti citati nel prompt col loro token. Limiti dall'architettura (_on_arch_changed).
        from ui.widgets.reference_panel import ReferencePanel
        ref_box = QGroupBox(tr("vid.ref.group", "🎭 Riferimenti multipli"))
        ref_l = QVBoxLayout(ref_box)
        self.reference_panel = ReferencePanel()
        self.reference_panel.insert_token.connect(self._insert_prompt_token)
        # H3 Ref2VA (riferimenti attivi): niente LoRA turbo 8 passi (esiste solo per FL2VA) e Veda disponibile
        self.reference_panel.enable.toggled.connect(self._sync_h3_refs)
        self._sync_h3_refs()
        ref_l.addWidget(self.reference_panel)
        # @Foto_N nel prompt: completamento digitando «@» + token evidenziati (unica convenzione dell'app)
        from ui.widgets import ref_prompt
        self._prompt_refs = ref_prompt.install(
            self.prompt, lambda: self.reference_panel.names() if self.reference_panel.enable.isChecked() else set())
        self.reference_panel.changed.connect(self._prompt_refs.highlighter.rehighlight)
        gen_l.addWidget(ref_box)

        # --- Restauro volti automatico (post-generazione, opzionale) ---
        frbox = QGroupBox(tr("vid.face.group")); frf = QFormLayout(frbox)
        self.fr_auto = QCheckBox(tr("vid.face.auto"))
        self.fr_auto.setToolTip(tr("vid.face.auto.tt"))
        frf.addRow(self.fr_auto)
        from core import face_restore_pipeline as _frp
        self.fr_auto_model = QComboBox(); self.fr_auto_model.addItems(_frp.FACERESTORE_MODELS)
        self.fr_auto_model.setToolTip(tr("vid.face.model.tt", "Modello AI per il restauro volti post-generazione."))
        frf.addRow(tr("vid.face.model_lbl"), self.fr_auto_model)
        fr_isr = QHBoxLayout()
        self.fr_auto_fidelity = QSlider(Qt.Horizontal); self.fr_auto_fidelity.setRange(0, 100)
        self.fr_auto_fidelity.setValue(70)
        self.fr_auto_fidelity_lbl = QLabel("0.70")
        self.fr_auto_fidelity.valueChanged.connect(
            lambda v: self.fr_auto_fidelity_lbl.setText(f"{v/100:.2f}"))
        self.fr_auto_fidelity.setToolTip(tr("vid.face.fidelity.tt"))
        fr_isr.addWidget(self.fr_auto_fidelity, 1); fr_isr.addWidget(self.fr_auto_fidelity_lbl)
        frf.addRow(tr("vid.face.fidelity_lbl"), self._wrap(fr_isr))
        # abilita i controlli solo se la spunta è attiva
        def _toggle_fr(on):
            self.fr_auto_model.setEnabled(on); self.fr_auto_fidelity.setEnabled(on)
        self.fr_auto.toggled.connect(_toggle_fr); _toggle_fr(False)
        fr_note = QLabel(tr("vid.face.note"))
        fr_note.setWordWrap(True); fr_note.setStyleSheet("color:#888;")
        frf.addRow(fr_note)

        # --- Face-swap automatico (post-generazione): impone un volto da foto su tutti i frame ---
        self.fs_auto = QCheckBox(tr("vid.faceswap.auto", "Face-swap: imponi un volto da foto"))
        self.fs_auto.setToolTip(tr("vid.faceswap.auto.tt",
            "Dopo la generazione sostituisce il volto in ogni frame con quello della foto scelta (InsightFace)."))
        frf.addRow(self.fs_auto)
        fs_row = QHBoxLayout()
        self.fs_face = QLineEdit(); self.fs_face.setReadOnly(True)
        self.fs_face.setPlaceholderText(tr("vid.faceswap.face.ph", "Foto del volto da imporre…"))
        self.fs_face_btn = QPushButton(tr("vid.faceswap.face.btn", "Scegli volto…"))
        self.fs_face_btn.clicked.connect(self._pick_faceswap)
        fs_row.addWidget(self.fs_face, 1); fs_row.addWidget(self.fs_face_btn)
        frf.addRow(tr("vid.faceswap.face_lbl", "Volto sorgente"), self._wrap(fs_row))

        def _toggle_fs(on):
            self.fs_face.setEnabled(on); self.fs_face_btn.setEnabled(on)
        self.fs_auto.toggled.connect(_toggle_fs); _toggle_fs(False)

        # --- Fluidifica (RIFE): interpola frame → meno scatti, più fps ---
        self.rife_auto = QCheckBox(tr("vid.rife.auto", "Fluidifica (RIFE): meno scatti, più fps"))
        self.rife_auto.setToolTip(tr("vid.rife.auto.tt",
            "A fine generazione interpola frame intermedi (anti-judder); raddoppia/quadruplica gli fps."))
        frf.addRow(self.rife_auto)
        self.rife_factor = QComboBox(); self.rife_factor.addItems(["x2", "x4"])
        self.rife_factor.setToolTip(tr("vid.rife.factor.tt", "x2 raddoppia gli fps, x4 li quadruplica"))
        frf.addRow(tr("vid.rife.factor_lbl", "Interpolazione"), self.rife_factor)

        def _toggle_rife(on):
            self.rife_factor.setEnabled(on)
        self.rife_auto.toggled.connect(_toggle_rife); _toggle_rife(False)
        gen_l.addWidget(frbox)
        gen_l.addStretch(1)

        # --- Sampler / Seed (spostati da Avanzate) ---
        self.sampler_seed = SamplerSeedGroup()
        root.addWidget(self.sampler_seed)

        # --- LoRA LTX-2 (integrato; NSFW pilotato dal selettore video) ---
        lbox = QGroupBox(tr("vid.lora.group")); ll = QVBoxLayout(lbox)
        self.lora = LoraTab(allow_nsfw_getter=lambda: self.nsfw.currentText() == "NSFW", embedded=True)
        ll.addWidget(self.lora)
        root.addWidget(lbox)
        root.addWidget(self.mode_tabs, 1)

        self._refilter_models()
        self._on_preset(self.res_preset.currentText())
        self._on_arch_changed()      # applica lo stato iniziale dell'architettura (LTX/LingBot)

    def _wrap(self, layout):
        w = QWidget(); w.setLayout(layout); return w

    # ── sotto-tab Genera | Modifica ──────────────────────────────────────────────
    def show_edit_tab(self) -> None:
        """Porta in primo piano «Modifica» (usato anche dal rimando di «Modifica Video AI»)."""
        self.mode_tabs.setCurrentIndex(1)

    def is_edit_tab(self) -> bool:
        return self.mode_tabs.currentIndex() == 1

    def _on_mode_tab(self, idx: int) -> None:
        if self._mode_sync:
            return
        if idx == 0 and self.v2v_panel.current_mode() != "normal":
            self._mode_sync = True
            try:
                self.v2v_panel.rb_normal.setChecked(True)          # in «Genera» si genera: modalità Normale
            finally:
                self._mode_sync = False

    def _on_panel_mode(self, _btn=None, checked: bool = True) -> None:
        if self._mode_sync or not checked:
            return
        if self.v2v_panel.current_mode() != "normal" and self.mode_tabs.currentIndex() == 0:
            self._mode_sync = True
            try:
                self.mode_tabs.setCurrentIndex(1)                  # una modalità di modifica vive in «Modifica»
            finally:
                self._mode_sync = False

    def _on_nsfw(self, *_):
        self._refilter_models()      # riselezione istantanea modelli utilizzabili
        self.lora.refresh()          # re-filtro LoRA per NSFW
        if hasattr(self, "v2v_panel"):
            self.v2v_panel.refresh_wan_loras()   # LoRA Wan di Talk/Canta

    # ── architettura video (LTX ↔ LingBot) ───────────────────────────────────
    def _arch(self) -> str:
        return (self.video_arch.currentData() if hasattr(self, "video_arch") else "ltx") or "ltx"

    def _insert_prompt_token(self, token: str) -> None:
        """Inserisce il token del riferimento nel prompt, nel punto del cursore (con lo spazio giusto)."""
        if not token or not hasattr(self, "prompt"):
            return
        cur = self.prompt.textCursor()
        before = self.prompt.toPlainText()[:cur.position()]
        cur.insertText(("" if (not before or before.endswith((" ", "\n"))) else " ") + token + " ")
        self.prompt.setTextCursor(cur)
        self.prompt.setFocus()

    def _apply_reference_limits(self, arch: str, caps: dict) -> None:
        """Limiti dei riferimenti secondo l'architettura: sono capacità del MODELLO, non preferenze.
        MiniMax H3 (Ref2VA): 9 immagini + 3 video + 3 audio, max 12 file, clip 2–15 s.
        LTX con la LoRA Multiple-Subject-Reference: 2–5 immagini, niente video/audio."""
        panel = getattr(self, "reference_panel", None)
        if panel is None:
            return
        panel.set_token_style("h3" if arch == "minimax_h3" else "ltx")
        if arch == "minimax_h3":
            panel.set_limits(images=9, videos=3, audios=3, total=12, min_items=1, clip_s=(2.0, 15.0))
        elif caps.get("multiref"):
            # LTX con la LoRA MSR: SOLO immagini, max 4 soggetti + 1 sfondo OBBLIGATORIO — i
            # riferimenti diventano una pseudo-sequenza e lo sfondo ne riempie i frame liberi.
            panel.set_limits(images=4, videos=0, audios=0, total=4, min_items=1,
                             needs_background=True)

    def _on_arch_changed(self, *_):
        arch = self._arch()
        try:
            user_prefs.set("video_arch", arch)
        except Exception:                                       # noqa: BLE001
            pass
        is_ling = arch == "lingbot"
        is_ltx = arch == "ltx"
        # DEFAULT SAMPLER PER-ARCH (23/07): LingBot con gli 8 step/cfg~1 ereditati dai default LTX
        # genera GARBAGE psichedelico (misurato: con 40 step/cfg 3.0 è pulito). I "suggested"
        # rispettano le modifiche esplicite dell'utente (stesso meccanismo dei modelli distilled).
        try:
            ss = getattr(self, "sampler_seed", None)
            if ss is not None:
                if is_ling:
                    ss.set_suggested_steps(40, note="LingBot")
                    ss.set_suggested_cfg(3.0, note="LingBot")
                elif arch in ("krea_rt", "self_forcing"):
                    ss.set_suggested_steps(4, note="Self-Forcing/Realtime (distillato)")
                    ss.set_suggested_cfg(1.0, note="Self-Forcing/Realtime")
        except Exception:                                       # noqa: BLE001
            pass
        # progressive disclosure del pannello LingBot (refiner/rewriter) — solo LingBot
        if hasattr(self, "lingbot_box"):
            self.lingbot_box.setVisible(is_ling)
        is_h3 = arch == "minimax_h3"
        if hasattr(self, "h3_box"):
            self.h3_box.setVisible(is_h3)
            self._sync_h3_te()
        # controlli comuni con i vincoli di H3 (24 fps fissi, 5–15 s, niente negative/CFG)
        self.fps.setEnabled(not is_h3)
        self.fps.setToolTip(tr("vid.h3.fixed", "Fisso con MiniMax H3") if is_h3 else
                            tr("vid.fps.tt", "Frame al secondo. 24 = cinematografico, 30 = TV, 60 = fluid."))
        if is_h3:
            if not hasattr(self, "_pre_h3_len"):
                self._pre_h3_len = self.length.value()
            self.fps.setValue(24)
            self.length.setRange(5.0, 15.0)
            self._sync_h3_duration()
            self._sync_h3_steps()
        else:
            self.length.setRange(1, 86400)
            if hasattr(self, "_pre_h3_len"):
                self.length.setValue(self._pre_h3_len)
                del self._pre_h3_len
            if hasattr(self, "sampler_seed"):
                self.sampler_seed.steps.setEnabled(True)
        for w in (getattr(self, "neg", None), getattr(getattr(self, "sampler_seed", None), "cfg", None)):
            if w is not None:
                w.setEnabled(not is_h3)
                w.setToolTip(tr("vid.h3.no_cfg", "MiniMax H3 ha la guidance distillata: niente negative "
                                "prompt né CFG.") if is_h3 else "")
        # abilita/disabilita i controlli SOLO-LTX (teacache/motore/attention/LoRA valgono solo per LTX;
        # v2v/storyboard secondo le capability dell'arch). Non li nasconde: tooltip esplicativo.
        from core.model_catalog import video_caps
        caps = video_caps(arch)
        self._apply_reference_limits(arch, caps)
        tip = tr("vid.arch.unavailable", "Non disponibile con l'architettura {a}").format(a=arch.upper())
        ltx_only = [
            (getattr(self, "v2v_panel", None), not caps["v2v"]),
            (getattr(self, "storyboard_panel", None), not caps["storyboard"]),
            (getattr(self, "reference_panel", None), not caps.get("multiref")),
            (getattr(self, "teacache", None), not is_ltx),
            (getattr(self, "video_engine_sel", None), not is_ltx),
            (getattr(self, "engine_row_lbl", None), not is_ltx),
            (getattr(self, "attn_backend", None), not (is_ltx or is_h3)),   # LingBot/Krea: sempre FlashAttention
            (getattr(self, "attn_row_lbl", None), not (is_ltx or is_h3)),
            (getattr(self, "lora", None), not (is_ltx or is_h3)),  # LTX e H3 (worker H3: LoRA al volo)
        ]
        # il pannello LoRA mostra le LoRA dell'arch attiva (registro: family «MiniMax-H3» / LTX-2 / LTX-2.5)
        _lora_arch = "MiniMax-H3" if is_h3 else ("LTX" if is_ltx else None)
        if _lora_arch and hasattr(self, "lora") and getattr(self.lora, "_arch_label", None) != _lora_arch:
            try:
                self.lora.set_arch(_lora_arch)
            except Exception:                                   # noqa: BLE001
                pass
        for w, disable in ltx_only:
            if w is None:
                continue
            w.setEnabled(not disable)
            try:
                w.setToolTip(tip if disable else "")
            except Exception:                                   # noqa: BLE001
                pass
        # ripopola il dropdown modello per l'architettura scelta
        self._refilter_models()                          # smista per architettura

    # ── MiniMax H3 ─────────────────────────────────────────────────────────────────────────────
    def _build_h3_box(self, root):
        """Opzioni SOLO di MiniMax H3 (progressive disclosure). Tutto il resto riusa i controlli comuni:
        formato = rapporto della risoluzione, durata 5–15 s, fps 24, «Audio nativo» = genera audio, passi."""
        from PySide6.QtWidgets import QLineEdit as _LE
        from ui.widgets.h3_dialogue import H3DialogueEditor
        self.h3_box = QGroupBox(tr("vid.h3.group", "Opzioni MiniMax H3"))
        f = QFormLayout(self.h3_box)
        self.h3_te = QComboBox()
        for tid, key, label in (("mmh3-te-nvfp4", "nvfp4", "Qwen3-VL-32B nvfp4-AWQ (15,7 GB, consigliato)"),
                                ("mmh3-te-fp8", "fp8", "Qwen3-VL-32B fp8 (26,5 GB, RTX 40xx)"),
                                ("mmh3-te-int8", "int8", "Qwen3-VL-32B int8-ConvRot (27 GB, qualità)"),
                                ("mmh3-te-heretic-nvfp4", "heretic", "Qwen3-VL-32B Heretic nvfp4 (senza censura)")):
            self.h3_te.addItem(tr(f"vid.h3.te.{key}", label), tid)
        self.h3_te.setToolTip(tr("vid.h3.te.tt",
            "Il text encoder legge il prompt (e le immagini di riferimento). Lavora da solo e viene liberato "
            "prima del video, quindi non si somma al transformer in VRAM."))
        f.addRow(tr("vid.h3.te_lbl", "Text encoder"), self.h3_te)
        self.h3_turbo = QComboBox()
        self.h3_turbo.addItem(tr("vid.h3.turbo.off", "Spento (qualità piena, ~49 passi)"), "")
        self.h3_turbo.addItem(tr("vid.h3.turbo.4", "Turbo 4 passi (lightx2v)"), "4")
        self.h3_turbo.addItem(tr("vid.h3.turbo.8", "Turbo 8 passi (più dettaglio)"), "8")
        self.h3_turbo.addItem(tr("vid.h3.turbo.v4", "Turbo v4 8 passi (larryvrh, anche coi riferimenti)"), "v4")
        self.h3_turbo.addItem(tr("vid.h3.turbo.dmad", "Turbo DMAD + v4 8 passi (audio migliore, sperimentale)"), "dmad")
        self.h3_turbo.setToolTip(tr("vid.h3.turbo.tt",
            "LoRA di distillazione lightx2v: da ~49 a 4 o 8 passi (10× più veloce). Con i riferimenti "
            "(Ref2VA) esiste solo la 4 passi, in anteprima."))
        self.h3_turbo.currentIndexChanged.connect(self._sync_h3_steps)
        f.addRow(tr("vid.h3.turbo_lbl", "Velocità"), self.h3_turbo)
        # Attenzione sparsa Veda (workers/veda_h3): il predittore è Ref2VA → vale solo coi riferimenti; senza, il
        # worker non la riceve (main_window). Il riepilogo «sparse/piena» finisce nel log del job.
        self.h3_veda = QCheckBox(tr("vid.h3.veda", "Attenzione sparsa Veda (più veloce)"))
        self.h3_veda.setToolTip(tr("vid.h3.veda.tt",
            "Calcola solo i blocchi d'attenzione che un predittore addestrato segna come utili: circa 1,4–2,2× "
            "più veloce sul denoise. Vale con i riferimenti (Ref2VA) e il Turbo 8 passi; i pesi del modello non "
            "cambiano. Se qualcosa non torna, il video usa da solo l'attenzione piena."))
        f.addRow("", self.h3_veda)
        # LightVAE (lightx2v): VAE video potato con la stessa interfaccia latente → nessun effetto sul denoise
        self.h3_lightvae = QComboBox()
        self.h3_lightvae.addItem(tr("vid.h3.vae.orig", "Originale (qualità piena)"), "")
        self.h3_lightvae.addItem(tr("vid.h3.vae.dec", "LightVAE: decoder Lite (decodifica 3× più veloce)"), "dec")
        self.h3_lightvae.addItem(tr("vid.h3.vae.encdec", "LightVAE: encoder + decoder Lite"), "encdec")
        self.h3_lightvae.setToolTip(tr("vid.h3.vae.tt",
            "Versione potata del VAE video di H3 (lightx2v): stessi latenti, quindi il modello non cambia. Il "
            "decoder Lite trasforma i latenti in fotogrammi circa 3 volte più in fretta con metà della memoria "
            "video; l'encoder Lite accelera anche la codifica di prima/ultima immagine e riferimenti. Può "
            "perdere un po' di dettaglio fine: confrontalo con l'originale."))
        f.addRow(tr("vid.h3.vae_lbl", "VAE video"), self.h3_lightvae)
        self.h3_canvas = QComboBox()
        self.h3_canvas.addItem(tr("vid.h3.canvas.768", "768p (nativa, massima qualità)"), "768")
        self.h3_canvas.addItem(tr("vid.h3.canvas.544", "544p (≈2,3× più veloce)"), "544")
        self.h3_canvas.setToolTip(tr("vid.h3.canvas.tt",
            "Dimensione della tela. Il formato (16:9, 9:16…) viene dalla risoluzione scelta sopra; H3 "
            "adatta i lati alla sua griglia (lato corto 768 o 544, multipli di 32)."))
        f.addRow(tr("vid.h3.canvas_lbl", "Tela"), self.h3_canvas)
        self.h3_dialogue = H3DialogueEditor()
        f.addRow(tr("vid.h3.dialogue_lbl", "Battute"), self.h3_dialogue)
        self.h3_soundscape = _LE()
        self.h3_soundscape.setPlaceholderText(tr("vid.h3.soundscape.ph", "es. pioggia leggera, traffico lontano"))
        self.h3_soundscape.setToolTip(tr("vid.h3.soundscape.tt", "Suoni d'ambiente da generare insieme al video."))
        f.addRow(tr("vid.h3.soundscape_lbl", "Suoni d'ambiente"), self.h3_soundscape)
        self.h3_music = _LE()
        self.h3_music.setPlaceholderText(tr("vid.h3.music.ph", "es. jazz lento al pianoforte"))
        self.h3_music.setToolTip(tr("vid.h3.music.tt", "Musica di sottofondo (non diegetica) da generare."))
        f.addRow(tr("vid.h3.music_lbl", "Musica"), self.h3_music)
        self.h3_dur_lbl = QLabel("")
        self.h3_dur_lbl.setStyleSheet("color:#74c0fc;")
        f.addRow("", self.h3_dur_lbl)
        note = QLabel(tr("vid.h3.note",
            "MiniMax H3 genera video e audio stereo nella stessa passata: 24 fps, 5–15 s, niente negative "
            "prompt (guidance distillata). Licenza MiniMax H3 Community: territori esclusi UE, Regno Unito, "
            "USA, Corea del Sud."))
        note.setWordWrap(True)
        note.setStyleSheet("color:#e8b339;")
        f.addRow(note)
        self.h3_box.setVisible(False)
        root.addWidget(self.h3_box)
        self.length.valueChanged.connect(self._sync_h3_duration)
        self.nsfw.currentTextChanged.connect(self._sync_h3_te)

    def _ltx_te_allowed(self) -> bool:
        """Riga encoder LTX visibile: transformer LTX 2.5 scelto + NSFW + edizione che ammette il 18+."""
        if self.nsfw.currentText() != "NSFW":
            return False
        d = self.model.currentData() if hasattr(self, "model") else None
        m = (d or {}).get("model") if isinstance(d, dict) else ""
        try:
            from core import ltx_release, nsfw_policy
            return ltx_release.detect_release(m or "") == "2.5" and nsfw_policy.adult_allowed()
        except Exception:                                        # noqa: BLE001
            return False

    def _sync_ltx_te(self, *_):
        on = self._ltx_te_allowed()
        self.ltx_te.setVisible(on)
        self.ltx_te_lbl.setVisible(on)
        if not on:
            self.ltx_te.setCurrentIndex(0)

    def _sync_h3_refs(self, *_):
        """Coi riferimenti H3 lavora in Ref2VA: la LoRA turbo 8 passi esiste solo per FL2VA (sceglierla faceva
        fallire Genera) e il predittore Veda è Ref2VA, quindi la sua casella ha senso solo qui."""
        from core import minimax_h3_spec as S
        refs = self.reference_panel.enable.isChecked()
        for key, (modes, _nfe, _shift) in S.TURBO.items():     # turbo senza versione Ref2VA (8 passi, dmad)
            if "ref2va" in modes:
                continue
            i = self.h3_turbo.findData(key)
            if i < 0:
                continue
            self.h3_turbo.view().setRowHidden(i, refs)
            if refs and self.h3_turbo.currentIndex() == i:
                self.h3_turbo.setCurrentIndex(self.h3_turbo.findData("4"))
        self.h3_veda.setEnabled(refs)
        if not refs:
            self.h3_veda.setChecked(False)

    def _sync_h3_te(self, *_):
        """Il text encoder «heretic» (senza censura) compare solo in modalità NSFW."""
        _i = self.h3_te.findData("mmh3-te-heretic-nvfp4")
        nsfw = self.nsfw.currentText() == "NSFW"
        self.h3_te.view().setRowHidden(_i, not nsfw)
        if not nsfw and self.h3_te.currentIndex() == _i:
            self.h3_te.setCurrentIndex(0)

    def _sync_h3_duration(self, *_):
        if not hasattr(self, "h3_dur_lbl") or self._arch() != "minimax_h3":
            return
        from core import minimax_h3_spec as S
        n = S.snap_frames(self.length.value())
        self.h3_dur_lbl.setText(tr("vid.h3.dur", "Durata effettiva: {s:.2f} s ({n} fotogrammi a 24 fps)")
                                .format(s=S.frames_seconds(n), n=n))

    def _sync_h3_steps(self, *_):
        if self._arch() != "minimax_h3":
            return
        from core import minimax_h3_spec as S
        t = self.h3_turbo.currentData() or ""
        ss = getattr(self, "sampler_seed", None)
        if ss is not None:
            ss.set_suggested_steps(S.TURBO[t][1] if t else S.DEFAULT_NFE, note="MiniMax H3")
            ss.steps.setEnabled(not t)

    @staticmethod
    def _model_key(combo, i):
        return (combo.itemData(i) or {}).get("key")

    def _finish_model_menu(self, prev_key, candidates: set):
        """Dopo aver ripopolato il menu modelli: riseleziona la scelta di prima; se e' SPARITA (matrice
        dell'edizione, modalita' NSFW...) resta il segnaposto «scegli un modello» (ui/model_menu.py)."""
        from ui import model_menu as mm
        mm.settle(self.model, prev_key, candidates, key_of=lambda i: self._model_key(self.model, i),
                  data={"kind": "placeholder", "placeholder": True, "key": mm.PLACEHOLDER})

    def model_is_placeholder(self) -> bool:
        from ui import model_menu as mm
        return mm.is_placeholder(self.model)

    def _refilter_h3_models(self):
        """Dropdown modello = profili di quantizzazione H3 (il checkpoint FL2VA o Ref2VA lo decide la
        modalità: con i riferimenti serve il Ref2VA). Non scaricati → evidenziati, download su selezione."""
        from core import weights_catalog as wc
        from core import minimax_h3_spec as S
        from core import weights_tooltips as _wt
        from PySide6.QtGui import QColor
        entries = {e["id"]: e for e in wc.load()}
        from ui import model_menu as _mm
        prev = _mm.current_key(self.model, lambda i: self._model_key(self.model, i))
        labels = {"nvfp4": tr("vid.h3.prof.nvfp4", "nvfp4 · 12,5 GB (RTX 50xx)"),
                  "fp8": tr("vid.h3.prof.fp8", "fp8 · 21 GB (RTX 40xx)"),
                  "int8": tr("vid.h3.prof.int8", "int8-ConvRot · 21 GB (tutte le GPU)"),
                  "int4": tr("vid.h3.prof.int4", "int4 W4A8 · 12 GB (tutte le GPU)")}
        self.model.blockSignals(True)
        self.model.clear()
        for prof, (fl, _ref, _te) in S.PROFILES.items():
            e = entries.get(fl)
            if not e or not wc.menu_allowed(e):        # riga model.<id> dell'edizione
                continue
            try:
                present = wc.is_present(e)
            except Exception:                                   # noqa: BLE001
                present = False
            self.model.addItem(("" if present else "⬇ ") + "MiniMax H3 " + labels[prof],
                               {"kind": "minimax_h3", "profile": prof, "cat_id": fl, "present": present,
                                "key": f"minimax_h3:{prof}"})
            i = self.model.count() - 1
            self.model.setItemData(i, _wt.tooltip_for(e) + "\n" + (tr("vid.present", "presente") if present
                                                                   else tr("vid.missing", "da scaricare")),
                                   Qt.ToolTipRole)
            if not present:
                self.model.setItemData(i, QColor("#e8b339"), Qt.ForegroundRole)
        # DiT personalizzati (finetune 18+): solo se l'edizione e la modalità NSFW li ammettono
        _adult = True
        try:
            from core import nsfw_policy
            _adult = nsfw_policy.adult_allowed() and self.nsfw.currentText() == "NSFW"
        except Exception:                                       # noqa: BLE001
            pass
        for cid, meta in S.CUSTOM_DITS.items():
            e = entries.get(cid)
            if not e or not wc.menu_allowed(e) or (e.get("sensitivity") == "adult" and not _adult):
                continue
            try:
                present = wc.is_present(e)
            except Exception:                                   # noqa: BLE001
                present = False
            self.model.addItem(("" if present else "⬇ ") + "🔞 MiniMax H3 · " + meta["label"],
                               {"kind": "minimax_h3", "profile": "int8", "h3_dit": cid, "cat_id": cid,
                                "present": present, "key": f"minimax_h3:{cid}"})
            i = self.model.count() - 1
            self.model.setItemData(i, _wt.tooltip_for(e) + "\n" + (tr("vid.present", "presente") if present
                                                                   else tr("vid.missing", "da scaricare")),
                                   Qt.ToolTipRole)
            if not present:
                self.model.setItemData(i, QColor("#e8b339"), Qt.ForegroundRole)
        self._finish_model_menu(prev, {f"minimax_h3:{p}" for p in S.PROFILES}
                                | {f"minimax_h3:{c}" for c in S.CUSTOM_DITS})
        self.model.blockSignals(False)

    def _refilter_lingbot_models(self):
        """Popola il dropdown con le varianti LingBot (Dense/MoE) risolte dal catalogo pesi.
        A differenza di LTX (file singoli) i modelli LingBot sono snapshot diffusers → voci logiche."""
        from core import weights_catalog as wc
        from core import model_precision as _mp
        from PySide6.QtGui import QColor
        entries = {e["id"]: e for e in wc.load()}
        _free_gb = _mp.gpu_free_gb()
        from ui import model_menu as _mm
        prev = _mm.current_key(self.model, lambda i: self._model_key(self.model, i))
        self.model.blockSignals(True)
        self.model.clear()
        # (id_catalogo, variante, etichetta)
        variants = [
            ("lingbot_dense", "dense", tr("vid.lingbot.dense", "Dense 1.3B (leggero, tutte le GPU)")),
            ("lingbot_dense_fp8", "dense", tr("vid.lingbot.dense_fp8", "Dense 1.3B fp8 (consumer)")),
            ("lingbot_moe_uint4", "moe", tr("vid.lingbot.moe_uint4", "MoE 30B uint4 (qualità, consumer)")),
            ("lingbot_moe", "moe", tr("vid.lingbot.moe", "MoE 30B bf16 (qualità max, VRAM alta)")),
        ]
        for cat_id, variant, label in variants:
            e = entries.get(cat_id)
            if not e or not wc.menu_allowed(e):        # riga model.<id> dell'edizione
                continue
            present = False
            model_dir = ""
            try:
                present = wc.is_present(e)
                model_dir = str(wc.dest_for(e))
            except Exception:                                   # noqa: BLE001
                pass
            badge = "" if present else "⬇ "                    # non scaricato → offerto al download
            gb = e.get("approx_gb", 0)
            self.model.addItem(f"{badge}{label}  · ~{gb}GB",
                               {"kind": "lingbot", "variant": variant, "cat_id": cat_id,
                                "model": model_dir if present else "", "present": present, "key": f"lingbot:{cat_id}"})
            # tooltip ESPLICATIVO unificato (cosa fa / quando preferirlo, i18n) + stato
            from core import weights_tooltips as _wt
            _tt = _wt.tooltip_for(e) + "\n" + (tr("vid.present", "presente") if present
                                               else tr("vid.missing", "da scaricare"))
            self.model.setItemData(self.model.count() - 1, _tt, Qt.ToolTipRole)
            if not present:
                self.model.setItemData(self.model.count() - 1, QColor("#e8b339"), Qt.ForegroundRole)
        self._finish_model_menu(prev, {f"lingbot:{c}" for c, _v, _l in variants})
        self.model.blockSignals(False)
        # NB: NON offrire il download qui (refilter è anche programmatico) — l'offerta parte solo su
        # selezione ESPLICITA dell'utente (model.activated → _on_model_activated), mai in automatico.

    def _on_model_activated(self, *_):
        """Solo su click utente: se ha scelto un modello del catalogo non presente, scaricalo (con deps).
        Generico per OGNI arch a snapshot (LingBot, Krea Realtime, Self-Forcing realtime…)."""
        self._maybe_offer_lingbot_download()

    def _maybe_offer_lingbot_download(self):
        """Se il modello del catalogo selezionato (o una sua dipendenza) non è presente, ne avvia il download.
        Generico per OGNI arch a snapshot. Conferma con la dimensione TOTALE (modello + base, es. Krea Realtime
        richiede anche la base Wan) perché la connessione può essere metered."""
        data = self.model.currentData() or {}
        cat_id = data.get("cat_id")
        if not cat_id or data.get("present"):                 # nulla da scaricare
            return
        from core import weights_catalog as wc
        entries = {e["id"]: e for e in wc.load()}
        e = entries.get(cat_id)
        if not e:
            return
        try:
            missing = wc.missing_ids_with_deps(e, strict=True)   # modello + dipendenze mancanti (base Wan inclusa)
        except wc.DepNotAllowedError as ex:                    # dipendenza non inclusa nell'edizione
            from PySide6.QtWidgets import QMessageBox as _QMB
            _QMB.warning(self, tr("weights.dep.blocked.title", "Dipendenza non inclusa nell'edizione"),
                         tr("weights.dep.blocked.msg",
                            "Il modello '{m}' richiede {dep}, che la tua edizione non include: "
                            "download annullato.").format(m=ex.owner, dep=" / ".join(ex.alternatives)))
            return
        if not missing:
            return
        gb = sum(int(entries.get(m, {}).get("approx_gb", 0) or 0) for m in missing)
        from PySide6.QtWidgets import QMessageBox
        r = QMessageBox.question(
            self, tr("vid.model.dl.title", "Scaricare il modello?"),
            tr("vid.model.dl.msg", "Il modello '{m}' (e le sue dipendenze) non è scaricato: {ids} · ~{gb}GB. "
               "Scaricarlo ora?").format(m=cat_id, ids=", ".join(missing), gb=gb),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if r == QMessageBox.Yes:
            self._download_lingbot(missing)

    def _download_lingbot(self, ids):
        """Scarica uno o più pesi del catalogo (modello + deps) via scripts/download_weights.py con dialog di
        progresso (non blocca la UI). `ids` = str singolo o lista di id (deps prima)."""
        import sys
        from pathlib import Path
        from PySide6.QtCore import QProcess
        from PySide6.QtWidgets import QProgressDialog
        id_list = [ids] if isinstance(ids, str) else list(ids)
        ids_arg = ",".join(id_list)
        script = str(Path(__file__).resolve().parents[2] / "scripts" / "download_weights.py")
        dlg = QProgressDialog(tr("vid.model.dl.progress", "Scaricamento {m}…").format(m=ids_arg),
                              tr("common.cancel", "Annulla"), 0, 0, self)
        dlg.setWindowTitle(tr("vid.model.dl.title", "Scaricamento modello"))
        dlg.setMinimumWidth(420)
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)

        def _fin(*_):
            dlg.close()
            self._on_arch_changed()                             # ripopola per l'arch corrente (riflette la presenza)
        proc.finished.connect(_fin)
        dlg.canceled.connect(proc.kill)
        proc.start(sys.executable, ["-u", script, "--ids", ids_arg])
        dlg.show()

    def _refilter_diffusers_arch_models(self, items: list):
        """Popola il dropdown con modelli-snapshot diffusers di un'arch (es. Krea Realtime) dal catalogo.
        items = [(cat_id, variant, label)]. Generico (come _refilter_lingbot_models). Data kind = arch."""
        from core import weights_catalog as wc
        from PySide6.QtGui import QColor
        arch = self._arch()
        entries = {e["id"]: e for e in wc.load()}
        from ui import model_menu as _mm
        prev = _mm.current_key(self.model, lambda i: self._model_key(self.model, i))
        self.model.blockSignals(True)
        self.model.clear()
        for cat_id, variant, label in items:
            e = entries.get(cat_id)
            if not e or not wc.menu_allowed(e):        # riga model.<id> dell'edizione
                continue
            present, model_dir = False, ""
            try:
                present = wc.ready_with_deps(e)                  # presente = modello + base Wan (deps) scaricati
                model_dir = str(wc.dest_for(e))
            except Exception:                                   # noqa: BLE001
                pass
            badge = "" if present else "⬇ "
            gb = e.get("approx_gb", 0)
            self.model.addItem(f"{badge}{label}  · ~{gb}GB",
                               {"kind": arch, "variant": variant, "cat_id": cat_id,
                                "model": model_dir if present else "", "present": present,
                                "key": f"{arch}:{cat_id}"})
            from core import weights_tooltips as _wt         # tooltip esplicativo unificato (i18n)
            self.model.setItemData(self.model.count() - 1, _wt.tooltip_for(e), Qt.ToolTipRole)
            if not present:
                self.model.setItemData(self.model.count() - 1, QColor("#e8b339"), Qt.ForegroundRole)
        self._finish_model_menu(prev, {f"{arch}:{c}" for c, _v, _l in items})
        self.model.blockSignals(False)

    def _resolve_model_path(self, name: str) -> str | None:
        """Risolve il nome relativo del modello video al path assoluto (per la detection header)."""
        import os
        if not name:
            return None
        if os.path.isfile(name):
            return name
        for root in (r"E:\AI\Models\diffusion_models", r"E:\AI\Models\Checkpoints",
                     r"E:\AI\Models\checkpoints"):
            p = os.path.join(root, name)
            if os.path.isfile(p):
                return p
        return None

    def _refilter_krea_models(self):
        # due varianti selezionabili: 14B "quality" + 1.3B Self-Forcing "realtime" (leggero, ~realtime 5090)
        self._refilter_diffusers_arch_models([
            ("krea_rt", "quality", tr("vid.arch.krea_rt.model", "Krea Realtime 14B (autoregressivo, qualità)")),
            ("self_forcing_dmd", "realtime", tr("vid.arch.sf13b.model", "Self-Forcing 1.3B (realtime, leggero)")),
        ])

    def _refilter_models(self):
        """Ripopola il menu modello PER L'ARCHITETTURA scelta. Se l'architettura è CAMBIATA e il modello concreto
        scelto prima appartiene a un'altra, il combo mostra «scegli un modello» (ui/model_menu): mai il primo
        modello della nuova architettura scelto al posto dell'utente."""
        from ui import model_menu as _mm
        arch = self._arch() if hasattr(self, "video_arch") else "ltx"
        prev = _mm.current_key(self.model, lambda i: self._model_key(self.model, i))
        arch_changed = getattr(self, "_menu_arch", arch) != arch
        self._menu_arch = arch
        self._refilter_models_body()
        if arch_changed and prev not in (None, _mm.PLACEHOLDER, "auto"):
            self.model.blockSignals(True)
            try:
                self._finish_model_menu(prev, None)
            finally:
                self.model.blockSignals(False)
            self._on_model_changed()

    def _refilter_models_body(self):
        """Corpo del ripopolamento (vedi `_refilter_models`). Lo chiamano anche l'inventario asincrono, il filtro
        NSFW e la precisione: prima rimetteva sempre i checkpoint LTX, anche con LingBot/Krea/H3."""
        arch = self._arch() if hasattr(self, "video_arch") else "ltx"
        if arch == "lingbot":
            return self._refilter_lingbot_models()
        if arch == "krea_rt":
            return self._refilter_krea_models()
        if arch == "minimax_h3" and hasattr(self, "h3_box"):
            return self._refilter_h3_models()
        nsfw = self.nsfw.currentText() == "NSFW"
        cur_data = self.model.currentData() or {}
        cur_key = cur_data.get("key")
        self.model.blockSignals(True)
        self.model.clear()
        self.model.addItem(tr("vid.model.auto"), {"kind": "auto", "key": "auto"})
        # precisione: filtro + colori VRAM. free_gb calcolato UNA volta (no nvidia-smi per item).
        from core import model_precision as _mp
        from PySide6.QtGui import QColor
        _free_gb = _mp.gpu_free_gb()
        _pf = self.precision.currentData() if hasattr(self, "precision") else "all"

        def add(label: str, kind: str, model: str, is_nsfw: bool, enabled: bool = True):
            if (not nsfw) and is_nsfw:
                return
            _info = _mp.classify(model, _free_gb, file_path=self._resolve_model_path(model))
            if _pf and _pf != "all" and _info["precision"] != _pf:
                return                                       # filtrato per precisione scelta
            badge = "🔞 " if is_nsfw else ""
            fav = "⭐ " if user_prefs.is_fav_model(model) else ""
            key = f"{kind}:{model}"
            self.model.addItem(f"{fav}{badge}{label}", {"kind": kind, "model": model, "key": key})
            # tooltip con steps/cfg consigliati e tipo modello
            _steps = self._mcat.recommended_steps(model)
            _cfg = self._mcat.recommended_cfg(model)
            _note = self._mcat.recommended_note(model)
            _tip_parts = [model, tr("vid.model.tip_type").format(kind=kind)]
            if _steps is not None:
                _tip_parts.append(tr("vid.model.tip_steps").format(steps=_steps))
            if _cfg is not None:
                _tip_parts.append(tr("vid.model.tip_cfg").format(cfg=_cfg))
            if _note:
                _tip_parts.append(tr("vid.model.tip_note").format(note=_note))
            self.model.setItemData(self.model.count() - 1, "\n".join(_tip_parts), Qt.ToolTipRole)
            _col = _mp.QT_COLORS.get(_info["color"])
            if _col:
                self.model.setItemData(self.model.count() - 1, QColor(_col), Qt.ForegroundRole)
            if not enabled:
                item = self.model.model().item(self.model.count() - 1)
                if item:
                    item.setEnabled(False)

        # fav-first: i preferiti compaiono in cima a ogni categoria
        def fav_first(lst):
            return sorted(lst, key=lambda m: (not user_prefs.is_fav_model(m), m.lower()))
        diffusion = fav_first([m for m in self._diffusion_models if self._is_ltx_video_model(m)])
        checkpoints = fav_first([m for m in self._checkpoint_models if self._is_ltx_video_model(m)])
        ggufs = fav_first([m for m in self._gguf_models if self._is_ltx_video_model(m)])

        # Nessuna whitelist di compatibilità: TUTTI i modelli LTX video sono esposti.
        if diffusion:
            for m in diffusion:
                add(f"{m}  · diffusion", "diffusion", m, self._is_nsfw_model(m))
        else:
            for label, model in MODEL_FILES.items():
                add(label, "diffusion", model, label in _MODEL_NSFW)

        if ggufs:
            for m in ggufs:
                add(f"{m}  · GGUF", "gguf", m, self._is_nsfw_model(m))
        else:
            add(tr("vid.model.gguf_fallback"), "gguf", GGUF_FALLBACK_FILE, False)

        for m in checkpoints:
            add(f"{m}  · checkpoint", "checkpoint", m, self._is_nsfw_model(m))

        # Se l'utente ha già scelto qualcosa di concreto, preserva quella scelta nel refilter.
        matched = False
        if cur_key and cur_key != "auto":
            for i in range(self.model.count()):
                data = self.model.itemData(i) or {}
                if data.get("key") == cur_key:
                    self.model.setCurrentIndex(i)
                    matched = True
                    break
        # NESSUN auto-select silenzioso di un modello concreto (regola utente 22/07: «è l'utente
        # che sceglie cosa fare, non la app che prende il suo posto»). Senza scelta preservata
        # resta l'item «Auto» (indice 0): lì l'automatismo è DICHIARATO nell'etichetta.
        if not matched:
            self.model.setCurrentIndex(0)
        self.model.blockSignals(False)
        self._on_model_changed()

    def _update_model_star(self, *_):
        """Aggiorna l'icona stella accanto al dropdown in base al modello selezionato."""
        if not hasattr(self, "model_star"):
            return
        data = self.model.currentData() or {}
        model = data.get("model", "")
        is_fav = bool(model) and model != "auto" and user_prefs.is_fav_model(model)
        self.model_star.setText("★" if is_fav else "☆")
        self.model_star.setStyleSheet(
            "QToolButton{border:none;font-size:16px;color:%s;}" % ("#f5c518" if is_fav else "#777"))

    def _toggle_fav_model_current(self):
        """Segna/desegna come preferito il modello attualmente selezionato, poi riordina."""
        data = self.model.currentData() or {}
        model = data.get("model", "")
        if not model or model == "auto":
            return
        user_prefs.toggle_fav_model(model)
        # ricostruisci la lista (i preferiti vanno in cima) preservando la selezione
        self._refilter_models()
        self._update_model_star()

    @staticmethod
    def _score_model_for_vram(name: str, vram_gb: float | None) -> int:
        """Punteggio: alto = preferito. Base/dev > distilled; precisione massima che entra in VRAM."""
        n = (name or "").lower()
        s = 0
        # base vs distilled: forte penalità per i distilled (modalità veloce, qualità inferiore)
        if "distill" in n:
            s -= 1000
        if "dev" in n:
            s += 50
        # precisione: alto = qualità
        if "bf16" in n or "fp16" in n:
            prec = 4
        elif "fp8" in n:
            prec = 3
        elif "fp4" in n:
            prec = 2
        elif "gguf" in n or "_q4" in n or "_q5" in n or "_q6" in n or "_q8" in n:
            prec = 1
        else:
            prec = 3   # assunzione conservativa
        s += prec * 10
        # gate VRAM: penalizza precisioni che non entrano nella VRAM disponibile (margine per attivazioni)
        if vram_gb is not None:
            # 22B: BF16/FP16 ≈ 44 GB · FP8 ≈ 22 GB · FP4 ≈ 11 GB
            if prec == 4 and vram_gb < 48:
                s -= 200
            if prec == 3 and vram_gb < 22:
                s -= 200
            if prec == 2 and vram_gb < 14:
                s -= 100
        # preferisci 22B (qualità maggiore) su 19B, se la VRAM lo permette
        if "22b" in n:
            s += 5
        elif "19b" in n:
            s += 3
        return s

    def _pick_best_model_index(self) -> int | None:
        """Indice del modello migliore (scorre la combo, salta «Auto» e disabilitati)."""
        best_i, best_s = None, None
        for i in range(self.model.count()):
            data = self.model.itemData(i) or {}
            if data.get("kind") in (None, "auto", "unsupported"):
                continue
            # rispetta gli item disabilitati
            item = self.model.model().item(i)
            if item is not None and not item.isEnabled():
                continue
            score = self._score_model_for_vram(data.get("model", ""), self._vram_gb)
            if best_s is None or score > best_s:
                best_s, best_i = score, i
        return best_i

    def _is_ltx_video_model(self, name: str) -> bool:
        """Transformer LTX per il menu «LTX»: lo decide l'HEADER/la cartella (core/arch_filter); il catalogo esterno e
        il nome solo se il file non si riconosce. Prima il catalogo vinceva sempre: un Qwen21 etichettato «LTX» e il
        file ausiliario `_ltx23_aux_components` finivano nel menu video."""
        from core import arch_filter
        return arch_filter.is_ltx_transformer(name, self._mcat.arch_of(name))

    @staticmethod
    def _detected_family(name: str) -> str:
        try:
            from core import model_detect, offline_inventory
            import os
            for sub in ("diffusion_models", "Checkpoints"):
                p = os.path.join(str(offline_inventory.MODELS_ROOT), sub,
                                 (name or "").replace("/", os.sep))
                if os.path.isfile(p):
                    return str(model_detect.get(p).get("family") or "").lower()
        except Exception:                               # noqa: BLE001
            pass
        return ""

    def _is_nsfw_model(self, name: str) -> bool:
        # FASE D: lista keyword UNIFICATA (nsfw_policy) — prima quattro euristiche diverse in
        # quattro file. Il flag del catalogo (che include l'override utente) resta prioritario.
        from core import nsfw_policy
        return self._mcat.is_nsfw(name) or nsfw_policy.looks_nsfw(name)

    def _on_model_changed(self, *_):
        """Aggiorna Steps/CFG/Sampler consigliati al cambio modello.

        NOTA: ora il template_engine inietta SEMPRE LTXVScheduler + NAG (vedi workflow di
        riferimento). Con questa pipeline la config ottimale è la stessa per tutti i modelli
        — anche il distilled rende molto meglio a 32 step che a 8: sampler `euler_ancestral_cfg_pp`,
        CFG 1.0 (la guida negativa la dà il NAG), ~32 step. Manteniamo questi valori per tutti,
        senza più scendere a 8 step sui distilled. L'utente può comunque abbassare gli step a
        mano per bozze veloci."""
        sel = self.model.currentData() or {}
        model = sel.get("model", "")
        if not model:
            return
        note = self._mcat.recommended_note(model)
        if not hasattr(self, "sampler_seed"):
            return
        # Config qualità allineata al reference, uniforme per tutti i modelli
        self.sampler_seed.set_suggested_steps(32, note or tr("vid.suggested.steps"))
        self.sampler_seed.set_suggested_cfg(1.0, note or tr("vid.suggested.cfg"))
        self.sampler_seed.set_suggested_sampler("euler_ancestral_cfg_pp")
        self.sampler_seed.set_suggested_scheduler("karras")
        # AUTO-COMPONENTI per arch video: rileva la famiglia dall'header → filtra le LoRA per arch
        # (LTX/Wan/Hunyuan) e imposta l'encoder associato. Come per le immagini, l'utente può cambiare.
        self._sync_video_arch(model)

    # mappa famiglia rilevata → (arch LoRA, encoder consigliato per la voce 'auto')
    _VID_FAMILY = {
        "ltxv": ("LTX", "Gemma/T5 (LTX-2)"),
        "wan21": ("Wan21", "umt5-xxl"), "wan22": ("Wan22", "umt5-xxl"),
        "hunyuan_video": ("Hunyuan", "llava/llama (HunyuanVideo)"),
    }

    def _sync_video_arch(self, model: str):
        """Rileva l'architettura del modello video (detector header) e adatta LoRA + encoder."""
        fp = self._resolve_model_path(model)
        if not fp:
            return
        try:
            from core import model_detect
            det = model_detect.components_for_path(fp)
        except Exception:                                   # noqa: BLE001
            return
        fam = det.get("family")
        arch, enc = self._VID_FAMILY.get(fam, (None, None))
        if arch and hasattr(self, "lora"):
            try:
                if getattr(self.lora, "_arch_label", None) != arch:
                    self.lora.set_arch(arch)                 # filtra le LoRA per l'arch del modello
            except Exception:                               # noqa: BLE001
                pass
        # encoder associato: se il selettore encoder ha la voce 'auto', annota cosa userà
        if enc and hasattr(self, "sampler_seed") and hasattr(self.sampler_seed, "set_encoder_hint"):
            try:
                self.sampler_seed.set_encoder_hint(enc)
            except Exception:                               # noqa: BLE001
                pass

    def set_model_inventory(self, diffusion_models: list[str] | None = None,
                            checkpoint_models: list[str] | None = None,
                            gguf_models: list[str] | None = None,
                            vram_gb: float | None = None):
        self._mcat.reload_if_changed()
        self._diffusion_models = list(diffusion_models or [])
        self._checkpoint_models = list(checkpoint_models or [])
        self._gguf_models = list(gguf_models or [])
        if vram_gb is not None:
            self._vram_gb = float(vram_gb)
        self._inventory_ready = True
        self._refilter_models()

    def set_lora_inventory(self, loras: list[str]):
        self.lora.set_available_loras(loras)

    def _on_preset(self, name: str):
        wh = RES_PRESETS.get(name)
        custom = wh is None
        self.width.setEnabled(custom); self.height.setEnabled(custom)
        if wh:
            self.width.setValue(wh[0]); self.height.setValue(wh[1])

    def _on_preset_apply(self, idx: int):
        """Applica il preset selezionato a Steps, CFG, Sampler, risoluzione, FPS, durata."""
        if idx <= 0:
            return
        cfg = self.preset_combo.currentData()
        if not cfg:
            return
        if cfg.get("steps"):
            self.sampler_seed.steps.setValue(cfg["steps"])
        if cfg.get("cfg") is not None:
            self.sampler_seed.cfg.setValue(cfg["cfg"])
        if cfg.get("teacache") is not None and hasattr(self, "teacache"):
            self.teacache.setValue(float(cfg["teacache"]))   # scritto NEL controllo (regola: preset→GUI)
        if cfg.get("sampler"):
            i = self.sampler_seed.sampler.findText(cfg["sampler"])
            if i >= 0:
                self.sampler_seed.sampler.setCurrentIndex(i)
        if cfg.get("scheduler") and hasattr(self.sampler_seed, "scheduler"):
            i = self.sampler_seed.scheduler.findText(cfg["scheduler"])
            if i >= 0:
                self.sampler_seed.scheduler.setCurrentIndex(i)
        if cfg.get("width") and cfg.get("height"):
            # cerca il preset di risoluzione più vicino
            matched = False
            for _rname, wh in RES_PRESETS.items():
                if wh and wh[0] == cfg["width"] and wh[1] == cfg["height"]:
                    self.res_preset.setCurrentText(_rname)
                    matched = True
                    break
            if not matched:
                self.res_preset.setCurrentText("Personalizzato")
                self.width.setValue(cfg["width"])
                self.height.setValue(cfg["height"])
        if cfg.get("fps"):
            self.fps.setValue(cfg["fps"])
        if cfg.get("length_s"):
            self.length.setValue(cfg["length_s"])
        # LoRA del preset (ripristino autoritativo: se assenti dal preset, non tocca la selezione)
        if "loras" in cfg:
            self.lora.gui_set_state({"loras": cfg.get("loras", [])})
        # resetta a "— Scegli preset —" dopo l'applicazione
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)

    # ---------- preset personali ----------
    def _reload_presets(self):
        """Popola il combo: built-in + preset personali (⭐) da user_prefs."""
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem(tr("vid.preset.choose"), None)
        customs = user_prefs.custom_presets()
        if customs:
            self.preset_combo.addItem(tr("vid.preset.mine"), None)
            for name, cfg in customs.items():
                self.preset_combo.addItem(f"⭐ {name}", {**cfg, "_custom": True, "_name": name})
                self.preset_combo.setItemData(self.preset_combo.count() - 1,
                                              cfg.get("description", tr("vid.preset.personal")), Qt.ToolTipRole)
            self.preset_combo.addItem(tr("vid.preset.system"), None)
        for pname, pcfg in VIDEO_PRESETS.items():
            self.preset_combo.addItem(pname, pcfg)
            self.preset_combo.setItemData(self.preset_combo.count() - 1,
                                          pcfg.get("description", ""), Qt.ToolTipRole)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)

    def _current_settings_as_preset(self) -> dict:
        """Cattura le impostazioni video correnti in un dict-preset."""
        return {
            "steps": self.sampler_seed.steps.value(),
            "cfg": float(self.sampler_seed.cfg.value()),
            "sampler": self.sampler_seed.sampler.currentText(),
            "scheduler": self.sampler_seed.scheduler.currentText(),
            "width": round32(self.width.value()),
            "height": round32(self.height.value()),
            "fps": self.fps.value(),
            "length_s": self.length.value(),
            # LoRA selezionate (comfy_name + strength) → il preset le ripristina
            "loras": self.lora.gui_get_state().get("loras", []),
            "description": tr("vid.preset.personal"),
        }

    def _save_current_preset(self):
        name, ok = QInputDialog.getText(self, tr("vid.preset.save.title"), tr("vid.preset.save.prompt"))
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in user_prefs.custom_presets():
            if QMessageBox.question(self, tr("vid.preset.overwrite.title"), tr("vid.preset.overwrite.msg").format(name=name)) != QMessageBox.Yes:
                return
        user_prefs.save_preset(name, self._current_settings_as_preset())
        self._reload_presets()
        # seleziona il preset appena salvato
        i = self.preset_combo.findText(f"⭐ {name}")
        if i >= 0:
            self.preset_combo.blockSignals(True); self.preset_combo.setCurrentIndex(i); self.preset_combo.blockSignals(False)

    def _delete_selected_preset(self):
        data = self.preset_combo.currentData()
        if not (isinstance(data, dict) and data.get("_custom")):
            QMessageBox.information(self, tr("vid.preset.del.title"), tr("vid.preset.del.none"))
            return
        name = data.get("_name", "")
        if QMessageBox.question(self, tr("vid.preset.del.confirm.title"), tr("vid.preset.del.confirm.msg").format(name=name)) != QMessageBox.Yes:
            return
        user_prefs.delete_preset(name)
        self._reload_presets()

    # ---------- API ----------
    def set_facerestore_models(self, models: list[str]):
        """Riempie la dropdown del restauro volti con i modelli REALMENTE installati in ComfyUI
        (evita di offrire GFPGAN non scaricati → ComfyUI fallirebbe). Conserva la selezione se valida."""
        if not models:
            return
        cur = self.fr_auto_model.currentText()
        self.fr_auto_model.blockSignals(True)
        self.fr_auto_model.clear()
        self.fr_auto_model.addItems(models)
        i = self.fr_auto_model.findText(cur)
        self.fr_auto_model.setCurrentIndex(i if i >= 0 else 0)
        self.fr_auto_model.blockSignals(False)

    def _pick_pulid_kf(self):
        """Selettore foto volto (multiplo) per il primo frame PuLID → i2v."""
        from PySide6.QtWidgets import QFileDialog
        paths, _ = QFileDialog.getOpenFileNames(self, tr("vid.pulid_kf.browse", "Sfoglia…"), "",
                                                "Immagini (*.png *.jpg *.jpeg *.webp *.bmp)")
        if paths:
            self.pulid_kf_face.setText("|".join(paths))

    def get_values(self) -> dict:
        v = {
            "prompt_positive": self.prompt.toPlainText().strip(),
            "prompt_negative": self.neg.toPlainText().strip(),
            "nsfw_mode": "nsfw" if self.nsfw.currentText() == "NSFW" else "sfw",
            # TE LTX-2.5 alternativo (id di catalogo) solo se ammesso: altrimenti l'ufficiale della release
            "ltx_te": (self.ltx_te.currentData() or "") if self._ltx_te_allowed() else "",
            "width": round32(self.width.value()),
            "height": round32(self.height.value()),
            "fps": self.fps.value(),
            "length_s": self.length.value(),
            "ref_strength": self.img_strength.value() / 100.0,
            "downscale_half": self.downscale_half.isChecked(),
            "upscale_x2": self.upscale_x2.isChecked(),
            "refine_hr": self.refine_hr.isChecked(),
            "i2v_method": self.i2v_method.currentData() or "addguide",
            "teacache_thresh": float(self.teacache.value()) if hasattr(self, "teacache") else 0.0,
            "video_arch": self._arch(),
            "decode_lowmem": bool(self.decode_lowmem.isChecked()) if hasattr(self, "decode_lowmem") else False,
            "native_audio": bool(self.native_audio.isChecked()) if hasattr(self, "native_audio") else False,
        }
        # VIDEO LUNGO "Continuo" → instrada al motore Self-Forcing (una ripresa continua, senza tagli).
        if hasattr(self, "long_mode") and self.long_mode.currentData() == "continuous":
            v["video_arch"] = "self_forcing"
            v["long_mode"] = "continuous"
        sel_data = self.model.currentData() or {"kind": "auto"}
        kind = sel_data.get("kind")
        if kind == "krea_rt":
            v["model_kind"] = "krea_rt"
            v["model"] = sel_data.get("model", "")
            # variante REALTIME (Self-Forcing 1.3B) → arch/worker dedicato; "quality" resta il 14B krea_rt
            if sel_data.get("variant") == "realtime" or sel_data.get("cat_id") == "self_forcing_dmd":
                v["video_arch"] = "self_forcing"
                v["krea_rt_variant"] = "realtime"
        elif kind == "minimax_h3":
            v["model_kind"] = "minimax_h3"
            v["model"] = ""
            v["h3_profile"] = sel_data.get("profile", "nvfp4")
            v["h3_dit"] = sel_data.get("h3_dit", "")              # DiT personalizzato (finetune), se scelto
            v["h3_te"] = self.h3_te.currentData() or "mmh3-te-nvfp4"
            v["h3_turbo"] = self.h3_turbo.currentData() or ""
            v["h3_veda"] = self.h3_veda.isChecked()
            v["h3_lightvae"] = self.h3_lightvae.currentData() or ""
            v["h3_canvas"] = self.h3_canvas.currentData() or "768"
            v["h3_dialogue"] = self.h3_dialogue.value()
            v["h3_soundscape"] = self.h3_soundscape.text().strip()
            v["h3_music"] = self.h3_music.text().strip()
        elif kind == "lingbot":
            # LingBot: model = cartella diffusers (vuota → il worker risolve dal catalogo), variante + refiner
            v["model_kind"] = "lingbot"
            v["model"] = sel_data.get("model", "")
            v["lingbot_variant"] = sel_data.get("variant", "dense")
            v["run_refiner"] = bool(getattr(self, "lingbot_refiner", None) and self.lingbot_refiner.isChecked()
                                    and sel_data.get("variant") == "moe")
            v["lingbot_rewriter"] = bool(getattr(self, "lingbot_rewriter", None) and self.lingbot_rewriter.isChecked())
        elif kind == "gguf":
            v["model_kind"] = "gguf"
            v["model_gguf"] = sel_data.get("model")
            v["model"] = sel_data.get("model", "")
        elif kind == "diffusion":
            model = sel_data.get("model", "")
            v["model_standard"] = model
            v["model"] = model
            v["model_kind"] = "eros" if "eros" in model.lower() else "standard"
        else:
            v["model_kind"] = "auto"
            v["model"] = "auto"
        return v

    def get_face_restore_auto(self) -> dict | None:
        """Config del restauro volti automatico post-generazione, o None se disattivo."""
        if not self.fr_auto.isChecked():
            return None
        return {
            "model": self.fr_auto_model.currentText(),
            "fidelity": self.fr_auto_fidelity.value() / 100.0,
            "detection": "retinaface_resnet50",
        }

    def get_face_swap_auto(self) -> dict | None:
        """Config del face-swap automatico post-generazione (volto da foto), o None se disattivo."""
        if not self.fs_auto.isChecked():
            return None
        face = self.fs_face.text().strip()
        return {"face": face} if face else None

    def _pick_faceswap(self):
        from PySide6.QtWidgets import QFileDialog
        fn, _ = QFileDialog.getOpenFileName(
            self, tr("vid.faceswap.face.btn", "Scegli volto…"), "",
            tr("common.file_image_filter", "Immagini (*.png *.jpg *.jpeg *.webp *.bmp)"))
        if fn:
            self.fs_face.setText(fn)

    def get_rife_auto(self) -> dict | None:
        """Config dell'interpolazione RIFE post-generazione, o None se disattiva."""
        if not self.rife_auto.isChecked():
            return None
        return {"factor": int(self.rife_factor.currentText().lstrip("x"))}

    def get_sampler_values(self) -> dict:
        return self.sampler_seed.get_values()

    def wants_fixed_seed(self) -> bool:
        return self.sampler_seed.wants_fixed_seed()

    def selected_loras(self) -> list[dict]:
        return self.lora.selected_loras()

    def trigger_warnings(self, prompt: str) -> list[str]:
        return self.lora.trigger_warnings(prompt)
