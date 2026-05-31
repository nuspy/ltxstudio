"""Sampler/Seed (Steps, CFG, Sampler, Scheduler, Seed) — gruppo riusabile nel tab Video.
Il tab Avanzate funge da hub per le nuove funzionalità (Navigator, Workflow, Analizzatore, Restauro, Modelli).
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QGroupBox, QLabel, QPushButton, QHBoxLayout, QFrame, QTabWidget,
)

# Sampler VERIFICATI su LTX-2.3 (test reali 30/05): la famiglia RES (res_2s/res_multistep/…),
# pur ottima in teoria per il flow-matching, in PRATICA ROMPE LTX-2.3 → output a rumore
# verde/cyan. Confermato A/B: eros+res_multistep_cfg_pp = garbage; eros+euler_ancestral_cfg_pp =
# perfetto. Quindi qui solo i sampler euler/dpmpp che funzionano davvero. `lcm` escluso (solo LCM).
SAMPLERS = [
    "euler_ancestral_cfg_pp",    # DEFAULT — reference AuthorStorm, qualità migliore su LTX
    "euler_cfg_pp",              # deterministico pulito
    "euler",                     # baseline / bozze veloci
    "dpmpp_2m",                  # unica alternativa di altra famiglia (regge)
]   # rimossi: euler_ancestral (ridondante vs cfg_pp), dpmpp_2m_sde (SDE→flicker nel video)

# "LTXVScheduler (shift)" = default video: usa l'LTXVScheduler con gli SHIFT (sotto). Scegliendo
# uno scheduler CLASSICO, il template_engine instrada il sampler su un BasicScheduler con quello
# scheduler → la scelta diventa EFFETTIVA anche sul video. Ordinati per qualità su flow/LTX.
# Il primo valore DEVE combaciare con TemplateEngine._LTXV_SCHED_OPTION.
SCHEDULERS = [
    "LTXVScheduler (shift)",   # DEFAULT — il vero scheduler del video LTX
    "linear_quadratic",        # schedule nativo LTX
    "bong_tangent",            # flow-aware, miglior alternativa
    "karras",                  # dettaglio (diffusione/immagini)
    "beta",                    # distilled
]   # rimossi: kl_optimal (≈bong_tangent), beta57/exponential/sgm_uniform/simple/normal (generici)

# Encoder testo del VIDEO LTX (nodo 5079), ordinati per QUALITÀ/aderenza al prompt.
# Stessa dimensione (12B) → nessuna differenza di velocità: si sceglie solo per resa.
VIDEO_ENCODERS = [
    ("Auto (da SFW/NSFW)", None),   # default: il preset sceglie (SFW→full, NSFW→abliterated)
    ("Gemma 3 12B — full (qualità max)", "gemma_3_12B_it.safetensors"),
    ("Gemma 3 12B — abliterated (NSFW, decensurato)", "gemma_3_12B_it_abliterated.safetensors"),
]


class SamplerSeedGroup(QGroupBox):
    """Steps / CFG / Sampler / Scheduler / Seed — gruppo riusabile (embeddato nel tab Video).

    Default qualità (modello base, non distilled): 30 step, CFG 3.5, dpmpp_2m_sde + karras.
    Quando si seleziona un modello distilled, `set_suggested_steps/cfg` (chiamato da VideoTab
    via `model_catalog.recommended_*`) abbassa i valori per quel modello (es. 8 / 0.9).
    """

    # Default allineati al workflow di riferimento ad alta qualità:
    # LTXVScheduler (iniettato dal template_engine) + 32 step + sampler cfg_pp + CFG 1.0
    # (la guida negativa è fornita dal NAG, quindi il CFGGuider resta a 1.0).
    _DEFAULT_STEPS = 32
    _DEFAULT_CFG = 1.0
    _DEFAULT_SAMPLER = "euler_ancestral_cfg_pp"
    _DEFAULT_SCHEDULER = "LTXVScheduler (shift)"

    def __init__(self):
        super().__init__("Sampler / Seed")
        self._steps_user_modified = False
        self._cfg_user_modified = False
        self._sampler_user_modified = False
        self._scheduler_user_modified = False
        f = QFormLayout(self)

        self.steps = QSpinBox()
        self.steps.setRange(1, 60)
        self.steps.setValue(self._DEFAULT_STEPS)
        self.steps.valueChanged.connect(lambda _: setattr(self, "_steps_user_modified", True))

        self.cfg = QDoubleSpinBox()
        self.cfg.setRange(0.0, 15.0)
        self.cfg.setSingleStep(0.1)
        self.cfg.setValue(self._DEFAULT_CFG)
        self.cfg.valueChanged.connect(lambda _: setattr(self, "_cfg_user_modified", True))
        # Steps e CFG sulla stessa riga (compattazione)
        steps_cfg = QHBoxLayout()
        steps_cfg.addWidget(QLabel("Steps")); steps_cfg.addWidget(self.steps)
        steps_cfg.addSpacing(12)
        steps_cfg.addWidget(QLabel("CFG")); steps_cfg.addWidget(self.cfg)
        steps_cfg.addStretch(1)
        f.addRow("Steps / CFG:", steps_cfg)

        self.sampler = QComboBox()
        self.sampler.addItems(SAMPLERS)
        if self._DEFAULT_SAMPLER in SAMPLERS:
            self.sampler.setCurrentText(self._DEFAULT_SAMPLER)
        self.sampler.currentIndexChanged.connect(
            lambda _: setattr(self, "_sampler_user_modified", True))
        self.sampler.setToolTip(
            "Sampler VERIFICATI compatibili con LTX-2.3:\n"
            "• euler_ancestral_cfg_pp — DEFAULT, qualità migliore (reference AuthorStorm)\n"
            "• euler_cfg_pp / euler — deterministici, robusti\n"
            "• dpmpp_2m / _sde — alternative valide\n"
            "⚠ I sampler RES (res_2s, res_multistep…) sono stati RIMOSSI: in teoria ottimi per il "
            "flow, ma in pratica producono RUMORE verde su LTX-2.3 (verificato A/B)."
        )
        self.scheduler = QComboBox()
        self.scheduler.addItems(SCHEDULERS)
        if self._DEFAULT_SCHEDULER in SCHEDULERS:
            self.scheduler.setCurrentText(self._DEFAULT_SCHEDULER)
        self.scheduler.currentIndexChanged.connect(
            lambda _: setattr(self, "_scheduler_user_modified", True))
        self.scheduler.setToolTip(
            "Ora EFFETTIVO anche sul video:\n"
            "• LTXVScheduler (shift) — default: usa l'LTXVScheduler con gli SHIFT qui sotto "
            "(consigliato, flow-aware, shape-aware)\n"
            "• linear_quadratic — schedule nativo LTX\n"
            "• bong_tangent / kl_optimal / beta57 — alta qualità (flow-aware)\n"
            "• karras — dettaglio (diffusione) · beta — distilled · exponential — morbido\n"
            "Scegliendo uno scheduler classico, gli Shift LTX vengono ignorati."
        )
        # Sampler e Scheduler sulla stessa riga (compattazione)
        samp_sched = QHBoxLayout()
        samp_sched.addWidget(self.sampler, 1)
        samp_sched.addWidget(self.scheduler, 1)
        f.addRow("Sampler / Sched.:", samp_sched)

        # SHIFT di LTXVScheduler = leva qualità REALE del video (sposta il campionamento tra
        # struttura/movimento e dettaglio). Costo nullo, alto impatto.
        shift_row = QHBoxLayout()
        self.base_shift = QDoubleSpinBox()
        self.base_shift.setRange(0.10, 6.00); self.base_shift.setSingleStep(0.05)
        self.base_shift.setValue(0.95)
        self.base_shift.setToolTip(
            "Shift BASE di LTXVScheduler (default 0.95). Più ALTO = più step su struttura e "
            "MOVIMENTO (coerenza temporale, meno flicker); più BASSO = più dettaglio fine ma "
            "rischio di incoerenza tra frame. È la leva qualità #1 del video LTX."
        )
        self.max_shift = QDoubleSpinBox()
        self.max_shift.setRange(0.50, 12.00); self.max_shift.setSingleStep(0.05)
        self.max_shift.setValue(2.05)
        self.max_shift.setToolTip(
            "Shift MAX di LTXVScheduler (default 2.05). Limite superiore dello shift dipendente "
            "dalla risoluzione/lunghezza. Alzalo per clip più lunghe/grandi se il movimento è incerto."
        )
        shift_row.addWidget(QLabel("base")); shift_row.addWidget(self.base_shift)
        shift_row.addWidget(QLabel("max")); shift_row.addWidget(self.max_shift)
        f.addRow("Shift (LTX):", shift_row)

        self.encoder = QComboBox()
        for label, fn in VIDEO_ENCODERS:
            self.encoder.addItem(label, fn)
        self.encoder.setToolTip(
            "Encoder testo del VIDEO LTX — incide soprattutto sull'ADERENZA al prompt "
            "(composizione, soggetto, azione), meno sulla texture:\n"
            "• Gemma 3 12B full — massima fedeltà semantica, ma 'ammorbidisce' i contenuti espliciti\n"
            "• Gemma 3 12B abliterated — versione decensurata: molto migliore su prompt NSFW spinti\n"
            "Stesso 12B → nessuna perdita di velocità tra i due. '⛔ assente' = file non ancora presente."
        )
        f.addRow("Text encoder:", self.encoder)

        self.prompt_enhance = QCheckBox("🪄 Migliora prompt (LTX enhancer)")
        self.prompt_enhance.setChecked(False)
        self.prompt_enhance.setToolTip(
            "Se attivo, un LLM (Gemma) ESPANDE il tuo prompt in una descrizione cinematografica "
            "dettagliata (stile, visivi, azioni, AUDIO/dialoghi) prima di darlo al modello video. "
            "Utile per prompt brevi; con prompt già dettagliati lascialo OFF. In NSFW inietta "
            "automaticamente la direttiva di sblocco così non rifiuta di descrivere i soggetti.")
        f.addRow(self.prompt_enhance)

        self.fixed_seed = QCheckBox("Seed fisso")
        f.addRow(self.fixed_seed)

        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setValue(777)
        f.addRow("Seed:", self.seed)

    # ---------- suggest API (chiamata da VideoTab quando cambia il modello) ----------
    def set_suggested_steps(self, steps: int, note: str = ""):
        tip = f"Consigliato per questo modello: {steps}"
        if note:
            tip += f" ({note})"
        self.steps.setToolTip(tip)
        if not self._steps_user_modified or self.steps.value() == self._DEFAULT_STEPS:
            self._steps_user_modified = False
            self.steps.setValue(steps)
            self._steps_user_modified = False

    def set_suggested_sampler(self, name: str):
        """Imposta il sampler consigliato per il modello, rispettando l'eventuale scelta utente."""
        if not self._sampler_user_modified or self.sampler.currentText() == self._DEFAULT_SAMPLER:
            self._sampler_user_modified = False
            idx = self.sampler.findText(name)
            if idx >= 0:
                self.sampler.setCurrentIndex(idx)
            self._sampler_user_modified = False

    def set_suggested_scheduler(self, name: str):
        """Imposta lo scheduler consigliato per il modello, rispettando l'eventuale scelta utente."""
        if not self._scheduler_user_modified or self.scheduler.currentText() == self._DEFAULT_SCHEDULER:
            self._scheduler_user_modified = False
            idx = self.scheduler.findText(name)
            if idx >= 0:
                self.scheduler.setCurrentIndex(idx)
            self._scheduler_user_modified = False

    def set_suggested_cfg(self, cfg: float, note: str = ""):
        tip = f"Consigliato per questo modello: {cfg}"
        if note:
            tip += f" ({note})"
        self.cfg.setToolTip(tip)
        if not self._cfg_user_modified or abs(self.cfg.value() - self._DEFAULT_CFG) < 0.01:
            self._cfg_user_modified = False
            self.cfg.setValue(cfg)
            self._cfg_user_modified = False

    # ---------- read ----------
    def get_values(self) -> dict:
        v = {
            "steps": self.steps.value(),
            "cfg": self.cfg.value(),
            "sampler": self.sampler.currentText(),
            "scheduler": self.scheduler.currentText(),
            "ltxv_base_shift": self.base_shift.value(),
            "ltxv_max_shift": self.max_shift.value(),
        }
        enc = self.encoder.currentData()
        if enc:   # None = "Auto" → NON forzare: il preset NSFW/SFW sceglie l'encoder
            v["text_encoder"] = enc
        v["prompt_enhance"] = self.prompt_enhance.isChecked()
        if self.fixed_seed.isChecked():
            v["seed"] = self.seed.value()
        return v

    def set_encoder_availability(self, available: list[str]):
        """Marca '⛔ assente' gli encoder non ancora presenti in ComfyUI (es. abliterated in download)."""
        avail = set(available or [])
        for i in range(self.encoder.count()):
            base = self.encoder.itemText(i).split("  ⛔")[0]
            fn = self.encoder.itemData(i)
            self.encoder.setItemText(i, base + ("  ⛔ assente" if avail and fn not in avail else ""))

    def wants_fixed_seed(self) -> bool:
        return self.fixed_seed.isChecked()


# ══════════════════════════════════════════════════════════════════════════════
# Tab Avanzate — hub delle funzioni avanzate / nuove feature
# ══════════════════════════════════════════════════════════════════════════════

class AdvancedTab(QWidget):
    """Hub visivo per le funzionalità avanzate: Workflow, Analizzatore, Navigator, Restauro, Modelli."""

    # Segnali emessi dai pulsanti — collegati a MainWindow dopo la costruzione
    open_workflow   = Signal()   # apre WorkflowDialog
    open_analyzer   = Signal()   # apre SettingsAnalyzerDialog
    open_navigator  = Signal()   # apre NavigatorDialog

    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        self.sub = QTabWidget()
        outer.addWidget(self.sub)
        self.sub.addTab(self._build_tools_tab(), "🛠 Strumenti")
        # Host per Import Models (riempito da attach_models dopo la costruzione)
        self._models_host = QWidget(); QVBoxLayout(self._models_host).setContentsMargins(0, 0, 0, 0)
        self.sub.addTab(self._models_host, "📦 Import Models")

    def _build_tools_tab(self) -> QWidget:
        host = QWidget(); root = QVBoxLayout(host); root.setSpacing(10)
        intro = QLabel("<b>Funzioni avanzate</b> — strumenti e impostazioni di LTX Studio.")
        intro.setWordWrap(True); intro.setStyleSheet("color: #aaa; padding: 4px;")
        root.addWidget(intro)

        # ── Generazione ──
        gen_grp = QGroupBox("🎬 Generazione"); gv = QVBoxLayout(gen_grp)
        gv.addWidget(self._big_btn(
            "🎬  Workflow Predefiniti…",
            "35 workflow ottimizzati: Talking Head, Landscape Cinematografico, Product Showcase, "
            "Action Scene, Cyberpunk, Fashion, e molti altri.", self.open_workflow))
        gv.addWidget(self._big_btn(
            "🔍  Analizza Impostazioni…",
            "Controlla steps, CFG, risoluzione, LoRA strength e conflitti NSFW/architettura. "
            "Suggerisce auto-fix dove possibile.", self.open_analyzer))
        root.addWidget(gen_grp)

        # ── Galleria ──
        nav_grp = QGroupBox("🗂 Galleria"); nv = QVBoxLayout(nav_grp)
        nv.addWidget(self._big_btn(
            "🗂  Navigator Gallery…  (Ctrl+G)",
            "Sfoglia le generazioni precedenti con anteprima, metadati completi e riutilizzo "
            "parametri.", self.open_navigator))
        root.addWidget(nav_grp)

        # ── Video lunghi (auto-extend) ──
        root.addWidget(self._build_longvideo_group())

        # ── Nota impostazioni sampler ──
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setStyleSheet("color: #333;")
        root.addWidget(sep)
        note = QLabel(
            "⚙️  <b>Steps / CFG / Sampler / Scheduler / Seed</b> → tab <b>Video</b>.<br>"
            "🔧 <b>Migliora qualità</b> (upscale foto/video, restauro, colorazione) → tab dedicato.<br>"
            "📦 <b>Import Models</b> + patch di compatibilità ComfyUI → sotto-tab qui accanto.")
        note.setWordWrap(True); note.setStyleSheet("color: #777; font-size: 11px; padding: 4px;")
        root.addWidget(note)
        root.addStretch(1)
        return host

    def _build_longvideo_group(self) -> QGroupBox:
        """Impostazioni per i video lunghi: auto-extend oltre la soglia + parametri continuità."""
        grp = QGroupBox("🎞 Video lunghi (lunghezza illimitata)")
        f = QFormLayout(grp)
        self.lv_auto = QCheckBox("Genera automaticamente i video più lunghi della soglia (a spezzoni)")
        self.lv_auto.setChecked(True)
        self.lv_auto.setToolTip("Se attivo, quando la durata richiesta nel tab Video supera la soglia "
                                "qui sotto, il video viene prodotto in una passata base + estensioni "
                                "automatiche a spezzoni (lunghezza illimitata, VRAM costante).")
        f.addRow(self.lv_auto)
        self.lv_threshold = QDoubleSpinBox(); self.lv_threshold.setRange(4, 30); self.lv_threshold.setValue(20.0)
        self.lv_threshold.setSuffix(" s")
        self.lv_threshold.setToolTip("Durata massima della singola passata LTX. Oltre questa, scatta "
                                     "l'estensione automatica. LTX-2.3 arriva a ~20s nativi in passata "
                                     "singola → tienila a 20.")
        f.addRow("Soglia passata singola:", self.lv_threshold)
        self.lv_seg = QDoubleSpinBox(); self.lv_seg.setRange(4, 15); self.lv_seg.setValue(10.0)
        self.lv_seg.setSuffix(" s")
        self.lv_seg.setToolTip("Lunghezza di ogni spezzone I2V concatenato. Più lungo = meno giunzioni "
                               "= meno transizioni. Consigliato 10s.")
        f.addRow("Spezzone estensione:", self.lv_seg)
        self.lv_overlap = QSpinBox(); self.lv_overlap.setRange(0, 48); self.lv_overlap.setValue(24)
        self.lv_overlap.setToolTip("Fotogrammi DROPPATI all'inizio di ogni spezzone (nascondono la "
                                   "transizione I2V dal frame statico). 24 = 1s @24fps. Più alti = "
                                   "transizioni più pulite ma più contenuto scartato.")
        f.addRow("Overlap/drop (frame):", self.lv_overlap)
        self.lv_context = QSpinBox(); self.lv_context.setRange(8, 64); self.lv_context.setValue(16)
        self.lv_context.setToolTip("Fotogrammi della coda usati come contesto per lo spezzone successivo.")
        f.addRow("Contesto (frame):", self.lv_context)
        self.lv_ctx_strength = QDoubleSpinBox(); self.lv_ctx_strength.setRange(0.0, 1.0)
        self.lv_ctx_strength.setSingleStep(0.05); self.lv_ctx_strength.setValue(0.5)
        self.lv_ctx_strength.setToolTip("Quanto il contesto vincola lo spezzone: 1.0=molto ancorato "
                                        "(meno deriva, meno movimento), 0.0=libero (più movimento, più deriva).")
        f.addRow("Forza contesto:", self.lv_ctx_strength)
        return grp

    # ---- API ----
    def attach_models(self, widget: QWidget):
        """Inserisce il tab Import Models come sotto-tab di Avanzate."""
        self._models_host.layout().addWidget(widget)

    def long_video_config(self) -> dict:
        return {
            "auto": self.lv_auto.isChecked(),
            "threshold": float(self.lv_threshold.value()),
            "seg_seconds": float(self.lv_seg.value()),
            "overlap": int(self.lv_overlap.value()),
            "context": int(self.lv_context.value()),
            "context_strength": float(self.lv_ctx_strength.value()),
        }

    @staticmethod
    def _big_btn(label: str, tooltip: str, signal: Signal) -> QWidget:
        """Crea un pulsante grande con descrizione sotto."""
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)

        btn = QPushButton(label)
        btn.setMinimumHeight(44)
        btn.setStyleSheet(
            "QPushButton { font-size: 13px; text-align: left; padding: 6px 12px; "
            "border-radius: 4px; background: #2a2a3a; border: 1px solid #444; }"
            "QPushButton:hover { background: #3a3a5a; border-color: #6677cc; }"
            "QPushButton:pressed { background: #1a1a2a; }"
        )
        btn.setToolTip(tooltip)
        btn.clicked.connect(signal.emit)
        h.addWidget(btn)

        return container
