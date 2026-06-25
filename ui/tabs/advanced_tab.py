"""Sampler/Seed (Steps, CFG, Sampler, Scheduler, Seed) — gruppo riusabile nel tab Video.
Il tab Avanzate funge da hub per le nuove funzionalità (Navigator, Workflow, Analizzatore, Restauro, Modelli).
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QGroupBox, QLabel, QPushButton, QHBoxLayout, QFrame, QTabWidget,
)
from ui.i18n_manager import tr

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
    ("adv.encoder.auto", None),   # default: il preset sceglie (SFW→full, NSFW→abliterated)
    ("adv.encoder.gemma_full", "gemma_3_12B_it.safetensors"),
    ("adv.encoder.gemma_abliterated", "gemma_3_12B_it_abliterated.safetensors"),
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
    # Preset qualità Fast/Pro/HQ → (steps, cfg). Il primo è "Personalizzato" (no-op, steps/cfg None).
    _QUALITY_PRESETS = [
        ("adv.quality.custom", None, None),
        ("adv.quality.fast", 8, 1.0),
        ("adv.quality.pro", 30, 3.0),
        ("adv.quality.hq", 50, 4.0),
    ]

    def __init__(self):
        super().__init__(tr("adv.sampler.group"))
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
        steps_cfg.addWidget(QLabel(tr("adv.steps_lbl"))); steps_cfg.addWidget(self.steps)
        steps_cfg.addSpacing(12)
        steps_cfg.addWidget(QLabel(tr("adv.cfg_lbl"))); steps_cfg.addWidget(self.cfg)
        steps_cfg.addStretch(1)
        f.addRow(tr("adv.steps_cfg_lbl"), steps_cfg)

        # Preset qualità Fast/Pro/HQ (modalità ufficiali LTX-2.3): scorciatoia che imposta steps+cfg.
        # "Personalizzato" (default) = non tocca nulla; scegliere un preset compila i campi sopra,
        # che restano modificabili a mano. NB: non cambia il MODELLO (la vecchia quality-mode
        # Distilled/Base era diversa ed è stata rimossa).
        self.quality_preset = QComboBox()
        for key, _s, _c in self._QUALITY_PRESETS:
            self.quality_preset.addItem(tr(key), (_s, _c))
        self.quality_preset.setToolTip(tr("adv.quality.tt"))
        self.quality_preset.currentIndexChanged.connect(self._apply_quality_preset)
        f.addRow(tr("adv.quality_lbl"), self.quality_preset)

        self.sampler = QComboBox()
        self.sampler.addItems(SAMPLERS)
        if self._DEFAULT_SAMPLER in SAMPLERS:
            self.sampler.setCurrentText(self._DEFAULT_SAMPLER)
        self.sampler.currentIndexChanged.connect(
            lambda _: setattr(self, "_sampler_user_modified", True))
        self.sampler.setToolTip(tr("adv.sampler.tt"))
        self.scheduler = QComboBox()
        self.scheduler.addItems(SCHEDULERS)
        if self._DEFAULT_SCHEDULER in SCHEDULERS:
            self.scheduler.setCurrentText(self._DEFAULT_SCHEDULER)
        self.scheduler.currentIndexChanged.connect(
            lambda _: setattr(self, "_scheduler_user_modified", True))
        self.scheduler.setToolTip(tr("adv.scheduler.tt"))
        # Sampler e Scheduler sulla stessa riga (compattazione)
        samp_sched = QHBoxLayout()
        samp_sched.addWidget(self.sampler, 1)
        samp_sched.addWidget(self.scheduler, 1)
        f.addRow(tr("adv.sampler_sched_lbl"), samp_sched)

        # SHIFT di LTXVScheduler = leva qualità REALE del video (sposta il campionamento tra
        # struttura/movimento e dettaglio). Costo nullo, alto impatto.
        shift_row = QHBoxLayout()
        self.base_shift = QDoubleSpinBox()
        self.base_shift.setRange(0.10, 6.00); self.base_shift.setSingleStep(0.05)
        self.base_shift.setValue(0.95)
        self.base_shift.setToolTip(tr("adv.shift_base.tt"))
        self.max_shift = QDoubleSpinBox()
        self.max_shift.setRange(0.50, 12.00); self.max_shift.setSingleStep(0.05)
        self.max_shift.setValue(2.05)
        self.max_shift.setToolTip(tr("adv.shift_max.tt"))
        shift_row.addWidget(QLabel(tr("adv.shift_base_lbl"))); shift_row.addWidget(self.base_shift)
        shift_row.addWidget(QLabel(tr("adv.shift_max_lbl"))); shift_row.addWidget(self.max_shift)
        f.addRow(tr("adv.shift_lbl"), shift_row)

        self.encoder = QComboBox()
        for label_key, fn in VIDEO_ENCODERS:
            self.encoder.addItem(tr(label_key), fn)
        self.encoder.setToolTip(tr("adv.encoder.tt"))
        f.addRow(tr("adv.encoder_lbl"), self.encoder)

        self.prompt_enhance = QCheckBox(tr("adv.prompt_enhance"))
        self.prompt_enhance.setChecked(False)
        self.prompt_enhance.setToolTip(tr("adv.prompt_enhance.tt"))
        f.addRow(self.prompt_enhance)

        self.fixed_seed = QCheckBox(tr("adv.fixed_seed"))
        f.addRow(self.fixed_seed)

        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setValue(777)
        f.addRow(tr("adv.seed_lbl"), self.seed)

    def _apply_quality_preset(self, *_):
        """Preset Fast/Pro/HQ: compila steps+cfg (restano modificabili). 'Personalizzato'=no-op."""
        data = self.quality_preset.currentData()
        if not data:
            return
        steps, cfg = data
        if steps is not None:
            self.steps.setValue(int(steps))
        if cfg is not None:
            self.cfg.setValue(float(cfg))

    # ---------- suggest API (chiamata da VideoTab quando cambia il modello) ----------
    def set_suggested_steps(self, steps: int, note: str = ""):
        tip = tr("adv.suggested.tip").format(value=steps)
        if note:
            tip += tr("adv.suggested.note").format(note=note)
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
        tip = tr("adv.suggested.tip").format(value=cfg)
        if note:
            tip += tr("adv.suggested.note").format(note=note)
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
            self.encoder.setItemText(i, base + (("  " + tr("adv.encoder.absent")) if avail and fn not in avail else ""))

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
        self.sub.addTab(self._build_tools_tab(), tr("adv.tab.tools"))
        # Host per Import Models (riempito da attach_models dopo la costruzione)
        self._models_host = QWidget(); QVBoxLayout(self._models_host).setContentsMargins(0, 0, 0, 0)
        self.sub.addTab(self._models_host, tr("adv.tab.models"))

    def _build_tools_tab(self) -> QWidget:
        host = QWidget(); root = QVBoxLayout(host); root.setSpacing(10)
        intro = QLabel(tr("adv.intro.lbl"))
        intro.setWordWrap(True); intro.setStyleSheet("color: #aaa; padding: 4px;")
        root.addWidget(intro)

        # ── Generazione ──
        gen_grp = QGroupBox(tr("adv.gen.group")); gv = QVBoxLayout(gen_grp)
        gv.addWidget(self._big_btn(
            tr("adv.workflow.btn"),
            tr("adv.workflow.tt"), self.open_workflow))
        gv.addWidget(self._big_btn(
            tr("adv.analyzer.btn"),
            tr("adv.analyzer.tt"), self.open_analyzer))
        root.addWidget(gen_grp)

        # ── Galleria ──
        nav_grp = QGroupBox(tr("adv.gallery.group")); nv = QVBoxLayout(nav_grp)
        nv.addWidget(self._big_btn(
            tr("adv.navigator.btn"),
            tr("adv.navigator.tt"), self.open_navigator))
        root.addWidget(nav_grp)

        # ── Video lunghi (auto-extend) ──
        root.addWidget(self._build_longvideo_group())

        # ── Nota impostazioni sampler ──
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setStyleSheet("color: #333;")
        root.addWidget(sep)
        note = QLabel(tr("adv.note.lbl"))
        note.setWordWrap(True); note.setStyleSheet("color: #777; font-size: 11px; padding: 4px;")
        root.addWidget(note)
        root.addStretch(1)
        return host

    def _build_longvideo_group(self) -> QGroupBox:
        """Impostazioni per i video lunghi: auto-extend oltre la soglia + parametri continuità."""
        grp = QGroupBox(tr("adv.lv.group"))
        f = QFormLayout(grp)
        self.lv_auto = QCheckBox(tr("adv.lv.auto"))
        self.lv_auto.setChecked(True)
        self.lv_auto.setToolTip(tr("adv.lv.auto.tt"))
        f.addRow(self.lv_auto)
        self.lv_threshold = QDoubleSpinBox(); self.lv_threshold.setRange(4, 30); self.lv_threshold.setValue(20.0)
        self.lv_threshold.setSuffix(" s")
        self.lv_threshold.setToolTip(tr("adv.lv.threshold.tt"))
        f.addRow(tr("adv.lv.threshold_lbl"), self.lv_threshold)
        self.lv_seg = QDoubleSpinBox(); self.lv_seg.setRange(4, 15); self.lv_seg.setValue(10.0)
        self.lv_seg.setSuffix(" s")
        self.lv_seg.setToolTip(tr("adv.lv.seg.tt"))
        f.addRow(tr("adv.lv.seg_lbl"), self.lv_seg)
        self.lv_overlap = QSpinBox(); self.lv_overlap.setRange(0, 48); self.lv_overlap.setValue(24)
        self.lv_overlap.setToolTip(tr("adv.lv.overlap.tt"))
        f.addRow(tr("adv.lv.overlap_lbl"), self.lv_overlap)
        self.lv_context = QSpinBox(); self.lv_context.setRange(8, 64); self.lv_context.setValue(16)
        self.lv_context.setToolTip(tr("adv.lv.context.tt"))
        f.addRow(tr("adv.lv.context_lbl"), self.lv_context)
        self.lv_ctx_strength = QDoubleSpinBox(); self.lv_ctx_strength.setRange(0.0, 1.0)
        self.lv_ctx_strength.setSingleStep(0.05); self.lv_ctx_strength.setValue(0.5)
        self.lv_ctx_strength.setToolTip(tr("adv.lv.ctx_strength.tt"))
        f.addRow(tr("adv.lv.ctx_strength_lbl"), self.lv_ctx_strength)
        # --- anti-drift (degrado cumulativo della scena nei video molto lunghi) ---
        self.lv_ref_anchor = QDoubleSpinBox(); self.lv_ref_anchor.setRange(0.0, 1.0)
        self.lv_ref_anchor.setSingleStep(0.05); self.lv_ref_anchor.setValue(0.25)
        self.lv_ref_anchor.setToolTip(tr("adv.lv.ref_anchor.tt"))
        f.addRow(tr("adv.lv.ref_anchor_lbl"), self.lv_ref_anchor)
        self.lv_anchor_every = QSpinBox(); self.lv_anchor_every.setRange(0, 20); self.lv_anchor_every.setValue(0)
        self.lv_anchor_every.setToolTip(tr("adv.lv.anchor_every.tt"))
        f.addRow(tr("adv.lv.anchor_every_lbl"), self.lv_anchor_every)
        # --- resume dopo crash ---
        self.lv_resume = QCheckBox(tr("adv.lv.resume"))
        self.lv_resume.setChecked(True)
        self.lv_resume.setToolTip(tr("adv.lv.resume.tt"))
        f.addRow(self.lv_resume)
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
            "ref_anchor_strength": float(self.lv_ref_anchor.value()),
            "anchor_every": int(self.lv_anchor_every.value()),
            "resume": self.lv_resume.isChecked(),
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
