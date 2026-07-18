"""Tab 2 — Frame 1 / Prompt + editor inpaint/outpaint + generazione immagine di partenza.

- Canvas editor (mask_canvas) con inpaint/outpaint.
- Generazione immagine di partenza: arch FLUX / Z-Image / Ernie, checkpoint/modello filtrato,
  LoRA filtrate per architettura con icona 🔞 NSFW (come nel tab LoRA).
- Inpaint/outpaint: modello + LoRA compatibili (anch'esse con 🔞).
La generazione vera (FLUX Fill / txt2img) è cablata in main_window.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit, QLabel,
    QComboBox, QSlider, QFormLayout, QGroupBox, QSpinBox, QPushButton, QLineEdit,
    QCheckBox, QTabWidget, QScrollArea, QFrame, QDoubleSpinBox, QToolButton,
    QListWidget, QFileDialog,
)

from ui.mask_canvas import MaskCanvas
from ui.tabs.lora_tab import LoraTab
from core.lora_catalog import LoraCatalog
from core.model_catalog import ModelCatalog
from core import user_prefs
from core.gen_pipeline import GenPipeline
from core.flux2_encoder import is_compatible_encoder
from ui.i18n_manager import tr

DEFAULT_NEG = ("pc game, console game, video game, cartoon, childish, ugly, deformed, "
               "low quality, blurry, watermark, text")

# inpaint/outpaint: etichetta -> {arch (per LoRA), unet (file modello), steps}
# NB: il DEFAULT (prima voce) deve essere un modello NATIVO funzionante (FLUX.1 Fill), non FLUX.2 Klein
# (ramo edit-evoluto ancora su ComfyUI/deferred #14/#44): altrimenti inpaint/outpaint non generano di default.
# NB (regola utente): le label mostrano SEMPRE il file reale — «ciò che scegli è ciò che gira», nessun modello
# nascosto dietro un nome generico. Nessuna sostituzione automatica del checkpoint.
INPAINT_MODELS = {
    "FLUX_DEV_Fill  (Fill inpaint/outpaint, 22GB)":
        {"arch": "Flux", "unet": "Flux\\FLUX_DEV_Fill.safetensors", "steps": 40},
    "fluxunchained-fill-hyp8-Q5  (Fill GGUF, RAM bassa)":
        {"arch": "Flux", "unet": "Flux\\fluxunchained-fill-hyp8-Q5_K_S.gguf", "steps": 8},
    # FLUX BASE (64 canali) → FluxInpaintPipeline: SEGUE il prompt (sostituisci/aggiungi ciò che descrivi).
    "getphat Flux Reality v30 Hardcore FP8  (base, NSFW)":
        {"arch": "Flux", "unet": "Flux\\getphat Flux Reality v30 Hardcore FP8.safetensors", "steps": 28},
    "FLUX_Krea_Dev  (base, foto)":
        {"arch": "Flux", "unet": "Flux\\FLUX_Krea_Dev.safetensors", "steps": 28},
    "miraclein_flux2klein_nsfw_v2  (FLUX.2 Klein, edit)":
        {"arch": "Flux2", "unet": "Flux\\miraclein_flux2klein_nsfw_v2_fp8.safetensors", "steps": 12},
    "Nessuno (solo LTX I2V)": {"arch": None, "unet": None, "steps": 0},
}
# generazione: arch -> (architettura catalogo LoRA, keyword per filtrare i modelli in diffusion_models)
GEN_ARCH = {
    "FLUX": ("Flux", ("flux", "getphat", "fuxcapacity")),   # "krea" tolto: ora ambiguo con Krea 2
    "Krea 2": ("Krea2", ("krea-2", "krea2")),               # arch immagine alternativa (Turbo/Raw)
    "Z-Image": ("Z-Image", ("z-image", "z_image", "zimage")),
    "Ernie": ("Ernie", ("ernie",)),
}


def _is_flux2(name: str) -> bool:
    """FLUX.2 (es. FLUX_2_Dev): VAE/encoder diversi da FLUX.1 → non supportato dalle pipeline FLUX.1."""
    n = (name or "").lower()
    return "flux_2" in n or "flux2" in n or "flux 2" in n


def fill_lora_combo(combo: QComboBox, arch: str | None, allow_nsfw: bool = True):
    """Popola un combo con le LoRA dell'architettura (filtrate per NSFW), icona 🔞 per le NSFW."""
    combo.clear(); combo.addItem(tr("f1.inp.none"), None)
    if not arch:
        return
    try:
        cat = LoraCatalog(compatible_arch=arch)
        for e in cat.loras(allow_nsfw=allow_nsfw):
            label = ("🔞 " if e.nsfw else "") + e.name
            combo.addItem(label, e.comfy_name)
    except Exception:
        pass


# modelli upscale preferiti (i più potenti per foto reali), se presenti
_UPSCALE_PREFERRED = ("4xNomos8kSCHAT-L.pth", "4xRealWebPhoto_v4_dat2.safetensors",
                      "4xNomos2_hq_dat2.safetensors", "4x-UltraSharpV2.safetensors",
                      "4x-UltraSharp.safetensors")


class Frame1Tab(QWidget):
    request_generate = Signal()         # genera immagine di partenza
    request_segment_text = Signal()     # seleziona oggetto da testo (CLIPSeg+SAM)
    request_upscale = Signal()          # aumenta definizione (upscale AI)
    request_recolor = Signal()          # cambia colore (Kontext prompt-based)
    request_restore_photo = Signal()    # restaura / colora foto (invia al tab Restauro)
    request_integrate = Signal()        # integra immagini (spostato qui dal canvas)
    request_compose = Signal()          # componi più foto (multi-reference / stitch)

    def __init__(self):
        super().__init__()
        self._models: list[str] = []      # diffusion_models disponibili (da ComfyUI)
        self._inpaint_avail: set[str] = set()   # unet_name realmente presenti in ComfyUI
        # Checkpoint FLUX (es. getphat Reality, fuxcapacity) montati come modelli inpaint
        # via CheckpointLoaderSimple. label_inpaint -> ckpt_name (relativo a models/checkpoints).
        self._dynamic_checkpoints: dict[str, str] = {}
        self._mcat = ModelCatalog()       # per architettura + nsfw dei modelli

        # spazio di lavoro + selezione AI: vivono nel pannello principale (esposti a main_window)
        self.canvas = MaskCanvas()

        # controlli in SOTTO-TAB (tutti allineati in alto)
        # Il sub-tab "Immagini" è stato rimosso: i suoi controlli sono migrati altrove
        # (NSFW → «Genera immagine», Prompt/Negativo → tab Video). _prompt_tab() viene
        # comunque chiamato per istanziare i widget placeholder (img_nsfw, prompt, neg)
        # necessari come back-compat per le call-site esistenti.
        _placeholder = self._prompt_tab()        # crea self.img_nsfw / self.prompt / self.neg
        _placeholder.setVisible(False)
        sub = QTabWidget(); self.sub = sub        # esposto per navigazione/test dei sotto-tab
        sub.addTab(self._scroll(self._gen_box()), tr("f1.tab.gen"))
        sub.addTab(self._scroll(self._inpaint_box()), tr("f1.tab.inpaint"))
        sub.addTab(self._scroll(self._color_box()), tr("f1.tab.color"))
        sub.addTab(self._scroll(self._upscale_box()), tr("f1.tab.upscale"))
        self.segment_box = self._segment_box()    # selezione AI (sotto il canvas, nel pannello principale)

        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(sub)
        lay.addWidget(_placeholder)     # hidden: tiene vivi i widget back-compat
        self._reload_inpaint_loras(); self._reload_gen(); self._reload_flux_combos()

    def _scroll(self, w: QWidget) -> QScrollArea:
        """Wrappa il widget in alto (stretch sotto) dentro uno scroll → controlli allineati in alto."""
        cont = QWidget(); cv = QVBoxLayout(cont); cv.setContentsMargins(8, 8, 8, 8)
        cv.addWidget(w); cv.addStretch(1)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QScrollArea.NoFrame); sc.setWidget(cont)
        return sc

    def _prompt_tab(self) -> QWidget:
        # NSFW selector → spostato in "Genera immagine"; Prompt globale/negativo → spostati
        # nel tab Video. Resta un placeholder pronto ad accogliere altri controlli "Immagini".
        # NOTE: self.prompt / self.neg / self.img_nsfw vengono creati comunque (alias da
        # main_window.py post-init verso i widget del tab Video) per non rompere
        # get_values() e tutte le call-site esistenti.
        w = QWidget(); bl = QVBoxLayout(w)
        # img_nsfw è creato qui ma il suo widget viene inserito in _gen_box (vedi sotto).
        self.img_nsfw = QComboBox(); self.img_nsfw.addItems(["SFW", "NSFW"])
        self.img_nsfw.currentTextChanged.connect(self._on_img_nsfw)
        # placeholder text — il tab è ora vuoto, in attesa di altri controlli immagine.
        info = QLabel(tr("f1.info.lbl"))
        info.setWordWrap(True); info.setStyleSheet("color:#888; padding:8px;")
        bl.addWidget(info)
        # widget back-compat per get_values() — sostituiti con alias da main_window
        # subito dopo la creazione di video_tab, ma servono già qui per evitare AttributeError.
        self.prompt = QPlainTextEdit(); self.prompt.setVisible(False)
        self.neg = QPlainTextEdit(DEFAULT_NEG); self.neg.setVisible(False)
        bl.addWidget(self.prompt); bl.addWidget(self.neg)
        return w

    # ---------- generazione immagine di partenza ----------
    def _gen_box(self) -> QGroupBox:
        box = QGroupBox(tr("f1.gen.group"))
        f = QFormLayout(box)
        self.img_nsfw.setToolTip(tr("f1.gen.content.tt", "Scegli SFW (contenuti sicuri) o NSFW (per adulti): filtra modelli e LoRA di conseguenza."))
        f.addRow(tr("f1.gen.content_lbl"), self.img_nsfw)
        self.gen_arch = QComboBox(); self.gen_arch.addItems(list(GEN_ARCH.keys()))
        self.gen_arch.currentTextChanged.connect(self._reload_gen)
        self.gen_arch.currentTextChanged.connect(self._sync_gen_lora_arch)
        self.gen_arch.currentTextChanged.connect(self._apply_arch_defaults)
        self.gen_arch.setToolTip(tr("f1.gen.arch.tt", "Architettura del modello immagine (FLUX, Z-Image o Ernie): determina modelli, encoder e LoRA compatibili."))
        f.addRow(tr("f1.gen.arch_lbl"), self.gen_arch)
        # Selettore MOTORE di generazione: Comfy (model_management ComfyUI, VRAM dinamica, FLUX.2 ~11x più
        # veloce, scala 8→48GB) vs Diffusers (pipeline diffusers). Persistito in user_prefs.
        self.gen_engine_sel = QComboBox()
        self.gen_engine_sel.addItem(tr("f1.gen.engine.comfy", "Comfy (veloce, VRAM dinamica)"), "comfy")
        self.gen_engine_sel.addItem(tr("f1.gen.engine.diffusers", "Diffusers"), "diffusers")
        try:
            from core import user_prefs as _up
            i = self.gen_engine_sel.findData(_up.get_engine("image"))
            self.gen_engine_sel.setCurrentIndex(i if i >= 0 else 0)
            self.gen_engine_sel.currentIndexChanged.connect(
                lambda *_: _up.set_engine("image", self.gen_engine_sel.currentData()))
        except Exception:                                       # noqa: BLE001
            pass
        self.gen_engine_sel.setToolTip(tr("f1.gen.engine.tt",
            "Comfy: gestione VRAM dinamica (consigliato, scala su tutte le GPU). Diffusers: pipeline classica."))
        f.addRow(tr("f1.gen.engine_lbl", "Motore"), self.gen_engine_sel)
        self.gen_model = QComboBox()
        self.gen_model.setToolTip(tr("f1.gen.model.tt", "Modello usato per generare l'immagine; i colori indicano la fattibilità con la VRAM disponibile."))
        self.gen_model.currentIndexChanged.connect(self._update_gen_star)
        self.gen_model.currentIndexChanged.connect(self._refilter_gen_encoder)
        self.gen_model.currentIndexChanged.connect(self._update_auto_components)
        self.gen_model.activated.connect(self._maybe_offer_gen_download)   # download-on-select (Krea 2)
        gmrow = QHBoxLayout(); gmrow.setContentsMargins(0, 0, 0, 0)
        gmrow.addWidget(self.gen_model, 1)
        # filtro precisione (fp16/fp8/fp4/gguf) + colori VRAM (verde/giallo/rosso) sul dropdown modelli
        from ui.widgets.precision_combo import make_precision_selector
        self.gen_precision = make_precision_selector()
        self.gen_precision.setMaximumWidth(110)
        self.gen_precision.currentIndexChanged.connect(lambda *_: self._reload_gen())
        gmrow.addWidget(self.gen_precision)
        self.gen_star = QToolButton(); self.gen_star.setText("☆")
        self.gen_star.setStyleSheet("QToolButton{border:none;font-size:16px;color:#777;}")
        self.gen_star.setToolTip(tr("f1.gen.star.tt"))
        self.gen_star.clicked.connect(self._toggle_gen_fav)
        gmrow.addWidget(self.gen_star)
        gmw = QWidget(); gmw.setLayout(gmrow); f.addRow(tr("f1.model_lbl"), gmw)
        # Encoder / VAE / Type CLIP — popolati dall'inventario ComfyUI via set_gen_inventory().
        # I default si auto-impostano al cambio di architettura (_apply_arch_defaults).
        self.gen_encoder = QComboBox(); self.gen_encoder.setEditable(False)
        self.gen_encoder.setToolTip(tr("f1.gen.encoder.tt", "Text encoder che interpreta il prompt; 'auto' sceglie quello adatto al modello."))
        f.addRow(tr("f1.gen.encoder_lbl"), self.gen_encoder)
        self.gen_clip_type = QComboBox()
        f.addRow(tr("f1.gen.clip_lbl"), self.gen_clip_type)
        self.gen_vae = QComboBox()
        self.gen_vae.setToolTip(tr("f1.gen.vae.tt", "VAE che decodifica il latente in immagine; 'auto' usa quello del modello."))
        f.addRow(tr("f1.gen.vae_lbl"), self.gen_vae)
        # LoRA multi-select (come nel tab Video)
        lora_group = QGroupBox(tr("f1.gen.lora.group"))
        lora_vl = QVBoxLayout(lora_group)
        self.gen_lora_tab = LoraTab(allow_nsfw_getter=self._allow_nsfw, embedded=True, arch="Flux")
        lora_vl.addWidget(self.gen_lora_tab)
        f.addRow(lora_group)
        # Sampler / scheduler / steps / CFG con default applicati per arch.
        # "auto" come prima voce: lascia decidere al pipeline il default per architettura.
        self.gen_sampler = QComboBox()
        self.gen_sampler.addItem("auto", None)
        # euler/dpmpp affidabili in cima. I RES restano disponibili (su Flux spesso ok) ma NON
        # in cima: su LTX-2.3 rompono (rumore) — verificato — quindi niente default rischiosi.
        self.gen_sampler.addItems(["euler", "euler_ancestral_cfg_pp", "dpmpp_2m", "res_2s"])
        self.gen_sampler.setToolTip(tr("f1.gen.sampler.tt"))
        f.addRow(tr("f1.sampler_lbl"), self.gen_sampler)
        self.gen_scheduler = QComboBox()
        self.gen_scheduler.addItem("auto", None)
        self.gen_scheduler.addItems(["karras", "bong_tangent", "beta", "simple"])
        self.gen_scheduler.setToolTip(tr("f1.gen.scheduler.tt"))
        f.addRow(tr("f1.scheduler_lbl"), self.gen_scheduler)
        self.gen_enhance = QCheckBox(tr("f1.gen.enhance"))
        self.gen_enhance.setChecked(False)
        self.gen_enhance.setToolTip(tr("f1.gen.enhance.tt"))
        f.addRow(self.gen_enhance)
        self.gen_steps = QSpinBox(); self.gen_steps.setRange(1, 80); self.gen_steps.setValue(20)
        self.gen_steps.setToolTip(tr("f1.steps.tt", "Numero di passi di diffusione: più passi = più dettaglio e tempi più lunghi."))
        f.addRow(tr("f1.steps_lbl"), self.gen_steps)
        self.gen_cfg = QDoubleSpinBox(); self.gen_cfg.setRange(0.0, 20.0); self.gen_cfg.setSingleStep(0.1)
        self.gen_cfg.setValue(3.5)
        self.gen_cfg.setToolTip(tr("f1.gen.cfg.tt", "Guidance (CFG): quanto l'immagine aderisce al prompt; valori troppo alti creano artefatti."))
        f.addRow(tr("f1.cfg_lbl"), self.gen_cfg)
        # Seed + checkbox random/fisso
        seedrow = QHBoxLayout()
        self.gen_seed = QSpinBox(); self.gen_seed.setRange(0, 2_147_483_647); self.gen_seed.setValue(0)
        self.gen_seed.setToolTip(tr("f1.seed.tt", "Seme del generatore casuale: fissarlo rende la generazione riproducibile."))
        self.gen_seed_rand = QCheckBox(tr("f1.seed_rand")); self.gen_seed_rand.setChecked(True)
        self.gen_seed_rand.setToolTip(tr("f1.seed_rand.tt", "Se attivo usa un seme casuale a ogni generazione; disattiva per fissarlo."))
        self.gen_seed.setEnabled(False)
        self.gen_seed_rand.toggled.connect(lambda on: self.gen_seed.setEnabled(not on))
        seedrow.addWidget(self.gen_seed, 1); seedrow.addWidget(self.gen_seed_rand)
        seedw = QWidget(); seedw.setLayout(seedrow); f.addRow(tr("f1.seed_lbl"), seedw)
        # Prompt positivo dedicato (opzionale) + Prompt negativo
        self.gen_prompt = QPlainTextEdit(); self.gen_prompt.setMaximumHeight(40)
        self.gen_prompt.setPlaceholderText(tr("f1.gen.prompt.ph"))
        self.gen_prompt.setToolTip(tr("f1.gen.prompt.tt", "Prompt positivo dedicato all'immagine di partenza (se vuoto usa quello globale)."))
        from PySide6.QtWidgets import QWidget as _QW, QHBoxLayout as _QH
        _pbb = QToolButton(); _pbb.setText("🧱"); _pbb.setCursor(Qt.PointingHandCursor)
        _pbb.setToolTip(tr("f1.gen.builder.tt", "Costruttore di prompt strutturato (campi → prompt)."))
        _pbb.clicked.connect(lambda: self._open_prompt_builder())
        _pw = _QW(); _pl = _QH(_pw); _pl.setContentsMargins(0, 0, 0, 0); _pl.setSpacing(4)
        _pl.addWidget(self.gen_prompt, 1); _pl.addWidget(_pbb)
        f.addRow(tr("f1.gen.prompt_lbl"), _pw)
        self.gen_neg = QPlainTextEdit(DEFAULT_NEG); self.gen_neg.setMaximumHeight(40)
        self.gen_neg.setPlaceholderText(tr("f1.gen.neg.ph"))
        self.gen_neg.setToolTip(tr("f1.gen.neg.tt", "Prompt negativo: elementi da evitare nell'immagine generata."))
        f.addRow(tr("f1.neg_lbl"), self.gen_neg)
        self.gen_btn = QPushButton(tr("f1.gen.btn"))
        self.gen_btn.setToolTip(tr("f1.gen.btn.tt", "Genera l'immagine di partenza con i parametri impostati."))
        self.gen_btn.clicked.connect(lambda: self.request_generate.emit())

        def _open_prompt_builder(_self=self):
            from ui.widgets.prompt_builder import PromptBuilderDialog
            _self._pb_dlg = PromptBuilderDialog(_self)
            _self._pb_dlg.applied.connect(_self._apply_built_prompt)
            _self._pb_dlg.show()

        def _apply_built_prompt(p, n, _self=self):
            if p:
                _self.gen_prompt.setPlainText(p)
            if n:
                _self.gen_neg.setPlainText(n)
        self._open_prompt_builder = _open_prompt_builder
        self._apply_built_prompt = _apply_built_prompt
        f.addRow(self.gen_btn)
        self.gen_warn_lbl = QLabel("")
        self.gen_warn_lbl.setWordWrap(True)
        self.gen_warn_lbl.setStyleSheet("color:#e67e22; font-size:11px;")
        self.gen_warn_lbl.setVisible(False)
        f.addRow(self.gen_warn_lbl)
        # --- Face adapter PuLID: imprime l'identità di un volto da una foto (solo FLUX) ---
        pulid_group = QGroupBox(tr("f1.pulid.group")); self._pulid_group = pulid_group
        pf = QFormLayout(pulid_group)
        self.gen_pulid_enable = QCheckBox(tr("f1.pulid.enable"))
        self.gen_pulid_enable.setToolTip(tr("f1.pulid.enable.tt"))
        pf.addRow(self.gen_pulid_enable)
        prow = QHBoxLayout()
        self.gen_pulid_face = QLineEdit(); self.gen_pulid_face.setPlaceholderText(tr("f1.pulid.face.ph"))
        self.gen_pulid_face.setToolTip(tr("f1.pulid.face.tt", "Percorso della foto del volto da cui PuLID copia l'identità (solo FLUX)."))
        self.gen_pulid_face_btn = QPushButton(tr("f1.pulid.face.btn"))
        self.gen_pulid_face_btn.setToolTip(tr("f1.pulid.face.btn.tt", "Sfoglia il disco per scegliere la foto del volto di riferimento."))
        self.gen_pulid_face_btn.clicked.connect(self._pick_pulid_face)
        prow.addWidget(self.gen_pulid_face, 1); prow.addWidget(self.gen_pulid_face_btn)
        pf.addRow(tr("f1.pulid.face_lbl"), self._wrapw(prow))
        self.gen_pulid_strength = QDoubleSpinBox()
        self.gen_pulid_strength.setRange(0.0, 3.0); self.gen_pulid_strength.setSingleStep(0.05)
        self.gen_pulid_strength.setValue(0.8)
        self.gen_pulid_strength.setToolTip(tr("f1.pulid.strength.tt"))
        pf.addRow(tr("f1.pulid.strength_lbl"), self.gen_pulid_strength)
        # Parametri avanzati PuLID-FLUX ufficiali (prima inutilizzati)
        self.gen_pulid_start = QSpinBox(); self.gen_pulid_start.setRange(0, 10); self.gen_pulid_start.setValue(0)
        self.gen_pulid_start.setToolTip(tr("f1.pulid.start.tt",
            "Da quale step iniziare a iniettare l'identità (0 = subito). Valori 1-4 lasciano formare prima la "
            "posa/composizione dal prompt, poi applicano il volto → pose e prompt più fedeli."))
        pf.addRow(tr("f1.pulid.start_lbl", "Start step"), self.gen_pulid_start)
        self.gen_pulid_truecfg = QDoubleSpinBox(); self.gen_pulid_truecfg.setRange(1.0, 10.0)
        self.gen_pulid_truecfg.setSingleStep(0.5); self.gen_pulid_truecfg.setValue(1.0)
        self.gen_pulid_truecfg.setToolTip(tr("f1.pulid.truecfg.tt",
            "True CFG (1.0 = veloce/off). >1 migliora l'aderenza al prompt e riduce il dominio dell'identità "
            "(più lento). Attivo solo se la pipeline lo supporta."))
        pf.addRow(tr("f1.pulid.truecfg_lbl", "True CFG"), self.gen_pulid_truecfg)
        f.addRow(pulid_group)
        # Default coerenti con l'architettura iniziale
        self._apply_arch_defaults(self.gen_arch.currentText())
        return box

    def _pick_pulid_face(self):
        """Selettore foto volto per PuLID — accetta PIÙ foto (media embedding = identità più robusta)."""
        from PySide6.QtWidgets import QFileDialog
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("f1.pulid.face.btn"), "",
            "Immagini (*.png *.jpg *.jpeg *.webp *.bmp)")
        if paths:
            self.gen_pulid_face.setText("|".join(paths))     # più foto separate da | → media in PuLID

    # Default per architettura: CENTRALIZZATI in core/model_catalog.ARCH_DEFAULTS (review FASE G) —
    # un solo posto da aggiornare quando cambia un'arch, GUI e backend leggono la stessa fonte.
    from core.model_catalog import ARCH_DEFAULTS as _ARCH_DEFAULTS

    def _apply_arch_defaults(self, arch_lbl: str):
        if hasattr(self, "_pulid_group"):                    # PuLID è solo FLUX: disabilita su Z-Image/Ernie
            is_flux = (arch_lbl == "FLUX")
            self._pulid_group.setEnabled(is_flux)
            self._pulid_group.setToolTip("" if is_flux else tr("f1.pulid.only_flux",
                "PuLID (face adapter) funziona solo con l'architettura FLUX."))
        d = self._ARCH_DEFAULTS.get(arch_lbl)
        if not d:
            return
        # Default ARCH = "auto" su sampler/scheduler/clip_type: l'utente vede subito che è
        # in modalità auto, e il pipeline applica i valori corretti del modello scelto.
        if hasattr(self, "gen_sampler"):
            self.gen_sampler.setCurrentIndex(0)              # "auto"
            self.gen_sampler.setToolTip(tr("f1.gen.default_for.tt").format(arch=arch_lbl, val=d['sampler']))
        if hasattr(self, "gen_scheduler"):
            self.gen_scheduler.setCurrentIndex(0)            # "auto"
            self.gen_scheduler.setToolTip(tr("f1.gen.default_for.tt").format(arch=arch_lbl, val=d['scheduler']))
        if hasattr(self, "gen_steps"):
            self.gen_steps.setValue(int(d["steps"]))
        if hasattr(self, "gen_cfg"):
            self.gen_cfg.setValue(float(d["cfg"]))
        if hasattr(self, "gen_clip_type"):
            self.gen_clip_type.setCurrentIndex(0)            # "auto"
            self.gen_clip_type.setToolTip(tr("f1.gen.default_for.tt").format(arch=arch_lbl, val=d['clip_type']))

    def _sync_gen_lora_arch(self, arch_lbl: str):
        """Sincronizza l'architettura del LoraTab con quella selezionata per la generazione."""
        arch_map = {"FLUX": "Flux", "Z-Image": "Z-Image", "Ernie": "Ernie"}
        arch = arch_map.get(arch_lbl, "Flux")
        self.gen_lora_tab.set_arch(arch)

    def _allow_nsfw(self) -> bool:
        return self.img_nsfw.currentText() == "NSFW"

    def _on_img_nsfw(self, *_):
        """Cambio NSFW/SFW immagine → re-filtro istantaneo di modelli e LoRA."""
        self._reload_gen()
        self._reload_inpaint_loras()
        self._reload_flux_combos()

    def _color_name_combo(self, combo):
        """Colora le voci di un combo i cui item TESTO sono nomi-file di modelli (verde/giallo/rosso)."""
        try:
            from core import model_precision as _mp
            from PySide6.QtGui import QColor
            free = _mp.gpu_free_gb()
            for i in range(combo.count()):
                info = _mp.classify(combo.itemText(i), free)
                col = _mp.QT_COLORS.get(info["color"])
                if col:
                    combo.setItemData(i, QColor(col), Qt.ForegroundRole)
                combo.setItemData(i, f"{info['precision'].upper()} · ~{info['footprint_gb']}GB", Qt.ToolTipRole)
        except Exception:                                    # noqa: BLE001
            pass

    def _color_curated_combo(self, combo, spec: dict, *, key: str = "unet"):
        """Colora le voci di un combo CURATO (label→dict) per fattibilità VRAM (verde/giallo/rosso),
        classificando il file modello in spec[label][key]. Usato per inpaint/integra (liste brevi)."""
        try:
            from core import model_precision as _mp
            from PySide6.QtGui import QColor
            free = _mp.gpu_free_gb()
            for i in range(combo.count()):
                label = combo.itemText(i)
                ent = spec.get(label) or {}
                fname = ent.get(key)
                if not fname:
                    continue
                info = _mp.classify(fname, free)
                col = _mp.QT_COLORS.get(info["color"])
                if col:
                    combo.setItemData(i, QColor(col), Qt.ForegroundRole)
                combo.setItemData(i, f"{info['precision'].upper()} · ~{info['footprint_gb']}GB", Qt.ToolTipRole)
        except Exception:                                    # noqa: BLE001
            pass

    def _reload_gen(self, *_):
        arch_lbl = self.gen_arch.currentText()
        if arch_lbl == "Krea 2":                              # snapshot diffusers → varianti dal catalogo (no file)
            self._reload_gen_krea2()
            return
        _lora_arch, kws = GEN_ARCH.get(arch_lbl, (None, ()))
        allow = self._allow_nsfw()
        models = []
        # FLUX: includi sia FLUX.1 che FLUX.2 da diffusion_models, escludendo solo
        # Fill (inpaint) e Kontext (editing). Aggiungi anche i checkpoint Flux.
        if arch_lbl == "FLUX":
            for m in (self._models or []):
                ml = m.lower()
                # Escludi i NON-generativi: Fill (inpaint), Kontext (editing), Redux (adattatore di
                # image-variation) e Klein (FLUX.2 edit-instruct, role=edit nel detector: il loader
                # t2i flux2 non mappa le sue chiavi → 'double_blocks.7.img_mlp.2'; vive nel ramo inpaint).
                if "fill" in ml or "kontext" in ml or "redux" in ml or "klein" in ml:
                    continue
                if any(k in ml for k in kws):
                    if not allow and self._mcat.is_nsfw(m):
                        continue
                    models.append(m)
            # Checkpoint Flux (lista dinamica dall'inventario ComfyUI)
            for c in (getattr(self, "_gen_checkpoint_models", []) or []):
                cl = c.lower()
                if "kontext" in cl or "redux" in cl or _is_flux2(c):
                    continue
                if any(k in cl for k in kws):
                    if not allow and self._mcat.is_nsfw(c):
                        continue
                    if c not in models:
                        models.append(c)
        else:
            for m in self._models:
                ml = m.lower()
                if not any(k in ml for k in kws):
                    continue
                if not allow and self._mcat.is_nsfw(m):
                    continue
                models.append(m)
        # fav-first: i modelli preferiti in cima (i nomi restano i filename reali, nessun prefisso
        # nel testo → currentText() resta valido per get_gen_config)
        models = sorted(models, key=lambda m: (not user_prefs.is_fav_model(m), m.lower()))
        # filtro precisione + colori VRAM (verde/giallo/rosso) via helper riutilizzabile
        from ui.widgets import precision_combo as _pc
        prec = self.gen_precision.currentData() if hasattr(self, "gen_precision") else "all"
        if models:
            _pc.populate(self.gen_model, models, prec or "all",
                         prepend=[(tr("f1.gen.auto"), "__auto__")])
            if self.gen_model.count() <= 1:    # filtro troppo restrittivo → nessun modello di quella precisione
                self.gen_model.addItem(tr("f1.gen.no_model"), None)
        else:
            self.gen_model.blockSignals(True); self.gen_model.clear()
            self.gen_model.addItem(tr("f1.gen.no_model"), None)
            self.gen_model.blockSignals(False)
        self._update_gen_star()
        # Controlla se il golden template è disponibile per questa arch
        arch_key = {"FLUX": "flux", "Z-Image": "zimage", "Ernie": "ernie"}.get(arch_lbl, "flux")
        golden_ok = GenPipeline.golden_exists(arch_key)
        if hasattr(self, "gen_warn_lbl"):
            if not golden_ok:
                self.gen_warn_lbl.setText(
                    tr("f1.gen.warn.tt").format(arch=arch_lbl, key=arch_key))
                self.gen_warn_lbl.setVisible(True)
                self.gen_btn.setEnabled(False)
            else:
                self.gen_warn_lbl.setVisible(False)
                self.gen_btn.setEnabled(True)
        if hasattr(self, "gen_lora_tab"):
            self.gen_lora_tab.refresh()

    def _inp_label(self) -> str:
        return self.inp_model.currentText().split("  ⛔")[0] if self.inp_model.count() else ""

    def _update_inp_star(self, *_):
        if not hasattr(self, "inp_star"):
            return
        is_fav = user_prefs.is_fav_model("inpaint:" + self._inp_label())
        self.inp_star.setText("★" if is_fav else "☆")
        self.inp_star.setStyleSheet(
            "QToolButton{border:none;font-size:16px;color:%s;}" % ("#f5c518" if is_fav else "#777"))

    def _toggle_inp_fav(self):
        lbl = self._inp_label()
        if not lbl:
            return
        user_prefs.toggle_fav_model("inpaint:" + lbl)
        self._update_inp_star()

    def _reload_gen_krea2(self, *_):
        """Popola il dropdown con le varianti Krea 2 (Turbo/Raw) dal catalogo pesi, con badge ⬇ se non presenti
        (Krea 2 è uno snapshot diffusers, non un checkpoint-file → download-on-select come per il video)."""
        from core import weights_catalog as wc
        from PySide6.QtGui import QColor
        entries = {e["id"]: e for e in wc.load()}
        self.gen_model.blockSignals(True)
        self.gen_model.clear()
        for cat_id, label in (("krea2_turbo", tr("f1.gen.krea2.turbo", "Krea 2 Turbo (8 step, veloce)")),
                              ("krea2_raw", tr("f1.gen.krea2.raw", "Krea 2 Raw (52 step, qualità)"))):
            e = entries.get(cat_id)
            if not e:
                continue
            present = False
            try:
                present = wc.ready_with_deps(e)
            except Exception:                                 # noqa: BLE001
                pass
            gb = e.get("approx_gb", 0)
            self.gen_model.addItem(f"{'' if present else '⬇ '}{label}  · ~{gb}GB", cat_id)
            if not present:
                self.gen_model.setItemData(self.gen_model.count() - 1, QColor("#e8b339"), Qt.ForegroundRole)
        self.gen_model.blockSignals(False)
        self._update_gen_star()

    def _maybe_offer_gen_download(self, *_):
        """Solo su click utente: se l'arch immagine è a snapshot (Krea 2) e la variante scelta non è presente,
        offre il download (con dipendenze) via download_weights, come nel tab Video."""
        if self.gen_arch.currentText() != "Krea 2":
            return
        cat_id = self.gen_model.currentData()
        if not isinstance(cat_id, str) or not cat_id.startswith("krea2"):
            return
        from core import weights_catalog as wc
        entries = {e["id"]: e for e in wc.load()}
        e = entries.get(cat_id)
        if not e:
            return
        missing = wc.missing_ids_with_deps(e)
        if not missing:
            return
        gb = sum(int(entries.get(m, {}).get("approx_gb", 0) or 0) for m in missing)
        from PySide6.QtWidgets import QMessageBox
        r = QMessageBox.question(
            self, tr("f1.gen.dl.title", "Scaricare il modello?"),
            tr("f1.gen.dl.msg", "Il modello '{m}' non è scaricato: {ids} · ~{gb}GB. Scaricarlo ora?").format(
                m=cat_id, ids=", ".join(missing), gb=gb),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if r == QMessageBox.Yes:
            self._download_gen_weights(missing)

    def _download_gen_weights(self, ids):
        """Scarica pesi del catalogo (modello immagine + deps) con dialog di progresso non bloccante."""
        import sys
        from pathlib import Path
        from PySide6.QtCore import QProcess
        from PySide6.QtWidgets import QProgressDialog
        ids_arg = ",".join([ids] if isinstance(ids, str) else list(ids))
        script = str(Path(__file__).resolve().parents[2] / "scripts" / "download_weights.py")
        dlg = QProgressDialog(tr("f1.gen.dl.progress", "Scaricamento {m}…").format(m=ids_arg),
                              tr("common.cancel", "Annulla"), 0, 0, self)
        dlg.setWindowTitle(tr("f1.gen.dl.title", "Scaricamento modello"))
        dlg.setMinimumWidth(420)
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)

        def _fin(*_):
            dlg.close()
            self._reload_gen_krea2()                          # riflette la nuova presenza (toglie il ⬇)
        proc.finished.connect(_fin)
        dlg.canceled.connect(proc.kill)
        proc.start(sys.executable, ["-u", script, "--ids", ids_arg])
        dlg.show()

    def _gen_model_name(self) -> str:
        """Nome modello selezionato; risolve la voce '⚡ Auto' al miglior modello dell'arch corrente
        (stesso criterio del motore auto_select: preferiti + qualità di famiglia)."""
        if self.gen_arch.currentText() == "Krea 2":           # snapshot: nessun file-modello (risolto dal worker)
            return ""
        if not self.gen_model.count():
            return ""
        if self.gen_model.currentData() == "__auto__":
            cands = [self.gen_model.itemData(i) for i in range(self.gen_model.count())
                     if self.gen_model.itemData(i) not in ("__auto__", None)]
            if not cands:
                return ""
            try:
                from core import auto_select
                return max(cands, key=lambda m: (user_prefs.is_fav_model(m), auto_select._pref(m)))
            except Exception:                            # noqa: BLE001
                return cands[0]
        t = self.gen_model.currentText()
        return "" if t.startswith("(") else t

    def _update_gen_star(self, *_):
        if not hasattr(self, "gen_star"):
            return
        m = self._gen_model_name()
        is_fav = bool(m) and user_prefs.is_fav_model(m)
        self.gen_star.setText("★" if is_fav else "☆")
        self.gen_star.setStyleSheet(
            "QToolButton{border:none;font-size:16px;color:%s;}" % ("#f5c518" if is_fav else "#777"))

    def _toggle_gen_fav(self):
        m = self._gen_model_name()
        if not m:
            return
        user_prefs.toggle_fav_model(m)
        self._reload_gen()      # riordina (preferiti in cima)

    def _reload_flux_combos(self):
        """Re-popola i combo FLUX:
        - `up_refine_model` (rifinitura upscale): solo FLUX.2 (escludendo Fill)
        - `integrate_model`: contestuale al motore Integra selezionato
            * Kontext (FLUX.1)  → modelli che contengono "kontext" (es. FLUX_Kontext_Dev)
            * FLUX.2 (sperimentale) → modelli FLUX.2 (escludendo Fill)
        """
        allow = self._allow_nsfw()
        # rifinitura upscale → sempre FLUX.2
        flux2 = [m for m in self._models
                 if _is_flux2(m) and "fill" not in m.lower()
                 and (allow or not self._mcat.is_nsfw(m))]
        refine_combo = getattr(self, "up_refine_model", None)
        if refine_combo is not None:
            cur = refine_combo.currentText()
            refine_combo.clear()
            refine_combo.addItems(flux2) if flux2 else refine_combo.addItem(tr("f1.flux2.none"))
            if flux2:
                self._color_name_combo(refine_combo)
            if cur in flux2:
                refine_combo.setCurrentText(cur)
        # integra: liste separate per engine
        ic = getattr(self, "integrate_model", None)
        if ic is not None:
            cur = ic.currentText()
            engine = (self.integrate_engine.currentIndex()
                      if hasattr(self, "integrate_engine") else 0)
            if engine == 2:                        # Krea 2 Edit → varianti Turbo/Raw (snapshot, no file)
                ic.clear()
                for cid, lbl in (("krea2_turbo", tr("f1.gen.krea2.turbo", "Krea 2 Turbo (8 step, veloce)")),
                                 ("krea2_raw", tr("f1.gen.krea2.raw", "Krea 2 Raw (52 step, qualità)"))):
                    ic.addItem(lbl, {"krea2_variant": "raw" if cid.endswith("raw") else "turbo"})
                if hasattr(self, "integrate_puzzle"):
                    self.integrate_puzzle.setEnabled(False); self.integrate_puzzle.setChecked(False)
                return
            if engine == 1:                        # FLUX.2
                pool = flux2
                empty = tr("f1.flux2.none")
            else:                                  # Kontext (FLUX.1)
                pool = [m for m in self._models
                        if "kontext" in m.lower() and "fill" not in m.lower()
                        and not _is_flux2(m)
                        and (allow or not self._mcat.is_nsfw(m))]
                empty = tr("f1.kontext.none")
            ic.clear()
            ic.addItems(pool) if pool else ic.addItem(empty)
            if pool:
                self._color_name_combo(ic)
            if cur in pool:
                ic.setCurrentText(cur)
        # puzzle_lora è una LoRA FLUX.1 → disabilita in modalità FLUX.2
        if hasattr(self, "integrate_puzzle") and hasattr(self, "integrate_engine"):
            is_flux2_mode = (self.integrate_engine.currentIndex() == 1)
            self.integrate_puzzle.setEnabled(not is_flux2_mode)
            if is_flux2_mode:
                self.integrate_puzzle.setChecked(False)
                self.integrate_puzzle.setToolTip(tr("f1.integrate.puzzle.tt.flux2"))
            else:
                self.integrate_puzzle.setToolTip(tr("f1.integrate.puzzle.tt.on"))

    # ---------- inpaint / outpaint ----------
    def _photo_field(self, placeholder: str) -> dict:
        """Campo 'foto da modificare': path + Sfoglia + X.

        Regola (richiesta utente 13/07): se il campo è VUOTO, la foto da modificare è quella già
        caricata NEL CANVAS. Prima la GUI chiedeva una foto ma il campo per darla non esisteva nella
        sezione: l'unico modo era caricarla nel canvas, senza che nulla lo dicesse.
        Ritorna {"row": widget da mettere nel form, "edit": QLineEdit con il path}.
        """
        row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(4)
        edit = QLineEdit(); edit.setPlaceholderText(placeholder)
        edit.setToolTip(tr("f1.photo.tt",
                           "Foto da modificare. Se lasci vuoto, viene usata l'immagine del canvas."))
        b_open = QPushButton("📂"); b_open.setFixedWidth(30)
        b_open.setToolTip(tr("f1.photo.browse.tt", "Scegli una foto dal disco"))
        b_clr = QPushButton("✕"); b_clr.setFixedWidth(26)
        b_clr.setToolTip(tr("f1.photo.clear.tt", "Svuota → usa la foto del canvas"))

        def _browse():
            fn, _ = QFileDialog.getOpenFileName(self, tr("f1.photo.browse.tt", "Scegli una foto"), "",
                                                tr("canvas.image_filter", "Immagini (*.png *.jpg *.jpeg *.webp)"))
            if fn:
                edit.setText(fn)

        b_open.clicked.connect(_browse)
        b_clr.clicked.connect(lambda: edit.clear())
        h.addWidget(edit, 1); h.addWidget(b_open); h.addWidget(b_clr)
        return {"row": row, "edit": edit}

    def _inpaint_box(self) -> QWidget:
        """Tab Inpaint/Outpaint diviso in DUE sezioni indipendenti, ognuna con i suoi
        parametri (anche duplicati: negativo/steps/seed) e il proprio bottone di generazione:

        1. «Inpaint / Outpaint» — riempi area mascherata o estendi oltre i bordi.
        2. «Integra» — fondi immagini extra incollate sul canvas (era sul canvas, ora qui).
        """
        wrap = QWidget(); wv = QVBoxLayout(wrap); wv.setContentsMargins(0, 0, 0, 0); wv.setSpacing(10)

        # Styling: bordo + titolo evidente per evitare che le due sezioni si "fondano"
        # visivamente. Applicato sul wrapper così è ereditato da entrambi i QGroupBox.
        wrap.setStyleSheet(
            "QGroupBox {"
            "  border: 1px solid #555;"
            "  border-radius: 6px;"
            "  margin-top: 14px;"
            "  padding-top: 10px;"
            "  font-weight: bold;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  subcontrol-position: top left;"
            "  left: 10px;"
            "  padding: 0 6px;"
            "  color: #c8d7e3;"
            "}"
        )

        # ============ SEZIONE 1: INPAINT / OUTPAINT ============
        ipo_box = QGroupBox(tr("f1.inpaint.group"))
        f = QFormLayout(ipo_box)
        # Inpaint/outpaint lavorano SEMPRE sulla foto del canvas (è lì che disegni la maschera): nessun
        # campo file qui. Il campo esiste solo in "Integra", dove la base può essere un'altra foto.
        self.inpaint_prompt = QPlainTextEdit(); self.inpaint_prompt.setMaximumHeight(36)
        self.inpaint_prompt.setPlaceholderText(tr("f1.inpaint.prompt.ph"))
        self.inpaint_prompt.setToolTip(tr("f1.inpaint.prompt.tt", "Descrivi cosa disegnare nell'area mascherata (inpaint)."))
        f.addRow(tr("f1.inpaint.prompt_lbl"), self.inpaint_prompt)
        self.edit_enhance = QCheckBox(tr("f1.edit.enhance", "✨ Arricchisci prompt (AI)"))
        self.edit_enhance.setToolTip(tr("f1.edit.enhance.tt",
            "Prima di elaborare, un LLM locale arricchisce il prompt di modifica (inpaint/outpaint/integra). "
            "Invisibile, opzionale."))
        f.addRow("", self.edit_enhance)
        self.outpaint_prompt = QPlainTextEdit(); self.outpaint_prompt.setMaximumHeight(36)
        self.outpaint_prompt.setPlaceholderText(tr("f1.outpaint.prompt.ph"))
        self.outpaint_prompt.setToolTip(tr("f1.outpaint.prompt.tt", "Descrivi cosa generare nell'area estesa oltre i bordi (outpaint)."))
        f.addRow(tr("f1.outpaint.prompt_lbl"), self.outpaint_prompt)
        self.inp_neg = QPlainTextEdit(DEFAULT_NEG); self.inp_neg.setMaximumHeight(36)
        self.inp_neg.setPlaceholderText(tr("f1.inpaint.neg.ph"))
        self.inp_neg.setToolTip(tr("f1.neg.tt", "Elementi da evitare nell'area rigenerata."))
        f.addRow(tr("f1.neg_lbl"), self.inp_neg)
        srow = QHBoxLayout()
        self.strength = QSlider(Qt.Horizontal); self.strength.setRange(0, 100); self.strength.setValue(75)
        self.strength.setToolTip(tr("f1.inpaint.strength.tt", "Quanto l'area mascherata viene rigenerata: 1.00 = totale, valori bassi preservano l'originale."))
        self.strength_lbl = QLabel("0.75")
        self.strength.valueChanged.connect(lambda v: self.strength_lbl.setText(f"{v/100:.2f}"))
        srow.addWidget(self.strength); srow.addWidget(self.strength_lbl)
        sw = QWidget(); sw.setLayout(srow); f.addRow(tr("f1.inpaint.strength_lbl"), sw)
        # GUIDANCE (CFG) esplicita — distinta dalla "Forza": quanto la modifica segue il PROMPT (alta=decisa).
        self.inp_guidance = QDoubleSpinBox(); self.inp_guidance.setRange(1.0, 30.0); self.inp_guidance.setSingleStep(0.5)
        self.inp_guidance.setValue(30.0)
        self.inp_guidance.setToolTip(tr("f1.inpaint.guidance.tt",
            "Guidance (CFG): quanto la modifica SEGUE il prompt. Alta (≈30) = modifica decisa e aderente; "
            "bassa = più morbida/creativa. Diverso dalla «Forza mantenimento» (quanto preserva l'originale)."))
        f.addRow(tr("f1.inpaint.guidance_lbl", "Guidance (aderenza prompt)"), self.inp_guidance)
        # feather della ricomposizione fuori-maschera: 0 = automatico (proporzionale), 2-16 = manuale
        self.inp_feather = QSpinBox(); self.inp_feather.setRange(0, 16); self.inp_feather.setValue(0)
        self.inp_feather.setSpecialValueText(tr("f1.inpaint.feather.auto", "Auto"))
        self.inp_feather.setToolTip(tr("f1.inpaint.feather.tt",
            "Sfumatura del bordo tra area rigenerata e originale (px): Auto = proporzionale alla foto; "
            "valori alti = transizione più morbida ma anello di modifica più largo."))
        f.addRow(tr("f1.inpaint.feather_lbl", "Sfumatura bordo (px)"), self.inp_feather)
        self.inp_model = QComboBox(); self.inp_model.addItems(list(INPAINT_MODELS.keys()))
        self.inp_model.setToolTip(tr("f1.inpaint.model.tt", "Modello usato per inpaint/outpaint (FLUX Fill o checkpoint compatibile)."))
        self._color_curated_combo(self.inp_model, INPAINT_MODELS, key="unet")   # colori VRAM (verde/giallo/rosso)
        self.inp_model.currentTextChanged.connect(self._reload_inpaint_loras)
        self.inp_model.currentTextChanged.connect(self._update_inp_star)
        self.inp_model.currentTextChanged.connect(self._auto_inp_guidance)
        imrow = QHBoxLayout(); imrow.setContentsMargins(0, 0, 0, 0)
        imrow.addWidget(self.inp_model, 1)
        self.inp_star = QToolButton(); self.inp_star.setText("☆")
        self.inp_star.setStyleSheet("QToolButton{border:none;font-size:16px;color:#777;}")
        self.inp_star.setToolTip(tr("f1.inpaint.star.tt"))
        self.inp_star.clicked.connect(self._toggle_inp_fav)
        imrow.addWidget(self.inp_star)
        imw = QWidget(); imw.setLayout(imrow); f.addRow(tr("f1.model_lbl"), imw)
        self.inp_lora = QComboBox()
        self.inp_lora.setToolTip(tr("f1.inpaint.lora.tt", "LoRA opzionale applicata durante inpaint/outpaint per stile o soggetto."))
        f.addRow(tr("f1.inpaint.lora_lbl"), self.inp_lora)
        self.inp_steps = QSpinBox(); self.inp_steps.setRange(1, 80); self.inp_steps.setValue(20)
        self.inp_steps.setToolTip(tr("f1.steps.tt", "Numero di passi di diffusione: più passi = più dettaglio e tempi più lunghi."))
        f.addRow(tr("f1.steps_lbl"), self.inp_steps)
        seedrow = QHBoxLayout()
        self.inp_seed = QSpinBox(); self.inp_seed.setRange(0, 2_147_483_647); self.inp_seed.setValue(0)
        self.inp_seed.setToolTip(tr("f1.seed.tt", "Seme del generatore casuale: fissarlo rende la generazione riproducibile."))
        self.inp_seed_rand = QCheckBox(tr("f1.seed_rand")); self.inp_seed_rand.setChecked(True)
        self.inp_seed_rand.setToolTip(tr("f1.seed_rand.tt", "Se attivo usa un seme casuale a ogni generazione; disattiva per fissarlo."))
        self.inp_seed.setEnabled(False)
        self.inp_seed_rand.toggled.connect(lambda on: self.inp_seed.setEnabled(not on))
        seedrow.addWidget(self.inp_seed, 1); seedrow.addWidget(self.inp_seed_rand)
        sw2 = QWidget(); sw2.setLayout(seedrow); f.addRow(tr("f1.seed_lbl"), sw2)
        # bottoni Genera per inpaint/outpaint
        brow = QHBoxLayout()
        b_in = QPushButton(tr("f1.inpaint.btn")); b_in.clicked.connect(lambda: self.canvas.request_inpaint.emit())
        b_in.setToolTip(tr("f1.inpaint.run.tt", "Riempi l'area mascherata mantenendo intatto il resto dell'immagine."))
        b_out = QPushButton(tr("f1.outpaint.btn")); b_out.clicked.connect(lambda: self.canvas.request_outpaint.emit())
        b_out.setToolTip(tr("f1.outpaint.run.tt", "Estendi l'immagine oltre i bordi generando il nuovo contenuto."))
        brow.addWidget(b_in); brow.addWidget(b_out)
        bw = QWidget(); bw.setLayout(brow); f.addRow(bw)
        wv.addWidget(ipo_box)
        # separatore visivo orizzontale fra le due sezioni
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet("color: #444; margin: 6px 0;")
        wv.addWidget(sep)

        # ============ SEZIONE 2: INTEGRA ============
        int_box = QGroupBox(tr("f1.integrate.group"))
        g = QFormLayout(int_box)
        # Foto di BASE su cui fondere i layer. Vuoto = la foto del canvas (com'era, ma ora è detto).
        self.int_photo = self._photo_field(tr("f1.photo.ph", "(vuoto = la foto nel canvas)"))
        g.addRow(tr("f1.photo_base_lbl", "Foto di base"), self.int_photo["row"])
        self.integrate_prompt = QPlainTextEdit(); self.integrate_prompt.setMaximumHeight(36)
        self.integrate_prompt.setPlaceholderText(tr("f1.integrate.prompt.ph"))
        self.integrate_prompt.setToolTip(tr("f1.integrate.prompt.tt", "Istruzione su come fondere le immagini extra nella scena."))
        g.addRow(tr("f1.integrate.prompt_lbl"), self.integrate_prompt)
        self.integrate_engine = QComboBox()
        self.integrate_engine.addItems([tr("f1.engine.kontext"), tr("f1.engine.flux2"),
                                        tr("f1.engine.krea2edit", "Krea 2 Edit")])
        self.integrate_engine.setToolTip(tr("f1.integrate.engine.tt"))
        # Cambio engine → ripopola il combo modelli (Kontext FLUX.1 / FLUX.2 / Krea 2 Turbo-Raw) +
        # abilitazione del puzzle_lora (LoRA FLUX.1) + visibilità dello slider grounding (solo Krea 2 Edit).
        self.integrate_engine.currentIndexChanged.connect(self._on_integrate_engine_changed)
        g.addRow(tr("f1.integrate.engine_lbl"), self.integrate_engine)
        # Slider GROUNDING (edit↔fedeltà identità) — solo per Krea 2 Edit. 512=edit forte … 1536=preserva
        # identità (def 768). Manopola caratteristica del port comfyui-krea2edit.
        grrow = QHBoxLayout()
        self.edit_grounding = QSlider(Qt.Horizontal); self.edit_grounding.setRange(512, 1536)
        self.edit_grounding.setSingleStep(64); self.edit_grounding.setValue(768)
        self.edit_grounding_lbl = QLabel("768")
        self.edit_grounding.valueChanged.connect(lambda v: self.edit_grounding_lbl.setText(str(int(v))))
        self.edit_grounding.setToolTip(tr("f1.edit.grounding.tt",
                                          "Bassa = edit più forte; alta = preserva meglio l'identità (Krea 2 Edit)."))
        grrow.addWidget(self.edit_grounding, 1); grrow.addWidget(self.edit_grounding_lbl)
        self._edit_grounding_row = self._wrapw(grrow)
        self._edit_grounding_label = QLabel(tr("f1.edit.grounding_lbl", "Fedeltà / Edit"))
        g.addRow(self._edit_grounding_label, self._edit_grounding_row)
        self._edit_grounding_label.setVisible(False); self._edit_grounding_row.setVisible(False)
        self.integrate_puzzle = QCheckBox(tr("f1.integrate.puzzle"))
        self.integrate_puzzle.setChecked(False)   # OFF di default: introduce un artefatto a cornice
        self.integrate_puzzle.setToolTip(tr("f1.integrate.puzzle.tt"))
        g.addRow("", self.integrate_puzzle)
        self.integrate_model = QComboBox()
        self.integrate_model.setToolTip(tr("f1.integrate.model.tt", "Modello usato per l'integrazione (Kontext FLUX.1 o FLUX.2)."))
        # Label dinamica: aggiornata da _on_integrate_engine_changed.
        self._integrate_model_label = QLabel(tr("f1.model_lbl"))
        g.addRow(self._integrate_model_label, self.integrate_model)
        # Campi duplicati (negativo/steps/seed) dedicati all'integrazione: il pipeline
        # generate_integrate userà questi al posto di quelli inpaint.
        self.int_neg = QPlainTextEdit(DEFAULT_NEG); self.int_neg.setMaximumHeight(36)
        self.int_neg.setPlaceholderText(tr("f1.inpaint.neg.ph"))
        self.int_neg.setToolTip(tr("f1.neg.tt", "Elementi da evitare nell'area rigenerata."))
        g.addRow(tr("f1.neg_lbl"), self.int_neg)
        self.int_steps = QSpinBox(); self.int_steps.setRange(1, 80); self.int_steps.setValue(20)
        self.int_steps.setToolTip(tr("f1.steps.tt", "Numero di passi di diffusione: più passi = più dettaglio e tempi più lunghi."))
        g.addRow(tr("f1.steps_lbl"), self.int_steps)
        # Guidance (FluxGuidance) regolabile: Kontext FLUX.1 ~2.5; FLUX.2 ~4. Più alta = più aderenza
        # al prompt/riferimento (e più "forzatura"); più bassa = integrazione più morbida.
        grow = QHBoxLayout()
        self.int_guidance = QSlider(Qt.Horizontal); self.int_guidance.setRange(10, 100)  # /10 → 1.0..10.0
        self.int_guidance.setValue(25)
        self.int_guidance_lbl = QLabel("2.5")
        self.int_guidance.valueChanged.connect(lambda v: self.int_guidance_lbl.setText(f"{v/10:.1f}"))
        self.int_guidance.setToolTip(tr("f1.integrate.guidance.tt"))
        grow.addWidget(self.int_guidance, 1); grow.addWidget(self.int_guidance_lbl)
        g.addRow(tr("f1.integrate.guidance_lbl"), self._wrapw(grow))
        intseedrow = QHBoxLayout()
        self.int_seed = QSpinBox(); self.int_seed.setRange(0, 2_147_483_647); self.int_seed.setValue(0)
        self.int_seed.setToolTip(tr("f1.seed.tt", "Seme del generatore casuale: fissarlo rende la generazione riproducibile."))
        self.int_seed_rand = QCheckBox(tr("f1.seed_rand")); self.int_seed_rand.setChecked(True)
        self.int_seed_rand.setToolTip(tr("f1.seed_rand.tt", "Se attivo usa un seme casuale a ogni generazione; disattiva per fissarlo."))
        self.int_seed.setEnabled(False)
        self.int_seed_rand.toggled.connect(lambda on: self.int_seed.setEnabled(not on))
        intseedrow.addWidget(self.int_seed, 1); intseedrow.addWidget(self.int_seed_rand)
        sw3 = QWidget(); sw3.setLayout(intseedrow); g.addRow(tr("f1.seed_lbl"), sw3)
        # bottone Integra spostato qui dal canvas/workspace
        b_int = QPushButton(tr("f1.integrate.btn"))
        b_int.setToolTip(tr("f1.integrate.run.tt", "Fondi le immagini incollate sul canvas nella scena in modo coerente."))
        b_int.clicked.connect(lambda: self.request_integrate.emit())
        g.addRow(b_int)
        wv.addWidget(int_box)

        # ============ SEZIONE 3: COMPONI PIÙ FOTO (senza incollare → niente occlusione) ============
        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine); sep2.setFrameShadow(QFrame.Sunken)
        sep2.setStyleSheet("color: #444; margin: 6px 0;")
        wv.addWidget(sep2)
        comp_box = QGroupBox(tr("f1.compose.group"))
        cg = QVBoxLayout(comp_box)
        info = QLabel(tr("f1.compose.info")); info.setWordWrap(True)
        info.setStyleSheet("color:#9bd; font-size:11px;")
        cg.addWidget(info)
        # LISTA VUOTA = usa il CANVAS: la base è la foto del canvas e le foto da comporre sono i suoi
        # LAYER, ciascuno con la SUA maschera e con posizione/scala/rotazione che gli hai dato.
        # (Con la lista piena, invece, le foto vengono affiancate e la collocazione la sceglie il
        # modello: nessun controllo spaziale — è un limite dei modelli, non della GUI.)
        hint = QLabel(tr("f1.compose.canvas_hint",
                         "💡 Lascia la lista VUOTA per usare il canvas: base = foto del canvas, "
                         "elementi = i layer, con le loro maschere, posizioni, scale e rotazioni."))
        hint.setWordWrap(True); hint.setStyleSheet("color:#8c8; font-size:11px;")
        cg.addWidget(hint)
        self.compose_list = QListWidget(); self.compose_list.setMaximumHeight(110)
        self.compose_list.setToolTip(tr("f1.compose.list.tt"))
        cg.addWidget(self.compose_list)
        crow = QHBoxLayout()
        b_add = QPushButton(tr("f1.compose.add")); b_add.clicked.connect(self._compose_add)
        b_add.setToolTip(tr("f1.compose.add.tt", "Aggiungi una o più foto da combinare senza incollarle sul canvas."))
        b_del = QPushButton(tr("f1.compose.remove")); b_del.clicked.connect(self._compose_remove)
        b_del.setToolTip(tr("f1.compose.remove.tt", "Rimuovi la foto selezionata dalla lista di composizione."))
        b_msk = QPushButton(tr("f1.compose.mask")); b_msk.clicked.connect(self._compose_mask)
        b_msk.setToolTip(tr("f1.compose.mask.tt"))
        crow.addWidget(b_add); crow.addWidget(b_del); crow.addWidget(b_msk)
        cg.addLayout(crow)
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel(tr("f1.compose.method_lbl")))
        self.compose_method = QComboBox()
        self.compose_method.addItems([tr("f1.compose.method.multiref"), tr("f1.compose.method.stitch")])
        self.compose_method.setToolTip(tr("f1.compose.method.tt"))
        mrow.addWidget(self.compose_method, 1)
        cg.addLayout(mrow)
        self.compose_prompt = QPlainTextEdit(); self.compose_prompt.setMaximumHeight(48)
        self.compose_prompt.setPlaceholderText(tr("f1.compose.prompt.ph"))
        self.compose_prompt.setToolTip(tr("f1.compose.prompt.tt", "Descrivi la scena finale che combina i soggetti delle foto elencate."))
        cg.addWidget(self.compose_prompt)
        b_comp = QPushButton(tr("f1.compose.btn"))
        b_comp.setToolTip(tr("f1.compose.run.tt", "Componi le foto elencate in un'unica immagine coerente."))
        b_comp.clicked.connect(lambda: self.request_compose.emit())
        cg.addWidget(b_comp)
        wv.addWidget(comp_box)
        self._compose_masks: dict[str, str] = {}

        return wrap

    # ---------- cambia colore (Kontext prompt-based, stile MagicQuill) ----------
    def _color_box(self) -> QGroupBox:
        box = QGroupBox(tr("f1.color.group"))
        f = QFormLayout(box)
        crow = QHBoxLayout()
        self.color_target = QLineEdit(); self.color_target.setPlaceholderText(tr("f1.color.target.ph"))
        self.color_target.setToolTip(tr("f1.color.target.tt", "Colore desiderato (nome o codice esadecimale) da applicare al soggetto."))
        self.color_swatch = QLabel(); self.color_swatch.setFixedSize(26, 26)
        self.color_swatch.setStyleSheet("border:1px solid #555; border-radius:5px;")
        cbtn = QPushButton(tr("f1.color.pick.btn")); cbtn.clicked.connect(self._pick_color)
        cbtn.setToolTip(tr("f1.color.pick.tt", "Apri il selettore colore per scegliere la tinta da applicare."))
        crow.addWidget(self.color_target, 1); crow.addWidget(self.color_swatch); crow.addWidget(cbtn)
        f.addRow(tr("f1.color.color_lbl"), self._wrapw(crow))
        self.color_subject = QLineEdit()
        self.color_subject.setPlaceholderText(tr("f1.color.subject.ph"))
        self.color_subject.setToolTip(tr("f1.color.subject.tt", "Soggetto o area di cui cambiare colore (es. 'la maglietta', 'l'auto')."))
        f.addRow(tr("f1.color.subject_lbl"), self.color_subject)
        self.color_engine = QComboBox()
        self.color_engine.addItems([tr("f1.engine.kontext"), tr("f1.engine.flux2"),
                                    tr("f1.engine.krea2edit", "Krea 2 Edit")])
        self.color_engine.setToolTip(tr("f1.color.engine.tt", "Motore per il cambio colore: Kontext (FLUX.1), FLUX.2 o Krea 2 Edit."))
        f.addRow(tr("f1.engine_lbl"), self.color_engine)
        self.color_steps = QSpinBox(); self.color_steps.setRange(1, 80); self.color_steps.setValue(20)
        self.color_steps.setToolTip(tr("f1.steps.tt", "Numero di passi di diffusione: più passi = più dettaglio e tempi più lunghi."))
        f.addRow(tr("f1.steps_lbl"), self.color_steps)
        self.color_btn = QPushButton(tr("f1.color.btn"))
        self.color_btn.setToolTip(tr("f1.color.run.tt", "Applica il nuovo colore al soggetto indicato lasciando invariato il resto."))
        self.color_btn.clicked.connect(lambda: self.request_recolor.emit())
        f.addRow(self.color_btn)
        note = QLabel(tr("f1.color.note"))
        note.setWordWrap(True); note.setStyleSheet("color:#888;")
        f.addRow(note)
        return box

    def _wrapw(self, layout) -> QWidget:
        w = QWidget(); w.setLayout(layout); return w

    def _pick_color(self):
        from PySide6.QtWidgets import QColorDialog
        c = QColorDialog.getColor()
        if c.isValid():
            self.color_target.setText(c.name())
            self.color_swatch.setStyleSheet(f"background:{c.name()}; border:1px solid #555; border-radius:5px;")

    @staticmethod
    def _engine_val(idx: int) -> str:
        """Indice combo motore → valore backend. 0=Kontext(FLUX.1), 1=FLUX.2, 2=Krea 2 Edit."""
        return {0: "kontext", 1: "flux2", 2: "krea2"}.get(int(idx), "kontext")

    def edit_grounding_px(self) -> int:
        return int(self.edit_grounding.value()) if hasattr(self, "edit_grounding") else 768

    def _integrate_krea2_variant(self) -> str:
        """Variante Krea 2 (turbo/raw) scelta nel combo Integra quando engine=Krea 2 Edit; altrimenti turbo."""
        if getattr(self, "integrate_engine", None) is None or self.integrate_engine.currentIndex() != 2:
            return "turbo"
        d = self.integrate_model.currentData() if hasattr(self, "integrate_model") else None
        return (d or {}).get("krea2_variant", "turbo") if isinstance(d, dict) else "turbo"

    def get_color_config(self) -> dict:
        return {
            "target": self.color_target.text().strip(),
            "subject": self.color_subject.text().strip(),
            "engine": self._engine_val(self.color_engine.currentIndex()),
            "grounding": self.edit_grounding_px(),
            "steps": self.color_steps.value(),
        }

    def _reload_inpaint_loras(self, *_):
        label = self.inp_model.currentText().split("  ⛔")[0]
        # Tutte le voci dinamiche FLUX (checkpoint o UNET) usano lo stesso pool LoRA Flux.
        if label in self._dynamic_checkpoints or label in getattr(self, "_dynamic_unets", {}):
            fill_lora_combo(self.inp_lora, "Flux", allow_nsfw=self._allow_nsfw())
            return
        spec = INPAINT_MODELS.get(label, {})
        fill_lora_combo(self.inp_lora, spec.get("arch"), allow_nsfw=self._allow_nsfw())
        # allinea gli steps al default consigliato del modello (l'utente può ri-modificarli)
        if spec.get("steps"):
            self.inp_steps.setValue(int(spec["steps"]))

    # ---------- selezione oggetto (GroundingDINO / CLIPSeg + SAM) ----------
    def _segment_box(self) -> QGroupBox:
        box = QGroupBox(tr("f1.seg.group"))
        v = QVBoxLayout(box)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel(tr("f1.seg.text_lbl")))
        self.seg_text = QLineEdit(); self.seg_text.setPlaceholderText(tr("f1.seg.text.ph"))
        self.seg_text.setToolTip(tr("f1.seg.text.tt", "Testo che descrive l'oggetto da selezionare automaticamente (es. 'il cane')."))
        self.seg_text.returnPressed.connect(lambda: self.request_segment_text.emit())
        r1.addWidget(self.seg_text, 1)
        seg_btn = QPushButton(tr("f1.seg.btn")); seg_btn.clicked.connect(lambda: self.request_segment_text.emit())
        seg_btn.setToolTip(tr("f1.seg.run.tt", "Individua e seleziona l'oggetto descritto usando il detector + SAM."))
        r1.addWidget(seg_btn)
        v.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel(tr("f1.seg.detector_lbl")))
        self.seg_detector = QComboBox()
        # Il «Detector» riguarda SOLO la selezione DA TESTO (pulsante 🎯). La selezione col
        # PUNTATORE (click/box sul canvas) usa sempre SAM vanilla in automatico e non passa
        # da qui. Niente «SAM v1» qui: da testo SAM da solo non funziona (serve un detector) e
        # in passato finiva su CLIPSeg (non installato) → errore.
        self.seg_detector.addItem(tr("f1.seg.detector.dino_sam"), userData="grounding_dino")
        self.seg_detector.addItem(tr("f1.seg.detector.dino"), userData="grounding_dino_only")
        self.seg_detector.setToolTip(tr("f1.seg.detector.tt", "Detector per la selezione da testo: GroundingDINO con o senza raffinamento SAM."))
        self.seg_detector.setCurrentIndex(0)
        r2.addWidget(self.seg_detector, 1)
        r2.addWidget(QLabel(tr("f1.seg.threshold_lbl")))
        self.seg_threshold = QDoubleSpinBox()
        self.seg_threshold.setRange(0.10, 0.95); self.seg_threshold.setSingleStep(0.05)
        # 0.50 per GroundingDINO: 0.30-0.40 selezionava spesso l'intera immagine
        # su prompt generici (es. "viso"). Valori alti (0.50-0.70) sono più precisi.
        self.seg_threshold.setValue(0.50)
        self.seg_threshold.setToolTip(tr("f1.seg.threshold.tt"))
        r2.addWidget(self.seg_threshold)
        v.addLayout(r2)
        self.seg_detector.currentIndexChanged.connect(self._on_seg_detector_changed)
        r3 = QHBoxLayout()
        r3.addWidget(QLabel(tr("f1.seg.sam_lbl")))
        self.seg_sam = QComboBox(); self.seg_sam.setMinimumWidth(200)
        self.seg_sam.setToolTip(tr("f1.seg.sam.tt"))
        r3.addWidget(self.seg_sam, 1)
        v.addLayout(r3)
        hint = QLabel(tr("f1.seg.hint"))
        hint.setTextFormat(Qt.RichText)
        hint.setStyleSheet("color:#888;"); hint.setWordWrap(True); v.addWidget(hint)
        return box

    def _on_integrate_engine_changed(self, _idx: int):
        """Cambio motore Integra → ricarica modelli compatibili + label dinamica + slider grounding."""
        if _idx == 2:
            engine_label = tr("f1.integrate.model.krea2", "Modello (Krea 2)")
        elif _idx == 1:
            engine_label = tr("f1.integrate.model.flux2")
        else:
            engine_label = tr("f1.integrate.model.kontext")
        if hasattr(self, "_integrate_model_label"):
            self._integrate_model_label.setText(engine_label)
        # slider grounding visibile SOLO per Krea 2 Edit
        if hasattr(self, "_edit_grounding_row"):
            self._edit_grounding_row.setVisible(_idx == 2)
            self._edit_grounding_label.setVisible(_idx == 2)
        self._reload_flux_combos()

    def _on_seg_detector_changed(self, _idx: int):
        # Solo detector da testo (GroundingDINO): la soglia è sempre rilevante. Default 0.50.
        self.seg_threshold.setEnabled(True)

    def set_sam_models(self, models: list[str]):
        self.seg_sam.clear()
        if models:
            # Questo menu serve SOLO alla selezione a click/box (puntatore), che usa il
            # SAMLoader di Impact-Pack: carica solo SAM *vanilla* (sam_vit_b/l/h). Un
            # SAM-HQ qui darebbe mismatch di state_dict → vanilla in cima e selezionato.
            # (La selezione «da testo»/GroundingDINO usa SAM-HQ in automatico, non questo menu.)
            vanilla = [m for m in models if "hq" not in m.lower()]
            hq = [m for m in models if "hq" in m.lower()]
            self.seg_sam.addItems(vanilla + hq)
            if not vanilla:
                # solo SAM-HQ disponibili: il puntatore non può funzionare finché non si
                # aggiunge un SAM vanilla in models/sams.
                self.seg_sam.insertItem(0, tr("f1.seg.sam.need_vanilla"))
                self.seg_sam.setCurrentIndex(0)
        else:
            self.seg_sam.addItem(tr("f1.seg.sam.none"))

    def get_segment_config(self) -> dict:
        sam = self.seg_sam.currentText() if self.seg_sam.count() else ""
        if sam.startswith("(") or sam.startswith("⚠"):
            sam = ""
        detector = self.seg_detector.currentData() if hasattr(self, "seg_detector") else "grounding_dino"
        thr = float(self.seg_threshold.value()) if hasattr(self, "seg_threshold") else 0.50
        return {"text": self.seg_text.text().strip(),
                "detector": detector,
                "threshold": thr,
                "sam_model": sam or None,
                "clipseg_threshold": thr}  # back-compat con vecchio main_window

    # ---------- aumenta definizione (upscale AI) ----------
    def _upscale_box(self) -> QGroupBox:
        box = QGroupBox(tr("f1.upscale.group"))
        f = QFormLayout(box)
        self.up_model = QComboBox(); self.up_model.setMinimumWidth(220)
        self.up_model.setToolTip(tr("f1.upscale.model.tt", "Modello ESRGAN per l'aumento di risoluzione dell'immagine."))
        f.addRow(tr("f1.model_lbl"), self.up_model)
        srow = QHBoxLayout()
        self.up_scale = QComboBox(); self.up_scale.addItems(["2x", "3x", "4x"]); self.up_scale.setCurrentText("4x")
        self.up_scale.setToolTip(tr("f1.upscale.scale.tt", "Fattore di ingrandimento della risoluzione (2×, 3× o 4×)."))
        srow.addWidget(self.up_scale)
        srow.addWidget(QLabel(tr("f1.upscale.mode_lbl")))
        self.up_mode = QComboBox(); self.up_mode.addItems([tr("f1.upscale.mode.fast"), tr("f1.upscale.mode.refine")])
        self.up_mode.setToolTip(tr("f1.upscale.mode.tt", "Modalità: 'veloce' solo upscaler; 'rifinitura' aggiunge un passaggio FLUX per più dettaglio."))
        self.up_mode.currentIndexChanged.connect(self._on_up_mode)
        srow.addWidget(self.up_mode, 1)
        sw = QWidget(); sw.setLayout(srow); f.addRow(tr("f1.upscale.factor_lbl"), sw)
        # rifinitura AI (FLUX img2img a denoise basso)
        self.up_refine_model = QComboBox()
        self.up_refine_model.setToolTip(tr("f1.upscale.refine_model.tt", "Modello FLUX.2 usato per la rifinitura AI dopo l'upscale."))
        f.addRow(tr("f1.upscale.refine_model_lbl"), self.up_refine_model)
        rrow = QHBoxLayout()
        self.up_refine_str = QSlider(Qt.Horizontal); self.up_refine_str.setRange(5, 70); self.up_refine_str.setValue(30)
        self.up_refine_str.setToolTip(tr("f1.upscale.refine_str.tt", "Intensità della rifinitura AI: alta ridisegna di più, bassa preserva l'originale."))
        self.up_refine_str_lbl = QLabel("0.30")
        self.up_refine_str.valueChanged.connect(lambda v: self.up_refine_str_lbl.setText(f"{v/100:.2f}"))
        rrow.addWidget(self.up_refine_str); rrow.addWidget(self.up_refine_str_lbl)
        self.up_refine_steps = QSpinBox(); self.up_refine_steps.setRange(4, 40); self.up_refine_steps.setValue(12)
        self.up_refine_steps.setToolTip(tr("f1.steps.tt", "Numero di passi di diffusione: più passi = più dettaglio e tempi più lunghi."))
        rrow.addWidget(QLabel(tr("f1.steps_lbl"))); rrow.addWidget(self.up_refine_steps)
        rw = QWidget(); rw.setLayout(rrow); f.addRow(tr("f1.upscale.refine_strength_lbl"), rw)
        up_btn = QPushButton(tr("f1.upscale.btn"))
        up_btn.setToolTip(tr("f1.upscale.run.tt", "Avvia l'aumento di definizione dell'immagine corrente."))
        up_btn.clicked.connect(lambda: self.request_upscale.emit())
        f.addRow(up_btn)
        hint = QLabel(tr("f1.upscale.hint"))
        hint.setStyleSheet("color:#888;"); hint.setWordWrap(True); f.addRow(hint)

        from PySide6.QtWidgets import QFrame as _QFrame
        sep = _QFrame(); sep.setFrameShape(_QFrame.HLine)
        sep.setStyleSheet("color:#333; margin-top:6px;")
        f.addRow(sep)
        restore_btn = QPushButton(tr("f1.restore.btn"))
        restore_btn.setToolTip(tr("f1.restore.tt"))
        restore_btn.clicked.connect(lambda: self.request_restore_photo.emit())
        f.addRow(restore_btn)
        self._on_up_mode()
        return box

    def _on_up_mode(self, *_):
        refine = self.up_mode.currentIndex() == 1
        for w in (self.up_refine_model, self.up_refine_str, self.up_refine_str_lbl, self.up_refine_steps):
            w.setEnabled(refine)

    def set_upscale_models(self, models: list[str]):
        self.up_model.clear()
        if not models:
            self.up_model.addItem(tr("f1.upscale.none")); return
        pref = [m for m in _UPSCALE_PREFERRED if m in models]      # preferiti (più potenti) davanti
        rest = [m for m in models if m not in pref]
        for m in pref + rest:
            self.up_model.addItem(m)
        try:                                                       # colori VRAM (header-aware)
            from ui.widgets.precision_combo import color_existing
            from pathlib import Path as _P
            _r = _P(r"E:\AI\Models\Upscale")
            color_existing(self.up_model, name_to_path=lambda n: str(_r / n) if not _P(n).is_file() else n)
        except Exception:                                          # noqa: BLE001
            pass

    def get_upscale_config(self) -> dict:
        m = self.up_model.currentText() if self.up_model.count() else ""
        if m.startswith("("):
            m = ""
        rm = self.up_refine_model.currentText() if self.up_refine_model.count() else ""
        if rm.startswith("("):
            rm = ""
        return {
            "model": m or None, "scale": float(self.up_scale.currentText().rstrip("x")),
            "refine": self.up_mode.currentIndex() == 1,
            "refine_model": rm or None,
            "refine_denoise": self.up_refine_str.value() / 100.0,
            "refine_steps": self.up_refine_steps.value(),
        }

    # ---------- inventario modelli (da ComfyUI, passato da main_window) ----------
    def set_model_inventory(self, diffusion_models: list[str],
                            checkpoint_models: list[str] | None = None):
        self._models = list(diffusion_models)
        self._gen_checkpoint_models = list(checkpoint_models or [])
        self._mcat.reload_if_changed()
        if checkpoint_models is not None:
            self._refresh_inpaint_checkpoints(checkpoint_models)
        self._reload_gen()
        self._reload_flux_combos()        # modelli FLUX per rifinitura upscale e integrazione

    def set_gen_inventory(self, text_encoders: list[str] | None = None,
                          vaes: list[str] | None = None,
                          clip_types: list[str] | None = None):
        """Popola i dropdown Text encoder / CLIP type / VAE della sezione «Genera immagine»
        con i valori reali esposti da ComfyUI (CLIPLoader/VAELoader). Mantiene la selezione
        corrente se ancora valida; altrimenti applica il default dell'architettura.
        """
        if text_encoders is not None and hasattr(self, "gen_encoder"):
            self._all_text_encoders = list(text_encoders)   # inventario completo (per il refiltro)
            self._refilter_gen_encoder()                     # mostra solo gli encoder compatibili
        if clip_types is not None and hasattr(self, "gen_clip_type"):
            cur = self.gen_clip_type.currentText()
            self.gen_clip_type.clear()
            self.gen_clip_type.addItem("auto", None)
            for t in clip_types:
                self.gen_clip_type.addItem(t, t)
            if cur and cur != "auto":
                i = self.gen_clip_type.findText(cur)
                if i >= 0: self.gen_clip_type.setCurrentIndex(i)
            # forza i default per arch (clip_type → auto)
            self._apply_arch_defaults(self.gen_arch.currentText())
        if vaes is not None and hasattr(self, "gen_vae"):
            cur = self.gen_vae.currentText()
            self.gen_vae.clear()
            self.gen_vae.addItem(tr("f1.gen.auto_from_model"), None)
            for v in vaes:
                self.gen_vae.addItem(v, v)
            if cur:
                i = self.gen_vae.findText(cur)
                if i >= 0: self.gen_vae.setCurrentIndex(i)

    def _refilter_gen_encoder(self, *_):
        """Aggiorna il dropdown «Text encoder» in base al modello selezionato.

        Per i modelli FLUX.2 mostra SOLO gli encoder compatibili (Mistral-3 / Qwen3):
        gli altri (Gemma, T5, CLIP-L, …) caricati come `flux2` fanno crashare il sampler
        con «Input img and txt tensors must have 3 dimensions». Per gli altri modelli
        mostra l'intero inventario. Mantiene la selezione corrente se resta valida."""
        if not hasattr(self, "gen_encoder") or not getattr(self, "_all_text_encoders", None):
            return
        model = self.gen_model.currentText()
        flux2 = _is_flux2(model)
        cur = self.gen_encoder.currentData()
        self.gen_encoder.blockSignals(True)
        self.gen_encoder.clear()
        self.gen_encoder.addItem(tr("f1.gen.auto_from_model"), None)
        for e in self._all_text_encoders:
            if flux2 and not is_compatible_encoder(model, e):
                continue                                  # encoder incompatibile → nascosto
            self.gen_encoder.addItem(e, e)
        idx = self.gen_encoder.findData(cur) if cur else 0
        self.gen_encoder.setCurrentIndex(idx if idx >= 0 else 0)  # se sparito → «auto»
        self.gen_encoder.blockSignals(False)

    def _update_auto_components(self, *_):
        """Auto-resolve componenti: rilevata l'ARCHITETTURA del modello scelto (detector header), mostra
        sulle voci 'auto' di VAE/encoder/CLIP la famiglia + il componente consigliato. L'utente può sempre
        scegliere un altro valore (override). Niente forzature: aggiorna solo l'etichetta/tooltip di 'auto'."""
        if not hasattr(self, "gen_vae"):
            return
        name = self.gen_model.currentText()
        det = {}
        if name and "auto" not in name.lower():
            import os as _os
            from pathlib import Path as _P
            from core import model_detect
            roots = [_P(r"E:\AI\Models\diffusion_models"), _P(r"E:\AI\Models\Checkpoints"),
                     _P(r"E:\AI\Models\checkpoints")]
            fp = name if _os.path.isfile(name) else next(
                (str(r / name) for r in roots if (r / name).is_file()), None)
            if fp:
                try:
                    det = model_detect.components_for_path(fp)
                except Exception:                            # noqa: BLE001
                    det = {}
        fam = det.get("family")
        # ripristina le etichette base, poi annota se abbiamo una famiglia
        base_vae = tr("f1.gen.auto_from_model")
        base_enc = tr("f1.gen.auto_from_model")
        if fam:
            vae = det.get("vae") or "auto"
            tes = det.get("text_encoders") or []
            te0 = tes[0] if tes else "auto"
            hint_v = f"{base_vae} — {fam}: {str(vae).replace(chr(92), '/').split('/')[-1]}"
            hint_e = f"{base_enc} — {fam}: {te0}"
            tip = tr("f1.gen.auto.detected", "Architettura rilevata: {fam}. 'auto' userà VAE/encoder adatti."
                     ).format(fam=fam)
            if self.gen_vae.count():
                self.gen_vae.setItemText(0, hint_v); self.gen_vae.setItemData(0, tip, Qt.ToolTipRole)
            if self.gen_encoder.count():
                self.gen_encoder.setItemText(0, hint_e); self.gen_encoder.setItemData(0, tip, Qt.ToolTipRole)
            if self.gen_clip_type.count():
                self.gen_clip_type.setItemData(0, tip, Qt.ToolTipRole)
        else:
            if self.gen_vae.count():
                self.gen_vae.setItemText(0, base_vae)
            if self.gen_encoder.count():
                self.gen_encoder.setItemText(0, base_enc)

    # Popola dinamicamente il dropdown Inpaint con TUTTI i modelli FLUX disponibili:
    # - checkpoint Flux (CheckpointLoaderSimple → ramo `checkpoint` del pipeline)
    # - UNET Flux/Fill da diffusion_models (UNETLoader → ramo `unet`)
    # - UNET FLUX.2 (es. FLUX.2 [dev]): ora gestiti da `_build_flux2` (VAE flux2 + encoder
    #   per-variante). Restano fuori solo i Kontext (modello editing, non inpaint).
    def _refresh_inpaint_checkpoints(self, checkpoint_models: list[str]):
        # heuristic: identifica file FLUX/FLUX.2-compatibili dal nome (catalogo è opzionale)
        def is_flux_like(name: str) -> bool:
            n = (name or "").lower()
            if "kontext" in n:
                return False
            # esclusioni tecniche
            if any(x in n for x in ("text_encoders", "vae", ".vae.", "clip")):
                return False
            # keyword positive (catalogo o derivate)
            return any(k in n for k in (
                "flux", "getphat", "fuxcapacity", "krea", "reality", "fluxunchained",
                "sigma", "verus", "pixelwave",
            ))
        ckpts = [c for c in checkpoint_models if is_flux_like(c)]
        # in più: prendi anche i UNET Flux da diffusion_models (Fill, Dev, ecc.)
        unets = [m for m in (self._models or []) if is_flux_like(m)]
        # rimuovi voci dinamiche precedenti
        for label in list(self._dynamic_checkpoints.keys()):
            idx = self.inp_model.findText(label)
            if idx >= 0:
                self.inp_model.removeItem(idx)
        self._dynamic_checkpoints.clear()
        # inserisci nuove voci PRIMA di "Nessuno (solo LTX I2V)"
        none_idx = self.inp_model.findText("Nessuno (solo LTX I2V)")
        insert_at = none_idx if none_idx >= 0 else self.inp_model.count()
        import os as _os
        # `_dynamic_checkpoints` mappa label → file checkpoint (ramo CheckpointLoaderSimple).
        # I UNET Flux/Fill di diffusion_models vanno via `_dynamic_unets` (UNETLoader).
        for ckpt in sorted(ckpts):
            base = _os.path.splitext(_os.path.basename(ckpt))[0]
            label = f"{base}  · checkpoint Flux"
            self._dynamic_checkpoints[label] = ckpt
            self.inp_model.insertItem(insert_at, label)
            insert_at += 1
        # UNET-style FLUX (diffusion_models): non duplicare i file già esposti dalle
        # voci hardcoded di INPAINT_MODELS (fluxunchained-fill / FLUX_DEV_Fill).
        hardcoded_unets = {(spec.get("unet") or "").lower()
                           for spec in INPAINT_MODELS.values() if spec.get("unet")}
        self._dynamic_unets = getattr(self, "_dynamic_unets", {})
        for label in list(self._dynamic_unets.keys()):
            idx = self.inp_model.findText(label)
            if idx >= 0:
                self.inp_model.removeItem(idx)
        self._dynamic_unets.clear()
        for u in sorted(unets):
            if u.lower() in hardcoded_unets:
                continue
            base = _os.path.splitext(_os.path.basename(u))[0]
            label = f"{base}  · UNET Flux"
            self._dynamic_unets[label] = u
            self.inp_model.insertItem(insert_at, label)
            insert_at += 1

    def set_inpaint_availability(self, unet_names: list[str]):
        """Marca nel menu i modelli inpaint NON ancora presenti in ComfyUI (es. in download)."""
        self._inpaint_avail = set(unet_names or [])
        for i in range(self.inp_model.count()):
            label = self.inp_model.itemText(i).split("  ⛔")[0]
            # Modelli dinamici (checkpoint Flux o UNET Flux): vengono dall'inventario
            # ComfyUI → sono per definizione installati, niente marker ⛔.
            if label in self._dynamic_checkpoints or label in getattr(self, "_dynamic_unets", {}):
                self.inp_model.setItemText(i, label)
                continue
            spec = INPAINT_MODELS.get(label, {})
            unet = spec.get("unet")
            if unet and unet not in self._inpaint_avail:
                self.inp_model.setItemText(i, label + "  ⛔ non installato")
            else:
                self.inp_model.setItemText(i, label)

    # ---------- API per il controller ----------
    def set_video_aspect(self, w: int, h: int):
        self.canvas.set_aspect(w, h)

    def get_values(self) -> dict:
        return {
            "prompt_positive": self.prompt.toPlainText().strip(),
            "prompt_negative": self.neg.toPlainText().strip(),
        }

    def _auto_inp_guidance(self, label: str):
        """Auto-imposta la Guidance secondo il modello inpaint: Fill dev=alta (30), distillato/Hyper=bassa (2.5),
        FLUX base=media (3.5), Klein=bassa (4). Guida l'utente al valore giusto (poi può cambiarlo)."""
        if not hasattr(self, "inp_guidance"):
            return
        l = (label or "").lower()
        if "hyper" in l or "hyp8" in l or "hyp-8" in l or "distill" in l or "turbo" in l:
            g = 2.5
        elif "klein" in l or "flux.2" in l or "flux2" in l:
            g = 4.0
        elif "base" in l:                       # FLUX base 64ch → FluxInpaintPipeline (segue il prompt)
            g = 3.5
        else:                                   # FLUX Fill dev (context-fill, guidance alta)
            g = 30.0
        self.inp_guidance.setValue(g)

    def get_inpaint_config(self) -> dict:
        label = self.inp_model.currentText().split("  ⛔")[0]
        # Risoluzione voce dinamica:
        # - checkpoint Flux (CheckpointLoaderSimple): _dynamic_checkpoints
        # - UNET Flux/Fill (UNETLoader):              _dynamic_unets
        # - hardcoded (FLUX Fill / FLUX Fill GGUF):    INPAINT_MODELS
        ckpt = self._dynamic_checkpoints.get(label)
        dyn_unet = getattr(self, "_dynamic_unets", {}).get(label)
        if ckpt:
            spec = {"arch": "Flux", "unet": None, "steps": self.inp_steps.value()}
        elif dyn_unet:
            spec = {"arch": "Flux", "unet": dyn_unet, "steps": self.inp_steps.value()}
        else:
            spec = INPAINT_MODELS.get(label, {})
        unet = spec.get("unet")
        available = (unet is None) or (unet in self._inpaint_avail) if self._inpaint_avail else True
        imodel = self.integrate_model.currentText() if self.integrate_model.count() else ""
        if imodel.startswith("("):
            imodel = ""
        return {
            "inpaint_prompt": self.inpaint_prompt.toPlainText().strip(),
            "outpaint_prompt": self.outpaint_prompt.toPlainText().strip(),
            "integrate_prompt": self.integrate_prompt.toPlainText().strip(),
            "integrate_model": imodel or None,
            "integrate_engine": self._engine_val(self.integrate_engine.currentIndex()),
            "integrate_variant": self._integrate_krea2_variant(),   # turbo/raw quando engine=krea2
            "grounding": self.edit_grounding_px(),
            "integrate_puzzle_lora": ("MagicQuill\\puzzle_lora.safetensors"
                                      if self.integrate_puzzle.isChecked() else None),
            # Campi dedicati all'integrazione (duplicati rispetto a inpaint/outpaint)
            "integrate_negative": self.int_neg.toPlainText().strip(),
            "integrate_steps": self.int_steps.value(),
            "integrate_guidance": self.int_guidance.value() / 10.0,
            "integrate_seed": None if self.int_seed_rand.isChecked() else self.int_seed.value(),
            # FOTO DI BASE di "Integra", scelta a mano. Vuoto ⇒ si usa quella del CANVAS (main_window
            # la carica nel canvas prima di procedere). Inpaint/outpaint usano solo il canvas.
            "integrate_photo": (self.int_photo["edit"].text().strip()
                                if hasattr(self, "int_photo") else ""),
            # Campi inpaint/outpaint
            "negative": self.inp_neg.toPlainText().strip(),
            "strength": self.strength.value() / 100.0,
            "guidance": float(self.inp_guidance.value()),   # CFG esplicita dalla GUI (aderenza al prompt)
            "feather": int(self.inp_feather.value()),       # 0 = auto proporzionale
            "model": label,
            "unet": unet,
            "checkpoint": ckpt,
            "steps": self.inp_steps.value(),
            "seed": None if self.inp_seed_rand.isChecked() else self.inp_seed.value(),
            "lora": self.inp_lora.currentData(),
            "available": available,
            "enhance": self.edit_enhance.isChecked() if hasattr(self, "edit_enhance") else False,
        }

    # ---------- composizione multi-foto (multi-reference / stitch) ----------
    def _compose_add(self):
        from PySide6.QtWidgets import QFileDialog
        files, _ = QFileDialog.getOpenFileNames(
            self, tr("f1.compose.add"), "", "Immagini (*.png *.jpg *.jpeg *.webp *.bmp)")
        for f in files:
            self.compose_list.addItem(f)

    def _compose_remove(self):
        for it in self.compose_list.selectedItems():
            self._compose_masks.pop(it.text().split("  🎭")[0], None)
            self.compose_list.takeItem(self.compose_list.row(it))

    def _compose_mask(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        it = self.compose_list.currentItem()
        if not it:
            QMessageBox.information(self, tr("f1.compose.group"), tr("f1.compose.mask.pick_first")); return
        key = it.text().split("  🎭")[0]
        f, _ = QFileDialog.getOpenFileName(
            self, tr("f1.compose.mask"), "", "Maschera (*.png *.jpg *.jpeg *.bmp)")
        if f:
            self._compose_masks[key] = f
            it.setText(key + "  🎭")

    def get_compose_config(self) -> dict:
        imgs = [self.compose_list.item(i).text().split("  🎭")[0] for i in range(self.compose_list.count())]
        return {
            "images": imgs,
            "masks": dict(getattr(self, "_compose_masks", {})),
            "method": "stitch" if self.compose_method.currentIndex() == 1 else "multiref",
            "prompt": self.compose_prompt.toPlainText().strip(),
            "engine": self._engine_val(self.integrate_engine.currentIndex()),
            "variant": self._integrate_krea2_variant(),            # turbo/raw quando engine=krea2
            "grounding": self.edit_grounding_px(),
            "model": (self.integrate_model.currentText() if self.integrate_model.count() else "") or None,
        }

    def get_gen_config(self) -> dict:
        enc = self.gen_encoder.currentData() if hasattr(self, "gen_encoder") else None
        vae = self.gen_vae.currentData() if hasattr(self, "gen_vae") else None
        # `auto` → None (il pipeline sceglie il default per arch).
        ctype = self.gen_clip_type.currentData() if hasattr(self, "gen_clip_type") else None
        smp = self.gen_sampler.currentText() if hasattr(self, "gen_sampler") else None
        if smp == "auto": smp = None
        sch = self.gen_scheduler.currentText() if hasattr(self, "gen_scheduler") else None
        if sch == "auto": sch = None
        # Seed: random se checkbox attivo → None (il caller userà random.randint())
        seed_fixed = (not self.gen_seed_rand.isChecked()) if hasattr(self, "gen_seed_rand") else False
        pulid_face = None
        if getattr(self, "gen_pulid_enable", None) and self.gen_pulid_enable.isChecked():
            _f = self.gen_pulid_face.text().strip()
            pulid_face = _f or None                          # può contenere più path separati da | (media)
        return {
            "pulid_face": pulid_face,
            "pulid_strength": (self.gen_pulid_strength.value() if hasattr(self, "gen_pulid_strength") else 0.8),
            "pulid_start_step": (self.gen_pulid_start.value() if hasattr(self, "gen_pulid_start") else 0),
            "pulid_true_cfg": (self.gen_pulid_truecfg.value() if hasattr(self, "gen_pulid_truecfg") else 1.0),
            "arch": {"FLUX": "flux", "Krea 2": "krea2", "Z-Image": "zimage", "Ernie": "ernie"}.get(self.gen_arch.currentText(), "flux"),
            "model": self._gen_model_name(),
            # Krea 2: variante (turbo/raw) dalla voce selezionata nel dropdown → il worker risolve lo snapshot
            "krea2_variant": ("raw" if str(self.gen_model.currentData() or "").endswith("raw") else "turbo")
                             if self.gen_arch.currentText() == "Krea 2" else None,
            "loras": self.gen_lora_tab.selected_loras(),
            "steps": self.gen_steps.value(),
            "cfg": float(self.gen_cfg.value()) if hasattr(self, "gen_cfg") else None,
            "sampler": smp,
            "scheduler": sch,
            "encoder": enc,
            "clip_type": ctype,
            "vae": vae,
            "seed": self.gen_seed.value() if seed_fixed else None,
            "prompt": self.gen_prompt.toPlainText().strip() or self.prompt.toPlainText().strip(),
            "negative": self.gen_neg.toPlainText().strip() if hasattr(self, "gen_neg") else "",
            "enhance": self.gen_enhance.isChecked() if hasattr(self, "gen_enhance") else False,
            "nsfw": self._allow_nsfw(),
        }

    @property
    def image_path(self):
        return self.canvas.image_path
