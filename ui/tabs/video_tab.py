"""Tab Video: caratteristiche video (fps, risoluzione, durata, NSFW/SFW, modello),
+ Sampler/Seed (spostati da Avanzate), + LoRA LTX-2 (integrato dal tab LoRA).
Il selettore NSFW/SFW del video re-filtra istantaneamente modelli e LoRA."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
    QHBoxLayout, QLabel, QGroupBox, QSlider, QCheckBox, QPushButton,
    QPlainTextEdit, QToolButton, QInputDialog, QMessageBox, QScrollArea, QLineEdit,
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
    "10EROS (NSFW)":                  "ltx2310eros_v1.safetensors",
}
_MODEL_NSFW = {"10EROS (NSFW)"}      # voci modello considerate NSFW (nascoste in SFW)
GGUF_FALLBACK_FILE = "LTXVideo\\LTX-2\\quantstack\\ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf"

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
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QScrollArea.NoFrame)
        host = QWidget(); root = QVBoxLayout(host); sc.setWidget(host); outer.addWidget(sc)

        # --- Prompt (spostato qui dal tab Frame 1) ---
        prompt_group = QGroupBox(tr("vid.prompt.group"))
        prompt_vl = QVBoxLayout(prompt_group)
        prompt_vl.addWidget(QLabel(tr("vid.prompt.lbl")))
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText(tr("vid.prompt.ph"))
        self.prompt.setMaximumHeight(100)
        prompt_vl.addWidget(self.prompt)
        prompt_vl.addWidget(QLabel(tr("vid.neg.lbl")))
        self.neg = QPlainTextEdit(DEFAULT_NEG)
        self.neg.setMaximumHeight(70)
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
        form.addRow(tr("vid.content_lbl"), self.nsfw)
        self.model = QComboBox()
        self.model.setMaxVisibleItems(20)       # fix: mostra tutti i modelli nel popup
        self.model.currentIndexChanged.connect(self._on_model_changed)
        self.model.currentIndexChanged.connect(self._update_model_star)
        mrow = QHBoxLayout(); mrow.setContentsMargins(0, 0, 0, 0)
        mrow.addWidget(self.model, 1)
        self.model_star = QToolButton(); self.model_star.setText("☆")
        self.model_star.setStyleSheet("QToolButton{border:none;font-size:16px;color:#777;}")
        self.model_star.setToolTip(tr("vid.model_star.tt"))
        self.model_star.clicked.connect(self._toggle_fav_model_current)
        mrow.addWidget(self.model_star)
        form.addRow(tr("vid.model_lbl"), self._wrap(mrow))
        self.res_preset = QComboBox(); self.res_preset.addItems(list(RES_PRESETS.keys()))
        self.res_preset.currentTextChanged.connect(self._on_preset)
        form.addRow(tr("vid.res_lbl"), self.res_preset)
        wh = QHBoxLayout()
        self.width = QSpinBox(); self.width.setRange(256, 2048); self.width.setSingleStep(32); self.width.setValue(704)
        self.height = QSpinBox(); self.height.setRange(256, 2048); self.height.setSingleStep(32); self.height.setValue(1280)
        wh.addWidget(QLabel(tr("vid.w_lbl"))); wh.addWidget(self.width)
        wh.addWidget(QLabel(tr("vid.h_lbl"))); wh.addWidget(self.height)
        form.addRow("", self._wrap(wh))
        # FPS e Durata sulla stessa riga (compattazione richiesta)
        self.fps = QSpinBox(); self.fps.setRange(1, 60); self.fps.setValue(24)
        self.length = QDoubleSpinBox(); self.length.setRange(1, 60); self.length.setValue(8.0); self.length.setSuffix(" s")
        fpsdur = QHBoxLayout()
        fpsdur.addWidget(QLabel(tr("vid.fps_lbl"))); fpsdur.addWidget(self.fps)
        fpsdur.addSpacing(12)
        fpsdur.addWidget(QLabel(tr("vid.dur_lbl"))); fpsdur.addWidget(self.length, 1)
        form.addRow(tr("vid.fps_dur_lbl"), self._wrap(fpsdur))
        # Forza immagine: quanto la foto di partenza è rispettata nel VIDEO (nodo 5647 → LTXVAddGuide.strength).
        # Default 0.50 ("soft reference" come il workflow di riferimento): valori alti (0.8+) ancorano
        # troppo al fotogramma → movimento rigido/innaturale; 0.45-0.6 lasciano al modello libertà di
        # movimento mantenendo l'identità. Alza verso 0.8 solo se vuoi aderenza stretta all'immagine.
        isr = QHBoxLayout()
        self.img_strength = QSlider(Qt.Horizontal); self.img_strength.setRange(0, 100); self.img_strength.setValue(50)
        self.img_strength_lbl = QLabel("0.50")
        self.img_strength.valueChanged.connect(lambda v: self.img_strength_lbl.setText(f"{v/100:.2f}"))
        self.img_strength.setToolTip(tr("vid.img_strength.tt"))
        isr.addWidget(self.img_strength, 1); isr.addWidget(self.img_strength_lbl)
        form.addRow(tr("vid.img_strength_lbl"), self._wrap(isr))
        # Metodo di condizionamento immagine→video (solo quando si parte da un'immagine)
        self.i2v_method = QComboBox()
        self.i2v_method.addItem(tr("vid.i2v_method.addguide"), "addguide")
        self.i2v_method.addItem(tr("vid.i2v_method.inplace"), "inplace")
        self.i2v_method.setToolTip(tr("vid.i2v_method.tt"))
        form.addRow(tr("vid.i2v_method_lbl"), self.i2v_method)
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

        # --- V2V: editing video (Normal / Talk / ReTake) ---
        from ui.widgets.v2v_panel import V2VPanel
        v2v_box = QGroupBox(tr("vid.v2v.group"))
        v2v_l = QVBoxLayout(v2v_box)
        self.v2v_panel = V2VPanel()
        v2v_l.addWidget(self.v2v_panel)
        root.addWidget(v2v_box)

        # --- Storyboard multi-scena (immagini-keyframe a tempo → video senza montaggio) ---
        from ui.widgets.storyboard_panel import StoryboardPanel
        sb_box = QGroupBox(tr("vid.sb.group"))
        sb_l = QVBoxLayout(sb_box)
        self.storyboard_panel = StoryboardPanel()
        sb_l.addWidget(self.storyboard_panel)
        root.addWidget(sb_box)

        # --- Restauro volti automatico (post-generazione, opzionale) ---
        frbox = QGroupBox(tr("vid.face.group")); frf = QFormLayout(frbox)
        self.fr_auto = QCheckBox(tr("vid.face.auto"))
        self.fr_auto.setToolTip(tr("vid.face.auto.tt"))
        frf.addRow(self.fr_auto)
        from core import face_restore_pipeline as _frp
        self.fr_auto_model = QComboBox(); self.fr_auto_model.addItems(_frp.FACERESTORE_MODELS)
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
        root.addWidget(frbox)

        # --- Sampler / Seed (spostati da Avanzate) ---
        self.sampler_seed = SamplerSeedGroup()
        root.addWidget(self.sampler_seed)

        # --- LoRA LTX-2 (integrato; NSFW pilotato dal selettore video) ---
        lbox = QGroupBox(tr("vid.lora.group")); ll = QVBoxLayout(lbox)
        self.lora = LoraTab(allow_nsfw_getter=lambda: self.nsfw.currentText() == "NSFW", embedded=True)
        ll.addWidget(self.lora)
        root.addWidget(lbox, 1)

        self._refilter_models()
        self._on_preset(self.res_preset.currentText())

    def _wrap(self, layout):
        w = QWidget(); w.setLayout(layout); return w

    def _on_nsfw(self, *_):
        self._refilter_models()      # riselezione istantanea modelli utilizzabili
        self.lora.refresh()          # re-filtro LoRA per NSFW

    def _refilter_models(self):
        nsfw = self.nsfw.currentText() == "NSFW"
        cur_data = self.model.currentData() or {}
        cur_key = cur_data.get("key")
        self.model.blockSignals(True)
        self.model.clear()
        self.model.addItem(tr("vid.model.auto"), {"kind": "auto", "key": "auto"})

        def add(label: str, kind: str, model: str, is_nsfw: bool, enabled: bool = True):
            if (not nsfw) and is_nsfw:
                return
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
            if not enabled:
                item = self.model.model().item(self.model.count() - 1)
                if item:
                    item.setEnabled(False)

        # fav-first: i preferiti compaiono in cima a ogni categoria
        def fav_first(lst):
            return sorted(lst, key=lambda m: (not user_prefs.is_fav_model(m), m.lower()))
        diffusion = fav_first([m for m in self._diffusion_models if self._is_ltx_video_model(m)])
        checkpoints = fav_first([m for m in self._checkpoint_models if self._is_ltx_video_model(m)])
        ggufs = fav_first([m for m in self._gguf_models if "ltx" in m.lower()])

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
        # cur_key=="auto" deriva dal populate iniziale (inventario vuoto): NON considerarlo
        # come scelta utente, così l'auto-select può prendere il miglior modello concreto.
        matched = False
        if cur_key and cur_key != "auto":
            for i in range(self.model.count()):
                data = self.model.itemData(i) or {}
                if data.get("key") == cur_key:
                    self.model.setCurrentIndex(i)
                    matched = True
                    break
        # Auto-select del miglior modello (VRAM-aware) se nessuna scelta utente preservata.
        if not matched and self._inventory_ready and not self._auto_selected:
            best_idx = self._pick_best_model_index()
            if best_idx is not None:
                self.model.setCurrentIndex(best_idx)
                self._auto_selected = True
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
        low = (name or "").lower()
        if "text_projection" in low or "lora" in low:
            return False
        return (self._mcat.arch_of(name).upper() == "LTX") or "ltx" in low or "eros" in low

    def _is_nsfw_model(self, name: str) -> bool:
        low = (name or "").lower()
        return self._mcat.is_nsfw(name) or any(k in low for k in ("eros", "nsfw", "xxx", "adult", "porn"))

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

    def get_values(self) -> dict:
        v = {
            "prompt_positive": self.prompt.toPlainText().strip(),
            "prompt_negative": self.neg.toPlainText().strip(),
            "nsfw_mode": "nsfw" if self.nsfw.currentText() == "NSFW" else "sfw",
            "width": round32(self.width.value()),
            "height": round32(self.height.value()),
            "fps": self.fps.value(),
            "length_s": self.length.value(),
            "ref_strength": self.img_strength.value() / 100.0,
            "downscale_half": self.downscale_half.isChecked(),
            "upscale_x2": self.upscale_x2.isChecked(),
            "refine_hr": self.refine_hr.isChecked(),
            "i2v_method": self.i2v_method.currentData() or "addguide",
        }
        sel_data = self.model.currentData() or {"kind": "auto"}
        kind = sel_data.get("kind")
        if kind == "gguf":
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
