"""Finestra principale LTX Studio (MVP): tab + preview live + Genera cablato al backend."""
from __future__ import annotations

import random
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal, Qt
from PySide6.QtGui import QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget, QPushButton,
    QLabel, QProgressBar, QPlainTextEdit, QMessageBox,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.comfy_driver import ComfyDriver           # noqa: E402
from core.template_engine import TemplateEngine     # noqa: E402
from core.state_manager import StateManager         # noqa: E402
from core.inpaint_pipeline import InpaintPipeline   # noqa: E402
from core.gen_pipeline import GenPipeline           # noqa: E402
from core.segment_pipeline import SegmentPipeline   # noqa: E402
from core.upscale_pipeline import UpscalePipeline    # noqa: E402
from core.integrate_pipeline import IntegratePipeline # noqa: E402
from core.kontext_pipeline import KontextPipeline     # noqa: E402
from core.video_enhance_pipeline import VideoEnhancePipeline  # noqa: E402
from core.extend_pipeline import ExtendPipeline         # noqa: E402
from core.face_restore_pipeline import FaceRestorePipeline  # noqa: E402
from core import gui_state                                # noqa: E402  (persistenza impostazioni GUI)
from ui.tabs.video_def_tab import VideoDefTab          # noqa: E402
from ui.tabs.video_extend_tab import VideoExtendTab    # noqa: E402
from ui.tabs.restore_tab import RestoreTab             # noqa: E402
from ui.tabs.import_models_tab import ImportModelsTab  # noqa: E402
from ui.tabs.video_tab import VideoTab              # noqa: E402
from ui.tabs.frame1_tab import Frame1Tab            # noqa: E402
from ui.tabs.lora_tab import LoraTab                # noqa: E402
from ui.tabs.advanced_tab import AdvancedTab        # noqa: E402
from ui.tabs.audio_tab import AudioTab              # noqa: E402
from ui.tabs.controlnet_tab import ControlNetTab    # noqa: E402
from ui.settings_dialog import SettingsDialog       # noqa: E402
from ui.wizard import SetupWizard                   # noqa: E402
from ui.progress_dialog import GenProgressDialog     # noqa: E402
from ui.video_player_dialog import VideoPlayerDialog # noqa: E402
from ui.navigator_dialog import NavigatorDialog      # noqa: E402
from core.metadata_saver import MetadataSaver        # noqa: E402
from core.restore_pipeline import RestorePipeline    # noqa: E402

OUTPUT_DIR = Path(r"E:\AI\ltxstudio\output")


class _HoldCompareLabel(QLabel):
    """Mostra il risultato; mentre tieni premuto il mouse mostra l'immagine di partenza (confronto)."""
    def __init__(self, result_pix, before_pix=None):
        super().__init__()
        self._result = result_pix
        self._before = before_pix
        self.setAlignment(Qt.AlignCenter)
        if before_pix is not None:
            self.setCursor(Qt.PointingHandCursor)
        self._show(result_pix)

    def _show(self, pix):
        if pix and not pix.isNull():
            self.setPixmap(pix.scaled(620, 620, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def mousePressEvent(self, e):
        if self._before is not None:
            self._show(self._before)

    def mouseReleaseEvent(self, e):
        self._show(self._result)


class ComfyBridge(QObject):
    """Marshalla le callback del WS (thread di rete) verso slot Qt (thread UI)."""
    progress = Signal(int, int)
    preview = Signal(bytes)
    status = Signal(str)
    executing = Signal(str)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LTX Studio — LTX-2.3 (MVP)")
        self.resize(1180, 740)

        self.driver = ComfyDriver()
        self.bridge = ComfyBridge()
        try:
            self.engine = TemplateEngine()
            self._engine_err = None
        except Exception as e:  # template non ancora catturato
            self.engine = None
            self._engine_err = str(e)

        self._current_pid: str | None = None
        self._inpaint_pid: str | None = None
        self._generation_cache: dict[str, dict] = {}   # prompt_id → meta dict per sidecar
        self._inpaint_mode: str = "inpaint"
        self._inpaint_bbox: tuple[int, int, int, int] | None = None
        self._inpaint_before: str | None = None    # immagine di partenza, per confronto nel dialog risultato
        self._overrides: dict = {}
        self.state = StateManager()
        self.inpainter = InpaintPipeline()
        self.genpipe = GenPipeline()
        self.segpipe = SegmentPipeline()
        self.upscaler = UpscalePipeline()
        self.integrator = IntegratePipeline()
        self.kontext = KontextPipeline()      # integra/colore stile MagicQuill (Kontext/FLUX.2)
        self.enhancer = VideoEnhancePipeline()  # definizione video (SeedVR2 / ESRGAN / RIFE)
        self.extender = ExtendPipeline()        # prolunga video (LTX-2.3 nativo, multi-segmento)
        self.facerestorer = FaceRestorePipeline()   # restauro volti per-frame (CodeFormer/GFPGAN)
        self.restorer = RestorePipeline()       # restauro foto + colorazione
        self._last_video_path: str | None = None
        self._sam_models: list[str] = []
        self._seg_pid: str | None = None
        self._inventory_set = False
        self.gen_dialog = GenProgressDialog(self, on_cancel=self._cancel_gen)
        self.video_player: VideoPlayerDialog | None = None
        self._build_ui()
        self._build_menu()
        self._wire_ws()
        # bottoni inpaint/outpaint del canvas + generazione immagine
        self.frame1_tab.canvas.request_inpaint.connect(lambda: self.generate_inpaint("inpaint"))
        self.frame1_tab.canvas.request_outpaint.connect(lambda: self.generate_inpaint("outpaint"))
        # Bottone «Integra immagini» spostato dal canvas al tab Inpaint/Outpaint
        self.frame1_tab.request_integrate.connect(self.generate_integrate)
        self.frame1_tab.request_generate.connect(self.generate_image)
        self.frame1_tab.request_upscale.connect(self.generate_upscale)
        self.frame1_tab.request_recolor.connect(self.generate_recolor)
        # selezione oggetto (SAM): testo dal tab, regione (click/box) dal canvas
        self.frame1_tab.request_segment_text.connect(self.segment_by_text)
        self.frame1_tab.canvas.request_segment_region.connect(self.segment_by_region)
        self.video_def_tab.request_enhance.connect(self.generate_video_enhance)
        self.video_def_tab.request_face_restore.connect(self.generate_face_restore)
        self.video_extend_tab.request_extend.connect(self.generate_video_extend)
        self.restore_tab.request_restore.connect(self.generate_restore)
        self.models_tab.catalog_updated.connect(self._on_catalog_updated)
        self.frame1_tab.request_restore_photo.connect(self._restore_from_canvas)
        # Segnali del tab Avanzate → funzioni della finestra principale
        self.advanced_tab.open_workflow.connect(self._open_workflow)
        self.advanced_tab.open_analyzer.connect(self._open_analyzer)
        self.advanced_tab.open_navigator.connect(self._open_navigator)
        if getattr(self, "_last_video_path", None):
            self.video_def_tab.set_video(self._last_video_path)
            self.video_extend_tab.set_video(self._last_video_path)

        # poll stato ComfyUI + completamento job
        self._status_timer = QTimer(self); self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start(3000); self._refresh_status()

        # ── Ripristino impostazioni GUI salvate (tutti i tab, automatico) ──
        # I campi statici si ripristinano subito; i menu popolati dall'inventario live di
        # ComfyUI (modelli, SAM, upscale, LoRA…) vengono ri-applicati una volta che
        # l'inventario è pronto, in _refresh_status (flag _gui_state_restored).
        self._saved_gui_state = gui_state.load()
        self._gui_state_restored = False
        QTimer.singleShot(0, self._restore_gui_state)

    # ---------- UI ----------
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # sinistra: tab
        self.tabs = QTabWidget()
        self.video_tab = VideoTab()
        self.frame1_tab = Frame1Tab()
        # I widget Prompt globale / Prompt negativo sono stati spostati nel tab Video.
        # Aliasiamo i nomi `frame1_tab.prompt` / `frame1_tab.neg` ai widget reali in
        # video_tab così tutte le call-site esistenti (get_values, restore, ecc.) continuano
        # a funzionare senza refactor. I widget placeholder creati in _prompt_tab() vengono
        # sostituiti dai veri editor del tab Video.
        self.frame1_tab.prompt = self.video_tab.prompt
        self.frame1_tab.neg = self.video_tab.neg
        self.lora_tab = self.video_tab.lora        # le LoRA ora vivono nel tab Video (sotto sampler/seed)
        self.advanced_tab = AdvancedTab()
        self.audio_tab = AudioTab()
        self.controlnet_tab = ControlNetTab()
        self.video_def_tab = VideoDefTab()
        self.video_extend_tab = VideoExtendTab()
        self.restore_tab = RestoreTab()
        self.models_tab = ImportModelsTab()
        self.tabs.addTab(self.frame1_tab, "Immagini")
        self.tabs.addTab(self.audio_tab, "Audio / Voce")
        self.tabs.addTab(self.video_tab, "Video")
        self.tabs.addTab(self.controlnet_tab, "ControlNet")
        self.tabs.addTab(self.video_extend_tab, "Prolunga video")
        # «Migliora qualità» = unione di Restauro foto (🖼 Foto) + Definizione video (🎬 Video)
        self.quality_tab = QTabWidget()
        self.quality_tab.addTab(self.restore_tab, "🖼 Foto")
        self.quality_tab.addTab(self.video_def_tab, "🎬 Video")
        self.tabs.addTab(self.quality_tab, "✨ Migliora qualità")
        # Import Models diventa un sotto-tab di Avanzate
        self.advanced_tab.attach_models(self.models_tab)
        self.tabs.addTab(self.advanced_tab, "Avanzate")
        # NB: i tab (immagine di editing) vengono aggiunti DOPO il pannello anteprima,
        #     così l'anteprima sta a SINISTRA e l'editing a DESTRA (posizioni invertite).

        # sincronizza il formato del canvas Frame1 con la risoluzione del Video
        def _sync_aspect():
            v = self.video_tab.get_values()
            self.frame1_tab.set_video_aspect(v["width"], v["height"])
        self.video_tab.width.valueChanged.connect(_sync_aspect)
        self.video_tab.height.valueChanged.connect(_sync_aspect)
        self.video_tab.res_preset.currentTextChanged.connect(lambda *_: _sync_aspect())
        _sync_aspect()

        # SINISTRA: spazio di lavoro grande = canvas + selezione AI + Genera/Interrompi
        # (niente anteprima: il sistema non la genera). Canvas e selezione vivono qui.
        work = QVBoxLayout()
        work.addWidget(self.frame1_tab.canvas, 1)        # spazio di lavoro per l'immagine
        work.addWidget(self.frame1_tab.segment_box)      # sistema di selezione con AI, sotto il canvas
        btns = QHBoxLayout()
        self.gen_btn = QPushButton("🎬 Genera"); self.gen_btn.setObjectName("generate")
        self.gen_btn.clicked.connect(self.on_generate)
        self.cancel_btn = QPushButton("⏹ Interrompi"); self.cancel_btn.clicked.connect(self.driver.interrupt)
        self.free_vram_btn = QPushButton("🗑 Libera VRAM")
        self.free_vram_btn.setToolTip("Scarica i modelli dalla VRAM di ComfyUI (libera memoria GPU)")
        self.free_vram_btn.clicked.connect(self._on_free_vram)
        self.navigator_btn = QPushButton("🗂 Navigator")
        self.navigator_btn.setToolTip("Sfoglia le generazioni precedenti e riutilizza i parametri (Ctrl+G)")
        self.navigator_btn.clicked.connect(self._open_navigator)
        btns.addWidget(self.gen_btn, 2); btns.addWidget(self.cancel_btn, 1)
        btns.addWidget(self.free_vram_btn, 1); btns.addWidget(self.navigator_btn, 1)
        work.addLayout(btns)
        # Shortcut Ctrl+G per il Navigator
        nav_shortcut = QShortcut(QKeySequence("Ctrl+G"), self)
        nav_shortcut.activated.connect(self._open_navigator)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
        work.addWidget(self.progress)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(110)
        work.addWidget(self.log)
        root.addLayout(work, 3)             # SINISTRA: spazio di lavoro (grande)
        root.addWidget(self.tabs, 2)        # DESTRA: tab di controllo (sotto-tab in Prompt/Frame 1)

        self.status_lbl = QLabel("…")
        self.statusBar().addWidget(self.status_lbl)
        if self._engine_err:
            self._log(f"[!] Template non caricato: {self._engine_err}")
            self._log("    Cattura il golden prompt (templates/README_capture.md) e riavvia.")

    def _build_menu(self):
        m = self.menuBar().addMenu("Strumenti")
        act_wiz = m.addAction("Wizard guidato…"); act_wiz.triggered.connect(self.open_wizard)
        act_set = m.addAction("Impostazioni modelli…"); act_set.triggered.connect(self.open_settings)
        act_nav = m.addAction("🗂 Navigator Gallery…  Ctrl+G"); act_nav.triggered.connect(self._open_navigator)
        act_ana = m.addAction("🔍 Analizza impostazioni…"); act_ana.triggered.connect(self._open_analyzer)
        act_wfl = m.addAction("🎬 Workflow predefiniti…"); act_wfl.triggered.connect(self._open_workflow)

    def open_settings(self):
        if not self.engine:
            QMessageBox.warning(self, "Template mancante", self._engine_err or ""); return
        dlg = SettingsDialog(self.engine, self.driver, self._overrides, self)
        if dlg.exec():
            self._overrides = dlg.overrides()
            self._log(f"Override modelli aggiornati: {len(self._overrides)} nodi")

    def open_wizard(self):
        wiz = SetupWizard(self)
        if wiz.exec():
            r = wiz.result_values()
            self.frame1_tab.prompt.setPlainText(r.get("prompt_positive", ""))
            self.video_tab.length.setValue(r.get("length_s", 8))
            self.video_tab.nsfw.setCurrentText("NSFW" if r.get("nsfw_mode") == "nsfw" else "SFW")
            self.state.update({"nsfw_mode": r.get("nsfw_mode"), "audio_mode": r.get("audio_mode")})
            self._log(f"Wizard: obiettivo '{r.get('goal')}' applicato.")

    def _wire_ws(self):
        self.driver.on_progress = lambda v, m: self.bridge.progress.emit(v, m)
        self.driver.on_preview = lambda b: self.bridge.preview.emit(b)
        self.driver.on_executing = lambda n: self.bridge.executing.emit(n or "")
        self.driver.on_status = lambda d: None
        self.bridge.progress.connect(self._on_progress)
        self.bridge.preview.connect(self._on_preview)
        self.bridge.executing.connect(self._on_executing)
        try:
            self.driver.start_ws()
        except RuntimeError as e:
            self._log(f"WS preview non disponibile: {e}")

    # ---------- slot ----------
    def _on_progress(self, v: int, m: int):
        self.progress.setRange(0, max(m, 1)); self.progress.setValue(v)
        self.gen_dialog.on_progress(v, m)

    def _on_preview(self, data: bytes):
        self.gen_dialog.on_preview(data)      # l'anteprima vive nel dialog di avanzamento

    def _on_executing(self, node_id: str):
        self.gen_dialog.on_executing(node_id)

    def _cancel_gen(self):
        try:
            self.driver.interrupt()
        except Exception:
            pass
        self._current_pid = self._inpaint_pid = self._seg_pid = None
        self._log("Generazione annullata dall'utente.")

    def _on_free_vram(self):
        ok = self.driver.free_memory()
        if ok:
            self._log("✅ Memoria VRAM liberata (modelli scaricati da ComfyUI).")
        else:
            self._log("[!] Impossibile liberare VRAM — ComfyUI offline o endpoint /free non supportato.")

    def _open_navigator(self):
        """Apre il Navigator Gallery (Ctrl+G)."""
        dlg = NavigatorDialog(
            output_dir=OUTPUT_DIR,
            apply_callback=self._apply_reuse_fields,
            parent=self,
        )
        dlg.exec()

    def _apply_reuse_fields(self, fields: dict):
        """Riceve i campi selezionati da ReuseDialog e li applica alla GUI."""
        if "prompt_positive" in fields:
            self.frame1_tab.prompt.setPlainText(fields["prompt_positive"])
        if "prompt_negative" in fields:
            # il campo si chiama `neg` in frame1_tab (QPlainTextEdit nel sotto-tab Prompt)
            self.frame1_tab.neg.setPlainText(fields["prompt_negative"])
        v = self.video_tab
        if "width" in fields:
            try:
                v.width.setValue(int(fields["width"]))
            except Exception:
                pass
        if "height" in fields:
            try:
                v.height.setValue(int(fields["height"]))
            except Exception:
                pass
        if "fps" in fields:
            try:
                v.fps.setValue(int(fields["fps"]))
            except Exception:
                pass
        if "length_s" in fields:
            try:
                v.length.setValue(float(fields["length_s"]))
            except Exception:
                pass
        # Sampler / seed (ora in advanced_tab o sampler_seed sub-widget del video_tab)
        ss = getattr(v, "sampler_seed", None)
        if ss:
            if "steps" in fields:
                try:
                    ss.steps.setValue(int(fields["steps"]))
                except Exception:
                    pass
            if "cfg" in fields:
                try:
                    ss.cfg.setValue(float(fields["cfg"]))
                except Exception:
                    pass
            if "seed" in fields:
                try:
                    ss.seed.setValue(int(fields["seed"]))
                    ss.fixed_seed.setChecked(True)
                except Exception:
                    pass
            if "sampler" in fields:
                idx = ss.sampler.findText(str(fields["sampler"]))
                if idx >= 0:
                    ss.sampler.setCurrentIndex(idx)
            if "scheduler" in fields:
                idx = ss.scheduler.findText(str(fields["scheduler"]))
                if idx >= 0:
                    ss.scheduler.setCurrentIndex(idx)
        if "nsfw_mode" in fields:
            idx = v.nsfw.findText(str(fields["nsfw_mode"]).upper())
            if idx >= 0:
                v.nsfw.setCurrentIndex(idx)
        if "model" in fields:
            idx = v.model.findText(str(fields["model"]))
            if idx >= 0:
                v.model.setCurrentIndex(idx)
        if "audio_mode" in fields:
            at = self.audio_tab
            mode = str(fields["audio_mode"])
            # Mappa audio_mode → attributo del radio button in AudioTab
            _audio_map = {"native": "rb_native", "user": "rb_user",
                          "clone": "rb_clone", "mute": "rb_mute"}
            rb_attr = _audio_map.get(mode)
            if rb_attr:
                rb = getattr(at, rb_attr, None)
                if rb:
                    rb.setChecked(True)
        if "loras" in fields and isinstance(fields["loras"], list):
            # Riattiva le LoRA elencate nel sidecar con la strength salvata
            lora_tab = self.lora_tab
            for entry in fields["loras"]:
                name_comfy = entry.get("name", "")
                strength = float(entry.get("strength", 1.0))
                # Cerca per comfy_name nella lista visibile
                for row_name, row in lora_tab._rows.items():
                    e = row["entry"]
                    if e.comfy_name == name_comfy or row_name == name_comfy:
                        row["cb"].setChecked(True)
                        row["sp"].setValue(strength)
                        break
        self._log(f"[Navigator] {len(fields)} parametri ripristinati nella GUI.")

    def _restore_gui_state(self):
        """Applica le impostazioni GUI salvate a tutti i tab (best-effort)."""
        try:
            gui_state.restore(self, getattr(self, "_saved_gui_state", None) or {})
        except Exception as e:
            try:
                self._log(f"[warn] ripristino impostazioni GUI: {e}")
            except Exception:
                pass

    def closeEvent(self, event):
        """Salva lo stato di tutti i controlli GUI alla chiusura (così al riavvio si ritrova
        com'era). Generico: copre automaticamente i campi nuovi senza manutenzione."""
        try:
            gui_state.store(self)
        except Exception:
            pass
        super().closeEvent(event)

    def _refresh_status(self):
        up = self.driver.is_up()
        self.status_lbl.setText(
            f"ComfyUI: {'🟢 connesso' if up else '🔴 offline'} @ {self.driver.cfg.http}  |  "
            f"Template: {'OK' if self.engine else 'mancante'}")
        self.gen_btn.setEnabled(up and self.engine is not None)
        if up and not self._inventory_set:
            try:
                diffusion_models = self.driver.models_in("diffusion_models")
                checkpoint_models = self.driver.models_in("checkpoints")
                lora_models = self.driver.models_in("loras")
                self.frame1_tab.set_model_inventory(diffusion_models, checkpoint_models)
                oi = self.driver.object_info()
                if self.engine:
                    self.engine.set_object_info(oi)  # per resolve_everywhere
                self._apply_comfy_inventory(oi)      # disponibilità inpaint + modelli SAM
                self.video_tab.set_model_inventory(
                    diffusion_models, checkpoint_models, self._video_gguf_models(oi),
                    vram_gb=self.driver.gpu_vram_gb())
                self.video_tab.set_lora_inventory(lora_models)
                self._inventory_set = True
                # menu dinamici ora popolati → ri-applica le impostazioni salvate (una volta)
                if getattr(self, "_saved_gui_state", None) and not self._gui_state_restored:
                    self._gui_state_restored = True
                    self._restore_gui_state()
            except Exception:
                pass
        if self._current_pid:
            self._check_completion()
        if self._inpaint_pid:
            self._check_inpaint_done()
        if self._seg_pid:
            self._check_segment_done()

    def _apply_comfy_inventory(self, oi: dict):
        """Estrae da object_info le liste reali: inpaint, SAM, upscale. Gestisce sia il
        formato v1 [[opzioni],…] sia v3 ['COMBO',{'options':[…]}]."""
        def names(cls, inp):
            spec = oi.get(cls, {}).get("input", {}).get("required", {}).get(inp)
            if not isinstance(spec, list) or not spec:
                return []
            head = spec[0]
            if isinstance(head, list):
                return list(head)                                  # v1
            if head == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
                return list(spec[1].get("options", []))            # v3
            return []
        unets = names("UNETLoader", "unet_name") + names("UnetLoaderGGUF", "unet_name")
        self.frame1_tab.set_inpaint_availability(unets)
        # l'extend usa un UNET distilled hardcoded: correggilo se non è installato (file rinominato)
        if hasattr(self.extender, "ensure_unet"):
            self.extender.ensure_unet(names("UNETLoader", "unet_name"))
        all_sam = names("SAMLoader", "model_name")
        # Filtra:
        # 1) Solo file nella root di sams/ (esclude DreamO/BEN2 ecc.)
        # 2) Esclude SAM 2 (sam2_*) — non sono compatibili con SAMLoader di Impact-Pack
        #    che usa `segment_anything.sam_model_registry` (Meta SAM v1). I file SAM 2 hanno
        #    formato `{"model": ...}` → load_state_dict fallisce con "Unexpected key 'model'".
        #    SAM 2 va usato tramite ComfyUI-segment-anything-2 (kijai), già installato:
        #    è coperto dal flusso GroundingDINO+SAM-HQ, non da SAMLoader/Impact.
        def _is_sam_v1(n: str) -> bool:
            nl = (n or "").lower()
            if not nl:
                return False
            if nl.startswith("sam2") or nl.startswith("sam_2") or "sam2_" in nl or "sam_2_" in nl:
                return False
            # accetta esplicitamente i prefissi noti di SAM v1 / SAM-HQ
            return nl.startswith(("sam_vit", "sam_hq_", "mobile_sam"))
        self._sam_models = [m for m in all_sam
                            if "/" not in m and "\\" not in m and _is_sam_v1(m)]
        self.frame1_tab.set_sam_models(self._sam_models)
        upscale_models = names("UpscaleModelLoader", "model_name")
        self.frame1_tab.set_upscale_models(upscale_models)
        self.video_def_tab.set_upscale_models(upscale_models)
        self.restore_tab.set_upscale_models(upscale_models)
        # Restauro volti: popola la dropdown con i modelli REALMENTE presenti (solo quelli scaricati,
        # es. codeformer) → niente più selezione di un GFPGAN assente che fa fallire ComfyUI.
        fr_models = names("FaceRestoreModelLoader", "model_name")
        if fr_models and hasattr(self.video_tab, "set_facerestore_models"):
            self.video_tab.set_facerestore_models(fr_models)
        # Modelli FLUX.2 per colorizzazione (da diffusion_models)
        diff_models = names("UNETLoader", "unet_name")
        self.restore_tab.set_flux2_models(diff_models)
        # Risoluzione encoder/VAE/clip-type per Z-Image ed Ernie (gen immagini hand-built)
        _tes = names("CLIPLoader", "clip_name")
        _vaes = names("VAELoader", "vae_name")
        _ctypes = names("CLIPLoader", "type")
        self.genpipe.set_inventory(text_encoders=_tes, vaes=_vaes, clip_types=_ctypes)
        # Popola anche i dropdown UI della sezione «Genera immagine»
        if hasattr(self.frame1_tab, "set_gen_inventory"):
            self.frame1_tab.set_gen_inventory(text_encoders=_tes, vaes=_vaes, clip_types=_ctypes)
        # Disponibilità encoder VIDEO (gemma full / abliterated) nel gruppo Sampler/Seed
        ss = getattr(self.video_tab, "sampler_seed", None)
        if ss is not None and hasattr(ss, "set_encoder_availability"):
            ss.set_encoder_availability(_tes)

    def _video_gguf_models(self, oi: dict) -> list[str]:
        """Raccoglie i loader GGUF supportati dal workflow video, se presenti in ComfyUI."""
        out: list[str] = []
        for cls, inp in (("GGUFLoaderKJ", "model_name"), ("UnetLoaderGGUF", "unet_name")):
            spec = oi.get(cls, {}).get("input", {}).get("required", {}).get(inp)
            if not isinstance(spec, list) or not spec:
                continue
            head = spec[0]
            options = head if isinstance(head, list) else (
                spec[1].get("options", []) if head == "COMBO" and len(spec) > 1
                and isinstance(spec[1], dict) else [])
            for name in options:
                if name not in out:
                    out.append(name)
        return out

    def _log(self, msg: str):
        self.log.appendPlainText(msg)

    def _open_analyzer(self):
        """Analizza le impostazioni correnti e mostra il dialog con i risultati."""
        from core.settings_analyzer import SettingsAnalyzer
        from ui.settings_analyzer_dialog import SettingsAnalyzerDialog
        vals = {}
        vals.update(self.video_tab.get_values())
        vals.update(self.video_tab.get_sampler_values())
        loras = self.lora_tab.selected_loras()
        analyzer = SettingsAnalyzer()
        result = analyzer.analyze(
            steps=int(vals.get("steps", 20)),
            cfg=float(vals.get("cfg", 3.0)),
            width=int(vals.get("width", 704)),
            height=int(vals.get("height", 480)),
            nsfw_mode=str(vals.get("nsfw_mode", "sfw")),
            model_name=str(vals.get("model", "")),
            loras=loras,
            sampler=str(vals.get("sampler", "")),
            scheduler=str(vals.get("scheduler", "")),
            audio_mode=str(self.audio_tab.get_audio_config().get("audio_mode", "native")),
        )
        dlg = SettingsAnalyzerDialog(result, autofix_callback=self._apply_analyzer_fixes, parent=self)
        dlg.exec()

    def _open_workflow(self):
        """Apre il dialog dei workflow predefiniti."""
        from ui.workflow_dialog import WorkflowDialog
        dlg = WorkflowDialog(apply_callback=self._apply_workflow_fields, parent=self)
        dlg.exec()

    def _apply_workflow_fields(self, fields: dict, workflow_name: str = ""):
        """Applica i campi di un workflow alla GUI (come _apply_reuse_fields ma con audio_mode)."""
        # Prima applica i parametri comuni (riusa _apply_reuse_fields)
        self._apply_reuse_fields(fields)
        # Poi gestisci audio_mode (speciale: imposta il radio del tab Audio)
        # audio_mode è già gestito da _apply_reuse_fields (chiamata sopra)
        if workflow_name:
            self._log(f"Workflow applicato: '{workflow_name}'")

    def _apply_analyzer_fixes(self, fixes: dict):
        """Applica le correzioni suggerite dall'analizzatore alla GUI."""
        v = self.video_tab
        ss = getattr(v, "sampler_seed", None)
        if "steps" in fixes and ss:
            try:
                ss.steps.setValue(int(fixes["steps"]))
            except Exception:
                pass
        if "cfg" in fixes and ss:
            try:
                ss.cfg.setValue(float(fixes["cfg"]))
            except Exception:
                pass
        if "width" in fixes:
            try:
                v.width.setValue(int(fixes["width"]))
            except Exception:
                pass
        if "height" in fixes:
            try:
                v.height.setValue(int(fixes["height"]))
            except Exception:
                pass
        self._log(f"[Analyzer] Applicati fix: {fixes}")

    # ---------- generazione ----------
    def on_generate(self):
        if not self.engine:
            QMessageBox.warning(self, "Template mancante", self._engine_err or "Golden prompt non caricato.")
            return
        vals = {}
        vals.update(self.frame1_tab.get_values())
        vals.update(self.video_tab.get_values())
        vals.update(self.video_tab.get_sampler_values())   # Steps/CFG/Sampler/Seed (ora sotto Video)
        if not vals.get("prompt_positive"):
            QMessageBox.warning(self, "Prompt vuoto", "Inserisci un prompt nel tab Prompt / Frame 1.")
            return
        if not self.video_tab.wants_fixed_seed():
            vals["seed"] = random.randint(0, 2**31)

        # ── V2V (ReTake / Talk): editing di un video sorgente ──
        v2v = self.video_tab.v2v_panel.get_v2v_config() if hasattr(self.video_tab, "v2v_panel") else {"v2v_mode": "normal"}
        self._v2v_keep_audio_src = None
        _mode = v2v.get("v2v_mode", "normal")
        if _mode != "normal":
            src = v2v.pop("v2v_video_path", None)
            if _mode == "talk" and not src:
                # TALK SENZA VIDEO → genera il parlato DALL'IMMAGINE (Frame 1) come I2V normale:
                # niente maschera V2V; voce dall'albero del tab Audio, dialogo dal prompt principale.
                if not self.frame1_tab.image_path:
                    QMessageBox.warning(self, "Talk", "Per Talk: fornisci un VIDEO da modificare, "
                                        "oppure un'IMMAGINE nel Frame 1 (genero il parlato dall'immagine).")
                    return
                self._log("[Talk] nessun video → genero dall'immagine (I2V + lip-sync da audio/prompt).")
                # NON impostiamo v2v_mode → prosegue il percorso I2V normale (use_image più sotto).
            else:
                errs = self.video_tab.v2v_panel.validation_errors()
                if errs:
                    QMessageBox.warning(self, f"Manca qualcosa per {_mode.upper()}",
                                        "Per generare in questa modalità:\n\n• " + "\n• ".join(errs))
                    return
                vname = self._upload_video(src) if src else None
                if not vname:
                    QMessageBox.warning(self, "Video sorgente", "Impossibile caricare il video sorgente in ComfyUI.")
                    return
                v2v["v2v_video"] = vname
                w, h = self._probe_resolution(src)
                if w and h:
                    vals["width"], vals["height"] = w, h
                dur = self._probe_duration(src)
                if dur > 0:
                    vals["length_s"] = dur
                if _mode == "talk":
                    # Talk DA VIDEO: la finestra di rigenerazione = TUTTO il video → l'INTERO dialogo
                    # viene sintetizzato (prima [0,5s] di default → seconda metà del testo persa).
                    v2v["v2v_start"] = 0.0
                    v2v["v2v_end"] = dur if dur > 0 else v2v.get("v2v_end", 5.0)
                if _mode == "retake" and not v2v.get("v2v_regen_audio"):
                    self._v2v_keep_audio_src = src
                vals.update(v2v)

        # ── VIDEO LUNGHI: auto-extend oltre la soglia ──
        # Se la durata richiesta supera la soglia (impostata in Avanzate), generiamo una
        # passata base = soglia, poi al completamento estendiamo automaticamente a spezzoni
        # fino alla durata target (lunghezza illimitata, VRAM costante).
        self._auto_extend = None
        try:
            lv = self.advanced_tab.long_video_config()
        except Exception:
            lv = {"auto": False}
        target_len = float(vals.get("length_s") or 0)
        if lv.get("auto") and target_len > lv["threshold"] + 0.5:
            self._auto_extend = {
                "target": target_len, "produced": lv["threshold"],
                "prompt": vals.get("prompt_positive", ""),
                "negative": vals.get("prompt_negative", ""),
                "seg_seconds": lv["seg_seconds"], "overlap": lv["overlap"],
                "context": lv["context"], "context_strength": lv["context_strength"],
            }
            vals["length_s"] = lv["threshold"]          # passata base limitata alla soglia
            self._log(f"[Video lungo] durata {target_len:.0f}s > soglia {lv['threshold']:.0f}s → "
                      f"passata base {lv['threshold']:.0f}s poi estensione automatica.")

        # ── RESTAURO VOLTI automatico: passaggio finale post-generazione (opzionale) ──
        # Al completamento del video (DOPO l'eventuale auto-extend) applichiamo il restauro
        # volti per-fotogramma sul file finale. Submit separato → nessun conflitto VRAM con LTX.
        self._auto_face_restore = None
        try:
            fr_cfg = self.video_tab.get_face_restore_auto()
        except Exception:
            fr_cfg = None
        if fr_cfg:
            from core.face_restore_pipeline import node_installed as _fr_installed
            if _fr_installed():
                self._auto_face_restore = fr_cfg
                self._log(f"[Restauro volti] attivo: verrà applicato a fine generazione "
                          f"(fidelity {fr_cfg['fidelity']:.2f}, {fr_cfg['model']}).")
            else:
                QMessageBox.information(self, "Restauro volti non disponibile",
                    "Il restauro volti automatico è attivo ma il custom node facerestore_cf non è "
                    "installato. Installalo da «Migliora qualità → Video» (poi riavvia ComfyUI). "
                    "Procedo SENZA restauro volti per questa generazione.")
                self._log("[Restauro volti] nodo non installato → salto il passaggio.")

        # aggiorna stato per i conflitti (audio, controlnet, modello, nsfw)
        audio = self.audio_tab.get_audio_config()
        # CONSENSO TTS: se è stato generato un file TTS qui (audio_tab) ma l'utente NON
        # ha esplicitamente impostato la modalità "user", chiedo se vuole usarlo come
        # traccia. Se rifiuta, resta la modalità corrente (es. native → audio dal prompt).
        pending_tts = self.audio_tab.has_pending_tts() if hasattr(self.audio_tab, "has_pending_tts") else None
        if pending_tts and audio.get("audio_mode") != "user":
            ans = QMessageBox.question(self, "Audio TTS disponibile",
                f"È stato generato un audio TTS:\n\n  {Path(pending_tts).name}\n\n"
                "Vuoi usarlo come traccia audio del video?\n"
                "(annulla = usa l'audio del prompt — modalità nativa)",
                QMessageBox.Yes | QMessageBox.No)
            if ans == QMessageBox.Yes:
                self.audio_tab.apply_tts_as_user_audio()
                audio = self.audio_tab.get_audio_config()      # ricarica con audio_mode=user
                self._log(f"[TTS] uso il file generato come audio: {Path(pending_tts).name}")
        # ── Clona voce con TESTO → sintesi TTS adesso, poi usa come traccia utente ──
        # Se l'utente ha scelto «Clona voce» E ha scritto un testo nel TTS, sintetizziamo
        # l'audio con la voce clonata (CosyVoice ecc.) e lo usiamo come traccia (user audio).
        # Se il testo è vuoto, resta clone (solo il campione → LTXVReferenceAudio).
        if audio.get("audio_mode") == "clone" and (audio.get("tts_text") or "").strip() \
                and audio.get("clone_voice") is not None:
            try:
                from core.tts_engine import synthesize as _tts_synth
                self._log(f"[TTS] sintesi voce clonata '{audio['clone_voice'].name}' "
                          f"({audio.get('tts_engine')})…")
                out = _tts_synth(audio.get("tts_engine") or "edge",
                                 audio["clone_voice"], audio["tts_text"].strip())
                audio["audio_mode"] = "user"; audio["audio_file"] = str(out)
                audio["audio_user"] = True; audio["audio_clone"] = False
                self._log(f"[TTS] audio clonato pronto: {Path(str(out)).name} → traccia del video")
            except Exception as e:
                QMessageBox.warning(self, "Sintesi TTS fallita",
                    f"Non sono riuscito a sintetizzare la voce clonata:\n{e}\n\n"
                    "Procedo in modalità riferimento (il modello genera il parlato dal prompt "
                    "usando solo il timbro del campione).")
                self._log(f"[warn] sintesi TTS clone fallita: {e} → fallback a riferimento")
                # resta clone: usa solo il campione
        # Fallback se l'utente ha selezionato un audio_mode che richiede un file ma non
        # l'ha caricato: trattare come "native" → pruna audio_user/voice nel workflow.
        # Senza questo, i nodi VHS_LoadAudioUpload (5098/5669) restano col path hardcoded
        # del golden template (un mp3 inesistente in ComfyUI/input/) → validation fail.
        if audio.get("audio_mode") == "user" and not audio.get("audio_file"):
            self._log("[warn] audio_mode='user' senza file caricato → fallback a 'native'. "
                      "Carica un file nel tab Audio per usare un audio personalizzato.")
            audio["audio_mode"] = "native"; audio["audio_user"] = False
            audio["audio_native"] = True
        if audio.get("audio_mode") == "clone" and not audio.get("voice_sample"):
            self._log("[warn] audio_mode='clone' senza campione voce → fallback a 'native'. "
                      "Scegli/crea nel tab Audio una voce con campione per il voice cloning.")
            audio["audio_mode"] = "native"; audio["audio_clone"] = False
            audio["audio_native"] = True
        # ── EROS + audio nativo = RONZIO: il finetune NSFW eros NON è addestrato sull'audio →
        # l'audio "nativo" (generato da LTX) esce come ronzio. Avvisa (una volta) e suggerisci. ──
        _uses_eros = (vals.get("model_kind") == "eros"
                      or (vals.get("nsfw_mode") == "nsfw" and vals.get("model_kind") == "auto"))
        if _uses_eros and audio.get("audio_mode") == "native" and not audio.get("audio_mute"):
            self._log("[Audio] ⚠ eros non genera audio (solo ronzio): usa «Clona voce»/«Audio fornito "
                      "da me», o cambia modello (dev-fp8/distilled), o «Muto».")
            if not getattr(self, "_warned_eros_audio", False):
                self._warned_eros_audio = True
                QMessageBox.warning(self, "Audio con eros",
                    "Il modello eros (finetune NSFW) NON è addestrato sull'audio → l'audio «nativo» "
                    "esce come RONZIO.\n\nPer avere audio:\n"
                    "• tab Audio → «Clona voce» (TTS) o «Audio fornito da me» (file): la voce non la "
                    "genera eros;\n"
                    "• oppure cambia modello (dev-fp8 / distilled 1.1);\n"
                    "• oppure «Muto».\n\nProcedo comunque con questa generazione.")
        # ── VIDEO LUNGO + voce: la voce (TTS/file) è UNA traccia lunga → affettala per segmento ──
        # Ogni spezzone usa la sua FINESTRA temporale della voce (il dialogo è già tolto dal prompt
        # degli extend; LTX fa solo visivo+ambiente). Risolve "5 minuti di parlato nei 10s della base".
        if self._auto_extend is not None and audio.get("audio_mode") == "user" and audio.get("audio_file"):
            self._auto_extend["tts_full"] = audio["audio_file"]
            _base = self._slice_audio(audio["audio_file"], 0.0, float(vals.get("length_s", 10.0)))
            if _base:
                audio["audio_file"] = str(Path(r"E:\AI\Comfy_Nexus\ComfyUI\input") / _base)
                self._log("[Video lungo] voce lunga → affetto il TTS per segmento (base + extend).")
        vals["audio_mode"] = audio["audio_mode"]      # → collapse_audio_selector in build_prompt
        # LTX-2.3 (audio-video) tende a CANTARE il dialogo del prompt invece di pronunciarlo.
        # Per far PARLARE ma non CANTARE: sopprimo solo il canto nel negativo (NON tocco
        # talking/speaking/lip-sync, così il parlato e il labiale restano).
        if audio.get("suppress_singing"):
            anti = "singing, song, music, musical, melody, humming, chanting, opera, lyrics, musical performance"
            neg = vals.get("prompt_negative", "")
            if "singing" not in neg:
                vals["prompt_negative"] = (neg + ", " + anti).strip(", ") if neg else anti
        cnet = self.controlnet_tab.get_values()
        self.state.update({
            "nsfw_mode": vals.get("nsfw_mode", "sfw"),
            "audio_mute": audio.get("audio_mute", False),
            "audio_native": audio["audio_native"], "audio_user": audio["audio_user"],
            "audio_clone": audio["audio_clone"], "audio_mode": audio["audio_mode"],
            "voice_sample": audio.get("voice_sample"),
            "controlnet_enabled": cnet["controlnet_enabled"],
            "controlnet_ref_video": cnet.get("controlnet_ref_video"),
        }, auto_apply_implies=True)
        blocking = [c for c in self.state.evaluate(auto_apply_implies=False) if c.severity == "blocking"]
        if blocking:
            QMessageBox.warning(self, "Conflitti da risolvere", "\n".join("• " + c.message for c in blocking))
            return

        # LoRA + trigger word
        loras = [{"name": l["name"], "strength": l["strength"]} for l in self.lora_tab.selected_loras()]
        if self.lora_tab.auto_insert_triggers():
            # opzione attiva: inserisci automaticamente i trigger mancanti, senza chiedere
            trigs = self.lora_tab.missing_trigger_words(vals["prompt_positive"])
            if trigs:
                new_prompt = (vals["prompt_positive"] + ", " + ", ".join(trigs)).strip(", ")
                self.frame1_tab.prompt.setPlainText(new_prompt)
                vals["prompt_positive"] = new_prompt
                self._log(f"Trigger word inserite nel prompt: {', '.join(trigs)}")
        else:
            warns = self.lora_tab.trigger_warnings(vals["prompt_positive"])
            if warns:
                box = QMessageBox(self)
                box.setWindowTitle("Trigger word mancanti")
                box.setText("Alcune LoRA hanno trigger word non presenti nel prompt:\n\n" + "\n".join(warns))
                b_ins = box.addButton("Inserisci e genera", QMessageBox.AcceptRole)
                b_gen = box.addButton("Genera comunque", QMessageBox.YesRole)
                box.addButton("Annulla", QMessageBox.RejectRole)
                box.exec()
                clicked = box.clickedButton()
                if clicked == b_ins:
                    trigs = self.lora_tab.missing_trigger_words(vals["prompt_positive"])
                    if trigs:
                        new_prompt = (vals["prompt_positive"] + ", " + ", ".join(trigs)).strip(", ")
                        self.frame1_tab.prompt.setPlainText(new_prompt)
                        vals["prompt_positive"] = new_prompt
                elif clicked != b_gen:
                    return
        # seleziona il template in base ai moduli attivi (i golden specifici se presenti, altrimenti superset)
        use_image = bool(self.frame1_tab.image_path)
        mode = self._pick_template(use_image, audio["audio_mode"], cnet["controlnet_enabled"],
                                   vals.get("model_kind", "auto"))
        self.engine.use_template(mode)
        self._log(f"Template: {self.engine.active_template_name}")

        # upload input moduli + set binding (immagine canvas / video ctrl / voce)
        try:
            self._prepare_module_inputs(vals, use_image, audio, cnet)
        except Exception as e:
            self._log(f"[warn] preparazione input moduli: {e}")

        # TEXT2VIDEO: azzera ref_strength (→ nodo 5647, forza del keyframe1) così l'immagine bakeata
        # di default del template ("magicquill (4).png") viene RIGENERATA dal rumore invece di
        # apparire all'inizio del video. In I2V resta la forza scelta dall'utente.
        if not use_image:
            vals["ref_strength"] = 0.0

        try:
            prompt = self.engine.build_prompt(vals, loras=loras, overrides=self._overrides)
            # pruning: se siamo sul superset, disattiva i moduli NON richiesti
            if self.engine.active_template_name == "superset":
                # prompt enhancer: opzionale via toggle (default OFF → prompt grezzo).
                # Se ON, NON lo disattiviamo (l'LLM espande il prompt; la direttiva NSFW è
                # iniettata in build_prompt). Se OFF, lo pruniamo (evita input mancanti).
                disable = [] if vals.get("prompt_enhance") else ["prompt_enhancer", "prompt_relay"]
                if not cnet["controlnet_enabled"]:
                    disable.append("controlnet")
                if audio["audio_mode"] != "clone":
                    disable.append("voice")          # nodo voce ID-LoRA (LTXVReferenceAudio)
                if audio["audio_mode"] != "user":
                    disable.append("audio_user")     # sorgente audio caricata dall'utente
                # keyframe extra 8..2 → disattivati. keyframe1 NON è un modulo disattivabile
                # (la sua neutralizzazione in text2video avviene azzerando ref_strength sotto).
                disable += [f"keyframe{i}" for i in range(8, 1, -1)]
                # LoRA distilled OFF per qualità: la distilled-dynamic (modulo "distilled", nodi
                # 6428/6429) è una distillazione per pochi step che AMMORBIDISCE il dettaglio fine
                # dei frame generati. La pruniamo sempre → il modello base gira pulito (l'utente non
                # ha problemi di velocità). Riattivabile rendendola un toggle se servirà una modalità veloce.
                disable.append("distilled")
                prompt, warns = self.engine.disable_modules(prompt, disable)
                for wmsg in warns:
                    self._log("[prune] " + wmsg)
                self._log(f"[prune] moduli disattivati richiesti: {disable} → {len(prompt)} nodi")
            # ricorda se questa generazione è NATIVA (no downscale) → l'anti-moiré post serve solo lì
            self._last_gen_native = not vals.get("downscale_half", False)
            # Video lunghi: cattura il contesto per rigenerare i segmenti I2V con la pipeline base
            if self._auto_extend is not None:
                import copy as _copy
                self._auto_extend.update({
                    "base_vals": _copy.deepcopy(vals), "loras": loras,
                    "audio": audio, "cnet": cnet, "fps": int(vals.get("fps", 24)),
                })
            self._current_pid = self.driver.submit(prompt)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            # Cache dei parametri per il sidecar JSON (salvato a completamento)
            self._generation_cache[self._current_pid] = MetadataSaver.build_meta_from_vals(
                vals, loras, audio, cnet,
                template_name=self.engine.active_template_name,
                prompt_id=self._current_pid,
                image_path=self.frame1_tab.image_path,
            )
            self._log(f"Inviato. prompt_id={self._current_pid}  ({len(prompt)} nodi)")
            self.progress.setValue(0)
            self.gen_dialog.start(prompt, "Generazione video")
        except Exception as e:
            QMessageBox.critical(self, "Errore invio", str(e))
            self._log(f"[ERR] {e}")

    def _pick_template(self, use_image: bool, audio_mode: str, cnet_enabled: bool,
                       model_kind: str = "auto") -> str:
        from core.template_engine import TemplateEngine
        avail = set(TemplateEngine.available_templates().keys())
        active = []
        if use_image: active.append("image")
        if cnet_enabled: active.append("controlnet")
        if audio_mode == "clone": active.append("voice")
        if audio_mode == "user": active.append("user_audio")
        # GGUF (RAM bassa): usa il golden ridotto in T2V, oppure sostituisce il loader nel superset.
        if model_kind == "gguf" and "gguf" in avail:
            if active:
                self._log("[i] GGUF con moduli attivi: uso il superset sostituendone il loader modello.")
            else:
                return "gguf"
        if not active:
            return "t2v" if "t2v" in avail else "superset"
        if len(active) == 1:
            cand = {"image": "i2v", "controlnet": "controlnet",
                    "voice": "voice", "user_audio": "audio_user"}.get(active[0])
            if cand in avail:
                return cand
        return "superset" if "superset" in avail else ("t2v" if "t2v" in avail else "superset")

    def _prepare_module_inputs(self, vals: dict, use_image: bool, audio: dict, cnet: dict):
        """Esporta/carica gli input dei moduli su ComfyUI e imposta i binding."""
        def upload(path):
            r = self.driver.upload_image(path)
            name = r.get("name", "")
            sub = r.get("subfolder", "")
            return f"{sub}/{name}" if sub else name
        # immagine iniziale dal canvas (con maschera per inpaint, se presente)
        if use_image:
            exp = self.frame1_tab.canvas.export(OUTPUT_DIR)
            up = upload(exp["image"])
            vals["ref_image_file"] = up
            if exp.get("has_mask") and exp.get("mask"):
                try:
                    self.driver.upload_mask(exp["mask"], up.split("/")[-1], up.rsplit("/", 1)[0] if "/" in up else "", "input")
                except Exception as e:
                    self._log(f"[warn] upload maschera: {e}")
        # ControlNet
        if cnet.get("controlnet_enabled") and cnet.get("controlnet_ref_video"):
            vals["control_video"] = upload(cnet["controlnet_ref_video"])
            # Mappa UI index → ImpactSwitch.select del workflow (5564):
            #   input1 = AIO/Depth (DepthAnythingV2)  → select=1
            #   input2 = Canny                        → select=2
            #   input3 = DWPose                       → select=3
            ui_to_select = {0: 2, 1: 1, 2: 3}     # UI: 0 canny, 1 depth, 2 pose
            vals["controlnet_type"] = ui_to_select.get(int(cnet.get("controlnet_type", 0)), 2)
        # voce / audio utente
        # CLONE: il sample vocale serve SOLO come riferimento timbrico per LTXVReferenceAudio
        # (5669), che condiziona LTX-2.3 a generare l'audio del personaggio con quella voce.
        # Il latent audio del concat AV deve restare vuoto (LTXVEmptyLatentAudio, 5652) →
        # il modello sintetizza l'audio del parlato direttamente dal prompt testuale.
        # USER: l'utente fornisce la traccia audio già pronta (5098 → encode → 5089) e il
        # modello la usa come guida temporale/contenutistica del parlato.
        if audio.get("audio_mode") == "clone" and audio.get("voice_sample"):
            vals["voice_sample_file"] = upload(audio["voice_sample"])
        elif audio.get("audio_mode") == "user" and audio.get("audio_file"):
            vals["user_audio_file"] = upload(audio["audio_file"])

    # ---------- inpaint / outpaint (genera la foto PRIMA del video) ----------
    def _upload_local(self, path: str) -> str:
        r = self.driver.upload_image(path)
        name, sub = r.get("name", ""), r.get("subfolder", "")
        return f"{sub}/{name}" if sub else name

    def generate_inpaint(self, mode: str):
        if not self.driver.is_up():
            QMessageBox.warning(self, "ComfyUI offline", "Avvia ComfyUI per generare."); return
        if not self.frame1_tab.image_path:
            QMessageBox.warning(self, "Nessuna immagine", "Carica una foto nel canvas."); return
        if self._inpaint_pid:
            QMessageBox.information(self, "Attendi", "Generazione immagine già in corso."); return
        cfg = self.frame1_tab.get_inpaint_config()
        if cfg.get("unet") and not cfg.get("available", True):
            QMessageBox.warning(self, "Modello non installato",
                                f"Il modello inpaint selezionato non è ancora presente in ComfyUI:\n\n"
                                f"{cfg.get('unet')}\n\n"
                                f"Se è in download, attendi il termine e l'ingest; poi riavvia ComfyUI.")
            return
        ptxt = (cfg["inpaint_prompt"] if mode == "inpaint" else cfg["outpaint_prompt"]) \
            or self.frame1_tab.get_values().get("prompt_positive", "")
        retention = cfg.get("strength", 0.75)
        denoise = 1.0 if mode == "outpaint" else max(0.5, min(1.0, 1.0 - 0.4 * retention))
        # guard: serve un'area da rigenerare, altrimenti il modello non fa nulla
        if mode == "outpaint" and not self.frame1_tab.canvas.has_outpaint_area():
            QMessageBox.information(self, "Niente da estendere",
                                    "La foto riempie già tutto il frame: non c'è bordo da generare.\n\n"
                                    "Riduci lo zoom della foto (rotella in modalità ✋ Sposta) per lasciare "
                                    "spazio vuoto attorno, poi rilancia l'outpaint."); return
        if mode == "inpaint" and not self.frame1_tab.canvas.has_mask():
            QMessageBox.information(self, "Nessuna maschera",
                                    "Disegna una maschera (✏ Maschera / 🪣 Fill / lazzo / selezione oggetto) "
                                    "sull'area da rigenerare prima dell'inpaint."); return
        try:
            exp = self.frame1_tab.canvas.export_inpaint(OUTPUT_DIR, mode)
            self._inpaint_bbox = exp.get("bbox")    # per rimettere l'inpaint al suo posto
            self._inpaint_before = exp.get("image") # per confronto prima/dopo nel dialog
            img_name = self._upload_local(exp["image"])
            mask_name = self._upload_local(exp["mask"])
            seed = cfg["seed"] if cfg.get("seed") is not None else random.randint(0, 2**31)
            steps = cfg.get("steps") or 20
            negative = cfg.get("negative") or self.frame1_tab.get_values().get("prompt_negative", "")
            # feather/overlap della maschera per fondere la giunzione (no linee di separazione).
            # Outpaint: la crescita DEVE superare la sfumatura, altrimenti la sfumatura sborda nel
            # bordo lasciando una fascia scura non riempita (distacco). grow(40) > feather(24) →
            # bordo pieno fino alla foto, transizione tutta dentro la foto (overlap).
            feather, grow = (24, 40) if mode == "outpaint" else (12, 0)
            prompt = self.inpainter.build(
                img_name, ptxt, mode=mode, mask_file=mask_name, denoise=denoise,
                seed=seed, lora=cfg.get("lora"), feather=feather, mask_grow=grow,
                unet=cfg.get("unet"), checkpoint=cfg.get("checkpoint"),
                steps=steps, negative=negative)
            self._inpaint_pid = self.driver.submit(prompt)
            self._inpaint_mode = mode
            self.gen_dialog.start(prompt, "Outpainting" if mode == "outpaint" else "Inpainting")
            self._log(f"{mode}: inviato (id={self._inpaint_pid}, steps={steps}, seed={seed}, denoise={denoise:.2f}).")
        except Exception as e:
            QMessageBox.critical(self, f"Errore {mode}", str(e)); self._log(f"[ERR] {mode}: {e}")

    def generate_integrate(self):
        """Integra l'oggetto incollato nella scena con FLUX **Kontext** (o FLUX.2): il composito
        (scena + prop) è dato come RIFERIMENTO + prompt → il modello RIGENERA/armonizza (luce,
        ombre, prospettiva), non incolla l'originale. Prompt opzionale (default auto)."""
        if not self.driver.is_up():
            QMessageBox.warning(self, "ComfyUI offline", "Avvia ComfyUI per generare."); return
        if not self.frame1_tab.image_path:
            QMessageBox.warning(self, "Nessuna immagine", "Carica una foto base nel canvas."); return
        if self._inpaint_pid:
            QMessageBox.information(self, "Attendi", "Generazione già in corso."); return
        if not self.frame1_tab.canvas.has_layers():
            QMessageBox.information(self, "Nessuna immagine da integrare",
                                    "Aggiungi una o più immagini con «+ Img» e posizionale, "
                                    "poi premi «🧩 Integra immagini»."); return
        cfg = self.frame1_tab.get_inpaint_config()
        engine = cfg.get("integrate_engine", "kontext")
        # Modello deciso dall'utente nel combo (contestuale all'engine selezionato).
        # Sia Kontext che FLUX.2 leggono dallo stesso campo `integrate_model`. Se vuoto,
        # KontextPipeline ricade sul default hardcoded `FLUX_Kontext_Dev.safetensors`.
        model = cfg.get("integrate_model")
        if engine == "flux2" and not model:
            QMessageBox.warning(self, "Modello FLUX.2 mancante",
                                "Per il motore FLUX.2 scegli un modello FLUX.2 nel menu integra."); return
        ptxt = cfg.get("integrate_prompt") or ""        # opzionale: default auto in KontextPipeline
        try:
            exp = self.frame1_tab.canvas.export_integrate(OUTPUT_DIR)
            if not exp.get("has_region"):
                QMessageBox.information(self, "Niente da integrare",
                                        "Le immagini non hanno né maschera né area valida nel frame."); return
            self._inpaint_bbox = None
            self._inpaint_before = exp.get("image")     # confronto prima/dopo (composito di partenza)
            img_name = self._upload_local(exp["image"])
            # Campi dedicati alla sezione Integra (separati da quelli di inpaint/outpaint).
            int_seed_cfg = cfg.get("integrate_seed")
            seed = int_seed_cfg if int_seed_cfg is not None else random.randint(0, 2**31)
            steps = cfg.get("integrate_steps") or 20
            negative = cfg.get("integrate_negative") or ""
            aux = cfg.get("integrate_puzzle_lora") or None    # puzzle_lora opzionale (solo Kontext)
            prompt = self.kontext.integrate(img_name, ptxt, engine=engine, model=model,
                                            negative=negative, seed=seed, steps=steps, aux_lora=aux)
            self._inpaint_pid = self.driver.submit(prompt)
            self._inpaint_mode = "integrazione"
            self.gen_dialog.start(prompt, f"Integrazione ({'FLUX.2' if engine == 'flux2' else 'Kontext'})")
            self._log(f"Integrazione {engine}: inviata (steps={steps}, seed={seed}, puzzle_lora={'sì' if aux else 'no'}).")
        except Exception as e:
            QMessageBox.critical(self, "Errore integrazione", str(e)); self._log(f"[ERR] integrate: {e}")

    def generate_recolor(self):
        """Cambia colore (Kontext/FLUX.2 prompt-based): ricolora il soggetto mantenendo texture/luce."""
        if not self.driver.is_up():
            QMessageBox.warning(self, "ComfyUI offline", "Avvia ComfyUI per generare."); return
        if not self.frame1_tab.image_path:
            QMessageBox.warning(self, "Nessuna immagine", "Carica o genera un'immagine nel canvas."); return
        if self._inpaint_pid:
            QMessageBox.information(self, "Attendi", "Generazione già in corso."); return
        cc = self.frame1_tab.get_color_config()
        if not cc["target"]:
            QMessageBox.warning(self, "Colore mancante",
                                "Indica il colore di destinazione (es. «rosso acceso» o #cc0000)."); return
        engine = cc["engine"]; model = None
        if engine == "flux2":
            model = self.frame1_tab.get_inpaint_config().get("integrate_model")
            if not model:
                QMessageBox.warning(self, "Modello FLUX.2 mancante",
                                    "Per il motore FLUX.2 scegli un modello nel tab Inpaint/Outpaint."); return
        try:
            img_name = self._upload_local(self.frame1_tab.image_path)
            self._inpaint_bbox = None
            self._inpaint_before = self.frame1_tab.image_path     # confronto prima/dopo
            seed = random.randint(0, 2**31)
            prompt = self.kontext.recolor(img_name, cc["target"], subject=cc["subject"],
                                          engine=engine, model=model, steps=cc["steps"], seed=seed)
            self._inpaint_pid = self.driver.submit(prompt)
            self._inpaint_mode = "cambio colore"
            self.gen_dialog.start(prompt, f"Cambio colore ({'FLUX.2' if engine == 'flux2' else 'Kontext'})")
            self._log(f"Cambio colore: {cc['subject'] or 'soggetto'} → {cc['target']} ({engine}).")
        except Exception as e:
            QMessageBox.critical(self, "Errore cambio colore", str(e)); self._log(f"[ERR] recolor: {e}")

    def generate_video_enhance(self):
        """Definizione video: restore + super-risoluzione (+ opz. FPS) sull'ultimo video generato."""
        import shutil
        if not self.driver.is_up():
            QMessageBox.warning(self, "ComfyUI offline", "Avvia ComfyUI per migliorare il video."); return
        cfg = self.video_def_tab.get_config()
        if not cfg.get("video"):
            QMessageBox.warning(self, "Nessun video", "Genera un video o caricane uno nel tab."); return
        if self._current_pid or self._inpaint_pid:
            QMessageBox.information(self, "Attendi", "Generazione già in corso."); return
        src = Path(cfg["video"])
        if not src.exists():
            QMessageBox.warning(self, "File mancante", f"Video non trovato: {src}"); return
        try:
            comfy_input = Path(r"E:\AI\Comfy_Nexus\ComfyUI\input"); comfy_input.mkdir(parents=True, exist_ok=True)
            dest = comfy_input / src.name
            if str(dest) != str(src):
                shutil.copy2(src, dest)        # VHS_LoadVideo carica per nome dalla input dir
            fps = self.video_tab.get_values().get("fps", 24)
            prompt = self.enhancer.build(
                src.name, engine=cfg["engine"], seedvr2_model=cfg["seedvr2_model"],
                resolution=cfg["resolution"], blocks_to_swap=cfg["blocks_to_swap"],
                batch_size=cfg["batch_size"], esrgan_model=cfg["esrgan_model"],
                target_scale=cfg["target_scale"], fps=int(fps), rife=cfg["rife"],
                rife_multiplier=cfg["rife_multiplier"], rife_ckpt=cfg["rife_ckpt"],
                seed=random.randint(0, 2**31))
            self._current_pid = self.driver.submit(prompt)
            self.gen_dialog.start(prompt, f"Definizione video ({cfg['engine']})")
            self._log(f"Definizione video: {cfg['engine']} su {src.name} ({len(prompt)} nodi). "
                      f"Il modello SeedVR2 può scaricarsi al primo uso.")
        except Exception as e:
            QMessageBox.critical(self, "Errore definizione video", str(e)); self._log(f"[ERR] enhance: {e}")

    def generate_video_extend(self):
        """Prolunga video: estensione LTX-2.3 nativa multi-segmento sull'ultimo video (o caricato)."""
        import shutil
        if not self.driver.is_up():
            QMessageBox.warning(self, "ComfyUI offline", "Avvia ComfyUI per prolungare il video."); return
        cfg = self.video_extend_tab.get_config()
        if not cfg.get("video"):
            QMessageBox.warning(self, "Nessun video", "Genera un video o caricane uno nel tab."); return
        if self._current_pid or self._inpaint_pid:
            QMessageBox.information(self, "Attendi", "Generazione già in corso."); return
        src = Path(cfg["video"])
        if not src.exists():
            QMessageBox.warning(self, "File mancante", f"Video non trovato: {src}"); return
        if not cfg["prompt"]:
            QMessageBox.warning(self, "Prompt mancante", "Scrivi come prosegue la scena."); return
        try:
            comfy_input = Path(r"E:\AI\Comfy_Nexus\ComfyUI\input"); comfy_input.mkdir(parents=True, exist_ok=True)
            dest = comfy_input / src.name
            if str(dest) != str(src):
                shutil.copy2(src, dest)
            v = self.video_tab.get_values()
            n_seg = self.extender.segment_count(cfg["add_seconds"], cfg.get("seg_seconds", 4.0))
            # forza contesto: override dall'auto-extend, altrimenti dai settings Avanzate
            ctx_strength = getattr(self, "_auto_extend_ctx_strength", None)
            if ctx_strength is None:
                try:
                    ctx_strength = self.advanced_tab.long_video_config().get("context_strength", 0.5)
                except Exception:
                    ctx_strength = 0.5
            self._auto_extend_ctx_strength = None      # consuma l'override una sola volta
            # EREDITA i parametri di qualità dalla passata base (stesso modello/sampler/step/cfg/
            # encoder/shift) → l'estensione resta nello stesso spazio visivo (no rumore/incoerenza).
            sv = self.video_tab.get_sampler_values()
            ext_model = v.get("model_standard")        # modello diffusion attivo (None se gguf/auto → fallback distilled)
            prompt = self.extender.build(
                src.name, cfg["prompt"], width=v["width"], height=v["height"], fps=int(v.get("fps", 24)),
                add_seconds=cfg["add_seconds"], seg_seconds=cfg.get("seg_seconds", 4.0),
                overlap_frames=cfg["overlap_frames"], context_frames=cfg["context_frames"],
                overlap_strength=float(ctx_strength),
                steps=int(sv.get("steps") or cfg["steps"]), cfg=float(sv.get("cfg") or 1.0),
                seed=random.randint(0, 2**31), negative=v.get("prompt_negative", ""),
                unet=ext_model, sampler=sv.get("sampler") or "euler_ancestral_cfg_pp",
                base_shift=float(sv.get("ltxv_base_shift", 0.95)),
                max_shift=float(sv.get("ltxv_max_shift", 2.05)),
                clip1=sv.get("text_encoder"))
            self._current_pid = self.driver.submit(prompt)
            self.gen_dialog.start(prompt, "Prolunga video (LTX-2.3, illimitato)")
            self._log(f"Prolunga video: +{cfg['add_seconds']:.0f}s in {n_seg} spezzoni su {src.name} "
                      f"({len(prompt)} nodi).")
        except Exception as e:
            QMessageBox.critical(self, "Errore prolunga video", str(e)); self._log(f"[ERR] extend: {e}")

    def generate_face_restore(self):
        """Restauro volti per-frame (CodeFormer/GFPGAN) sul video del tab Migliora qualità."""
        import shutil
        if not self.driver.is_up():
            QMessageBox.warning(self, "ComfyUI offline", "Avvia ComfyUI per elaborare."); return
        if self._current_pid or self._inpaint_pid:
            QMessageBox.information(self, "Attendi", "Elaborazione già in corso."); return
        cfg = self.video_def_tab.get_face_restore_config()
        if not cfg.get("video"):
            QMessageBox.warning(self, "Nessun video", "Carica o genera un video."); return
        src = Path(cfg["video"])
        if not src.exists():
            QMessageBox.warning(self, "File mancante", f"Video non trovato: {src}"); return
        try:
            from core.face_restore_pipeline import node_installed, ensure_model
            if not node_installed():
                QMessageBox.warning(self, "Nodo mancante",
                    "Il custom node facerestore_cf non è installato. Usa «📥 Installa face-restore» "
                    "nel tab Migliora qualità → Video, poi riavvia ComfyUI."); return
            ok, msg = ensure_model(cfg["model"], log=self._log)
            if not ok:
                QMessageBox.warning(self, "Modello restauro mancante", msg); return
            comfy_input = Path(r"E:\AI\Comfy_Nexus\ComfyUI\input"); comfy_input.mkdir(parents=True, exist_ok=True)
            dest = comfy_input / src.name
            if str(dest) != str(src):
                shutil.copy2(src, dest)
            v = self.video_tab.get_values()
            prompt = self.facerestorer.build(
                src.name, fps=int(v.get("fps", 24)), fidelity=cfg["fidelity"],
                model=cfg["model"], detection=cfg["detection"])
            self._current_pid = self.driver.submit(prompt)
            self.gen_dialog.start(prompt, "Restauro volti (CodeFormer/GFPGAN)")
            self._log(f"Restauro volti su {src.name} (fidelity {cfg['fidelity']:.2f}, {cfg['model']}).")
        except Exception as e:
            QMessageBox.critical(self, "Errore restauro volti", str(e)); self._log(f"[ERR] face restore: {e}")

    def generate_image(self):
        if not self.driver.is_up():
            QMessageBox.warning(self, "ComfyUI offline", "Avvia ComfyUI per generare."); return
        if self._inpaint_pid:
            QMessageBox.information(self, "Attendi", "Generazione immagine già in corso."); return
        cfg = self.frame1_tab.get_gen_config()
        if not cfg["prompt"]:
            QMessageBox.warning(self, "Prompt vuoto", "Inserisci un prompt immagine o globale."); return
        if not cfg["model"] or cfg["model"].startswith("("):
            QMessageBox.warning(self, "Nessun modello", f"Nessun modello {cfg['arch']} trovato."); return
        v = self.video_tab.get_values()
        loras = [{"name": l["name"], "strength": l["strength"]} for l in cfg.get("loras", [])]
        # FLUX: distinguo checkpoint vs UNET (CheckpointLoaderSimple vs UNETLoader)
        ckpts = set(getattr(self.frame1_tab, "_gen_checkpoint_models", []) or [])
        is_checkpoint = cfg["arch"] == "flux" and cfg["model"] in ckpts
        # Seed dalla UI: None = random, altrimenti il valore fisso scelto dall'utente
        seed = cfg.get("seed") if cfg.get("seed") is not None else random.randint(0, 2**31)
        # Negativo dalla sezione gen; fallback al prompt negativo globale del tab Video
        negative = cfg.get("negative") or self.frame1_tab.get_values().get("prompt_negative", "")
        try:
            prompt = self.genpipe.build(
                cfg["arch"],
                cfg["model"] if not is_checkpoint else "",
                cfg["prompt"],
                width=v["width"], height=v["height"],
                seed=seed, steps=cfg["steps"],
                loras=loras,
                negative=negative,
                sampler=cfg.get("sampler"), scheduler=cfg.get("scheduler"),
                cfg=cfg.get("cfg"),
                encoder=cfg.get("encoder"), clip_type=cfg.get("clip_type"),
                vae=cfg.get("vae"),
                checkpoint=cfg["model"] if is_checkpoint else None,
                enhance=cfg.get("enhance", False), nsfw=cfg.get("nsfw", False),
            )
            self._inpaint_pid = self.driver.submit(prompt)
            self._inpaint_mode = "generazione immagine"
            self._inpaint_bbox = None
            self._inpaint_before = None      # da zero: nessun confronto
            self.gen_dialog.start(prompt, f"Generazione immagine ({cfg['arch']})")
            self._log(f"Generazione immagine ({cfg['arch']} · {cfg['model']}) inviata…")
        except Exception as e:
            QMessageBox.critical(self, "Errore generazione", str(e)); self._log(f"[ERR] gen: {e}")

    def generate_upscale(self):
        """Aumenta la definizione dell'immagine corrente del canvas con un modello AI di super-risoluzione."""
        if not self.driver.is_up():
            QMessageBox.warning(self, "ComfyUI offline", "Avvia ComfyUI per generare."); return
        if not self.frame1_tab.image_path:
            QMessageBox.warning(self, "Nessuna immagine", "Carica o genera un'immagine nel canvas."); return
        if self._inpaint_pid:
            QMessageBox.information(self, "Attendi", "Operazione già in corso."); return
        cfg = self.frame1_tab.get_upscale_config()
        if not cfg.get("model"):
            QMessageBox.warning(self, "Nessun modello upscale",
                                "Nessun modello di super-risoluzione in elenco. Riavvia l'app per "
                                "ricaricare i modelli (sono in E:\\AI\\Models\\Upscale)."); return
        if cfg.get("refine") and not cfg.get("refine_model"):
            QMessageBox.warning(self, "Modello rifinitura mancante",
                                "La rifinitura AI richiede un modello FLUX in diffusion_models. "
                                "Selezionane uno nel menu «Modello rifinitura»."); return
        self._inpaint_before = self.frame1_tab.image_path   # confronto prima/dopo
        try:
            img = self._upload_local(self.frame1_tab.image_path)
            rprompt = self.frame1_tab.get_values().get("prompt_positive", "")
            rneg = self.frame1_tab.get_values().get("prompt_negative", "")
            prompt = self.upscaler.build(
                img, cfg["model"], target_scale=cfg["scale"],
                refine=cfg.get("refine", False), refine_model=cfg.get("refine_model"),
                refine_denoise=cfg.get("refine_denoise", 0.3), refine_steps=cfg.get("refine_steps", 12),
                refine_prompt=rprompt, refine_negative=rneg, seed=random.randint(0, 2**31))
            self._inpaint_pid = self.driver.submit(prompt)
            self._inpaint_mode = "aumento definizione"
            self._inpaint_bbox = None
            extra = " + rifinitura AI" if cfg.get("refine") else ""
            self.gen_dialog.start(prompt, f"Aumento definizione ({cfg['scale']:.0f}x{extra})")
            self._log(f"Upscale {cfg['scale']:.0f}x con {cfg['model']}{extra} inviato…")
        except Exception as e:
            QMessageBox.critical(self, "Errore upscale", str(e)); self._log(f"[ERR] upscale: {e}")

    def _check_inpaint_done(self):
        try:
            hist = self.driver.history(self._inpaint_pid)
        except Exception:
            return
        entry = hist.get(self._inpaint_pid)
        if not entry:
            return
        st = entry.get("status", {})
        if st.get("status_str") == "error":
            self.gen_dialog.finish()
            self._log(f"[ERR] {self._inpaint_mode} fallito"); self._inpaint_pid = None; return
        if not st.get("completed"):
            return
        pid = self._inpaint_pid; self._inpaint_pid = None
        self.gen_dialog.finish()
        for out in entry.get("outputs", {}).values():
            for item in out.get("images", []) or []:
                fn = item.get("filename")
                if not fn:
                    continue
                try:
                    data = self.driver.download_view(fn, item.get("subfolder", ""), item.get("type", "output"))
                    rp = OUTPUT_DIR / f"result_{fn}"; rp.write_bytes(data)
                    self._show_inpaint_result(str(rp)); return
                except Exception as e:
                    self._log(f"[ERR] download risultato: {e}")

    def _show_inpaint_result(self, path: str):
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        dlg = QDialog(self); dlg.setWindowTitle(f"Risultato {self._inpaint_mode} — Accetta o Annulla")
        v = QVBoxLayout(dlg)
        before_pix = QPixmap(self._inpaint_before) if self._inpaint_before else None
        if before_pix is not None and before_pix.isNull():
            before_pix = None
        lbl = _HoldCompareLabel(QPixmap(path), before_pix)
        v.addWidget(lbl)
        if before_pix is not None:
            hint = QLabel("🖱️ Tieni premuto sull'immagine per vedere l'originale (confronto prima/dopo)")
            hint.setStyleSheet("color:#9aa1b0;"); hint.setAlignment(Qt.AlignCenter); v.addWidget(hint)
        bb = QDialogButtonBox()
        b_ok = bb.addButton("✔ Accetta (usa questa)", QDialogButtonBox.AcceptRole)
        bb.addButton("↩ Revert (scarta)", QDialogButtonBox.RejectRole)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if dlg.exec():
            if self._inpaint_mode == "inpaint" and self._inpaint_bbox:
                self.frame1_tab.canvas.load_inpaint_result(path, self._inpaint_bbox)  # rimette al suo posto, niente bordo
            else:
                self.frame1_tab.canvas.load_result(path)
            self._log(f"{self._inpaint_mode}: risultato accettato e caricato nel canvas.")
        else:
            self._log(f"{self._inpaint_mode}: risultato scartato (revert).")

    # ---------- selezione oggetto (SAM / CLIPSeg) ----------
    def _segment_preconds(self) -> bool:
        if not self.driver.is_up():
            QMessageBox.warning(self, "ComfyUI offline", "Avvia ComfyUI per segmentare."); return False
        if not self.frame1_tab.image_path:
            QMessageBox.warning(self, "Nessuna immagine", "Carica una foto nel canvas."); return False
        if self._seg_pid:
            QMessageBox.information(self, "Attendi", "Selezione in corso."); return False
        return True

    def segment_by_text(self):
        if not self._segment_preconds():
            return
        cfg = self.frame1_tab.get_segment_config()
        if not cfg.get("text"):
            QMessageBox.warning(self, "Testo mancante", "Scrivi cosa selezionare (es. 'costume da bagno')."); return
        try:
            img = self._upload_local(self.frame1_tab.canvas._composite(OUTPUT_DIR))
            detector = cfg.get("detector", "grounding_dino")
            threshold = float(cfg.get("threshold", cfg.get("clipseg_threshold", 0.30)))
            prompt = self.segpipe.build_text(
                img, cfg["text"],
                detector=detector,
                sam_model=cfg.get("sam_model"),
                threshold=threshold,
            )
            self._seg_pid = self.driver.submit(prompt)
            det_label = {"grounding_dino": "GroundingDINO+SAM-HQ",
                         "grounding_dino_only": "GroundingDINO",
                         "sam_only": "SAM"}.get(detector, detector)
            self._log(f"Selezione '{cfg['text']}' inviata via {det_label} (soglia {threshold:.2f})…")
        except Exception as e:
            QMessageBox.critical(self, "Errore selezione", str(e)); self._log(f"[ERR] segment text: {e}")

    def segment_by_region(self):
        if not self._segment_preconds():
            return
        cfg = self.frame1_tab.get_segment_config()
        if not self._sam_models:
            QMessageBox.warning(self, "SAM non disponibile",
                                "La selezione a click/riquadro richiede un modello SAM in models/sams.\n"
                                "Ho mappato E:\\AI\\Models\\Segmentation in extra_model_paths.yaml: "
                                "riavvia ComfyUI per caricarlo."); return
        # Il puntatore (Impact SAMLoader) carica solo SAM vanilla: se la scelta è un SAM-HQ
        # o assente, ripiega su un vanilla disponibile (sam_vit_b/l/h).
        sam = cfg.get("sam_model")
        if not sam or "hq" in sam.lower():
            vanilla = next((m for m in self._sam_models if "hq" not in m.lower()), None)
            if not vanilla:
                QMessageBox.warning(self, "Serve un SAM vanilla",
                                    "La selezione a click/box richiede un SAM vanilla "
                                    "(es. sam_vit_b / sam_vit_l) in models/sams. "
                                    "Ho solo modelli SAM-HQ (usati dalla selezione «da testo»).\n\n"
                                    "Aggiungi un sam_vit_*.pth e riavvia ComfyUI, oppure usa la "
                                    "selezione «da testo» (GroundingDINO).")
                return
            sam = vanilla
        try:
            exp = self.frame1_tab.canvas.export_region_mask(OUTPUT_DIR)
            img = self._upload_local(exp["image"]); region = self._upload_local(exp["mask"])
            # la soglia UI ora AGISCE sul puntatore: più bassa = selezione più ampia (SAM mask).
            seg_thr = float(cfg.get("threshold", cfg.get("clipseg_threshold", 0.50)))
            prompt = self.segpipe.build_region(img, region, sam_model=sam,
                                               detection_hint=exp.get("hint", "center-1"),
                                               threshold=seg_thr)
            self._seg_pid = self.driver.submit(prompt)
            self._log(f"Selezione magnetica ({exp.get('hint')}) inviata…")
        except Exception as e:
            QMessageBox.critical(self, "Errore selezione", str(e)); self._log(f"[ERR] segment region: {e}")

    def _check_segment_done(self):
        try:
            hist = self.driver.history(self._seg_pid)
        except Exception:
            return
        entry = hist.get(self._seg_pid)
        if not entry:
            return
        st = entry.get("status", {})
        if st.get("status_str") == "error":
            self.gen_dialog.finish()
            self._log("[ERR] selezione fallita (controlla che il modello SAM sia caricato).")
            self._seg_pid = None; return
        if not st.get("completed"):
            return
        self._seg_pid = None
        self.gen_dialog.finish()
        for out in entry.get("outputs", {}).values():
            for item in out.get("images", []) or []:
                fn = item.get("filename")
                if not fn:
                    continue
                try:
                    data = self.driver.download_view(fn, item.get("subfolder", ""), item.get("type", "output"))
                    rp = OUTPUT_DIR / f"segmask_{fn}"; rp.write_bytes(data)
                    self.frame1_tab.canvas.load_mask_result(str(rp))
                    self._log("Selezione applicata alla maschera (modificabile con pennello/gomma).")
                    return
                except Exception as e:
                    self._log(f"[ERR] download maschera: {e}")

    def _check_completion(self):
        try:
            hist = self.driver.history(self._current_pid)
        except Exception:
            return
        entry = hist.get(self._current_pid)
        if not entry:
            return
        st = entry.get("status", {})
        if st.get("status_str") == "error":
            self.gen_dialog.finish()
            self._log("[ERR] esecuzione fallita")
            self._current_pid = None
            return
        if st.get("completed"):
            pid = self._current_pid; self._current_pid = None
            self.gen_dialog.finish()
            saved = 0
            video_path: str | None = None
            meta_to_save = self._generation_cache.pop(pid, None)
            for out in entry.get("outputs", {}).values():
                for key in ("gifs", "videos", "images"):
                    for item in out.get(key, []) or []:
                        fn = item.get("filename")
                        if not fn:
                            continue
                        try:
                            data = self.driver.download_view(fn, item.get("subfolder", ""), item.get("type", "output"))
                            dest = OUTPUT_DIR / fn; dest.write_bytes(data)
                            # Salva sidecar JSON accanto al file generato
                            if meta_to_save:
                                try:
                                    MetadataSaver.save_sidecar(str(dest), meta_to_save)
                                except Exception as se:
                                    self._log(f"[warn] sidecar: {se}")
                            self._log(f"✔ salvato: {dest}")
                            saved += 1
                            if fn.lower().endswith((".mp4", ".webm", ".mkv", ".mov", ".gif")):
                                video_path = str(dest)
                        except Exception as e:
                            self._log(f"[ERR] download {fn}: {e}")
            self.progress.setValue(self.progress.maximum())
            self._log(f"Completato ({saved} file).")
            if video_path:
                self._last_video_path = video_path        # per i tab Definizione / Prolunga video
                if hasattr(self, "video_def_tab"):
                    self.video_def_tab.set_video(video_path)
                if hasattr(self, "video_extend_tab"):
                    self.video_extend_tab.set_video(video_path)
                # ── VIDEO LUNGHI: chaining I2V con la pipeline base (sostituisce l'IAMCCS) ──
                # Se siamo in un chaining attivo, questo video è un SEGMENTO → unisci + continua.
                if getattr(self, "_extend_state", None) is not None:
                    self._handle_extend_completion(video_path)
                    return
                # Prima estensione: la base è pronta → avvia il chaining I2V.
                ae = getattr(self, "_auto_extend", None)
                if ae and ae.get("base_vals"):
                    self._auto_extend = None              # consuma
                    if (ae["target"] - ae["produced"]) >= 1.0:
                        self._begin_i2v_extend(video_path, ae)
                        return
                # ── RESTAURO VOLTI automatico: ultimo passaggio sul video finale ──
                fr = getattr(self, "_auto_face_restore", None)
                if fr:
                    self._auto_face_restore = None          # consuma (evita loop sul restauro)
                    if self._run_auto_face_restore(video_path, fr):
                        return                              # il completamento mostrerà il video restaurato
                # ReTake "tieni audio originale": muxa l'audio del video sorgente sul risultato.
                if getattr(self, "_v2v_keep_audio_src", None):
                    video_path = self._mux_source_audio(video_path, self._v2v_keep_audio_src)
                    self._v2v_keep_audio_src = None
                # SOSPESO (richiesta utente): il post-process _clean.mp4 non serve più — l'originale è
                # già a posto e il re-encode peggiorava il video. Mostriamo direttamente l'originale.
                # video_path = self._clean_video(video_path, antimoire=getattr(self, "_last_gen_native", True))
                self._show_video(video_path)

    # ──────────────── VIDEO LUNGHI: estensione per concatenazione I2V ────────────────
    # Sostituisce l'IAMCCS (che dava rumore): ogni segmento è generato con la pipeline BASE
    # provata, in modalità I2V dall'ULTIMO frame del video accumulato → pulito e coerente.
    def _begin_i2v_extend(self, base_video: str, ae: dict):
        # produced = durata REALE della base (LTX a volte genera meno della soglia richiesta:
        # es. soglia 20s ma base 10s). Usare la stima ae["produced"]=soglia faceva fermare presto
        # l'accumulo (25s richiesti → 16s reali). Misuriamo sempre il file vero.
        real_base = self._probe_duration(base_video) or ae["produced"]
        self._extend_state = {
            "target": ae["target"], "produced": real_base,
            "seg_seconds": ae["seg_seconds"], "accumulated": base_video,
            "base_vals": ae["base_vals"], "loras": ae["loras"],
            "audio": ae["audio"], "cnet": ae["cnet"], "fps": ae["fps"], "segs": 0,
            "overlap": int(ae.get("overlap", 24)),   # frame di SOVRAPPOSIZIONE fusi alla giunzione
            "tail_trim": 10,                          # frame tagliati a FINE segmento (freeze finale LTX)
            "tts_full": ae.get("tts_full"),           # traccia voce completa → affettata per segmento
        }
        self._log(f"[Video lungo] base pronta → estensione I2V fino a {ae['target']:.0f}s")
        self._start_extend_segment()

    def _start_extend_segment(self):
        import copy as _copy
        es = self._extend_state
        try:
            vals = _copy.deepcopy(es["base_vals"])
            vals.pop("ref_image_file", None)
            # Evita la RIPETIZIONE del parlato negli spezzoni extend: togli il dialogo tra virgolette
            # dal prompt (es. «she says: "…"»). Il dialogo lo dice SOLO la base; gli extend continuano
            # scena+ambiente (l'audio è già in continuità via la maschera audio). Tieni il resto del prompt.
            import re as _re
            _pp = vals.get("prompt_positive", "")
            _stripped = _re.sub(r'["“”][^"“”]*["“”]', '', _pp)
            vals["prompt_positive"] = _re.sub(r'\s{2,}', ' ', _stripped).strip() or _pp
            # MULTI-FRAME: condiziona sugli ultimi ~16 frame IN MOTO (saltando il freeze finale) →
            # continuità di moto/scena (no salto). Fallback ad AddGuide single-frame se il clip fallisce.
            # MULTI-FRAME: il contesto = esattamente gli ULTIMI `overlap` frame IN MOTO di A (saltando
            # il freeze finale). Il segmento li RIGENERA all'inizio (ancora) → in stitch fondiamo quella
            # finestra (seam tra contenuti identici = niente zoom-jump, coda in moto copre la rampa).
            # Rileva la fine della parte "viva" (scarta la coda degradata in nebbia di LTX) → il
            # contesto NON parte da nebbia → niente collasso a nero del segmento successivo.
            es["a_good_end"] = self._last_detailed_time(es["accumulated"], es["fps"], es["tail_trim"])
            tail = self._extract_tail_clip(es["accumulated"], es["overlap"], es["a_good_end"], es["fps"])
            if tail:
                vals["ref_video_file"] = tail
                vals["i2v_method"] = "multiframe"
            else:
                vals["ref_image_file"] = self._upload_local(self._extract_last_frame(es["accumulated"]))
                vals["i2v_method"] = "addguide"
            vals["ref_strength"] = 0.6   # ancora più stretta → i primi `overlap` frame combaciano meglio
            # genera abbastanza da coprire il netto + l'overlap (fuso) + il freeze finale (tagliato)
            remaining = es["target"] - es["produced"]
            consume_s = (es["overlap"] + es["tail_trim"]) / max(1, es["fps"])
            vals["length_s"] = max(4.0, min(es["seg_seconds"], remaining + consume_s))
            # durata overlap (s) per la continuità AUDIO del segmento (audio mask: rigenera [overlap_s, len])
            vals["v2v_overlap_s"] = es["overlap"] / max(1, es["fps"])
            # VOCE LUNGA (TTS): usa la FETTA della traccia voce per questo segmento (parlato continuo,
            # niente ripetizione). Allineo ~overlap_s prima (il segmento riproduce l'overlap).
            if es.get("tts_full"):
                seg_audio_start = max(0.0, es["produced"] - vals["v2v_overlap_s"])
                _slice = self._slice_audio(es["tts_full"], seg_audio_start, vals["length_s"])
                if _slice:
                    vals["user_audio_file"] = _slice
                    vals["audio_mode"] = "user"
                    es["audio"] = dict(es["audio"], audio_mode="user")
                    vals["extend_audio_continuity"] = False   # la voce è il TTS, non l'audio LTX
                else:
                    # TTS finito → questo segmento resta ambiente LTX (dialogo già tolto dal prompt)
                    vals["audio_mode"] = "native"
                    es["audio"] = dict(es["audio"], audio_mode="native")
            # FORZA il superset (la base I2V validata): build + prune come la generazione normale
            self.engine.use_template("superset")
            prompt = self.engine.build_prompt(vals, loras=es["loras"], overrides=self._overrides)
            disable = [] if vals.get("prompt_enhance") else ["prompt_enhancer", "prompt_relay"]
            if not es["cnet"].get("controlnet_enabled"):
                disable.append("controlnet")
            if es["audio"].get("audio_mode") != "clone":
                disable.append("voice")
            if es["audio"].get("audio_mode") != "user":
                disable.append("audio_user")
            disable += [f"keyframe{i}" for i in range(8, 1, -1)]
            disable.append("distilled")   # qualità: niente LoRA distilled anche nei segmenti extend
            prompt, _ = self.engine.disable_modules(prompt, disable)
            es["segs"] += 1
            self._current_pid = self.driver.submit(prompt)
            self.gen_dialog.start(prompt, f"Video lungo: segmento {es['segs']} "
                                          f"({es['produced']:.0f}/{es['target']:.0f}s)")
            self._log(f"[Video lungo] segmento {es['segs']}: I2V multi-frame, +{vals['length_s']:.0f}s")
        except Exception as e:
            self._log(f"[ERR] segmento extend: {e}")
            final = es["accumulated"]; self._extend_state = None
            self._show_video(final)

    def _handle_extend_completion(self, segment_video: str):
        es = self._extend_state
        try:
            es["accumulated"] = self._stitch_videos(es["accumulated"], segment_video, es["fps"])
            # produced = durata REALE dell'accumulato (non una stima) → l'accumulo si ferma quando
            # il video VERO raggiunge il target. Fallback alla stima se il probe fallisce.
            real = self._probe_duration(es["accumulated"])
            es["produced"] = real if real > 0 else es["produced"] + max(
                1.0, es["seg_seconds"] - (es["overlap"] + es["tail_trim"]) / max(1, es["fps"]))
            self._log(f"[Video lungo] segmento unito → {es['produced']:.1f}/{es['target']:.0f}s (reale)")
        except Exception as e:
            self._log(f"[ERR] stitch segmento: {e}")
        if es["produced"] < es["target"] - 0.5 and es["segs"] < 60:
            self._start_extend_segment()
        else:
            final = es["accumulated"]; self._extend_state = None
            self._log(f"[Video lungo] ✔ COMPLETATO: {es['produced']:.0f}s")
            self._show_video(final)

    def _extract_last_frame(self, video: str) -> str:
        import subprocess
        out = OUTPUT_DIR / "_extend_lastframe.png"
        # prendi il frame ~0.3s prima della fine (l'ultimissimo può essere già il freeze finale)
        subprocess.run(["ffmpeg", "-y", "-sseof", "-0.35", "-i", video, "-vframes", "1", str(out)],
                       capture_output=True)
        return str(out)

    def _extract_tail_clip(self, video: str, overlap_frames: int, end_time: float,
                           fps: int) -> str | None:
        """Estrae gli ultimi `overlap_frames` frame IN MOTO che terminano a `end_time` (calcolato da
        `_last_detailed_time` → ULTIMO frame ancora 'vivo', PRIMA della nebbia/degradazione di fine
        clip LTX). Questa finestra è la stessa che `_stitch_videos` fonderà alla giunzione: il segmento
        successivo la rigenera (ancora) → niente zoom-jump, e la coda IN MOTO (non nebbia!) copre la
        rampa del nuovo segmento → niente stop e niente collasso a nero. Ritorna il nome. None se fallisce."""
        import subprocess
        try:
            comfy_in = Path(r"E:\AI\Comfy_Nexus\ComfyUI\input")
            comfy_in.mkdir(parents=True, exist_ok=True)
            name = "_extend_tail.mp4"
            out = comfy_in / name
            dur = self._probe_duration(video)
            win = max(0.3, overlap_frames / max(1, fps))     # finestra = esattamente i frame di overlap
            end = end_time if (end_time and end_time > win) else (max(win, dur - 0.4) if dur > 0 else None)
            start = max(0.0, end - win) if end else 0.0
            cmd = ["ffmpeg", "-y", "-ss", str(start)]
            if end:
                cmd += ["-to", str(end)]
            cmd += ["-i", video, "-c:v", "libx264", "-an", str(out)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            return name if (r.returncode == 0 and out.exists()) else None
        except Exception:
            return None

    @staticmethod
    def _probe_duration(video: str) -> float:
        import subprocess
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nw=1:nk=1", video], capture_output=True, text=True)
        try:
            return float(r.stdout.strip())
        except Exception:
            return 0.0

    @staticmethod
    def _probe_resolution(video: str):
        import subprocess
        try:
            r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                                "stream=width,height", "-of", "csv=p=0:s=x", video],
                               capture_output=True, text=True)
            w, h = r.stdout.strip().split("x")[:2]
            return int(w), int(h)
        except Exception:
            return None, None

    def _upload_video(self, path: str):
        """Copia il video sorgente nella input di ComfyUI → nome per VHS_LoadVideo (modalità V2V)."""
        import shutil
        try:
            comfy_in = Path(r"E:\AI\Comfy_Nexus\ComfyUI\input")
            comfy_in.mkdir(parents=True, exist_ok=True)
            name = "_v2v_source" + Path(path).suffix.lower()
            dst = comfy_in / name
            shutil.copy2(path, dst)
            return name if dst.exists() else None
        except Exception as e:
            self._log(f"[ERR] upload video V2V: {e}")
            return None

    @staticmethod
    def _frame_brightness(video: str, t: float):
        """Luminanza media (0-255) di un frame del video al tempo t. None se fallisce."""
        import subprocess
        p = OUTPUT_DIR / "_bright_probe.png"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", video,
                        "-frames:v", "1", "-vf", "scale=96:-1", str(p)], capture_output=True)
        try:
            from PIL import Image
            px = list(Image.open(str(p)).convert("L").getdata())
            return sum(px) / len(px) if px else None
        except Exception:
            return None

    @staticmethod
    def _frame_detail(png: str) -> float | None:
        """Dettaglio/contrasto di un frame = deviazione standard della luminanza. Un frame degradato
        in nebbia informe (tipica fine clip LTX) ha stddev BASSA; un frame ricco di dettaglio alta."""
        try:
            from PIL import Image
            import statistics
            im = Image.open(png).convert("L")
            return statistics.pstdev(list(im.getdata()))
        except Exception:
            return None

    def _last_detailed_time(self, video: str, fps: int, tail_trim: int = 10) -> float:
        """Timestamp dell'ULTIMO frame ancora 'vivo' (dettaglio sopra soglia), scartando la coda
        degradata in nebbia informe di fine clip LTX. Evita che il contesto multi-frame del segmento
        successivo parta da nebbia → collasso a nero (bug 'segmento 2 nero'). Campiona l'ultimo ~3s,
        soglia = 70% del picco di dettaglio recente. Ritorna min(good_end, dur−tail_trim) per saltare
        anche l'eventuale freeze. Fallback prudente se il sampling fallisce."""
        import subprocess
        dur = self._probe_duration(video)
        if dur <= 0:
            return dur
        floor = max(0.0, dur - tail_trim / max(1, fps))   # comunque salta il freeze finale
        span = min(3.0, dur)
        n = max(4, int(span / 0.2))
        samples = []
        probe = OUTPUT_DIR / "_deg_probe.png"
        for i in range(n + 1):
            t = dur - span + i * (span / n)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", video,
                            "-frames:v", "1", "-vf", "scale=192:-1", str(probe)], capture_output=True)
            d = self._frame_detail(str(probe))
            if d is not None:
                samples.append((t, d))
        if not samples:
            return floor
        peak = max(d for _, d in samples)
        thr = peak * 0.7
        good = [t for t, d in samples if d >= thr]
        good_end = max(good) if good else max(0.0, dur - span * 0.5)
        return min(good_end, floor)

    def _stitch_videos(self, a: str, b: str, fps: int) -> str:
        """Unisce a + b con overlap-blend (semantica ImageBatchExtendWithOverlap del workflow di
        riferimento V2V_Extend). B NON viene droppato all'inizio: i suoi primi `overlap` frame
        RIGENERANO la coda di A (stesso contenuto, ancorato dal guide multi-frame). Tagliamo il
        freeze finale di A così la finestra di blend cade su frame IN MOTO, poi dissolviamo
        sull'INTERA finestra di overlap (= overlap/fps, non 0.25s) tra contenuti identici → seam
        invisibile (niente zoom-jump) e la coda in moto di A copre la rampa di B (niente stop).
        Prova con audio (acrossfade), ripiega su solo-video se a non ha audio."""
        import subprocess
        es = self._extend_state
        out = OUTPUT_DIR / f"ltxstudio_longvideo_{es['segs']:03d}.mp4"
        ov = int(es.get("overlap", 24))          # frame di sovrapposizione fusi alla giunzione
        tail = int(es.get("tail_trim", 10))      # frame minimi tagliati a FINE (freeze finale LTX)
        xf = ov / max(1, fps)                     # la dissolvenza copre TUTTA la finestra di overlap
        # A: taglia alla fine della parte VIVA (a_good_end, calcolato da _last_detailed_time = stessa
        #    finestra che B rigenera) → niente nebbia degradata nel blend, contenuti che combaciano.
        dur_a = self._probe_duration(a)
        a_good = es.get("a_good_end")
        a_end = a_good if (a_good and a_good > xf + 0.3) else (
            max(xf + 0.3, dur_a - tail / max(1, fps)) if dur_a > 0 else dur_a)
        a_trim = f"start=0:end={a_end}"
        # B: tieni dall'inizio; taglia alla sua fine VIVA (scarta la nebbia di fine clip di B → il
        #    video finale non finisce in dissolvenza nebbiosa).
        dur_b = self._probe_duration(b)
        b_good = self._last_detailed_time(b, fps, tail)
        b_end = b_good if (b_good and b_good > xf + 0.3) else (
            max(xf + 0.3, dur_b - tail / max(1, fps)) if dur_b > 0 else dur_b)
        b_trim = f"start=0:end={b_end}"
        offset = max(0.1, a_end - xf)             # il blend inizia `ov` frame prima della fine di A
        # ESPOSIZIONE: il segmento B (extend) tende a uscire più SCURO del precedente (roundtrip VAE
        # encode→rigenera della coda + coda presa da zona già più dim) → seconda parte SOTTOESPOSTA.
        # Compensiamo agganciando la luminanza di B a quella di A alla giunzione (brightness additivo).
        b_eq = ""
        a_luma = self._frame_brightness(a, max(0.0, a_end - 0.15))
        b_luma = self._frame_brightness(b, 0.15)
        if a_luma is not None and b_luma is not None and b_luma > 1:
            delta = max(-0.2, min(0.2, (a_luma - b_luma) / 255.0))
            if abs(delta) > 0.02:
                b_eq = f",eq=brightness={delta:.3f}"
                self._log(f"[Video lungo] compenso esposizione segmento: brightness {delta:+.3f}")
        # DEBAND sul video finale → attenua il banding/dithering nei gradienti lisci (nebbia/bagliori).
        vchain = (f"[0:v]trim={a_trim},setpts=PTS-STARTPTS[v0];"
                  f"[1:v]trim={b_trim},setpts=PTS-STARTPTS{b_eq}[v1];"
                  f"[v0][v1]xfade=transition=fade:duration={xf}:offset={offset},deband[outv]")
        # RONZIO a ~55-60Hz del modello audio LTX (hum tipo rete elettrica): highpass 70Hz + NOTCH
        # stretto a 57Hz lo eliminano (verificato: i picchi 55/59Hz spariscono) mantenendo i bassi
        # legittimi (più di un highpass a 100Hz) e senza intaccare voce/ambiente.
        # Filtri PER-SEGMENTO prima del crossfade (il notch applicato DOPO l'acrossfade "suonava"
        # attraverso la giunzione = il "crash"). Crossfade triangolare (c1/c2=tri) = transizione liscia.
        _audio_clean = "highpass=f=70,bandreject=f=57:width_type=h:w=20"
        achain = (f"[0:a]atrim={a_trim},asetpts=PTS-STARTPTS,{_audio_clean}[a0];"
                  f"[1:a]atrim={b_trim},asetpts=PTS-STARTPTS,{_audio_clean}[a1];"
                  f"[a0][a1]acrossfade=d={xf}:c1=tri:c2=tri[outa]")

        def _try(filtergraph, maps, pix, codec_args):
            cmd = ["ffmpeg", "-y", "-i", a, "-i", b, "-filter_complex", filtergraph]
            for m in maps:
                cmd += ["-map", m]
            cmd += ["-r", str(fps), "-pix_fmt", pix] + codec_args + [str(out)]
            return subprocess.run(cmd, capture_output=True, text=True)

        # qualità: 10-bit + CRF basso (visivamente lossless). Tenta libx264 10-bit; se il build non lo
        # supporta, ripiega a 8-bit; e su solo-video se A non ha audio.
        hi10 = ["-c:v", "libx264", "-crf", "16", "-preset", "slow", "-c:a", "aac", "-b:a", "192k"]
        hi8 = ["-c:v", "libx264", "-crf", "16", "-preset", "slow", "-c:a", "aac", "-b:a", "192k"]
        attempts = [
            (f"{vchain};{achain}", ["[outv]", "[outa]"], "yuv420p10le", hi10),  # 10-bit + audio
            (f"{vchain};{achain}", ["[outv]", "[outa]"], "yuv420p",     hi8),   # 8-bit + audio
            (vchain,               ["[outv]"],           "yuv420p10le", ["-c:v", "libx264", "-crf", "16", "-preset", "slow"]),  # 10-bit solo video
            (vchain,               ["[outv]"],           "yuv420p",     ["-c:v", "libx264", "-crf", "16", "-preset", "slow"]),  # 8-bit solo video
        ]
        last = None
        for fg, maps, pix, args in attempts:
            r = _try(fg, maps, pix, args)
            if r.returncode == 0 and out.exists():
                return str(out)
            last = r
        raise RuntimeError(f"ffmpeg overlap-blend stitch: {(last.stderr or '')[-300:] if last else 'n/a'}")

    def _run_auto_face_restore(self, video_path: str, cfg: dict) -> bool:
        """Lancia il restauro volti sul video finale come passaggio automatico. Riusa la
        FaceRestorePipeline. Ritorna True se il submit è partito (il risultato verrà gestito
        dal normale _check_completion), False se va saltato (mostra il video originale)."""
        import shutil
        try:
            src = Path(video_path)
            if not src.exists():
                return False
            from core.face_restore_pipeline import ensure_model
            ok, msg = ensure_model(cfg["model"], log=self._log)
            if not ok:
                self._log(f"[Restauro volti] {msg} → mostro il video non restaurato.")
                return False
            comfy_input = Path(r"E:\AI\Comfy_Nexus\ComfyUI\input"); comfy_input.mkdir(parents=True, exist_ok=True)
            dest = comfy_input / src.name
            if str(dest) != str(src):
                shutil.copy2(src, dest)
            v = self.video_tab.get_values()
            prompt = self.facerestorer.build(
                src.name, fps=int(v.get("fps", 24)), fidelity=cfg["fidelity"],
                model=cfg["model"], detection=cfg.get("detection", "retinaface_resnet50"))
            self._current_pid = self.driver.submit(prompt)
            self.gen_dialog.start(prompt, "Restauro volti (CodeFormer/GFPGAN)")
            self._log(f"[Restauro volti] passaggio finale su {src.name} "
                      f"(fidelity {cfg['fidelity']:.2f}, {cfg['model']}).")
            return True
        except Exception as e:
            self._log(f"[ERR] restauro volti automatico: {e} → mostro il video non restaurato.")
            return False

    def generate_restore(self, cfg: dict):
        """Restauro / colorazione foto tramite RestorePipeline."""
        if not self.driver.is_up():
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "ComfyUI offline", "Avvia ComfyUI per elaborare."); return
        if self._inpaint_pid:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Attendi", "Elaborazione già in corso."); return

        image_path = cfg.get("image_path", "")
        if image_path == "<from_canvas>":
            image_path = self.frame1_tab.image_path or ""
        if not image_path:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Nessuna immagine", "Seleziona un'immagine da elaborare."); return

        op = cfg.get("operation", "restore")
        try:
            img_name = self._upload_local(image_path)
            self._inpaint_before = image_path
            self._inpaint_bbox = None
            seed = random.randint(0, 2**31)

            if op == "restore":
                prompt = self.restorer.build_restore(
                    img_name,
                    upscale_model=cfg["upscale_model"],
                    target_scale=cfg["target_scale"],
                    denoise=cfg["denoise"],
                    denoise_strength=cfg["denoise_strength"],
                    seed=seed)
                label = f"Restauro ({cfg['target_scale']:.0f}×)"
            elif op == "colorize":
                prompt = self.restorer.build_colorize(
                    img_name,
                    flux2_model=cfg["flux2_model"],
                    subject=cfg["subject"],
                    style=cfg["style"],
                    seed=seed,
                    steps=cfg["color_steps"])
                label = "Colorazione B&W"
            else:  # restore_and_colorize
                prompt = self.restorer.build_restore_and_colorize(
                    img_name,
                    upscale_model=cfg["upscale_model"],
                    flux2_model=cfg["flux2_model"],
                    subject=cfg["subject"],
                    style=cfg["style"],
                    seed=seed,
                    steps=cfg["color_steps"])
                label = "Restauro + Colorazione"

            self._inpaint_pid = self.driver.submit(prompt)
            self._inpaint_mode = label
            self.gen_dialog.start(prompt, label)
            self._log(f"{label}: inviato (seed={seed}).")
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Errore restauro", str(e))
            self._log(f"[ERR] restauro: {e}")

    def _restore_from_canvas(self):
        """Porta al tab Restauro con l'immagine del canvas pre-selezionata."""
        img = self.frame1_tab.image_path or ""
        if not img:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Nessuna immagine",
                "Carica o genera un'immagine nel canvas prima di usare il Restauro.")
            return
        # Pre-imposta il path nel tab Restauro
        self.restore_tab._input_path = img
        self.restore_tab.path_edit.setText(img)
        # Porta al tab «Migliora qualità» → sotto-tab Foto (Restauro)
        self.tabs.setCurrentWidget(self.quality_tab)
        self.quality_tab.setCurrentWidget(self.restore_tab)
        self._log(f"[Restauro] Immagine canvas inviata al tab Migliora qualità → Foto: {img}")

    def _on_catalog_updated(self):
        """Ricarica il catalogo LoRA dopo un ingest completato."""
        try:
            self.lora_tab.catalog.load()
            self.lora_tab.refresh()
            self._log("[Modelli] Catalogo LoRA ricaricato dopo ingest.")
        except Exception as e:
            self._log(f"[warn] Ricarica catalogo: {e}")

    def _slice_audio(self, src: str, start_s: float, dur_s: float) -> str | None:
        """Estrae [start_s, start_s+dur_s] dall'audio `src` → file nella input di ComfyUI (per usarlo
        come audio 'user' di un segmento del video lungo). Ritorna il NOME del file, None se vuoto/errore."""
        import subprocess
        try:
            comfy_in = Path(r"E:\AI\Comfy_Nexus\ComfyUI\input")
            comfy_in.mkdir(parents=True, exist_ok=True)
            name = f"_tts_seg_{int(start_s*100):07d}.wav"
            out = comfy_in / name
            r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, start_s):.3f}",
                                "-t", f"{max(0.1, dur_s):.3f}", "-i", src, "-ac", "2", "-ar", "44100",
                                str(out)], capture_output=True, text=True)
            if r.returncode == 0 and out.exists() and out.stat().st_size > 1024:
                return name
            return None
        except Exception as e:
            self._log(f"[ERR] slice audio TTS: {e}")
            return None

    def _mux_source_audio(self, video: str, audio_src: str) -> str:
        """ReTake 'tieni audio originale': rimpiazza la traccia audio del risultato con quella del
        video sorgente. Ritorna il path muxato (o l'originale se fallisce)."""
        import subprocess
        out = Path(video).with_name(Path(video).stem + "_srcaudio.mp4")
        r = subprocess.run(["ffmpeg", "-y", "-i", video, "-i", audio_src, "-map", "0:v:0",
                            "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                            "-shortest", str(out)], capture_output=True, text=True)
        if r.returncode == 0 and out.exists():
            self._log(f"[ReTake] audio originale muxato → {out.name}")
            return str(out)
        return video

    def _clean_video(self, path: str, antimoire: bool = True) -> str:
        """Post-process SOLO AUDIO del video finale: highpass 70 + notch 57Hz → toglie il RONZIO
        ~55-60Hz del modello audio LTX. Il VIDEO viene COPIATO (niente filtro: il moiré è risolto alla
        sorgente col VAEDecodeTiled + output h264 come i workflow di riferimento; nessun denoise che
        sfochi). Se non c'è audio, ritorna l'originale. `antimoire` ignorato (mantenuto per compat)."""
        import subprocess
        src = Path(path)
        # se non ha traccia audio, niente da pulire
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                                "stream=codec_name", "-of", "csv=p=0", str(src)], capture_output=True, text=True)
        if not probe.stdout.strip():
            return str(src)
        out = src.with_name(src.stem + "_clean.mp4")
        cmd = ["ffmpeg", "-y", "-i", str(src), "-c:v", "copy",
               "-af", "highpass=f=70,bandreject=f=57:width_type=h:w=20", "-c:a", "aac", "-b:a", "192k", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and out.exists():
            self._log(f"[pulizia audio] ronzio rimosso → {out.name}")
            return str(out)
        self._log("[pulizia audio] fallita → uso l'originale")
        return str(src)

    def _show_video(self, path: str):
        # rendi l'ultimo video disponibile come sorgente V2V ("Usa ultima generazione")
        try:
            if hasattr(self.video_tab, "v2v_panel"):
                self.video_tab.v2v_panel.set_last_generation(path)
        except Exception:
            pass
        if self.video_player is None:
            self.video_player = VideoPlayerDialog(self)
        self.video_player.play_file(path)
