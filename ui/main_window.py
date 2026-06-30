"""Finestra principale LTX Studio (MVP): tab + preview live + Genera cablato al backend."""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, QThread, Signal, Qt
from PySide6.QtGui import QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget, QPushButton,
    QLabel, QProgressBar, QPlainTextEdit, QMessageBox, QProgressDialog, QApplication,
    QMenu, QToolButton, QSplitter,
)

from ui.i18n_manager import tr, i18n, LANGUAGES, LANG_FLAGS
from ui.job_queue import GenerationQueue, JobQueuePanel

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
from core import features, gating_apply, watermark        # noqa: E402  (feature-gating per edizione)
from core import backend                                  # noqa: E402  (router backend nativo/ComfyUI)
from ui.tabs.video_def_tab import VideoDefTab          # noqa: E402
from ui.tabs.video_extend_tab import VideoExtendTab    # noqa: E402
from ui.tabs.restore_tab import RestoreTab             # noqa: E402
from ui.tabs.trellis_tab import TrellisTab             # noqa: E402
from ui.tabs.rig_tab import RigTab                     # noqa: E402
from ui.tabs.import_models_tab import ImportModelsTab  # noqa: E402
from ui.tabs.convert_tab import ConvertTab             # noqa: E402
from ui.tabs.anim_tab import AnimTab                   # noqa: E402
from ui.tabs.train_tab import TrainTab                 # noqa: E402
from ui.tabs.lora_video_tab import LoraVideoTab        # noqa: E402
from ui.tabs.lora_image_tab import LoraImageTab        # noqa: E402
from ui.tabs.music_tab import MusicTab                 # noqa: E402
from ui.tabs.world_tab import WorldTab                 # noqa: E402
from ui.tabs.weights_tab import WeightsTab             # noqa: E402
from core.mocap_service import MocapService            # noqa: E402
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
from core.trellis_pipeline import TrellisPipeline    # noqa: E402
from core.rig_pipeline import RigPipeline            # noqa: E402

OUTPUT_DIR = Path(r"E:\AI\ltxstudio\output")
# Cartella output di ComfyUI (stessa macchina) — per recuperare i .glb di TRELLIS/rig.
# NB: run_nvidia.bat avvia ComfyUI con --output-directory E:\AI\Comfy_Nexus\Output\
COMFY_OUTPUT_DIR = Path(r"E:\AI\Comfy_Nexus\Output")


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


class _NativeImageWorker(QThread):
    """Esegue una generazione immagine sul backend NATIVO (InferenceServer) fuori dal thread UI."""
    done = Signal(dict)

    def __init__(self, params: dict, module: str = "image"):
        super().__init__()
        self.params = params
        self.module = module

    def run(self):
        try:
            if self.module == "image":
                # Router motore: comfy-native (model_management ComfyUI) o diffusers, da user_prefs;
                # comfy per t2i con fallback automatico a diffusers, il resto su diffusers.
                from core import img_router
                res = img_router.generate(self.params, timeout=3600)
            else:
                from core import native_driver
                res = native_driver.run_and_wait(self.module, self.params, timeout=3600)
        except Exception as e:                          # noqa: BLE001
            res = {"status": "error", "error": str(e)}
        self.done.emit(res)


class _SeedVR2Worker(QThread):
    """SeedVR2 super-risoluzione video (CLI standalone, venv isolato). Stream log + path finale."""
    done = Signal(dict)
    progress = Signal(str)

    def __init__(self, src: str, out: str, *, resolution: int, batch_size: int, chunk_size: int):
        super().__init__()
        self.src, self.out = src, out
        self.resolution, self.batch_size, self.chunk_size = resolution, batch_size, chunk_size

    def run(self):
        from core import seedvr2_service as sv
        try:
            sv.upscale_video(self.src, self.out, resolution=self.resolution, batch_size=self.batch_size,
                             chunk_size=self.chunk_size, on_log=lambda l: self.progress.emit(l))
            self.done.emit({"status": "success", "output": [self.out]})
        except Exception as e:                          # noqa: BLE001
            self.done.emit({"status": "error", "error": str(e)})


class _FlashVSRWorker(QThread):
    """FlashVSR v1.1 super-risoluzione video x4 (repo diffsynth, venv isolato). Stream log + path."""
    done = Signal(dict)
    progress = Signal(str)

    def __init__(self, src: str, out: str):
        super().__init__()
        self.src, self.out = src, out

    def run(self):
        from core import flashvsr_service as fv
        try:
            fv.upscale_video(self.src, self.out, on_log=lambda l: self.progress.emit(l))
            self.done.emit({"status": "success", "output": [self.out]})
        except Exception as e:                          # noqa: BLE001
            self.done.emit({"status": "error", "error": str(e)})


class _VideoEditAIWorker(QThread):
    """#24/#59/#60 editing video AI: (inpaint) SAM2 track→guide magenta nel venv inference_server, poi
    LTX-2 edit_lora nel venv ltx2; (edit/style) direttamente LTX-2 col video sorgente come guide."""
    done = Signal(dict)
    progress = Signal(str)
    INF_PY = r"E:\AI\Tools\inference_server_venv\Scripts\python.exe"
    LTX2_PY = r"E:\AI\Tools\ltx2_venv\Scripts\python.exe"

    def __init__(self, cfg: dict, out_dir: str):
        super().__init__()
        self.cfg = cfg
        self.out_dir = out_dir

    def run(self):
        import json as _j, os as _os, subprocess as _sp, time as _t
        try:
            cfg = self.cfg
            src = cfg["video"]
            base = _os.path.splitext(_os.path.basename(src))[0]
            stamp = int(_t.time())
            out = _os.path.join(self.out_dir, f"edit_{cfg['mode']}_{base}_{stamp}.mp4")
            params = {"prompt": cfg["prompt"], "negative": "worst quality, blurry",
                      "edit_lora": cfg["edit_lora"], "control_strength": float(cfg.get("strength", 1.0)),
                      "width": 512, "height": 320, "num_frames": 25, "fps": 25,
                      "seed": 42, "offload": "cpu", "output": out}
            if cfg["mode"] == "inpaint":                       # #59 parametri inpaint dalla GUI
                params["control_strength"] = float(cfg.get("guide", 0.4))
                params["cfg_scale"] = float(cfg.get("cfg_scale", 6.0))
                params["mask_dilate"] = int(cfg.get("mask_dilate", 24))
            if cfg["mode"] == "inpaint":                       # SAM2 track → latent-inpaint (vinpaint)
                self.progress.emit("[edit] SAM2: traccio l'oggetto sul clip…")
                gpath = _os.path.join(self.out_dir, f"_guide_{base}_{stamp}.mp4")
                seg = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
                                    "inference_server", "backends", "video_segment.py")
                r = _sp.run([self.INF_PY, "-u", seg, src, cfg["object"], gpath],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
                line = next((l for l in (r.stdout or "").splitlines() if l.startswith("GUIDE:")), "")
                if not line:
                    self.done.emit({"status": "error", "error": "tracking SAM2 fallito: " + (r.stderr or "")[-300:]}); return
                # video_segment salva frames e masks accanto al video sorgente in <base>_track/
                track_dir = _os.path.splitext(src)[0] + "_track"
                params["original_frames_dir"] = _os.path.join(track_dir, "frames")
                params["inpaint_masks_dir"] = _os.path.join(track_dir, "masks")
                params["skip_stage_2"] = False
                self.progress.emit("[edit] maschera tracciata → latent-inpaint")
            else:
                params["control_video"] = src                  # edit/style: video sorgente come guide
            jf = _os.path.join(self.out_dir, f"_editp_{stamp}.json")
            _j.dump(params, open(jf, "w"))
            worker = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
                                   "workers", "ltx2_infer.py")
            self.progress.emit("[edit] genero il video (LTX-2)…")
            r2 = _sp.run([self.LTX2_PY, "-u", worker, jf], capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
            if _os.path.exists(out):
                self.done.emit({"status": "success", "output": [out]})
            else:
                self.done.emit({"status": "error", "error": (r2.stdout or "")[-200:] + (r2.stderr or "")[-300:]})
        except Exception as e:                              # noqa: BLE001
            self.done.emit({"status": "error", "error": str(e)})


class _NativeMultiTalkWorker(QThread):
    """Lip-sync MultiTalk (Wan2.1) come subprocess nel venv ComfyUI; riporta progress e path finale."""
    done = Signal(dict)
    progress = Signal(str)

    COMFY_PY = r"E:\AI\Comfy_Nexus\ComfyUI\venv\Scripts\python.exe"

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        import json as _json, os as _os, subprocess as _sp, tempfile as _tf
        try:
            worker = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "workers", "wan_multitalk.py")
            jf = _tf.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
            _json.dump(self.params, jf); jf.close()
            proc = _sp.Popen([self.COMFY_PY, "-u", worker, jf.name],
                             stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, encoding="utf-8", errors="replace")
            out_path = None
            for line in proc.stdout:
                line = line.rstrip()
                if line.startswith("[") or line.startswith("OK:") or "Error" in line:
                    self.progress.emit(line)
                if line.startswith("OK:"):
                    out_path = line[3:].strip()
            proc.wait()
            if out_path:
                self.done.emit({"status": "success", "output": [out_path]})
            else:
                self.done.emit({"status": "error", "error": "MultiTalk: nessun output prodotto"})
        except Exception as e:                          # noqa: BLE001
            self.done.emit({"status": "error", "error": str(e)})


class _FaceSwapWorker(QThread):
    """Face-swap post-process su un video (InsightFace inswapper) fuori dal thread UI."""
    done = Signal(dict)
    progress = Signal(str)

    def __init__(self, video: str, face: str, out: str):
        super().__init__()
        self.video, self.face, self.out = video, face, out

    def run(self):
        try:
            from core import faceswap
            if not faceswap.available():
                self.done.emit({"status": "error",
                                "error": "inswapper_128.onnx mancante in E:\\AI\\Models\\insightface."})
                return
            out = faceswap.swap_video(self.video, self.face, self.out,
                                      progress=lambda i, n: self.progress.emit(f"frame {i}/{n}"))
            self.done.emit({"status": "success", "output": [out]})
        except Exception as e:                          # noqa: BLE001
            self.done.emit({"status": "error", "error": str(e)})


class _RifeWorker(QThread):
    """Fluidificazione RIFE post-process (anti-judder / +fps) fuori dal thread UI."""
    done = Signal(dict)
    progress = Signal(str)

    def __init__(self, video: str, out: str, factor: int = 2):
        super().__init__()
        self.video, self.out, self.factor = video, out, factor

    def run(self):
        try:
            from core import rife
            if not rife.available():
                self.done.emit({"status": "error", "error": "Practical-RIFE non trovato."})
                return
            out = rife.interpolate_video(self.video, self.out, factor=self.factor,
                                         progress=lambda i, n: self.progress.emit(f"frame {i}/{n}"))
            self.done.emit({"status": "success", "output": [out]})
        except Exception as e:                          # noqa: BLE001
            self.done.emit({"status": "error", "error": str(e)})


class _NativeLtx2Worker(QThread):
    """Generazione video NATIVA LTX-2.3 (GGUF, zero ComfyUI) come subprocess nel venv ltx2.
    Legge i params da JSON, esegue workers/ltx2_infer.py (auto-GGUF + chunking per i lunghi),
    riporta progress e il path finale (riga 'OK:<path>')."""
    done = Signal(dict)
    progress = Signal(str)

    LTX2_PY = r"E:\AI\Tools\ltx2_venv\Scripts\python.exe"

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        import json as _json, os as _os, subprocess as _sp, tempfile as _tf
        try:
            worker = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "workers", "ltx2_infer.py")
            if not _os.path.exists(self.LTX2_PY):
                self.done.emit({"status": "error", "error": f"venv ltx2 mancante: {self.LTX2_PY}"})
                return
            jf = _tf.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
            _json.dump(self.params, jf); jf.close()
            env = dict(_os.environ)
            env["HF_HUB_OFFLINE"] = "1"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            proc = _sp.Popen([self.LTX2_PY, "-u", worker, jf.name], env=env,
                             stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, encoding="utf-8", errors="replace")
            out_path = None
            for line in proc.stdout:
                line = line.rstrip()
                if line.startswith("[") or line.startswith("OK:") or "Error" in line or "error" in line:
                    self.progress.emit(line)
                if line.startswith("OK:"):
                    out_path = line[3:].strip()
            proc.wait()
            if out_path and _os.path.exists(out_path):
                self.done.emit({"status": "success", "output": [out_path]})
            else:
                self.done.emit({"status": "error", "error": "LTX-2 nativo: nessun output prodotto"})
        except Exception as e:                          # noqa: BLE001
            self.done.emit({"status": "error", "error": str(e)})


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("window.title"))
        self.resize(1180, 740)

        self.driver = ComfyDriver()
        self.bridge = ComfyBridge()
        try:
            self.engine = TemplateEngine()
            self._engine_err = None
        except Exception as e:  # template non ancora catturato
            self.engine = None
            self._engine_err = str(e)

        self.jobs = GenerationQueue(self)       # coda SERIALE delle generazioni (no overlap/cancel reciproci)
        self._current_pid: str | None = None
        self._inpaint_pid: str | None = None
        self._trellis_pid: str | None = None    # job TRELLIS.2 in corso
        self._trellis_prefix: str = ""          # prefisso file atteso in output ComfyUI
        self._rig_pid: str | None = None        # job rigging SkinTokens in corso
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
        self.trellisser = TrellisPipeline()     # immagine → modello 3D (TRELLIS.2)
        self.rigger = RigPipeline()             # mesh → rig automatico (SkinTokens)
        self._last_video_path: str | None = None
        self._sam_models: list[str] = []
        self._seg_pid: str | None = None
        self._inventory_set = False
        self._text_encoders: list[str] = []     # encoder testo installati (cache per auto-download FLUX.2)
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
        self.frame1_tab.request_compose.connect(self.generate_compose)
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
        self.trellis_tab.request_trellis.connect(self.generate_trellis)
        self.rig_tab.request_rig.connect(self.generate_rig)
        # Convertitore + operazioni mesh nei visori 3D (Blender headless)
        self.convert_tab.request_convert.connect(self.run_convert)
        self._wire_viewer_ops(self.trellis_tab)
        self._wire_viewer_ops(self.rig_tab)
        # Animazione / Mocap
        self.anim_tab.request_mocap_start.connect(self.start_mocap)
        self.anim_tab.request_mocap_stop.connect(self.mocap.stop)
        self.anim_tab.request_retarget.connect(self.run_retarget)
        self.mocap.frame.connect(self.anim_tab.set_preview)
        self.mocap.finished_ok.connect(self._on_mocap_done)
        self.mocap.failed.connect(self._on_mocap_failed)
        self._wire_viewer_ops(self.anim_tab)
        self.models_tab.catalog_updated.connect(self._on_catalog_updated)
        self.frame1_tab.request_restore_photo.connect(self._restore_from_canvas)
        # Segnali del tab Avanzate → funzioni della finestra principale
        self.advanced_tab.open_workflow.connect(self._open_workflow)
        self.advanced_tab.open_analyzer.connect(self._open_analyzer)
        self.advanced_tab.open_navigator.connect(self._open_navigator)
        if getattr(self, "_last_video_path", None):
            self.video_def_tab.set_video(self._last_video_path)
            self.video_extend_tab.set_video(self._last_video_path)

        # Auto-avvio ComfyUI se non è già in esecuzione (il polling sotto lo rileva quando è pronto).
        self._maybe_autostart_native()
        # ComfyUI NON viene più avviato in automatico (migrazione zero-comfy): solo il server nativo.
        # _maybe_autostart_comfyui() resta definito per il fallback manuale/diagnostica, non chiamato.

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

        # ── Feature-gating per EDIZIONE: nasconde tab/funzioni non incluse e
        #    mostra l'age-gate se l'edizione consente contenuti 18+. In dev
        #    (core/edition.py assente) il profilo è permissivo (tutto visibile).
        try:
            report = gating_apply.apply(self)
            self._log(f"[Edizione: {report['edition']}] tab rimossi: "
                      f"{len(report['removed_tabs'])}")
        except Exception as e:                       # noqa: BLE001
            self._log(f"[gating] errore non bloccante: {e}")

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
        self.trellis_tab = TrellisTab()
        self.rig_tab = RigTab()
        self.convert_tab = ConvertTab()
        self.anim_tab = AnimTab()
        self.train_tab = TrainTab()
        self.lora_video_tab = LoraVideoTab()
        self.lora_image_tab = LoraImageTab()
        self.train_tab.lora_ready.connect(lambda _p: self._refresh_status())
        self.music_tab = MusicTab()
        self.world_tab = WorldTab()
        self.weights_tab = WeightsTab()
        self.mocap = MocapService(self)
        self.models_tab = ImportModelsTab()
        self.tabs.addTab(self.frame1_tab, tr("tab.images"))
        self.tabs.addTab(self.audio_tab, tr("tab.audio_voice"))
        self.tabs.addTab(self.video_tab, tr("tab.video"))
        self.tabs.addTab(self.controlnet_tab, tr("tab.controlnet"))
        self.tabs.addTab(self.video_extend_tab, tr("tab.extend_video"))
        from ui.tabs.video_edit_ai_tab import VideoEditAITab          # #24/#59/#60 editing video AI
        self.video_edit_ai_tab = VideoEditAITab()
        self.video_edit_ai_tab.request_edit.connect(self._gen_video_edit_ai)
        self.tabs.addTab(self.video_edit_ai_tab, tr("tab.video_edit_ai", "✨ Modifica Video AI"))
        # «Migliora qualità» = unione di Restauro foto (🖼 Foto) + Definizione video (🎬 Video)
        self.quality_tab = QTabWidget()
        self.quality_tab.addTab(self.restore_tab, tr("tab.quality.photo"))
        self.quality_tab.addTab(self.video_def_tab, tr("tab.quality.video"))
        self.tabs.addTab(self.quality_tab, tr("tab.quality"))
        # Modello 3D da immagine (TRELLIS.2) — tab separato
        self.tabs.addTab(self.trellis_tab, tr("tab.trellis"))
        # Rig 3D automatico (SkinTokens) — tab separato
        self.tabs.addTab(self.rig_tab, tr("tab.rig"))
        # Convertitore di formati 3D (Blender headless) — tab separato
        self.tabs.addTab(self.convert_tab, tr("tab.convert"))
        # Animazione / Mocap (FreeMoCap single-camera + retarget) — tab separato
        self.tabs.addTab(self.anim_tab, tr("tab.anim"))
        # Allena LoRA (FLUX.1/FLUX.2 via ai-toolkit) — tab separato
        self.tabs.addTab(self.train_tab, tr("tab.train", "Allena LoRA"))
        # LoRA Video LTX-2.3 (dataset auto-generato: personaggi/oggetti/luoghi) — tab separato
        self.tabs.addTab(self.lora_video_tab, tr("tab.lora_video", "LoRA Video"))
        # LoRA Foto FLUX.2 (dataset immagini auto-generato) — tab separato
        self.tabs.addTab(self.lora_image_tab, tr("tab.lora_image", "LoRA Foto"))
        # Musica (HeartMuLa canzoni/strumentale + MMAudio sonorizzazione) — tab separato
        self.tabs.addTab(self.music_tab, tr("tab.music", "Musica"))
        # Mondo 3D (WorldGen: scena/ambiente da foto, mesh+splat, player) — tab separato
        self.tabs.addTab(self.world_tab, tr("tab.world", "Mondo 3D"))
        # Pesi modelli (wizard download da HuggingFace, locale-first) — tab separato
        self.tabs.addTab(self.weights_tab, tr("tab.weights", "Pesi modelli"))
        # Import Models diventa un sotto-tab di Avanzate
        self.advanced_tab.attach_models(self.models_tab)
        self.tabs.addTab(self.advanced_tab, tr("tab.advanced"))
        # NB: i tab (immagine di editing) vengono aggiunti DOPO il pannello anteprima,
        #     così l'anteprima sta a SINISTRA e l'editing a DESTRA (posizioni invertite).

        # sincronizza il formato del canvas Frame1 con la risoluzione del Video
        def _sync_aspect():
            v = self.video_tab.get_values()
            self.frame1_tab.set_video_aspect(v["width"], v["height"])
        self.video_tab.width.valueChanged.connect(_sync_aspect)
        self.video_tab.height.valueChanged.connect(_sync_aspect)
        self.video_tab.res_preset.currentTextChanged.connect(lambda *_: _sync_aspect())
        # menu Proporzioni sulla BARRA del canvas → aggiorna la risoluzione (sorgente: video_tab spinbox).
        def _on_canvas_resolution(w, h):
            self.video_tab.res_preset.setCurrentText("Personalizzato")   # custom → spinbox abilitati
            self.video_tab.width.setValue(int(w)); self.video_tab.height.setValue(int(h))
        self.frame1_tab.canvas.resolution_chosen.connect(_on_canvas_resolution)
        _sync_aspect()

        # SINISTRA: spazio di lavoro grande = canvas + selezione AI + Genera/Interrompi
        # (niente anteprima: il sistema non la genera). Canvas e selezione vivono qui.
        work_top = QWidget()
        work = QVBoxLayout(work_top); work.setContentsMargins(0, 0, 0, 0)
        work.addWidget(self.frame1_tab.canvas, 1)        # spazio di lavoro per l'immagine
        work.addWidget(self.frame1_tab.segment_box)      # sistema di selezione con AI, sotto il canvas
        btns = QHBoxLayout()
        self.gen_btn = QPushButton(tr("btn.generate")); self.gen_btn.setObjectName("generate")
        self.gen_btn.clicked.connect(self.on_generate)
        self.cancel_btn = QPushButton(tr("btn.stop")); self.cancel_btn.clicked.connect(self.driver.interrupt)
        self.free_vram_btn = QPushButton(tr("btn.free_vram"))
        self.free_vram_btn.setToolTip(tr("btn.free_vram.tt"))
        self.free_vram_btn.clicked.connect(self._on_free_vram)
        self.navigator_btn = QPushButton(tr("btn.navigator"))
        self.navigator_btn.setToolTip(tr("btn.navigator.tt"))
        self.navigator_btn.clicked.connect(self._open_navigator)
        btns.addWidget(self.gen_btn, 2); btns.addWidget(self.cancel_btn, 1)
        btns.addWidget(self.free_vram_btn, 1); btns.addWidget(self.navigator_btn, 1)
        work.addLayout(btns)
        # Shortcut Ctrl+G per il Navigator
        nav_shortcut = QShortcut(QKeySequence("Ctrl+G"), self)
        nav_shortcut.activated.connect(self._open_navigator)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
        work.addWidget(self.progress)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)   # altezza regolabile dallo splitter

        # ── Separatori TRASCINABILI (QSplitter) ──────────────────────────────────────────
        # Verticale a SINISTRA: area di lavoro (canvas+selezione+pulsanti) ↕ log
        self.left_splitter = QSplitter(Qt.Vertical)
        self.left_splitter.addWidget(work_top)
        self.left_splitter.addWidget(self.log)
        self.left_splitter.setStretchFactor(0, 6); self.left_splitter.setStretchFactor(1, 1)
        self.left_splitter.setChildrenCollapsible(False); self.left_splitter.setHandleWidth(6)
        self.left_splitter.setSizes([620, 110])
        # Orizzontale PRINCIPALE: area di lavoro ↔ tab dei moduli di controllo
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(self.left_splitter)
        self.main_splitter.addWidget(self.tabs)
        self.main_splitter.setStretchFactor(0, 3); self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setChildrenCollapsible(False); self.main_splitter.setHandleWidth(6)
        root.addWidget(self.main_splitter)

        # Pannello CODA generazioni (dock): lista job + stato + annulla. Vedi ui/job_queue.py
        from PySide6.QtWidgets import QDockWidget
        self.queue_dock = QDockWidget(tr("queue.title", "Coda generazioni"), self)
        self.queue_dock.setObjectName("queue_dock")
        self.queue_panel = JobQueuePanel(self.jobs, self.queue_dock)
        self.queue_dock.setWidget(self.queue_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.queue_dock)
        self.jobs.changed.connect(self._refresh_status)

        self.status_lbl = QLabel("…")
        self.statusBar().addWidget(self.status_lbl)
        if self._engine_err:
            self._log(f"[!] Template non caricato: {self._engine_err}")
            self._log("    Cattura il golden prompt (templates/README_capture.md) e riavvia.")

    def _enqueue(self, label, worker, on_done=None, on_progress=None):
        """Accoda un worker di generazione nella coda seriale (no sovrapposizioni). Vedi ui/job_queue.py.
        Sostituisce il vecchio pattern worker.done.connect(...)+worker.start() in tutti i punti."""
        return self.jobs.enqueue(label, worker, on_done=on_done, on_progress=on_progress)

    def _build_menu(self):
        m = self.menuBar().addMenu(tr("menu.tools"))
        act_wiz = m.addAction(tr("menu.wizard")); act_wiz.triggered.connect(self.open_wizard)
        act_set = m.addAction(tr("menu.settings")); act_set.triggered.connect(self.open_settings)
        act_nav = m.addAction(tr("menu.navigator_gallery")); act_nav.triggered.connect(self._open_navigator)
        act_ana = m.addAction(tr("menu.analyzer")); act_ana.triggered.connect(self._open_analyzer)
        act_wfl = m.addAction(tr("menu.workflow")); act_wfl.triggered.connect(self._open_workflow)
        act_film = m.addAction(tr("menu.film_studio", "🎬 AI Film Studio")); act_film.triggered.connect(self._open_film_studio)
        # ── selettore lingua (sempre visibile nella status bar) ──────────────────────
        self._lang_btn = QToolButton()
        self._lang_btn.setText(f"{LANG_FLAGS.get(i18n.current_lang(), '🌐')} {LANGUAGES.get(i18n.current_lang(), '')}")
        self._lang_btn.setToolTip(tr("lang.icon_tt"))
        self._lang_btn.setPopupMode(QToolButton.InstantPopup)
        lang_menu = QMenu(self._lang_btn)
        for code, name in LANGUAGES.items():
            flag = LANG_FLAGS.get(code, "")
            act = lang_menu.addAction(f"{flag} {name}")
            act.setData(code)
            if code == i18n.current_lang():
                act.setCheckable(True); act.setChecked(True)
            act.triggered.connect(lambda _=False, c=code: self._change_language(c))
        self._lang_btn.setMenu(lang_menu)
        self.statusBar().addPermanentWidget(self._lang_btn)

    def _change_language(self, lang_code: str):
        """Cambia lingua, aggiorna il pulsante e avvisa di riavviare."""
        changed = i18n.set_language(lang_code)
        if not changed:
            return
        flag = LANG_FLAGS.get(lang_code, "🌐")
        name = LANGUAGES.get(lang_code, lang_code)
        self._lang_btn.setText(f"{flag} {name}")
        QMessageBox.information(
            self,
            tr("lang.changed.title"),
            tr("lang.changed.msg"),
        )

    def open_settings(self):
        if not self.engine:
            QMessageBox.warning(self, tr("mw.template_missing_title"), self._engine_err or ""); return
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

    def _open_film_studio(self):
        """Apre la finestra AI Film Studio (Text-to-Film, backend nativo)."""
        from ui.film_studio.film_studio_window import FilmStudioWindow
        if getattr(self, "_film_win", None) is None:
            self._film_win = FilmStudioWindow(self)
        self._film_win.show(); self._film_win.raise_(); self._film_win.activateWindow()

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
            # dimensioni canvas persistite via video_tab.width/height: ri-sincronizza l'aspetto del
            # canvas + l'etichetta del menu Proporzioni dopo il restore (i valori sono già ripristinati).
            try:
                v = self.video_tab.get_values()
                self.frame1_tab.set_video_aspect(int(v["width"]), int(v["height"]))
            except Exception:                            # noqa: BLE001
                pass
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

    def _maybe_autostart_native(self):
        """Avvia l'InferenceServer nativo (porta 8189) se non risponde. Necessario per le feature
        migrate (immagini/inpaint/kontext/upscale nativi) — sostituisce gradualmente ComfyUI."""
        try:
            from core import native_driver
            if native_driver.is_up():
                return
            from core.native_launcher import launch
            ok, msg = launch()
            self._log(f"[Nativo] {msg}" if ok else f"[Nativo] auto-avvio saltato: {msg}")
        except Exception as e:                   # noqa: BLE001
            self._log(f"[Nativo] auto-avvio errore: {e}")

    def _maybe_autostart_comfyui(self):
        """Se ComfyUI non risponde su 127.0.0.1:8188, lo avvia in automatico (finestra separata,
        via run_nvidia.bat). Non blocca: `_refresh_status` rileva quando è pronto e carica
        l'inventario. Se è già attivo, non fa nulla."""
        try:
            if self.driver.is_up():
                return
            from core.comfy_launcher import launch
            ok, msg = launch()
            self._log(f"[ComfyUI] {msg}" if ok else f"[ComfyUI] auto-avvio saltato: {msg}")
            if ok and hasattr(self, "status_lbl"):
                self.status_lbl.setText(tr("mw.comfy_autostart_status"))
        except Exception as e:                   # noqa: BLE001
            self._log(f"[ComfyUI] auto-avvio errore: {e}")

    def _ensure_flux2_encoder(self, unet: str | None) -> bool:
        """Pre-check FLUX.2: se il modello è FLUX.2 e il suo encoder testo manca, OFFRE di scaricarlo
        (Comfy-Org via huggingface_hub). Ritorna True se si può procedere (encoder presente o appena
        scaricato), False se l'utente annulla o il download fallisce. Per i non-FLUX.2 ritorna True."""
        try:
            from core.gen_pipeline import is_flux2
            if not unet or not is_flux2(unet):
                return True
            from core.flux2_encoder import resolve_encoder, txt_in_indim
            _enc, err = resolve_encoder(unet, self._text_encoders)
            if not err:
                return True                          # encoder già presente
            from core.encoder_downloader import source_for
            indim = txt_in_indim(unet)
            src = source_for(indim)
            if not src:
                return True                          # nessuna fonte nota → il build mostrerà il messaggio
            repo, _f, local_name, gb, label = src
            ans = QMessageBox.question(
                self, tr("mw.flux2_encoder_missing_title"),
                tr("mw.flux2_encoder_missing_msg").format(label=label, gb=gb, repo=repo),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ans != QMessageBox.Yes:
                return False
            ok, msg, rel = self._run_encoder_download(indim)
            if ok and rel:
                if rel not in self._text_encoders:
                    self._text_encoders.append(rel)
                for p in (self.inpainter, self.genpipe, self.kontext):
                    try:
                        p.set_inventory(text_encoders=self._text_encoders)
                    except Exception:
                        pass
                self._log(f"[FLUX.2] {msg} → {rel}")
                QMessageBox.information(
                    self, tr("mw.encoder_ready_title"),
                    tr("mw.encoder_ready_msg").format(rel=rel))
                return True
            QMessageBox.warning(self, tr("mw.download_failed_title"),
                                tr("mw.download_failed_msg").format(msg=msg))
            return False
        except Exception as e:                       # noqa: BLE001
            self._log(f"[FLUX.2] pre-check encoder errore: {e}")
            return True                              # non bloccare: il build deciderà

    def _run_encoder_download(self, indim: int):
        """Esegue il download in un thread tenendo viva la UI (dialog indeterminato). Ritorna
        (ok, msg, nome_relativo)."""
        import concurrent.futures
        import time as _time
        from core.encoder_downloader import download, source_for
        gb = source_for(indim)[3] if source_for(indim) else 0.0
        dlg = QProgressDialog(
            tr("mw.encoder_download_progress").format(gb=gb), "", 0, 0, self)
        dlg.setWindowTitle(tr("mw.encoder_download_title"))
        dlg.setCancelButton(None)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.show()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(download, indim, log=self._log)
            while not fut.done():
                QApplication.processEvents()
                _time.sleep(0.05)
            res = fut.result()
        dlg.close()
        return res

    def _refresh_status(self):
        from core import native_driver, offline_inventory
        # NO-COMFY: se tutti i moduli sono nativi, NON interrogare ComfyUI (spento di proposito) ad ogni
        # tick → evita la chiamata HTTP bloccante che rallenta la GUI. Durante un job, assumi nativo up
        # (il polling di completion già conferma) per non bloccare il main thread sul /system_stats occupato.
        _all_native = all(backend.is_native(m) for m in ("image", "video", "enhance", "segment", "face"))
        up = False if _all_native else self.driver.is_up()
        if (self._current_pid or self._inpaint_pid or self._seg_pid) and _all_native:
            native_up = True
        else:
            native_up = native_driver.is_up()
        # NO-COMFY: se ComfyUI è assente ma il nativo è attivo, l'inventario viene dal FILESYSTEM.
        ready = up or native_up
        _conn = tr("mw.status_connected") if up else (
            "NATIVO" if native_up else tr("mw.status_offline"))
        _tpl = tr("mw.status_template_ok") if self.engine else tr("mw.status_template_missing")
        self.status_lbl.setText(
            tr("mw.status_line").format(conn=_conn, http=self.driver.cfg.http, tpl=_tpl))
        self.gen_btn.setEnabled(ready and self.engine is not None)
        if ready and not self._inventory_set:
            try:
                src = self.driver if up else offline_inventory   # ComfyUI o filesystem
                diffusion_models = src.models_in("diffusion_models")
                checkpoint_models = src.models_in("checkpoints")
                lora_models = src.models_in("loras")
                self.frame1_tab.set_model_inventory(diffusion_models, checkpoint_models)
                oi = src.object_info()
                if self.engine:
                    self.engine.set_object_info(oi)  # per resolve_everywhere
                self._apply_comfy_inventory(oi)      # disponibilità inpaint + modelli SAM
                self.video_tab.set_model_inventory(
                    diffusion_models, checkpoint_models, self._video_gguf_models(oi),
                    vram_gb=src.gpu_vram_gb())
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
        if self._trellis_pid:
            self._check_trellis_done()
        if self._rig_pid:
            self._check_rig_done()

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
        self._text_encoders = list(_tes)     # cache per l'auto-download encoder FLUX.2
        self.genpipe.set_inventory(text_encoders=_tes, vaes=_vaes, clip_types=_ctypes)
        self.inpainter.set_inventory(text_encoders=_tes)   # encoder FLUX.2 per-variante (inpaint/outpaint)
        self.kontext.set_inventory(text_encoders=_tes)     # encoder FLUX.2 per-variante (integra/colore)
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

    # ---------- gating per edizione (helper) ----------
    def _feature_gate(self, fid: str) -> bool:
        """True se la feature è disponibile nell'edizione; altrimenti mostra un
        messaggio di upsell e ritorna False."""
        if features.is_enabled(fid):
            return True
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, tr("gate.locked.title"),
                                tr("gate.locked.msg").format(edition=features.edition()))
        self._log(f"[gating] funzione '{fid}' non disponibile in '{features.edition()}'.")
        return False

    def _gen_limit_ok(self) -> bool:
        """Limite generazioni della demo (cap.gen_images.params.gen_limit)."""
        from core import usage
        lim = int(features.param("cap.gen_images", "gen_limit", 0) or 0)
        if usage.limit_reached(lim):
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, tr("gate.limit.title"),
                                    tr("gate.limit.msg").format(n=lim))
            return False
        return True

    def _clamp_resolution(self, vals: dict) -> None:
        """Applica il tetto di risoluzione dell'edizione (cap.gen_images.params.max_px)."""
        max_px = int(features.param("cap.gen_images", "max_px", 0) or 0)
        if not max_px:
            return
        for key in ("width", "height"):
            if vals.get(key) and int(vals[key]) > max_px:
                vals[key] = max_px

    # ---------- generazione ----------
    def on_generate(self):
        if not self.engine:
            QMessageBox.warning(self, tr("mw.template_missing_title"), self._engine_err or tr("mw.golden_prompt_not_loaded"))
            return
        vals = {}
        vals.update(self.frame1_tab.get_values())
        vals.update(self.video_tab.get_values())
        vals.update(self.video_tab.get_sampler_values())   # Steps/CFG/Sampler/Seed (ora sotto Video)
        if not vals.get("prompt_positive"):
            QMessageBox.warning(self, tr("mw.empty_prompt_title"), tr("mw.empty_prompt_msg"))
            return
        if not self.video_tab.wants_fixed_seed():
            vals["seed"] = random.randint(0, 2**31)

        # ── V2V (ReTake / Talk): editing di un video sorgente ──
        v2v = self.video_tab.v2v_panel.get_v2v_config() if hasattr(self.video_tab, "v2v_panel") else {"v2v_mode": "normal"}
        self._v2v_keep_audio_src = None
        _mode = v2v.get("v2v_mode", "normal")
        # Lip-sync MultiTalk (Wan2.1) per Talk/Canta → motore nativo dedicato, salta il flusso ComfyUI
        if _mode in ("sing", "talk") and self._maybe_multitalk(vals, v2v, _mode):
            return
        if _mode != "normal":
            src = v2v.pop("v2v_video_path", None)
            if _mode == "talk" and not src:
                # TALK SENZA VIDEO → genera il parlato DALL'IMMAGINE (Frame 1) come I2V normale:
                # niente maschera V2V; voce dall'albero del tab Audio, dialogo dal prompt principale.
                if not self.frame1_tab.image_path:
                    QMessageBox.warning(self, tr("mw.talk_title"), tr("mw.talk_need_input_msg"))
                    return
                self._log("[Talk] nessun video → genero dall'immagine (I2V + lip-sync da audio/prompt).")
                # NON impostiamo v2v_mode → prosegue il percorso I2V normale (use_image più sotto).
            else:
                errs = self.video_tab.v2v_panel.validation_errors()
                if errs:
                    QMessageBox.warning(self, tr("mw.missing_for_mode_title").format(mode=_mode.upper()),
                                        tr("mw.missing_for_mode_msg") + "\n• ".join(errs))
                    return
                vname = self._upload_video(src) if src else None
                if not vname:
                    QMessageBox.warning(self, tr("mw.source_video_title"), tr("mw.source_video_upload_failed_msg"))
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
                # ReTake NATIVO: rigenera la sezione [start,end] del video sorgente col worker LTX-2 (RetakePipeline)
                if _mode == "retake" and backend.is_native("video"):
                    return self._retake_native(vals, src, v2v)
                vals.update(v2v)

        # ── STORYBOARD multi-scena: immagini-keyframe a tempo → video senza montaggio ──
        sb_panel = getattr(self.video_tab, "storyboard_panel", None)
        scenes = sb_panel.get_storyboard() if sb_panel else []
        if scenes:
            errs = sb_panel.validation_errors()
            if errs:
                QMessageBox.warning(self, tr("mw.sb_title", "Storyboard"),
                                    tr("mw.sb_invalid", "Storyboard non valido:") + "\n• " + "\n• ".join(errs))
                return
            if backend.is_native("video"):
                # NATIVO: niente upload ComfyUI, si usano i path LOCALI (→ keyframe in _generate_video_native)
                vals["storyboard"] = [{"image": sc["image"], "second": sc["second"],
                                       "strength": sc["strength"]} for sc in scenes]
            else:
                uploaded = []
                for sc in scenes:
                    name = self._upload_local(sc["image"])
                    if not name:
                        QMessageBox.warning(self, tr("mw.sb_title", "Storyboard"),
                                            tr("mw.sb_upload_failed", "Upload immagine scena fallito: {img}").format(img=sc["image"]))
                        return
                    uploaded.append({"image": name, "second": sc["second"], "strength": sc["strength"]})
                vals["storyboard"] = uploaded
            self._log(tr("mw.sb_log", "[Storyboard] {n} scene-keyframe → interpolazione multi-scena.").format(n=len(scenes)))

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
                QMessageBox.information(self, tr("mw.face_restore_unavailable_title"),
                    tr("mw.face_restore_unavailable_msg"))
                self._log("[Restauro volti] nodo non installato → salto il passaggio.")

        # aggiorna stato per i conflitti (audio, controlnet, modello, nsfw)
        audio = self.audio_tab.get_audio_config()
        # CONSENSO TTS: se è stato generato un file TTS qui (audio_tab) ma l'utente NON
        # ha esplicitamente impostato la modalità "user", chiedo se vuole usarlo come
        # traccia. Se rifiuta, resta la modalità corrente (es. native → audio dal prompt).
        pending_tts = self.audio_tab.has_pending_tts() if hasattr(self.audio_tab, "has_pending_tts") else None
        if pending_tts and audio.get("audio_mode") != "user":
            ans = QMessageBox.question(self, tr("mw.tts_available_title"),
                tr("mw.tts_available_msg").format(name=Path(pending_tts).name),
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
                QMessageBox.warning(self, tr("mw.tts_synth_failed_title"),
                    tr("mw.tts_synth_failed_msg").format(error=e))
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
            QMessageBox.warning(self, tr("mw.conflicts_title"), "\n".join("• " + c.message for c in blocking))
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
                box.setWindowTitle(tr("mw.trigger_missing_title"))
                box.setText(tr("mw.trigger_missing_msg") + "\n".join(warns))
                b_ins = box.addButton(tr("mw.btn_insert_and_generate"), QMessageBox.AcceptRole)
                b_gen = box.addButton(tr("mw.btn_generate_anyway"), QMessageBox.YesRole)
                box.addButton(tr("mw.btn_cancel"), QMessageBox.RejectRole)
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
        # BACKEND NATIVO (LTX-2.3 GGUF, zero ComfyUI): ramo additivo, attivo solo se backend_config
        # video=native. Bypassa la costruzione del workflow ComfyUI/TemplateEngine.
        if backend.is_native("video"):
            self._log("Backend video: NATIVO (LTX-2.3 GGUF)")
            return self._generate_video_native(vals, audio, loras, use_image)
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
            self.gen_dialog.start(prompt, tr("mw.gen_video_label"))
        except Exception as e:
            QMessageBox.critical(self, tr("mw.send_error_title"), str(e))
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
        # NATIVO/no-comfy: gli handler nativi leggono i path LOCALI → niente upload, ritorna il path.
        if not self.driver.is_up():
            return path
        r = self.driver.upload_image(path)
        name, sub = r.get("name", ""), r.get("subfolder", "")
        return f"{sub}/{name}" if sub else name

    def generate_inpaint(self, mode: str):
        if not self._feature_gate("cap.inpaint"):
            return
        from core import backend
        if not backend.is_native("image") and not self.driver.is_up():
            QMessageBox.warning(self, tr("mw.comfy_offline_title"), tr("mw.comfy_offline_generate_msg")); return
        if not self.frame1_tab.image_path:
            QMessageBox.warning(self, tr("mw.no_image_title"), tr("mw.load_photo_canvas_msg")); return
        if self._inpaint_pid:
            QMessageBox.information(self, tr("mw.wait_title"), tr("mw.image_gen_in_progress_msg")); return
        cfg = self.frame1_tab.get_inpaint_config()
        if cfg.get("unet") and not cfg.get("available", True):
            QMessageBox.warning(self, tr("mw.model_not_installed_title"),
                                tr("mw.inpaint_model_not_installed_msg").format(unet=cfg.get('unet')))
            return
        # FLUX.2: se manca l'encoder del modello, offri di scaricarlo prima di costruire.
        if not self._ensure_flux2_encoder(cfg.get("unet")):
            return
        ptxt = (cfg["inpaint_prompt"] if mode == "inpaint" else cfg["outpaint_prompt"]) \
            or self.frame1_tab.get_values().get("prompt_positive", "")
        retention = cfg.get("strength", 0.75)
        denoise = 1.0 if mode == "outpaint" else max(0.5, min(1.0, 1.0 - 0.4 * retention))
        # guard: serve un'area da rigenerare, altrimenti il modello non fa nulla
        if mode == "outpaint" and not self.frame1_tab.canvas.has_outpaint_area():
            QMessageBox.information(self, tr("mw.nothing_to_extend_title"),
                                    tr("mw.nothing_to_extend_msg")); return
        if mode == "inpaint" and not self.frame1_tab.canvas.has_mask():
            QMessageBox.information(self, tr("mw.no_mask_title"),
                                    tr("mw.no_mask_msg")); return
        try:
            exp = self.frame1_tab.canvas.export_inpaint(OUTPUT_DIR, mode)
            self._inpaint_bbox = exp.get("bbox")    # per rimettere l'inpaint al suo posto
            self._inpaint_before = exp.get("image") # per confronto prima/dopo nel dialog
            seed = cfg["seed"] if cfg.get("seed") is not None else random.randint(0, 2**31)
            steps = cfg.get("steps") or 20
            negative = cfg.get("negative") or self.frame1_tab.get_values().get("prompt_negative", "")
            # ── BACKEND NATIVO (FLUX.1 Fill): niente ComfyUI. FLUX.2 Klein e i checkpoint
            #    restano su ComfyUI finché non migrati (vedi docs/zero_comfy.md Fase A).
            from core import backend
            from core.gen_pipeline import is_flux2
            _unet = cfg.get("unet") or ""
            if (backend.is_native("image") and not cfg.get("checkpoint")
                    and _unet and not is_flux2(_unet)):
                self._start_native_inpaint(exp, _unet, ptxt, negative, steps, seed, retention, mode)
                return
            img_name = self._upload_local(exp["image"])
            mask_name = self._upload_local(exp["mask"])
            # feather/overlap della maschera per fondere la giunzione (no linee di separazione).
            # Outpaint: la crescita DEVE superare la sfumatura, altrimenti la sfumatura sborda nel
            # bordo lasciando una fascia scura non riempita (distacco). grow(40) > feather(24) →
            # bordo pieno fino alla foto, transizione tutta dentro la foto (overlap).
            feather, grow = (24, 40) if mode == "outpaint" else (12, 0)
            prompt = self.inpainter.build(
                img_name, ptxt, mode=mode, mask_file=mask_name, denoise=denoise,
                seed=seed, lora=cfg.get("lora"), feather=feather, mask_grow=grow,
                unet=cfg.get("unet"), checkpoint=cfg.get("checkpoint"),
                steps=steps, negative=negative, retention=retention)
            self._inpaint_pid = self.driver.submit(prompt)
            self._inpaint_mode = mode
            self.gen_dialog.start(prompt, tr("mw.outpainting_label") if mode == "outpaint" else tr("mw.inpainting_label"))
            self._log(f"{mode}: inviato (id={self._inpaint_pid}, steps={steps}, seed={seed}, denoise={denoise:.2f}).")
        except Exception as e:
            QMessageBox.critical(self, tr("mw.error_with_mode_title").format(mode=mode), str(e)); self._log(f"[ERR] {mode}: {e}")

    def generate_integrate(self):
        """Integra l'oggetto incollato nella scena con FLUX **Kontext** (o FLUX.2): il composito
        (scena + prop) è dato come RIFERIMENTO + prompt → il modello RIGENERA/armonizza (luce,
        ombre, prospettiva), non incolla l'originale. Prompt opzionale (default auto)."""
        from core import backend
        if not backend.is_native("image") and not self.driver.is_up():
            QMessageBox.warning(self, tr("mw.comfy_offline_title"), tr("mw.comfy_offline_generate_msg")); return
        if not self.frame1_tab.image_path:
            QMessageBox.warning(self, tr("mw.no_image_title"), tr("mw.load_base_photo_canvas_msg")); return
        if self._inpaint_pid:
            QMessageBox.information(self, tr("mw.wait_title"), tr("mw.gen_in_progress_msg")); return
        if not self.frame1_tab.canvas.has_layers():
            QMessageBox.information(self, tr("mw.no_image_to_integrate_title"),
                                    tr("mw.no_image_to_integrate_msg")); return
        cfg = self.frame1_tab.get_inpaint_config()
        engine = cfg.get("integrate_engine", "kontext")
        # Modello deciso dall'utente nel combo (contestuale all'engine selezionato).
        # Sia Kontext che FLUX.2 leggono dallo stesso campo `integrate_model`. Se vuoto,
        # KontextPipeline ricade sul default hardcoded `FLUX_Kontext_Dev.safetensors`.
        model = cfg.get("integrate_model")
        if engine == "flux2" and not model:
            QMessageBox.warning(self, tr("mw.flux2_model_missing_title"),
                                tr("mw.flux2_model_missing_integrate_msg")); return
        # FLUX.2: se manca l'encoder del modello, offri di scaricarlo prima di costruire.
        if not self._ensure_flux2_encoder(model):
            return
        ptxt = cfg.get("integrate_prompt") or ""        # opzionale: default auto in KontextPipeline
        try:
            exp = self.frame1_tab.canvas.export_integrate(OUTPUT_DIR)
            if not exp.get("has_region"):
                QMessageBox.information(self, tr("mw.nothing_to_integrate_title"),
                                        tr("mw.nothing_to_integrate_msg")); return
            self._inpaint_bbox = None
            self._inpaint_before = exp.get("image")     # confronto prima/dopo (composito di partenza)
            img_name = self._upload_local(exp["image"])
            # Campi dedicati alla sezione Integra (separati da quelli di inpaint/outpaint).
            int_seed_cfg = cfg.get("integrate_seed")
            seed = int_seed_cfg if int_seed_cfg is not None else random.randint(0, 2**31)
            steps = cfg.get("integrate_steps") or 20
            negative = cfg.get("integrate_negative") or ""
            aux = cfg.get("integrate_puzzle_lora") or None    # puzzle_lora opzionale (solo Kontext)
            guidance = cfg.get("integrate_guidance", 2.5)
            # ── BACKEND NATIVO (Kontext FLUX.1): niente ComfyUI. FLUX.2 e puzzle_lora → ComfyUI.
            from core import backend
            if backend.is_native("image") and engine == "kontext" and not aux:
                self._start_native_kontext(exp, model, ptxt, negative, steps, seed, guidance)
                return
            prompt = self.kontext.integrate(img_name, ptxt, engine=engine, model=model,
                                            negative=negative, seed=seed, steps=steps,
                                            guidance=guidance, aux_lora=aux)
            self._inpaint_pid = self.driver.submit(prompt)
            self._inpaint_mode = "integrazione"
            self.gen_dialog.start(prompt, tr("mw.integrate_label").format(engine='FLUX.2' if engine == 'flux2' else 'Kontext'))
            self._log(f"Integrazione {engine}: inviata (steps={steps}, seed={seed}, puzzle_lora={'sì' if aux else 'no'}).")
        except Exception as e:
            QMessageBox.critical(self, tr("mw.integrate_error_title"), str(e)); self._log(f"[ERR] integrate: {e}")

    def generate_compose(self):
        """Componi 2+ foto SEPARATE in un'unica immagine senza incollarle (niente occlusione):
        multi-reference (catena ReferenceLatent) o stitch (affianca). FLUX.1 (Kontext) o FLUX.2.
        Maschere opzionali per-foto (prep_masked) per indicare cosa usare."""
        from core import backend
        if not backend.is_native("image") and not self.driver.is_up():
            QMessageBox.warning(self, tr("mw.comfy_offline_title"), tr("mw.comfy_offline_generate_msg")); return
        if self._inpaint_pid:
            QMessageBox.information(self, tr("mw.wait_title"), tr("mw.gen_in_progress_msg")); return
        cfg = self.frame1_tab.get_compose_config()
        imgs = cfg.get("images") or []
        if len(imgs) < 2:
            QMessageBox.information(self, tr("f1.compose.group"), tr("mw.compose_need_two_msg")); return
        engine = cfg.get("engine", "kontext")
        model = cfg.get("model")
        if engine == "flux2" and not model:
            QMessageBox.warning(self, tr("mw.flux2_model_missing_title"),
                                tr("mw.flux2_model_missing_integrate_msg")); return
        if not self._ensure_flux2_encoder(model):
            return
        masks = cfg.get("masks") or {}
        # ── BACKEND NATIVO (FLUX.1 Kontext): affianca le foto (pre-mascherate) in un unico
        #    riferimento (stitch lato Python) → Kontext nativo. Il multi-reference VERO (N latenti
        #    separati) richiede FLUX.2 → resta su ComfyUI (posticipato).
        from core import backend
        if backend.is_native("image") and engine == "kontext":
            try:
                import os
                from PIL import Image
                prepped = []
                for f in imgs:
                    src = self.kontext.prep_masked(f, masks[f], OUTPUT_DIR) if f in masks else f
                    prepped.append(Image.open(src).convert("RGB"))
                h = min(im.height for im in prepped)
                row = [im.resize((max(1, int(im.width * h / im.height)), h)) for im in prepped]
                comp = Image.new("RGB", (sum(im.width for im in row), h), (255, 255, 255))
                x = 0
                for im in row:
                    comp.paste(im, (x, 0)); x += im.width
                cpath = os.path.join(OUTPUT_DIR, "compose_stitch.png"); comp.save(cpath)
                self._inpaint_before = None
                self._start_native_kontext({"image": cpath}, model,
                    cfg.get("prompt") or "combine these reference images into one coherent scene",
                    "", 20, random.randint(0, 2**31), 2.5)
                return
            except Exception as e:                       # noqa: BLE001
                self._log(f"[compose nativo] fallback ComfyUI: {e}")
        try:
            uploaded = []
            for f in imgs:
                src = f
                if f in masks:                          # applica la maschera "cosa usare" a monte
                    src = self.kontext.prep_masked(f, masks[f], OUTPUT_DIR)
                uploaded.append(self._upload_local(src))
            seed = random.randint(0, 2**31)
            graph = self.kontext.compose(uploaded, cfg.get("prompt") or "", method=cfg.get("method", "multiref"),
                                         engine=engine, model=model, seed=seed)
            self._inpaint_pid = self.driver.submit(graph)
            self._inpaint_mode = "composizione"
            self._inpaint_bbox = None
            self._inpaint_before = None
            self.gen_dialog.start(graph, tr("mw.compose_label").format(
                n=len(imgs), method=cfg.get("method", "multiref")))
            self._log(f"Composizione {engine} ({cfg.get('method')}): {len(imgs)} foto, seed={seed}.")
        except Exception as e:
            QMessageBox.critical(self, tr("mw.integrate_error_title"), str(e)); self._log(f"[ERR] compose: {e}")

    def generate_recolor(self):
        """Cambia colore (Kontext/FLUX.2 prompt-based): ricolora il soggetto mantenendo texture/luce."""
        if not self._feature_gate("cap.inpaint"):
            return
        from core import backend
        if not backend.is_native("image") and not self.driver.is_up():
            QMessageBox.warning(self, tr("mw.comfy_offline_title"), tr("mw.comfy_offline_generate_msg")); return
        if not self.frame1_tab.image_path:
            QMessageBox.warning(self, tr("mw.no_image_title"), tr("mw.load_or_gen_image_canvas_msg")); return
        if self._inpaint_pid:
            QMessageBox.information(self, tr("mw.wait_title"), tr("mw.gen_in_progress_msg")); return
        cc = self.frame1_tab.get_color_config()
        if not cc["target"]:
            QMessageBox.warning(self, tr("mw.color_missing_title"),
                                tr("mw.color_missing_msg")); return
        engine = cc["engine"]; model = None
        if engine == "flux2":
            model = self.frame1_tab.get_inpaint_config().get("integrate_model")
            if not model:
                QMessageBox.warning(self, tr("mw.flux2_model_missing_title"),
                                    tr("mw.flux2_model_missing_inpaint_tab_msg")); return
        # ── BACKEND NATIVO (recolor Kontext FLUX.1): niente ComfyUI. FLUX.2 → ComfyUI.
        from core import backend
        if backend.is_native("image") and engine == "kontext":
            what = (cc.get("subject") or "").strip() or "the indicated subject"
            rprompt = (f"Recolor {what} to {cc['target']}. Keep the original texture, material, "
                       f"lighting and shadows; change only the color.")
            self._inpaint_before = self.frame1_tab.image_path
            self._start_native_kontext({"image": self.frame1_tab.image_path}, None, rprompt, "",
                                       cc["steps"], random.randint(0, 2**31), 2.5)
            return
        try:
            img_name = self._upload_local(self.frame1_tab.image_path)
            self._inpaint_bbox = None
            self._inpaint_before = self.frame1_tab.image_path     # confronto prima/dopo
            seed = random.randint(0, 2**31)
            prompt = self.kontext.recolor(img_name, cc["target"], subject=cc["subject"],
                                          engine=engine, model=model, steps=cc["steps"], seed=seed)
            self._inpaint_pid = self.driver.submit(prompt)
            self._inpaint_mode = "cambio colore"
            self.gen_dialog.start(prompt, tr("mw.recolor_label").format(engine='FLUX.2' if engine == 'flux2' else 'Kontext'))
            self._log(f"Cambio colore: {cc['subject'] or 'soggetto'} → {cc['target']} ({engine}).")
        except Exception as e:
            QMessageBox.critical(self, tr("mw.recolor_error_title"), str(e)); self._log(f"[ERR] recolor: {e}")

    def generate_video_enhance(self):
        """Definizione video: restore + super-risoluzione (+ opz. FPS) sull'ultimo video generato."""
        import shutil
        if backend.is_native("video"):
            return self._enhance_video_native()
        if not self.driver.is_up():
            QMessageBox.warning(self, tr("mw.comfy_offline_title"), tr("mw.comfy_offline_enhance_msg")); return
        cfg = self.video_def_tab.get_config()
        if not cfg.get("video"):
            QMessageBox.warning(self, tr("mw.no_video_title"), tr("mw.no_video_msg")); return
        if self._current_pid or self._inpaint_pid:
            QMessageBox.information(self, tr("mw.wait_title"), tr("mw.gen_in_progress_msg")); return
        src = Path(cfg["video"])
        if not src.exists():
            QMessageBox.warning(self, tr("mw.file_missing_title"), tr("mw.video_not_found_msg").format(path=src)); return
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
            self.gen_dialog.start(prompt, tr("mw.video_def_label").format(engine=cfg['engine']))
            self._log(f"Definizione video: {cfg['engine']} su {src.name} ({len(prompt)} nodi). "
                      f"Il modello SeedVR2 può scaricarsi al primo uso.")
        except Exception as e:
            QMessageBox.critical(self, tr("mw.video_def_error_title"), str(e)); self._log(f"[ERR] enhance: {e}")

    def generate_video_extend(self):
        """Prolunga video: estensione LTX-2.3 nativa multi-segmento sull'ultimo video (o caricato)."""
        import shutil
        if backend.is_native("video"):
            return self._extend_video_native()
        if not self.driver.is_up():
            QMessageBox.warning(self, tr("mw.comfy_offline_title"), tr("mw.comfy_offline_extend_msg")); return
        cfg = self.video_extend_tab.get_config()
        if not cfg.get("video"):
            QMessageBox.warning(self, tr("mw.no_video_title"), tr("mw.no_video_msg")); return
        if self._current_pid or self._inpaint_pid:
            QMessageBox.information(self, tr("mw.wait_title"), tr("mw.gen_in_progress_msg")); return
        src = Path(cfg["video"])
        if not src.exists():
            QMessageBox.warning(self, tr("mw.file_missing_title"), tr("mw.video_not_found_msg").format(path=src)); return
        if not cfg["prompt"]:
            QMessageBox.warning(self, tr("mw.prompt_missing_title"), tr("mw.extend_prompt_missing_msg")); return
        try:
            comfy_input = Path(r"E:\AI\Comfy_Nexus\ComfyUI\input"); comfy_input.mkdir(parents=True, exist_ok=True)
            dest = comfy_input / src.name
            if str(dest) != str(src):
                shutil.copy2(src, dest)
            v = self.video_tab.get_values()
            # Regola: <=20s → 1 spezzone (no giunzioni); >20s → spezzoni da 18s con sovrapposizione.
            n_seg, seg_seconds = self.extender.plan_segments(
                cfg["add_seconds"], cfg.get("seg_seconds"))
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
                add_seconds=cfg["add_seconds"], seg_seconds=seg_seconds, segments=n_seg,
                overlap_frames=cfg["overlap_frames"], context_frames=cfg["context_frames"],
                overlap_strength=float(ctx_strength),
                steps=int(sv.get("steps") or cfg["steps"]), cfg=float(sv.get("cfg") or 1.0),
                seed=random.randint(0, 2**31), negative=v.get("prompt_negative", ""),
                unet=ext_model, sampler=sv.get("sampler") or "euler_ancestral_cfg_pp",
                base_shift=float(sv.get("ltxv_base_shift", 0.95)),
                max_shift=float(sv.get("ltxv_max_shift", 2.05)),
                clip1=sv.get("text_encoder"))
            self._current_pid = self.driver.submit(prompt)
            self.gen_dialog.start(prompt, tr("mw.extend_video_label"))
            self._log(f"Prolunga video: +{cfg['add_seconds']:.0f}s in {n_seg} spezzoni su {src.name} "
                      f"({len(prompt)} nodi).")
        except Exception as e:
            QMessageBox.critical(self, tr("mw.extend_video_error_title"), str(e)); self._log(f"[ERR] extend: {e}")

    def generate_face_restore(self):
        """Restauro volti per-frame (CodeFormer/GFPGAN) sul video del tab Migliora qualità."""
        import shutil
        cfg0 = self.video_def_tab.get_face_restore_config()
        if backend.is_native("face"):
            return self._face_restore_native(cfg0)
        if not self.driver.is_up():
            QMessageBox.warning(self, tr("mw.comfy_offline_title"), tr("mw.comfy_offline_process_msg")); return
        if self._current_pid or self._inpaint_pid:
            QMessageBox.information(self, tr("mw.wait_title"), tr("mw.processing_in_progress_msg")); return
        cfg = self.video_def_tab.get_face_restore_config()
        if not cfg.get("video"):
            QMessageBox.warning(self, tr("mw.no_video_title"), tr("mw.load_or_gen_video_msg")); return
        src = Path(cfg["video"])
        if not src.exists():
            QMessageBox.warning(self, tr("mw.file_missing_title"), tr("mw.video_not_found_msg").format(path=src)); return
        try:
            from core.face_restore_pipeline import node_installed, ensure_model
            if not node_installed():
                QMessageBox.warning(self, tr("mw.node_missing_title"),
                    tr("mw.face_restore_node_missing_msg")); return
            ok, msg = ensure_model(cfg["model"], log=self._log)
            if not ok:
                QMessageBox.warning(self, tr("mw.restore_model_missing_title"), msg); return
            comfy_input = Path(r"E:\AI\Comfy_Nexus\ComfyUI\input"); comfy_input.mkdir(parents=True, exist_ok=True)
            dest = comfy_input / src.name
            if str(dest) != str(src):
                shutil.copy2(src, dest)
            v = self.video_tab.get_values()
            prompt = self.facerestorer.build(
                src.name, fps=int(v.get("fps", 24)), fidelity=cfg["fidelity"],
                model=cfg["model"], detection=cfg["detection"])
            self._current_pid = self.driver.submit(prompt)
            self.gen_dialog.start(prompt, tr("mw.face_restore_label"))
            self._log(f"Restauro volti su {src.name} (fidelity {cfg['fidelity']:.2f}, {cfg['model']}).")
        except Exception as e:
            QMessageBox.critical(self, tr("mw.face_restore_error_title"), str(e)); self._log(f"[ERR] face restore: {e}")

    def generate_image(self):
        from core import backend
        if not backend.is_native("image") and not self.driver.is_up():
            QMessageBox.warning(self, tr("mw.comfy_offline_title"), tr("mw.comfy_offline_generate_msg")); return
        if self._inpaint_pid:
            QMessageBox.information(self, tr("mw.wait_title"), tr("mw.image_gen_in_progress_msg")); return
        if not self._gen_limit_ok():
            return
        cfg = self.frame1_tab.get_gen_config()
        if not cfg["prompt"]:
            QMessageBox.warning(self, tr("mw.empty_prompt_title"), tr("mw.empty_image_prompt_msg")); return
        if not cfg["model"] or cfg["model"].startswith("("):
            QMessageBox.warning(self, tr("mw.no_model_title"), tr("mw.no_model_arch_msg").format(arch=cfg['arch'])); return
        # FLUX.2: se manca l'encoder del modello, offri di scaricarlo prima di costruire.
        if cfg.get("arch") == "flux" and not self._ensure_flux2_encoder(cfg["model"]):
            return
        v = self.video_tab.get_values()
        self._clamp_resolution(v)        # tetto risoluzione per edizione (es. demo 768px)
        loras = [{"name": l["name"], "strength": l["strength"]} for l in cfg.get("loras", [])]
        # FLUX: distinguo checkpoint vs UNET (CheckpointLoaderSimple vs UNETLoader)
        ckpts = set(getattr(self.frame1_tab, "_gen_checkpoint_models", []) or [])
        is_checkpoint = cfg["arch"] == "flux" and cfg["model"] in ckpts
        # Seed dalla UI: None = random, altrimenti il valore fisso scelto dall'utente
        seed = cfg.get("seed") if cfg.get("seed") is not None else random.randint(0, 2**31)
        # Negativo dalla sezione gen; fallback al prompt negativo globale del tab Video
        negative = cfg.get("negative") or self.frame1_tab.get_values().get("prompt_negative", "")
        # Backend astratto: se il modulo "image" è su "native", usa l'InferenceServer (no ComfyUI).
        from core import backend
        if backend.is_native("image"):
            self._start_native_image(cfg, v, loras, seed, negative); return
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
            from core import usage
            usage.increment()                 # conteggio per il limite della demo
            self._inpaint_mode = "generazione immagine"
            self._inpaint_bbox = None
            self._inpaint_before = None      # da zero: nessun confronto
            self.gen_dialog.start(prompt, tr("mw.image_gen_label").format(arch=cfg['arch']))
            self._log(f"Generazione immagine ({cfg['arch']} · {cfg['model']}) inviata…")
        except Exception as e:
            QMessageBox.critical(self, tr("mw.gen_error_title"), str(e)); self._log(f"[ERR] gen: {e}")

    def _start_native_image(self, cfg, v, loras, seed, negative):
        """Genera un'immagine via backend nativo (InferenceServer) invece di ComfyUI."""
        from core import native_driver
        if not native_driver.is_up():
            QMessageBox.warning(self, tr("mw.native_off_title", "Backend nativo offline"),
                tr("mw.native_off_msg", "L'InferenceServer nativo (porta 8189) non risponde. "
                   "Avvialo, oppure rimetti il backend 'image' su ComfyUI."))
            return
        params = {"model": cfg["model"], "arch": cfg["arch"], "prompt": cfg["prompt"],
                  "negative": negative, "width": v["width"], "height": v["height"],
                  "steps": cfg["steps"], "guidance": cfg.get("cfg") or 3.5,
                  "seed": seed, "loras": loras,
                  "pulid_face": cfg.get("pulid_face"), "pulid_strength": cfg.get("pulid_strength", 0.8)}
        self._log("Generazione immagine (backend NATIVO) inviata…")
        self._native_worker = _NativeImageWorker(params)
        self._enqueue(tr("queue.job.image", "Immagine"), self._native_worker, self._on_native_image_done)

    def _on_native_image_done(self, res: dict):
        if res.get("status") != "success":
            self._log(f"[native] errore: {res.get('error','')[:300]}")
            QMessageBox.critical(self, tr("mw.gen_error_title", "Errore"), str(res.get("error", ""))[:1000])
            return
        out = (res.get("output") or [None])[0]
        self._log(f"[native] immagine generata: {out}")
        if out:
            # dialog anteprima + Accetta / Revert (generazione da zero: nessun "prima" da confrontare)
            self._inpaint_mode = tr("mw.mode_generate", "generazione")
            self._inpaint_bbox = None
            self._show_inpaint_result(out)

    def _generate_video_native(self, vals: dict, audio: dict, loras: list, use_image: bool):
        """Generazione video NATIVA (LTX-2.3 GGUF, zero ComfyUI) — ramo attivo solo se
        backend_config video=native. Costruisce i params dal tab Video/Avanzate e lancia il worker
        ltx2 (auto-GGUF + chunking per i lunghi). Audio (native/user) → a2v (canto/lip-sync)."""
        import time as _t
        fps = int(self.video_tab.fps.value())
        length_s = float(self.video_tab.length.value()) if hasattr(self.video_tab, "length") else 5.0
        nf = max(9, int(round(length_s * fps)) // 8 * 8 + 1)
        out = str(OUTPUT_DIR / f"ltx_native_{int(_t.time())}.mp4")
        params = {
            "prompt": vals.get("prompt_positive", ""),
            "negative": vals.get("prompt_negative", "worst quality, blurry"),
            "width": int(self.video_tab.width.value()),
            "height": int(self.video_tab.height.value()),
            "num_frames": nf, "fps": fps,
            "steps": int(self.video_tab.sampler_seed.steps.value()),
            "guidance": float(self.video_tab.sampler_seed.cfg.value()),
            "seed": int(self.video_tab.sampler_seed.seed.value()),
            "offload": "cpu", "loras": loras, "output": out,
        }
        # MODELLO scelto nel tab Video → al worker. Senza questo il worker usava SEMPRE il checkpoint
        # di default → "funziona solo un modello" (qualunque scelta nel dropdown era ignorata).
        if vals.get("model_kind") == "gguf" and vals.get("model_gguf"):
            params["gguf_transformer"] = vals["model_gguf"]   # base checkpoint resta default per vae/gemma
        elif vals.get("model") and vals.get("model") not in ("", "auto"):
            params["model"] = vals["model"]
        if use_image and self.frame1_tab.image_path:
            params["image_path"] = self.frame1_tab.image_path
            params["ref_strength"] = float(vals.get("ref_strength", 1.0))
        # audio reale → audio→video (canto/lip-sync) nativo
        af = audio.get("audio_file")
        if af and os.path.exists(af):
            params["audio_path"] = af
            # a2v: il video deve coprire TUTTA la durata dell'audio (altrimenti l'immagine si "congela"
            # all'ultimo frame mentre la canzone continua). Ricalcola i frame dalla durata dell'audio e
            # limita l'audio muxato alla durata del video (coerenza durate).
            adur = self._probe_duration(af) or 0.0
            if adur > length_s:
                length_s = adur
                nf = max(9, int(round(length_s * fps)) // 8 * 8 + 1)
                params["num_frames"] = nf
                self._log(f"[nativo] a2v: durata video allineata all'audio ({adur:.1f}s, {nf} frame)")
            params["audio_max_duration"] = length_s
        # video lungo → chunking memory-bounded (chunk da ~5s)
        if nf > 150:
            params["chunk_frames"] = 121
        # CONTROLNET (#24): preprocessa il ref-video (canny/depth/pose) e passalo al worker come guida
        cnet = self.controlnet_tab.get_values()
        if cnet.get("controlnet_enabled") and cnet.get("controlnet_ref_video"):
            try:
                from core import controlnet_preprocess as cnp
                kind = {0: "canny", 1: "depth", 2: "pose"}.get(int(cnet.get("controlnet_type", 0)), "canny")
                ctrl_out = str(OUTPUT_DIR / f"_ctrl_{kind}_{int(_t.time())}.mp4")
                cnp.preprocess_video(cnet["controlnet_ref_video"], kind, ctrl_out, fps=fps)
                params["control_video"] = ctrl_out
                params["control_type"] = kind
                params["control_strength"] = float(cnet.get("controlnet_strength", 0.8) or 0.8)
                params.pop("image_path", None)
                self._log(f"[nativo] ControlNet {kind} → guida video preprocessata")
            except Exception as e:                          # noqa: BLE001
                self._log(f"[nativo] ControlNet preprocess fallito ({e}) → genero senza guida")
        # STORYBOARD multi-scena → keyframe a tempo (il worker li condiziona a frame diversi)
        sb = vals.get("storyboard")
        if sb:
            kfs = [{"path": sc["image"], "frame": max(0, min(nf - 1, int(round(float(sc["second"]) * fps)))),
                    "strength": float(sc.get("strength", 1.0))} for sc in sb if sc.get("image")]
            if kfs:
                params["keyframes"] = kfs
                params.pop("image_path", None)        # i keyframe sostituiscono il singolo frame iniziale
                self._log(f"[nativo] storyboard: {len(kfs)} keyframe a tempo → video multi-scena")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._log(f"[nativo] video LTX-2 GGUF: {params['width']}x{params['height']} {nf} frame "
                  f"{'+audio' if params.get('audio_path') else ''} {'(chunk)' if params.get('chunk_frames') else ''}")
        self._ltx2_worker = _NativeLtx2Worker(params)
        self._enqueue(tr("queue.job.video", "Video"), self._ltx2_worker, self._on_multitalk_done,
                      on_progress=lambda s: self._log("[nativo] " + s))

    def _retake_native(self, vals: dict, src: str, v2v: dict):
        """ReTake NATIVO (#24): rigenera la sezione [start,end] del video sorgente (RetakePipeline LTX-2).
        Zero ComfyUI."""
        import time as _t
        if not src or not os.path.exists(src):
            QMessageBox.warning(self, tr("mw.source_video_title", "Video"), tr("mw.no_video_msg", "Scegli un video.")); return
        out = str(OUTPUT_DIR / f"retake_{int(_t.time())}.mp4")
        params = {
            "retake": True, "video_path": src, "prompt": vals.get("prompt_positive", ""),
            "negative": vals.get("prompt_negative", "") or "worst quality, blurry",
            "start_time": float(v2v.get("v2v_start", 0.0)), "end_time": float(v2v.get("v2v_end", 5.0)),
            "regenerate_video": True, "regenerate_audio": bool(v2v.get("v2v_regen_audio", False)),
            "steps": int(self.video_tab.sampler_seed.steps.value()), "guidance": float(self.video_tab.sampler_seed.cfg.value()),
            "seed": int(self.video_tab.sampler_seed.seed.value()), "offload": "cpu", "output": out,
        }
        self._log(f"[nativo] ReTake [{params['start_time']:.1f}–{params['end_time']:.1f}]s su {Path(src).name}…")
        self._ltx2_worker = _NativeLtx2Worker(params)
        self._enqueue(tr("queue.job.video_retake", "Video ReTake"), self._ltx2_worker, self._on_multitalk_done,
                      on_progress=lambda s: self._log("[nativo] " + s))

    def _gen_video_edit_ai(self, cfg: dict):
        """#24/#59/#60 Modifica Video AI: edit-anything / inpaint (SAM2 track) / style — worker async."""
        self.video_edit_ai_tab.log.appendPlainText(tr("vedit.start", "[edit] avvio {m}…").format(m=cfg["mode"]))
        self._edit_ai_worker = _VideoEditAIWorker(cfg, str(OUTPUT_DIR))
        self._enqueue(tr("queue.job.video_edit", "Modifica Video AI"), self._edit_ai_worker,
                      self._on_video_edit_ai_done, on_progress=self.video_edit_ai_tab.log.appendPlainText)

    def _on_video_edit_ai_done(self, res: dict):
        if res.get("status") == "success":
            out = (res.get("output") or [""])[0]
            self.video_edit_ai_tab.log.appendPlainText(tr("vedit.done", "✅ Pronto: {o}").format(o=out))
            self._last_video_path = out
        else:
            self.video_edit_ai_tab.log.appendPlainText(tr("vedit.fail", "❌ Errore: {e}").format(e=res.get("error", "")[:300]))

    def _enhance_video_native(self):
        """Definizione video NATIVA: super-risoluzione per-frame (ESRGAN/spandrel via inference_server,
        handler video_upscale) + opz. restore volti / FPS. Zero ComfyUI."""
        import random as _r
        cfg = self.video_def_tab.get_config()
        src = cfg.get("video")
        if not src or not os.path.exists(src):
            QMessageBox.warning(self, tr("mw.no_video_title", "Video"), tr("mw.no_video_msg", "Scegli un video.")); return
        engine = cfg.get("engine", "seedvr2")
        if engine == "seedvr2":                          # SeedVR2 nativo (diffusion VSR, coerenza temporale)
            from core import seedvr2_service as sv
            errs = sv.preflight()
            if errs:
                QMessageBox.warning(self, "SeedVR2", "\n".join(errs) +
                    "\n\n" + tr("mw.seedvr2_install_hint", "Installa con: python scripts/install_seedvr2.py")); return
            out = str(OUTPUT_DIR / f"seedvr2_{Path(src).stem}.mp4")
            self._log(f"[nativo] SeedVR2 super-risoluzione (res {cfg.get('resolution',1080)}) su {Path(src).name}…")
            self._sv2_worker = _SeedVR2Worker(src, out, resolution=int(cfg.get("resolution", 1080) or 1080),
                                              batch_size=int(cfg.get("batch_size", 5) or 5), chunk_size=120)
            self._enqueue(tr("queue.job.seedvr2", "Super-risoluzione (SeedVR2)"), self._sv2_worker,
                          self._on_multitalk_done, on_progress=self._log)
            return
        if engine == "flashvsr":                         # FlashVSR v1.1 nativo (x4 streaming, diffsynth)
            from core import flashvsr_service as fv
            errs = fv.preflight()
            if errs:
                QMessageBox.warning(self, "FlashVSR", "\n".join(errs) +
                    "\n\n" + tr("mw.flashvsr_install_hint", "Installa con: python scripts/install_flashvsr.py")); return
            out = str(OUTPUT_DIR / f"flashvsr_{Path(src).stem}.mp4")
            self._log(f"[nativo] FlashVSR x4 super-risoluzione su {Path(src).name}…")
            self._fv_worker = _FlashVSRWorker(src, out)
            self._enqueue(tr("queue.job.flashvsr", "Super-risoluzione (FlashVSR)"), self._fv_worker,
                          self._on_multitalk_done, on_progress=self._log)
            return
        # esrgan: upscale per-frame nativo (spandrel)
        params = {
            "video": src, "model": cfg.get("esrgan_model"),
            "scale": int(cfg.get("target_scale", 2) or 2), "seed": _r.randint(0, 2**31),
        }
        self._log(f"[nativo] definizione video (upscale per-frame x{params['scale']}) su {Path(src).name}…")
        self._native_worker = _NativeImageWorker(params, module="video_upscale")
        self._enqueue(tr("queue.job.video_upscale", "Definizione video (upscale)"), self._native_worker,
                      self._on_multitalk_done)

    def _extend_video_native(self):
        """Prolunga un video NATIVAMENTE (LTX-2.3 GGUF): i2v dall'ULTIMO frame del video sorgente per
        N secondi, poi concatena originale + continuazione. Zero ComfyUI."""
        import subprocess as _sp, time as _t
        cfg = self.video_extend_tab.get_config()
        src = cfg.get("video")
        if not src or not os.path.exists(src):
            QMessageBox.warning(self, tr("mw.no_video_title", "Video"), tr("mw.no_video_msg", "Scegli un video da prolungare.")); return
        if not cfg.get("prompt"):
            QMessageBox.warning(self, tr("mw.prompt_missing_title", "Prompt"), tr("mw.extend_prompt_missing_msg", "Inserisci un prompt per l'estensione.")); return
        v = self.video_tab.get_values()
        fps = int(v.get("fps", 25) or 25)
        add_s = float(cfg.get("add_seconds", 5) or 5)
        nf = max(9, int(add_s * fps) // 8 * 8 + 1)
        lastf = str(OUTPUT_DIR / f"_ext_last_{int(_t.time())}.png")
        _sp.run(["ffmpeg", "-v", "error", "-sseof", "-0.12", "-i", src, "-vframes", "1", lastf, "-y"], check=False)
        if not os.path.exists(lastf):
            QMessageBox.warning(self, tr("mw.extend_video_error_title", "Prolunga"), "Impossibile estrarre l'ultimo frame."); return
        cont = str(OUTPUT_DIR / f"_ext_cont_{int(_t.time())}.mp4")
        self._ext_src = src
        params = {
            "image_path": lastf, "prompt": cfg["prompt"],
            "negative": v.get("prompt_negative", "") or "worst quality, blurry",
            "width": int(v.get("width", 512)), "height": int(v.get("height", 512)),
            "num_frames": nf, "fps": fps, "steps": int(self.video_tab.sampler_seed.steps.value()),
            "guidance": float(self.video_tab.sampler_seed.cfg.value()), "seed": int(self.video_tab.sampler_seed.seed.value()),
            "offload": "cpu", "ref_strength": 1.0, "output": cont,
        }
        if nf > 150:
            params["chunk_frames"] = 121
        self._log(f"[nativo] prolunga +{add_s:.0f}s ({nf} frame) da {Path(src).name}…")
        self._ltx2_worker = _NativeLtx2Worker(params)
        self._enqueue(tr("queue.job.video_extend", "Prolunga video"), self._ltx2_worker,
                      self._on_extend_native_done, on_progress=lambda s: self._log("[nativo] " + s))

    def _on_extend_native_done(self, res: dict):
        if res.get("status") != "success":
            self._log(f"[nativo] extend errore: {res.get('error','')[:300]}")
            QMessageBox.critical(self, tr("mw.extend_video_error_title", "Prolunga"), str(res.get("error", ""))[:1000]); return
        import subprocess as _sp, time as _t
        cont = (res.get("output") or [None])[0]
        src = getattr(self, "_ext_src", None)
        final = str(OUTPUT_DIR / f"extended_{int(_t.time())}.mp4")
        lst = str(OUTPUT_DIR / "_ext_list.txt")
        try:
            # concat con re-encode (codec/risoluzione possono differire tra sorgente e continuazione)
            with open(lst, "w", encoding="utf-8") as fh:
                fh.write(f"file '{src}'\nfile '{cont}'\n")
            _sp.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", final, "-y"], check=False)
        except Exception as e:                              # noqa: BLE001
            self._log(f"[nativo] concat extend fallito ({e}) → uso la sola continuazione")
            final = cont
        out = final if os.path.exists(final) else cont
        self._on_multitalk_done({"status": "success", "output": [out]})

    def _start_native_multitalk(self, params: dict):
        """Lip-sync/canto NATIVO via a2v LTX-2.3 (GGUF, zero ComfyUI). Sostituisce MultiTalk/Wan
        (era accoppiato al venv ComfyUI, rotto dal rename). Sorgente volto = foto Frame 1 o primo
        frame del video sorgente; audio reale (parlato/canzone). Chunking per i lunghi."""
        import subprocess as _sp, time as _t
        img = params.get("image_path")
        src = params.get("source_video")
        if not img and src:                              # talk da video → estrai il primo frame
            img = str(OUTPUT_DIR / f"_talk_src_{int(_t.time())}.png")
            _sp.run(["ffmpeg", "-v", "error", "-i", src, "-vframes", "1", img, "-y"], check=False)
        nf = int(params.get("num_frames", 121))
        nat = {
            "image_path": img, "audio_path": params.get("audio_path"),
            "prompt": params.get("prompt", ""),
            "negative": params.get("negative", "") or "worst quality, blurry",
            "width": int(params.get("width", 512)), "height": int(params.get("height", 512)),
            "num_frames": nf, "fps": int(params.get("fps", 25)),
            "steps": 8, "guidance": 3.0, "seed": 8, "offload": "cpu",
            "ref_strength": 1.0, "output": params.get("output", str(OUTPUT_DIR / "talk_native.mp4")),
        }
        if nf > 150:
            nat["chunk_frames"] = 121
        self._log(f"[lip-sync nativo] a2v LTX-2 GGUF — {nf} frame {nat['width']}x{nat['height']}…")
        self._mt_worker = _NativeLtx2Worker(nat)
        self._enqueue(tr("queue.job.lipsync", "Lip-sync"), self._mt_worker, self._on_multitalk_done,
                      on_progress=lambda s: self._log("[lip-sync] " + s))

    def _on_multitalk_done(self, res: dict):
        if res.get("status") != "success":
            self._log(f"[MultiTalk] errore: {res.get('error','')[:300]}")
            QMessageBox.critical(self, tr("mw.gen_error_title", "Errore"), str(res.get("error", ""))[:1000])
            return
        out = (res.get("output") or [None])[0]
        self._last_video_path = out
        self._log(f"[MultiTalk] video pronto: {out}")
        # post-process: face-swap automatico se attivo (impone un volto da foto su tutti i frame)
        fs = self.video_tab.get_face_swap_auto() if hasattr(self.video_tab, "get_face_swap_auto") else None
        if fs and fs.get("face") and out:
            from pathlib import Path as _P
            swapped = str(_P(out).with_name(_P(out).stem + "_faceswap.mp4"))
            self._log("[Face-swap] applico il volto da foto al video…")
            self._fs_worker = _FaceSwapWorker(out, fs["face"], swapped)
            self._enqueue(tr("queue.job.faceswap", "Face-swap"), self._fs_worker, self._on_faceswap_done,
                          on_progress=lambda s: self._log("[Face-swap] " + s))
            return
        self._maybe_rife(out)

    def _on_faceswap_done(self, res: dict):
        if res.get("status") != "success":
            self._log(f"[Face-swap] errore: {res.get('error','')[:300]}")
            QMessageBox.warning(self, "Face-swap", str(res.get("error", ""))[:800])
            return
        out = (res.get("output") or [None])[0]
        self._last_video_path = out
        self._log(f"[Face-swap] video finale: {out}")
        self._maybe_rife(out)

    def _maybe_rife(self, out):
        """Post-process RIFE (fluidifica) se attivo, altrimenti annuncia il video."""
        rf = self.video_tab.get_rife_auto() if hasattr(self.video_tab, "get_rife_auto") else None
        if rf and out:
            from pathlib import Path as _P
            fluid = str(_P(out).with_name(_P(out).stem + f"_rife{rf['factor']}x.mp4"))
            self._log(f"[RIFE] fluidifico il video (x{rf['factor']})…")
            self._rife_worker = _RifeWorker(out, fluid, int(rf["factor"]))
            self._enqueue(tr("queue.job.rife", "Interpolazione FPS (RIFE)"), self._rife_worker,
                          self._on_rife_done, on_progress=lambda s: self._log("[RIFE] " + s))
            return
        self._announce_video(out)

    def _on_rife_done(self, res: dict):
        if res.get("status") != "success":
            self._log(f"[RIFE] errore: {res.get('error','')[:300]}")
            self._announce_video(self._last_video_path)   # annuncia comunque il video pre-RIFE
            return
        out = (res.get("output") or [None])[0]
        self._last_video_path = out
        self._log(f"[RIFE] video fluidificato: {out}")
        self._announce_video(out)

    def _announce_video(self, out):
        # Finestra di conferma SEMPRE presente; con anteprima video (player) se QtMultimedia è
        # disponibile, altrimenti thumbnail del primo frame. Accetta = tieni, Revert = scarta il file.
        if not out or not os.path.exists(out):
            QMessageBox.information(self, tr("mw.native_done_title", "Video pronto"),
                                    tr("mw.native_done_msg", "Generato: {f}").format(f=out))
            return
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        dlg = QDialog(self); dlg.setWindowTitle(tr("mw.video_result_title", "Video generato"))
        v = QVBoxLayout(dlg)
        player = None
        try:                                             # anteprima video reale (player con loop)
            from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
            from PySide6.QtMultimediaWidgets import QVideoWidget
            from PySide6.QtCore import QUrl
            vw = QVideoWidget(); vw.setMinimumSize(480, 320); v.addWidget(vw)
            player = QMediaPlayer(dlg); ao = QAudioOutput(dlg)
            player.setAudioOutput(ao); player.setVideoOutput(vw)
            player.setSource(QUrl.fromLocalFile(out))
            player.setLoops(QMediaPlayer.Infinite)
            player.play()
        except Exception:                                # noqa: BLE001 — fallback: thumbnail primo frame
            try:
                import subprocess, tempfile
                from inference_server.backends.video_postprocess import _ffmpeg
                thumb = os.path.join(tempfile.gettempdir(), "ltx_vid_thumb.png")
                subprocess.run([_ffmpeg(), "-y", "-i", out, "-frames:v", "1", thumb],
                               capture_output=True, timeout=30)
                if os.path.exists(thumb):
                    lbl = QLabel(); lbl.setPixmap(QPixmap(thumb).scaledToWidth(480, Qt.SmoothTransformation))
                    lbl.setAlignment(Qt.AlignCenter); v.addWidget(lbl)
            except Exception:                            # noqa: BLE001
                v.addWidget(QLabel(tr("mw.video_no_preview", "Anteprima non disponibile.")))
        v.addWidget(QLabel(os.path.basename(out)))
        bb = QDialogButtonBox()
        bb.addButton(tr("mw.btn_accept", "Accetta"), QDialogButtonBox.AcceptRole)
        bb.addButton(tr("mw.btn_revert", "Revert"), QDialogButtonBox.RejectRole)
        b_open = bb.addButton(tr("mw.btn_open", "Apri"), QDialogButtonBox.ActionRole)
        b_open.clicked.connect(lambda: os.startfile(out) if hasattr(os, "startfile") else None)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        keep = bool(dlg.exec())
        if player is not None:
            player.stop()
        if keep:
            self._last_video_path = out
            try:
                self.video_tab.v2v_panel.set_last_generation(out)
            except Exception:                            # noqa: BLE001
                pass
            self._log(f"Video accettato: {out}")
        else:
            try:
                os.remove(out)
            except Exception:                            # noqa: BLE001
                pass
            self._log("Video scartato (revert).")

    def _maybe_multitalk(self, vals: dict, v2v: dict, mode: str) -> bool:
        """Instrada Talk/Canta al motore MultiTalk nativo. Ritorna True se gestito (→ on_generate
        deve fermarsi). MultiTalk è audio-driven: serve un audio REALE (file/canzone)."""
        if mode == "sing":
            audio_path = v2v.get("sing_audio_path")
            sep = bool(v2v.get("sing_separate_vocals", True))
            ascale = float(v2v.get("sing_audio_scale", 2.0))
        else:                                            # talk: audio dal tab Audio (file utente/TTS)
            ac = self.audio_tab.get_audio_config() if hasattr(self, "audio_tab") else {}
            audio_path = ac.get("audio_file")
            sep = False
            ascale = 2.0
        if not audio_path or not Path(audio_path).exists():
            QMessageBox.warning(self, tr("mw.multitalk_need_audio_title", "Audio mancante"),
                tr("mw.multitalk_need_audio_msg",
                   "La modalità lip-sync (MultiTalk) richiede un file audio reale: "
                   "scegli una canzone in 'Canta', o un file/voce nel tab Audio per 'Talk'."))
            return True
        src_video = v2v.get("v2v_video_path")            # talk da video esistente (→ primo frame)
        image_path = None if src_video else (self.frame1_tab.image_path or None)
        if not src_video and not image_path:
            QMessageBox.warning(self, tr("mw.talk_title", "Talk/Canta"),
                                tr("mw.talk_need_input_msg", "Serve una foto (Frame 1) o un video sorgente."))
            return True
        fps = int(vals.get("fps", 25) or 25)
        length = float(vals.get("length_s") or 0)
        if length <= 0:
            length = self._probe_duration(audio_path) or 5.0
        # durata segmento + overlap crossfade dalla GUI (Avanzate → Video lungo); default chunk 15s
        try:
            lv = self.advanced_tab.long_video_config()
            seg_frames = int(float(lv.get("seg_seconds", 15) or 15) * fps)
            overlap_frames = int(lv.get("overlap", 12) or 12)
            ref_anchor = float(lv.get("ref_anchor_strength", 0.25))
            anchor_every = int(lv.get("anchor_every", 0) or 0)
            resume = bool(lv.get("resume", True))
        except Exception:                               # noqa: BLE001
            seg_frames, overlap_frames = int(15 * fps), 12
            ref_anchor, anchor_every, resume = 0.25, 0, True
        params = {
            "image_path": image_path, "source_video": src_video,
            "audio_path": audio_path,
            "prompt": vals.get("prompt_positive", ""), "negative": vals.get("prompt_negative", ""),
            "width": int(vals.get("width", 640) or 640), "height": int(vals.get("height", 640) or 640),
            "num_frames": max(25, int(length * fps)), "fps": fps,
            "separate_vocals": sep, "audio_scale": ascale, "blocks_to_swap": "auto", "context": True,
            "seg_frames": seg_frames, "overlap_frames": overlap_frames,
            "ref_anchor_strength": ref_anchor, "anchor_every": anchor_every, "resume": resume,
            "output": str(OUTPUT_DIR / f"multitalk_{mode}.mp4"),
        }
        self._start_native_multitalk(params)
        return True

    def _start_native_inpaint(self, exp, unet, prompt, negative, steps, seed, retention, mode):
        """Inpaint/outpaint via backend NATIVO (FluxFillPipeline) — niente ComfyUI.
        Passa i path LOCALI di immagine+maschera (image_flux li apre con PIL)."""
        import os
        from core import native_driver
        from core.inpaint_pipeline import InpaintPipeline
        if not native_driver.is_up():
            QMessageBox.warning(self, tr("mw.native_off_title", "Backend nativo offline"),
                tr("mw.native_off_msg", "L'InferenceServer nativo (porta 8189) non risponde. "
                   "Avvialo, oppure rimetti il backend 'image' su ComfyUI."))
            return
        model = unet if os.path.isabs(unet) else os.path.join(r"E:\AI\Models\diffusion_models", unet)
        # FLUX.2 Klein (miraclein) = arch distillata edit-instruct con encoder Qwen3: instradata a
        # 'flux2_inpaint' (Flux2KleinInpaintPipeline). FLUX.1 Fill resta 'inpaint'. La guidance per Klein
        # è bassa (CFG reale ~4), per Fill è alta (~30) → effective_guidance(flux2=...).
        from core.gen_pipeline import is_flux2
        _f2 = bool(unet and is_flux2(unet))
        arch = "flux2_inpaint" if _f2 else "inpaint"
        guidance = InpaintPipeline.effective_guidance(unet, 30.0, retention, flux2=_f2)
        params = {"model": model, "arch": arch, "prompt": prompt, "negative": negative,
                  "image_path": exp["image"], "mask_path": exp["mask"],
                  "steps": steps, "guidance": guidance, "seed": seed}
        self._inpaint_mode = mode
        self._log(f"{mode} (backend NATIVO) inviato (steps={steps}, seed={seed}, guidance={guidance:.1f})…")
        self._native_worker = _NativeImageWorker(params)
        self._enqueue(tr("queue.job.inpaint", "Inpaint/Outpaint"), self._native_worker,
                      self._on_native_inpaint_done)

    def _start_native_kontext(self, exp, model, prompt, negative, steps, seed, guidance):
        """Kontext/integra via backend NATIVO (FluxKontextPipeline) — niente ComfyUI.
        Il composito (scena+oggetto) è il riferimento; il modello rigenera/armonizza."""
        import os
        from core import native_driver
        from core.kontext_pipeline import KONTEXT_MODEL
        if not native_driver.is_up():
            QMessageBox.warning(self, tr("mw.native_off_title", "Backend nativo offline"),
                tr("mw.native_off_msg", "L'InferenceServer nativo (porta 8189) non risponde."))
            return
        unet = model or KONTEXT_MODEL
        mp = unet if os.path.isabs(unet) else os.path.join(r"E:\AI\Models\diffusion_models", unet)
        if not prompt:
            prompt = ("Seamlessly integrate the added object into the scene: match the lighting, "
                      "shadows, color and perspective so it looks naturally part of the photo.")
        params = {"model": mp, "arch": "kontext", "prompt": prompt, "negative": negative,
                  "image_path": exp["image"], "steps": steps, "guidance": guidance, "seed": seed}
        self._inpaint_mode = "integrazione"
        self._log(f"Integra Kontext (backend NATIVO) inviato (steps={steps}, seed={seed})…")
        self._native_worker = _NativeImageWorker(params)
        self._enqueue(tr("queue.job.integrate", "Integra (Kontext)"), self._native_worker,
                      self._on_native_inpaint_done)

    def _on_native_inpaint_done(self, res: dict):
        if res.get("status") != "success":
            self._log(f"[native inpaint] errore: {res.get('error','')[:300]}")
            QMessageBox.critical(self, tr("mw.gen_error_title", "Errore"), str(res.get("error", ""))[:1000])
            return
        out = (res.get("output") or [None])[0]
        self._log(f"[native inpaint] generato: {out}")
        if out:
            self._show_inpaint_result(out)   # dialog anteprima prima/dopo + Accetta / Revert

    def _start_native_upscale(self, model, scale, image_path=None):
        """Upscale via backend NATIVO (ESRGAN/spandrel) — niente ComfyUI."""
        from core import native_driver
        if not native_driver.is_up():
            QMessageBox.warning(self, tr("mw.native_off_title", "Backend nativo offline"),
                tr("mw.native_off_msg", "L'InferenceServer nativo (porta 8189) non risponde."))
            return
        params = {"model": model, "arch": "upscale", "image_path": image_path or self.frame1_tab.image_path,
                  "target_scale": float(scale), "seed": 0}
        self._log(f"Upscale (backend NATIVO) inviato (×{scale})…")
        self._native_worker = _NativeImageWorker(params, module="upscale")
        self._enqueue(tr("queue.job.upscale", "Upscale immagine"), self._native_worker,
                      self._on_native_inpaint_done)

    def generate_upscale(self):
        """Aumenta la definizione dell'immagine corrente del canvas con un modello AI di super-risoluzione."""
        if not self._feature_gate("cap.upscale_image"):
            return
        from core import backend
        if not backend.is_native("image") and not self.driver.is_up():
            QMessageBox.warning(self, tr("mw.comfy_offline_title"), tr("mw.comfy_offline_generate_msg")); return
        if not self.frame1_tab.image_path:
            QMessageBox.warning(self, tr("mw.no_image_title"), tr("mw.load_or_gen_image_canvas_msg")); return
        if self._inpaint_pid:
            QMessageBox.information(self, tr("mw.wait_title"), tr("mw.operation_in_progress_msg")); return
        cfg = self.frame1_tab.get_upscale_config()
        if not cfg.get("model"):
            QMessageBox.warning(self, tr("mw.no_upscale_model_title"),
                                tr("mw.no_upscale_model_msg")); return
        if cfg.get("refine") and not cfg.get("refine_model"):
            QMessageBox.warning(self, tr("mw.refine_model_missing_title"),
                                tr("mw.refine_model_missing_msg")); return
        # ── BACKEND NATIVO (ESRGAN via spandrel): upscale puro senza ComfyUI.
        #    La rifinitura FLUX resta su ComfyUI finché non migrata.
        from core import backend
        if backend.is_native("image") and not cfg.get("refine"):
            self._inpaint_before = self.frame1_tab.image_path
            self._start_native_upscale(cfg["model"], cfg.get("target_scale") or cfg.get("scale") or 2.0)
            return
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
            extra = tr("mw.upscale_refine_extra") if cfg.get("refine") else ""
            self.gen_dialog.start(prompt, tr("mw.upscale_label").format(scale=cfg['scale'], extra=extra))
            self._log(f"Upscale {cfg['scale']:.0f}x con {cfg['model']}{extra} inviato…")
        except Exception as e:
            QMessageBox.critical(self, tr("mw.upscale_error_title"), str(e)); self._log(f"[ERR] upscale: {e}")

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
                    watermark.apply_to_file(str(rp))          # watermark obbligatorio (edizioni demo/supporter)
                    self._show_inpaint_result(str(rp)); return
                except Exception as e:
                    self._log(f"[ERR] download risultato: {e}")

    def _show_inpaint_result(self, path: str):
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        dlg = QDialog(self); dlg.setWindowTitle(tr("mw.result_dialog_title").format(mode=self._inpaint_mode))
        v = QVBoxLayout(dlg)
        before_pix = QPixmap(self._inpaint_before) if self._inpaint_before else None
        if before_pix is not None and before_pix.isNull():
            before_pix = None
        lbl = _HoldCompareLabel(QPixmap(path), before_pix)
        v.addWidget(lbl)
        if before_pix is not None:
            hint = QLabel(tr("mw.compare_hint"))
            hint.setStyleSheet("color:#9aa1b0;"); hint.setAlignment(Qt.AlignCenter); v.addWidget(hint)
        bb = QDialogButtonBox()
        b_ok = bb.addButton(tr("mw.btn_accept"), QDialogButtonBox.AcceptRole)
        bb.addButton(tr("mw.btn_revert"), QDialogButtonBox.RejectRole)
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
        if not self._feature_gate("cap.sam"):
            return False
        if not self.driver.is_up():
            QMessageBox.warning(self, tr("mw.comfy_offline_title"), tr("mw.comfy_offline_segment_msg")); return False
        if not self.frame1_tab.image_path:
            QMessageBox.warning(self, tr("mw.no_image_title"), tr("mw.load_photo_canvas_msg")); return False
        if self._seg_pid:
            QMessageBox.information(self, tr("mw.wait_title"), tr("mw.selection_in_progress_msg")); return False
        return True

    def segment_by_text(self):
        if not self._segment_preconds():
            return
        cfg = self.frame1_tab.get_segment_config()
        if not cfg.get("text"):
            QMessageBox.warning(self, tr("mw.text_missing_title"), tr("mw.segment_text_missing_msg")); return
        if backend.is_native("segment"):
            # detector "grounding_dino_only" → solo GroundingDINO (box, no raffinamento SAM2)
            return self._segment_native(text=cfg["text"],
                                        threshold=float(cfg.get("threshold", cfg.get("clipseg_threshold", 0.30))),
                                        use_sam=(cfg.get("detector") != "grounding_dino_only"))
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
            QMessageBox.critical(self, tr("mw.selection_error_title"), str(e)); self._log(f"[ERR] segment text: {e}")

    def segment_by_region(self):
        if not self._segment_preconds():
            return
        cfg = self.frame1_tab.get_segment_config()
        if backend.is_native("segment"):
            try:
                exp = self.frame1_tab.canvas.export_region_mask(OUTPUT_DIR)
                return self._segment_native(region_mask=exp["mask"], image_path=exp["image"])
            except Exception as e:                          # noqa: BLE001
                QMessageBox.critical(self, tr("mw.selection_error_title"), str(e)); return
        if not self._sam_models:
            QMessageBox.warning(self, tr("mw.sam_unavailable_title"),
                                tr("mw.sam_unavailable_msg")); return
        # Il puntatore (Impact SAMLoader) carica solo SAM vanilla: se la scelta è un SAM-HQ
        # o assente, ripiega su un vanilla disponibile (sam_vit_b/l/h).
        sam = cfg.get("sam_model")
        if not sam or "hq" in sam.lower():
            vanilla = next((m for m in self._sam_models if "hq" not in m.lower()), None)
            if not vanilla:
                QMessageBox.warning(self, tr("mw.sam_vanilla_needed_title"),
                                    tr("mw.sam_vanilla_needed_msg"))
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
            QMessageBox.critical(self, tr("mw.selection_error_title"), str(e)); self._log(f"[ERR] segment region: {e}")

    def _face_restore_native(self, cfg: dict):
        """Restauro volti NATIVO (CodeFormer/GFPGAN via inference_server, handler 'video_face'). Zero ComfyUI."""
        import random as _r
        src = cfg.get("video")
        if not src or not os.path.exists(src):
            QMessageBox.warning(self, tr("mw.no_video_title", "Video"), tr("mw.load_or_gen_video_msg", "Carica o genera un video.")); return
        params = {
            "image_path": src, "model": cfg.get("model_path") or cfg.get("model"),
            "fidelity": float(cfg.get("fidelity", 0.5) or 0.5),
            "detection": cfg.get("detection"), "seed": _r.randint(0, 2**31),
        }
        self._log(f"[nativo] restauro volti su {Path(src).name} (fidelity {params['fidelity']:.2f})…")
        self._native_worker = _NativeImageWorker(params, module="video_face")
        self._enqueue(tr("queue.job.video_face", "Restauro volti video"), self._native_worker,
                      self._on_multitalk_done)

    def _segment_native(self, *, text: str = "", threshold: float = 0.30,
                        region_mask: str = "", image_path: str = "", use_sam: bool = True):
        """Segmentazione NATIVA (GroundingDINO+SAM2 via inference_server, handler 'segment'). Zero ComfyUI.
        Per testo: usa il composite locale; per regione: usa image/region mask locali (no upload)."""
        # PROPORZIONI: per la selezione a TESTO usa la foto SORGENTE reale (alle sue dimensioni native),
        # non il composite del canvas — così la maschera non viene distorta a quadrato. Fallback al composite.
        if not image_path and not region_mask:
            src = getattr(self.frame1_tab.canvas, "_image_path", None)
            image_path = src if (src and os.path.exists(src)) else ""
        img = image_path or self.frame1_tab.canvas._composite(OUTPUT_DIR)
        params = {"image_path": img}
        if region_mask:
            params["region_mask"] = region_mask
            self._log(tr("mw.segment_native_region", "[nativo] selezione magnetica…"))
        else:
            params["text"] = text
            params["threshold"] = threshold
            params["use_sam"] = use_sam
            _mode = "GroundingDINO+SAM2" if use_sam else "solo GroundingDINO"
            self._log(f"[nativo] selezione '{text}' ({_mode}, soglia {threshold:.2f})…")
        self._seg_native_worker = _NativeImageWorker(params, module="segment")
        self._enqueue(tr("queue.job.segment", "Segmentazione"), self._seg_native_worker,
                      self._on_segment_native_done)

    def _on_segment_native_done(self, res: dict):
        if res.get("status") != "success":
            self._log(f"[nativo] segment errore: {str(res.get('error', ''))[:200]}")
            QMessageBox.critical(self, tr("mw.selection_error_title"), str(res.get("error", ""))[:600]); return
        mask = (res.get("output") or [None])[0]
        if mask and os.path.exists(mask):
            self.frame1_tab.canvas.load_mask_result(mask)
            self._log(tr("mw.segment_native_applied", "Selezione applicata alla maschera (modificabile)."))

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
                if hasattr(self, "music_tab"):
                    self.music_tab.set_last_video(video_path)   # per "Sonorizza ultimo video"
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
            # Continuità audio dalla coda SOLO se la coda ha davvero una traccia audio (la base muta,
            # o eros, produce video senza audio → niente da encodare). Il blocco TTS sotto può
            # comunque forzarla a False (in quel caso la voce è il TTS affettato, non l'audio LTX).
            vals["extend_audio_continuity"] = bool(tail) and self._has_audio_stream(es["accumulated"])
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
            self.gen_dialog.start(prompt, tr("mw.long_video_segment_label").format(
                seg=es['segs'], produced=es['produced'], target=es['target']))
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
            cmd += ["-i", video, "-c:v", "libx264"]
            # CONSERVA l'audio della coda se la sorgente ne ha uno: serve a _extend_audio_continuity
            # (LTXVAudioVAEEncode legge l'audio della coda). Con `-an` il file restava senza stream
            # audio e ffmpeg falliva ("Output file does not contain any stream"). Se la sorgente è
            # muta, resta `-an` (niente continuità audio, gestita a monte).
            cmd += (["-c:a", "aac"] if self._has_audio_stream(video) else ["-an"])
            cmd += [str(out)]
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
    def _has_audio_stream(video: str) -> bool:
        """True se il video contiene almeno una traccia audio (ffprobe). Evita di chiedere a
        LTXVAudioVAEEncode di leggere l'audio da un file muto (crash ffmpeg 'no stream')."""
        import subprocess
        try:
            r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                                "-show_entries", "stream=index", "-of", "csv=p=0", video],
                               capture_output=True, text=True)
            return bool(r.stdout.strip())
        except Exception:
            return False

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
            if backend.is_native("face"):                 # auto-chain NATIVO (zero ComfyUI)
                self._face_restore_native({
                    "video": str(src), "model": cfg.get("model"),
                    "model_path": cfg.get("model_path"), "fidelity": cfg.get("fidelity", 0.5),
                    "detection": cfg.get("detection", "retinaface_resnet50")})
                return True
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
            self.gen_dialog.start(prompt, tr("mw.face_restore_label"))
            self._log(f"[Restauro volti] passaggio finale su {src.name} "
                      f"(fidelity {cfg['fidelity']:.2f}, {cfg['model']}).")
            return True
        except Exception as e:
            self._log(f"[ERR] restauro volti automatico: {e} → mostro il video non restaurato.")
            return False

    def _faceswap_photo(self, cfg: dict):
        """T5: face-swap su FOTO (core/faceswap.swap_image, InsightFace nativo). Zero ComfyUI."""
        from PySide6.QtWidgets import QMessageBox
        image_path = cfg.get("image_path", "")
        if image_path == "<from_canvas>":
            image_path = self.frame1_tab.image_path or ""
        src_face = cfg.get("source_face", "")
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, tr("mw.no_image_title"), tr("mw.select_image_to_process_msg")); return
        if not src_face or not os.path.exists(src_face):
            QMessageBox.warning(self, tr("res.faceswap.group"), tr("res.faceswap.need_source",
                "Seleziona la foto del volto sorgente.")); return
        out = str(OUTPUT_DIR / f"faceswap_{Path(image_path).stem}.png")

        class _FSWorker(QThread):
            done = Signal(dict)

            def run(self):
                try:
                    from core import faceswap
                    r = faceswap.swap_image(image_path, src_face, out)
                    self.done.emit({"status": "success", "output": [r]})
                except Exception as e:  # noqa: BLE001
                    self.done.emit({"status": "error", "error": str(e)})

        self._log(f"[nativo] face-swap foto: {Path(src_face).name} → {Path(image_path).name}…")
        self._fsphoto_worker = _FSWorker()
        self._enqueue(tr("queue.job.faceswap_photo", "Face-swap foto"), self._fsphoto_worker,
                      self._on_native_image_done)

    def generate_restore(self, cfg: dict):
        """Restauro / colorazione / face-swap foto."""
        if cfg.get("operation") == "faceswap":           # T5: face-swap su foto (nativo, no ComfyUI)
            return self._faceswap_photo(cfg)
        if not self.driver.is_up():
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, tr("mw.comfy_offline_title"), tr("mw.comfy_offline_process_msg")); return
        if self._inpaint_pid:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, tr("mw.wait_title"), tr("mw.processing_in_progress_msg")); return

        image_path = cfg.get("image_path", "")
        if image_path == "<from_canvas>":
            image_path = self.frame1_tab.image_path or ""
        if not image_path:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, tr("mw.no_image_title"), tr("mw.select_image_to_process_msg")); return

        op = cfg.get("operation", "restore")
        # ── BACKEND NATIVO: il restauro 'restore' è upscale ESRGAN (nativo via spandrel).
        #    colorize/restore_and_colorize usano FLUX.2 → restano su ComfyUI (posticipato).
        from core import backend
        if op == "restore" and backend.is_native("image"):
            self._inpaint_before = image_path
            self._start_native_upscale(cfg["upscale_model"], cfg["target_scale"], image_path=image_path)
            return
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
                label = tr("mw.restore_label").format(scale=cfg['target_scale'])
            elif op == "colorize":
                prompt = self.restorer.build_colorize(
                    img_name,
                    flux2_model=cfg["flux2_model"],
                    subject=cfg["subject"],
                    style=cfg["style"],
                    seed=seed,
                    steps=cfg["color_steps"])
                label = tr("mw.colorize_label")
            else:  # restore_and_colorize
                prompt = self.restorer.build_restore_and_colorize(
                    img_name,
                    upscale_model=cfg["upscale_model"],
                    flux2_model=cfg["flux2_model"],
                    subject=cfg["subject"],
                    style=cfg["style"],
                    seed=seed,
                    steps=cfg["color_steps"])
                label = tr("mw.restore_colorize_label")

            self._inpaint_pid = self.driver.submit(prompt)
            self._inpaint_mode = label
            self.gen_dialog.start(prompt, label)
            self._log(f"{label}: inviato (seed={seed}).")
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, tr("mw.restore_error_title"), str(e))
            self._log(f"[ERR] restauro: {e}")

    def _restore_from_canvas(self):
        """Porta al tab Restauro con l'immagine del canvas pre-selezionata."""
        img = self.frame1_tab.image_path or ""
        if not img:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, tr("mw.no_image_title"),
                tr("mw.load_image_before_restore_msg"))
            return
        # Pre-imposta il path nel tab Restauro
        self.restore_tab._input_path = img
        self.restore_tab.path_edit.setText(img)
        # Porta al tab «Migliora qualità» → sotto-tab Foto (Restauro)
        self.tabs.setCurrentWidget(self.quality_tab)
        self.quality_tab.setCurrentWidget(self.restore_tab)

    # ---------- TRELLIS.2 (immagine → modello 3D) ----------
    def _trellis_native(self, cfg: dict, image_path: str):
        """TRELLIS 3D NATIVO (venv isolato, subprocess) — immagine→GLB. Zero ComfyUI."""
        from PySide6.QtWidgets import QMessageBox
        from core import trellis_service
        errs = trellis_service.preflight(image_path)
        if errs:
            QMessageBox.warning(self, "TRELLIS", "\n".join(errs)); return
        seed = random.randint(0, 2**31 - 1) if cfg.get("seed_random", True) else int(cfg.get("seed", 0))
        name = f"trellis_{Path(image_path).stem}_{seed}"
        steps = int(cfg.get("steps", 12) or 12)
        job = trellis_service.make_job(
            image_path, name, seed=seed, ss_steps=steps, slat_steps=steps,
            simplify=float(cfg.get("simplify", 0.95) or 0.95),
            texture_size=int(cfg.get("texture_size", 1024) or 1024),
            on_log=self._log)
        out_glb = str(trellis_service.glb_path(name))
        self._log(f"[nativo] TRELLIS 3D su {Path(image_path).name} (res {cfg.get('resolution','')})…")

        class _TrellisWorker(QThread):
            done = Signal(dict)

            def run(self):
                import os as _os, threading as _th
                try:
                    ev = _th.Event(); rc = {"v": None}
                    job.on_done = lambda code: (rc.__setitem__("v", code), ev.set())
                    job.start()                              # Job è asincrono (subprocess) → attendi on_done
                    ev.wait()
                    if rc["v"] == 0 and _os.path.exists(out_glb):
                        self.done.emit({"status": "success", "output": [out_glb]})
                    else:
                        self.done.emit({"status": "error",
                                        "error": f"TRELLIS terminato (exit {rc['v']}); output mancante: {out_glb}"})
                except Exception as e:  # noqa: BLE001
                    self.done.emit({"status": "error", "error": str(e)})

            def stop(self):                                  # cancel dalla coda → killa il subprocess
                try:
                    job.stop()
                except Exception:                            # noqa: BLE001
                    pass

        self._trellis_worker = _TrellisWorker()
        self._enqueue(tr("queue.job.trellis", "Modello 3D (TRELLIS)"), self._trellis_worker,
                      self._on_trellis_native_done)

    def _on_trellis_native_done(self, res: dict):
        from PySide6.QtWidgets import QMessageBox
        if res.get("status") != "success":
            self._log(f"[trellis] errore: {res.get('error', '')[:300]}")
            QMessageBox.critical(self, "TRELLIS", str(res.get("error", ""))[:1000]); return
        glb = (res.get("output") or [None])[0]
        self._log(f"[trellis] mesh 3D pronta: {glb}")
        try:                                              # carica nel visore 3D del tab se disponibile
            if glb and hasattr(self, "trellis_tab") and hasattr(self.trellis_tab, "load_model"):
                self.trellis_tab.load_model(glb)
        except Exception:  # noqa: BLE001
            pass

    def generate_trellis(self, cfg: dict):
        """Genera un modello 3D dall'immagine selezionata (nativo se installato, altrimenti ComfyUI)."""
        from PySide6.QtWidgets import QMessageBox
        from core import trellis_service
        if not trellis_service.is_installed() and not self.driver.is_up():
            QMessageBox.warning(self, tr("mw.comfy_offline_title"),
                                tr("mw.comfy_offline_process_msg")); return
        if self._trellis_pid or self._inpaint_pid:
            QMessageBox.information(self, tr("mw.wait_title"),
                                   tr("mw.processing_in_progress_msg")); return

        image_path = cfg.get("image_path", "")
        force_no_bg = False
        if image_path == "<from_canvas>":
            image_path = self.frame1_tab.image_path or ""
        elif image_path == "<from_canvas_masked>":
            # Soggetto isolato dalla maschera del canvas (sfondo trasparente):
            # esclude sfondo/oggetti indesiderati e disattiva il remove-bg automatico.
            if not self.frame1_tab.canvas.has_mask():
                QMessageBox.warning(self, tr("trellis.no_mask.title"),
                                    tr("trellis.no_mask.msg")); return
            exp = self.frame1_tab.canvas.export_subject_rgba(OUTPUT_DIR)
            image_path = exp.get("image") or ""
            force_no_bg = True
        if not image_path:
            QMessageBox.warning(self, tr("mw.no_image_title"),
                                tr("mw.select_image_to_process_msg")); return

        # BACKEND NATIVO (TRELLIS via venv isolato, zero ComfyUI) se installato
        try:
            from core import trellis_service
            if trellis_service.is_installed():
                return self._trellis_native(cfg, image_path)
        except Exception:  # noqa: BLE001
            pass

        # Verifica che i nodi TRELLIS.2 siano installati in ComfyUI
        try:
            if "LoadTrellis2Models" not in self.driver.object_info("LoadTrellis2Models"):
                QMessageBox.warning(self, tr("trellis.missing.title"),
                                    tr("trellis.missing.msg")); return
        except Exception:
            QMessageBox.warning(self, tr("trellis.missing.title"),
                                tr("trellis.missing.msg")); return

        try:
            img_name = self._upload_local(image_path)
            seed = random.randint(0, 2**31 - 1) if cfg.get("seed_random", True) else int(cfg.get("seed", 0))
            self._trellis_prefix = f"ltxstudio_trellis_{seed}"
            prompt = self.trellisser.build(
                img_name,
                resolution=cfg.get("resolution", "1024_cascade"),
                precision=cfg.get("precision", "auto"),
                attn_backend=cfg.get("attn_backend", "auto"),
                remove_background=False if force_no_bg else cfg.get("remove_background", True),
                background_color=cfg.get("background_color", "black"),
                seed=seed,
                ss_guidance_strength=cfg.get("guidance", 6.5),
                shape_guidance_strength=cfg.get("guidance", 6.5),
                ss_sampling_steps=cfg.get("steps", 12),
                shape_sampling_steps=cfg.get("steps", 12),
                max_tokens=cfg.get("max_tokens", 49152),
                target_face_count=cfg.get("target_face_count", 500000),
                texture=cfg.get("texture", True),
                tex_guidance_strength=cfg.get("tex_guidance", 3.0),
                tex_sampling_steps=cfg.get("tex_steps", 12),
                texture_size=cfg.get("texture_size", 2048),
                file_format=cfg.get("file_format", "glb"),
                filename_prefix=self._trellis_prefix,
            )
            self._trellis_pid = self.driver.submit(prompt)
            label = tr("trellis.run.label").format(res=cfg.get("resolution", ""))
            self.gen_dialog.start(prompt, label)
            self._log(f"{label}: inviato (seed={seed}).")
        except Exception as e:
            QMessageBox.critical(self, tr("trellis.error.title"), str(e))
            self._log(f"[ERR] TRELLIS: {e}")

    def _check_trellis_done(self):
        try:
            hist = self.driver.history(self._trellis_pid)
        except Exception:
            return
        entry = hist.get(self._trellis_pid)
        if not entry:
            return
        st = entry.get("status", {})
        if st.get("status_str") == "error":
            self.gen_dialog.finish()
            self._log("[ERR] generazione 3D fallita"); self._trellis_pid = None; return
        if not st.get("completed"):
            return
        self._trellis_pid = None
        self.gen_dialog.finish()
        glb = self._latest_trellis_file()
        if not glb:
            self._log("[ERR] nessun file 3D prodotto da TRELLIS."); return
        # Copia nell'output di LTX Studio + mostra nel visore
        try:
            dest = OUTPUT_DIR / Path(glb).name
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copyfile(glb, dest)
        except Exception:
            dest = Path(glb)   # fallback: usa direttamente il file di ComfyUI
        self._log(f"✔ modello 3D salvato: {dest}")
        self.trellis_tab.show_result(str(dest))
        # rendi disponibile al tab Rig come "ultimo modello 3D"
        self.rig_tab.set_last_mesh(str(dest))
        self.tabs.setCurrentWidget(self.trellis_tab)

    def _latest_trellis_file(self) -> str | None:
        """Trova il .glb/.obj/... più recente col prefisso atteso nella output di ComfyUI."""
        try:
            cands = [f for f in COMFY_OUTPUT_DIR.glob(f"{self._trellis_prefix}*")
                     if f.suffix.lower() in (".glb", ".obj", ".ply", ".stl", ".gltf")]
            if not cands:
                return None
            return str(max(cands, key=lambda f: f.stat().st_mtime))
        except Exception:
            return None

    # ---------- Rig 3D (SkinTokens) ----------
    def _rig_native(self, cfg: dict, mesh_path: str):
        """Rig SkinTokens NATIVO (venv isolato + Blender headless) — mesh→GLB rigged. Zero ComfyUI."""
        from PySide6.QtWidgets import QMessageBox
        from core import rig_service, blender_installer
        errs = rig_service.preflight(mesh_path)
        if errs:
            QMessageBox.warning(self, "SkinTokens", "\n".join(errs)); return
        if not blender_installer.find_blender():           # Blender richiesto per skinning/export
            if not self._offer_blender_install():
                return
            return
        name = f"rig_{Path(mesh_path).stem}"
        out = str(rig_service.OUT_DIR / rig_service._safe(name) / "rigged.glb")
        job = rig_service.make_job(mesh_path, name, on_log=self._log)
        self._log(f"[nativo] Rig SkinTokens su {Path(mesh_path).name}…")

        class _RigWorker(QThread):
            done = Signal(dict)

            def run(self):
                import os as _os, threading as _th
                try:
                    ev = _th.Event(); rc = {"v": None}
                    job.on_done = lambda code: (rc.__setitem__("v", code), ev.set())
                    job.start()                              # Job asincrono → attendi on_done
                    ev.wait()
                    if rc["v"] == 0 and _os.path.exists(out):
                        self.done.emit({"status": "success", "output": [out]})
                    else:
                        self.done.emit({"status": "error",
                                        "error": f"Rig terminato (exit {rc['v']}); output mancante: {out}"})
                except Exception as e:  # noqa: BLE001
                    self.done.emit({"status": "error", "error": str(e)})

            def stop(self):
                try:
                    job.stop()
                except Exception:                            # noqa: BLE001
                    pass

        self._rig_worker = _RigWorker()
        self._enqueue(tr("queue.job.rig", "Rig 3D"), self._rig_worker, self._on_trellis_native_done)

    def generate_rig(self, cfg: dict):
        """Rigging automatico di una mesh (nativo SkinTokens se installato, altrimenti ComfyUI)."""
        import os, shutil
        from PySide6.QtWidgets import QMessageBox
        from core import rig_service
        if not rig_service.is_installed() and not self.driver.is_up():
            QMessageBox.warning(self, tr("mw.comfy_offline_title"),
                                tr("mw.comfy_offline_process_msg")); return
        if self._rig_pid or self._trellis_pid:
            QMessageBox.information(self, tr("mw.wait_title"),
                                   tr("mw.processing_in_progress_msg")); return

        mesh_path = cfg.get("mesh_path", "")
        if not mesh_path or not Path(mesh_path).exists():
            QMessageBox.warning(self, tr("rig.no_mesh.title"), tr("rig.no_mesh.msg")); return

        if rig_service.is_installed():                     # BACKEND NATIVO (zero ComfyUI)
            return self._rig_native(cfg, mesh_path)

        # Pre-check nodo installato
        try:
            if "SkinTokenRigTrimesh" not in self.driver.object_info("SkinTokenRigTrimesh"):
                QMessageBox.warning(self, tr("rig.missing.title"), tr("rig.missing.msg")); return
        except Exception:
            QMessageBox.warning(self, tr("rig.missing.title"), tr("rig.missing.msg")); return

        # Pre-check Blender (richiesto per l'export headless su Python 3.12).
        # Se manca, offri l'installazione automatica via winget (previo consenso).
        from core import blender_installer
        if not blender_installer.find_blender():
            if not self._offer_blender_install():
                return
            # installazione avviata in background: l'utente riproverà dopo il riavvio
            return

        try:
            self._rig_prefix = "3D/ltxstudio_rig"
            prompt = self.rigger.build(
                mesh_path,
                file_format=cfg.get("file_format", "glb"),
                skeleton_template=cfg.get("skeleton_template", "Keep model names"),
                use_transfer=cfg.get("use_transfer", True),
                use_postprocess=cfg.get("use_postprocess", False),
                group_per_vertex=cfg.get("group_per_vertex", 4),
                bottom_center_origin=cfg.get("bottom_center_origin", False),
                smooth_angle=cfg.get("smooth_angle", 55.0),
                top_k=cfg.get("top_k", 5),
                top_p=cfg.get("top_p", 0.95),
                temperature=cfg.get("temperature", 1.0),
                repetition_penalty=cfg.get("repetition_penalty", 2.0),
                num_beams=cfg.get("num_beams", 10),
                filename_prefix=self._rig_prefix,
            )
            self._rig_pid = self.driver.submit(prompt)
            label = tr("rig.run.label").format(tpl=cfg.get("skeleton_template", ""))
            self.gen_dialog.start(prompt, label)
            self._log(f"{label}: inviato.")
        except Exception as e:
            QMessageBox.critical(self, tr("rig.error.title"), str(e))
            self._log(f"[ERR] rig: {e}")

    def _check_rig_done(self):
        try:
            hist = self.driver.history(self._rig_pid)
        except Exception:
            return
        entry = hist.get(self._rig_pid)
        if not entry:
            return
        st = entry.get("status", {})
        if st.get("status_str") == "error":
            self.gen_dialog.finish()
            self._log("[ERR] rigging fallito"); self._rig_pid = None; return
        if not st.get("completed"):
            return
        self._rig_pid = None
        self.gen_dialog.finish()
        rigged = self._latest_rig_file()
        if not rigged:
            self._log("[ERR] nessun file rigged prodotto."); return
        try:
            dest = OUTPUT_DIR / Path(rigged).name
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copyfile(rigged, dest)
        except Exception:
            dest = Path(rigged)
        self._log(f"✔ modello rigged salvato: {dest}")
        self.rig_tab.show_result(str(dest))
        # handoff continuità: rendi il rig disponibile al tab Animazione
        self.anim_tab.set_last_rig(str(dest))
        self.tabs.setCurrentWidget(self.rig_tab)

    def _latest_rig_file(self) -> str | None:
        """Trova il file rigged più recente (SkinTokens salva in output/3D/)."""
        try:
            base = COMFY_OUTPUT_DIR / "3D"
            cands = [f for f in base.glob("ltxstudio_rig*")
                     if f.suffix.lower() in (".glb", ".fbx", ".gltf")]
            if not cands:
                return None
            return str(max(cands, key=lambda f: f.stat().st_mtime))
        except Exception:
            return None

    # ---------- Convertitore + operazioni mesh (Blender headless) ----------
    def _wire_viewer_ops(self, tab):
        """Collega i segnali del visore 3D interattivo (se presente) agli handler."""
        viewer = getattr(tab, "viewer", None)
        if viewer is None:
            return
        if hasattr(viewer, "request_mesh_op"):
            viewer.request_mesh_op.connect(self.run_mesh_op)
        if hasattr(viewer, "request_import"):
            viewer.request_import.connect(lambda t=tab: self._viewer_import(t))
        if hasattr(viewer, "request_export"):
            viewer.request_export.connect(lambda t=tab: self._viewer_export(t))

    def _viewer_import(self, tab):
        from PySide6.QtWidgets import QFileDialog
        from ui.tabs.convert_tab import INPUT_FILTER
        path, _ = QFileDialog.getOpenFileName(
            self, tr("viewer.import"), str(Path.home()), INPUT_FILTER)
        if not path:
            return
        if hasattr(tab, "set_input_path"):
            tab.set_input_path(path)
        elif hasattr(tab, "set_last_mesh"):
            tab.set_last_mesh(path)
        if hasattr(tab, "show_result"):
            tab.show_result(path)
        self._log(f"[3D] importato nello stadio: {path}")

    def _viewer_export(self, tab):
        from PySide6.QtWidgets import QMessageBox
        v = getattr(tab, "viewer", None)
        path = v.get_edited_path() if (v and hasattr(v, "get_edited_path")) else ""
        if not path:
            QMessageBox.information(self, tr("viewer.export"), tr("viewer.no_asset"))
            return
        self.convert_tab.set_input_path(path)
        self.tabs.setCurrentWidget(self.convert_tab)

    def run_convert(self, cfg: dict):
        """Converte un file 3D nel formato scelto (Blender headless)."""
        import os
        from core.blender_service import blender_available
        if not cfg.get("input"):
            return
        if not blender_available():
            self._offer_blender_install(); return
        out_dir = OUTPUT_DIR / "converted"; out_dir.mkdir(parents=True, exist_ok=True)
        args = dict(cfg)
        args["output"] = str(out_dir / f"{Path(cfg['input']).stem}.{cfg['format']}")
        self.convert_tab.log_msg(tr("convert.running").format(fmt=cfg["format"]))
        self._start_blender("convert.py", args, self._on_convert_done, self._on_blender_err)

    def _on_convert_done(self, res: dict):
        import os
        out = res.get("output", "")
        self.convert_tab.log_msg(tr("convert.done").format(path=out))
        self._log(f"✔ convertito: {out}")
        try:
            os.startfile(str(Path(out).parent))  # noqa: S606 (Windows)
        except Exception:
            pass

    def run_mesh_op(self, cfg: dict):
        """Esegue un'operazione mesh (separa/cancella) in-place via Blender e ricarica il viewer."""
        from core.blender_service import blender_available
        inp = cfg.get("input")
        if not inp:
            return
        if not blender_available():
            self._offer_blender_install(); return
        args = dict(cfg)
        args["output"] = inp                                   # editing in-place
        args["format"] = Path(inp).suffix.lstrip(".") or "glb"
        self._mesh_op_tab = self._tab_owning_viewer(inp)
        self._log(tr("viewer.mesh_op_running").format(op=cfg.get("op")))
        self._start_blender("mesh_ops.py", args, self._on_mesh_op_done, self._on_blender_err)

    def _on_mesh_op_done(self, res: dict):
        out = res.get("output", "")
        self._log(f"✔ operazione mesh completata ({len(res.get('objects', []))} oggetti).")
        tab = getattr(self, "_mesh_op_tab", None)
        if tab is not None and out and hasattr(tab, "show_result"):
            tab.show_result(out)

    def _tab_owning_viewer(self, path: str):
        for tab in (self.trellis_tab, self.rig_tab, getattr(self, "anim_tab", None)):
            if tab is None:
                continue
            v = getattr(tab, "viewer", None)
            if v is not None and hasattr(v, "get_edited_path") and v.get_edited_path() == path:
                return tab
        return self.trellis_tab

    def _start_blender(self, script: str, args: dict, on_ok, on_err, timeout: int = 900):
        from core.blender_service import BlenderWorker
        worker = BlenderWorker(script, args, timeout)
        self._blender_jobs = getattr(self, "_blender_jobs", [])
        self._blender_jobs.append(worker)                      # no GC finché vivo
        worker.finished_ok.connect(on_ok)
        worker.failed.connect(on_err)
        worker.finished.connect(
            lambda w=worker: self._blender_jobs.remove(w) if w in self._blender_jobs else None)
        worker.start()

    def _on_blender_err(self, msg: str):
        from PySide6.QtWidgets import QMessageBox
        self._log(f"[ERR] Blender: {msg}")
        QMessageBox.warning(self, tr("convert.error.title"), msg)

    # ---------- Animazione / Mocap ----------
    def start_mocap(self, cfg: dict):
        """Avvia la cattura webcam (worker MediaPipe→BVH nel venv mocap)."""
        from PySide6.QtWidgets import QMessageBox
        from core.mocap_service import mocap_env_ready
        if not mocap_env_ready():
            QMessageBox.warning(self, tr("anim.env.title"), tr("anim.env.msg"))
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._mocap_bvh = str(OUTPUT_DIR / "mocap_capture.bvh")
        preview = str(OUTPUT_DIR / "mocap_preview.png")
        if self.mocap.start(self._mocap_bvh, preview, camera=int(cfg.get("camera", 0))):
            self.anim_tab.set_capturing(True)
            self._log(tr("anim.cam.started"))

    def _on_mocap_done(self, bvh: str, frames: int):
        self.anim_tab.set_capturing(False)
        self.anim_tab.set_anim(bvh)
        self._log(tr("anim.cam.done").format(frames=frames, path=bvh))

    def _on_mocap_failed(self, msg: str):
        from PySide6.QtWidgets import QMessageBox
        self.anim_tab.set_capturing(False)
        self._log(f"[ERR] mocap: {msg}")
        QMessageBox.warning(self, tr("anim.env.title"), msg)

    def run_retarget(self, cfg: dict):
        """Applica l'animazione (BVH) al rig via Blender (retarget su armatura esistente)."""
        from PySide6.QtWidgets import QMessageBox
        from core.blender_service import blender_available
        model = cfg.get("model", "")
        anim = cfg.get("anim", "")
        if not model or not Path(model).exists():
            QMessageBox.warning(self, tr("anim.need.title"), tr("anim.need_rig.msg")); return
        if not anim or not Path(anim).exists():
            QMessageBox.warning(self, tr("anim.need.title"), tr("anim.need_anim.msg")); return
        if not blender_available():
            self._offer_blender_install(); return
        out_dir = OUTPUT_DIR / "animated"; out_dir.mkdir(parents=True, exist_ok=True)
        args = dict(cfg)
        args["output"] = str(out_dir / f"{Path(model).stem}_anim.{cfg.get('format', 'fbx')}")
        self._log(tr("anim.retarget.running"))
        self._start_blender("retarget.py", args, self._on_retarget_done, self._on_blender_err)

    def _on_retarget_done(self, res: dict):
        out = res.get("output", "")
        rt = res.get("retargeted", False)
        self._log(tr("anim.retarget.done").format(path=out, mode="rig esistente" if rt else "armatura BVH"))
        if out and Path(out).suffix.lower() in (".glb", ".gltf"):
            self.anim_tab.show_result(out)
        else:
            import os
            try:
                os.startfile(str(Path(out).parent))  # noqa: S606
            except Exception:
                pass

    def _offer_blender_install(self) -> bool:
        """Chiede consenso e avvia l'installazione automatica di Blender (winget)."""
        from PySide6.QtWidgets import QMessageBox, QProgressDialog
        from core import blender_installer
        if not blender_installer.winget_available():
            QMessageBox.warning(self, tr("rig.no_blender.title"), tr("rig.no_blender.msg"))
            return False
        ans = QMessageBox.question(
            self, tr("rig.blender_install.title"), tr("rig.blender_install.msg"),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ans != QMessageBox.Yes:
            return False
        dlg = QProgressDialog(tr("rig.blender_install.progress"), "", 0, 0, self)
        dlg.setWindowTitle(tr("rig.blender_install.title"))
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setRange(0, 0)
        dlg.show()
        worker = blender_installer.BlenderInstallWorker()
        self._blender_worker = worker        # mantieni il riferimento (no GC)
        worker.finished_ok.connect(self._on_blender_installed)
        worker.failed.connect(self._on_blender_install_failed)
        worker.finished.connect(dlg.close)
        worker.start()
        self._log(tr("rig.blender_install.progress"))
        return True

    def _on_blender_installed(self, path: str):
        from PySide6.QtWidgets import QMessageBox
        self._log(f"✔ Blender installato: {path}")
        QMessageBox.information(self, tr("rig.blender_ok.title"),
                               tr("rig.blender_ok.msg").format(path=path))

    def _on_blender_install_failed(self, msg: str):
        from PySide6.QtWidgets import QMessageBox
        self._log(f"[ERR] install Blender: {msg}")
        QMessageBox.warning(self, tr("rig.blender_fail.title"),
                            tr("rig.blender_fail.msg").format(err=msg))
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
        path = watermark.apply_to_file(path)        # watermark obbligatorio (edizioni demo/supporter)
        # rendi l'ultimo video disponibile come sorgente V2V ("Usa ultima generazione")
        try:
            if hasattr(self.video_tab, "v2v_panel"):
                self.video_tab.v2v_panel.set_last_generation(path)
        except Exception:
            pass
        if self.video_player is None:
            self.video_player = VideoPlayerDialog(self)
        self.video_player.play_file(path)
