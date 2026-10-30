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
    QMenu, QToolButton, QSplitter, QInputDialog,
)

from ui.i18n_manager import tr, i18n, LANGUAGES, LANG_FLAGS
from ui.job_queue import GenerationQueue, JobQueuePanel
from ui import model_menu as mm
from ui.tabs.frame1_tab import QWEN21_ENGINE_IDX, FLUX3_ENGINE_IDX

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
from core import video_edit_modes as vem                  # noqa: E402  (modalità Modifica Video AI)
from core import features, gating_apply, watermark        # noqa: E402  (feature-gating per edizione)
from core import user_prefs                                # noqa: E402  (varianti UI Classica/Pro + preferenze)
from core import backend                                  # noqa: E402  (router backend nativo/ComfyUI)
from ui.tabs.video_def_tab import VideoDefTab          # noqa: E402
from ui.tabs.video_extend_tab import VideoExtendTab    # noqa: E402
from ui.tabs.restore_tab import RestoreTab             # noqa: E402
from ui.tabs.trellis_tab import TrellisTab             # noqa: E402
from ui.tabs.avatar_tab import AvatarTab               # noqa: E402
from ui.tabs.rig_tab import RigTab                     # noqa: E402
from ui.tabs.import_models_tab import ImportModelsTab  # noqa: E402
from ui.tabs.convert_tab import ConvertTab             # noqa: E402
from ui.tabs.anim_tab import AnimTab                   # noqa: E402
from ui.tabs.train_tab import TrainTab                 # noqa: E402
from ui.tabs.lora_video_tab import LoraVideoTab        # noqa: E402
from ui.tabs.lora_image_tab import LoraImageTab        # noqa: E402
from ui.tabs.train_lora_video_tab import TrainLoraVideoTab  # noqa: E402
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


def _kill_proc_tree(proc):
    """#91: termina un subprocess Popen E I SUOI FIGLI (anti-orfano VRAM). Necessario perché i worker
    video (ltx2_infer/comfy) lanciano un python-venv che è padre dell'interprete reale: terminare solo
    il padre lascerebbe il figlio a calcolare e a tenere la VRAM. Windows → taskkill /T /F (albero)."""
    if proc is None:
        return
    try:
        if proc.poll() is not None:                         # già terminato
            return
    except Exception:                                       # noqa: BLE001
        return
    import sys as _sys, subprocess as _sp
    if _sys.platform == "win32":
        try:
            _sp.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=10)
            return
        except Exception:                                   # noqa: BLE001
            pass
    try:
        proc.terminate(); proc.wait(timeout=5)
    except Exception:                                       # noqa: BLE001
        try:
            proc.kill()
        except Exception:                                   # noqa: BLE001
            pass


class _NativeImageWorker(QThread):
    """Esegue una generazione immagine sul backend NATIVO (InferenceServer) fuori dal thread UI."""
    done = Signal(dict)

    # encoder FLUX.2 Klein scelto nella GUI (callable → id di catalogo o ""): lo collega MainWindow, così vale per
    # tutti i job Klein (inpaint, Integra, Componi, Colore) senza toccare ogni punto che li crea
    klein_te_source = None

    def __init__(self, params: dict, module: str = "image"):
        super().__init__()
        import time as _t
        params.setdefault("_t_button", _t.time())           # timeline unica dal click "Genera" (wall-clock)
        if (module == "image" and str(params.get("arch") or "").startswith("flux2") and "klein_te" not in params
                and _NativeImageWorker.klein_te_source is not None):
            try:
                _kte = _NativeImageWorker.klein_te_source()
            except Exception:                                   # noqa: BLE001
                _kte = ""
            if _kte:
                params["klein_te"] = _kte
        print(f"[gen-timeline] t0 = click Genera (module={module}, arch={params.get('arch','')})", flush=True)
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


class _VideoComfyWorker(QThread):
    """Motore video alternativo COMFY-NATIVE (model_management ComfyUI, VRAM dinamica, no-OOM) per
    t2v/i2v semplici. Selettore in video_tab; default resta ltx2_infer (copre tutte le feature)."""
    done = Signal(dict)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self.proc = None                                    # #91: subprocess, per terminarlo alla chiusura

    def stop(self):
        _kill_proc_tree(getattr(self, "proc", None))

    def run(self):
        try:
            from core import ltx2_service
            res = ltx2_service.generate(self.params, timeout=3600)
        except Exception as e:                              # noqa: BLE001
            res = {"status": "error", "error": str(e)}
        self.done.emit(res)


class _SeedVR2Worker(QThread):
    """SeedVR2 super-risoluzione video (CLI standalone, venv isolato). Stream log + path finale."""
    done = Signal(dict)
    progress = Signal(str)

    def __init__(self, src: str, out: str, *, resolution: int, batch_size: int, chunk_size: int,
                 model: str = "", block_swap: int = 0, temporal_overlap: int = 0):
        super().__init__()
        self.src, self.out = src, out
        self.resolution, self.batch_size, self.chunk_size = resolution, batch_size, chunk_size
        self.model, self.block_swap, self.temporal_overlap = model, block_swap, temporal_overlap

    def run(self):
        from core import seedvr2_service as sv
        try:
            sv.upscale_video(self.src, self.out, resolution=self.resolution, batch_size=self.batch_size,
                             chunk_size=self.chunk_size, model=self.model or "", block_swap=self.block_swap,
                             temporal_overlap=self.temporal_overlap,
                             on_log=lambda l: self.progress.emit(l))
            self.done.emit({"status": "success", "output": [self.out]})
        except Exception as e:                          # noqa: BLE001
            self.done.emit({"status": "error", "error": str(e)})


class _FlashVSRWorker(QThread):
    """FlashVSR v1.1 super-risoluzione video fino a ×4, default ×2 (repo diffsynth, venv isolato). Stream log + path."""
    done = Signal(dict)
    progress = Signal(str)

    def __init__(self, src: str, out: str, target_scale: float = 2.0):
        super().__init__()
        self.src, self.out, self.target_scale = src, out, target_scale

    def run(self):
        from core import flashvsr_service as fv
        try:
            fv.upscale_video(self.src, self.out, target_scale=self.target_scale,
                             on_log=lambda l: self.progress.emit(l))
            self.done.emit({"status": "success", "output": [self.out]})
        except Exception as e:                          # noqa: BLE001
            self.done.emit({"status": "error", "error": str(e)})


class _FlashVSRPlusWorker(QThread):
    """FlashVSR+ (Turbo/Base) super-risoluzione video fino a ×4 (pacchetto FlashVSR_Plus, SDPA, VRAM adattiva). Log + path."""
    done = Signal(dict)
    progress = Signal(str)

    def __init__(self, src: str, out: str, target_scale: float = 2.0, mode: str = "turbo",
                 temporal_compression: int = 4):
        super().__init__()
        self.src, self.out, self.target_scale = src, out, target_scale
        self.mode, self.temporal_compression = mode, temporal_compression

    def run(self):
        from core import flashvsrplus_service as fvp
        try:
            fvp.upscale_video(self.src, self.out, target_scale=self.target_scale, mode=self.mode,
                              temporal_compression=self.temporal_compression,
                              on_log=lambda l: self.progress.emit(l))
            self.done.emit({"status": "success", "output": [self.out]})
        except Exception as e:                          # noqa: BLE001
            self.done.emit({"status": "error", "error": str(e)})


class _H3UpsampleWorker(QThread):
    """H3 Video Upsampler (MiniMax H3 nativo, venv diffusers_next): upscaler latente + ri-disegno a tile. Log + path."""
    done = Signal(dict)
    progress = Signal(str)

    def __init__(self, src: str, out: str, params: dict):
        super().__init__()
        self.src, self.out, self.params = src, out, dict(params)

    def run(self):
        from core import h3_upsample_service as hs
        try:
            cfg = {k: v for k, v in self.params.items() if k not in ("video", "output")}
            hs.upscale_video(self.src, self.out, on_log=lambda l: self.progress.emit(l), **cfg)
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
        self._ic_job = None                                    # IcJob (Fase 1 IC-LoRA), se la modalità lo usa

    def _probe_wh(self, video: str) -> tuple[int, int]:
        """Risoluzione (w, h) del video via ffprobe. (0, 0) se illeggibile."""
        try:
            import json as _j2, subprocess as _sp2
            r = _sp2.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "json", video], capture_output=True, text=True, timeout=30)
            s = _j2.loads(r.stdout)["streams"][0]
            return int(s["width"]), int(s["height"])
        except Exception:                                       # noqa: BLE001
            return 0, 0

    def run(self):
        import json as _j, os as _os, subprocess as _sp, time as _t
        try:
            cfg = self.cfg
            src = cfg["video"]
            base = _os.path.splitext(_os.path.basename(src))[0]
            stamp = int(_t.time())
            # COORDINAMENTO (mancava): SAM2/GroundingDINO (~3-5GB) e poi LTX-2 partono da questo
            # worker come subprocess DIRETTI, fuori dal dispatch — libera gli altri motori e
            # aspetta il rilascio effettivo prima del primo load GPU.
            try:
                from core import mem_coordinator as _mc
                _mc.free_and_wait("video", need_mb=6000, wait_s=20)
            except Exception:                               # noqa: BLE001
                pass
            out = _os.path.join(self.out_dir, f"edit_{cfg['mode']}_{base}_{stamp}.mp4")
            params = {"prompt": cfg["prompt"], "negative": "worst quality, blurry",
                      "edit_lora": cfg["edit_lora"], "control_strength": float(cfg.get("strength", 1.0)),
                      "width": int(cfg.get("width", 512)), "height": int(cfg.get("height", 320)),
                      "num_frames": int(cfg.get("num_frames", 25)), "fps": int(cfg.get("fps", 25)),
                      # niente offload forzato: 'cpu' = block-streaming = pinned memory ⇒ SEGFAULT
                      # col pagefile da 16GB. In auto il worker sceglie residente (più veloce). (13/07)
                      "seed": 42, "output": out}
            for _k in ("model", "gguf_transformer", "ltx_release"):   # checkpoint ESATTAMENTE come scelto nel tab Video
                if cfg.get(_k):
                    params[_k] = cfg[_k]
            if cfg["mode"] == "inpaint":                       # #59 inpaint: SAM2 track → latent-inpaint (vinpaint)
                # Logica condivisa con la REST API (core/video_inpaint.py): LoRA t2v (a parole) o r2v (con foto di
                # riferimento, scelta dell'utente), maschere SAM2, segmenti per i clip lunghi.
                from core import video_inpaint as _vi
                _miss = _vi.missing_weight(cfg)
                if _miss:
                    self.done.emit({"status": "error", "error": _miss}); return
                params = _vi.inpaint_params(cfg, out)
                if _vi.has_reference(cfg):
                    # punto di aggancio @image_N (task 6): qui i riferimenti sono UNO (la foto del campo); quando il
                    # meccanismo dei token sara' nel ramo, i nomi citati nel prompt vanno risolti prima di arrivare qui.
                    _note = _vi.fit_reference(params, self.out_dir, stamp)
                    if _note:
                        self.progress.emit("[inpaint] " + _note)
                    self.progress.emit(f"[inpaint] foto di riferimento → LoRA r2v ({_os.path.basename(cfg['ref_image'])})")
                if params["num_frames"] > _vi.SEGMENT_FRAMES:
                    self.progress.emit(f"[inpaint] clip lungo ({params['num_frames']} frame): "
                                       f"{_vi.segments_info(params['num_frames'])} segmenti da {_vi.SEGMENT_FRAMES}")
                gpath = _os.path.join(self.out_dir, f"_guide_{base}_{stamp}.mp4")
                try:
                    _vi.attach_tracking(params, src, cfg["object"], gpath, int(cfg.get("mask_dilate", 24)),
                                        log=self.progress.emit)
                except RuntimeError as _e:
                    self.done.emit({"status": "error", "error": str(_e)}); return
            elif cfg["mode"] == "replace_char":                # #102 Sostituisci personaggio (union-control-ref)
                from core import controlnet_preprocess as _cp
                # PREFLIGHT: il peso union-control-ref deve esistere (altrimenti il worker degrada IN SILENZIO
                # a ti2v ignorando canny+union-control). Errore chiaro con hint.
                _lp = _os.path.join(r"E:\AI\Models\LoRA", str(cfg["edit_lora"]).replace("/", _os.sep))
                if not _os.path.exists(_lp):
                    self.done.emit({"status": "error", "error":
                        f"Peso «Sostituisci personaggio» mancante:\n{_lp}\nScaricalo dal tab «Pesi modelli» "
                        "(IC-LoRA union-control-ref di LTX-2.3)."}); return
                # VALIDAZIONE immagine di riferimento vs aspetto del video (best practice: primo frame stessa inquadratura)
                try:
                    from PIL import Image as _Im
                    _rw, _rh = _Im.open(cfg["ref_image"]).size
                    _vw, _vh = self._probe_wh(src)
                    if _vw and _vh and _rh and _vh and abs(_rw / _rh - _vw / _vh) > 0.15:
                        self.progress.emit(f"[replace] ⚠ immagine di riferimento {_rw}x{_rh}: aspetto diverso dal "
                                           f"video {_vw}x{_vh} → usa il PRIMO FRAME del video col nuovo personaggio.")
                except Exception:                              # noqa: BLE001
                    pass
                canny_v = _os.path.join(self.out_dir, f"_canny_{base}_{stamp}.mp4")
                self.progress.emit("[replace] canny del video sorgente (guida struttura)…")
                _cp.preprocess_video(src, "canny", canny_v)    # struttura/movimento preservati
                params["control_video"] = canny_v
                params["image_path"] = cfg.get("ref_image")    # immagine sostitutiva = riferimento (frame 0)
                params["control_strength"] = 0.6               # union-control-ref (ricetta SECourses)
                params["ref_strength"] = float(cfg.get("ref_strength", 0.8))   # #102 forza identità del riferimento
                params["loras"] = [{"name": r"E:\AI\Models\LoRA\LTX-2\LTX2.3_Crisp_Enhance.safetensors",
                                    "strength": 1.0}]          # enhance nitidezza (chiave "name" = convenzione worker)
                params["_mux_audio_from"] = src                # preserva l'audio originale
                # #102 PRECISIONE: se l'utente indica il personaggio da sostituire → SAM2 lo traccia e il worker
                # ricompone SOLO quell'area (sfondo intatto al pixel). Vuoto = sostituzione su tutto il frame.
                if cfg.get("object"):
                    self.progress.emit("[replace] SAM2: traccio il personaggio da sostituire…")
                    _seg = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)),
                                         "inference_server", "backends", "video_segment.py")
                    _guide = _os.path.join(self.out_dir, f"_rcguide_{base}_{stamp}.mp4")
                    _sp.run([self.INF_PY, "-u", _seg, src, cfg["object"], _guide,
                             str(int(cfg.get("mask_dilate", 24)))],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
                    _td = _os.path.splitext(src)[0] + "_track"
                    if _os.path.isdir(_os.path.join(_td, "masks")):
                        params["char_mask_dir"] = _os.path.join(_td, "masks")
                        params["original_frames_dir"] = _os.path.join(_td, "frames")
                        params["mask_blur"] = float(cfg.get("mask_blur", 6.0))
                    else:
                        self.progress.emit("[replace] ⚠ tracking SAM2 fallito → sostituzione su tutto il frame.")
            elif cfg["mode"] in vem.MODE_SPECS and vem.MODE_SPECS[cfg["mode"]].cap:
                # Fase 1 IC-LoRA (restauro/outpaint/restyle/cameraman): logica condivisa con la REST API
                # (core/video_edit_ic.py): preflight peso, tetti, segmenti, preprocess, checkpoint come scelto.
                from core import video_edit_ic as _vic
                try:
                    _ij = _vic.build_job(cfg, self.out_dir, stamp, out, log=lambda m: self.progress.emit(m))
                except _vic.IcJobError as _e:
                    self.done.emit({"status": "error", "error": str(_e)}); return
                for _n in _ij.notes:
                    self.progress.emit(f"[edit] {_n}")
                params = _ij.params                            # sostituisce i params storici (incl. _strict_ic, model)
                self._ic_job = _ij
            else:
                params["control_video"] = src                  # edit/style: video sorgente come guide
            jf = _os.path.join(self.out_dir, f"_editp_{stamp}.json")
            _j.dump(params, open(jf, "w"))
            worker = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)),
                                   "workers", "ltx2_infer.py")
            self.progress.emit("[edit] genero il video (LTX-2)…")
            try:                                            # attesa piena pre-22B (il SAM2 è appena uscito:
                from core.ltx_video_job import free_video_vram as _fvv   # la sua VRAM sta rientrando)
                _fvv(arch="ltx")
            except Exception:                               # noqa: BLE001
                pass
            r2 = _sp.run([self.LTX2_PY, "-u", worker, jf], capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
            if _os.path.exists(out) and getattr(self, "_ic_job", None) is not None:
                try:                                           # post IC-LoRA: outpaint (gamma+originale) e audio originale
                    from core import video_edit_ic as _vic2
                    out = _vic2.finalize(self._ic_job, out, log=lambda m: self.progress.emit(m))
                except Exception as _e2:                       # noqa: BLE001
                    self.done.emit({"status": "error", "error": f"post-processo fallito: {_e2}"}); return
                self.done.emit({"status": "success", "output": self._ic_job.outputs(out)})
            elif _os.path.exists(out):
                _mux = params.get("_mux_audio_from")           # #102 rimetti l'audio originale sul video rigenerato
                if _mux and _os.path.exists(_mux):
                    muxed = out[:-4] + "_audio.mp4"
                    try:
                        _sp.run(["ffmpeg", "-y", "-i", out, "-i", _mux, "-c:v", "copy",
                                 "-map", "0:v:0", "-map", "1:a:0?", "-shortest", muxed],
                                capture_output=True, text=True)
                        if _os.path.exists(muxed):
                            out = muxed
                    except Exception:                          # noqa: BLE001
                        pass
                self.done.emit({"status": "success", "output": [out]})
            else:
                self.done.emit({"status": "error", "error": (r2.stdout or "")[-200:] + (r2.stderr or "")[-300:]})
        except Exception as e:                              # noqa: BLE001
            self.done.emit({"status": "error", "error": str(e)})


class _NativeMultiTalkWorker(QThread):
    """Lip-sync MultiTalk (Wan2.1 14B) NATIVO, senza ComfyUI: subprocess nel venv consolidato (core/multitalk_service,
    motore workers/wan_mt). Riporta progress e path finale; stop() termina il processo."""
    done = Signal(dict)
    progress = Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self.proc = None

    def stop(self):
        _kill_proc_tree(getattr(self, "proc", None))

    def run(self):
        from core import multitalk_service as mts
        try:
            out = mts.run_job(self.params, on_log=lambda l: self.progress.emit(l),
                              on_proc=lambda p: setattr(self, "proc", p))
            self.done.emit({"status": "success", "output": [out]})
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


class _LongCatWorker(QThread):
    """Avatar parlante LongCat 1.5 (subprocess nel venv dedicato) fuori dal thread UI."""
    done = Signal(dict)
    progress = Signal(str)

    def __init__(self, params: dict, service=None):
        super().__init__()
        self.params = params
        self.service = service                          # None = LongCat; core.minimax_avatar = MiniMax H3
        self._job = None

    def run(self):
        try:
            from core import longcat_service as _lc
            lc = self.service or _lc
            errs = lc.preflight(self.params)
            if errs:
                self.done.emit({"status": "error", "error": "\n".join(errs)}); return
            lines: list[str] = []

            def _log(s: str):
                lines.append(s)
                self.progress.emit(s)
            self._job = lc.make_job(self.params, on_log=_log)
            rc = self._job.run()
            out = self._job.output
            if rc == 0 and out and os.path.exists(out):
                self.done.emit({"status": "success", "output": [out]})
            else:
                err = next((l for l in reversed(lines) if l.startswith("ERRORE")), "") or "\n".join(lines[-12:])
                self.done.emit({"status": "error", "error": err or f"worker avatar terminato con codice {rc}"})
        except Exception as e:                          # noqa: BLE001
            self.done.emit({"status": "error", "error": str(e)})

    def stop(self):
        if self._job is not None:
            self._job.stop()


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
        self.proc = None                                    # #91: subprocess, per terminarlo alla chiusura

    def stop(self):
        """#91: ferma la generazione. Path SERVER (:8190): cancel HTTP cooperativo — MAI il kill,
        il processo server è condiviso e deve sopravvivere. Path subprocess: kill dell'albero."""
        _jid = getattr(self, "_server_jid", None)
        if _jid:
            try:
                from core import ltx2_service
                ltx2_service.cancel_native(_jid)
            except Exception:                               # noqa: BLE001
                pass
            return
        _kill_proc_tree(getattr(self, "proc", None))

    LINGBOT_PY = r"E:\AI\Tools\lingbot_venv311\Scripts\python.exe"  # py3.11+torch2.11+FA2 (ricetta corretta)

    def run(self):
        import json as _json, os as _os, subprocess as _sp, tempfile as _tf
        try:
            # arch → venv+worker dedicati (ltx/lingbot/krea_rt) dal dispatch centrale. Default LTX-2.
            # `runner_for` (NON _ARCH_RUNNER): per l'arch 'ltx' il venv dipende dalla RELEASE del
            # checkpoint, e _ARCH_RUNNER["ltx"] è cablato sulla 2.3 → prima la 2.5 non partiva mai
            # da qui, contro il docstring di runner_for che dichiara questo il punto UNICO.
            _arch = (self.params.get("arch") or "ltx").lower()
            from core.ltx_video_job import runner_for as _runner_for
            py, worker = _runner_for(self.params)
            if _arch == "ltx":
                self.progress.emit(f"[release] LTX {self.params.get('ltx_release') or '2.3'} "
                                   f"→ {_os.path.basename(_os.path.dirname(_os.path.dirname(py)))}")
            if not _os.path.exists(py):
                self.done.emit({"status": "error",
                                "error": f"venv per arch '{_arch}' mancante: {py}. Esegui lo script install_*."})
                return
            # #4 separazione voce (Canta): isola i vocals dalla canzone PRIMA del lip-sync (in-process,
            # venv comfy con demucs). Fail-safe: se fallisce, resta l'audio originale.
            if self.params.pop("_sep_vocals", False) and self.params.get("audio_path"):
                from core import vocal_separation as _vs
                self.progress.emit("[canta] separo la voce dalla base (demucs)…")
                self.params["audio_path"] = _vs.separate_vocals(
                    self.params["audio_path"], on_log=lambda l: self.progress.emit("[demucs] " + l))
            # #3 talk region: cattura e rimuovi i parametri di compositing PRIMA del dump JSON
            _talk = {"region": self.params.pop("_talk_region", None),
                     "src": self.params.pop("_talk_source", None),
                     "offset": self.params.pop("_talk_offset", 0)}
            # ── RESIDENZA DI DEFAULT (unificazione motori): per arch 'ltx' il job va al SERVER
            # persistente :8190 (pipeline + parcheggio RAM sopravvivono tra le generazioni →
            # niente reload da SSD). Fallback AUTOMATICO al subprocess per-job se il server non
            # parte o muore a metà. Opt-out: LTX_VIDEO_SERVER=0 / user_prefs 'video_server'.
            out_path, _tail, _err_extra = None, [], ""
            _served = False
            if _arch == "ltx":
                try:
                    from core import ltx2_service as _svc
                    # Il server :8190 è avviato SEMPRE col venv 2.3 (ltx2_service.LTX2_PY). Un job
                    # con release diversa finirebbe nel PACCHETTO 2.3, che non sa caricare un text
                    # encoder da file singolo e muore con «No files matching pattern
                    # 'tokenizer.model'». gen_service ha sempre avuto questa guardia; qui mancava,
                    # e la GUI ci mandava dentro anche i job 2.5. Release ≠ default → subprocess,
                    # che sceglie il venv giusto via runner_for().
                    from core import ltx_release as _lrel
                    _rel = self.params.get("ltx_release") or _lrel.DEFAULT_RELEASE
                    if _rel != _lrel.DEFAULT_RELEASE:
                        self.progress.emit(
                            f"[nativo] LTX {_rel}: il server persistente monta la "
                            f"{_lrel.DEFAULT_RELEASE} → subprocess dedicato")
                    elif _svc.server_enabled() and _svc.ensure_server():
                        self.progress.emit("[nativo] server video persistente :8190 (residenza)")
                        out_path, _served, _err_extra = self._run_via_server(_svc)
                except Exception as _se:                    # noqa: BLE001
                    self.progress.emit(f"[nativo] server non disponibile ({_se}) → subprocess")
                    _served = False
            if not _served:
                # COORDINAMENTO (mancava su TUTTO il ramo subprocess: LingBot/krea_rt/self_forcing
                # e il fallback ltx): libera gli altri motori e ASPETTA il rilascio effettivo,
                # come fa run_clip — senza, il worker caricava con FLUX/Krea2 ancora residenti.
                try:
                    from core.ltx_video_job import free_video_vram as _fvv
                    _fvv(model=str(self.params.get("model", "") or ""),
                         gguf=str(self.params.get("gguf_transformer", "") or ""), arch=_arch)
                except Exception:                           # noqa: BLE001
                    pass
                jf = _tf.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
                _json.dump(self.params, jf); jf.close()
                env = dict(_os.environ)
                env["HF_HUB_OFFLINE"] = "1"
                env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                proc = _sp.Popen([py, "-u", worker, jf.name], env=env,
                                 stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, encoding="utf-8", errors="replace")
                self.proc = proc                            # #91: esposto per stop() alla chiusura
                # Output COMPLETO su file, sovrascritto a ogni generazione. In memoria se ne
                # tenevano le ultime 40 righe, e in un traceback lungo l'ultima — quella col TIPO
                # di eccezione — resta fuori: si vede dove si è rotto, non perché.
                try:
                    _dir = _os.path.dirname(str(self.params.get("output") or "")) or str(OUTPUT_DIR)
                    _lg = open(_os.path.join(_dir, "gui_last_worker.log"),
                               "w", encoding="utf-8", errors="replace")
                except (OSError, TypeError):
                    _lg = None
                for line in proc.stdout:
                    line = line.rstrip()
                    if _lg is not None:
                        _lg.write(line + "\n"); _lg.flush()
                    if line:
                        _tail.append(line)
                        if len(_tail) > 40:
                            _tail.pop(0)
                    if line.startswith("[") or line.startswith("OK:") or "Error" in line or "error" in line \
                            or "Traceback" in line:
                        self.progress.emit(line)
                    if line.startswith("OK:"):
                        out_path = line[3:].strip()
                proc.wait()
                if _lg is not None:
                    _lg.close()
            # #3 talk region: composita solo bocca/volto sul frame sorgente (reimagine=off)
            if out_path and _os.path.exists(out_path) and _talk["region"] and _talk["src"]:
                from core import talk_region as _tr
                self.progress.emit(f"[talk] compositing regione '{_talk['region']}' (solo bocca/volto)…")
                comp = out_path.rsplit(".", 1)[0] + "_region.mp4"
                out_path = _tr.composite_talk_region(
                    out_path, _talk["src"], comp, region=_talk["region"],
                    mask_offset=int(_talk["offset"] or 0), on_log=lambda l: self.progress.emit("[talk] " + l))
            if out_path and _os.path.exists(out_path):
                self.done.emit({"status": "success", "output": [out_path]})
            else:
                # NON più "nessun output" e basta: mostra il MOTIVO reale (path server: l'errore del
                # job — il tail del traceback lato :8190; path subprocess: ring-buffer stdout + exit).
                _why = _err_extra or "\n".join(_tail[-18:]) or "(nessun output dal worker)"
                _p = getattr(self, "proc", None)
                _rc = f" (exit {_p.returncode})" if _p is not None else ""
                # La CAUSA in cima. Il traceback è lungo e il dialog lo tronca: l'ultima riga —
                # quella che dice il TIPO di eccezione, l'unica che serve davvero — finiva fuori
                # dallo schermo, e restava visibile solo il punto in cui si è rotto. Qui si pesca
                # la riga finale non vuota (formato «ValueError: messaggio») e si mette in testa.
                _causa = ""
                for _ln in reversed((_why or "").splitlines()):
                    _s = _ln.strip()
                    if _s and not _s.startswith(("File ", "  ", "Traceback")):
                        _causa = _s
                        break
                _testa = f"⛔ {_causa}\n\n" if _causa else ""
                # un OOM / VRAM insufficiente non è «nessun output»: il titolo dice la causa vera
                _esito = "memoria GPU insufficiente" if (
                    "VRAM" in _causa or "memoria GPU" in _causa or "out of memory" in _causa.lower()
                ) else "nessun output"
                self.done.emit({"status": "error",
                                "error": f"LTX-2 nativo: {_esito}{_rc}.\n{_testa}{_why}\n\n"
                                         f"Log completo: {OUTPUT_DIR}\\gui_last_worker.log"})
        except Exception as e:                          # noqa: BLE001
            self.done.emit({"status": "error", "error": str(e)})

    def _run_via_server(self, svc) -> tuple:
        """Submit+poll sul server video persistente. Ritorna (out_path, served, err):
        served=False → il server è morto/il job è andato perso → il chiamante fa fallback al
        subprocess con gli STESSI params (già ripuliti dal pre-processing: retry idempotente)."""
        import time as _t
        try:
            jid = svc.submit_native(dict(self.params))
        except Exception as e:                          # noqa: BLE001
            self.progress.emit(f"[nativo] submit al server fallito ({e}) → subprocess")
            return None, False, ""
        self._server_jid = jid
        _net_fails, _last_msg, _last_step = 0, "", -1
        while True:
            _t.sleep(2.5)
            try:
                h = svc.poll_native(jid)
                _net_fails = 0
            except Exception:                           # noqa: BLE001
                _net_fails += 1
                if _net_fails >= 5 and not svc.is_up():
                    self.progress.emit("[nativo] server video morto a metà job → fallback subprocess")
                    self._server_jid = None
                    return None, False, ""
                continue
            st = h.get("status")
            pr = h.get("progress") or {}
            if pr.get("msg") and pr["msg"] != _last_msg:
                _last_msg = pr["msg"]
                self.progress.emit("[stage] " + _last_msg)
            if pr.get("step") is not None and pr.get("step") != _last_step:
                _last_step = pr["step"]
                self.progress.emit(f"[nativo] step {pr.get('step')}/{pr.get('total', '?')}")
            if st == "success":
                self._server_jid = None
                return (h.get("output") or [None])[0], True, ""
            if st == "cancelled":
                self._server_jid = None
                return None, True, "generazione annullata dall'utente"
            if st == "error":
                self._server_jid = None
                _err = str(h.get("error", ""))
                # «size mismatch … shape in current model is torch.Size([3840…])» = il server ha
                # costruito il modello con i DEFAULT del configuratore (30 teste × 128) invece che
                # con la config del checkpoint: il suo stato è sporco, e resterà tale per ogni job
                # successivo finché non lo si riavvia. Il subprocess, che parte pulito, carica lo
                # stesso checkpoint senza un errore — verificato. Quindi si ricade lì invece di
                # consegnare all'utente un traceback che non può risolvere.
                if "size mismatch" in _err or "Error(s) in loading state_dict" in _err:
                    self.progress.emit("[nativo] stato del server incoerente col checkpoint "
                                       "→ lo azzero e passo al subprocess")
                    try:
                        svc.reset_server()
                    except Exception:                   # noqa: BLE001
                        pass
                    return None, False, ""
                return None, True, _err[-1500:]
            if st == "unknown":                         # server riavviato: job perso → fallback
                self._server_jid = None
                return None, False, ""


class _ModelAuditWorker(QObject):
    """F2 unified-model-manager: audit del registro dimensioni modelli in background all'avvio.
    Solo header safetensors + JSON (model_detect.scan + calibrate_all livello-1): nessun torch,
    nessuna GPU — sicuro nel processo GUI. Thread Python DAEMON (non QThread): chiudere l'app mentre
    l'audit gira non deve abortire il processo («QThread destroyed while running»)."""
    done = Signal(dict)

    def start(self):
        import threading
        threading.Thread(target=self.run, name="model-audit", daemon=True).start()

    def run(self):
        try:
            from core import startup_model_audit
            self.done.emit(startup_model_audit.audit())
        except Exception as e:                          # noqa: BLE001
            self.done.emit({"error": str(e)})


class _NativeMigrateWorker(QObject):
    """Converte in formato NATIVO i single-file ComfyUI già su disco (copiati a mano, collegati dall'installer):
    sul PC dell'utente la conversione è automatica sia al download (core/downloader) sia qui, all'avvio.
    Solo voci di catalogo col campo `convert` il cui sorgente esiste: di norma non fa nulla. Niente GPU.
    Thread Python DAEMON (non QThread): chiudere l'app durante una conversione non deve abortire il processo
    («QThread destroyed while running»); la conversione è comunque sicura (scrive un .part, il sorgente si
    cancella solo dopo la verifica) e riparte al prossimo avvio."""
    log = Signal(str)

    def start(self):
        import threading
        threading.Thread(target=self._run, name="native-migrate", daemon=True).start()

    def _run(self):
        try:
            from core import native_convert as nc
            done = nc.migrate_all(log=self.log.emit)
            if done:
                self.log.emit(f"[native] convertiti in formato nativo: {', '.join(done)}")
        except Exception as e:                          # noqa: BLE001 — non blocca l'avvio
            self.log.emit(f"[native] conversione all'avvio non riuscita: {e}")


class MainWindow(QMainWindow):
    # anteprime risultato pronte (dal thread di decode) → dialog sul thread GUI (queued connection)
    _result_prev_ready = Signal(str, object, object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("window.title"))
        self.resize(1180, 740)
        self._result_prev_ready.connect(self._show_inpaint_result_ready)

        self.driver = ComfyDriver()
        self.bridge = ComfyBridge()
        try:
            self.engine = TemplateEngine()
            self._engine_err = None
        except Exception as e:  # template non ancora catturato
            self.engine = None
            self._engine_err = str(e)

        self.jobs = GenerationQueue(self)       # coda SERIALE delle generazioni (no overlap/cancel reciproci)
        # lock GPU inter-processo: se un ALTRO processo (API/training/musica) tiene la GPU, la coda
        # attende e lo dice in statusbar (ereditato anche dalla variante Pro, che subclassa questa).
        self.jobs.gpu_waiting.connect(
            lambda tag: self.statusBar().showMessage(
                tr("queue.gpu_busy", "GPU occupata da {tag} — il job parte appena libera.")
                .format(tag=tag), 4000))
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
        self.frame1_tab.request_magicquill.connect(self.generate_magicquill)
        self.frame1_tab.request_generate.connect(self.generate_image)
        self.frame1_tab.request_upscale.connect(self.generate_upscale)
        self.frame1_tab.request_recolor.connect(self.generate_recolor)
        # selezione oggetto (SAM): testo dal tab, regione (click/box) dal canvas
        self.frame1_tab.request_segment_text.connect(self.segment_by_text)
        self.frame1_tab.canvas.request_segment_region.connect(self.segment_by_region)
        self.video_def_tab.request_enhance.connect(self.generate_video_enhance)
        self.video_def_tab.request_face_restore.connect(self.generate_face_restore)
        self.video_def_tab.request_exr.connect(self.generate_exr_export)
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

        # Solo il server NATIVO viene avviato (migrazione zero-ComfyUI completata): l'auto-avvio ComfyUI è
        # stato rimosso (#51). Il polling sotto rileva il server nativo quando è pronto.
        self._maybe_autostart_native()

        # poll stato ComfyUI + completamento job
        self._status_timer = QTimer(self); self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start(3000); self._refresh_status()

        # ── F2 unified-model-manager: audit del registro dimensioni modelli all'apertura ──
        # Thread in background (solo header+JSON, zero torch/GPU): verifica se i modelli locali sono
        # tutti classificati; il risultato (self._audit_result) alimenta il calibratore livello-2 (F7).
        QTimer.singleShot(2000, self._start_model_audit)
        QTimer.singleShot(5000, self._start_native_migration)   # single-file ComfyUI su disco → nativi

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

    # ---------- F2: audit registro dimensioni modelli (avvio) ----------
    def _start_native_migration(self):
        self._migrate_worker = _NativeMigrateWorker()
        self._migrate_worker.log.connect(self._log)
        self._migrate_worker.start()

    def _start_model_audit(self):
        self._log(tr("audit.running", "Verifica registro modelli…"))
        self._audit_worker = _ModelAuditWorker()
        self._audit_worker.done.connect(self._on_model_audit_done)
        self._audit_worker.start()

    def _on_model_audit_done(self, res: dict):
        if res.get("error"):
            self._log(f"[audit] errore non bloccante: {res['error']}")
            return
        self._audit_result = res                        # consumato dal calibratore livello-2 (F7)
        msg = tr("audit.summary",
                 "Modelli: {t} — {m} misurati, {e} stimati, {u} sconosciuti, {k} da misurare").format(
            t=res["total"], m=len(res["measured"]), e=len(res["estimated"]),
            u=len(res["unknown"]), k=len(res["to_measure"]))
        self._log("[audit] " + msg)
        try:
            self.statusBar().showMessage(msg, 20000)
        except Exception:                               # noqa: BLE001
            pass
        self._maybe_start_calibration(res)

    def _maybe_start_calibration(self, res: dict):
        """F7: dialog di CONSENSO (una volta, policy salvata) + misura reale in background a GPU
        inattiva. La misura carica un modello alla volta nel venv giusto e si interrompe
        all'istante se accodi una generazione."""
        to_measure = [t for t in res.get("to_measure", []) if t.get("venv")]
        if not to_measure:
            return
        from core import user_prefs as _up
        pol = str(_up.get("calib_policy", ""))
        if not pol:                                     # primo avvio post-feature: chiedi
            from PySide6.QtWidgets import QMessageBox
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle(tr("calib.title", "Misura reale dei modelli"))
            _min = max(1, len(to_measure))
            box.setText(tr("calib.msg",
                "{k} modelli non hanno ancora un peso VRAM misurato (~{m} min in totale, a GPU "
                "inattiva).\nMisurarli rende precisa la gestione automatica della memoria.\n\n"
                "La misura NON parte mai durante le tue generazioni e si interrompe subito se ne "
                "avvii una.").format(k=_min, m=_min))
            b_now = box.addButton(tr("calib.now", "Misura a GPU libera"), QMessageBox.AcceptRole)
            box.addButton(tr("calib.later", "Solo al primo uso"), QMessageBox.RejectRole)
            b_never = box.addButton(tr("calib.never", "Mai"), QMessageBox.DestructiveRole)
            box.exec()
            pol = ("idle" if box.clickedButton() is b_now
                   else ("never" if box.clickedButton() is b_never else "on_use"))
            _up.set("calib_policy", pol)
        if pol != "idle":
            return
        from core.vram_calibrator import CalibrationScheduler
        self._calib = CalibrationScheduler(self.jobs, self)
        self._calib.progress.connect(self._log)
        self._calib.set_targets(to_measure)
        self._calib.start()

    # ---------- UI ----------
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # sinistra: tab
        self.tabs = QTabWidget()
        self.video_tab = VideoTab()
        self.frame1_tab = Frame1Tab()
        # rapporto del canvas per il pannello riquadri FLUX.3 (nessun torch.cuda: solo i valori del tab Video)
        self.frame1_tab.size_provider = lambda: (int(self.video_tab.get_values().get("width", 1024)),
                                                 int(self.video_tab.get_values().get("height", 1024)))
        self.frame1_tab.gen_arch.currentTextChanged.connect(self._apply_flux3_size_cap)
        self._apply_flux3_size_cap()
        _NativeImageWorker.klein_te_source = self.frame1_tab.klein_te_choice      # encoder Klein scelto in Inpaint
        self.video_tab.request_pulid_keyframe.connect(self._gen_pulid_keyframe_for_video)
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
        self.avatar_tab = AvatarTab()
        self.rig_tab = RigTab()
        self.convert_tab = ConvertTab()
        self.anim_tab = AnimTab()
        self.train_tab = TrainTab()
        self.lora_video_tab = LoraVideoTab()
        self.lora_image_tab = LoraImageTab()
        self.train_lora_video_tab = TrainLoraVideoTab()
        self.train_tab.lora_ready.connect(lambda _p: self._refresh_status())
        self.music_tab = MusicTab()
        self.world_tab = WorldTab()
        self.world_tab.get_canvas_image = lambda: (self.frame1_tab.image_path or "")   # #96.5: fallback immagine dal canvas
        # PONTE catena 3D (review): la scena WorldGen non era collegata al resto — ora il GLB entra nel
        # tab Rig come "ultima mesh" (stesso pattern Trellis→Rig→Anim).
        self.world_tab.mesh_ready.connect(lambda g: self.rig_tab.set_last_mesh(g))
        self.weights_tab = WeightsTab()
        # FASE D: pesi arrivati (download/collega/importa) → l'inventario dei dropdown si RIFÀ da
        # solo (prima _inventory_set restava True per sempre: serviva riavviare l'app).
        self.weights_tab.inventory_changed.connect(self.refresh_model_inventory)
        self.mocap = MocapService(self)
        self.models_tab = self._make_models_tab()           # Classica o Pro (variante utente)
        self.tabs.addTab(self.frame1_tab, tr("tab.images"))
        self.tabs.addTab(self.audio_tab, tr("tab.audio_voice"))
        self.tabs.addTab(self.video_tab, tr("tab.video"))
        self.tabs.addTab(self.controlnet_tab, tr("tab.controlnet"))
        self.tabs.addTab(self.video_extend_tab, tr("tab.extend_video"))
        from ui.tabs.video_edit_ai_tab import VideoEditAITab          # #24/#59/#60 editing video AI
        self.video_edit_ai_tab = VideoEditAITab()
        self.video_edit_ai_tab.request_edit.connect(self._gen_video_edit_ai)
        # checkpoint scelto nel tab Video: lo si MOSTRA (e si avvisa/blocca) nel tab Modifica Video AI, mai cambiato
        self.video_edit_ai_tab.set_model_getter(self.video_tab.get_values)
        self.video_tab.model.currentIndexChanged.connect(self.video_edit_ai_tab.refresh_checkpoint_hint)
        self.video_edit_ai_tab.request_open_video_tab.connect(lambda: self.tabs.setCurrentWidget(self.video_tab))
        self.video_edit_ai_tab.request_open_weights_tab.connect(lambda: self.tabs.setCurrentWidget(self.weights_tab))
        self.video_def_tab.request_open_weights_tab.connect(lambda: self.tabs.setCurrentWidget(self.weights_tab))
        # Doppiaggio da testo: «ho un mio audio» → Talk/Canta del tab Video (esistenti, non toccate); «ultimo audio»
        self.video_edit_ai_tab.request_open_v2v.connect(self._open_v2v_from_vedit)
        self.video_edit_ai_tab.set_last_audio_getter(lambda: self.avatar_tab._last_audio())
        # Movimento da tracce: immagine dal canvas del Frame 1, e rimando dal tab Video («Guida il movimento con tracce…»)
        self.video_edit_ai_tab.set_canvas_image_getter(lambda: (self.frame1_tab.image_path or ""))
        self.video_tab.request_motion_tracks.connect(self._open_motion_tracks_from_video)
        self.tabs.addTab(self.video_edit_ai_tab, tr("tab.video_edit_ai", "✨ Modifica Video AI"))
        # Avatar parlante (LongCat 1.5): foto/testo + voce → persona che parla, durata dall'audio
        self.avatar_tab.get_canvas_image = lambda: (self.frame1_tab.image_path or "")
        self.frame1_tab.canvas.image_changed.connect(self.avatar_tab.on_canvas_image_changed)
        self.avatar_tab.load_into_canvas.connect(
            lambda p: self.frame1_tab.canvas.load_image(p, fit_mode="canvas"))
        self.audio_tab.tts_generated.connect(self.avatar_tab.set_last_audio)
        self.music_tab.music_ready.connect(self.avatar_tab.set_last_audio)
        self.avatar_tab.request_avatar.connect(self.generate_avatar)
        self.avatar_tab.on_canvas_image_changed(self.frame1_tab.image_path or "")
        self.tabs.addTab(self.avatar_tab, tr("tab.avatar", "Avatar"))
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
        # LoRA Train Img. — dataset MANUALE (immagini + caption) FLUX.1/FLUX.2 via ai-toolkit
        self.tabs.addTab(self.train_tab, tr("tab.train", "LoRA Train Img."))
        # LoRA Wizard Vid. — dataset AUTO-generato da soggetti (LTX-2.3)
        self.tabs.addTab(self.lora_video_tab, tr("tab.lora_video", "LoRA Wizard Vid."))
        # LoRA Wizard Img. — dataset AUTO-generato da soggetti (FLUX.2)
        self.tabs.addTab(self.lora_image_tab, tr("tab.lora_image", "LoRA Wizard Img."))
        # Train LoRA Video — dataset MANUALE da spezzoni video esistenti (LTX-2.3)
        self.tabs.addTab(self.train_lora_video_tab, tr("tab.train_lora_video", "Train LoRA Video"))
        # Musica (HeartMuLa canzoni/strumentale + MMAudio sonorizzazione) — tab separato
        self.tabs.addTab(self.music_tab, tr("tab.music", "Musica"))
        # Mondo 3D (WorldGen: scena/ambiente da foto, mesh+splat, player) — tab separato
        self.tabs.addTab(self.world_tab, tr("tab.world", "Mondo 3D"))
        # Pesi modelli (wizard download da HuggingFace, locale-first) — tab separato
        self.tabs.addTab(self.weights_tab, tr("tab.weights", "Pesi modelli"))
        # Import Models diventa un sotto-tab di Avanzate
        self.advanced_tab.attach_models(self.models_tab)
        self.tabs.addTab(self.advanced_tab, tr("tab.advanced"))
        # #96.4: icona-emoji distintiva per ogni tab (language-independent, idempotente)
        _tab_emoji = {
            id(self.frame1_tab): "🖼️", id(self.audio_tab): "🎙️", id(self.video_tab): "🎬",
            id(self.controlnet_tab): "🎛️", id(self.video_extend_tab): "⏩",
            id(self.video_edit_ai_tab): "✨", id(self.avatar_tab): "🗣️", id(self.quality_tab): "💎",
            id(self.trellis_tab): "🧊", id(self.rig_tab): "🦴", id(self.convert_tab): "🔄",
            id(self.anim_tab): "🕺", id(self.train_tab): "🎓", id(self.lora_video_tab): "🎞️",
            id(self.lora_image_tab): "🖌️", id(self.music_tab): "🎵", id(self.world_tab): "🌍",
            id(self.weights_tab): "📦", id(self.advanced_tab): "⚙️",
        }
        for _i in range(self.tabs.count()):
            _em = _tab_emoji.get(id(self.tabs.widget(_i)))
            _txt = self.tabs.tabText(_i)
            if _em and not _txt.startswith(_em):
                self.tabs.setTabText(_i, f"{_em} {_txt.lstrip('✨ ').strip()}" if _txt.startswith("✨") else f"{_em} {_txt}")
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
        # Genera/Interrompi SPOSTATI nel footer ancorato del tab Video («Genera Video», richiesta
        # utente): qui restano solo Libera VRAM e Navigator. self.gen_btn/self.cancel_btn sono
        # ALIAS ai bottoni del video_tab così tutte le call-site esistenti restano valide.
        self.gen_btn = self.video_tab.generate_btn
        self.gen_btn.clicked.connect(self.on_generate)
        self.cancel_btn = self.video_tab.stop_btn
        self.cancel_btn.clicked.connect(self.driver.interrupt)
        self.free_vram_btn = QPushButton(tr("btn.free_vram"))
        self.free_vram_btn.setToolTip(tr("btn.free_vram.tt"))
        self.free_vram_btn.clicked.connect(self._on_free_vram)
        self.navigator_btn = QPushButton(tr("btn.navigator"))
        self.navigator_btn.setToolTip(tr("btn.navigator.tt"))
        self.navigator_btn.clicked.connect(self._open_navigator)
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
        # #96.1: barra tab MULTI-RIGA — la QTabBar nativa scorre in orizzontale; la nascondo e la
        # sostituisco con pulsanti in FlowLayout che vanno a capo → tutti i moduli visibili senza scroll.
        from ui.widgets.flow_layout import FlowLayout
        from PySide6.QtWidgets import QButtonGroup, QVBoxLayout as _QVBox
        self.tabs.tabBar().hide()
        _navw = QWidget(); self._tab_nav_layout = FlowLayout(_navw, margin=2, spacing=3)
        self._tab_nav_group = QButtonGroup(self); self._tab_nav_group.setExclusive(True)
        self._tab_nav_btns = []
        for _i in range(self.tabs.count()):
            _b = QToolButton(); _b.setText(self.tabs.tabText(_i)); _b.setCheckable(True)
            _b.setToolButtonStyle(Qt.ToolButtonTextOnly); _b.setCursor(Qt.PointingHandCursor)
            _b.setStyleSheet("QToolButton{border:1px solid #444;border-radius:6px;padding:3px 9px;}"
                             "QToolButton:hover{border-color:#6aa;}"
                             "QToolButton:checked{background:#3a6ea5;color:white;border-color:#3a6ea5;}")
            _b.setChecked(_i == self.tabs.currentIndex())
            _b.clicked.connect(lambda _c=False, _idx=_i: self.tabs.setCurrentIndex(_idx))
            self._tab_nav_group.addButton(_b, _i); self._tab_nav_layout.addWidget(_b); self._tab_nav_btns.append(_b)
        self.tabs.currentChanged.connect(
            lambda i: self._tab_nav_btns[i].setChecked(True) if 0 <= i < len(self._tab_nav_btns) else None)
        _tabs_container = QWidget(); _tcv = _QVBox(_tabs_container)
        _tcv.setContentsMargins(0, 0, 0, 0); _tcv.setSpacing(2)
        _tcv.addWidget(_navw); _tcv.addWidget(self.tabs, 1)
        self.main_splitter.addWidget(_tabs_container)
        self.main_splitter.setStretchFactor(0, 3); self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setChildrenCollapsible(False); self.main_splitter.setHandleWidth(6)
        root.addWidget(self.main_splitter)

        # Pannello CODA generazioni (dock): lista job + stato + annulla. Vedi ui/job_queue.py
        from PySide6.QtWidgets import QDockWidget
        self.queue_dock = QDockWidget(tr("queue.title", "Coda generazioni"), self)
        self.queue_dock.setObjectName("queue_dock")
        self.queue_panel = JobQueuePanel(self.jobs, self.queue_dock)
        self.queue_dock.setWidget(self.queue_panel)
        self.queue_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable
                                    | QDockWidget.DockWidgetFloatable)   # #96.3: chiudibile → si NASCONDE (non elimina)
        self.addDockWidget(Qt.RightDockWidgetArea, self.queue_dock)
        # #96.3: mini-icona nella nav-bar per ri-mostrare la coda quando è nascosta (QToolButton già importato)
        self._queue_btn = QToolButton(self)
        self._queue_btn.setText("📋"); self._queue_btn.setCheckable(True); self._queue_btn.setChecked(True)
        self._queue_btn.setToolTip(tr("queue.toggle.tt", "Mostra/nascondi la coda generazioni"))
        self._queue_btn.setStyleSheet("QToolButton{border:none;font-size:16px;padding:2px 6px;}")
        self._queue_btn.clicked.connect(lambda: self.queue_dock.setVisible(not self.queue_dock.isVisible()))
        self.queue_dock.visibilityChanged.connect(self._queue_btn.setChecked)
        self._tab_nav_layout.addWidget(self._queue_btn)   # #96.1/3: nella nav-bar (la tabBar nativa è nascosta)
        # Monitor memoria (RAM/VRAM dedicata/shared/GPU%) — finestra non-modale, toggle accanto alla coda
        self._mem_monitor = None
        self._mem_btn = QToolButton(self)
        self._mem_btn.setText("📊"); self._mem_btn.setCheckable(True)
        self._mem_btn.setToolTip(tr("mem.toggle.tt", "Monitor memoria (RAM/VRAM dedicata/shared/GPU)"))
        self._mem_btn.setStyleSheet("QToolButton{border:none;font-size:16px;padding:2px 6px;}")
        self._mem_btn.clicked.connect(self._toggle_mem_monitor)
        self._tab_nav_layout.addWidget(self._mem_btn)
        self.jobs.changed.connect(self._refresh_status)

        # La riga "ComfyUI: NATIVO @ http://… | Template: …" non compare più nella barra di stato:
        # il backend è nativo da tempo e nominare ComfyUI con l'indirizzo di un server che non si
        # usa confondeva e basta. La label resta viva (nascosta) perché _refresh_status continua a
        # scriverci: così lo stato è ancora interrogabile e riattivarla è una riga.
        self.status_lbl = QLabel("…")
        self.status_lbl.setVisible(False)
        if self._engine_err:
            self._log(f"[!] Template non caricato: {self._engine_err}")
            self._log("    Cattura il golden prompt (templates/README_capture.md) e riavvia.")

    def _enqueue(self, label, worker, on_done=None, on_progress=None):
        """Accoda un worker di generazione nella coda seriale (no sovrapposizioni). Vedi ui/job_queue.py.
        Sostituisce il vecchio pattern worker.done.connect(...)+worker.start() in tutti i punti.
        #97: scrive automaticamente un sidecar (output/sidecars/) coi params del worker ad ogni output."""
        def _wrapped(res):
            try:
                self._auto_sidecar(worker, res, label)
            except Exception:                                # noqa: BLE001
                pass
            if on_done:
                on_done(res)
        return self.jobs.enqueue(label, worker, on_done=_wrapped, on_progress=on_progress)

    def _auto_sidecar(self, worker, res, label):
        """#97: sidecar universale — dai params del worker + output del risultato. Best-effort (mai rompe)."""
        if not isinstance(res, dict) or res.get("status") not in (None, "success"):
            return
        outs = res.get("output") or []
        out = outs[0] if isinstance(outs, (list, tuple)) and outs else (outs if isinstance(outs, str) else None)
        params = getattr(worker, "params", None)
        if not out and isinstance(params, dict):
            out = params.get("output")
        if not out or not isinstance(out, str) or not os.path.exists(out):
            return
        meta = {"label": str(label), "output_kind": Path(out).suffix.lstrip(".").lower()}
        if isinstance(params, dict):                         # solo valori JSON-safe (niente tensori/oggetti)
            for k, v in params.items():
                if k == "output":
                    continue
                if isinstance(v, (str, int, float, bool)) or v is None:
                    meta[k] = v
                elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool, dict)) for x in v):
                    meta[k] = list(v)
        from core.metadata_saver import MetadataSaver
        MetadataSaver.save_sidecar(out, meta)

    def _build_menu(self):
        m = self.menuBar().addMenu(tr("menu.tools"))
        act_wiz = m.addAction(tr("menu.wizard")); act_wiz.triggered.connect(self.open_wizard)
        act_set = m.addAction(tr("menu.settings")); act_set.triggered.connect(self.open_settings)
        act_nav = m.addAction(tr("menu.navigator_gallery")); act_nav.triggered.connect(self._open_navigator)
        act_ana = m.addAction(tr("menu.analyzer")); act_ana.triggered.connect(self._open_analyzer)
        act_wfl = m.addAction(tr("menu.workflow")); act_wfl.triggered.connect(self._open_workflow)
        act_film = m.addAction(tr("menu.film_studio", "🎬 AI Film Studio")); act_film.triggered.connect(self._open_film_studio)
        # ── Interfaccia: versione Classica / Pro dei moduli (persistita) ─────────────
        from PySide6.QtGui import QActionGroup
        m.addSeparator()
        ui_menu = m.addMenu(tr("menu.ui_variant", "🎨 Interfaccia"))
        _variants = (("classic", tr("menu.variant.classic", "Classica")),
                     ("pro", tr("menu.variant.pro", "Pro ✨")))
        mod_menu = ui_menu.addMenu(tr("menu.ui_variant.models", "Tab Modelli"))
        self._grp_models = QActionGroup(self); self._grp_models.setExclusive(True)
        fs_menu = ui_menu.addMenu(tr("menu.ui_variant.filmstudio", "AI Film Studio"))
        self._grp_fs = QActionGroup(self); self._grp_fs.setExclusive(True)
        for grp, submenu, pref in ((self._grp_models, mod_menu, "ui_variant.models"),
                                   (self._grp_fs, fs_menu, "ui_variant.filmstudio")):
            for key, label in _variants:
                a = submenu.addAction(label); a.setCheckable(True); a.setData(key)
                a.setChecked(str(user_prefs.get(pref, "classic")) == key)
                grp.addAction(a)
        self._grp_models.triggered.connect(lambda a: self._set_models_variant(a.data()))
        self._grp_fs.triggered.connect(lambda a: self._set_filmstudio_variant(a.data()))
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

    def _toggle_mem_monitor(self):
        """Apri/chiudi la finestra non-modale del monitor memoria (RAM/VRAM/GPU)."""
        if self._mem_monitor is None:
            from ui.memory_monitor import MemoryMonitor
            self._mem_monitor = MemoryMonitor(self)
            self._mem_monitor.finished.connect(lambda *_: self._mem_btn.setChecked(False))
        if self._mem_monitor.isVisible():
            self._mem_monitor.hide()
            self._mem_btn.setChecked(False)
        else:
            self._mem_monitor.show()
            self._mem_monitor.raise_()
            self._mem_monitor.activateWindow()
            self._mem_btn.setChecked(True)

    def open_settings(self):
        if not self.engine:
            QMessageBox.warning(self, tr("mw.template_missing_title"), self._engine_err or ""); return
        dlg = SettingsDialog(self.engine, self.driver, self._overrides, self)
        if dlg.exec():
            self._overrides = dlg.overrides()
            self._log(f"Override modelli aggiornati: {len(self._overrides)} nodi")

    def open_wizard(self):
        wiz = SetupWizard(self)
        if not wiz.exec():
            return
        r = wiz.result_values()
        cat = r.get("category", "video")
        prompt = r.get("prompt_positive", "")
        # Stato condiviso (sempre): modalità contenuto + audio.
        self.state.update({"nsfw_mode": r.get("nsfw_mode"), "audio_mode": r.get("audio_mode")})
        nsfw_text = "NSFW" if r.get("nsfw_mode") == "nsfw" else "SFW"

        if cat == "video":
            self.frame1_tab.prompt.setPlainText(prompt)
            self.video_tab.length.setValue(r.get("length_s", 8))
            try:
                self.video_tab.nsfw.setCurrentText(nsfw_text)
            except Exception:
                pass
            self.tabs.setCurrentWidget(self.video_tab)
        elif cat == "image":
            self.frame1_tab.prompt.setPlainText(prompt)
            self.tabs.setCurrentWidget(self.frame1_tab)
        elif cat == "3d":
            # immagine→3D: nessun prompt, basta portare l'utente alla scheda giusta.
            self.tabs.setCurrentWidget(self.trellis_tab)
        elif cat == "music":
            # il prompt è la descrizione da cui generare il testo della canzone.
            if prompt and hasattr(self.music_tab, "lyr_prompt"):
                self.music_tab.lyr_prompt.setText(prompt)
            self.tabs.setCurrentWidget(self.music_tab)
        elif cat == "voice":
            if prompt and hasattr(self.audio_tab, "tts_text"):
                self.audio_tab.tts_text.setPlainText(prompt)
            self.tabs.setCurrentWidget(self.audio_tab)
        elif cat == "film":
            # AI Film Studio è una finestra separata: la apro col prompt come spunto.
            self._open_film_studio()
            win = getattr(self, "_film_win", None)
            if prompt and win is not None and hasattr(win, "seed_prompt"):
                try:
                    win.seed_prompt(prompt)
                except Exception:
                    pass
        self._log(f"Wizard: obiettivo '{r.get('goal_label', r.get('goal'))}' applicato.")

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

    def refresh_model_inventory(self):
        """FASE D weights-unification: RIFÀ l'inventario modelli dei dropdown (scan filesystem /
        object_info) al prossimo tick di _refresh_status. Prima `_inventory_set` non veniva MAI
        resettato: dopo un download/import i menu restavano vecchi fino al riavvio dell'app."""
        self._inventory_set = False
        self._log(tr("mw.inventory_refresh", "Modelli aggiornati: ricarico gli elenchi dei menu…"))

    def _cancel_gen(self):
        try:
            self.driver.interrupt()
        except Exception:
            pass
        self._current_pid = self._inpaint_pid = self._seg_pid = None
        self._log("Generazione annullata dall'utente.")

    def _on_free_vram(self):
        """Libera la VRAM di TUTTI i server GPU nativi (immagini :8189, Krea2 :8191, video :8190,
        krea_rt :8192) via mem_coordinator.free_all() = mode "destroy" ESPLICITO: anche i parcheggi
        RAM vanno all'OS («Libera VRAM» = libera davvero). Prima si POSTava `b"{}"` a 3 porte fisse
        → mode default "demote" (parcheggio tenuto, e col vecchio bug il video restava in VRAM) e
        :8191/:8192 non erano mai toccate. Comfy resta come extra."""
        _names = {8189: "immagini", 8191: "krea2", 8190: "video", 8192: "krea_rt"}
        try:
            from core import mem_coordinator
            freed = [_names.get(p, str(p)) for p in mem_coordinator.free_all(timeout=25)]
        except Exception:                                    # noqa: BLE001
            freed = []
        try:
            if self.driver.free_memory():                    # ComfyUI, se davvero attivo (extra)
                freed.append("comfy")
        except Exception:                                    # noqa: BLE001
            pass
        try:                                                 # mostra la VRAM libera dopo (feedback reale)
            from core.gpu_mem import free_vram_gb              # NVML/nvidia-smi: NIENTE contesto CUDA nell'app
            _free_gb = free_vram_gb()                          # (torch.cuda.* creerebbe un contesto → crash 3D)
        except Exception:                                    # noqa: BLE001
            _free_gb = 0
        if freed:
            self._log(f"✅ VRAM liberata: {', '.join(freed)} — ora liberi {_free_gb:.1f} GB.")
        else:
            self._log("[!] Nessun server GPU attivo da liberare (:8189/:8190/:8191/:8192 offline).")

    def _open_film_studio(self):
        """Apre la finestra AI Film Studio (Text-to-Film, backend nativo). Variante Classica/Pro."""
        from ui.film_studio.film_studio_window import FilmStudioWindow
        if getattr(self, "_film_win", None) is None:
            cls = FilmStudioWindow
            if str(user_prefs.get("ui_variant.filmstudio", "classic")) == "pro":
                try:
                    from ui.film_studio.film_studio_window_pro import FilmStudioWindowPro
                    cls = FilmStudioWindowPro
                except Exception:                            # noqa: BLE001 — Pro assente → Classica
                    cls = FilmStudioWindow
            self._film_win = cls(self)
        self._film_win.show(); self._film_win.raise_(); self._film_win.activateWindow()

    # ---- varianti UI Classica / Pro (switch da menu Interfaccia) ----
    def _make_models_tab(self):
        """Istanzia la tab Modelli nella variante scelta (Pro → fallback Classica se assente)."""
        if str(user_prefs.get("ui_variant.models", "classic")) == "pro":
            try:
                from ui.tabs.import_models_tab_pro import ImportModelsTabPro
                return ImportModelsTabPro()
            except Exception:                                # noqa: BLE001
                pass
        return ImportModelsTab()

    def _set_models_variant(self, key: str):
        if key not in ("classic", "pro") or str(user_prefs.get("ui_variant.models", "classic")) == key:
            return
        user_prefs.set("ui_variant.models", key)
        new = self._make_models_tab()                        # hot-swap dentro Avanzate
        try:
            new.catalog_updated.connect(self._on_catalog_updated)
        except Exception:                                    # noqa: BLE001
            pass
        self.advanced_tab.swap_models(new)
        self.models_tab = new

    def _set_filmstudio_variant(self, key: str):
        if key not in ("classic", "pro") or str(user_prefs.get("ui_variant.filmstudio", "classic")) == key:
            return
        user_prefs.set("ui_variant.filmstudio", key)
        win = getattr(self, "_film_win", None)               # riaprirà con la nuova classe
        if win is not None:
            try:
                win.close()
            except Exception:                                # noqa: BLE001
                pass
            self._film_win = None

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

    def _stop_all_workers(self):
        """#91: alla chiusura, TERMINA i worker di generazione e i loro sottoprocessi video, altrimenti
        restano ORFANI a calcolare e a tenere la VRAM (bug osservato: GPU al 100% a finestra chiusa)."""
        # worker video con subprocess (ltx2_infer / comfy-native)
        seen = set()
        cands = [getattr(self, "_ltx2_worker", None), getattr(self, "_native_worker", None)]
        try:                                                # worker attualmente in coda
            run = getattr(getattr(self, "jobs", None), "_running", None)
            if run is not None:
                cands.append(getattr(run, "worker", None))
        except Exception:                                   # noqa: BLE001
            pass
        for w in cands:
            if w is None or id(w) in seen:
                continue
            seen.add(id(w))
            try:
                if hasattr(w, "stop"):
                    w.stop()                                # termina il subprocess (albero)
            except Exception:                               # noqa: BLE001
                pass
            try:
                if hasattr(w, "isRunning") and w.isRunning():
                    w.wait(3000)
            except Exception:                               # noqa: BLE001
                pass

    def closeEvent(self, event):
        """Salva lo stato dei controlli GUI + TERMINA i worker/sottoprocessi (anti-orfano VRAM, #91)."""
        try:
            self._stop_all_workers()
        except Exception:                                   # noqa: BLE001
            pass
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

    def _ensure_native_ready(self, timeout: int = 180) -> bool:
        """Il backend nativo DEVE esserci quando premi Genera: se è caduto lo riavvia invece di
        sbatterti in faccia un errore. Mostra l'attesa (dialog annullabile) e ritorna True quando è
        pronto. L'errore compare solo se il riavvio fallisce davvero — allora c'è un problema vero.

        Il riavvio in sé sta in `native_driver.ensure_up()` (senza Qt, così vale anche per Film Studio,
        API e servizi): qui lo eseguiamo in un thread per non congelare la finestra.
        """
        import concurrent.futures
        import time as _time
        from core import native_driver
        if native_driver.is_up():
            return True
        if getattr(self, "_native_restarting", False):
            return False                                    # riavvio già in corso: niente doppioni
        self._native_restarting = True
        dlg = QProgressDialog(tr("mw.native_restart_msg"), tr("mw.native_restart_cancel"), 0, 0, self)
        dlg.setWindowTitle(tr("mw.native_restart_title"))
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.show()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(native_driver.ensure_up, timeout=float(timeout), log=self._log)
                while not fut.done():
                    QApplication.processEvents()
                    if dlg.wasCanceled():
                        self._log("[Nativo] riavvio annullato — il server continua a partire in background.")
                        return False                        # il thread finisce da solo: niente kill a metà avvio
                    _time.sleep(0.05)
                ok = bool(fut.result())
        finally:
            self._native_restarting = False
            dlg.close()
        if not ok:
            QMessageBox.warning(self, tr("mw.native_off_title", "Backend nativo offline"),
                                tr("mw.native_restart_failed_msg").format(timeout=timeout))
        return ok

    def _ernie_pe_ok(self, cfg: dict) -> bool:
        """«Migliora prompt» su ERNIE-Image usa il PE ufficiale (pe/model.safetensors, ~7.7 GB). Se manca: avviso con
        «Apri tab Pesi modelli» / «Continua senza» / «Annulla»; NESSUN download da qui. True = si può procedere."""
        try:
            from core import model_paths as _mp
            from inference_server.backends import ernie_pe
            repo = str(_mp.p("ernie-image", r"E:\AI\Models\diffusers\Ernie\ERNIE-Image"))
            if ernie_pe.available(repo):
                return True
        except Exception:                                    # noqa: BLE001 — il server verifica comunque
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(tr("ernie.pe.missing.title", "Enhancer ERNIE mancante"))
        box.setText(tr("ernie.pe.missing.body",
                       "«Migliora prompt» su ERNIE-Image usa l'enhancer ufficiale (pe/model.safetensors, ~7.7 GB) e non "
                       "è su disco. Non scarico nulla io: scaricalo dal tab «Pesi modelli» (voce «ernie-image-pe»); se hai "
                       "già ernie-image-prompt-enhancer.safetensors, va spostato in {dir}.").format(dir=os.path.join(repo, "pe")))
        b_w = box.addButton(tr("vedit.ckpt.btn_open_weights_tab", "Apri tab Pesi modelli"), QMessageBox.ActionRole)
        b_c = box.addButton(tr("ernie.pe.continue", "Continua senza migliorare"), QMessageBox.AcceptRole)
        box.addButton(tr("common.cancel", "Annulla"), QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is b_w:
            self.tabs.setCurrentWidget(self.weights_tab)
            return False
        if box.clickedButton() is b_c:
            cfg["enhance"] = False
            return True
        return False

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
        # CODA OCCUPATA = job in corso: i worker via _enqueue NON settano i *_pid, quindi prima
        # si finiva comunque su native_driver.is_up() → HTTP (1.5s) sul thread GUI ogni 3s mentre
        # il server era occupato dal denoise = micro-freeze ricorrenti (visto live 22/07).
        _busy = False
        try:
            _busy = self.jobs.is_busy()
        except Exception:                                   # noqa: BLE001
            pass
        if (_busy or self._current_pid or self._inpaint_pid or self._seg_pid) and _all_native:
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
        """GGUF video RIABILITATI (22-23/07, bc0c390): residenti quantizzati + dequant GPU sono
        ~alla pari dell'fp8 (3.7-3.8 s/step misurati sul Q4_K_M). La lista arriva dal FILESYSTEM
        (nell'inventario offline i .gguf non passano da object_info): senza, restava solo l'item
        di fallback Q4 e il Q6_K spariva dal menu."""
        import glob
        root = r"E:\AI\Models\diffusion_models"
        out = []
        for p in glob.glob(os.path.join(root, "**", "*.gguf"), recursive=True):
            if "ltx" in os.path.basename(p).lower():
                out.append(os.path.relpath(p, root))
        return sorted(set(out))

    def _log(self, msg: str):
        # THREAD-SAFE: appendPlainText tocca un QWidget → deve girare SOLO sul thread GUI. I job a subprocess
        # (3D TRELLIS, musica, world) chiamano _log dal loro thread PUMP (non-GUI): toccare il widget da lì è
        # comportamento indefinito → access violation NATIVA in Qt6Core/Qt6Gui (crash visto nel log eventi). Se
        # non siamo sul thread GUI, marshalliamo la chiamata con QMetaObject.invokeMethod (coda → thread GUI).
        from PySide6.QtCore import QThread, QMetaObject, Qt as _Qt, Q_ARG
        if QThread.currentThread() is self.thread():
            self.log.appendPlainText(msg)
        else:
            QMetaObject.invokeMethod(self.log, "appendPlainText", _Qt.QueuedConnection, Q_ARG(str, str(msg)))

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
        """True se la feature è disponibile nell'edizione; altrimenti mostra l'upsell
        (tier minimo + prezzo + link Patreon) e ritorna False. Fallback: messaggio generico."""
        if features.is_enabled(fid):
            return True
        self._log(f"[gating] funzione '{fid}' non disponibile in '{features.edition()}'.")
        from core import upsell as _up
        if _up.info(fid):                                    # sbloccabile → vetrina con CTA Patreon
            from ui.widgets.upsell import UpsellDialog
            info = features.manifest().get("features", {}).get(fid, {})
            UpsellDialog(fid, info.get("i18n_key", ""), self).exec()
            return False
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, tr("gate.locked.title"),
                                tr("gate.locked.msg").format(edition=features.edition()))
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
        if mm.blocked(self, self.video_tab.model):           # modello scelto sparito dal menu: mai sceglierne un altro
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
        # 'In ascolto (loop)': ritratto che ascolta da UNA foto (Frame 1), girabile in loop senza stacco.
        # Non è un V2V: usa la foto come i2v + keyframe auto → prosegue nel percorso NATIVO normale.
        if _mode == "listen":
            if not self.frame1_tab.image_path:
                QMessageBox.warning(self, tr("mw.listen_title", "Ritratto in ascolto"),
                                    tr("mw.listen_need_image", "Serve una foto nel Frame 1 (il volto che ascolterà)."))
                return
            vals["listen_loop"] = {"motion": v2v.get("listen_motion", 0.55),
                                   "mouth_closed": v2v.get("listen_mouth_closed", True)}
            vals["listen_seconds"] = float(v2v.get("listen_seconds", 8.0))
            vals["use_image_force"] = True                   # la foto è la sorgente (anche se il toggle i2v è off)
            self._log(f"[Ascolto] loop {vals['listen_seconds']:.0f}s da foto Frame 1 "
                      f"(motion {vals['listen_loop']['motion']:.2f}, bocca chiusa={vals['listen_loop']['mouth_closed']}).")
            _mode = "normal"                                 # salta il blocco V2V → percorso nativo standard
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

        # ── RIFERIMENTI MULTIPLI: identità/stile/movimento/voce, SENZA posizione temporale ──
        # (distinti dai keyframe dello storyboard: qui i file si citano nel prompt col loro token)
        ref_panel = getattr(self.video_tab, "reference_panel", None)
        refs = ref_panel.get_references() if ref_panel else []
        if refs:
            errs = ref_panel.validation_errors()
            if errs:
                QMessageBox.warning(self, tr("vid.ref.group", "Riferimenti multipli"),
                                    tr("mw.ref_invalid", "Riferimenti non validi:") + "\n• " + "\n• ".join(errs))
                return
            _perr = ref_panel.prompt_errors(vals.get("prompt_positive", ""))
            if _perr:                                        # @image_N che non esiste: mai generare in silenzio
                QMessageBox.warning(self, tr("vid.ref.group", "Riferimenti multipli"),
                                    tr("mw.ref_invalid", "Riferimenti non validi:") + "\n• " + "\n• ".join(_perr))
                return
            vals["references"] = refs
            _unc = ref_panel.uncited(vals.get("prompt_positive", ""))
            if _unc:                                         # passati comunque al modello, ma lo si dice
                self._log(tr("mw.ref_untokened",
                             "[Riferimenti] {toks} non compaiono nel prompt: il modello potrebbe ignorarli.")
                          .format(toks=", ".join(_unc)))
            # i token @image_N (e i vecchi <Picture N>) diventano la sintassi del modello (<Picture k>, Image k…)
            vals["prompt_positive"] = ref_panel.translate(vals.get("prompt_positive", ""))
            self._log(tr("mw.ref_log", "[Riferimenti] {n} file di riferimento.").format(n=len(refs)))

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
        # #95 (V4): clone SENZA testo = riferimento-voce (LTXVReferenceAudio) → NON supportato dal worker video
        # nativo (leggeva solo audio_file) → era un no-op silenzioso. Avvisa e procede senza voce.
        if audio.get("audio_mode") == "clone" and not (audio.get("tts_text") or "").strip():
            self._log("[warn] clone senza testo (riferimento voce) non è supportato dal motore video nativo: "
                      "inserisci un testo nel tab Audio (verrà sintetizzato) o carica un audio. Procedo senza voce.")
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
        if self._inpaint_pid:
            QMessageBox.information(self, tr("mw.wait_title"), tr("mw.image_gen_in_progress_msg")); return
        if mm.blocked(self, self.frame1_tab.inp_model):
            return
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
        qwen_edit = bool(cfg.get("qwen21_dit") or cfg.get("integrate_engine") == "qwen_image21")
        if cfg.get("enhance") and (ptxt or "").strip() and not qwen_edit:   # Qwen: arricchitore ufficiale
            try:
                from core import prompt_enhance as pe
                if pe.available():
                    ptxt = pe.enhance(ptxt, kind="image", nsfw=bool(self.frame1_tab.get_values().get("nsfw"))) or ptxt
            except Exception:                                    # noqa: BLE001
                pass
        retention = cfg.get("strength", 0.75)
        guidance_ui = cfg.get("guidance")                    # CFG esplicita dalla GUI (se impostata)
        denoise = 1.0 if mode == "outpaint" else self._denoise_from_fidelity(retention)
        # guard: serve un'area da rigenerare, altrimenti il modello non fa nulla
        if mode == "outpaint" and not self.frame1_tab.canvas.has_outpaint_area():
            QMessageBox.information(self, tr("mw.nothing_to_extend_title"),
                                    tr("mw.nothing_to_extend_msg")); return
        if mode == "inpaint" and not self.frame1_tab.canvas.has_mask():
            # SENZA maschera → NON rifiutare più: modifica dell'INTERA foto secondo la descrizione
            # (Kontext edit-instruct, guidance bassa ~2.5). Il prompt dice COSA cambiare in tutta la foto.
            if not (ptxt or "").strip():
                QMessageBox.information(self, tr("mw.no_mask_title"),
                    tr("mw.no_mask_edit_msg", "Nessuna maschera disegnata: scrivi nel prompt COSA "
                       "modificare e verrà applicato all'INTERA foto (edit globale Kontext).")); return
            try:
                exp = self.frame1_tab.canvas.export_inpaint(OUTPUT_DIR, mode)
                self._inpaint_bbox = exp.get("bbox")
                self._inpaint_before = exp.get("image")
                seed = cfg["seed"] if cfg.get("seed") is not None else random.randint(0, 2**31)
                steps = cfg.get("steps") or 28
                negative = cfg.get("negative") or ""
                self._log(tr("mw.no_mask_kontext_log",
                             "Nessuna maschera → modifica dell'INTERA foto secondo la descrizione (Kontext)"))
                q21 = cfg.get("qwen21_dit") or (cfg.get("integrate_qwen21_dit")
                                                 if cfg.get("integrate_engine") == "qwen_image21" else "")
                if q21:                                           # Qwen-Image 2.1 (modello inpaint o motore Integra)
                    self._start_native_qwen21_edit(q21, ptxt, refs=[exp["image"]], negative=negative, seed=seed,
                                                   loras=self._inp_loras(cfg), enhance=cfg.get("enhance", False))
                    return
                if cfg.get("integrate_engine") == "flux3":        # motore FLUX.3 Klein selezionato in Integra
                    self._start_native_flux3_edit(cfg.get("integrate_flux3_dit"), ptxt, refs=[exp["image"]], seed=seed,
                                                  boxes=cfg.get("boxes"), pup=cfg.get("flux3_pup", True),
                                                  mode="integrazione")
                    return
                if cfg.get("integrate_engine") == "krea2":        # motore Krea 2 Edit selezionato in Integra
                    self._start_native_krea2_edit([exp["image"]], ptxt, negative,
                                                  cfg.get("grounding", 768),
                                                  cfg.get("integrate_variant", "turbo"), seed,
                                                  label=tr("queue.job.krea2edit", "Krea 2 Edit"))
                    return
                # exp SENZA 'mask' → _start_native_kontext prende il ramo kontext puro (no Fill)
                self._start_native_kontext({"image": exp["image"]}, None, ptxt, negative, steps, seed, 2.5)
            except Exception as e:                               # noqa: BLE001
                QMessageBox.warning(self, tr("mw.error_title", "Errore"), str(e))
            return
        try:
            exp = self.frame1_tab.canvas.export_inpaint(OUTPUT_DIR, mode)
            self._inpaint_bbox = exp.get("bbox")    # per rimettere l'inpaint al suo posto
            self._inpaint_before = exp.get("image") # per confronto prima/dopo nel dialog
            seed = cfg["seed"] if cfg.get("seed") is not None else random.randint(0, 2**31)
            steps = cfg.get("steps") or 20
            negative = cfg.get("negative") or self.frame1_tab.get_values().get("prompt_negative", "")
            # ── BACKEND NATIVO: FLUX.1 Fill ('inpaint') E FLUX.2 Klein ('flux2_inpaint',
            #    Flux2KleinInpaintPipeline — #14/#78 completati). Solo i checkpoint restano su ComfyUI.
            if cfg.get("qwen21_dit"):                        # Qwen-Image 2.1: ritocco della sola area
                # stessa scala «forza» dei FLUX (mantenimento → denoise); l'outpaint rigenera tutto il bordo
                # Qwen: CFG VERA (true_cfg_scale) — 30 (default Fill) la brucia (chiazze verdi, 16/10 01:15). Stessa
                # protezione che _start_native_inpaint applica a Klein/Hyper. Tetto = DECISIONE (qui 6.0, come Klein).
                _qg = guidance_ui
                if _qg is not None and float(_qg) > 6.0:
                    self._log(f"[inpaint] Qwen-Image 2.1: guidance {float(_qg):.0f} clampata a 6.0 (CFG vera, "
                              f"oltre brucia l'immagine)")
                    _qg = 6.0
                self._start_native_qwen21_edit(cfg["qwen21_dit"], ptxt, init_image=exp["image"], mask=exp["mask"],
                                               strength=denoise, negative=negative, seed=seed, steps=steps,
                                               guidance=_qg, loras=self._inp_loras(cfg),
                                               feather=cfg.get("feather", 0), mode=mode,
                                               enhance=cfg.get("enhance", False))
                return
            from core import backend
            _unet = cfg.get("unet") or ""
            # Anche i CHECKPOINT (Checkpoints\*.safetensors, es. flux1-schnell-fp8) vanno sul backend nativo: image_flux li
            # risolve sotto Checkpoints. Prima restavano sul grafo ComfyUI → «urlopen error WinError 10061» (punto 15).
            _ckpt = cfg.get("checkpoint") or ""
            if backend.is_native("image") and (_unet or _ckpt):
                self._start_native_inpaint(exp, _unet or _ckpt, ptxt, negative, steps, seed, retention, mode,
                                           guidance_override=guidance_ui)
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
        if mm.blocked(self, self.frame1_tab.integrate_model):
            return
        # Foto di base: dal nuovo campo della tab, oppure — se vuoto — quella già nel canvas.
        if not self._resolve_photo(self.frame1_tab.get_inpaint_config().get("integrate_photo", "")):
            return
        if self._inpaint_pid:
            QMessageBox.information(self, tr("mw.wait_title"), tr("mw.gen_in_progress_msg")); return
        self._layers_from_compose_list()          # le foto della lista laterale ENTRANO nel canvas
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
        if cfg.get("enhance") and ptxt.strip() and engine != "qwen_image21":   # Qwen: arricchitore ufficiale
            try:
                from core import prompt_enhance as pe
                if pe.available():
                    ptxt = pe.enhance(ptxt, kind="image", nsfw=bool(self.frame1_tab.get_values().get("nsfw"))) or ptxt
            except Exception:                                # noqa: BLE001
                pass
        try:
            exp = self.frame1_tab.canvas.export_integrate(OUTPUT_DIR)
            if not exp.get("has_region"):
                QMessageBox.information(self, tr("mw.nothing_to_integrate_title"),
                                        tr("mw.nothing_to_integrate_msg")); return
            self._inpaint_bbox = None                   # Integra lavora sull'intera scena (niente crop)
            self._inpaint_before = exp.get("image")     # confronto prima/dopo (composito di partenza)
            img_name = self._upload_local(exp["image"])
            # Campi dedicati alla sezione Integra (separati da quelli di inpaint/outpaint).
            int_seed_cfg = cfg.get("integrate_seed")
            seed = int_seed_cfg if int_seed_cfg is not None else random.randint(0, 2**31)
            steps = cfg.get("integrate_steps") or 20
            negative = cfg.get("integrate_negative") or ""
            aux = cfg.get("integrate_puzzle_lora") or None    # puzzle_lora opzionale (solo Kontext)
            guidance = cfg.get("integrate_guidance", 2.5)
            if engine == "qwen_image21":                      # Qwen: armonizza solo l'area dei layer incollati
                self._start_native_qwen21_edit(
                    cfg.get("integrate_qwen21_dit"), ptxt or self._QWEN21_INTEGRATE_PROMPT,
                    init_image=exp["image"], mask=exp.get("mask") or "", strength=1.0 if exp.get("mask") else None,
                    refs=[] if exp.get("mask") else [exp["image"]], negative=negative, seed=seed,
                    mode="integrazione", label=tr("queue.job.qwen21edit", "Qwen-Image 2.1 Edit"),
                    enhance=bool(cfg.get("enhance") and ptxt.strip()))
                return
            if engine == "flux3":                             # FLUX.3: modifica per istruzione del composito (+ riquadri)
                self._start_native_flux3_edit(
                    cfg.get("integrate_flux3_dit"), ptxt or self._QWEN21_INTEGRATE_PROMPT,
                    init_image=exp["image"] if exp.get("mask") else "", mask=exp.get("mask") or "",
                    refs=[] if exp.get("mask") else [exp["image"]], seed=seed, boxes=cfg.get("boxes"),
                    pup=cfg.get("flux3_pup", True), mode="integrazione")
                return
            if engine == "krea2":                             # motore Krea 2 Edit → edit del composito (single)
                self._start_native_krea2_edit([exp["image"]], ptxt, negative,
                                              cfg.get("grounding", 768),
                                              cfg.get("integrate_variant", "turbo"), seed,
                                              label=tr("queue.job.krea2edit", "Krea 2 Edit"))
                return
            # ── BACKEND NATIVO (Kontext FLUX.1): niente ComfyUI. «Integrazione MagicQuill» (puzzle_lora) = mode foreground.
            from core import backend
            if backend.is_native("image") and engine == "kontext" and aux:
                self._start_native_mq_foreground(exp, model, ptxt, steps, seed, guidance)
                return
            if backend.is_native("image") and engine in ("kontext", "flux2") and not aux:
                # PuLID DEDICATO all'Image to Image (container proprio nel pannello, separato da
                # quello di «Genera immagine» che agisce solo sulla generazione).
                _pl = ({"pulid_face": cfg.get("integrate_pulid_face"),
                        "pulid_strength": cfg.get("integrate_pulid_strength", 0.8),
                        "pulid_start_step": cfg.get("integrate_pulid_start_step", 0),
                        "pulid_true_cfg": cfg.get("integrate_pulid_true_cfg", 1.0)}
                       if cfg.get("integrate_pulid_face") and engine == "kontext" else None)
                self._start_native_kontext(exp, model, ptxt, negative, steps, seed, guidance, pulid=_pl)
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

    def _layers_from_compose_list(self) -> int:
        """Porta sul canvas le foto elencate nella lista laterale, con la loro maschera.

        Erano il buco di «Integra»: le foto della lista non erano layer del canvas, e Integra guarda
        solo i layer → chiedeva «aggiungi una o più immagini» anche quando la lista era piena. Le
        aggiungiamo solo se il canvas non ha già dei layer (altrimenti raddoppieremmo gli oggetti che
        hai posizionato a mano) e ci fermiamo a quelle non ancora presenti. Ritorna quante ne ha messe.
        """
        canvas = self.frame1_tab.canvas
        if canvas.has_layers():
            return 0
        cc = self.frame1_tab.get_compose_config()
        imgs = [p for p in cc.get("images", []) if p and os.path.exists(p)]
        masks = cc.get("masks", {})
        added = 0
        for i, p in enumerate(imgs):
            if canvas.add_layer(p, masks.get(p), index=i):
                added += 1
        if added:
            self._log(f"[integra] {added} foto dalla lista portate sul canvas (con le loro maschere) — "
                      f"spostale/ridimensionale, poi rilancia se vuoi cambiarne la posa.")
        return added

    def _resolve_photo(self, chosen: str) -> bool:
        """Risolve la FOTO DI BASE di «Integra»: campo della tab → canvas.

        Se il campo è VUOTO la base è la foto già nel canvas; se l'utente ne ha scelta una col campo
        file, la CARICHIAMO nel canvas — così maschere, layer e anteprima lavorano tutti sulla stessa
        immagine. Ritorna False (con avviso) solo se non c'è né l'una né l'altra. Inpaint/outpaint non
        passano di qui: la loro base è per definizione la foto su cui hai disegnato la maschera.
        """
        chosen = (chosen or "").strip()
        if chosen:
            if not os.path.exists(chosen):
                QMessageBox.warning(self, tr("mw.no_image_title"),
                                    tr("mw.photo_not_found", "Foto non trovata:\n{p}").format(p=chosen))
                return False
            if os.path.abspath(chosen) != os.path.abspath(self.frame1_tab.image_path or ""):
                self.frame1_tab.canvas.load_image(chosen)     # diventa la base del canvas
                self._log(f"[foto] uso la foto scelta: {os.path.basename(chosen)}")
            return True
        if not self.frame1_tab.image_path:
            QMessageBox.warning(self, tr("mw.no_image_title"), tr("mw.load_photo_canvas_msg"))
            return False
        return True                                           # campo vuoto → foto del canvas

    def generate_compose(self):
        """Componi 2+ foto SEPARATE in un'unica immagine senza incollarle (niente occlusione):
        multi-reference (catena ReferenceLatent) o stitch (affianca). FLUX.1 (Kontext) o FLUX.2.
        Maschere opzionali per-foto (prep_masked) per indicare cosa usare."""
        from core import backend
        if not backend.is_native("image") and not self.driver.is_up():
            QMessageBox.warning(self, tr("mw.comfy_offline_title"), tr("mw.comfy_offline_generate_msg")); return
        if self._inpaint_pid:
            QMessageBox.information(self, tr("mw.wait_title"), tr("mw.gen_in_progress_msg")); return
        if mm.blocked(self, self.frame1_tab.integrate_model):
            return
        cfg = self.frame1_tab.get_compose_config()
        imgs = cfg.get("images") or []
        # LISTA VUOTA ⇒ usa il CANVAS: base = foto del canvas, elementi = i suoi LAYER, ognuno con la
        # SUA maschera e con posizione/scala/rotazione date dall'utente. È l'unico modo di rispettare
        # davvero il layout: nessun modello FLUX ha un conditioning spaziale per i riferimenti — la
        # geometria va "cotta" nei pixel e nella maschera (paste-then-refine). export_integrate() fa
        # esattamente questo: composito con le trasformazioni + maschere per-layer + anello di edit.
        if not imgs:
            if not self.frame1_tab.image_path:
                QMessageBox.information(self, tr("f1.compose.group"),
                                        tr("mw.compose_need_canvas_msg",
                                           "Aggiungi foto alla lista, oppure carica una foto nel canvas "
                                           "e incollaci sopra i layer da comporre."))
                return
            if not self.frame1_tab.canvas.has_layers():
                QMessageBox.information(self, tr("f1.compose.group"),
                                        tr("mw.compose_need_layers_msg",
                                           "Il canvas non ha layer da comporre. Importa le foto sul canvas "
                                           "(posizionale/ruotale, disegna la loro maschera) oppure usa la lista."))
                return
            self._log("[componi] lista vuota → uso il canvas: base + layer con maschere e trasformazioni")
            return self.generate_integrate()                  # stessa meccanica: paste-then-refine
        if len(imgs) < 2:
            QMessageBox.information(self, tr("f1.compose.group"), tr("mw.compose_need_two_msg")); return
        engine = cfg.get("engine", "kontext")
        model = cfg.get("model")
        if engine == "qwen_image21":                      # Qwen: fino a 10 riferimenti VERI («image 1…10»)
            self._inpaint_before = None
            self._start_native_qwen21_edit(cfg.get("qwen21_dit"), cfg.get("prompt") or "",
                                           refs=imgs, mode="composizione",
                                           label=tr("queue.job.qwen21edit", "Qwen-Image 2.1 Edit"),
                                           enhance=self.frame1_tab.get_inpaint_config().get("enhance", False))
            return
        if engine == "flux3":                             # FLUX.3: fino a 10 riferimenti VERI («the first image»…)
            self._inpaint_before = None
            self._start_native_flux3_edit(cfg.get("flux3_dit"), cfg.get("prompt") or "", refs=imgs,
                                          boxes=None, pup=cfg.get("flux3_pup", True), mode="composizione")
            return
        if engine == "krea2":                             # Krea 2 Edit → multi-reference VERO (scena+soggetto,
            # 2 latenti separati ai frame RoPE 1/2 — non lo stitch di FLUX.1). Il modello ne gestisce max 2.
            self._start_native_krea2_edit(imgs[:2], cfg.get("prompt", ""), "",
                                          cfg.get("grounding", 768), cfg.get("variant", "turbo"),
                                          random.randint(0, 2**31),
                                          label=tr("queue.job.krea2edit", "Krea 2 Edit"))
            return
        if engine == "flux2" and not model:
            QMessageBox.warning(self, tr("mw.flux2_model_missing_title"),
                                tr("mw.flux2_model_missing_integrate_msg")); return
        if not self._ensure_flux2_encoder(model):
            return
        masks = cfg.get("masks") or {}
        from core import backend as _bk
        if engine == "flux2" and _bk.is_native("image"):      # FLUX.2 Klein: multi-riferimento VERO (task 9)
            srcs = [self.kontext.prep_masked(f, masks[f], OUTPUT_DIR) if f in masks else f for f in imgs]
            self._start_native_klein_multiref(srcs, cfg.get("prompt") or "", model)
            return
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
                    "", 20, random.randint(0, 2**31),
                    self.frame1_tab.get_inpaint_config().get("integrate_guidance", 2.5))   # guidance di Integra (Kontext)
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
        _ft = self.frame1_tab
        if mm.blocked(self, _ft.color_model if _ft.color_engine.currentIndex() in (QWEN21_ENGINE_IDX, FLUX3_ENGINE_IDX)
                      else _ft.integrate_model):
            return
        cc = self.frame1_tab.get_color_config()
        if not cc["target"]:
            QMessageBox.warning(self, tr("mw.color_missing_title"),
                                tr("mw.color_missing_msg")); return
        engine = cc["engine"]; model = None
        if engine == "qwen_image21":                      # Qwen-Image 2.1 → recolor per istruzione
            what = (cc.get("subject") or "").strip() or "the main subject"
            self._inpaint_before = self.frame1_tab.image_path
            self._start_native_qwen21_edit(
                cc.get("qwen21_dit"), f"Recolor {what} in image 1 to {cc['target']}. Keep the original texture, "
                f"material, lighting and shadows; change only the color.",
                refs=[self.frame1_tab.image_path], mode="cambio colore",
                label=tr("queue.job.qwen21edit", "Qwen-Image 2.1 Edit"))
            return
        if engine == "flux3":                             # FLUX.3 → recolor per istruzione
            what = (cc.get("subject") or "").strip() or "the main subject"
            self._inpaint_before = self.frame1_tab.image_path
            self._start_native_flux3_edit(
                cc.get("flux3_dit"), f"Recolor {what} in the first image to {cc['target']}. Keep the original texture, "
                f"material, lighting and shadows; change only the color.",
                refs=[self.frame1_tab.image_path], pup=False, mode="cambio colore")
            return
        if engine == "krea2":                             # Krea 2 Edit → recolor per istruzione (single)
            what = (cc.get("subject") or "").strip() or "the main subject"
            rprompt = (f"Recolor {what} to {cc['target']}. Keep the original texture, material, "
                       f"lighting and shadows; change only the color.")
            self._inpaint_before = self.frame1_tab.image_path
            self._start_native_krea2_edit([self.frame1_tab.image_path], rprompt, "",
                                          cc.get("grounding", 768), "turbo", random.randint(0, 2**31),
                                          label=tr("queue.job.krea2edit", "Krea 2 Edit"))
            return
        if engine == "flux2":
            from core.gen_pipeline import is_flux2           # (non è in scope a livello modulo)
            model = self.frame1_tab.get_inpaint_config().get("integrate_model")
            if not model:
                QMessageBox.warning(self, tr("mw.flux2_model_missing_title"),
                                    tr("mw.flux2_model_missing_inpaint_tab_msg")); return
            if not is_flux2(model):                          # #95 I3: il combo Integra può mostrare un modello Kontext (FLUX.1)
                QMessageBox.warning(self, tr("mw.flux2_model_missing_title"),
                    tr("mw.recolor_flux2_wrong_arch_msg",
                       "Il modello scelto non è FLUX.2. Imposta il motore della sezione Integra su FLUX.2 e "
                       "seleziona un modello FLUX.2, oppure usa il motore Kontext per il recolor.")); return
        # ── BACKEND NATIVO (recolor Kontext FLUX.1): niente ComfyUI. FLUX.2 → ComfyUI.
        from core import backend
        if backend.is_native("image") and engine == "kontext":
            what = (cc.get("subject") or "").strip() or "the indicated subject"
            rprompt = (f"Recolor {what} to {cc['target']}. Keep the original texture, material, "
                       f"lighting and shadows; change only the color.")
            self._inpaint_before = self.frame1_tab.image_path
            self._start_native_kontext({"image": self.frame1_tab.image_path}, None, rprompt, "",
                                       cc["steps"], random.randint(0, 2**31), cc.get("guidance", 2.5))
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

    def generate_exr_export(self):
        """Export frame EXR float32 (VFX) del video nel tab Definizione — cap.exr_export, no GPU (ffmpeg)."""
        from core import features
        if not features.is_enabled("cap.exr_export"):
            self._show_upsell("cap.exr_export") if hasattr(self, "_show_upsell") else QMessageBox.information(
                self, tr("upsell.title", "Funzione non inclusa"),
                tr("upsell.exr.msg", "L'export HDR/EXR non è incluso in questa edizione."))
            return
        video = getattr(self.video_def_tab, "_video", None)
        if not video:
            QMessageBox.information(self, tr("def.no_video.title", "Nessun video"),
                                    tr("def.no_video.pick_msg", "Carica prima un video nel tab Definizione."))
            return
        linear = self.video_def_tab.exr_linear.isChecked()

        class _ExrWorker(QThread):
            done = Signal(dict)
            progress = Signal(str)

            def run(self):
                try:
                    from core.exr_export import video_to_exr
                    out = video_to_exr(video, linear=linear, progress=self.progress.emit)
                    self.done.emit({"status": "success", "output": [out]})
                except Exception as e:                       # noqa: BLE001
                    self.done.emit({"status": "error", "error": str(e)})

        self._exr_worker = _ExrWorker()
        self._log(tr("def.exr.start", "Export EXR avviato ({m})…").format(
            m="lineare" if linear else "gamma"))
        self._enqueue(tr("queue.job.exr", "Export EXR (VFX)"), self._exr_worker,
                      lambda res: self._log(
                          tr("def.exr.done", "✅ Frame EXR in: {o}").format(o=res.get("output", ["?"])[0])
                          if res.get("status") == "success" else
                          tr("def.exr.fail", "❌ Export EXR: {e}").format(e=res.get("error", "")[:300])),
                      on_progress=self._log)

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
            ok, msg = ensure_model(cfg["model"], log=self._log,
                                   confirm=lambda n, mb: QMessageBox.question(
                                       self, tr("weights.missing.title", "Pesi mancanti"),
                                       tr("weights.missing.offer_one",
                                          "Il modello {name} (~{mb}MB) non è presente. Scaricarlo ora?")
                                       .format(name=n, mb=mb)) == QMessageBox.Yes)
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
        if mm.blocked(self, self.frame1_tab.gen_model):      # modello scelto sparito dal menu: mai sceglierne un altro
            return
        cfg = self.frame1_tab.get_gen_config()
        if not cfg["prompt"]:
            QMessageBox.warning(self, tr("mw.empty_prompt_title"), tr("mw.empty_image_prompt_msg")); return
        if cfg.get("arch") in ("qwen_image21", "krea2", "flux3"):      # riferimenti nominati: @image_N inesistente = errore
            _gr = self.frame1_tab.gen_refs
            _perr = _gr.prompt_errors(cfg.get("prompt_raw") or "")
            if _perr:
                QMessageBox.warning(self, tr("f1.gen.refs.group", "Immagini di riferimento"), "\n".join(_perr))
                return
            _unc = _gr.uncited(cfg.get("prompt_raw") or "")
            if _unc and cfg.get("ref_images"):                # passate comunque, come fa il modello ufficiale
                self._log(tr("f1.gen.refs.uncited", "[Riferimenti] {toks} non sono citati nel prompt: vengono "
                             "comunque passati al modello.").format(toks=", ".join(_unc)))
        # Krea 2 è uno SNAPSHOT diffusers (nessun file-modello): usa krea2_variant, non un checkpoint → salta il
        # controllo "modello obbligatorio" (il worker risolve turbo/raw dal catalogo).
        if cfg.get("arch") not in ("krea2", "qwen_image21", "flux3") and (not cfg["model"] or cfg["model"].startswith("(")):
            QMessageBox.warning(self, tr("mw.no_model_title"), tr("mw.no_model_arch_msg").format(arch=cfg['arch'])); return
        if cfg.get("arch") == "qwen_image21":                 # DiT = voce di catalogo (+ deps)
            if not self._feature_gate("cap.image_qwen21"):
                return
            if cfg.get("qwen21_pe") and not self._feature_gate("cap.qwen21_prompt_enhancer"):
                return
            if "control_image" in cfg:
                if not self._feature_gate("cap.qwen21_controlnet"):
                    return
                if not os.path.isfile(cfg["control_image"] or ""):
                    QMessageBox.warning(self, tr("mw.no_image_title"), tr("mw.qwen21_no_control",
                        "ControlNet: carica una foto nel canvas o scegline una nel campo «Foto guida».")); return
                if cfg.get("ref_images"):
                    QMessageBox.information(self, tr("f1.gen.qwen21.cn.group", "Controllo struttura (ControlNet)"),
                        tr("mw.qwen21_control_refs", "Il ControlNet non si combina con i Riferimenti: disattiva "
                           "l'uno o l'altro.")); return
            if "init_image" in cfg and not os.path.isfile(cfg["init_image"] or ""):
                QMessageBox.warning(self, tr("mw.no_image_title"),
                                    tr("mw.qwen21_no_init", "img2img: carica una foto nel canvas o scegline una "
                                       "nel campo «Immagine iniziale».")); return
            if not self._ensure_qwen21_weights(cfg.get("qwen21_dit") or "qwen21-dit-fp8", cfg.get("qwen21_te"),
                                               cfg.get("qwen21_turbo"), pe=cfg.get("qwen21_pe", ""),
                                               control=bool(cfg.get("control_image"))):
                return
        if cfg.get("arch") == "flux3" and not self._flux3_preflight(cfg.get("flux3_dit"), cfg.get("flux3_pup", True),
                                                                     cfg.get("boxes"), len(cfg.get("ref_images") or []),
                                                                     cfg.get("flux3_te", "")):
            return
        # FLUX.2: se manca l'encoder del modello, offri di scaricarlo prima di costruire.
        if cfg.get("arch") == "flux" and not self._ensure_flux2_encoder(cfg["model"]):
            return
        if cfg.get("arch") == "ernie" and cfg.get("enhance") and not self._ernie_pe_ok(cfg):
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
        if cfg.get("arch") == "krea2" and cfg.get("ref_images"):   # Krea 2 con riferimenti = Krea 2 Edit (task 9)
            self._start_native_krea2_edit(cfg["ref_images"], cfg["prompt"], negative, 768,
                                          cfg.get("krea2_variant") or "turbo", seed,
                                          label=tr("queue.job.krea2edit", "Krea 2 Edit"),
                                          dit=cfg.get("krea2_dit"), te=cfg.get("krea2_te"))
            return
        if cfg.get("ctl_mode"):                               # ControlNet FLUX.1 / Z-Image o Redux (task 9)
            self._start_native_control(cfg, v, loras, seed, negative); return
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

    # ── Preview live per-step (backend nativo) — task #88 ────────────────────────────────────────
    def _start_native_preview(self, title: str, source: str = "image"):
        """Apre SUBITO la finestra di anteprima LIVE (progresso per-step + immagine del passo appena
        completato), prima del dialog finale. source='image' → polling HTTP /preview (:8189);
        'video' → polling del file core.preview_file (i worker video sono sottoprocessi). Best-effort."""
        try:
            from ui.progress_dialog import GenProgressDialog
            from PySide6.QtCore import QTimer
            if getattr(self, "_native_prev_dlg", None) is None:
                self._native_prev_dlg = GenProgressDialog(self)
                self._native_prev_timer = QTimer(self)
                self._native_prev_timer.timeout.connect(self._poll_native_preview)
            self._native_prev_seq = -1
            self._native_prev_src = source
            if source == "video":                          # azzera lo stato del job PRECEDENTE (era il
                try:                                       # "15/30" che appariva subito, residuo vecchio)
                    from core import preview_file
                    preview_file.reset()
                except Exception:                          # noqa: BLE001
                    pass
            self._native_prev_dlg.start({}, title)         # prompt vuoto: non serve la pesatura nodi comfy
            self._native_prev_dlg.cur_bar.setRange(0, 0)   # "busy" finché non arriva il primo step
            self._native_prev_dlg.on_native_message(tr("progress.step_load_model", "Caricamento componenti…"))
            self._native_prev_timer.start(250)
            # forza il RENDER IMMEDIATO della finestra (altrimenti l'event loop la disegna in ritardo,
            # dando l'impressione che "appaia alla fine")
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
        except Exception as e:                              # noqa: BLE001
            self._log(f"[preview] non disponibile: {e}")

    def _poll_native_preview(self):
        try:
            src = getattr(self, "_native_prev_src", "image")
            if src == "video":
                from core import preview_file
                m = preview_file.read()
                if not m:
                    return
                # fase (caricamento/generazione) finché non parte il sampling; poi step X/Y
                if m.get("step", 0) and m.get("total"):
                    self._native_prev_dlg.on_native_step(m["step"], m["total"])
                elif m.get("msg"):
                    self._native_prev_dlg.on_native_message(m["msg"])
                if m.get("png_seq", 0) != getattr(self, "_native_prev_seq", -1):
                    png = preview_file.read_png()
                    if png:
                        self._native_prev_seq = m["png_seq"]
                        self._native_prev_dlg.on_preview(png)
                return
            from core import native_driver
            p = native_driver.preview()
            if not p:
                return
            # fase (caricamento…) finché step==0; poi barra credibile step/total
            if p.get("step", 0) and p.get("total"):
                self._native_prev_dlg.on_native_step(p["step"], p["total"])
            elif p.get("msg"):
                self._native_prev_dlg.on_native_message(p["msg"])
            # ridisegna l'anteprima solo quando c'è un PNG nuovo
            if p.get("png") and p.get("png_seq", 0) != getattr(self, "_native_prev_seq", -1):
                self._native_prev_seq = p["png_seq"]
                self._native_prev_dlg.on_preview(p["png"])
        except Exception:                                  # noqa: BLE001
            pass

    def _stop_native_preview(self):
        try:
            if getattr(self, "_native_prev_timer", None) is not None:
                self._native_prev_timer.stop()
            if getattr(self, "_native_prev_dlg", None) is not None:
                self._native_prev_dlg.finish()
        except Exception:                                  # noqa: BLE001
            pass

    def _start_native_image(self, cfg, v, loras, seed, negative):
        """Genera un'immagine via backend nativo (InferenceServer) invece di ComfyUI."""
        from core import native_driver
        if not self._ensure_native_ready():   # caduto? lo riavvia da sé
            return
        params = {"model": cfg["model"], "arch": cfg["arch"], "prompt": cfg["prompt"],
                  "negative": negative, "width": v["width"], "height": v["height"],
                  "steps": cfg["steps"],
                  # 0.0 è un valore VERO (Krea 2 Turbo è guidance-distillato): `or 3.5` lo trasformava in 3.5 → reticolo
                  "guidance": cfg["cfg"] if cfg.get("cfg") is not None else 3.5,
                  "seed": seed, "loras": loras,
                  "krea2_variant": cfg.get("krea2_variant"),   # Krea 2: turbo/raw (snapshot, no file-modello)
                  "krea2_dit": cfg.get("krea2_dit"),           # Krea 2: transformer single-file (es. Noct V4)
                  "krea2_te": cfg.get("krea2_te"),             # Krea 2: text encoder single-file (uncensored)
                  # Qwen-Image 2.1: DiT, turbo, encoder, RGBA, formato 2K, riferimenti (≤10), img2img
                  **{k: cfg[k] for k in ("qwen21_dit", "qwen21_turbo", "qwen21_te", "transparent",
                                         "flux3_dit", "flux3_pup", "flux3_te", "boxes",
                                         "aspect_2k", "ref_images", "init_image", "strength",
                                         "qwen21_pe", "qwen21_pe_think", "control_image", "control_type",
                                         "control_scale", "control_preprocess") if k in cfg},
                  # #95 I1: componenti scelti dall'utente (erano scartati → il backend usava i default)
                  "sampler": cfg.get("sampler"), "scheduler": cfg.get("scheduler"),
                  "encoder": cfg.get("encoder"), "clip_type": cfg.get("clip_type"), "vae": cfg.get("vae"),
                  "enhance": cfg.get("enhance"), "nsfw": cfg.get("nsfw"),
                  "pulid_face": cfg.get("pulid_face"), "pulid_strength": cfg.get("pulid_strength", 0.8),
                  "pulid_start_step": cfg.get("pulid_start_step", 0), "pulid_true_cfg": cfg.get("pulid_true_cfg", 1.0)}
        # log dei settaggi EFFETTIVI (visibilità su cosa è auto e cosa è scelto dall'utente)
        self._log(f"Generazione immagine (NATIVO): modello={os.path.basename(str(cfg['model']))} · "
                  f"arch={cfg['arch']} · {v['width']}×{v['height']} · steps={cfg['steps']} · "
                  f"cfg={cfg.get('cfg') or 3.5} · sampler={cfg.get('sampler') or 'auto'} · "
                  f"encoder={os.path.basename(str(cfg.get('encoder') or 'auto'))} · seed={seed}")
        self._native_worker = _NativeImageWorker(params)
        self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"))
        self._enqueue(tr("queue.job.image", "Immagine"), self._native_worker, self._on_native_image_done)

    # ── ControlNet FLUX.1 / Z-Image, Redux, ControlNet Inpainting (Alimama): backend image_control (task 9) ──────────
    _CTL_CAPS = {"flux_cn": "cap.image_controlnet", "zimage_cn": "cap.image_controlnet",
                 "flux_redux": "cap.image_redux", "flux_cn_inpaint": "cap.image_cn_inpaint"}
    _CTL_LABELS = {"flux_cn": "ControlNet FLUX.1", "zimage_cn": "ControlNet Z-Image",
                   "flux_redux": "Redux", "flux_cn_inpaint": "ControlNet Inpainting (Alimama)"}

    def _missing_weights_notice(self, ids: list, feature: str) -> bool:
        """True se i pesi ci sono tutti. Altrimenti avviso con nome e GB di ciò che manca e «Apri tab Pesi modelli»:
        NESSUN download da qui (regola del task: l'utente scarica dal tab Pesi)."""
        try:
            from core import ensure_weights as ew
            miss = ew.missing(list(ids))
        except Exception:                                    # noqa: BLE001 — il backend verifica comunque
            return True
        if not miss:
            return True
        gb = round(sum(float(e.get("approx_gb", 0) or 0) for e in miss), 1)
        names = "\n".join(f"• {e.get('label') or e['id']}  ({e['id']}, ~{e.get('approx_gb', '?')} GB)" for e in miss)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(tr("ctl.missing.title", "Pesi mancanti"))
        box.setText(tr("ctl.missing.body",
                       "{feature} ha bisogno di pesi che non sono su disco (~{gb} GB in tutto):\n\n{names}\n\n"
                       "Non scarico nulla io: scaricali dal tab «Pesi modelli».").format(feature=feature, gb=gb, names=names))
        b_w = box.addButton(tr("vedit.ckpt.btn_open_weights_tab", "Apri tab Pesi modelli"), QMessageBox.ActionRole)
        box.addButton(tr("common.cancel", "Annulla"), QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is b_w:
            self.tabs.setCurrentWidget(self.weights_tab)
        return False

    def _control_weight_ids(self, mode: str, kind: str = "") -> list:
        ids = {"flux_cn": ["flux-cn-union-pro2"], "flux_cn_inpaint": ["flux-cn-inpaint-alimama"],
               "zimage_cn": ["zimage-cn-union21"], "flux_redux": ["flux-redux", "siglip-so400m-384"]}[mode]
        if kind == "depth":
            ids = ids + ["depth-anything-v2"]
        elif kind == "pose":
            ids = ids + ["openpose-body"]
        return ids

    def _run_control_job(self, params: dict, label: str, mode_log: str):
        """Accoda un job `image_control` (stessa anteprima live e stesso risultato del resto della tab Immagini)."""
        self._inpaint_mode = mode_log
        self._native_worker = _NativeImageWorker(params, module="image_control")
        self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"))
        self._enqueue(label, self._native_worker, self._on_native_inpaint_done)

    def _start_native_control(self, cfg, v, loras, seed, negative):
        """«Genera» con ControlNet (FLUX.1 / Z-Image) o Redux. `cfg["ctl_mode"]` dice quale."""
        mode = cfg["ctl_mode"]
        if not self._feature_gate(self._CTL_CAPS[mode]):
            return
        label = self._CTL_LABELS[mode]
        if mode == "flux_redux":
            refs = [p for p in (cfg.get("ref_images") or []) if p and os.path.isfile(p)]
            if not refs:
                QMessageBox.warning(self, tr("mw.no_image_title"), tr("ctl.redux.no_image",
                    "Redux: scegli l'immagine di riferimento o carica una foto nel canvas.")); return
        else:
            if not os.path.isfile(cfg.get("control_image") or ""):
                QMessageBox.warning(self, tr("mw.no_image_title"), tr("ctl.no_guide",
                    "ControlNet: carica una foto nel canvas o scegline una nel campo «Foto guida».")); return
        if not self._missing_weights_notice(self._control_weight_ids(mode, cfg.get("control_preprocess", "")), label):
            return
        if not self._ensure_native_ready():
            return
        params = {"mode": mode, "arch": cfg["arch"], "model": cfg["model"], "prompt": cfg["prompt"],
                  "negative": negative, "width": v["width"], "height": v["height"], "steps": cfg["steps"],
                  "guidance": cfg["cfg"] if cfg.get("cfg") is not None else 3.5, "seed": seed, "loras": loras}
        for k in ("control_image", "control_preprocess", "control_scale", "control_end", "ref_images", "redux_strength"):
            if k in cfg:
                params[k] = cfg[k]
        if mode == "zimage_cn":
            params["guidance"] = cfg["cfg"] if cfg.get("cfg") is not None else 0.0
        self._log(f"{label} (NATIVO): modello={os.path.basename(str(cfg['model']))} · {v['width']}×{v['height']} · "
                  f"passi={cfg['steps']} · forza={cfg.get('control_scale', cfg.get('redux_strength', '-'))} · seed={seed}")
        self._run_control_job(params, tr("queue.job.image_control", "Immagine ({what})").format(what=label), "generazione immagine")

    def _start_native_cn_inpaint(self, exp, unet, prompt, negative, steps, seed, mode, icfg):
        """Inpaint/outpaint con ControlNet Inpainting (Alimama) su un FLUX.1 base: il ControlNet legge l'immagine
        mascherata + la maschera e guida il modello base; fuori dalla maschera restano i pixel originali."""
        if not self._feature_gate("cap.image_cn_inpaint"):
            return
        if not self._missing_weights_notice(self._control_weight_ids("flux_cn_inpaint"), self._CTL_LABELS["flux_cn_inpaint"]):
            return
        if not self._ensure_native_ready():
            return
        model = unet if os.path.isabs(unet) else os.path.join(r"E:\AI\Models\diffusion_models", unet)
        params = {"mode": "flux_cn_inpaint", "arch": "flux", "model": model, "prompt": prompt, "negative": negative,
                  "image_path": exp["image"], "mask_path": exp["mask"], "steps": steps,
                  "guidance": float(icfg.get("cn_inpaint_guidance", 3.5)), "seed": seed,
                  "control_scale": float(icfg.get("cn_inpaint_scale", 0.9)),
                  "feather": int(icfg.get("feather", 0)), "outpaint": mode == "outpaint"}
        self._log(f"{mode} con ControlNet Inpainting (Alimama): modello={os.path.basename(model)} · passi={steps} · "
                  f"forza={params['control_scale']:.2f} · seed={seed}")
        self._run_control_job(params, tr("queue.job.inpaint", "Inpaint/Outpaint"), mode)

    def _on_native_image_done(self, res: dict):
        self._stop_native_preview()
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

    def _auto_pick_ltx_gguf(self, free_gb: float) -> str | None:
        """Sceglie un transformer LTX GGUF che ENTRA nella VRAM libera (la Q più alta possibile → miglior
        qualità). None se non ce ne sono. Budget = free - ~8GB (VAE/gemma/attivazioni)."""
        import glob
        root = r"E:\AI\Models\diffusion_models"
        cands = sorted({p for pat in ("*ltx*.gguf", "*LTX*.gguf")
                        for p in glob.glob(os.path.join(root, "**", pat), recursive=True)})
        if not cands:
            return None
        budget = max(0.0, float(free_gb) - 8.0) * 1024 ** 3
        best, best_sz = None, -1
        for c in cands:
            try:
                # F6: footprint dal DB canonico (per i GGUF ≈ dim file, ma se misurato è più preciso)
                sz = 0
                try:
                    from core import vram_estimate as _ve
                    _mb, _src = _ve.estimate_mb(c, working_mb=0)
                    sz = int(_mb * 2**20) if _mb else 0
                except Exception:                              # noqa: BLE001
                    pass
                sz = sz or os.path.getsize(c)
            except OSError:
                continue
            if sz <= budget and sz > best_sz:                  # il più grande che entra = miglior qualità
                best, best_sz = c, sz
        return best or min(cands, key=lambda c: os.path.getsize(c))   # se nessuno entra, il più piccolo

    def _confirm_model_swap(self, original: str, proposed: str, reason: str) -> str:
        """Chiede all'utente se ACCETTARE la riassegnazione automatica del modello (per VRAM/fallback) o
        MANTENERE la scelta originale. Ritorna il modello da usare. Preferenza salvabile (chiedi/accetta/
        mantieni) in user_prefs 'model_swap_policy'. Nel guscio condiviso → vale per Classica e Pro."""
        from pathlib import Path as _P
        try:
            from core import user_prefs as _up
            pol = str(_up.get("model_swap_policy", "ask"))
        except Exception:                                    # noqa: BLE001
            _up, pol = None, "ask"
        if pol == "accept":
            return proposed
        if pol == "keep":
            return original
        from PySide6.QtWidgets import QMessageBox, QCheckBox
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(tr("mw.model_swap.title", "Cambio modello automatico"))
        box.setText(tr("mw.model_swap.msg",
            "Il modello scelto «{orig}» {reason}.\nProposta automatica: «{prop}».\n\n"
            "Usare il modello proposto oppure MANTENERE il tuo (verrà caricato con offload dinamico)?").format(
            orig=(_P(original).name if original else "(automatico)"), prop=_P(proposed).name, reason=reason))
        b_use = box.addButton(tr("mw.model_swap.use", "Usa il proposto"), QMessageBox.AcceptRole)
        box.addButton(tr("mw.model_swap.keep", "Mantieni il mio"), QMessageBox.RejectRole)
        cb = QCheckBox(tr("mw.model_swap.remember", "Ricorda la scelta (ripristinabile in Impostazioni)"))
        box.setCheckBox(cb)
        box.exec()
        chose_proposed = box.clickedButton() is b_use
        if cb.isChecked() and _up is not None:
            _up.set("model_swap_policy", "accept" if chose_proposed else "keep")
        self._log(f"[modello] riassegnazione {'accettata' if chose_proposed else 'rifiutata (mantengo il mio)'}")
        return proposed if chose_proposed else original

    def _generate_video_native(self, vals: dict, audio: dict, loras: list, use_image: bool):
        """Generazione video NATIVA (LTX-2.3 GGUF, zero ComfyUI) — ramo attivo solo se
        backend_config video=native. Costruisce i params dal tab Video/Avanzate e lancia il worker
        ltx2 (auto-GGUF + chunking per i lunghi). Audio (native/user) → a2v (canto/lip-sync)."""
        import time as _t
        fps = int(self.video_tab.fps.value())
        length_s = float(self.video_tab.length.value()) if hasattr(self.video_tab, "length") else 5.0
        if vals.get("listen_loop"):
            # Render SEMPRE il solo loop base ~8s (frame che stanno in VRAM allo stage-2; 16/24s diretti
            # = troppi frame → spilling). La durata scelta si ottiene ripetendo il loop base (seamless).
            vals["listen_loop"]["total_seconds"] = float(vals.get("listen_seconds", 8.0))
            length_s = 8.0
        nf = max(9, int(round(length_s * fps)) // 8 * 8 + 1)
        out = str(OUTPUT_DIR / f"ltx_native_{int(_t.time())}.mp4")
        # LTX-2 (two-stage) richiede width/height multipli di 64 → arrotonda (min 64) per non far
        # saltare i chunk ("Resolution ... not divisible by 64" → 0 output). Vale per entrambi i motori.
        _r64 = lambda x: max(64, int(round(int(x) / 64.0)) * 64)
        params = {
            "prompt": vals.get("prompt_positive", ""),
            "negative": vals.get("prompt_negative", "worst quality, blurry"),
            "width": _r64(self.video_tab.width.value()),
            "height": _r64(self.video_tab.height.value()),
            "num_frames": nf, "fps": fps,
            "steps": int(self.video_tab.sampler_seed.steps.value()),
            "guidance": float(self.video_tab.sampler_seed.cfg.value()),
            "seed": int(vals.get("seed", self.video_tab.sampler_seed.seed.value())),   # #95 V1: usa il seed randomizzato da on_generate
            "offload": "", "loras": loras, "output": out,   # "" = AUTO: il worker decide (none se ci sta,
            #                                                block-swap per-blocco se serve — come ComfyUI)
            "teacache_thresh": float(vals.get("teacache_thresh", 0.0) or 0.0),   # #104 cache velocità (0=off)
            "decode_lowmem": bool(vals.get("decode_lowmem")),   # tile VAE piccoli (film lunghi/4K, LTX)
            "native_audio": bool(vals.get("native_audio")),     # TIENI l'audio LTX-2 (parlato/suoni dal prompt)
        }
        # MODELLO scelto nel tab Video → al worker. Senza questo il worker usava SEMPRE il checkpoint
        # di default → "funziona solo un modello" (qualunque scelta nel dropdown era ignorata).
        if vals.get("model_kind") == "gguf" and vals.get("model_gguf"):
            params["gguf_transformer"] = vals["model_gguf"]   # base checkpoint resta default per vae/gemma
        elif vals.get("model") and vals.get("model") not in ("", "auto"):
            params["model"] = vals["model"]
        # RELEASE LTX dal CHECKPOINT (2.3 / 2.5): le due release hanno venv incompatibili
        # (2.5 richiede transformers>=5.8 per Gemma 4). Nessuno la impostava sul percorso GUI —
        # build_params() è il percorso API — quindi la 2.5 era IRRAGGIUNGIBILE dalla GUI: ogni
        # generazione finiva sul venv 2.3 e tutto il pack 2.5 scaricabile restava inerte.
        # Nome non riconosciuto → None → resta il default (2.3): nessuna sostituzione automatica.
        if (params.get("arch") or "ltx").lower() == "ltx":
            from core import ltx_release as _lrel
            _det = _lrel.detect_release(params.get("gguf_transformer") or params.get("model") or "")
            if _det:
                params["ltx_release"] = _det
            if _det == "2.5" and vals.get("ltx_te"):          # TE alternativo scelto (es. Gemma 4 uncensored)
                params["ltx_te"] = vals["ltx_te"]
        if (use_image or vals.get("use_image_force")) and self.frame1_tab.image_path:
            params["image_path"] = self.frame1_tab.image_path
            params["ref_strength"] = float(vals.get("ref_strength", 1.0))
            params["i2v_method"] = vals.get("i2v_method", "addguide")   # addguide=guida piena / inplace=soft-ref
        if vals.get("listen_loop"):                          # 'In ascolto (loop)': il worker costruisce i keyframe auto
            params["listen_loop"] = vals["listen_loop"]
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
        # Chunking SOLO per video davvero lunghi: LTX-2.3 regge ~20s in singolo shot sui 32GB. Spezzare a 5s
        # era troppo conservativo (introduceva chunk-boundary che potevano fallire → durata dimezzata muta).
        # Ora: ≤20s = singolo shot (durata piena garantita); >20s = chunk da ~20s memory-bounded.
        _single_max = int(20 * fps) // 8 * 8 + 1             # ~481 frame @24fps
        if nf > _single_max:
            # LV1/LV3: anche il video NATIVO normale ora rispetta gli slider «Video lungo» (Avanzate):
            # durata-segmento, overlap crossfade, anti-drift (re-anchor) e resume — prima inerti in nativo.
            try:
                _lv = self.advanced_tab.long_video_config()
                _seg = int(float(_lv.get("seg_seconds", 20) or 20) * fps)
                _chunk = min(_seg, _single_max) if _seg >= 9 else _single_max
                params["chunk_frames"] = max(9, _chunk // 8 * 8 + 1)
                params["overlap_frames"] = int(_lv.get("overlap", 0) or 0)
                params["ref_anchor_strength"] = float(_lv.get("ref_anchor_strength", 0.0) or 0.0)
                params["anchor_every"] = int(_lv.get("anchor_every", 0) or 0)
                params["resume"] = bool(_lv.get("resume", False))
            except Exception:                                # noqa: BLE001
                params["chunk_frames"] = _single_max
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
                params["conditioning_attention"] = float(cnet.get("conditioning_attention", 1.0) or 1.0)
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
                # ControlNet + storyboard NON convivono: col video-guida la pipeline diventa IC-LoRA,
                # che accetta UNA sola immagine → i keyframe sparivano in silenzio. Meglio dirlo prima
                # di occupare la GPU per una generazione che non fa quello che l'utente ha chiesto.
                if params.get("control_video"):
                    QMessageBox.warning(
                        self, tr("mw.sb_title", "Storyboard"),
                        tr("mw.sb_controlnet_conflict",
                           "Storyboard e ControlNet non si possono usare insieme: con il video-guida "
                           "il modello accetta una sola immagine e i {n} keyframe verrebbero ignorati.\n"
                           "Disattiva ControlNet, oppure lo storyboard.").format(n=len(kfs)))
                    return
                params["keyframes"] = kfs
                params.pop("image_path", None)        # i keyframe sostituiscono il singolo frame iniziale
                self._log(f"[nativo] storyboard: {len(kfs)} keyframe a tempo → video multi-scena")
        # RIFERIMENTI MULTIPLI → al worker così come sono (path+tipo+token+descrizione): NON sono
        # keyframe, quindi non diventano `keyframes` e non toccano `image_path`.
        if vals.get("references"):
            params["references"] = vals["references"]
            _n = {k: sum(1 for r in vals["references"] if r["kind"] == k)
                  for k in ("image", "video", "audio")}
            self._log(f"[nativo] riferimenti: {_n['image']} immagini, {_n['video']} video, "
                      f"{_n['audio']} audio")
            # Su LTX i riferimenti arrivano dalla LoRA Multiple-Subject-Reference, che è un FILE
            # DIVERSO per release (2.3 e 2.5 sono addestrate ciascuna sulla propria architettura):
            # la scelta la fa la release, mai un fallback all'altra.
            # NB: `params["arch"]` viene impostato più sotto (blocchi per architettura), qui non
            # c'è ancora → si guarda la scelta del tab, altrimenti con H3 si cercherebbe la LoRA LTX.
            if (vals.get("video_arch") or "ltx").lower() == "ltx":
                from core import ltx_release as _lr
                _rel = params.get("ltx_release") or _lr.DEFAULT_RELEASE
                _msr = _lr.get(_rel).msr_lora
                if not _msr or not os.path.exists(_msr):
                    _mid = f"ltx{_rel.replace('.', '')}-lora-multisubject"
                    from core import ensure_weights as _ew
                    if not _ew.offer_download_gui(self, [_mid], f"LTX {_rel} — riferimenti multipli"):
                        self._log(tr("weights.missing.title", "Pesi mancanti") + f" → {_mid}")
                        return
                    _msr = _lr.get(_rel).msr_lora
                params["msr_lora"] = _msr
                self._log(f"[msr] LoRA riferimenti LTX {_rel}: {os.path.basename(_msr)}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # ── 2ª ARCHITETTURA: LingBot-Video ──────────────────────────────────────────────
        # Se l'utente ha scelto LingBot nel tab Video, instrada al worker dedicato e salta tutta la
        # logica motore/GGUF LTX (LingBot ha il suo venv + gestione memoria). Prompt JSON + rewriter
        # sono gestiti a monte (Fase 5) se abilitati.
        if vals.get("video_arch") == "lingbot":
            params["arch"] = "lingbot"
            params["lingbot_variant"] = vals.get("lingbot_variant", "dense")
            if vals.get("run_refiner"):
                params["run_refiner"] = True
            params.pop("gguf_transformer", None)                # LingBot non usa i GGUF LTX
            # #16/#17 Caption JSON OBBLIGATORIA: LingBot col prompt in prosa genera GARBAGE — serve lo schema
            # ricco. Sempre: rewriter proprietario (Qwen3.6-27B+LoRA) se scaricato, altrimenti composer
            # locale Gemma (prompt_struct.to_lingbot_json). Il risultato va in params["prompt_json"].
            try:
                from core import prompt_rewrite_lingbot as _rw
                _dur = max(1.0, int(params.get("num_frames", 121)) / float(params.get("fps", 24)))
                self._log("[lingbot] prompt → caption JSON ricca"
                          + ("" if _rw.available() else " (composer locale Gemma)"))
                _pj, _neg = _rw.rewrite(params.get("prompt", ""),
                                        first_frame=params.get("image_path", "") or "",
                                        nsfw=(vals.get("nsfw_mode") == "nsfw"))
                if _pj:
                    params["prompt_json"] = _pj
                if _neg:
                    params["negative"] = _neg
            except Exception as _rex:                           # noqa: BLE001
                self._log(f"[lingbot] composer caption fallito ({_rex}) → prosa incapsulata dal worker")
            self._log(f"[lingbot] LingBot-Video ({params['lingbot_variant']}, "
                      f"refiner={'on' if params.get('run_refiner') else 'off'}): "
                      f"{params['width']}x{params['height']} {nf} frame")
            self._ltx2_worker = _NativeLtx2Worker(params)
            self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"), source="video")
            self._enqueue(tr("queue.job.video", "Video"), self._ltx2_worker, self._on_multitalk_done,
                          on_progress=lambda s: self._log("[lingbot] " + s))
            return
        # ── MiniMax H3: omni-modale (video + audio stereo nella STESSA passata) ─────────
        if vals.get("video_arch") == "minimax_h3":
            from core import minimax_h3_spec as _S
            params["arch"] = "minimax_h3"
            params.pop("gguf_transformer", None)
            for _k in ("h3_profile", "h3_dit", "h3_te", "h3_turbo", "h3_canvas", "h3_dialogue", "h3_soundscape",
                       "h3_music", "h3_lightvae"):
                if _k in vals:
                    params[_k] = vals[_k]
            params.setdefault("h3_profile", "nvfp4")
            # Veda: predittore Ref2VA → solo coi riferimenti (senza, la casella non ha effetto)
            if vals.get("h3_veda") and params.get("references"):
                if not self._feature_gate("cap.video_h3_veda"):
                    return
                params["h3_veda"] = True
            params["duration_s"] = float(vals.get("length_s") or 5.0)
            params["num_frames"] = _S.snap_frames(params["duration_s"])
            params["fps"] = _S.FPS
            params["generate_audio"] = bool(vals.get("native_audio", True))
            params.pop("negative", None)                  # guidance distillata: niente negative prompt
            _ab = self.video_tab.attn_backend.currentData() if hasattr(self.video_tab, "attn_backend") else ""
            params["attention_backend"] = {"flash": "flash", "sage": "sage"}.get(_ab or "", "")
            # CHECKPOINT DIVERSO per modalità (scelta di modalità, non di modello): i riferimenti richiedono
            # il Ref2VA, il resto il FL2VA. Profilo/TE/turbo li ha scelti l'utente.
            try:
                _ids = list(dict.fromkeys(_S.weight_ids(params).values()))
            except ValueError as _e:
                QMessageBox.warning(self, "MiniMax H3", str(_e))
                return
            try:                                          # consenso primo-uso col motore unico
                from core import ensure_weights as _ew
                if not _ew.offer_download_gui(self, _ids, "MiniMax H3"):
                    self._log(tr("weights.missing.title", "Pesi mancanti") + " → MiniMax H3")
                    return
            except Exception:                             # noqa: BLE001 — l'errore chiaro del worker copre
                pass
            self._log(f"[minimax_h3] {'Ref2VA (riferimenti)' if params.get('references') else 'FL2VA'} · "
                      f"profilo {params['h3_profile']} · {params['num_frames']} fotogrammi "
                      f"({_S.frames_seconds(params['num_frames']):.2f}s) · audio "
                      f"{'sì' if params['generate_audio'] else 'no'}"
                      f"{' · turbo ' + params['h3_turbo'] + ' passi' if params.get('h3_turbo') else ''}")
            self._ltx2_worker = _NativeLtx2Worker(params)
            self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"), source="video")
            self._enqueue(tr("queue.job.video", "Video"), self._ltx2_worker, self._on_multitalk_done,
                          on_progress=lambda s: self._log("[minimax_h3] " + s))
            return
        # ── 3ª ARCHITETTURA: Krea Realtime (autoregressivo) ─────────────────────────────
        if vals.get("video_arch") == "krea_rt":
            params["arch"] = "krea_rt"
            params.pop("gguf_transformer", None)
            # "Krea Realtime" = motore Self-Forcing 1.3B (VALIDATO: realtime, streaming-su-disco + sfratto
            # KV-cache → durata illimitata). Il redirect è in core/ltx_video_job._ARCH_RUNNER (opt-in 14B via
            # KREA_RT_USE_14B=1). Modalità continua di default (finestra scorrevole + attention-sink).
            params.setdefault("kv_local_attn", int(vals.get("kv_local_attn", 21)))
            params.setdefault("kv_sink", int(vals.get("kv_sink", 3)))
            # prompt TESTO semplice (no JSON). num_blocks/durata li decide il worker per VRAM/durata.
            self._log(f"[krea_rt] Krea Realtime (Self-Forcing 1.3B, autoregressivo realtime): "
                      f"{params['width']}x{params['height']} {nf} frame")
            self._ltx2_worker = _NativeLtx2Worker(params)
            self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"), source="video")
            self._enqueue(tr("queue.job.video", "Video"), self._ltx2_worker, self._on_multitalk_done,
                          on_progress=lambda s: self._log("[krea_rt] " + s))
            return
        # ── variante REALTIME di Krea: Self-Forcing 1.3B (autoregressivo leggero, ~realtime sulla 5090) ──
        if vals.get("video_arch") == "self_forcing":
            params["arch"] = "self_forcing"
            params.pop("gguf_transformer", None)                # prompt TESTO semplice; durata=num_frames (rolling)
            # Modalità CONTINUA di default (validata): finestra scorrevole 21 latenti (~5s, in-distribuzione) con
            # SFRATTO della KV-cache + attention-sink → durata ILLIMITATA a memoria costante, seam eliminati.
            params.setdefault("kv_local_attn", int(vals.get("kv_local_attn", 21)))
            params.setdefault("kv_sink", int(vals.get("kv_sink", 3)))
            params.setdefault("no_seam_repair", True)           # il sink pulisce alla radice, niente blend post
            self._log(f"[self_forcing] Self-Forcing 1.3B realtime (continuo, finestra scorrevole+sink): "
                      f"{params['width']}x{params['height']} {nf} frame")
            self._ltx2_worker = _NativeLtx2Worker(params)
            self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"), source="video")
            self._enqueue(tr("queue.job.video", "Video"), self._ltx2_worker, self._on_multitalk_done,
                          on_progress=lambda s: self._log("[self_forcing] " + s))
            return
        self._log(f"[nativo] video LTX-2 GGUF: {params['width']}x{params['height']} {nf} frame "
                  f"{'+audio' if params.get('audio_path') else ''} {'(chunk)' if params.get('chunk_frames') else ''}")
        # SELETTORE MOTORE VIDEO: default 'ltx2_infer' (tutte le feature). 'comfy' = comfy-native
        # (model_management ComfyUI, VRAM dinamica/no-OOM) ma SOLO t2v/i2v semplici → per audio/
        # controlnet/storyboard/lungo ricade su ltx2_infer. Scelta in user_prefs (dropdown video_tab).
        from core import user_prefs as _up
        # DEFAULT = comfy-native (block-swap dinamico del fp8, come ComfyUI): sui 32GB tiene il 22B in VRAM e
        # sposta solo i blocchi in eccesso → veloce, niente dequant GGUF per-layer. Il motore ltx2_infer
        # (nativo puro) ha una gestione memoria più rudimentale sul 22B → resta come opzione (audio/controlnet/
        # lungo lo usano comunque). Cambiabile dal dropdown motore video in Impostazioni.
        _engine = _up.get_engine("video", "comfy")
        # ComfyUI RIMOSSO (.bau): il motore comfy video non è più disponibile (COMFY_PY inesistente) → forza il
        # NATIVO LTX-2 (ltx2_infer), altrimenti _VideoComfyWorker fallisce silenziosamente ("non genera").
        try:
            from core.ltx2_service import COMFY_PY as _CPY
            if _engine == "comfy" and not os.path.exists(_CPY):
                _engine = "ltx2_infer"
                self._log("[motore] ComfyUI non disponibile → video NATIVO LTX-2")
        except Exception:                                       # noqa: BLE001
            pass
        # comfy copre: t2v/i2v/i2v-soft + KEYFRAME/storyboard (LTXVAddGuide). Restano su ltx2_infer solo
        # audio(a2v)/controlnet/long-chunk finché non portati anch'essi su comfy.
        _simple = not (params.get("audio_path") or params.get("control_video")
                       or params.get("chunk_frames"))
        # AUTO-CALIBRAZIONE VRAM: il 22B bf16 non entra sotto ~40GB liberi (dequant → ~44GB). Se t2v/i2v
        # SEMPLICE e VRAM insufficiente: 1) auto-seleziona un LTX GGUF (dequant al volo, entra nei 32GB) →
        # resta NATIVO; 2) fallback comfy SOLO se davvero disponibile; 3) altrimenti avvisa (no output muto).
        if _engine != "comfy" and _simple and not params.get("gguf_transformer"):
            # F6 unified-model-manager: VRAM LIBERA reale (gpu_mem/NVML, non il totale) e soglia dal
            # peso VERO del modello scelto (transformer_mb: misurato → header) invece del 40GB fisso.
            # Fallback: se il DB non conosce il modello, resta la soglia storica 40GB.
            try:
                from core import gpu_mem as _gm
                _free = float(_gm.free_vram_mb()) / 1024.0
            except Exception:                                   # noqa: BLE001
                try:
                    from core import model_precision as _mp
                    _free = float(_mp.gpu_free_gb())
                except Exception:                               # noqa: BLE001
                    _free = 999.0
            _need_gb = 40.0
            try:
                from core import vram_estimate as _ve2
                _mdl = str(params.get("model", "") or "")
                if _mdl and os.path.exists(_mdl):
                    _tr = _ve2.transformer_mb(_mdl, fp8_storage="fp8" in os.path.basename(_mdl).lower())
                    if _tr > 0:
                        _need_gb = _tr / 1024.0 + 8.0           # transformer + working video
            except Exception:                                   # noqa: BLE001
                pass
            if _free < _need_gb:
                # NIENTE proposta di swap a GGUF: i GGUF video sono rimossi dal prodotto (13/07,
                # ~500s/step) — il dialog proponeva un modello che ora fallisce sempre. Il modello
                # SCELTO resta (regola utente: mai cambiare modello senza consenso): si carica con
                # offload dinamico per-blocco (lento ma genera).
                self._log(f"[motore] VRAM ~{_free:.0f}GB < necessaria ~{_need_gb:.0f}GB "
                          "(misura PRE-liberazione: gli altri motori vengono sfrattati prima del "
                          "load — la decisione residente/offload la fa il worker a VRAM liberata)")
        if _engine == "comfy" and _simple:
            self._log("[motore] comfy-native (VRAM dinamica, no-OOM)")
            self._ltx2_worker = _VideoComfyWorker(params)
            self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"), source="video")
            self._enqueue(tr("queue.job.video", "Video"), self._ltx2_worker, self._on_multitalk_done)
            return
        if _engine == "comfy":
            self._log("[motore] comfy non copre audio/controlnet/storyboard/lungo → uso ltx2_infer")
        # FASE G weights-unification: consenso PRIMO-USO — componenti obbligatorie della release
        # LTX assenti (es. Gemma TE su installazione fresca) → dialog di download col motore
        # unico. Cabla ltx_release.missing_catalog_ids (il preflight che era dead code).
        if (params.get("arch") or "ltx").lower() == "ltx":
            try:
                from core import ensure_weights as ew
                from core import ltx_release as _lr
                _rel = params.get("ltx_release") or _lr.DEFAULT_RELEASE
                _ids = _lr.missing_catalog_ids(_rel)
                if _ids and not ew.offer_download_gui(self, _ids, f"LTX {_rel}"):
                    self._log(tr("weights.missing.title", "Pesi mancanti") + f" → LTX {_rel}")
                    return
            except Exception:                               # noqa: BLE001 — l'errore chiaro del worker copre
                pass
        self._ltx2_worker = _NativeLtx2Worker(params)
        # il motore nativo LTX-2 scrive la preview per-step su preview_file (#92) → mostra il dialog anteprima
        # anche qui (prima appariva solo sul ramo comfy-native).
        self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"), source="video")
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
            "seed": int(vals.get("seed", self.video_tab.sampler_seed.seed.value())),   # #95 V1: seed randomizzato
            # niente offload forzato: 'cpu' = pinned memory ⇒ SEGFAULT col pagefile da 16GB (13/07)
            "output": out,
        }
        self._log(f"[nativo] ReTake [{params['start_time']:.1f}–{params['end_time']:.1f}]s su {Path(src).name}…")
        self._ltx2_worker = _NativeLtx2Worker(params)
        self._enqueue(tr("queue.job.video_retake", "Video ReTake"), self._ltx2_worker, self._on_multitalk_done,
                      on_progress=lambda s: self._log("[nativo] " + s))

    def _probe_video_dims(self, video: str):
        """(width, height, nb_frames, fps) del video via ffprobe; (0,0,0,0) se illeggibile."""
        try:
            import json as _j, subprocess as _sp
            r = _sp.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                         "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate",
                         "-of", "json", video], capture_output=True, text=True, timeout=60)
            s = _j.loads(r.stdout)["streams"][0]
            num, den = (s.get("r_frame_rate", "25/1") + "/1").split("/")[:2]
            fps = round(float(num) / float(den)) if float(den) else 25
            return int(s["width"]), int(s["height"]), int(s.get("nb_read_frames", 0) or 0), fps
        except Exception:                                    # noqa: BLE001
            return 0, 0, 0, 0

    def _gen_video_edit_ai(self, cfg: dict):
        """#24/#59/#60 Modifica Video AI: edit-anything / inpaint (SAM2 track) / style — worker async."""
        _spec = vem.MODE_SPECS.get(cfg.get("mode", ""))
        if _spec is not None and _spec.cap and not self._feature_gate(_spec.cap):
            return                                           # edizione senza la funzione: upsell mostrato dal gate
        if cfg.get("mode") == "inpaint" and cfg.get("r2v") and not self._feature_gate("cap.video_inpaint_r2v"):
            return                                           # inpaint con foto di riferimento (LoRA r2v): gate proprio
        vv = self.video_tab.get_values()
        _fps = int(vv.get("fps", 25) or 25)
        _r64e = lambda x: max(64, round(int(x) / 64) * 64)   # #95 V2: LTX richiede multipli di 64
        # L'EDIT deve PRESERVARE risoluzione/durata/fps del video SORGENTE (le maschere SAM2 e il canny sono
        # tracciati sul sorgente → l'output deve combaciare, altrimenti è distorto e di durata sbagliata).
        # Prima si usavano i controlli t2v del tab Video (704×1280/8s) → output non allineato al sorgente.
        _sw, _sh, _snf, _sfps = self._probe_video_dims(cfg.get("video", ""))
        if _sw and _sh:
            cfg.setdefault("width", _r64e(_sw)); cfg.setdefault("height", _r64e(_sh))
            cfg.setdefault("fps", int(_sfps or _fps))
            cfg.setdefault("num_frames", max(9, (_snf // 8 * 8 + 1) if _snf else
                                             int(float(vv.get("length_s", 1) or 1) * _fps) // 8 * 8 + 1))
        else:                                                # probe fallito → fallback ai controlli del tab
            cfg.setdefault("width", _r64e(vv.get("width", 512) or 512))
            cfg.setdefault("height", _r64e(vv.get("height", 320) or 320))
            cfg.setdefault("fps", _fps)
            cfg.setdefault("num_frames", max(9, int(float(vv.get("length_s", 1) or 1) * _fps) // 8 * 8 + 1))
        if cfg.get("enhance") and (cfg.get("prompt") or "").strip():   # arricchimento prompt invisibile (opzionale)
            try:
                from core import prompt_enhance as pe
                if pe.available():
                    cfg["prompt"] = pe.enhance(cfg["prompt"], kind="video", nsfw=bool(vv.get("nsfw"))) or cfg["prompt"]
            except Exception:                                          # noqa: BLE001
                pass
        self.video_edit_ai_tab.log.appendPlainText(tr("vedit.start", "[edit] avvio {m}…").format(m=cfg["mode"]))
        self._edit_ai_worker = _VideoEditAIWorker(cfg, str(OUTPUT_DIR))
        self._enqueue(tr("queue.job.video_edit", "Modifica Video AI"), self._edit_ai_worker,
                      self._on_video_edit_ai_done, on_progress=self.video_edit_ai_tab.log.appendPlainText)

    def _open_motion_tracks_from_video(self):
        """Dal tab Video a Modifica Video AI nella modalita' «Movimento da tracce», con l'immagine di partenza del tab Video
        (quella del canvas Frame 1) gia' impostata. Il tab Video non viene toccato."""
        t = self.video_edit_ai_tab
        self.tabs.setCurrentWidget(t)
        t.select_mode("motion_track")
        img = self.frame1_tab.image_path or ""
        if img and os.path.isfile(img):
            t.set_track_image(img)
            self._log(tr("vedit.mt.handoff_ok", "[Movimento da tracce] immagine di partenza dal canvas: disegna le tracce e genera."))
        else:
            self._log(tr("vedit.mt.handoff_none", "[Movimento da tracce] nessuna immagine nel canvas del Frame 1: scegli un'immagine di partenza."))

    def _open_v2v_from_vedit(self, mode: str, video: str, audio: str):
        """Dal «Doppiaggio da testo» di Modifica Video AI a Talk (video + audio) o Canta (foto + canzone) del pannello
        V2V del tab Video, con la modalita' selezionata e video/audio gia' compilati quando noti. Le due modalita' e il
        loro codice restano intatti: qui si impostano solo i loro controlli esistenti."""
        panel = self.video_tab.v2v_panel
        self.tabs.setCurrentWidget(self.video_tab)
        self.video_tab.show_edit_tab()                       # Talk/Canta vivono nel sotto-tab «Modifica» del tab Video
        name = Path(audio).name if audio else ""
        if mode == "sing":
            panel.rb_sing.setChecked(True)
            if audio and os.path.isfile(audio):
                panel._sing_audio_path = audio
                panel.lbl_sing_audio.setText(f"🎵 {name}")
                panel.lbl_sing_audio.setStyleSheet("")
            self._log("[Doppiaggio] aperto «Canta» nel tab Video: serve la foto del Frame 1 e la canzone."
                      + (f" Audio impostato: {name}." if name else ""))
            return
        panel.rb_talk.setChecked(True)
        if video and os.path.isfile(video):
            panel._set_video(video)
        if audio and os.path.isfile(audio):                  # Talk prende l'audio dal tab Audio («Audio fornito da me»)
            self.audio_tab._audio_file = audio
            self.audio_tab.file_lbl.setText(name)
            self.audio_tab.rb_user.setChecked(True)
            self.audio_tab._on_mode()
        self._log("[Doppiaggio] aperto «Talk» nel tab Video."
                  + (f" Video e audio ({name}) già impostati." if audio else " Scegli l'audio nel tab Audio."))

    def _on_video_edit_ai_done(self, res: dict):
        if res.get("status") == "success":
            outs = res.get("output") or [""]
            out = outs[0]
            self.video_edit_ai_tab.log.appendPlainText(tr("vedit.done", "✅ Pronto: {o}").format(o=out))
            for _x in outs[1:]:                              # hdr: HEVC PQ / cartella EXR accanto all'anteprima SDR
                self.video_edit_ai_tab.log.appendPlainText(tr("vedit.done_extra", "   + {o}").format(o=_x))
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
        # #10 RIFE dal tab Definizione: se attivo, fluidifica DOPO l'upscale (consumato in _maybe_rife).
        self._pending_def_rife = ({"factor": int(cfg.get("rife_multiplier", 2) or 2)}
                                  if cfg.get("rife") else None)
        # CATENA: dopo la super-risoluzione applica restauro volti (poi RIFE) in sequenza, senza rilanciare.
        self._pending_def_face = (self.video_def_tab.get_face_restore_config()
                                  if cfg.get("chain_face") else None)
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
                                              batch_size=int(cfg.get("batch_size", 5) or 5), chunk_size=120,
                                              model=cfg.get("seedvr2_model", "") or "",
                                              block_swap=int(cfg.get("blocks_to_swap", 0) or 0),
                                              temporal_overlap=int(cfg.get("temporal_overlap", 3) or 0))
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
            _fvs = float(cfg.get("target_scale", 2.0) or 2.0)
            self._log(f"[nativo] FlashVSR super-risoluzione (x{_fvs:g}) su {Path(src).name}…")
            self._fv_worker = _FlashVSRWorker(src, out, target_scale=_fvs)
            self._enqueue(tr("queue.job.flashvsr", "Super-risoluzione (FlashVSR)"), self._fv_worker,
                          self._on_multitalk_done, on_progress=self._log)
            return
        if engine == "flashvsr_plus":                    # FlashVSR+ Turbo/Base (pacchetto nativo, SDPA)
            from core import flashvsrplus_service as fvp
            _fp = dict(cfg.get("fvp") or {})
            _mode = _fp.get("mode", "turbo")
            errs = fvp.preflight(_mode)
            if errs:
                QMessageBox.warning(self, "FlashVSR+", "\n".join(errs) +
                    "\n\n" + tr("mw.fvsrplus_install_hint", "Installa il codice con: python scripts/install_flashvsrplus.py "
                                  "e scarica i pesi dal tab «Pesi modelli» (flashvsr-plus-turbo / flashvsr-plus-base).")); return
            out = str(OUTPUT_DIR / f"flashvsrplus_{_mode}_{Path(src).stem}.mp4")
            _fvs = float(cfg.get("target_scale", 2.0) or 2.0)
            _tc = int(_fp.get("temporal_compression", 4) or 4)
            self._log(f"[nativo] FlashVSR+ {_mode} super-risoluzione (x{_fvs:g}, movimento {_tc}) su {Path(src).name}…")
            self._fvp_worker = _FlashVSRPlusWorker(src, out, target_scale=_fvs, mode=_mode, temporal_compression=_tc)
            self._enqueue(tr("queue.job.fvsrplus", "Super-risoluzione (FlashVSR+)"), self._fvp_worker,
                          self._on_multitalk_done, on_progress=self._log)
            return
        if engine == "h3_upsampler":                     # MiniMax H3 Video Upsampler nativo (upscaler latente + tile guidati)
            if not self._feature_gate("cap.video_h3_upsampler"):
                return
            from core import h3_upsample_service as hs
            p = dict(cfg.get("h3up") or {})
            p["video"] = src
            errs = hs.preflight(p)                       # valori, venv, pesi (nessun download)
            if errs:
                QMessageBox.warning(self, "H3 Video Upsampler", "\n".join(errs)); return
            out = str(OUTPUT_DIR / f"h3up_{Path(src).stem}.mp4")
            self._log(f"[nativo] H3 Video Upsampler ({p.get('megapixels')} MP, denoise {p.get('denoise')}, "
                      f"{'guidato' if p.get('guide') else 'senza guida'}) su {Path(src).name}…")
            self._h3up_worker = _H3UpsampleWorker(src, out, p)
            self._enqueue(tr("queue.job.h3up", "Super-risoluzione (H3 Video Upsampler)"), self._h3up_worker,
                          self._on_multitalk_done, on_progress=self._log)
            return
        # esrgan: upscale per-frame nativo (spandrel). CHIAVI CORRETTE: il backend video_postprocess.run()
        # legge 'image_path' e 'target_scale' (prima passavamo 'video'/'scale' → KeyError = upscale ESRGAN
        # video ROTTO, e comunque il fattore ignorato).
        params = {
            "image_path": src, "model": cfg.get("esrgan_model"),
            "target_scale": int(cfg.get("target_scale", 2) or 2), "seed": _r.randint(0, 2**31),
        }
        self._log(f"[nativo] definizione video (upscale per-frame x{params['target_scale']}) su {Path(src).name}…")
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
        # steps: il controllo del tab Prolunga ha priorità (era ignorato → si usava quello del tab Video)
        ext_steps = int(cfg.get("steps", 0) or 0) or int(self.video_tab.sampler_seed.steps.value())
        lastf = str(OUTPUT_DIR / f"_ext_last_{int(_t.time())}.png")
        _sp.run(["ffmpeg", "-v", "error", "-sseof", "-0.12", "-i", src, "-vframes", "1", lastf, "-y"], check=False)
        if not os.path.exists(lastf):
            QMessageBox.warning(self, tr("mw.extend_video_error_title", "Prolunga"), "Impossibile estrarre l'ultimo frame."); return
        cont = str(OUTPUT_DIR / f"_ext_cont_{int(_t.time())}.mp4")
        self._ext_src = src
        _r64x = lambda x: max(64, round(int(x) / 64) * 64)   # #95 V2: LTX richiede multipli di 64
        _seed = (int(self.video_tab.sampler_seed.seed.value()) if self.video_tab.wants_fixed_seed()
                 else random.randint(0, 2**31))              # #95 V1: seed random se non fissato
        params = {
            "image_path": lastf, "prompt": cfg["prompt"],
            "negative": v.get("prompt_negative", "") or "worst quality, blurry",
            "width": _r64x(v.get("width", 512)), "height": _r64x(v.get("height", 512)),
            "num_frames": nf, "fps": fps, "steps": ext_steps,
            "guidance": float(self.video_tab.sampler_seed.cfg.value()), "seed": _seed,
            # niente offload forzato: 'cpu' = pinned memory ⇒ SEGFAULT col pagefile da 16GB (13/07)
            "ref_strength": 1.0, "output": cont,
            "teacache_thresh": float(v.get("teacache_thresh", 0.0) or 0.0),
            "decode_lowmem": bool(v.get("decode_lowmem")),
            "native_audio": bool(v.get("native_audio")),
        }
        # MODELLO, LoRA e RELEASE dal tab Video, come per la generazione normale. Qui non venivano
        # passati affatto: la prolunga caricava SEMPRE il checkpoint di default (ltx-2.3-22b-dev-fp8,
        # 23,8 GB) più la LoRA distilled (7,2 GB) — 31 GB su 31,84 di scheda, quindi OOM prima ancora
        # di iniziare, qualunque cosa fosse selezionata nel tab. Con un checkpoint distilled la LoRA
        # non serve e lo spazio torna.
        if v.get("model_kind") == "gguf" and v.get("model_gguf"):
            params["gguf_transformer"] = v["model_gguf"]
        elif v.get("model") and v.get("model") not in ("", "auto"):
            params["model"] = v["model"]
        # Le LoRA NON stanno in get_values(): la generazione normale le prende dal tab LoRA
        # (main_window:2157). Senza, la prolunga girava sempre senza LoRA — o peggio, col worker
        # che caricava la distilled di default perché il checkpoint non era quello scelto.
        _lrs = [{"name": l["name"], "strength": l["strength"]}
                for l in self.lora_tab.selected_loras()] if hasattr(self, "lora_tab") else []
        if _lrs:
            params["loras"] = _lrs
        from core import ltx_release as _lrel_x
        _det_x = _lrel_x.detect_release(params.get("gguf_transformer") or params.get("model") or "")
        if _det_x:
            params["ltx_release"] = _det_x
        self._log(f"[nativo] prolunga: modello={os.path.basename(str(params.get('gguf_transformer') or params.get('model') or 'default'))} "
                  f"release={params.get('ltx_release', '2.3')} {params['width']}x{params['height']} "
                  f"steps={ext_steps} lora={len(_lrs or [])}")
        # seg_seconds → dimensione del chunk in-process (memory-bounded) per estensioni lunghe;
        # il worker segmenta quando num_frames > chunk_frames (continuità via ultimo frame).
        seg_s = float(cfg.get("seg_seconds", 0) or 0)
        chunk = int(seg_s * fps) // 8 * 8 + 1 if seg_s > 0 else 121
        if nf > chunk:
            params["chunk_frames"] = max(9, chunk)
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
        if params.get("lipsync_engine") == "multitalk":          # scelta dell'utente: MultiTalk (Wan 2.1) nativo
            return self._start_wan_multitalk(params)
        import subprocess as _sp, time as _t
        img = params.get("image_path")
        src = params.get("source_video")
        if not img and src:                              # talk da video → estrai il primo frame
            img = str(OUTPUT_DIR / f"_talk_src_{int(_t.time())}.png")
            _sp.run(["ffmpeg", "-v", "error", "-i", src, "-vframes", "1", img, "-y"], check=False)
        fps = int(params.get("fps", 25) or 25)               # NameError: 'fps' era usato più sotto senza esistere
        nf = int(params.get("num_frames", 121) or 0)
        if nf < 9:                                           # durata 0/assente → il video usciva a 0s (o crash)
            nf = max(9, int(round(float(params.get("seconds", 5) or 5) * fps)) // 8 * 8 + 1)
            self._log(f"[lip-sync] durata non valida → uso {nf} frame ({nf / fps:.1f}s)")
        _r64m = lambda x: max(64, round(int(x) / 64) * 64)   # #95 V2: dimensioni multiple di 64
        _mt_seed = params.get("seed")                        # #95 V3: seed dal chiamante (re-roll possibile), altrimenti random
        _mt_seed = int(_mt_seed) if _mt_seed is not None else random.randint(0, 2**31)
        nat = {
            "image_path": img, "audio_path": params.get("audio_path"),
            "prompt": params.get("prompt", ""),
            "negative": params.get("negative", "") or "worst quality, blurry",
            "width": _r64m(params.get("width", 512)), "height": _r64m(params.get("height", 512)),
            "num_frames": nf, "fps": fps,
            "steps": int(params.get("steps", 8)), "guidance": float(params.get("guidance", 3.0)),
            # NIENTE "offload": "cpu" — il block-streaming carica i pesi in PINNED MEMORY e col pagefile
            # da 16GB l'allocazione fallisce ⇒ SEGFAULT (exit 139) dopo un denoise perfetto, senza mp4.
            # In auto il worker tiene il transformer residente: niente crash e ~3× più veloce. (13/07)
            "seed": _mt_seed,
            # #4 forza dell'audio sul video (sing_audio_scale) → modality_scale del guider audio (G5: default 2.0
            # coerente con la GUI; il canto usa il valore del pannello, il talk 2.0).
            "audio_scale": float(params.get("audio_scale", 2.0) or 2.0),
            # #4 separa la voce dalla base prima del lip-sync (Canta) — gestito nel thread del worker
            "_sep_vocals": bool(params.get("separate_vocals", False)),
            "ref_strength": 1.0, "output": params.get("output", str(OUTPUT_DIR / "talk_native.mp4")),
            # VIDEO LUNGHI (LV2): anti-drift + resume + overlap crossfade nel worker chunked
            "overlap_frames": int(params.get("overlap_frames", 0) or 0),
            "ref_anchor_strength": float(params.get("ref_anchor_strength", 0.0) or 0.0),
            "anchor_every": int(params.get("anchor_every", 0) or 0),
            "resume": bool(params.get("resume", False)),
        }
        # #3 talk region: se c'è una regione e reimagine è OFF, composita solo bocca/volto sul
        #    frame sorgente statico DOPO l'a2v (post-step nel worker).
        if params.get("talk_region") and not params.get("talk_reimagine") and img:
            nat["_talk_region"] = params.get("talk_region")
            nat["_talk_offset"] = int(params.get("talk_mask_offset", 0) or 0)
            nat["_talk_source"] = img
        _single_max = int(20 * fps) // 8 * 8 + 1             # ≤20s = singolo shot (LTX-2.3 regge)
        # LV2: usa la durata-segmento della GUI (Avanzate → Video lungo) come chunk, cappata al max per-shot
        # e quantizzata (≡1 mod 8). Prima il chunk era fisso a 20s ignorando lo slider.
        _seg = int(params.get("seg_frames", 0) or 0)
        _chunk = min(_seg, _single_max) if _seg >= 9 else _single_max
        _chunk = max(9, _chunk // 8 * 8 + 1)
        if nf > _chunk:
            nat["chunk_frames"] = _chunk
        self._log(f"[lip-sync nativo] a2v LTX-2 fp8 — {nf} frame ({nf / fps:.1f}s) "
                  f"{nat['width']}x{nat['height']}…")
        self._mt_worker = _NativeLtx2Worker(nat)
        self._enqueue(tr("queue.job.lipsync", "Lip-sync"), self._mt_worker, self._on_multitalk_done,
                      on_progress=lambda s: self._log("[lip-sync] " + s))

    def _on_multitalk_done(self, res: dict):
        self._stop_native_preview()
        # Handler condiviso di TUTTI i video nativi (t2v/i2v/a2v/retake/controlnet), non solo MultiTalk:
        # etichetta generica "[video]" per non confondere (un t2v fallito non è un errore MultiTalk).
        if res.get("status") != "success":
            self._log(f"[video] errore: {res.get('error','')[:300]}")
            QMessageBox.critical(self, tr("mw.gen_error_title", "Errore"), str(res.get("error", ""))[:1000])
            return
        out = (res.get("output") or [None])[0]
        self._last_video_path = out
        self._log(f"[video] pronto: {out}")
        # CATENA tab Definizione: super-risoluzione → RESTAURO VOLTI → (RIFE), in sequenza. Consuma il flag una
        # sola volta: il face-restore ri-entra qui con _pending_def_face=None → prosegue a face-swap/RIFE.
        _pf = getattr(self, "_pending_def_face", None)
        if _pf and out:
            self._pending_def_face = None
            self._log("[catena] restauro volti sul video migliorato…")
            self._face_restore_native({**_pf, "video": out})
            return
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
        """Post-process RIFE (fluidifica) se attivo, altrimenti annuncia il video.
        Priorità: richiesta esplicita dal tab Definizione (#10) → auto-RIFE del tab Video."""
        rf = getattr(self, "_pending_def_rife", None)
        self._pending_def_rife = None                        # consuma una sola volta
        if not rf:
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

    def _materialize_talk_audio(self, ac: dict) -> str | None:
        """G3 — per Talk: se il tab Audio non ha un file pronto, prova a produrlo senza bloccare l'utente:
        (a) un TTS già generato nel tab Audio (pending) → usalo; (b) voce «clone» + testo → sintesi TTS ora.
        Ritorna il path o None (per i casi 'native'/'mute' che davvero non hanno una traccia). Non solleva."""
        try:
            pend = self.audio_tab.has_pending_tts() if hasattr(self.audio_tab, "has_pending_tts") else None
            if pend and Path(str(pend)).exists():
                self._log(f"[Talk] uso il TTS già generato: {Path(str(pend)).name}")
                return str(pend)
            if ac.get("audio_mode") == "clone" and (ac.get("tts_text") or "").strip() \
                    and ac.get("clone_voice") is not None:
                from core.tts_engine import synthesize as _tts_synth
                self._log(f"[Talk] sintesi voce clonata '{ac['clone_voice'].name}' ({ac.get('tts_engine')})…")
                out = _tts_synth(ac.get("tts_engine") or "edge", ac["clone_voice"], ac["tts_text"].strip())
                if out and Path(str(out)).exists():
                    self._log(f"[Talk] audio clonato pronto: {Path(str(out)).name}")
                    return str(out)
        except Exception as e:                              # noqa: BLE001
            self._log(f"[Talk] materializzazione audio fallita: {e}")
        return None

    def _start_wan_multitalk(self, params: dict):
        """Lip-sync MultiTalk NATIVO (Wan2.1 I2V 14B fp8 + adattatore audio): preflight con offerta di download dei pesi,
        poi job in coda. Parametri dal pannello Talk/Canta; guidance LTX ignorata (con la distillazione lightx2v cfg=1)."""
        from core import multitalk_service as mts
        g = dict(params)
        g["cfg"] = 1.0
        g["seed"] = g.get("seed") if g.get("seed") is not None else random.randint(0, 2**31)
        g.setdefault("output", str(OUTPUT_DIR / "multitalk.mp4"))
        miss = mts.missing_weights()
        if miss:
            try:
                from core import ensure_weights as ew
                if ew.missing(mts.weight_ids()) and not ew.offer_download_gui(self, mts.weight_ids(), "MultiTalk"):
                    return
            except Exception:                           # noqa: BLE001 — il preflight sotto da' il messaggio
                pass
        errs = mts.preflight(g)
        if errs:
            QMessageBox.warning(self, "MultiTalk", "\n".join(errs))
            return
        nf = int(g.get("num_frames", 81) or 81)
        self._log(f"[lip-sync MultiTalk] Wan2.1 14B nativo — {nf} frame ({nf / float(g.get('fps', 25) or 25):.1f}s) "
                  f"{g.get('width')}x{g.get('height')}…")
        self._mt_worker = _NativeMultiTalkWorker(g)
        self._enqueue(tr("queue.job.lipsync_multitalk", "Lip-sync (MultiTalk)"), self._mt_worker, self._on_multitalk_done,
                      on_progress=lambda s: self._log("[lip-sync] " + s))

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
            # G3 — se non c'è un file pronto, prova a MATERIALIZZARLO (clone+testo → TTS; TTS pending → usa),
            # come fa il ramo ComfyUI: prima Talk si bloccava «Audio mancante» anche con una voce clone scelta.
            if not (audio_path and Path(str(audio_path)).exists()):
                audio_path = self._materialize_talk_audio(ac)
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
        # G4 — talk DA VIDEO: la durata deve seguire il video sorgente (prima poteva troncare). Poi, se ancora
        # nulla, la durata dell'audio.
        if src_video and Path(str(src_video)).exists():
            _dv = self._probe_duration(src_video)
            if _dv and _dv > 0:
                length = _dv
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
            # propaga steps/cfg/seed scelti nel tab Video a talk/canto (prima usava i default 8/3.0)
            "steps": int(vals.get("steps", 8) or 8),
            "guidance": float(vals.get("cfg", vals.get("guidance", 3.0)) or 3.0),
            "seed": vals.get("seed"),
            "separate_vocals": sep, "audio_scale": ascale, "blocks_to_swap": "auto", "context": True,
            "seg_frames": seg_frames, "overlap_frames": overlap_frames,
            "ref_anchor_strength": ref_anchor, "anchor_every": anchor_every, "resume": resume,
            # #3 talk region: anima solo bocca/volto (reimagine=off) tenendo fermo il resto.
            # G2 — SOLO in modalità talk: nel canto (full-body) la regione-bocca non ha senso e non va passata.
            "talk_region": v2v.get("talk_region") if mode == "talk" else None,
            "talk_mask_offset": int(v2v.get("talk_mask_offset", 0) or 0) if mode == "talk" else 0,
            "talk_reimagine": bool(v2v.get("talk_reimagine", False)) if mode == "talk" else False,
            "output": str(OUTPUT_DIR / f"multitalk_{mode}.mp4"),
            "lipsync_engine": v2v.get("lipsync_engine", "ltx"),
            # LoRA Wan 2.1 scelte nella sezione «LoRA Wan» di Talk/Canta (si sommano alla distill lightx2v)
            "loras": list(v2v.get("wan_loras") or []),
        }
        self._start_native_multitalk(params)
        return True

    @staticmethod
    def _denoise_from_fidelity(fidelity) -> float:
        """Punto 13 dell'utente: lo slider è «Fedeltà all'originale» (0 = cambia tutto, 1 = immagine invariata);
        denoise/strength = 1 − valore. L'outpaint resta a 1,0 (i limiti per motore stanno nel backend)."""
        return max(0.0, min(1.0, 1.0 - float(fidelity)))

    def _start_native_inpaint(self, exp, unet, prompt, negative, steps, seed, retention, mode,
                              guidance_override=None):
        """Inpaint/outpaint via backend NATIVO (FluxFillPipeline) — niente ComfyUI.
        Passa i path LOCALI di immagine+maschera (image_flux li apre con PIL)."""
        import os
        from core import native_driver
        from core.inpaint_pipeline import InpaintPipeline
        _icfg = self.frame1_tab.get_inpaint_config()
        if _icfg.get("cn_inpaint"):                       # ControlNet Inpainting (Alimama) su FLUX.1 base (task 9)
            self._start_native_cn_inpaint(exp, unet, prompt, negative, steps, seed, mode, _icfg)
            return
        if not self._ensure_native_ready():   # caduto? lo riavvia da sé
            return
        model = unet if os.path.isabs(unet) else os.path.join(r"E:\AI\Models\diffusion_models", unet)
        # FLUX.2 Klein (miraclein) = arch distillata edit-instruct con encoder Qwen3: instradata a
        # 'flux2_inpaint' (Flux2KleinInpaintPipeline). FLUX.1 Fill resta 'inpaint'. La guidance per Klein
        # è bassa (CFG reale ~4), per Fill è alta (~30) → effective_guidance(flux2=...).
        from core.gen_pipeline import is_flux2
        _f2 = bool(unet and is_flux2(unet))
        arch = "flux2_inpaint" if _f2 else "inpaint"
        # Guidance ESPLICITA dalla GUI se impostata, ma CLAMPATA al modello: i distillati (Hyper/Turbo/8-step)
        # e Klein vanno a guidance BASSA — a 30 si BRUCIANO (macchie "come se mancassero step"). Fill dev = alta.
        from core.inpaint_pipeline import _is_hyper
        if guidance_override:
            g = float(guidance_override)
            guidance = min(g, 6.0) if _f2 else (min(g, 3.5) if _is_hyper(unet) else g)
            if guidance != g:
                self._log(f"[inpaint] modello distillato → guidance {g:.0f} clampata a {guidance:.1f} (evita macchie)")
        else:
            guidance = InpaintPipeline.effective_guidance(unet, 30.0, None, flux2=_f2)   # guidance NON più legata allo slider (ora «Fedeltà»: default 0 la dimezzerebbe)
        # #95 I5: forza (strength) → denoise, non solo guidance (outpaint=1.0, inpaint scala con retention)
        denoise = 1.0 if mode == "outpaint" else self._denoise_from_fidelity(retention)
        params = {"model": model, "arch": arch, "prompt": prompt, "negative": negative,
                  "image_path": exp["image"], "mask_path": exp["mask"],
                  "steps": steps, "guidance": guidance, "seed": seed, "denoise": denoise,
                  "feather": int(self.frame1_tab.get_inpaint_config().get("feather", 0)),
                  "outpaint": mode == "outpaint"}
        self._inpaint_mode = mode
        # log componenti/settaggi EFFETTIVI (visibilità: cosa è stato auto-selezionato davvero)
        self._log(f"{mode} (backend NATIVO): modello={os.path.basename(model)} · pipeline={arch} · "
                  f"steps={steps} · guidance={guidance:.1f} · denoise={denoise:.2f} · seed={seed}")
        if _f2 and int(steps) > 14 and mode == "inpaint":      # live 17/10: Klein a 21 passi = foto quasi invariata (MAD 4); a 12 = edit pieno
            self._log(tr("mw.klein_steps_warn", "⚠ FLUX.2 Klein è distillato: con {n} passi l'edit si indebolisce fino a lasciare la foto quasi "
                         "invariata (provato dal vivo: 12 passi = edit completo, 16 = parziale, 21 = invariato). Consigliati 12 passi.").format(n=int(steps)))
        self._native_worker = _NativeImageWorker(params)
        self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"))
        self._enqueue(tr("queue.job.inpaint", "Inpaint/Outpaint"), self._native_worker,
                      self._on_native_inpaint_done)

    def _start_native_kontext(self, exp, model, prompt, negative, steps, seed, guidance, pulid=None):
        """Kontext/integra via backend NATIVO (FluxKontextPipeline) — niente ComfyUI.
        Il composito (scena+oggetto) è il riferimento; il modello rigenera/armonizza."""
        import os
        from core import native_driver
        from core.kontext_pipeline import KONTEXT_MODEL
        if not self._ensure_native_ready():   # caduto? lo riavvia da sé
            return
        if not prompt:
            # IL PROMPT DEVE NOMINARE L'OGGETTO. Non è un dettaglio: con "the added object" il modello non
            # sa cosa ha davanti e lo REINVENTA (a parità di tutto il resto, al posto del calice mi ha
            # prodotto un aggeggio bianco con una faccina). Nominandolo — "wine glass" — la stessa identica
            # generazione, stesso seed, restituisce la mano che lo tiene. Quindi lo chiediamo, non lo
            # indoviniamo. NB "fingers wrapping around it" fa generare VETRI IN FRANTUMI: frase piana.
            what, ok = QInputDialog.getText(
                self, tr("mw.integrate_what_title", "Che cosa stai inserendo?"),
                tr("mw.integrate_what_msg",
                   "Descrivi in poche parole l'oggetto aggiunto (in inglese rende meglio):\n"
                   "es. «a glass of white wine», «a leather handbag», «a cat»"))
            what = (what or "").strip()
            if ok and what:
                prompt = (f"the hand holds {what}; match the lighting, shadows, color and perspective "
                          f"so it looks like a real part of the photo; photorealistic")
            else:
                prompt = ("the hand holds the added object naturally; match the lighting, shadows, color "
                          "and perspective so it looks like a real part of the photo; photorealistic")
            self._log(f"[integra] prompt: {prompt}")
        # INTEGRA MASK-BASED: se il canvas ha esportato la maschera della zona-edit, usa INPAINT
        # (FLUX Fill) su quella regione → rigenera/fonde l'oggetto nella scena (luce/ombre/bordi)
        # invece del kontext senza maschera (che preservava l'oggetto = solo posizionato). Vedi
        # export_integrate (oggetto-seme + alone bianco) + [[video-edit-inpaint-nativo]].
        mask = exp.get("mask")
        if mask and os.path.exists(mask):
            # MODELLO: il dropdown dell'utente comanda; il DEFAULT è un FLUX normale, NON il Fill.
            # È la differenza fra integrare e tirare a sorte: il Fill ignora ciò che sta sotto la maschera
            # e rigenera da zero (a parità di seed ho ottenuto una mano col calice, vetri in frantumi e un
            # paesaggio). Un FLUX normale entra in FluxInpaintPipeline, che ha `strength`: parte dai TUOI
            # pixel e li reinterpreta quanto basta — è il SetLatentNoiseMask della vecchia IntegratePipeline
            # (ComfyUI, 3/06), quella che davvero piegava il braccio e chiudeva le dita sull'oggetto.
            from core import arch_filter as _af
            _m = model or r"Flux\FLUX_Dev.safetensors"
            mp = _m if os.path.isabs(_m) else os.path.join(r"E:\AI\Models\diffusion_models", _m)
            params = {"model": mp, "arch": "flux2_inpaint" if _af.is_flux2(_m) else "inpaint", "prompt": prompt, "negative": negative,
                      "image_path": exp["image"], "mask_path": mask,
                      # guidance ≥3.5 (standard FLUX): col 2.5 ereditato dal tab (tarato per Kontext) il
                      # modello si allontana dal prompt e reinventa l'oggetto.
                      "steps": max(steps, 25), "guidance": max(3.5, float(guidance or 0)), "seed": seed,
                      # DENOISE 0.65 — misurato: a 0.55 l'oggetto resta incollato (nessuna mano), a 0.75
                      # il modello lo cancella. A 0.65 la mano lo afferra e l'oggetto resta il tuo.
                      "denoise": float(self.frame1_tab.get_inpaint_config().get("integrate_denoise") or 0.65),
                      # FEATHER AMPIO (default inpaint 6px): il backend ricompone i pixel originali fuori
                      # maschera; con la zona-edit larga un bordo netto disegna un RIQUADRO visibile.
                      "feather": 48}
            # NB: niente change-map (Differential Diffusion) qui — provata il 13/07 e RIMOSSA: peggiorava
            # il risultato rispetto all'alone bianco + maschera larga di export_integrate. Il supporto nel
            # backend resta (params["change_map"]), ma non lo alimentiamo.
            self._log(f"Integra mask-based (modello={os.path.basename(mp)}, backend NATIVO) inviato (steps={params['steps']}, seed={seed})…")
        else:
            unet = model or KONTEXT_MODEL
            mp = unet if os.path.isabs(unet) else os.path.join(r"E:\AI\Models\diffusion_models", unet)
            from core import arch_filter as _af
            # FLUX.2 (Klein) come motore di Integra: pipeline nativa Flux2KleinPipeline (edit per istruzione), mai il grafo ComfyUI
            params = {"model": mp, "arch": "flux2_kontext" if _af.is_flux2(unet) else "kontext", "prompt": prompt, "negative": negative,
                      "image_path": exp["image"], "steps": steps, "guidance": guidance, "seed": seed}
            if pulid and pulid.get("pulid_face"):             # identità volto (PuLID) anche sull'integra kontext
                params.update({k: pulid[k] for k in ("pulid_face", "pulid_strength",
                                                      "pulid_start_step", "pulid_true_cfg") if k in pulid})
                self._log("Integra Kontext + PuLID (identità volto) attivo")
            self._log(f"Integra Kontext (backend NATIVO) inviato (steps={steps}, seed={seed})…")
        self._inpaint_mode = "integrazione"
        self._native_worker = _NativeImageWorker(params)
        self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"))
        self._enqueue(tr("queue.job.integrate", "Image to Image (Kontext)"), self._native_worker,
                      self._on_native_inpaint_done)

    _QWEN21_INTEGRATE_PROMPT = ("Blend the pasted elements naturally into the scene: match lighting, shadows, "
                                "perspective and color grading, keep their identity and shape.")

    def _ensure_qwen21_weights(self, dit: str, te: str = "", turbo: str = "", *, pe: str = "",
                               control: bool = False) -> bool:
        """DiT Qwen scelto (+ deps, encoder scelto, LoRA turbo, arricchitore e ControlNet se richiesti)
        presente, o scaricato col consenso."""
        try:
            from core import ensure_weights as ew
            from core.qwen21_spec import TURBO, PE, CONTROLNET, is_turbo, te_for
            ids = [dit, te_for(dit, te)]                      # il TE non è più una dipendenza del DiT
            if turbo in TURBO and not is_turbo(dit):
                ids.append(TURBO[turbo][0])
            ids += [PE[k][0] for k in (("t2i", "i2i") if pe == "auto" else (pe,)) if k in PE]
            if control:
                ids.append(CONTROLNET)
                if self.frame1_tab.qwen21_control_params().get("control_preprocess") == "depth":
                    ids.append("depth-anything-v2")
            if ew.missing(ids) and not ew.offer_download_gui(self, ids, "Qwen-Image 2.1"):
                return False
        except Exception:                                     # noqa: BLE001 — il worker verifica comunque
            pass
        return True

    @staticmethod
    def _inp_loras(cfg: dict) -> list:
        return [{"name": cfg["lora"], "strength": 1.0}] if cfg.get("lora") else []

    def _canvas_output_size(self) -> dict:
        """Risoluzione d'uscita = quella del CANVAS visibile (task 16b). Caricando una foto in modalità «canvas» il
        canvas adotta la forma della foto (set_aspect) SENZA aggiornare i valori del tab Video: usare quelli dava
        1024×1024 → 704×1280 (formato cambiato, inquadratura tagliata). Se il canvas non è utilizzabile si ricade
        sui valori del tab Video. Sempre col tetto di risoluzione dell'edizione."""
        v = dict(self.video_tab.get_values())
        try:
            cv = self.frame1_tab.canvas
            cw, ch = int(cv.aspect_w), int(cv.aspect_h)
            if cw >= 64 and ch >= 64:
                v["width"], v["height"] = cw, ch
        except Exception:                                     # noqa: BLE001 — canvas assente: tab Video
            pass
        self._clamp_resolution(v)
        return v

    def _start_native_qwen21_edit(self, dit, prompt, *, refs=(), init_image="", mask="", strength=None,
                                  negative="", seed=None, steps=None, guidance=None, loras=None, feather=0,
                                  mode="qwen21_edit", label=None, enhance=False, extra=None):
        """Qwen-Image 2.1 nei pannelli di editing (Inpaint/Outpaint, Integra, Componi, Colore). Con `refs` è
        editing per istruzione (≤10 immagini, «image N» nel prompt); con `init_image`+`mask` rifà SOLO l'area
        mascherata (il resto resta l'originale al pixel). Passi/CFG: quelli consigliati per il DiT se non dati."""
        if not self._feature_gate("cap.image_qwen21"):
            return
        if not dit:
            QMessageBox.warning(self, tr("mw.no_model_title"),
                                tr("mw.qwen21_no_dit", "Scegli il modello Qwen-Image 2.1 nel menu del pannello."))
            return
        if not (prompt or "").strip():
            QMessageBox.information(self, tr("mw.no_mask_title", "Serve un'istruzione"),
                                    tr("mw.qwen21_no_prompt", "Scrivi COSA modificare nell'istruzione."))
            return
        refs = [r for r in (refs or []) if r]
        if not refs and not init_image:
            QMessageBox.warning(self, tr("mw.no_image_title"), tr("mw.load_photo_canvas_msg")); return
        te = self.frame1_tab.qwen21_te()
        if enhance and not self._feature_gate("cap.qwen21_prompt_enhancer"):
            enhance = False                                   # edizione senza arricchitori: si procede senza
        pe = self.frame1_tab.qwen21_pe_params("i2i") if enhance else {}
        extra = dict(extra or {})
        if not self._ensure_qwen21_weights(dit, te, pe=pe.get("qwen21_pe", ""),
                                           control=bool(extra.get("control_image"))) \
                or not self._ensure_native_ready():
            return
        from core.qwen21_spec import DITS, TURBO_STEPS, DEFAULT_STEPS, DEFAULT_GUIDANCE, MAX_REFS
        meta = DITS.get(dit, {})
        turbo = bool(meta.get("turbo"))
        params = {"arch": "qwen_image21", "qwen21_dit": dit, "qwen21_te": te, "prompt": prompt,
                  "negative": negative or "",
                  "seed": seed if seed is not None else random.randint(0, 2**31),
                  "steps": TURBO_STEPS if turbo else int(steps or meta.get("steps", DEFAULT_STEPS)),
                  "guidance": 1.0 if turbo else float(guidance if guidance is not None
                                                      else meta.get("cfg", DEFAULT_GUIDANCE)),
                  "ref_images": refs[:MAX_REFS], "loras": list(loras or []), **pe, **extra}
        if init_image:
            params["init_image"] = init_image
        if mask:
            params["mask_path"] = mask
            params["feather"] = int(feather or 0)
            if mode == "outpaint":
                params["outpaint"] = True       # il worker: riempimento dell'estensione + tono globale (task 14)
        else:                                  # risultato alla risoluzione e al rapporto del canvas (mai stirato);
            v = self._canvas_output_size()     # con la maschera vale il ritaglio del canvas, rimesso al pixel
            params["width"], params["height"] = int(v["width"]), int(v["height"])
        if strength is not None:
            params["strength"] = float(strength)
        self._inpaint_mode = mode
        self._native_worker = _NativeImageWorker(params)
        self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"))
        self._log(f"Qwen-Image 2.1 Edit ({dit}, {mode}, {len(refs)} rif., "
                  f"{'maschera' if mask else 'senza maschera'}, {params['steps']} passi) inviato…")
        self._enqueue(label or tr("queue.job.qwen21edit", "Qwen-Image 2.1 Edit"), self._native_worker,
                      self._on_native_inpaint_done)

    # ── FLUX.3 Image Klein (task 13): crea / modifica per istruzione / multi-riferimento / riquadri ───────────────
    def _apply_flux3_size_cap(self, *_):
        """Larghezza/altezza del tab Video: tetto 4096 con l'arch FLUX.3 Klein (nativo fino a 4096, verificato dal vivo), 2048 per
        tutti gli altri motori (invariati: uscendo da FLUX.3 un valore più grande si riporta al tetto)."""
        from core import flux3_spec as S3
        top = S3.MAX_SIDE if self.frame1_tab.gen_arch.currentText() == "FLUX.3 Klein" else 2048
        for spin in (self.video_tab.width, self.video_tab.height):
            if spin.maximum() != top:
                spin.setMaximum(top)

    def _flux3_weight_ids(self, dit: str, pup: bool, te: str = "") -> list:
        """Pesi che servono al job. Text encoder: la scelta (krea2|official) o, in automatico, il Krea2 uncensored se è su disco."""
        from core import flux3_spec as S3
        te_id = S3.te_id(te, S3.disk_present)           # «ufficiale» = flux3-te se c'è, altrimenti il TE dentro Krea 2 Turbo
        ids = [dit or S3.DEFAULT_DIT, te_id, S3.VAE]
        if te_id != S3.TE:
            ids.append(S3.TE_CONFIGS)               # tokenizer e config del TE a file singolo
        if pup:
            ids += [S3.PUP_DEFAULT, "flux3-pup-configs"]
        return ids

    def _flux3_preflight(self, dit, pup, boxes, n_refs, te="") -> bool:
        """Gate d'edizione, riquadri validi, pesi su disco (altrimenti avviso + «Apri tab Pesi», nessun download)."""
        if not self._feature_gate("cap.image_flux3"):
            return False
        if boxes and not self._feature_gate("cap.flux3_boxes"):
            return False
        if pup and not self._feature_gate("cap.flux3_upsampler"):
            return False
        if boxes:
            from core import flux3_caption as C3
            errs = C3.validate(boxes, n_refs)
            if errs:
                QMessageBox.warning(self, tr("f1.flux3.boxes.btn", "Riquadri…"), "\n".join(errs))
                return False
        return self._missing_weights_notice(self._flux3_weight_ids(dit, pup, te), "FLUX.3 Klein")

    def _start_native_flux3_edit(self, dit, prompt, *, refs=(), init_image="", mask="", boxes=None, pup=True, seed=None,
                                 steps=None, loras=None, feather=0, mode="flux3_edit", label=None):
        """FLUX.3 Klein nei pannelli di editing (Integra, Componi, Colore). Con `refs` è modifica per istruzione (≤10
        immagini, «the first image»… nel prompt); con `init_image`+`mask` è una modifica per istruzione sulla foto e
        il risultato si ricompone con la maschera binarizzata (FLUX.3 NON ha un inpaint nativo)."""
        if not (prompt or "").strip() and not boxes:
            QMessageBox.information(self, tr("mw.no_mask_title", "Serve un'istruzione"),
                                    tr("mw.qwen21_no_prompt", "Scrivi COSA modificare nell'istruzione."))
            return
        refs = [r for r in (refs or []) if r]
        if not refs and not init_image:
            QMessageBox.warning(self, tr("mw.no_image_title"), tr("mw.load_photo_canvas_msg")); return
        from core import flux3_spec as S3
        dit = dit or S3.DEFAULT_DIT
        te = self.frame1_tab.flux3_te_choice("f3e")
        if not self._flux3_preflight(dit, pup, boxes, len(refs) + (1 if init_image else 0), te) or not self._ensure_native_ready():
            return
        params = {"arch": "flux3", "flux3_dit": dit, "flux3_pup": bool(pup), "flux3_te": te, "prompt": prompt or "",
                  "seed": seed if seed is not None else random.randint(0, 2**31),
                  "steps": int(steps or S3.DEFAULT_STEPS), "ref_images": refs[:S3.MAX_REFS], "loras": list(loras or []),
                  "boxes": list(boxes or [])}
        if init_image:
            params["init_image"] = init_image
        if mask:
            params["mask_path"] = mask
            params["feather"] = int(feather or 0)
        else:                                  # risultato alla risoluzione e al rapporto del canvas (mai stirato)
            v = self.video_tab.get_values()
            self._clamp_resolution(v)
            params["width"], params["height"] = int(v["width"]), int(v["height"])
        self._inpaint_mode = mode
        self._native_worker = _NativeImageWorker(params)
        self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"))
        self._log(f"FLUX.3 Klein Edit ({mode}, {len(refs)} rif., {len(boxes or [])} riquadri, "
                  f"{'maschera (composito nostro)' if mask else 'senza maschera'}, {params['steps']} passi) inviato…")
        self._enqueue(label or tr("queue.job.flux3edit", "FLUX.3 Klein Edit"), self._native_worker,
                      self._on_native_inpaint_done)

    # ── MagicQuill V2 (task 9): backend image_mq (Kontext + LoRA di controllo) ─────────────────────────────────
    _MQ_LABEL = "MagicQuill V2"

    def _mq_model_path(self, model: str) -> str:
        return model if os.path.isabs(model) else os.path.join(r"E:\AI\Models\diffusion_models", model)

    def _magicquill_component_ok(self) -> bool:
        """Se l'edizione non include il codice di MagicQuill V2 (file esclusi dalla build) → avviso chiaro, mai un crash."""
        from core import optional_components as oc
        if oc.magicquill_installed():
            return True
        QMessageBox.information(self, tr("mq.pack.title", "MagicQuill non disponibile"),
                                tr("mq.pack.missing", oc.NOT_INSTALLED_MSG))
        return False

    def generate_magicquill(self):
        """Modifica locale / rimozione / bordi guidati / colore guidato con le LoRA MagicQuill V2 su FLUX.1 Kontext."""
        if not self._feature_gate("cap.image_magicquill"):
            return
        if not self._magicquill_component_ok():
            return
        ft = self.frame1_tab
        if mm.blocked(self, ft.mq_model):
            return
        cfg = ft.get_magicquill_config()
        mode = cfg["mode"]
        if not cfg["model"]:
            QMessageBox.warning(self, tr("mw.no_model_title"), tr("f1.mq.no_model",
                "MagicQuill: scegli un checkpoint FLUX.1 Kontext nel menu del pannello.")); return
        if not ft.image_path:
            QMessageBox.warning(self, tr("mw.no_image_title"), tr("mw.load_photo_canvas_msg")); return
        if self._inpaint_pid:
            QMessageBox.information(self, tr("mw.wait_title"), tr("mw.gen_in_progress_msg")); return
        if not ft.canvas.has_mask():
            QMessageBox.information(self, tr("mw.no_mask_title"), tr("f1.mq.no_mask",
                "Disegna prima sul canvas con il pennello maschera: l'area da modificare/rimuovere/colorare, "
                "oppure i tratti dei bordi.")); return
        if mode == "local" and not cfg["prompt"]:
            QMessageBox.information(self, tr("mw.no_mask_title"), tr("f1.mq.no_prompt",
                "Scrivi nel prompt cosa deve diventare l'area disegnata.")); return
        if not self._missing_weights_notice([f"mq2-{mode}-lora"], f"{self._MQ_LABEL} ({mode})"):
            return
        if not self._ensure_native_ready():
            return
        try:
            exp = ft.canvas.export_inpaint(OUTPUT_DIR, "inpaint")
        except Exception as e:                                     # noqa: BLE001
            QMessageBox.warning(self, tr("mw.error_title", "Errore"), str(e)); return
        self._inpaint_bbox = exp.get("bbox")
        self._inpaint_before = exp.get("image")
        seed = cfg["seed"] if cfg["seed"] is not None else random.randint(0, 2**31)
        params = {"arch": "kontext", "mode": mode, "model": self._mq_model_path(cfg["model"]),
                  "image_path": exp["image"], "prompt": cfg["prompt"], "steps": cfg["steps"],
                  "guidance": cfg["guidance"], "seed": seed}
        if mode in ("local", "removal"):
            params["mask_path"] = exp["mask"]
            params["grow"] = cfg["grow"]
            params["strength_local"] = cfg["strength"]
        elif mode == "edge":
            params["erase_path" if cfg["erase"] else "edge_path"] = exp["mask"]
            params["strength_edge"] = cfg["strength"]
        else:                                                       # color: l'area disegnata prende il colore scelto
            params["mask_path"] = exp["mask"]
            params["color_rgb"] = cfg["color_rgb"]
            params["strength_color"] = cfg["strength"]
        self._inpaint_mode = "magicquill"
        self._log(f"{self._MQ_LABEL} ({mode}, NATIVO): modello={os.path.basename(cfg['model'])} · passi={cfg['steps']} · "
                  f"guida={cfg['strength']:.2f} · seed={seed}")
        self._native_worker = _NativeImageWorker(params, module="image_mq")
        self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"))
        self._enqueue(tr("queue.job.magicquill", "MagicQuill V2"), self._native_worker, self._on_native_inpaint_done)

    def _start_native_mq_foreground(self, exp, model, prompt, steps, seed, guidance):
        """«Integrazione MagicQuill» (casella puzzle_lora di Integra) = mode foreground: l'immagine di Integra è già
        preparata (oggetto incollato + alone bianco), la maschera è la zona di edit."""
        if not self._feature_gate("cap.image_magicquill"):
            return
        if not self._magicquill_component_ok():
            return
        if not self._missing_weights_notice(["mq2-puzzle-lora"], f"{self._MQ_LABEL} (integrazione)"):
            return
        if not self._ensure_native_ready():
            return
        from core.kontext_pipeline import KONTEXT_MODEL
        mp = self._mq_model_path(model or KONTEXT_MODEL)
        params = {"arch": "kontext", "mode": "foreground", "model": mp, "image_path": exp["image"],
                  "mask_path": exp["mask"], "prepared": True, "prompt": prompt or
                  "Fill in the white region naturally and adapt the foreground into the background.",
                  "steps": steps, "guidance": guidance, "seed": seed}
        self._inpaint_mode = "integrazione"
        self._log(f"Integra {self._MQ_LABEL} (foreground, NATIVO): modello={os.path.basename(mp)} · passi={steps} · seed={seed}")
        self._native_worker = _NativeImageWorker(params, module="image_mq")
        self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"))
        self._enqueue(tr("queue.job.integrate", "Image to Image (Kontext)"), self._native_worker, self._on_native_inpaint_done)

    def _start_native_klein_multiref(self, imgs, prompt, model):
        """Componi con FLUX.2 Klein: le N foto sono N riferimenti VERI (latenti separati con id di posizione propri,
        in-context), non uno stitch. Misura = quella del canvas. Passi/guidance: i default del modello (image_flux)."""
        from core import image_refs as R
        if not self._feature_gate("cap.inpaint"):
            return
        if not self._ensure_native_ready():
            return
        mp = model if os.path.isabs(model) else os.path.join(r"E:\AI\Models\diffusion_models", model)
        v = self.video_tab.get_values()
        self._clamp_resolution(v)
        refs = [p for p in imgs if p][:R.MODEL_SYNTAX["flux2_klein"]["max_images"]]
        params = {"arch": "flux2_kontext", "model": mp, "prompt": prompt or "combine the reference images into one coherent scene",
                  "image_path": refs[0], "ref_images": refs, "gen_from_refs": True,
                  "width": int(v["width"]), "height": int(v["height"]), "seed": random.randint(0, 2**31)}
        self._inpaint_mode = "composizione"
        self._inpaint_bbox = None
        self._inpaint_before = None
        self._log(f"Componi FLUX.2 Klein (multi-riferimento, {len(refs)} foto, {params['width']}×{params['height']})…")
        self._native_worker = _NativeImageWorker(params)
        self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"))
        self._enqueue(tr("queue.job.compose", "Componi (FLUX.2 Klein)"), self._native_worker, self._on_native_inpaint_done)

    def _start_native_krea2_edit(self, images, prompt, negative, grounding, variant, seed,
                                 label="Krea 2 Edit", dit=None, te=None):
        """Krea 2 Edit — editing per istruzione (single/multi) NATIVO (Krea2EditPipeline, no ComfyUI).
        `images` = lista di path (1 = single: frame 1; 2 = multi: scena=1, soggetto=2)."""
        if not self._feature_gate("cap.krea2_edit"):   # feature d'edizione (plus+); DEV = permissivo
            return
        if not self._ensure_native_ready():
            return
        imgs = [p for p in (images or []) if p]
        if not imgs:
            QMessageBox.warning(self, tr("mw.no_model_title", "Manca l'immagine"),
                                tr("mw.krea2edit_no_image", "Krea 2 Edit: carica una foto (canvas o lista Componi).")); return
        if not (prompt or "").strip():
            QMessageBox.information(self, tr("mw.no_mask_title", "Serve un'istruzione"),
                                   tr("mw.krea2edit_no_prompt", "Scrivi COSA modificare nell'istruzione.")); return
        is_raw = (variant == "raw")
        params = {
            "arch": "krea2_edit", "krea2_variant": variant,
            "prompt": prompt, "negative": negative or "",
            "grounding_px": int(grounding or 768),
            # step/guidance tarati per l'EDIT (non i default t2i): Turbo 8/CFG1, Raw 20/CFG3.
            "steps": 20 if is_raw else 8,
            "guidance": 2.0 if is_raw else 0.0,
            "seed": seed,
        }
        if dit:                                            # finetune single-file scelto in «Crea» (es. Noct V4)
            params["krea2_dit"] = dit
        if te:                                             # text encoder single-file (es. uncensored)
            params["krea2_te"] = te
        if len(imgs) == 1:
            params["image_path"] = imgs[0]
        else:
            params["images"] = imgs[:2]
        self._inpaint_mode = "krea2_edit"
        self._native_worker = _NativeImageWorker(params)
        self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"))
        self._log(f"Krea 2 Edit ({variant}, {len(imgs)} img, grounding={params['grounding_px']}) inviato…")
        self._enqueue(tr("queue.job.krea2edit", label), self._native_worker, self._on_native_inpaint_done)

    def _gen_pulid_keyframe_for_video(self, faces: str, _prompt: str):
        """One-click: foto volto (PuLID) → genera il PRIMO FRAME con l'identità → lo carica in Frame1
        come start dell'i2v. PuLID è FLUX-only, il video è LTX: keyframe→i2v porta l'identità sul video."""
        faces = (faces or "").strip()
        if not faces:
            QMessageBox.information(self, tr("vid.pulid_kf_lbl", "Volto → primo frame (PuLID)"),
                                    tr("vid.pulid_kf.no_face", "Scegli prima una o più foto del volto.")); return
        from core import native_driver
        if not self._ensure_native_ready():   # caduto? lo riavvia da sé
            return
        vals = self.video_tab.get_values()
        gc = self.frame1_tab.get_gen_config()
        model = gc.get("model") if gc.get("arch") == "flux" else None   # riusa il modello FLUX di Frame1
        prompt = (_prompt or vals.get("prompt") or "").strip() or "a person, portrait, natural lighting, high detail"
        _r64 = lambda x: max(256, round(int(x) / 64) * 64)
        params = {"arch": "flux", "prompt": prompt, "pulid_face": faces, "pulid_strength": 0.85,
                  "pulid_start_step": int(gc.get("pulid_start_step", 0)),
                  "pulid_true_cfg": float(gc.get("pulid_true_cfg", 1.0)),
                  "width": _r64(vals.get("width", 768)), "height": _r64(vals.get("height", 512)),
                  "steps": 22, "guidance": 3.5, "seed": random.randint(0, 2**31)}
        if model:
            params["model"] = model
        self._log(f"Primo frame dal volto (PuLID) — {faces.count('|') + 1} foto…")
        self._pulid_kf_worker = _NativeImageWorker(params, module="image")
        self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"))
        self._enqueue(tr("queue.job.pulid_kf", "Primo frame (PuLID)"), self._pulid_kf_worker,
                      self._on_pulid_keyframe_done)

    def _on_pulid_keyframe_done(self, res: dict):
        self._stop_native_preview()
        if res.get("status") != "success":
            self._log(f"[pulid keyframe] errore: {res.get('error', '')[:300]}")
            QMessageBox.critical(self, tr("mw.gen_error_title", "Errore"), str(res.get("error", ""))[:1000]); return
        out = (res.get("output") or [None])[0]
        if not out:
            return
        self.frame1_tab.image_path = out                     # diventa lo start dell'i2v
        try:
            self.frame1_tab.canvas.load_image(out, fit_mode="canvas")   # programmatico: niente dialog
        except Exception:                                    # noqa: BLE001
            pass
        self._log(f"[pulid keyframe] primo frame pronto: {out} — ora genera il video (i2v).")
        QMessageBox.information(self, tr("vid.pulid_kf_lbl", "Volto → primo frame (PuLID)"),
            tr("vid.pulid_kf.ready", "Primo frame con l'identità pronto e caricato. Ora premi Genera video: "
               "l'i2v lo animerà mantenendo il volto."))

    def _on_native_inpaint_done(self, res: dict):
        self._stop_native_preview()
        if res.get("status") != "success":
            self._log(f"[native inpaint] errore: {res.get('error','')[:300]}")
            QMessageBox.critical(self, tr("mw.gen_error_title", "Errore"), str(res.get("error", ""))[:1000])
            return
        out = (res.get("output") or [None])[0]
        self._log(f"[native inpaint] generato: {out}")
        # #8 denoise classico post-upscale (restauro nativo) se richiesto dalla GUI
        dn = getattr(self, "_pending_denoise", None)
        self._pending_denoise = None                         # consuma una sola volta
        if out and dn:
            try:
                from core.denoise import denoise_image
                denoise_image(out, out, strength=dn)
                self._log(f"[native] denoise applicato (forza {dn:.2f}).")
            except Exception as e:                           # noqa: BLE001
                self._log(f"[native] denoise saltato: {e}")
        # FASE D: catena restore_and_colorize — upscale ESRGAN finito → parte la colorizzazione
        pc = getattr(self, "_pending_colorize", None)
        self._pending_colorize = None                        # consuma una sola volta
        if out and pc:
            self._log(tr("mw.restore_then_colorize_log",
                         "[restauro] upscale completato → avvio colorizzazione (catena nativa)."))
            self._start_native_colorize(pc, out)
            return
        # catena upscale+refine — ESRGAN finito → rifinitura FLUX img2img a denoise basso
        pr = getattr(self, "_pending_refine", None)
        self._pending_refine = None                          # consuma una sola volta
        if out and pr:
            self._log(tr("mw.upscale_then_refine_log",
                         "[upscale] ESRGAN completato → avvio rifinitura FLUX (catena nativa)."))
            self._start_native_refine(pr, out)
            return
        if out:
            self._show_inpaint_result(out)   # dialog anteprima prima/dopo + Accetta / Revert

    def _start_native_refine(self, rcfg: dict, image_path: str):
        """Rifinitura FLUX via backend NATIVO — replica del KSampler img2img del grafo comfy
        (denoise basso sull'immagine ingrandita). Riusa il ramo 'inpaint' VALIDATO
        (FluxInpaintPipeline con strength) con maschera TUTTA BIANCA + feather 0 = img2img
        esatto, senza nuove classi server-side. Con un checkpoint FLUX.2 la guard-famiglia
        del backend passa da sola a flux2_inpaint (Klein) dai canali."""
        import os
        import tempfile
        if not self._ensure_native_ready():
            return
        try:
            from PIL import Image
            with Image.open(image_path) as im:
                size = im.size
            mask = tempfile.NamedTemporaryFile(suffix="_refine_mask.png", delete=False)
            Image.new("L", size, 255).save(mask.name)
            mask.close()
        except Exception as e:                               # noqa: BLE001
            self._log(f"[refine] maschera non creata ({e}) → mostro l'upscale senza rifinitura.")
            self._show_inpaint_result(image_path)
            return
        m = rcfg["model"]
        mp = m if os.path.isabs(m) else os.path.join(r"E:\AI\Models\diffusion_models", m)
        seed = random.randint(0, 2**31)
        params = {"model": mp, "arch": "inpaint",
                  "prompt": rcfg.get("prompt", ""), "negative": rcfg.get("negative", ""),
                  "image_path": image_path, "mask_path": mask.name,
                  "steps": int(rcfg.get("steps", 12)),
                  "guidance": float(rcfg.get("guidance", 3.5)),
                  "denoise": float(rcfg.get("denoise", 0.3)),
                  "feather": 0, "seed": seed}
        self._inpaint_mode = tr("queue.job.refine", "Rifinitura FLUX")
        self._log(f"Rifinitura FLUX (backend NATIVO, {os.path.basename(mp)}) inviata "
                  f"(steps={params['steps']}, denoise={params['denoise']:.2f}, seed={seed})…")
        self._native_worker = _NativeImageWorker(params)
        self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"))
        self._enqueue(tr("queue.job.refine", "Rifinitura FLUX"), self._native_worker,
                      self._on_native_inpaint_done)

    def _start_native_colorize(self, cfg: dict, image_path: str):
        """Colorizzazione B&W via backend NATIVO — replica del grafo ComfyUI build_colorize
        (era l'ULTIMA generazione GUI rimasta su ComfyUI). L'immagine è il riferimento e
        l'istruzione descrive i colori: arch flux2_kontext (Klein); se il checkpoint scelto
        è un FLUX.1 la guard-famiglia del backend ripiega su FluxKontextPipeline dai canali."""
        import os
        if not self._ensure_native_ready():
            return
        subject = (cfg.get("subject") or "").strip()
        style = (cfg.get("style") or "natural realistic colors").strip()
        prompt = (f"colorized version of this black and white photograph, "
                  f"{subject + ', ' if subject else ''}"
                  f"{style}, photorealistic, natural skin tones, period-accurate colors, "
                  f"high quality colorization")
        m = cfg["flux2_model"]
        mp = m if os.path.isabs(m) else os.path.join(r"E:\AI\Models\diffusion_models", m)
        seed = random.randint(0, 2**31)
        params = {"model": mp, "arch": "flux2_kontext", "prompt": prompt, "negative": "",
                  "image_path": image_path,
                  "steps": int(cfg.get("color_steps", 20)),
                  "guidance": float(cfg.get("color_guidance", 2.5)),
                  "denoise": float(cfg.get("color_denoise", 0.85)),
                  "seed": seed}
        self._inpaint_mode = tr("mw.colorize_label")
        self._log(f"Colorize (backend NATIVO, {os.path.basename(mp)}) inviato "
                  f"(steps={params['steps']}, seed={seed})…")
        self._native_worker = _NativeImageWorker(params)
        self._start_native_preview(tr("progress.preview_title", "Anteprima generazione"))
        self._enqueue(tr("queue.job.colorize", "Colorizzazione"), self._native_worker,
                      self._on_native_inpaint_done)

    def _start_native_upscale(self, model, scale, image_path=None):
        """Upscale via backend NATIVO (ESRGAN/spandrel) — niente ComfyUI."""
        from core import native_driver
        if not self._ensure_native_ready():   # caduto? lo riavvia da sé
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
        _ft = self.frame1_tab
        if mm.blocked(self, _ft.up_model, _ft.up_refine_model if _ft.up_mode.currentIndex() == 1 else None):
            return
        cfg = self.frame1_tab.get_upscale_config()
        if not cfg.get("model"):
            QMessageBox.warning(self, tr("mw.no_upscale_model_title"),
                                tr("mw.no_upscale_model_msg")); return
        if cfg.get("refine") and not cfg.get("refine_model"):
            QMessageBox.warning(self, tr("mw.refine_model_missing_title"),
                                tr("mw.refine_model_missing_msg")); return
        # ── BACKEND NATIVO (ESRGAN via spandrel): upscale puro senza ComfyUI; con refine
        #    la rifinitura FLUX segue come 2° job in catena (_pending_refine, stesso pattern
        #    di _pending_colorize) — era l'ultimo ramo GUI che ancora chiedeva ComfyUI.
        from core import backend
        if backend.is_native("image"):
            self._inpaint_before = self.frame1_tab.image_path
            if cfg.get("refine"):
                self._pending_refine = {
                    "model": cfg["refine_model"],
                    "denoise": float(cfg.get("refine_denoise", 0.3)),
                    "steps": int(cfg.get("refine_steps", 12)),
                    "guidance": float(cfg.get("refine_guidance", 3.5)),
                    "prompt": self.frame1_tab.get_values().get("prompt_positive", ""),
                    "negative": self.frame1_tab.get_values().get("prompt_negative", ""),
                }
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

    @staticmethod
    def _load_scaled_image(path: str, box: int = 620):
        """Decodifica un'immagine GIÀ SCALATA al box richiesto (QImageReader.setScaledSize).
        Pensata per girare in un THREAD: un PNG upscalato 4x arriva a centinaia di MB e
        decodificarlo pieno sul thread GUI congelava l'app (visto live 22/07 dopo un upscale)."""
        from PySide6.QtGui import QImage, QImageReader
        try:
            r = QImageReader(path)
            r.setAutoTransform(True)
            sz = r.size()
            if sz.isValid() and (sz.width() > box or sz.height() > box):
                sz.scale(box, box, Qt.KeepAspectRatio)
                r.setScaledSize(sz)
            img = r.read()
            return None if img.isNull() else img
        except Exception:                                   # noqa: BLE001
            return None

    def _show_inpaint_result(self, path: str):
        """Prepara le anteprime (decode+scala) in un THREAD, poi mostra il dialog: la GUI non
        blocca mai, nemmeno con risultati upscalati giganti."""
        import threading
        _before = self._inpaint_before

        def _prep():
            res = self._load_scaled_image(path)
            bef = self._load_scaled_image(_before) if _before else None
            # torna sul thread GUI per il dialog via SIGNAL (queued): QPixmap/QDialog SOLO lì
            self._result_prev_ready.emit(path, res, bef)
        threading.Thread(target=_prep, daemon=True).start()

    def _show_inpaint_result_ready(self, path: str, res_img, bef_img):
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        dlg = QDialog(self); dlg.setWindowTitle(tr("mw.result_dialog_title").format(mode=self._inpaint_mode))
        v = QVBoxLayout(dlg)
        before_pix = QPixmap.fromImage(bef_img) if bef_img is not None else None
        result_pix = QPixmap.fromImage(res_img) if res_img is not None else QPixmap(path)
        lbl = _HoldCompareLabel(result_pix, before_pix)
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
        # Segmentazione NATIVA (GroundingDINO+SAM2 sul server nativo) → NON serve ComfyUI. Il requisito comfy
        # vale solo in modalità comfy. Copre sia "selezione testo" sia "seleziona oggetto" (stessa precond).
        if not backend.is_native("segment") and not self.driver.is_up():
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
            ok, msg = ensure_model(cfg["model"], log=self._log,
                                   confirm=lambda n, mb: QMessageBox.question(
                                       self, tr("weights.missing.title", "Pesi mancanti"),
                                       tr("weights.missing.offer_one",
                                          "Il modello {name} (~{mb}MB) non è presente. Scaricarlo ora?")
                                       .format(name=n, mb=mb)) == QMessageBox.Yes)
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
        from core import backend as _backend              # #95 I2: il restauro gira nativo → non richiede ComfyUI up
        _restore_native = (cfg.get("operation") in ("restore", "colorize", "restore_and_colorize")
                           and _backend.is_native("image"))
        if not _restore_native and not self.driver.is_up():
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
            # #8 denoise (classico Non-Local Means) come post-process dell'upscale nativo.
            self._pending_denoise = (float(cfg.get("denoise_strength", 0.2))
                                     if cfg.get("denoise") else None)
            self._start_native_upscale(cfg["upscale_model"], cfg["target_scale"], image_path=image_path)
            return
        if op in ("colorize", "restore_and_colorize") and not cfg.get("flux2_model"):
            from PySide6.QtWidgets import QMessageBox            # nessun modello FLUX.2 reale: non inviare il placeholder
            QMessageBox.warning(self, tr("mw.flux2_model_missing_title"),
                                tr("mw.flux2_model_missing_integrate_msg"))
            return
        # ── BACKEND NATIVO (FASE D 17/08): colorize = edit-per-riferimento (Klein/Kontext, la
        #    guard-famiglia del backend sceglie dai canali del checkpoint); restore_and_colorize =
        #    catena upscale ESRGAN nativo → colorize sull'output (_pending_colorize, come il denoise).
        if op in ("colorize", "restore_and_colorize") and backend.is_native("image"):
            self._inpaint_before = image_path
            if op == "restore_and_colorize":
                self._pending_colorize = dict(cfg)
                self._start_native_upscale(cfg["upscale_model"], cfg.get("target_scale", 2.0),
                                           image_path=image_path)
            else:
                self._start_native_colorize(cfg, image_path)
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
                    steps=cfg["color_steps"],
                    denoise=cfg.get("color_denoise", 0.85),
                    guidance=cfg.get("color_guidance", 2.5))
                label = tr("mw.colorize_label")
            else:  # restore_and_colorize
                prompt = self.restorer.build_restore_and_colorize(
                    img_name,
                    upscale_model=cfg["upscale_model"],
                    flux2_model=cfg["flux2_model"],
                    subject=cfg["subject"],
                    style=cfg["style"],
                    seed=seed,
                    steps=cfg["color_steps"],
                    denoise=cfg.get("color_denoise", 0.85),
                    guidance=cfg.get("color_guidance", 2.5),
                    target_scale=cfg.get("target_scale", 2.0))
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
    def _trellis_native(self, cfg: dict, image_path):
        """TRELLIS 3D NATIVO (venv isolato, subprocess) — immagine→GLB. Zero ComfyUI.
        `image_path`: str o LISTA di path (multi-vista: più foto dello stesso oggetto)."""
        from PySide6.QtWidgets import QMessageBox
        from core import trellis_service
        _primary = image_path[0] if isinstance(image_path, (list, tuple)) else image_path
        errs = trellis_service.preflight(_primary)
        if errs:
            QMessageBox.warning(self, "TRELLIS", "\n".join(errs)); return
        seed = random.randint(0, 2**31 - 1) if cfg.get("seed_random", True) else int(cfg.get("seed", 0))
        name = f"trellis_{Path(_primary).stem}_{seed}"
        steps = int(cfg.get("steps", 12) or 12)
        _trellis_tail = []                                   # ring-buffer per l'errore VERO del subprocess
        # LOG SU FILE (sopravvive a un eventuale crash dell'app): ogni riga del subprocess TRELLIS viene
        # scritta subito e FLUSHATA su output/3d/trellis_last.log → così vediamo l'errore reale anche se
        # l'app muore. (write+flush per riga: nessuna perdita al crash.)
        _tlog_path = Path(OUTPUT_DIR) / "3d" / "trellis_last.log"
        try:
            _tlog_path.parent.mkdir(parents=True, exist_ok=True)
            _tlog_fh = open(_tlog_path, "w", encoding="utf-8", buffering=1)
            _tlog_fh.write(f"[trellis] img={image_path} seed={seed} steps={steps} "
                           f"tex={cfg.get('texture_size')} faces={cfg.get('target_face_count')}\n"); _tlog_fh.flush()
        except Exception:                                    # noqa: BLE001
            _tlog_fh = None
        self._log(f"[trellis] log completo del subprocess in: {_tlog_path}")

        def _tlog(line):
            self._log(line)
            if line:
                _trellis_tail.append(str(line))
                if len(_trellis_tail) > 30:
                    _trellis_tail.pop(0)
                if _tlog_fh is not None:
                    try:
                        _tlog_fh.write(str(line) + "\n"); _tlog_fh.flush()
                        os.fsync(_tlog_fh.fileno())          # forza su disco: sopravvive al crash GPU/app
                    except Exception:                        # noqa: BLE001
                        pass
        job = trellis_service.make_job(
            image_path, name, seed=seed, ss_steps=steps, slat_steps=steps,
            simplify=float(cfg.get("simplify", 0.95) or 0.95),
            texture_size=int(cfg.get("texture_size", 1024) or 1024),
            target_faces=int(cfg.get("target_face_count", 0) or 0),   # #95: facce target dal tab (era ignorato)
            # Parametri GUI ora REALMENTE usati dal backend nativo TRELLIS.2 (prima ignorati):
            resolution=str(cfg.get("resolution") or "") or None,
            max_tokens=int(cfg.get("max_tokens", 0) or 0) or None,
            guidance=float(cfg.get("guidance", -1.0)),
            geo_steps=int(cfg.get("steps", 0) or 0) or None,
            tex_guidance=float(cfg.get("tex_guidance", -1.0)),
            tex_steps=int(cfg.get("tex_steps", 0) or 0) or None,
            texture=bool(cfg.get("texture", True)),
            remove_bg=bool(cfg.get("remove_background", True)),
            attn=str(cfg.get("attn_backend") or "auto"),
            precision=str(cfg.get("precision") or "auto"),
            bg_color=str(cfg.get("background_color") or "black"),
            on_log=_tlog)
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
                        _why = "\n".join(_trellis_tail[-14:]) or "(nessun log)"
                        self.done.emit({"status": "error",
                                        "error": f"TRELLIS terminato (exit {rc['v']}), output mancante.\n{_why}"})
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
        try:                                              # carica nel visore 3D del tab (show_result, NON load_model che è del viewer)
            if glb and hasattr(self, "trellis_tab") and hasattr(self.trellis_tab, "show_result"):
                self.trellis_tab.show_result(glb)
        except Exception:  # noqa: BLE001
            pass
        try:                                              # #95: parità col ramo Comfy → offri la mesh al tab Rig
            if glb and hasattr(self, "rig_tab") and hasattr(self.rig_tab, "set_last_mesh"):
                self.rig_tab.set_last_mesh(glb)
        except Exception:  # noqa: BLE001
            pass

    def _on_rig_native_done(self, res: dict):
        """Done handler del rig NATIVO: mostra nel visore Rig + handoff al tab Animazione (come il ramo Comfy)."""
        from PySide6.QtWidgets import QMessageBox
        if res.get("status") != "success":
            self._log(f"[rig] errore: {res.get('error', '')[:300]}")
            QMessageBox.critical(self, "Rig 3D", str(res.get("error", ""))[:1000]); return
        glb = (res.get("output") or [None])[0]
        self._log(f"[rig] modello riggato pronto: {glb}")
        try:                                              # visore del tab Rig
            if glb and hasattr(self, "rig_tab") and hasattr(self.rig_tab, "show_result"):
                self.rig_tab.show_result(glb)
        except Exception:  # noqa: BLE001
            pass
        try:                                              # handoff al tab Animazione (parità col ramo ComfyUI)
            if glb and hasattr(self, "anim_tab") and hasattr(self.anim_tab, "set_last_rig"):
                self.anim_tab.set_last_rig(glb)
        except Exception:  # noqa: BLE001
            pass

    # ── Avatar parlante (LongCat 1.5) ─────────────────────────────────────────────────────────────
    def generate_avatar(self, cfg: dict):
        if cfg.get("engine") == "minimax_h3":
            return self._generate_avatar_h3(cfg)
        if not self._feature_gate("cap.avatar_longcat"):
            return
        from core import longcat_service as lc
        if not lc.is_installed():
            QMessageBox.information(self, tr("tab.avatar", "Avatar"), tr(
                "avatar.err.not_installed",
                "Il motore LongCat Avatar non è installato. Installalo con:\n"
                "python scripts/install_longcat_avatar.py"))
            return
        prec = cfg.get("precision", "int8")
        try:
            from core import ensure_weights as ew
            ids = lc.weight_ids(prec)
            if ew.missing(ids) and not ew.offer_download_gui(self, ids, tr("tab.avatar", "Avatar")):
                return
        except Exception:                                # noqa: BLE001 — il preflight del worker protegge comunque
            pass
        errs = lc.preflight(cfg)
        if errs:
            QMessageBox.warning(self, tr("tab.avatar", "Avatar"), "\n".join(errs))
            return
        self._avatar_cfg = dict(cfg)
        self.avatar_tab.set_busy(True)
        self.avatar_tab.append_log(tr("avatar.log.start", "Avvio LongCat Avatar ({p})…").format(p=prec))
        self._log(f"[avatar] LongCat {cfg.get('mode')} {cfg.get('resolution')} {prec}")
        self._avatar_worker = _LongCatWorker(cfg)
        self._enqueue(tr("queue.job.avatar", "Avatar parlante"), self._avatar_worker, self._on_avatar_done,
                      on_progress=self.avatar_tab.append_log)

    def _generate_avatar_h3(self, cfg: dict):
        """2° motore avatar: MiniMax H3 (voce + video insieme, stesso worker del tab Video)."""
        if not self._feature_gate("cap.avatar_h3"):
            return
        from core import minimax_avatar as ma
        from core import minimax_h3_spec as S
        params = ma.build_params(cfg)
        try:
            from core import ensure_weights as ew
            ids = list(dict.fromkeys(S.weight_ids(params).values()))
            if ew.missing(ids) and not ew.offer_download_gui(self, ids, "MiniMax H3"):
                return
        except ValueError as e:
            QMessageBox.warning(self, tr("tab.avatar", "Avatar"), str(e))
            return
        except Exception:                                # noqa: BLE001 — il preflight protegge comunque
            pass
        errs = ma.preflight(params)
        if errs:
            QMessageBox.warning(self, tr("tab.avatar", "Avatar"), "\n".join(errs))
            return
        self._avatar_cfg = dict(cfg)
        self.avatar_tab.set_busy(True)
        self.avatar_tab.append_log(tr("avatar.log.start_h3", "Avvio MiniMax H3 ({p})…").format(p=params["h3_profile"]))
        self._log(f"[avatar] MiniMax H3 {'Ref2VA' if params.get('references') else 'FL2VA'} "
                  f"{params['num_frames']} fotogrammi")
        self._avatar_worker = _LongCatWorker(params, service=ma)
        self._enqueue(tr("queue.job.avatar", "Avatar parlante"), self._avatar_worker, self._on_avatar_done,
                      on_progress=self.avatar_tab.append_log)

    def _on_avatar_done(self, res: dict):
        if res.get("status") != "success":
            self.avatar_tab.set_busy(False)
            self.avatar_tab.append_log("✗ " + str(res.get("error", ""))[:2000])
            self._log(f"[avatar] errore: {str(res.get('error', ''))[:300]}")
            QMessageBox.critical(self, tr("mw.gen_error_title", "Errore"), str(res.get("error", ""))[:1500])
            return
        out = (res.get("output") or [None])[0]
        self._last_video_path = out
        cfg = getattr(self, "_avatar_cfg", {}) or {}
        if cfg.get("face_restore") and out:
            self.avatar_tab.append_log(tr("avatar.log.face", "Restauro volto…"))
            import random as _r
            params = {"image_path": out, "model": cfg["face_restore"], "fidelity": 0.6,
                      "seed": _r.randint(0, 2**31)}
            self._avatar_face_worker = _NativeImageWorker(params, module="video_face")
            self._enqueue(tr("queue.job.video_face", "Restauro volti video"), self._avatar_face_worker,
                          lambda r, _o=out: self._avatar_after_face(r, _o))
            return
        self._avatar_after_face({"status": "success", "output": [out]}, out)

    def _avatar_after_face(self, res: dict, fallback: str):
        ok = res.get("status") == "success"
        out = (res.get("output") or [None])[0] if ok else fallback
        if not ok:
            self.avatar_tab.append_log("✗ " + tr("avatar.log.face_err",
                                                 "restauro volto fallito: tengo il video originale"))
        cfg = getattr(self, "_avatar_cfg", {}) or {}
        if cfg.get("rife") and out:
            fluid = str(Path(out).with_name(Path(out).stem + "_rife2x.mp4"))
            self.avatar_tab.append_log(tr("avatar.log.rife", "Fluidifico a 50 fps (RIFE)…"))
            self._avatar_rife = _RifeWorker(out, fluid, 2)
            self._enqueue(tr("queue.job.rife", "Interpolazione FPS (RIFE)"), self._avatar_rife,
                          lambda r, _o=out: self._avatar_final(r, _o))
            return
        self._avatar_final({"status": "success", "output": [out]}, out)

    def _avatar_final(self, res: dict, fallback: str):
        ok = res.get("status") == "success"
        out = (res.get("output") or [None])[0] if ok else fallback
        if not ok:
            self.avatar_tab.append_log("✗ " + tr("avatar.log.rife_err", "RIFE fallito: tengo il video a 25 fps"))
        self._last_video_path = out
        self.avatar_tab.set_busy(False)
        self.avatar_tab.show_result(out)
        self.avatar_tab.append_log(tr("avatar.log.done", "✓ Video pronto: {f}").format(f=out))
        self._log(f"[avatar] pronto: {out}")

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

        # MULTI-VISTA: viste extra dal tab (immagini aggiuntive dello stesso oggetto). Il primario
        # risolto sopra (file/canvas/masked) resta la vista 1; le extra si accodano. Solo nativo.
        _views = [p for p in (cfg.get("image_paths") or [])
                  if p and not str(p).startswith("<") and os.path.exists(str(p)) and str(p) != image_path]

        # BACKEND NATIVO (TRELLIS via venv isolato, zero ComfyUI) se installato
        try:
            from core import trellis_service
            if trellis_service.is_installed():
                _inp = [image_path] + _views if _views else image_path
                if _views:
                    self._log(f"[trellis] multi-vista: {1 + len(_views)} immagini")
                return self._trellis_native(cfg, _inp)
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
        _fmt = cfg.get("file_format", "glb")
        out = str(rig_service.OUT_DIR / rig_service._safe(name) / f"rigged.{_fmt}")
        job = rig_service.make_job(                          # #95: inoltra la config del tab Rig (era ignorata)
            mesh_path, name,
            template=cfg.get("skeleton_template", "Mixamo"), file_format=_fmt,
            top_k=int(cfg.get("top_k", 50)), top_p=float(cfg.get("top_p", 0.95)),
            temperature=float(cfg.get("temperature", 1.0)),
            repetition_penalty=float(cfg.get("repetition_penalty", 1.0)),
            num_beams=int(cfg.get("num_beams", 1)),
            group_per_vertex=int(cfg.get("group_per_vertex", 4)), on_log=self._log)
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
        self._enqueue(tr("queue.job.rig", "Rig 3D"), self._rig_worker, self._on_rig_native_done)

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
        """Ricarica il catalogo LoRA dopo un ingest completato + RIFÀ l'inventario dropdown
        (FASE D: prima l'import Civitai aggiornava SOLO le LoRA — i checkpoint importati
        comparivano nei menu solo al riavvio)."""
        try:
            self.lora_tab.catalog.load()
            self.lora_tab.refresh()
            self._log("[Modelli] Catalogo LoRA ricaricato dopo ingest.")
        except Exception as e:
            self._log(f"[warn] Ricarica catalogo: {e}")
        self.refresh_model_inventory()

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
