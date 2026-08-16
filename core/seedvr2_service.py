"""SeedVR2 super-risoluzione VIDEO — STANDALONE, niente ComfyUI.

Usa il CLI standalone `inference_cli.py` del port Windows-compatibile (E:\\AI\\Tools\\SeedVR2,
numz/ComfyUI-SeedVR2_VideoUpscaler) come LIBRERIA via subprocess in venv isolato (come cosyvoice_service).
Pipeline 4-fasi (encode VAE → upscale DiT → decode VAE → color-correct) con block-swap + streaming chunk
per video lunghi. block_swap tarato DINAMICAMENTE sulla VRAM libera (mai valori fissi).

API:
  is_installed() -> bool
  preflight() -> list[str]
  upscale_video(inp, out, *, resolution=1080, model="", batch_size=5, chunk_size=0,
                temporal_overlap=0, color="lab", cuda_device="0", on_log=None) -> str
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

CLI_DIR = Path(r"E:\AI\Tools\SeedVR2")
_CONS = Path(r"E:\AI\Tools\inference_server_venv\Scripts\python.exe")   # venv consolidato (#61 — f5_tts/seedvr2 validati su transformers 4.57 del host)
VENV_PY = _CONS if _CONS.exists() else CLI_DIR / "venv" / "Scripts" / "python.exe"
CLI = CLI_DIR / "inference_cli.py"
# FASE F weights-unification: dir DAL registro unico (parent della voce seedvr2-3b-fp8);
# fallback al letterale storico se il catalogo non risolve.
try:
    from core import model_paths as _mp
    _p = _mp.path_of("seedvr2-3b-fp8")
    MODEL_DIR = _p.parent if _p else Path(r"E:\AI\Models\SeedVR2")
except Exception:                                        # noqa: BLE001
    MODEL_DIR = Path(r"E:\AI\Models\SeedVR2")
DEFAULT_MODEL = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"   # ~3GB, qualità buona, entra largo su 32GB


def is_installed() -> bool:
    return VENV_PY.exists() and CLI.exists()


def _vram_free_mb() -> int:
    # via nvidia-smi/NVML (core/gpu_mem): questo modulo gira anche nel processo GUI, che non deve
    # MAI creare un contesto CUDA (torch.cuda.mem_get_info lo creava — regola app-no-cuda-context).
    try:
        from core.gpu_mem import free_vram_mb
        return int(free_vram_mb())
    except Exception:                                       # noqa: BLE001
        return 0


def _auto_block_swap(model: str) -> tuple[int, str]:
    """block_swap dinamico per la VRAM libera (3B max 32 blocchi). Ritorna (blocks, offload_device)."""
    free = _vram_free_mb()
    if free == 0 or free >= 20000:        # >=20GB liberi → niente swap
        return 0, "none"
    if free >= 12000:
        return 8, "cpu"
    if free >= 8000:
        return 16, "cpu"
    return 32, "cpu"                       # <8GB → swap massimo


def preflight(model: str = "") -> list[str]:
    errs = []
    if not CLI.exists():
        errs.append(f"SeedVR2 CLI assente: clona numz/ComfyUI-SeedVR2_VideoUpscaler in {CLI_DIR}.")
    if not VENV_PY.exists():
        errs.append(f"venv SeedVR2 assente: crea {CLI_DIR}\\venv e installa requirements.txt "
                    f"(usa scripts/install_seedvr2.py).")
    # weights-unification C: il preflight controllava CLI+venv ma NON i pesi → "SeedVR2 fallito
    # (exit N)" criptico al primo uso. Ora l'assenza esce PRIMA, con la voce di catalogo.
    _m = MODEL_DIR / (model or DEFAULT_MODEL)
    if not _m.exists():
        errs.append(f"Pesi SeedVR2 assenti ({_m.name}): scaricali dal tab «Pesi modelli» "
                    "(voci seedvr2-3b-fp8 + seedvr2-vae, ~3.7GB).")
    return errs


def upscale_video(inp: str, out: str, *, resolution: int = 1080, model: str = "", batch_size: int = 5,
                  chunk_size: int = 0, temporal_overlap: int = 0, color: str = "lab",
                  cuda_device: str = "0", block_swap: int = 0, timeout_s: float | None = None,
                  on_log=None) -> str:
    """Super-risolve un video. batch_size deve essere 4n+1 (1,5,9,...). chunk_size>0 = streaming
    memory-bounded per video lunghi. block_swap>0 = override manuale (0 = auto sulla VRAM libera).
    `timeout_s` (o env LTX_SEEDVR2_TIMEOUT_S, default 3600) termina il processo se sfora (#156).
    Ritorna il path di output (mp4)."""
    errs = preflight(model)
    if errs:
        raise RuntimeError("; ".join(errs))
    model = model or DEFAULT_MODEL
    # LOCK GPU inter-processo PRIMA del free (altrimenti un altro processo può prendersi la VRAM
    # appena liberata). Rilascio nel finally del watchdog (mux/stego a valle sono CPU-only).
    _gl = None
    try:
        from core import gpu_lock as _gl
        _gl.acquire("upscale:seedvr2",
                    on_wait=lambda h: on_log and on_log(f"[gpu] occupata da {h.get('tag', '?')} — attendo…"))
    except Exception:                                       # noqa: BLE001
        _gl = None
    # COORDINAMENTO (mancava): libera gli altri motori e aspetta il rilascio EFFETTIVO PRIMA di
    # _auto_block_swap — la taratura leggeva la VRAM con FLUX/LTX ancora residenti e si
    # auto-castrava (swap massimo) o partiva nel buco del rilascio (shared-RAM).
    try:
        from core import mem_coordinator as _mc
        _mc.free_and_wait("video", need_mb=12000, wait_s=30)
    except Exception:                                       # noqa: BLE001
        pass
    blocks, offload = _auto_block_swap(model)
    if block_swap > 0:                       # override manuale dalla GUI (altrimenti resta auto)
        blocks = block_swap
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if batch_size % 4 != 1:                # forza 4n+1
        batch_size = (batch_size // 4) * 4 + 1
    cmd = [str(VENV_PY), "-u", str(CLI), inp, "--output", out, "--output_format", "mp4",
           "--model_dir", str(MODEL_DIR), "--dit_model", model, "--resolution", str(resolution),
           "--batch_size", str(batch_size), "--color_correction", color,
           "--cuda_device", cuda_device, "--video_backend", "ffmpeg"]
    if chunk_size > 0:
        cmd += ["--chunk_size", str(chunk_size), "--temporal_overlap", str(temporal_overlap or 3)]
    if blocks > 0:
        cmd += ["--blocks_to_swap", str(blocks), "--dit_offload_device", offload, "--vae_offload_device", offload]
    if timeout_s is None:
        timeout_s = float(os.environ.get("LTX_SEEDVR2_TIMEOUT_S", "3600") or 0)
    env = dict(os.environ, PYTHONPATH=str(CLI_DIR))
    try:
        proc = subprocess.Popen(cmd, cwd=str(CLI_DIR), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    except Exception:
        if _gl:
            _gl.release()
        raise
    killed = {"timeout": False}
    wd = None
    if timeout_s > 0:
        import threading
        def _kill():
            killed["timeout"] = True
            try:
                proc.kill()
            except Exception:                                # noqa: BLE001
                pass
        wd = threading.Timer(timeout_s, _kill); wd.daemon = True; wd.start()
    try:
        for line in proc.stdout:                            # stream log
            if on_log:
                on_log(line.rstrip())
        proc.wait()
    finally:
        if wd:
            wd.cancel()
        if _gl:
            _gl.release()
    if killed["timeout"]:
        raise RuntimeError(f"SeedVR2: timeout ({int(timeout_s)}s) — processo terminato.")
    if proc.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"SeedVR2 fallito (exit {proc.returncode}).")
    try:  # #107 firma invisibile di provenienza (default ON) — PRIMA del watermark visibile
        from core import stego
        if stego.enabled():
            stego.sign_file(out)
    except Exception:                                        # noqa: BLE001
        pass
    # #106 watermark obbligatorio: applicato QUI (fine del job SeedVR2 = file consegnato da GUI e
    # REST /video/enhance) e UNA sola volta per catena — se la GUI accoda RIFE dopo, core/rife.py
    # NON ri-marchia (la banda resta baked nei frame interpolati). RuntimeError → errore del job.
    try:
        from core import watermark
        out = watermark.apply_to_file(out)
    except RuntimeError:
        raise
    except Exception:                                        # noqa: BLE001
        pass
    return out
