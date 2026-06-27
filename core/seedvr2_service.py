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
VENV_PY = CLI_DIR / "venv" / "Scripts" / "python.exe"
CLI = CLI_DIR / "inference_cli.py"
MODEL_DIR = Path(r"E:\AI\Models\SeedVR2")
DEFAULT_MODEL = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"   # ~3GB, qualità buona, entra largo su 32GB


def is_installed() -> bool:
    return VENV_PY.exists() and CLI.exists()


def _vram_free_mb() -> int:
    try:
        import torch
        if torch.cuda.is_available():
            free, _tot = torch.cuda.mem_get_info()
            return int(free / 1048576)
    except Exception:                                       # noqa: BLE001
        pass
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


def preflight() -> list[str]:
    errs = []
    if not CLI.exists():
        errs.append(f"SeedVR2 CLI assente: clona numz/ComfyUI-SeedVR2_VideoUpscaler in {CLI_DIR}.")
    if not VENV_PY.exists():
        errs.append(f"venv SeedVR2 assente: crea {CLI_DIR}\\venv e installa requirements.txt "
                    f"(usa scripts/install_seedvr2.py).")
    return errs


def upscale_video(inp: str, out: str, *, resolution: int = 1080, model: str = "", batch_size: int = 5,
                  chunk_size: int = 0, temporal_overlap: int = 0, color: str = "lab",
                  cuda_device: str = "0", on_log=None) -> str:
    """Super-risolve un video. batch_size deve essere 4n+1 (1,5,9,...). chunk_size>0 = streaming
    memory-bounded per video lunghi. Ritorna il path di output (mp4)."""
    errs = preflight()
    if errs:
        raise RuntimeError("; ".join(errs))
    model = model or DEFAULT_MODEL
    blocks, offload = _auto_block_swap(model)
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
    env = dict(os.environ, PYTHONPATH=str(CLI_DIR))
    proc = subprocess.Popen(cmd, cwd=str(CLI_DIR), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:                                # stream log
        if on_log:
            on_log(line.rstrip())
    proc.wait()
    if proc.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"SeedVR2 fallito (exit {proc.returncode}).")
    return out
