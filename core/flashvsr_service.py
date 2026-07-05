"""FlashVSR v1.1 super-risoluzione VIDEO (x4, streaming) — STANDALONE, niente ComfyUI.

Usa il repo OpenImagingLab/FlashVSR (clonato in E:\\AI\\Tools\\FlashVSR, pipeline diffsynth) via
subprocess in venv isolato, attraverso il runner parametrico `workers/flashvsr_run.py` eseguito con
cwd=examples/WanVSR (per gli import `from utils...`/`from diffsynth...`). Pesi in E:\\AI\\Models\\FlashVSR.
L'audio originale viene rimuxato a valle (FlashVSR processa solo i frame). x4 fisso, streaming buffer
per video lunghi (memory-bounded).

API: is_installed() -> bool · preflight() -> list[str] · upscale_video(inp, out, *, scale=4.0,
     seed=0, sparse=2.0, local_range=11, on_log=None) -> str
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_DIR = Path(r"E:\AI\Tools\FlashVSR")
WANVSR_DIR = REPO_DIR / "examples" / "WanVSR"
_CONS = Path(r"E:\AI\Tools\inference_server_venv\Scripts\python.exe")   # venv consolidato (#61)
VENV_PY = _CONS if _CONS.exists() else REPO_DIR / "venv" / "Scripts" / "python.exe"
MODEL_DIR = Path(r"E:\AI\Models\FlashVSR\FlashVSR-v1.1")
RUNNER = Path(__file__).resolve().parents[1] / "workers" / "flashvsr_run.py"
REPO = "https://github.com/OpenImagingLab/FlashVSR"


def is_installed() -> bool:
    return VENV_PY.exists() and (WANVSR_DIR / "infer_flashvsr_v1.1_tiny.py").exists() and RUNNER.exists()


def preflight() -> list[str]:
    errs = []
    if not (WANVSR_DIR / "infer_flashvsr_v1.1_tiny.py").exists():
        errs.append(f"FlashVSR assente: clona {REPO} in {REPO_DIR}.")
    if not VENV_PY.exists():
        errs.append(f"venv FlashVSR assente: usa scripts/install_flashvsr.py.")
    if not (MODEL_DIR / "diffusion_pytorch_model_streaming_dmd.safetensors").exists():
        errs.append(f"Pesi FlashVSR assenti in {MODEL_DIR}.")
    return errs


def _mux_audio(src_with_audio: str, video_no_audio: str, out: str) -> str:
    """Rimette l'audio originale (FlashVSR processa solo i frame)."""
    import shutil
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-i", video_no_audio, "-i", src_with_audio,
                        "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0?",
                        "-shortest", out, "-y"], check=True)
        return out
    except Exception:                                       # noqa: BLE001
        shutil.copyfile(video_no_audio, out)               # sorgente senza audio → copia
        return out


def upscale_video(inp: str, out: str, *, scale: float = 4.0, seed: int = 0, sparse: float = 2.0,
                  local_range: int = 11, on_log=None) -> str:
    """Super-risolve un video x4 con FlashVSR (streaming). Ritorna il path mp4 (audio rimuxato)."""
    errs = preflight()
    if errs:
        raise RuntimeError("; ".join(errs))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = str(Path(out).with_name("_flashvsr_noaudio_" + Path(out).name))
    cmd = [str(VENV_PY), "-u", str(RUNNER), inp, tmp, str(MODEL_DIR),
           str(scale), str(seed), str(sparse), str(local_range)]
    env = dict(os.environ, PYTHONPATH=str(WANVSR_DIR) + os.pathsep + str(REPO_DIR))
    proc = subprocess.Popen(cmd, cwd=str(WANVSR_DIR), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        if on_log:
            on_log(line.rstrip())
    proc.wait()
    if proc.returncode != 0 or not os.path.exists(tmp):
        raise RuntimeError(f"FlashVSR fallito (exit {proc.returncode}).")
    res = _mux_audio(inp, tmp, out)
    try:
        os.remove(tmp)
    except Exception:                                      # noqa: BLE001
        pass
    # #106 watermark obbligatorio: fine del job FlashVSR (file consegnato da GUI e REST
    # /video/enhance), UNA volta per catena — l'eventuale RIFE a valle NON ri-marchia (banda
    # baked nei frame). RuntimeError (forced non applicabile) → errore del job.
    try:
        from core import watermark
        res = watermark.apply_to_file(res)
    except RuntimeError:
        raise
    except Exception:                                      # noqa: BLE001
        pass
    return res
