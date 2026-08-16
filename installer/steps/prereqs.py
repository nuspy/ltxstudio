"""Verifica dei prerequisiti runtime (idempotente): python / driver NVIDIA / torch / ffmpeg.
ComfyUI NON è più un prerequisito: il backend è nativo (lo shim comfy_compat è un extra
opzionale installabile dal tab «Importa modelli»)."""
from __future__ import annotations

import shutil
import subprocess
import sys


def _run(cmd) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return r.returncode == 0, (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def check_python() -> dict:
    ok = sys.version_info >= (3, 10)
    return {"label": "Python 3.10+", "ok": ok, "detail": sys.version.split()[0]}


def check_nvidia() -> dict:
    ok, out = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    return {"label": "Driver NVIDIA + CUDA", "ok": ok, "detail": out or "nvidia-smi assente"}


def check_torch() -> dict:
    try:
        import torch  # type: ignore
        cuda = getattr(torch.version, "cuda", None)
        ok = bool(torch.cuda.is_available())
        return {"label": "PyTorch + CUDA", "ok": ok, "detail": f"torch {torch.__version__}, cuda {cuda}"}
    except Exception:
        return {"label": "PyTorch + CUDA", "ok": False, "detail": "torch non importabile"}


def check_ffmpeg() -> dict:
    p = shutil.which("ffmpeg")
    return {"label": "ffmpeg", "ok": bool(p), "detail": p or "assente"}


def check_all() -> dict:
    return {
        "python": check_python(),
        "nvidia": check_nvidia(),
        "torch": check_torch(),
        "ffmpeg": check_ffmpeg(),
    }
