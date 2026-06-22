"""Verifica dei prerequisiti (idempotente) + individuazione/scelta di ComfyUI."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


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


def resolve_comfy_root(override: str | None) -> str | None:
    """Override → catalog_paths salvato → detect → chiede all'utente. Persiste la scelta."""
    try:
        from core import catalog_paths
    except Exception:
        catalog_paths = None
    if override and (Path(override) / "main.py").exists():
        _persist_comfy(catalog_paths, override)
        return override
    if catalog_paths is not None:
        saved = catalog_paths.load().get("comfy_root")
        if saved and (Path(saved) / "main.py").exists():
            return saved
        det = catalog_paths.detect_comfy_root()
        if det:
            _persist_comfy(catalog_paths, str(det))
            return str(det)
    # chiedi all'utente (ATTENZIONE: ComfyUI può essere in una cartella non standard)
    try:
        ans = input("ComfyUI non individuato. Inserisci la cartella di ComfyUI (con main.py): ").strip()
    except EOFError:
        return None
    if ans and (Path(ans) / "main.py").exists():
        _persist_comfy(catalog_paths, ans)
        return ans
    return None


def _persist_comfy(catalog_paths, root: str) -> None:
    if catalog_paths is None:
        return
    try:
        cfg = catalog_paths.load(); cfg["comfy_root"] = root; catalog_paths.save(cfg)
    except Exception:
        pass


def check_all(comfy_override: str | None = None) -> dict:
    return {
        "python": check_python(),
        "nvidia": check_nvidia(),
        "torch": check_torch(),
        "ffmpeg": check_ffmpeg(),
        "comfy_root": resolve_comfy_root(comfy_override),
    }
