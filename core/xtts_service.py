"""XTTS-v2 nativo (Coqui TTS) — STANDALONE, niente ComfyUI. Voice cloning 17 lingue (CC-BY-NC).

Gira nel venv isolato `E:\\AI\\Tools\\XTTS\\venv` (pip TTS) via subprocess (scripts/xtts_generate.py).
Stesso pattern di [[cosyvoice_service]]. speaker_wav + text (+language) → wav nella voce del campione.
"""
from __future__ import annotations

import re
from pathlib import Path

from core.music_service import Job

TOOLS = Path(r"E:\AI\Tools")
XT_DIR = TOOLS / "XTTS"
_CONS = Path(r"E:\AI\Tools\inference_server_venv\Scripts\python.exe")   # venv consolidato (#61)
XT_VENV_PY = _CONS if _CONS.exists() else XT_DIR / "venv" / "Scripts" / "python.exe"
GEN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "xtts_generate.py"
OUT_DIR = Path(r"E:\AI\ltxstudio\output\audio")


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", (name or "clone").strip()) or "clone"


def is_installed() -> bool:
    return XT_VENV_PY.exists() and (XT_DIR / ".ltx_ready").exists()


def make_job(text: str, prompt_wav: str, prompt_text: str = "", out_name: str = "clone",
             *, language: str = "it", **cbs) -> Job:
    """text → audio nella voce del campione `prompt_wav` (zero-shot). prompt_text ignorato (XTTS non lo usa)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{_safe(out_name)}.wav"
    cmd = [XT_VENV_PY, "-u", str(GEN_SCRIPT), "--text", text, "--ref_audio", prompt_wav,
           "--language", language or "it", "--output", str(out)]
    return Job(cmd, XT_DIR, **cbs)


def preflight(text: str, prompt_wav: str) -> list[str]:
    errs = []
    if not is_installed():
        errs.append("XTTS-v2 non installato: usa l'installer (scripts/install_xtts.py).")
    if not (text or "").strip():
        errs.append("Testo da sintetizzare vuoto.")
    if not prompt_wav or not Path(prompt_wav).is_file():
        errs.append(f"Campione vocale inesistente: {prompt_wav}")
    return errs
