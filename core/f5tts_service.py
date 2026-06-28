"""F5-TTS nativo — STANDALONE, niente ComfyUI. Voice cloning zero-shot SOTA (MIT).

Gira nel venv isolato `E:\\AI\\Tools\\F5-TTS\\venv` (pip f5-tts) via subprocess (scripts/f5tts_generate.py).
Stesso pattern di [[cosyvoice_service]]. ref_audio+ref_text+text → wav nella voce del campione.
"""
from __future__ import annotations

import re
from pathlib import Path

from core.music_service import Job

TOOLS = Path(r"E:\AI\Tools")
F5_DIR = TOOLS / "F5-TTS"
F5_VENV_PY = F5_DIR / "venv" / "Scripts" / "python.exe"
GEN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "f5tts_generate.py"
OUT_DIR = Path(r"E:\AI\ltxstudio\output\audio")


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", (name or "clone").strip()) or "clone"


def is_installed() -> bool:
    return F5_VENV_PY.exists() and (F5_DIR / ".ltx_ready").exists()


def make_job(text: str, prompt_wav: str, prompt_text: str = "", out_name: str = "clone", **cbs) -> Job:
    """text → audio nella voce del campione `prompt_wav` (zero-shot). prompt_text = trascrizione campione."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{_safe(out_name)}.wav"
    cmd = [F5_VENV_PY, "-u", str(GEN_SCRIPT), "--text", text, "--ref_audio", prompt_wav,
           "--ref_text", prompt_text or "", "--output", str(out)]
    return Job(cmd, F5_DIR, **cbs)


def preflight(text: str, prompt_wav: str) -> list[str]:
    errs = []
    if not is_installed():
        errs.append("F5-TTS non installato: usa l'installer (scripts/install_f5tts.py).")
    if not (text or "").strip():
        errs.append("Testo da sintetizzare vuoto.")
    if not prompt_wav or not Path(prompt_wav).is_file():
        errs.append(f"Campione vocale inesistente: {prompt_wav}")
    return errs
