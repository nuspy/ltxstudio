"""VibeVoice nativo — STANDALONE, niente ComfyUI. TTS multi-speaker / dialoghi lunghi (MIT).

Gira nel venv isolato `E:\\AI\\Tools\\VibeVoice\\venv` (repo VibeVoice-Community) via subprocess
(scripts/vibevoice_generate.py). Stesso pattern di [[cosyvoice_service]]. voice sample + text → wav.
NB: runner da validare al primo uso (clone repo + download modello VibeVoice-1.5B).
"""
from __future__ import annotations

import re
from pathlib import Path

from core.music_service import Job

TOOLS = Path(r"E:\AI\Tools")
VV_DIR = TOOLS / "VibeVoice"
VV_VENV_PY = VV_DIR / "venv" / "Scripts" / "python.exe"
GEN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vibevoice_generate.py"
OUT_DIR = Path(r"E:\AI\ltxstudio\output\audio")


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", (name or "clone").strip()) or "clone"


def is_installed() -> bool:
    return VV_VENV_PY.exists() and (VV_DIR / ".ltx_ready").exists()


def make_job(text: str, prompt_wav: str, prompt_text: str = "", out_name: str = "clone", **cbs) -> Job:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{_safe(out_name)}.wav"
    cmd = [VV_VENV_PY, "-u", str(GEN_SCRIPT), "--text", text, "--ref_audio", prompt_wav,
           "--output", str(out)]
    return Job(cmd, VV_DIR, **cbs)


def preflight(text: str, prompt_wav: str) -> list[str]:
    errs = []
    if not is_installed():
        errs.append("VibeVoice non installato: usa l'installer (scripts/install_vibevoice.py).")
    if not (text or "").strip():
        errs.append("Testo da sintetizzare vuoto.")
    if not prompt_wav or not Path(prompt_wav).is_file():
        errs.append(f"Campione vocale inesistente: {prompt_wav}")
    return errs
