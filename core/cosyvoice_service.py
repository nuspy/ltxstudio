"""Voce-clone CosyVoice2 — STANDALONE, niente ComfyUI. Sostituisce il custom node CosyVoice-ComfyUI.

Zero-shot voice cloning: da un campione vocale (wav) + testo → audio sintetizzato con quella voce.
Gira nel venv isolato `E:\\AI\\Tools\\CosyVoice\\venv` (repo FunAudioLLM/CosyVoice), via subprocess
(scripts/cosyvoice_generate.py). Si integra con la libreria voci dell'app (voices/samples).
"""
from __future__ import annotations

import re
from pathlib import Path

from core.music_service import Job

TOOLS = Path(r"E:\AI\Tools")
CV_DIR = TOOLS / "CosyVoice"
CV_VENV_PY = CV_DIR / "venv" / "Scripts" / "python.exe"
GEN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cosyvoice_generate.py"
OUT_DIR = Path(r"E:\AI\ltxstudio\output\audio")


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", (name or "clone").strip()) or "clone"


def is_installed() -> bool:
    return CV_VENV_PY.exists() and (CV_DIR / ".ltx_ready").exists()


def make_job(text: str, prompt_wav: str, prompt_text: str = "", out_name: str = "clone",
             *, instruct: str = "", **cbs) -> Job:
    """text → audio nella voce del campione `prompt_wav` (zero-shot). prompt_text = trascrizione del campione."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{_safe(out_name)}.wav"
    cmd = [CV_VENV_PY, "-u", str(GEN_SCRIPT), "--text", text, "--prompt_wav", prompt_wav,
           "--prompt_text", prompt_text or "", "--output", str(out)]
    if instruct:
        cmd += ["--instruct", instruct]
    return Job(cmd, CV_DIR, **cbs)


def preflight(text: str, prompt_wav: str) -> list[str]:
    errs = []
    if not is_installed():
        errs.append("CosyVoice non installato: usa l'installer (scripts/install_cosyvoice.py).")
    if not (text or "").strip():
        errs.append("Testo da sintetizzare vuoto.")
    if not prompt_wav or not Path(prompt_wav).is_file():
        errs.append(f"Campione vocale inesistente: {prompt_wav}")
    return errs
