"""Fish-Speech / OpenAudio nativo — STANDALONE, niente ComfyUI. Voice cloning (Apache-2.0).

venv isolato `E:\AI\Tools\fish-speech\venv` (repo fishaudio/fish-speech) via subprocess
(scripts/fish_generate.py). Pattern [[cosyvoice_service]]. NB: runner da validare al primo uso.
"""
from __future__ import annotations
import re
from pathlib import Path
from core.music_service import Job

TOOLS = Path(r"E:\AI\Tools")
FS_DIR = TOOLS / "fish-speech"
FS_VENV_PY = FS_DIR / "venv" / "Scripts" / "python.exe"
GEN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fish_generate.py"
OUT_DIR = Path(r"E:\AI\ltxstudio\output\audio")


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", (name or "clone").strip()) or "clone"


def is_installed() -> bool:
    return FS_VENV_PY.exists() and (FS_DIR / ".ltx_ready").exists()


def make_job(text: str, prompt_wav: str, prompt_text: str = "", out_name: str = "clone", **cbs) -> Job:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{_safe(out_name)}.wav"
    cmd = [FS_VENV_PY, "-u", str(GEN_SCRIPT), "--text", text, "--ref_audio", prompt_wav,
           "--ref_text", prompt_text or "", "--output", str(out)]
    return Job(cmd, FS_DIR, **cbs)


def preflight(text: str, prompt_wav: str) -> list[str]:
    errs = []
    if not is_installed():
        errs.append("Fish-Speech non installato: usa l'installer (scripts/install_fish.py).")
    if not (text or "").strip():
        errs.append("Testo da sintetizzare vuoto.")
    if not prompt_wav or not Path(prompt_wav).is_file():
        errs.append(f"Campione vocale inesistente: {prompt_wav}")
    return errs
