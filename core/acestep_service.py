"""ACE-Step XL 1.5 — generazione musica/canzoni STANDALONE, niente ComfyUI (alternativa a HeartMuLa).

venv isolato `E:\AI\Tools\ACE-Step\venv` (repo ace-step/ACE-Step) via subprocess. Apache-2.0.
"""
from __future__ import annotations
import re
from pathlib import Path
from core.music_service import Job

TOOLS = Path(r"E:\AI\Tools")
AS_DIR = TOOLS / "ACE-Step"
AS_VENV_PY = AS_DIR / "venv" / "Scripts" / "python.exe"
GEN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "acestep_generate.py"
OUT_DIR = Path(r"E:\AI\ltxstudio\output\audio")


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", (name or "song").strip()) or "song"


def is_installed() -> bool:
    return AS_VENV_PY.exists() and (AS_DIR / ".ltx_ready").exists()


def make_job(tags: str, lyrics: str = "", out_name: str = "song", *, duration: float = 60.0,
             steps: int = 27, guidance: float = 15.0, lazy: bool = False, **cbs) -> Job:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{_safe(out_name)}.wav"
    cmd = [AS_VENV_PY, "-u", str(GEN_SCRIPT), "--tags", tags, "--lyrics", lyrics or "",
           "--duration", float(duration), "--steps", int(steps), "--guidance", float(guidance),
           "--output", str(out)]
    if lazy:                                              # low-VRAM: CPU offload dei componenti
        cmd.append("--lazy")
    return Job(cmd, AS_DIR, **cbs)


def preflight(tags: str) -> list[str]:
    errs = []
    if not is_installed():
        errs.append("ACE-Step non installato: usa l'installer (scripts/install_acestep.py).")
    if not (tags or "").strip():
        errs.append("Tag/stile vuoti.")
    return errs
