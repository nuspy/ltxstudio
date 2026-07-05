r"""DiffPitcher — correzione d'intonazione voce (service subprocess). SCAFFOLD, test live pendente.

venv15 (con pyworld/pysptk/fastdtw/music21/tgt) + codice E:\AI\Tools\DiffPitcher-src + pesi E:\AI\Models\DiffPitcher.
Runner: scripts/diffpitcher_run.py. Output: <nome>_tuned.wav in output/.
"""
from __future__ import annotations

from pathlib import Path

from core.music_service import Job, MUSIC_OUTPUT_DIR, _safe

TOOLS = Path(r"E:\AI\Tools")
VENV15_PY = TOOLS / "ACE-Step" / "venv15" / "Scripts" / "python.exe"
SRC_DIR = TOOLS / "DiffPitcher-src"
W_DIR = Path(r"E:\AI\Models\DiffPitcher")
RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "diffpitcher_run.py"

_WEIGHTS = ("Diff-Pitcher_world_fixed_40.safetensors", "Diff-Pitcher_bigvgan_24khz_100band.safetensors")
_CONFIG = "config.json"


def is_installed() -> bool:
    return (VENV15_PY.exists() and (SRC_DIR / "pitch_controller").exists()
            and all((W_DIR / w).exists() for w in _WEIGHTS) and (W_DIR / _CONFIG).exists())


def out_path(out_name: str) -> Path:
    return MUSIC_OUTPUT_DIR / f"{_safe(out_name)}_tuned.wav"


def make_job(source: str, pitch_ref: str, out_name: str = "vocal", *,
             steps: int = 50, shift_semi: int = 0, **cbs) -> Job:
    MUSIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = out_path(out_name)
    cmd = [VENV15_PY, "-u", str(RUNNER), "--source", str(source), "--pitch-ref", str(pitch_ref),
           "--steps", int(steps), "--shift-semi", int(shift_semi), "--output", str(out)]
    return Job(cmd, SRC_DIR, **cbs)


def preflight(source: str, pitch_ref: str) -> list[str]:
    errs = []
    if not is_installed():
        errs.append("DiffPitcher non installato (venv15/sorgente/pesi mancanti).")
    if not source or not Path(source).is_file():
        errs.append("Seleziona la voce da correggere (sorgente).")
    if not pitch_ref or not Path(pitch_ref).is_file():
        errs.append("Seleziona la voce-guida (riferimento d'intonazione).")
    return errs
