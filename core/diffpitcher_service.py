r"""DiffPitcher — correzione d'intonazione voce (service subprocess). modo template e modo spartito MIDI validati a secco su CPU (17/10/2026).

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
_PITCHFORMER = "Diff-Pitcher_transformer_pitch_360.safetensors"        # solo modo spartito (MIDI)
_CONFIG = "config.json"
MODES = ("template", "score")


def is_installed(mode: str = "template") -> bool:
    weights = _WEIGHTS + ((_PITCHFORMER,) if mode == "score" else ())
    return (VENV15_PY.exists() and (SRC_DIR / "pitch_controller").exists()
            and all((W_DIR / w).exists() for w in weights) and (W_DIR / _CONFIG).exists())


def out_path(out_name: str) -> Path:
    return MUSIC_OUTPUT_DIR / f"{_safe(out_name)}_tuned.wav"


def make_job(source: str, pitch_ref: str = "", out_name: str = "vocal", *,
             steps: int = 50, shift_semi: int = 0, mode: str = "template", midi: str = "", **cbs) -> Job:
    """mode='template': `pitch_ref` = voce-guida; mode='score': `midi` = spartito (.mid/.midi)."""
    MUSIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = out_path(out_name)
    cmd = [VENV15_PY, "-u", str(RUNNER), "--mode", mode, "--source", str(source),
           "--steps", int(steps), "--shift-semi", int(shift_semi), "--output", str(out)]
    cmd += ["--midi", str(midi)] if mode == "score" else ["--pitch-ref", str(pitch_ref)]
    return Job(cmd, SRC_DIR, **cbs)


def preflight(source: str, pitch_ref: str = "", mode: str = "template", midi: str = "") -> list[str]:
    errs = []
    if mode not in MODES:
        return [f"Modo sconosciuto: {mode}."]
    if not is_installed(mode):
        errs.append("DiffPitcher non installato (venv15/sorgente/pesi mancanti"
                    + (": serve anche Diff-Pitcher_transformer_pitch_360.safetensors)." if mode == "score" else ")."))
    if not source or not Path(source).is_file():
        errs.append("Seleziona la voce da correggere (sorgente).")
    if mode == "template" and (not pitch_ref or not Path(pitch_ref).is_file()):
        errs.append("Seleziona la voce-guida (riferimento d'intonazione).")
    if mode == "score" and (not midi or not Path(midi).is_file() or Path(midi).suffix.lower() not in (".mid", ".midi")):
        errs.append("Seleziona lo spartito: un file MIDI (.mid / .midi).")
    return errs
