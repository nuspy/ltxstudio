r"""Router captioner per i dataset LoRA (immagini/video). Sceglie motore + venv + argomenti.

Motori:
- **joycaption** (consigliato): JoyCaption Beta One nf4, gira in venv15 (transformers+bitsandbytes), pesi
  E:\AI\Models\JoyCaption. Non censurato, modalità descriptive/tags/sd-prompt. ⚠️ da validare live.
- **florence2** (fallback leggero): gira nel venv di training ai-toolkit (se installato).

I wizard di auto-dataset (immagini/video) possono chiamare run_blocking() dopo aver costruito il dataset
per SOSTITUIRE le caption procedurali (trigger+tag) con caption VLM reali.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

TOOLS = Path(r"E:\AI\Tools")
VENV15_PY = TOOLS / "ACE-Step" / "venv15" / "Scripts" / "python.exe"
JOY_DIR = Path(r"E:\AI\Models\JoyCaption")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "caption_dataset.py"

MODES = ["descriptive", "tags", "sd-prompt"]


def joycaption_ready() -> bool:
    """True se venv15 + pesi JoyCaption COMPLETI (safetensors presenti, nessun .incomplete)."""
    if not (VENV15_PY.exists() and JOY_DIR.exists()):
        return False
    if any(JOY_DIR.rglob("*.incomplete")):
        return False                                         # download ancora in corso
    return any(JOY_DIR.glob("*.safetensors"))


def _florence_python() -> Path | None:
    try:
        from core import training_service as ts
        return ts.VENV_PYTHON if ts.VENV_PYTHON.exists() else None
    except Exception:                                        # noqa: BLE001
        return None


def available_engines() -> list[str]:
    """Motori captioner realmente disponibili (venv+pesi presenti)."""
    eng = []
    if joycaption_ready():
        eng.append("joycaption")
    if _florence_python() is not None:
        eng.append("florence2")
    return eng


def _python_for(engine: str) -> Path | None:
    if engine == "joycaption":
        return VENV15_PY if VENV15_PY.exists() else None
    return _florence_python()


def build_args(dataset_dir, engine: str = "joycaption", mode: str = "descriptive",
               trigger: str = "", overwrite: bool = False) -> list[str]:
    args = ["-u", str(SCRIPT), "--dir", str(dataset_dir), "--engine", engine, "--mode", mode]
    if trigger:
        args += ["--trigger", trigger]
    if overwrite:
        args += ["--overwrite"]
    return args


def _coordinate_vram() -> None:
    """COORDINAMENTO CONDIZIONALE (mancava): JoyCaption nf4 è un VLM ~8GB che partiva con gli
    altri motori residenti. Se la VRAM libera non basta → demote degli altri (parcheggio RAM,
    rientro veloce a caption finito) + attesa del rilascio effettivo."""
    try:
        from core import gpu_mem, mem_coordinator
        if gpu_mem.free_vram_mb() < 10000:
            mem_coordinator.free_for("llm", timeout=15)
            gpu_mem.wait_free_vram_mb(10000, timeout_s=20, tag="caption")
    except Exception:                                        # noqa: BLE001
        pass


def python_and_args(dataset_dir, engine: str = "joycaption", mode: str = "descriptive",
                    trigger: str = "", overwrite: bool = False):
    """(python_exe, args) per QProcess. python_exe = None se il venv del motore non è disponibile.
    Coordina la VRAM QUI (il chiamante lancia il QProcess subito dopo)."""
    py = _python_for(engine)
    if py:
        _coordinate_vram()
    return (str(py) if py else None), build_args(dataset_dir, engine, mode, trigger, overwrite)


def run_blocking(dataset_dir, engine: str = "joycaption", mode: str = "descriptive",
                 trigger: str = "", overwrite: bool = False, on_log=None) -> int:
    """Esecuzione BLOCCANTE (per i wizard: ricaptioning del dataset dopo la costruzione). Ritorna il returncode."""
    py = _python_for(engine)
    if py is None:
        if on_log:
            on_log(f"[caption] motore {engine} non disponibile (venv/pesi mancanti) — salto il captioning VLM.")
        return 1
    _coordinate_vram()
    cmd = [str(py)] + build_args(dataset_dir, engine, mode, trigger, overwrite)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:                                 # inoltra il log riga per riga
        if on_log:
            on_log(line.rstrip())
    return proc.wait()
