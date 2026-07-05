r"""ACE-Step **1.5** (fondazione F-D) — service subprocess. Motore musica di nuova generazione.

Differenze vs il v1 (core/acestep_service.py): DiT turbo 4B + planner **5Hz LM** (CoT), input strutturato
caption+lyrics+**metadata** (bpm/key/timesig/lingua) — la stessa separazione che avvicina l'aderenza a Suno.

Layout:
- venv:   E:\AI\Tools\ACE-Step\venv15         (copia del venv v1, già con tutte le deps di inferenza)
- codice: E:\AI\Tools\ACE-Step\ACE-Step-1.5-src (package `acestep`)
- pesi:   E:\AI\Tools\ACE-Step\Ace-Step1.5    (acestep-v15-turbo/, acestep-5Hz-lm-1.7B/, vae/, Qwen3-Embedding-0.6B/)

Runner: scripts/acestep15_generate.py (gira nel venv15). API confermata dal sorgente OSS.
"""
from __future__ import annotations

import re
from pathlib import Path

from core.music_service import Job

TOOLS = Path(r"E:\AI\Tools")
AS_DIR = TOOLS / "ACE-Step"
VENV15_PY = AS_DIR / "venv15" / "Scripts" / "python.exe"
SRC15_DIR = AS_DIR / "ACE-Step-1.5-src"
CKPT15_DIR = AS_DIR / "Ace-Step1.5"
GEN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "acestep15_generate.py"
OUT_DIR = Path(r"E:\AI\ltxstudio\output")

# sottocartelle-peso richieste (nomi come nel repo HF ACE-Step/Ace-Step1.5)
_DIT_SUB = "acestep-v15-turbo"
_LM_SUB = "acestep-5Hz-lm-1.7B"
_REQUIRED = (_DIT_SUB, _LM_SUB, "vae", "Qwen3-Embedding-0.6B")


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", (name or "song").strip()) or "song"


def _weights_ready() -> bool:
    """True se tutte le sottocartelle-peso esistono con almeno un .safetensors e nessun .incomplete."""
    if not CKPT15_DIR.exists() or any(CKPT15_DIR.rglob("*.incomplete")):
        return False
    for sub in _REQUIRED:
        d = CKPT15_DIR / sub
        if not d.exists() or not any(d.glob("*.safetensors")):
            return False
    return True


def is_installed() -> bool:
    return VENV15_PY.exists() and SRC15_DIR.exists() and (SRC15_DIR / "acestep").exists() and _weights_ready()


# mappa chiave task → flag CLI (valori non vuoti). task = dict prodotto dalla GUI per le modalità editing 1.5.
_TASK_FLAGS = {
    "task": "--task", "src_audio": "--src-audio", "ref_audio": "--ref-audio",
    "repaint_start": "--repaint-start", "repaint_end": "--repaint-end",
    "cover_strength": "--cover-strength", "global_caption": "--global-caption",
}


def make_job(caption: str, lyrics: str = "", out_name: str = "song", *, instrumental: bool = False,
             bpm: int = 0, keyscale: str = "", timesignature: str = "", language: str = "unknown",
             duration: float = -1.0, steps: int = 8, guidance: float = 1.0, seed: int = -1,
             think: bool = True, fmt: str = "wav", batch: int = 1, task: dict | None = None, **cbs) -> Job:
    """Costruisce il Job subprocess per la generazione ACE-Step 1.5. save_dir = output/ (root).
    task = dict opzionale per le modalità editing (cover/repaint/lego/extract/complete)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [VENV15_PY, "-u", str(GEN_SCRIPT),
           "--caption", caption, "--lyrics", lyrics or "",
           "--bpm", int(bpm or 0), "--keyscale", keyscale or "", "--timesignature", timesignature or "",
           "--language", language or "unknown", "--duration", float(duration),
           "--steps", int(steps), "--guidance", float(guidance), "--seed", int(seed),
           "--think", "true" if think else "false", "--format", fmt, "--batch", int(max(1, batch)),
           "--lm-model", _LM_SUB, "--dit-config", _DIT_SUB, "--output", str(OUT_DIR)]
    if instrumental:
        cmd.append("--instrumental")
    for key, flag in _TASK_FLAGS.items():                    # modalità editing 1.5
        if (task or {}).get(key) not in (None, ""):
            cmd += [flag, task[key]]
    return Job(cmd, SRC15_DIR, **cbs)


def preflight(caption: str) -> list[str]:
    errs = []
    if not (VENV15_PY.exists() and (SRC15_DIR / "acestep").exists()):
        errs.append("ACE-Step 1.5 non installato (venv15 o sorgente mancante).")
    elif not _weights_ready():
        missing = [s for s in _REQUIRED if not (CKPT15_DIR / s).exists() or not any((CKPT15_DIR / s).glob("*.safetensors"))]
        inc = "download incompleto" if any(CKPT15_DIR.rglob("*.incomplete")) else ""
        errs.append(f"ACE-Step 1.5: pesi non pronti ({inc or 'mancano: ' + ', '.join(missing)}).")
    if not (caption or "").strip():
        errs.append("Caption/stile vuoto.")
    return errs
