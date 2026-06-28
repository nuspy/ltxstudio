"""TRELLIS.2: immagine → modello 3D (mesh GLB con PBR) — STANDALONE, niente ComfyUI.

Gira nel proprio venv isolato `E:\\AI\\Tools\\TRELLIS2\\venv` (repo microsoft/TRELLIS), pilotato
via subprocess attraverso scripts/trellis_generate.py (headless). Sostituisce il custom node
ComfyUI-TRELLIS2. Pesi (microsoft/TRELLIS-image-large) scaricati da HF al primo uso.
Export FBX/OBJ dal GLB col Blender dell'app (riuso world_service.export_mesh).
"""
from __future__ import annotations

import re
from pathlib import Path

from core.music_service import Job          # job subprocess generico (riuso)

TOOLS = Path(r"E:\AI\Tools")
TRELLIS_DIR = TOOLS / "TRELLIS2"
TRELLIS_VENV_PY = TRELLIS_DIR / "venv" / "Scripts" / "python.exe"
GEN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trellis_generate.py"
OUT_DIR = Path(r"E:\AI\ltxstudio\output\3d")


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", (name or "model").strip()) or "model"


def is_installed() -> bool:
    # marker scritto dall'installer solo dopo `import trellis` riuscito (deps CUDA difficili su Windows)
    return TRELLIS_VENV_PY.exists() and (TRELLIS_DIR / ".ltx_ready").exists()


def glb_path(name: str) -> Path:
    return OUT_DIR / _safe(name) / "model.glb"


def make_job(image_path: str, name: str = "model", *, seed: int = 1,
             ss_steps: int = 12, slat_steps: int = 12, simplify: float = 0.95,
             texture_size: int = 1024, hf_token: str | None = None, **cbs) -> Job:
    out = OUT_DIR / _safe(name)
    out.mkdir(parents=True, exist_ok=True)
    # NB: trellis_generate.py (API ufficiale microsoft/TRELLIS.2) accetta SOLO questi argomenti.
    # ss_steps/slat_steps/simplify restano nella firma per compat con la GUI ma non sono passati
    # (la pipeline ufficiale usa i suoi default); simplify→decimation come target-facce indicativo.
    cmd = [TRELLIS_VENV_PY, "-u", str(GEN_SCRIPT), "--image", image_path, "--output_dir", str(out),
           "--seed", int(seed), "--texture_size", int(texture_size)]
    env = {"HF_TOKEN": hf_token, "HUGGING_FACE_HUB_TOKEN": hf_token} if hf_token else None
    return Job(cmd, TRELLIS_DIR, env=env, **cbs)


def preflight(image_path: str) -> list[str]:
    errs = []
    if not is_installed():
        errs.append("TRELLIS non installato: usa l'installer (scripts/install_trellis.py).")
    if not image_path or not Path(image_path).is_file():
        errs.append(f"Immagine inesistente: {image_path}")
    return errs
