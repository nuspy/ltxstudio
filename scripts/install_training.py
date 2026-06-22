"""Installer del modulo di TRAINING LoRA (FLUX.1/FLUX.2) basato su ai-toolkit (ostris).

Idempotente: clona se manca, altrimenti aggiorna; crea il venv solo se assente.

Esegue:
  1. Clona/aggiorna  ai-toolkit  in  E:\\AI\\Tools\\ai-toolkit
  2. Crea un venv ISOLATO            (NON quello di ComfyUI: dipendenze diverse)
  3. pip install torch 2.9.1 + cu128 (Blackwell/RTX 5090) nel venv
  4. pip install -r requirements.txt

Uso:
  python scripts\\install_training.py

Variabili d'ambiente opzionali:
  AI_TOOLKIT_DIR   override del percorso di installazione (default: E:\\AI\\Tools\\ai-toolkit)
  SKIP_TORCH       se impostata, salta l'installazione di torch (utile se già presente)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

AI_TOOLKIT_DIR = Path(os.environ.get("AI_TOOLKIT_DIR", r"E:\AI\Tools\ai-toolkit"))
REPO_URL = "https://github.com/ostris/ai-toolkit.git"
VENV_DIR = AI_TOOLKIT_DIR / "venv"
VENV_PY = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

TORCH_PKGS = ["torch==2.9.1", "torchvision==0.24.1"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None).returncode


def clone_or_update() -> int:
    if (AI_TOOLKIT_DIR / ".git").exists():
        print(f"[=] ai-toolkit già presente in {AI_TOOLKIT_DIR}: aggiorno")
        return run(["git", "-C", str(AI_TOOLKIT_DIR), "pull", "--ff-only"])
    print(f"[+] Clono ai-toolkit in {AI_TOOLKIT_DIR}")
    AI_TOOLKIT_DIR.parent.mkdir(parents=True, exist_ok=True)
    return run(["git", "clone", REPO_URL, str(AI_TOOLKIT_DIR)])


def make_venv() -> int:
    if VENV_PY.exists():
        print(f"[=] venv già presente: {VENV_DIR}")
        return 0
    print(f"[+] Creo venv: {VENV_DIR}")
    return run([sys.executable, "-m", "venv", str(VENV_DIR)])


def pip(*args: str) -> int:
    return run([VENV_PY, "-m", "pip", *args])


def main() -> int:
    print(f"ai-toolkit:  {AI_TOOLKIT_DIR}")
    print(f"venv python: {VENV_PY}")
    print(f"installer py: {sys.executable} ({sys.version.split()[0]})\n")

    if clone_or_update() != 0:
        print("[ERRORE] clone/aggiornamento ai-toolkit fallito."); return 1
    if make_venv() != 0:
        print("[ERRORE] creazione venv fallita."); return 1

    print("\n=== Aggiorno pip ===")
    pip("install", "--upgrade", "pip", "wheel", "setuptools")

    if not os.environ.get("SKIP_TORCH"):
        print("\n=== Installo torch cu128 (Blackwell/RTX 5090) ===")
        if pip("install", *TORCH_PKGS, "--index-url", TORCH_INDEX) != 0:
            print("[ERRORE] installazione torch fallita."); return 1
    else:
        print("\n[=] SKIP_TORCH impostata: salto torch.")

    req = AI_TOOLKIT_DIR / "requirements.txt"
    if req.exists():
        print("\n=== Installo requirements ai-toolkit ===")
        if pip("install", "-r", str(req)) != 0:
            print("[ERRORE] installazione requirements fallita."); return 1

    print("\n=== Dipendenze auto-captioning (Florence-2) ===")
    pip("install", "timm", "einops")     # best-effort: servono a Florence-2

    print("\n✔ Modulo training installato.")
    print("  - I modelli base (FLUX.1/FLUX.2, gated) vengono scaricati al primo run da HuggingFace.")
    print("  - Imposta un HF token nella tab Training per i modelli gated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
