"""Installer del backend nativo (InferenceServer) — venv isolato torch+diffusers, non ComfyUI.

Crea l'ambiente in cui gira `inference_server.server` (separato dall'app PyQt e da ComfyUI).
Locale al 100%: i pesi dei modelli si scaricano UNA VOLTA e poi l'inferenza è offline.

Uso: python scripts/install_inference_server.py
Env: SKIP_TORCH
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TOOLS = Path(os.environ.get("AI_TOOLS_DIR", r"E:\AI\Tools"))
VENV = TOOLS / "inference_server_venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
TORCH = ["torch", "torchvision"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
DEPS = ["diffusers>=0.33.1", "transformers>=4.48.3,<5", "accelerate>=1.1.1", "safetensors",  # transformers 5.x rompe from_single_file (CLIPTextModel.text_model)
        "sentencepiece", "protobuf", "pillow", "huggingface_hub", "peft",
        "fastapi", "uvicorn"]


def run(cmd: list) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode


def find_py() -> str:
    cand = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Python/Python312/python.exe"
    return str(cand) if cand.exists() else sys.executable


def main() -> int:
    if not VENV_PY.exists():
        print(f"[+] creo venv: {VENV}")
        if run([find_py(), "-m", "venv", str(VENV)]) != 0:
            return 1

    def pip(*a):
        return run([VENV_PY, "-m", "pip", *a])

    pip("install", "--upgrade", "pip", "wheel", "setuptools")
    if not os.environ.get("SKIP_TORCH"):
        if pip("install", *TORCH, "--index-url", TORCH_INDEX) != 0:
            print("[ERRORE] torch fallito."); return 1
    if pip("install", *DEPS) != 0:
        print("[ERRORE] deps fallite."); return 1
    print("\n✔ InferenceServer installato. Avvio:")
    print(f"  {VENV_PY} -m inference_server.server   (porta 8189)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
