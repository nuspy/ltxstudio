"""Installer SeedVR2 (super-risoluzione video, standalone) — crea venv isolato + requirements.

SeedVR2 gira via CLI standalone in E:\\AI\\Tools\\SeedVR2 (clone di numz/ComfyUI-SeedVR2_VideoUpscaler).
Questo script crea il venv e installa i requirements (torch + cv2 + safetensors + ...). I pesi (3B fp8
~3GB) vengono scaricati on-demand al primo upscale (model_registry). Zero ComfyUI.

Uso: python scripts/install_seedvr2.py
"""
import os
import subprocess
import sys
from pathlib import Path

CLI_DIR = Path(r"E:\AI\Tools\SeedVR2")
VENV = CLI_DIR / "venv"
REPO = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler"


def main():
    if not CLI_DIR.exists():
        print(f"[seedvr2] clono {REPO} → {CLI_DIR}")
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(CLI_DIR)], check=True)
    if not VENV.exists():
        print(f"[seedvr2] creo venv {VENV}")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    py = VENV / "Scripts" / "python.exe"
    print("[seedvr2] aggiorno pip + installo requirements (torch CUDA + cv2 + safetensors…)")
    subprocess.run([str(py), "-m", "pip", "install", "-U", "pip"], check=True)
    # torch CUDA 12.1 (allineato al resto dell'app); poi i requirements del repo
    subprocess.run([str(py), "-m", "pip", "install", "torch", "torchvision",
                    "--index-url", "https://download.pytorch.org/whl/cu128"], check=False)
    req = CLI_DIR / "requirements.txt"
    if req.exists():
        subprocess.run([str(py), "-m", "pip", "install", "-r", str(req)], check=False)
    print("[seedvr2] fatto. I pesi 3B fp8 (~3GB) si scaricano al primo upscale.")


if __name__ == "__main__":
    main()
