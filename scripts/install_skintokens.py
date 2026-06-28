"""Installer SkinTokens (rigging mesh→scheletro+skin) nativo — venv isolato, niente ComfyUI.

Repo VAST-AI/SkinTokens (TokenRig). Blender headless (riuso dell'app) per skinning/export.
NB: deps possono richiedere wheel precompilati su Windows (docs/precompiled.pdf). Uso: python scripts/install_skintokens.py
"""
import subprocess, sys
from pathlib import Path
SK = Path(r"E:\AI\Tools\SkinTokens"); VENV = SK / "venv"
REPO_DIR = SK / "SkinTokens"
REPO = "https://github.com/VAST-AI-Research/TokenRig"   # SkinTokens / TokenRig


def main():
    SK.mkdir(parents=True, exist_ok=True)
    if not REPO_DIR.exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(REPO_DIR)], check=False)
    if not VENV.exists():
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    py = VENV / "Scripts" / "python.exe"
    subprocess.run([str(py), "-m", "pip", "install", "-U", "pip", "wheel"], check=True)
    subprocess.run([str(py), "-m", "pip", "install", "torch",
                    "--index-url", "https://download.pytorch.org/whl/cu128"], check=False)
    subprocess.run([str(py), "-m", "pip", "install", "trimesh", "numpy", "scipy", "einops",
                    "safetensors", "huggingface_hub"], check=False)
    req = REPO_DIR / "requirements.txt"
    if req.exists():
        subprocess.run([str(py), "-m", "pip", "install", "-r", str(req)], check=False)
    chk = subprocess.run([str(py), "-c", "import torch; print('SK OK')"], capture_output=True, text=True)
    if "SK OK" in (chk.stdout or ""):
        (SK / ".ltx_ready").write_text("ok", encoding="utf-8")
        print("[skintokens] PRONTO (Blender per skinning/export: auto-install winget al primo uso).")
    else:
        print("[skintokens] deps incomplete:", (chk.stderr or "")[-800:])


if __name__ == "__main__":
    main()
