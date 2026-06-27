"""Installer FlashVSR v1.1 (super-risoluzione video x4) — venv isolato + diffsynth.

Repo: OpenImagingLab/FlashVSR (clonato in E:\\AI\\Tools\\FlashVSR). Crea il venv e installa torch CUDA +
requirements (diffsynth, einops, imageio-ffmpeg, ...). block_sparse_attn/flash_attn sono opzionali
(fallback ad attention standard se non compilano su Windows). I pesi sono già in E:\\AI\\Models\\FlashVSR.

Uso: python scripts/install_flashvsr.py
"""
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(r"E:\AI\Tools\FlashVSR")
VENV = REPO_DIR / "venv"
REPO = "https://github.com/OpenImagingLab/FlashVSR"


def main():
    if not REPO_DIR.exists():
        print(f"[flashvsr] clono {REPO}")
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(REPO_DIR)], check=True)
    if not VENV.exists():
        print(f"[flashvsr] creo venv {VENV}")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    py = VENV / "Scripts" / "python.exe"
    subprocess.run([str(py), "-m", "pip", "install", "-U", "pip"], check=True)
    print("[flashvsr] installo torch CUDA 12.4 + requirements…")
    subprocess.run([str(py), "-m", "pip", "install", "torch==2.6.0", "torchvision==0.21.0",
                    "torchaudio==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu124"], check=False)
    req = REPO_DIR / "requirements.txt"
    if req.exists():
        subprocess.run([str(py), "-m", "pip", "install", "-r", str(req)], check=False)
    # diffsynth in editable (è nel repo)
    subprocess.run([str(py), "-m", "pip", "install", "-e", str(REPO_DIR)], check=False)
    print("[flashvsr] fatto. block_sparse_attn/flash_attn sono opzionali (Windows: fallback standard).")


if __name__ == "__main__":
    main()
