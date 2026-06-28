"""Installer XTTS-v2 nativo (venv isolato + pip TTS Coqui) — voice cloning, niente ComfyUI.

Uso: python scripts/install_xtts.py
"""
import subprocess
import sys
from pathlib import Path

XT_DIR = Path(r"E:\AI\Tools\XTTS")
VENV = XT_DIR / "venv"


def main():
    XT_DIR.mkdir(parents=True, exist_ok=True)
    if not VENV.exists():
        print(f"[xtts] creo venv {VENV}")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    py = VENV / "Scripts" / "python.exe"
    subprocess.run([str(py), "-m", "pip", "install", "-U", "pip"], check=True)
    subprocess.run([str(py), "-m", "pip", "install", "torch", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu128"], check=False)
    # coqui-tts (fork mantenuto di TTS, compatibile py3.12) + transformers compatibile
    subprocess.run([str(py), "-m", "pip", "install", "coqui-tts"], check=False)
    (XT_DIR / ".ltx_ready").write_text("ok", encoding="utf-8")
    print("[xtts] fatto. Modello xtts_v2 scaricato al primo uso (accetta licenza CC-BY-NC).")


if __name__ == "__main__":
    main()
