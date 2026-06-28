"""Installer F5-TTS nativo (venv isolato + pip f5-tts) — voice cloning, niente ComfyUI.

Uso: python scripts/install_f5tts.py
"""
import subprocess
import sys
from pathlib import Path

F5_DIR = Path(r"E:\AI\Tools\F5-TTS")
VENV = F5_DIR / "venv"


def main():
    F5_DIR.mkdir(parents=True, exist_ok=True)
    if not VENV.exists():
        print(f"[f5tts] creo venv {VENV}")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    py = VENV / "Scripts" / "python.exe"
    subprocess.run([str(py), "-m", "pip", "install", "-U", "pip"], check=True)
    # torch cu128 (Blackwell sm_120) PRIMA, poi f5-tts (--no-deps di torch evitato: f5-tts pinna torch)
    subprocess.run([str(py), "-m", "pip", "install", "torch", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu128"], check=False)
    subprocess.run([str(py), "-m", "pip", "install", "f5-tts"], check=False)
    (F5_DIR / ".ltx_ready").write_text("ok", encoding="utf-8")
    print("[f5tts] fatto. Modelli scaricati al primo uso (HF SWivid/F5-TTS).")


if __name__ == "__main__":
    main()
