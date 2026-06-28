"""Installer VibeVoice nativo (clone repo VibeVoice-Community + venv torch cu128 + pip -e)."""
import subprocess, sys
from pathlib import Path
VV = Path(r"E:\AI\Tools\VibeVoice"); VENV = VV / "venv"
REPO = "https://github.com/vibevoice-community/VibeVoice"


def main():
    if not (VV / ".git").exists() and not (VV / "vibevoice").exists():
        VV.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(VV)], check=False)
    if not VENV.exists():
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    py = VENV / "Scripts" / "python.exe"
    subprocess.run([str(py), "-m", "pip", "install", "-U", "pip"], check=True)
    subprocess.run([str(py), "-m", "pip", "install", "torch", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu128"], check=False)
    subprocess.run([str(py), "-m", "pip", "install", "-e", str(VV), "torchcodec", "soundfile",
                    "transformers>=4.49,<5"], check=False)
    (VV / ".ltx_ready").write_text("ok", encoding="utf-8")
    print("[vibevoice] fatto (API runner da validare al primo uso).")


if __name__ == "__main__":
    main()
