"""Installer Fish-Speech nativo (clone fishaudio/fish-speech + venv torch cu128 + pip -e)."""
import subprocess, sys
from pathlib import Path
FS = Path(r"E:\AI\Tools\fish-speech"); VENV = FS / "venv"
REPO = "https://github.com/fishaudio/fish-speech"


def main():
    if not (FS / ".git").exists() and not (FS / "fish_speech").exists():
        FS.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(FS)], check=False)
    if not VENV.exists():
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    py = VENV / "Scripts" / "python.exe"
    subprocess.run([str(py), "-m", "pip", "install", "-U", "pip"], check=True)
    subprocess.run([str(py), "-m", "pip", "install", "torch", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu128"], check=False)
    subprocess.run([str(py), "-m", "pip", "install", "-e", str(FS), "torchcodec"], check=False)
    (FS / ".ltx_ready").write_text("ok", encoding="utf-8")
    print("[fish] fatto (API runner da validare al primo uso).")


if __name__ == "__main__":
    main()
