"""Installer ACE-Step nativo (clone ace-step/ACE-Step + venv torch cu128 + pip -e)."""
import subprocess, sys
from pathlib import Path
AS = Path(r"E:\AI\Tools\ACE-Step"); VENV = AS / "venv"; REPO_DIR = AS / "ACE-Step"
REPO = "https://github.com/ace-step/ACE-Step"


def main():
    AS.mkdir(parents=True, exist_ok=True)
    if not REPO_DIR.exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(REPO_DIR)], check=False)
    if not VENV.exists():
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    py = VENV / "Scripts" / "python.exe"
    subprocess.run([str(py), "-m", "pip", "install", "-U", "pip"], check=True)
    subprocess.run([str(py), "-m", "pip", "install", "torch", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu128"], check=False)
    subprocess.run([str(py), "-m", "pip", "install", "-e", str(REPO_DIR), "torchcodec"], check=False)
    chk = subprocess.run([str(py), "-c", "from acestep.pipeline_ace_step import ACEStepPipeline; print('AS OK')"],
                         capture_output=True, text=True)
    if "AS OK" in (chk.stdout or ""):
        (AS / ".ltx_ready").write_text("ok", encoding="utf-8")
        print("[acestep] PRONTO. Modello scaricato al primo uso (HF ACE-Step).")
    else:
        print("[acestep] import fallito:", (chk.stderr or "")[-800:])


if __name__ == "__main__":
    main()
