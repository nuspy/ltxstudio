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
    # 1) deps di ACE-Step (il suo setup può tirare un torch CPU di default: lo correggiamo dopo)
    subprocess.run([str(py), "-m", "pip", "install", "-e", str(REPO_DIR), "torchcodec"], check=False)
    # 2) trio torch cu128 PINNATO e installato PER ULTIMO → vince sulle deps di ACE-Step e resta ALLINEATO.
    #    Versioni coerenti obbligatorie: torch/torchvision/torchaudio dello stesso build cu128, altrimenti
    #    ABI mismatch ("Could not load libtorchaudio" / "operator torchvision::nms does not exist").
    #    cu128 = RTX 50xx sm_120. Vedi [[torchcodec-ffmpeg-windows]].
    subprocess.run([str(py), "-m", "pip", "install",
                    "torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0",
                    "--index-url", "https://download.pytorch.org/whl/cu128"], check=False)
    chk = subprocess.run([str(py), "-c", "from acestep.pipeline_ace_step import ACEStepPipeline; print('AS OK')"],
                         capture_output=True, text=True)
    if "AS OK" in (chk.stdout or ""):
        (AS / ".ltx_ready").write_text("ok", encoding="utf-8")
        print("[acestep] PRONTO. Modello scaricato al primo uso (HF ACE-Step).")
    else:
        print("[acestep] import fallito:", (chk.stderr or "")[-800:])


if __name__ == "__main__":
    main()
