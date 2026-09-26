"""Installer del motore Avatar LongCat-Video-Avatar 1.5 (venv dedicato, Windows, ZERO ComfyUI).

Codice UFFICIALE meituan-longcat/LongCat-Video (MIT) scaricato in sparse-checkout (solo il package
`longcat_video/` + i requirements: la connessione è a consumo, gli asset demo non servono).

Le istruzioni ufficiali sono per Linux multi-GPU (torch 2.6+cu124, flash-attn 2.7.4, torchrun+NCCL):
  · cu124 NON supporta Blackwell (RTX 50xx)  → torch 2.9.1 + cu130 (stesso stack del venv TRELLIS)
  · flash-attn 2.7.4 non ha wheel Windows    → flash-attn 2.8.3 precompilato (lo stesso di TRELLIS)
  · NCCL non esiste su Windows               → il worker gira single-GPU senza torchrun (gloo/stub)
  · il separatore vocale usa onnxruntime CPU (il modello è piccolo; evita il conflitto CUDA12/13)
  · `block_sparse_attention` importa triton a livello di modulo → triton-windows

Uso:  python scripts/install_longcat_avatar.py
Env:  AI_TOOLS_DIR (default E:\\AI\\Tools), SKIP_TORCH=1
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TOOLS = Path(os.environ.get("AI_TOOLS_DIR", r"E:\AI\Tools"))
ROOT = TOOLS / "LongCatAvatar"
VENV = ROOT / "venv"
VENV_PY = VENV / "Scripts" / "python.exe"
REPO_DIR = ROOT / "LongCat-Video"
REPO = "https://github.com/meituan-longcat/LongCat-Video"
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu130")
HF_WHEELS = "https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main"
WHEELS = ["flash_attn-2.8.3+torch2.9.1.cuda13.1-cp311-cp311-win_amd64.whl"]

# requirements.txt + requirements_avatar.txt ufficiali SENZA i pin incompatibili con torch 2.9/Windows
# (torch==2.6.0, flash-attn==2.7.4, onnxruntime==1.16.3, tritonserverclient, openai, streamlit).
DEPS = [
    "numpy>=1.26,<2.3", "diffusers==0.35.1", "transformers>=4.46,<5", "accelerate", "safetensors",
    "sentencepiece", "protobuf", "loguru", "einops", "ftfy", "psutil", "av", "opencv-python",
    "imageio", "imageio-ffmpeg", "pillow", "scipy", "scikit-image", "scikit-learn", "soundfile",
    "soxr", "librosa==0.11.0", "pyloudnorm", "audio-separator[cpu]==0.30.2", "onnx",
    "triton-windows", "huggingface_hub",
]


def _env() -> dict:
    """pip del venv appena creato non ha truststore: dietro antivirus/proxy TLS serve il bundle
    combinato certifi + store di Windows (core/net_tls)."""
    env = dict(os.environ)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from core.net_tls import child_env
        env.update(child_env())
    except Exception:                                                # noqa: BLE001
        pass
    return env


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    kw.setdefault("env", _env())
    return subprocess.run([str(c) for c in cmd], **kw)


def _py311() -> str:
    for c in (os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python311\python.exe"),
              r"C:\Python311\python.exe", r"C:\Program Files\Python311\python.exe"):
        if Path(c).is_file():
            return c
    raise SystemExit("Python 3.11 non trovato: il wheel flash-attn è cp311. Installa Python 3.11.")


def clone_repo() -> bool:
    if (REPO_DIR / "longcat_video").is_dir():
        print(f"[=] codice già presente: {REPO_DIR}")
        run(["git", "-C", REPO_DIR, "pull", "--ff-only"])
        return True
    ROOT.mkdir(parents=True, exist_ok=True)
    if run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, REPO_DIR]).returncode:
        return False
    return run(["git", "-C", REPO_DIR, "sparse-checkout", "set", "longcat_video"]).returncode == 0


def check_import() -> bool:
    code = (f"import sys; sys.path.insert(0, r'{REPO_DIR}');"
            "import torch, flash_attn;"
            "from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline;"
            "from longcat_video.modules.quantization import load_quantized_dit;"
            "from longcat_video.audio_process import get_audio_encoder;"
            "from audio_separator.separator import Separator;"
            "print('LONGCAT OK torch', torch.__version__, 'cuda', torch.cuda.is_available())")
    r = run([VENV_PY, "-c", code], capture_output=True, text=True)
    print((r.stdout or "").strip())
    if "LONGCAT OK" not in (r.stdout or ""):
        print("[!] verifica import fallita:\n", (r.stderr or "")[-2500:])
        return False
    return True


def main() -> int:
    if not clone_repo():
        print("[!] clone del repo LongCat-Video fallito"); return 1
    if not VENV_PY.exists():
        if run([_py311(), "-m", "venv", VENV]).returncode:
            print("[!] creazione venv fallita"); return 1
    run([VENV_PY, "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"])
    if os.environ.get("SKIP_TORCH") != "1":
        if run([VENV_PY, "-m", "pip", "install", "torch==2.9.1", "torchvision", "torchaudio",
                "--index-url", TORCH_INDEX]).returncode:
            print("[!] installazione torch fallita"); return 1
    for w in WHEELS:
        run([VENV_PY, "-m", "pip", "install", "--no-deps", f"{HF_WHEELS}/{w}"])
    if run([VENV_PY, "-m", "pip", "install", *DEPS]).returncode:
        print("[!] installazione dipendenze fallita"); return 1
    if not check_import():
        return 1
    (ROOT / ".ltx_ready").write_text("ok", encoding="utf-8")
    print(f"\n[OK] motore Avatar LongCat 1.5 pronto: {VENV_PY}\n"
          "     Pesi dal tab «Pesi modelli» (voci lcav-*): INT8 ufficiale ~21 GB totali, oppure FP8.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
