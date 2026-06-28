"""Installer TRELLIS.2 nativo (Windows) — repo UFFICIALE microsoft/TRELLIS.2 (MIT) + WHEEL precompilati.

ZERO ComfyUI: usa il codice ufficiale microsoft/TRELLIS.2 (github, MIT, niente comfy). Le deps CUDA
(nvdiffrast/spconv/cumesh/cumm/o_voxel/flex_gemm/flash_attn/sageattention) via WHEEL PRE-COMPILATI Windows
cp311/torch2.9.1 da MonsterMMORPG/Wan_GGUF (asset) → niente build MSVC. venv Python 3.11 (i wheel sono cp311).
Pesi microsoft/TRELLIS.2-4B scaricati da HF al primo uso. Vedi [[trellis-windows-wheels]] [[no-comfy-shim-rule]].

Uso: python scripts/install_trellis.py
"""
import os
import subprocess
import sys
from pathlib import Path

TR = Path(r"E:\AI\Tools\TRELLIS2")
VENV = TR / "venv"
REPO_DIR = TR / "TRELLIS2_official"
REPO = "https://github.com/microsoft/TRELLIS.2"
HF = "https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main"
WHEELS = [   # wheel precompilati Windows cp311 torch2.9.1 cu131 (asset) = deps di microsoft/TRELLIS.2
    "flash_attn-2.8.3+torch2.9.1.cuda13.1-cp311-cp311-win_amd64.whl",
    "sageattention-2.2.0+torch2.9.1.cuda13.1-cp39-abi3-win_amd64.whl",
    "nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl",
    "spconv_cu131-2.3.8+torch2.9.1-cp311-cp311-win_amd64.whl",
    "cumesh-0.0.1+torch291_cu13-cp311-cp311-win_amd64.whl",
    "cumm_cu_131_cuda_version-0.8.2+torch2.9.1-cp311-cp311-win_amd64.whl",
    "torch_cluster-1.6.3-cp311-cp311-win_amd64.whl",
    "torch_scatter-2.1.2+pt291cu13-cp311-cp311-win_amd64.whl",
    "flex_gemm-1.0.0-cp311-cp311-win_amd64.whl",
    "o_voxel-0.0.1-cp311-cp311-win_amd64.whl",
]


def _run(cmd, **k):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], **k)


def _py311() -> str:
    for c in (os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python311\python.exe"),
              r"C:\Python311\python.exe", r"C:\Program Files\Python311\python.exe"):
        if Path(c).is_file():
            return c
    raise SystemExit("Python 3.11 non trovato: i wheel TRELLIS sono cp311. Installa Python 3.11.")


def main():
    TR.mkdir(parents=True, exist_ok=True)
    if not REPO_DIR.exists():
        _run(["git", "clone", "--depth", "1", REPO, str(REPO_DIR)], check=False)
    if not VENV.exists():
        _run([_py311(), "-m", "venv", str(VENV)], check=True)
    py = VENV / "Scripts" / "python.exe"
    _run([py, "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"], check=True)
    _run([py, "-m", "pip", "install", "torch==2.9.1", "torchvision", "torchaudio",
          "--index-url", "https://download.pytorch.org/whl/cu130"], check=False)
    for w in WHEELS:
        _run([py, "-m", "pip", "install", "--no-deps", f"{HF}/{w}"], check=False)
    # deps Python del repo ufficiale (open) + utils3d pinnato
    _run([py, "-m", "pip", "install", "pillow", "imageio", "imageio-ffmpeg", "trimesh", "easydict",
          "opencv-python", "scipy", "einops", "transformers", "huggingface_hub", "safetensors",
          "rembg", "onnxruntime", "xatlas", "pymeshfix", "igraph", "tqdm", "open3d", "psutil",
          "plyfile", "numpy", "triton-windows"], check=False)   # triton-windows: richiesto da flex_gemm
    _run([py, "-m", "pip", "install", "--no-deps",
          "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"],
         check=False)
    # verifica import del codice UFFICIALE (no comfy) + o_voxel/nvdiffrast
    chk = _run([py, "-c",
                "import sys; sys.path.insert(0, r'%s'); "
                "from trellis2.pipelines import Trellis2ImageTo3DPipeline; "
                "import o_voxel, nvdiffrast.torch; print('TRELLIS2 OK')" % REPO_DIR],
               capture_output=True, text=True)
    if "TRELLIS2 OK" in (chk.stdout or ""):
        (TR / ".ltx_ready").write_text("ok", encoding="utf-8")
        print("[trellis] PRONTO (repo ufficiale microsoft/TRELLIS.2 + wheel, ZERO comfy). Pesi: HF al 1° uso.")
    else:
        print("[trellis] verifica fallita:\n", (chk.stderr or "")[-1800:])


if __name__ == "__main__":
    main()
