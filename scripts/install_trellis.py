"""Installer TRELLIS.2 nativo (Windows) — CODICE dal custom node MIT + WHEEL precompilati (asset).

Conforme licenze: il codice viene dal custom node `ComfyUI-TRELLIS2` (**MIT**, in Comfy_Nexus.bak), copiato
come package `trellis_nodes` + nostro shim `comfy` (no ComfyUI). Le deps CUDA difficili usano i WHEEL
PRECOMPILATI (asset) da MonsterMMORPG/Wan_GGUF (nvdiffrast/spconv/flash_attn/cumesh/...). Pesi TRELLIS.2-4B
già presenti (9.5GB in Comfy_Nexus.bak/models/trellis2) → symlink. NON copia codice del progetto SECourses
Patreon (non-open): solo approccio studiato (torch 2.9.1 + wheel). Vedi [[trellis-windows-wheels]] [[third-party-code-licensing]].
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

TR = Path(r"E:\AI\Tools\TRELLIS2")
VENV = TR / "venv"
NODES_SRC = Path(r"E:\AI\Comfy_Nexus.bak\ComfyUI\custom_nodes\ComfyUI-TRELLIS2\nodes")  # MIT
NODES_DST = TR / "trellis_nodes"
WEIGHTS_SRC = Path(r"E:\AI\Comfy_Nexus.bak\ComfyUI\models\trellis2")
MODELS_DIR = TR / "models" / "trellis2"
HF = "https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main"
WHEELS = [   # wheel precompilati Windows cp311 torch2.9.1 cu131 (asset) — niente build MSVC
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
    """Interprete Python 3.11 (i wheel TRELLIS sono cp311 → il venv DEVE essere 3.11, non 3.12)."""
    import os
    cands = [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python311\python.exe"),
        r"C:\Python311\python.exe",
        r"C:\Program Files\Python311\python.exe",
    ]
    for c in cands:
        if Path(c).is_file():
            return c
    raise SystemExit("Python 3.11 non trovato: i wheel TRELLIS sono cp311. Installa Python 3.11.")


def main():
    TR.mkdir(parents=True, exist_ok=True)
    # 1) codice MIT → package trellis_nodes (con __init__ vuoto: NON caricare i nodi comfy_api.io)
    if not NODES_DST.exists():
        shutil.copytree(NODES_SRC, NODES_DST, ignore=shutil.ignore_patterns("__pycache__"))
    (NODES_DST / "__init__.py").write_text("# trellis_nodes (codice MIT da ComfyUI-TRELLIS2)\n", encoding="utf-8")
    # 2) pesi (symlink, niente ricopia 9.5GB)
    MODELS_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not MODELS_DIR.exists() and WEIGHTS_SRC.exists():
        try:
            os.symlink(WEIGHTS_SRC, MODELS_DIR, target_is_directory=True)
        except OSError:
            print("  (symlink fallito — serve admin; copia manuale o junction)")
    # 3) venv Python 3.11 (cp311) + torch 2.9.1 cu130 (i wheel sono per 2.9.1/cp311)
    if not VENV.exists():
        _run([_py311(), "-m", "venv", str(VENV)], check=True)
    py = VENV / "Scripts" / "python.exe"
    _run([py, "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"], check=True)
    _run([py, "-m", "pip", "install", "torch==2.9.1", "torchvision",
          "--index-url", "https://download.pytorch.org/whl/cu130"], check=False)
    # 4) wheel CUDA precompilati (asset)
    for w in WHEELS:
        _run([py, "-m", "pip", "install", "--no-deps", f"{HF}/{w}"], check=False)
    # 5) deps Python (PyPI open) + utils3d dal commit pinnato
    _run([py, "-m", "pip", "install", "pillow", "imageio", "imageio-ffmpeg", "trimesh", "easydict",
          "opencv-python-headless", "scipy", "einops", "transformers", "huggingface_hub", "safetensors",
          "rembg", "onnxruntime", "xatlas", "pymeshfix", "igraph", "tqdm", "open3d", "psutil", "plyfile"],
         check=False)
    _run([py, "-m", "pip", "install", "--no-deps",
          "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"],
         check=False)
    # 6) verifica import della catena MIT via shim
    shim_dir = Path(__file__).resolve().parent
    chk = _run([py, "-c",
                "import sys; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s'); "
                "from trellis_comfy_shim import install_shim; install_shim(r'%s'); "
                "from trellis_nodes import stages; import nvdiffrast.torch; print('TRELLIS OK')"
                % (shim_dir, TR, str(MODELS_DIR.parent))],
               capture_output=True, text=True)
    if "TRELLIS OK" in (chk.stdout or ""):
        (TR / ".ltx_ready").write_text("ok", encoding="utf-8")
        print("[trellis] PRONTO (codice MIT + wheel precompilati + pesi).")
    else:
        print("[trellis] verifica fallita:\n", (chk.stderr or "")[-1800:])


if __name__ == "__main__":
    main()
