"""Installer TRELLIS.2 nativo (repo microsoft/TRELLIS + venv) — immagine→mesh 3D, niente ComfyUI.

NB: TRELLIS ha deps CUDA ostiche su Windows (spconv, nvdiffrast, diff-gaussian-rasterization, flash-attn,
kaolin). Dove possibile usare i wheel PRE-COMPILATI (vedi docs/precompiled.pdf, task #75) per evitare la
compilazione. Marker `.ltx_ready` scritto solo se `import trellis` riesce.

Uso: python scripts/install_trellis.py
"""
import subprocess
import sys
from pathlib import Path

TR = Path(r"E:\AI\Tools\TRELLIS2")
VENV = TR / "venv"
REPO_DIR = TR / "TRELLIS"
REPO = "https://github.com/microsoft/TRELLIS"


def _run(cmd, **k):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], **k)


def main():
    TR.mkdir(parents=True, exist_ok=True)
    if not REPO_DIR.exists():
        _run(["git", "clone", "--recurse-submodules", REPO, str(REPO_DIR)], check=False)
    if not VENV.exists():
        _run([sys.executable, "-m", "venv", str(VENV)], check=True)
    py = VENV / "Scripts" / "python.exe"
    _run([py, "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"], check=True)
    # torch cu128 (Blackwell sm_120)
    _run([py, "-m", "pip", "install", "torch", "torchvision",
          "--index-url", "https://download.pytorch.org/whl/cu128"], check=False)
    # deps base TRELLIS (le CUDA-pesanti separate)
    _run([py, "-m", "pip", "install", "pillow", "imageio", "imageio-ffmpeg", "trimesh", "easydict",
          "opencv-python-headless", "scipy", "einops", "transformers", "huggingface_hub", "safetensors",
          "rembg", "onnxruntime", "xatlas", "pyvista", "pymeshfix", "igraph", "tqdm"], check=False)
    # CUDA-heavy: spconv (sparse conv), flash-attn (usare wheel precompilato sm_120 se disponibile),
    # nvdiffrast + diff-gaussian-rasterization (da git). Possono fallire su Windows → l'app degrada.
    _run([py, "-m", "pip", "install", "spconv-cu126"], check=False)
    _run([py, "-m", "pip", "install", "flash-attn", "--no-build-isolation"], check=False)
    _run([py, "-m", "pip", "install", "git+https://github.com/NVlabs/nvdiffrast.git"], check=False)
    # verifica import trellis → scrive il marker solo se OK
    chk = _run([py, "-c", "import sys; sys.path.insert(0, r'%s'); import trellis; print('TRELLIS OK')" % REPO_DIR],
               capture_output=True, text=True)
    if "TRELLIS OK" in (chk.stdout or ""):
        (TR / ".ltx_ready").write_text("ok", encoding="utf-8")
        print("[trellis] PRONTO. Pesi microsoft/TRELLIS-image-large scaricati al primo uso.")
    else:
        print("[trellis] deps CUDA incomplete (Windows). Output:\n", (chk.stderr or "")[-1500:])
        print("[trellis] Usa i wheel precompilati (docs/precompiled.pdf) per flash-attn/spconv e riprova.")


if __name__ == "__main__":
    main()
