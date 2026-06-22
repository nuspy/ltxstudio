"""Installer del modulo MONDI 3D (WorldGen) — Windows nativo, venv isolato, non ComfyUI.

WorldGen è Linux-first: questo installer applica gli adattamenti necessari per Windows:
  - Python >= 3.11 (usa py -3.12; NON il 3.10 di sistema)
  - torch 2.7 cu128 (Blackwell) abbinato al wheel Windows di nunchaku
  - nunchaku: wheel Windows ufficiale (cp312) invece del wheel Linux pinnato in pyproject
  - pytorch3d: NON compilato → shim (solo le 2 funzioni quaternione↔matrice usate)
  - ml-sharp (submodule SSH, solo --use_sharp): saltato
  - install del package con --no-deps + deps elencate a mano (senza nunchaku/pytorch3d/xformers)

Scrive il marker `.ltx_ready` SOLO se `import worldgen` riesce.

Uso: python scripts/install_world.py
Env: SKIP_TORCH, NUNCHAKU_WHEEL (override URL)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = Path(os.environ.get("AI_TOOLS_DIR", r"E:\AI\Tools"))
WORLDGEN = TOOLS / "WorldGen"
VENV = WORLDGEN / "venv"
VENV_PY = VENV / "Scripts" / "python.exe"
SITE = VENV / "Lib" / "site-packages"
REPO = "https://github.com/ZiYang-xie/WorldGen.git"
SHIM = Path(__file__).resolve().parent / "worldgen_pt3d_shim.py"

# torch abbinato al wheel nunchaku Windows cp312
TORCH = ["torch==2.7.1", "torchvision==0.22.1"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
NUNCHAKU_WHEEL = os.environ.get(
    "NUNCHAKU_WHEEL",
    "https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.0/"
    "nunchaku-1.0.0+torch2.7-cp312-cp312-win_amd64.whl",
)
# deps WorldGen SENZA nunchaku (wheel Win a parte), pytorch3d (shim), xformers (non importato)
DEPS = [
    "diffusers>=0.33.1", "transformers>=4.48.3", "py360convert>=0.1.0", "einops>=0.7.0",
    "pillow>=8.0.0", "scikit-image>=0.24.0", "sentencepiece>=0.2.0", "opencv-python>=4.12.0.88",
    "peft>=0.7.1", "open3d>=0.19.0", "trimesh>=4.6.1", "timm>=1.0.15", "accelerate>=1.1.1",
    "matplotlib",  # dep di DA-2 (depth)
    "huggingface_hub", "viser @ git+https://github.com/ZiYang-xie/viser.git",
    "utils3d @ git+https://github.com/EasternJournalist/utils3d.git",
]


def run(cmd: list, cwd: Path | None = None) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None).returncode


def find_py() -> str | None:
    """Trova un interprete Python >= 3.11 (WorldGen lo richiede)."""
    cands = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Python/Python312/python.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Python/Python311/python.exe",
    ]
    for c in cands:
        if c.exists():
            return str(c)
    for args in (["py", "-3.12", "-c", "import sys;print(sys.executable)"],
                 ["py", "-3.11", "-c", "import sys;print(sys.executable)"]):
        try:
            r = subprocess.run(args, capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:                               # noqa: BLE001
            pass
    return None


def install_pt3d_shim() -> None:
    """Crea pytorch3d/transforms.py (shim) nel site-packages del venv."""
    pkg = SITE / "pytorch3d"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(SHIM, pkg / "transforms.py")
    print(f"[shim] pytorch3d.transforms → {pkg}")


def setup_da2() -> None:
    """DA-2 (depth) è un submodule NON pip-installabile: init + path src/ nel venv."""
    run(["git", "-C", str(WORLDGEN), "submodule", "update", "--init", "submodules/DA-2"])
    (SITE / "worldgen_da2.pth").write_text(
        str(WORLDGEN / "submodules" / "DA-2" / "src") + "\n", encoding="utf-8")
    print("[da2] submodule init + path aggiunto al venv")


def main() -> int:
    py = find_py()
    if not py:
        print("[ERRORE] serve Python 3.11+ (installa Python 3.12, poi riprova).")
        return 1
    print(f"WorldGen: {WORLDGEN}\npython 3.11+: {py}")

    if (WORLDGEN / ".git").exists():
        run(["git", "-C", str(WORLDGEN), "pull", "--ff-only"])
    else:
        WORLDGEN.parent.mkdir(parents=True, exist_ok=True)
        if run(["git", "clone", REPO, str(WORLDGEN)]) != 0:
            print("[ERRORE] clone fallito."); return 1

    if not VENV_PY.exists():
        print("[+] creo venv (3.12)")
        if run([py, "-m", "venv", str(VENV)]) != 0:
            return 1

    def pip(*a):
        return run([VENV_PY, "-m", "pip", *a])

    pip("install", "--upgrade", "pip", "wheel", "setuptools")
    if not os.environ.get("SKIP_TORCH"):
        print("=== torch 2.7 cu128 (abbinato al wheel nunchaku Windows) ===")
        if pip("install", *TORCH, "--index-url", TORCH_INDEX) != 0:
            print("[ERRORE] torch fallito."); return 1
    print("=== nunchaku (wheel Windows cp312) ===")
    if pip("install", NUNCHAKU_WHEEL) != 0:
        print("[!] wheel nunchaku non installato: imposta NUNCHAKU_WHEEL all'URL corretto "
              "(vedi github nunchaku-tech/nunchaku releases, cp312 win_amd64).")
    print("=== dipendenze WorldGen (no nunchaku/pytorch3d/xformers) ===")
    if pip("install", *DEPS) != 0:
        print("[!] alcune dipendenze non installate (vedi log).");
    print("=== package worldgen (editable, --no-deps) ===")
    run([VENV_PY, "-m", "pip", "install", "-e", ".", "--no-deps"], cwd=WORLDGEN)
    install_pt3d_shim()
    setup_da2()

    print("=== import test ===")
    rc = run([VENV_PY, "-c", "import worldgen; print('worldgen import OK')"])
    if rc == 0:
        (WORLDGEN / ".ltx_ready").write_text("ok", encoding="utf-8")
        print("\n✔ Modulo Mondi 3D installato (Windows). Pesi (FLUX.1-dev gated) al primo run.")
        return 0
    print("\n[ERRORE] import worldgen fallito: vedi traceback sopra (probabile mismatch "
          "API nunchaku o dipendenza mancante). Marker NON scritto.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
