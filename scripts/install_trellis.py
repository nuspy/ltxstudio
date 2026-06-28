"""Installer TRELLIS.2 nativo (Windows) — approccio SECourses con WHEEL PRE-COMPILATI (niente build MSVC).

Riferimento: E:\\AI\\ltxstudio\\Trellis (FurkanGozukara/MonsterMMORPG) — requirements_trellis.txt +
Windows_Install_or_Update.bat. Risolve nvdiffrast/spconv/flash_attn/cumesh via wheel precompilati
(huggingface MonsterMMORPG/Wan_GGUF) su torch 2.9.1 cu130 (sm_120 Blackwell). Vedi [[trellis-windows-wheels]].

Usa il fork Trellis_2_3D_Generator (TRELLIS.2-4B, pesi già in Comfy_Nexus.bak/models/trellis2) + UniRig.
Uso: python scripts/install_trellis.py
"""
import subprocess
import sys
from pathlib import Path

TR = Path(r"E:\AI\Tools\TRELLIS2")
VENV = TR / "venv"
REPO_DIR = TR / "Trellis_2_3D_Generator"
UNIRIG_DIR = TR / "UniRig"
FAITHC_DIR = TR / "FaithC"
HF = "https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main"
WHEELS_NODEPS = [          # wheel extra installati con --no-deps (dal Windows_Install.bat SECourses)
    f"{HF}/flex_gemm-1.0.0-cp311-cp311-win_amd64.whl",
    f"{HF}/o_voxel-0.0.1-cp311-cp311-win_amd64.whl",
    f"{HF}/spconv_cu131-2.3.8+torch2.9.1-cp311-cp311-win_amd64.whl",
]


def _run(cmd, **k):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], **k)


def main():
    TR.mkdir(parents=True, exist_ok=True)
    if not REPO_DIR.exists():
        _run(["git", "clone", "--recursive", "https://github.com/FurkanGozukara/Trellis_2_3D_Generator",
              str(REPO_DIR)], check=False)
    if not UNIRIG_DIR.exists():
        _run(["git", "clone", "https://github.com/FurkanGozukara/UniRig", str(UNIRIG_DIR)], check=False)
    if not FAITHC_DIR.exists():
        _run(["git", "clone", "https://github.com/Luo-Yihao/FaithC.git", str(FAITHC_DIR)], check=False)
    if not VENV.exists():
        _run([sys.executable, "-m", "venv", str(VENV)], check=True)
    py = VENV / "Scripts" / "python.exe"
    _run([py, "-m", "pip", "install", "--upgrade", "pip", "uv"], check=True)
    # requirements del fork (torch 2.9.1 cu130 + flash_attn/nvdiffrast/spconv/cumesh/... wheel precompilati)
    req = REPO_DIR / "requirements_trellis.txt"
    if not req.exists():
        req = Path(r"E:\AI\ltxstudio\Trellis\requirements_trellis.txt")   # fallback al progetto di riferimento
    env = {"UV_SKIP_WHEEL_FILENAME_CHECK": "1", "UV_LINK_MODE": "copy"}
    import os
    full_env = {**os.environ, **env}
    _run([py, "-m", "uv", "pip", "install", "-r", str(req)], check=False, env=full_env)
    for w in WHEELS_NODEPS:
        _run([py, "-m", "uv", "pip", "install", "--no-deps", w], check=False, env=full_env)
    if FAITHC_DIR.exists():
        _run([py, "-m", "uv", "pip", "install", "--no-build-isolation", "--no-deps", "-e", str(FAITHC_DIR)],
             check=False, env=full_env)
    # verifica: nvdiffrast + trellis importabili
    chk = _run([py, "-c", "import sys; sys.path.insert(0, r'%s'); import nvdiffrast.torch, trellis; print('TRELLIS OK')"
                % REPO_DIR], capture_output=True, text=True)
    if "TRELLIS OK" in (chk.stdout or ""):
        (TR / ".ltx_ready").write_text("ok", encoding="utf-8")
        print("[trellis] PRONTO (TRELLIS.2 + nvdiffrast wheel). Pesi: HF_model_downloader o models/trellis2.")
    else:
        print("[trellis] verifica fallita:\n", (chk.stderr or "")[-1500:])


if __name__ == "__main__":
    main()
