"""Installer del modulo MUSICA di LTX Studio (motori standalone, non ComfyUI).

Installa, in venv ISOLATI:
  - HeartMuLa (heartlib) — canzoni/strumentale (+ pesi HeartMuLa/HeartCodec da HuggingFace)
  - MMAudio                — sonorizzazione video→audio (pesi auto al primo run)

Idempotente (clona se manca, altrimenti aggiorna; non ricrea il venv se presente).

Uso:
  python scripts/install_music.py            # entrambi
  python scripts/install_music.py --heartmula
  python scripts/install_music.py --mmaudio

Env opzionali:
  SKIP_TORCH        salta l'installazione di torch
  SKIP_WEIGHTS      salta il download dei pesi HeartMuLa
  HF_TOKEN          token HuggingFace (se i repo lo richiedono)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

TOOLS = Path(os.environ.get("AI_TOOLS_DIR", r"E:\AI\Tools"))
TORCH_PKGS = ["torch", "torchvision", "torchaudio"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

HEARTLIB = TOOLS / "heartlib"
HEARTLIB_VENV_PY = HEARTLIB / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
MMAUDIO = TOOLS / "MMAudio"
MMAUDIO_VENV_PY = MMAUDIO / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

# repo HF dei pesi HeartMuLa → sottocartella locale in ckpt/
HEART_WEIGHTS = [
    ("HeartMuLa/HeartMuLaGen", "ckpt"),
    ("HeartMuLa/HeartMuLa-oss-3B-happy-new-year", "ckpt/HeartMuLa-oss-3B"),
    ("HeartMuLa/HeartCodec-oss-20260123", "ckpt/HeartCodec-oss"),
]


def run(cmd: list, cwd: Path | None = None) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None).returncode


def clone_or_update(dest: Path, url: str) -> int:
    if (dest / ".git").exists():
        print(f"[=] {dest.name}: aggiorno"); return run(["git", "-C", str(dest), "pull", "--ff-only"])
    print(f"[+] {dest.name}: clono"); dest.parent.mkdir(parents=True, exist_ok=True)
    return run(["git", "clone", url, str(dest)])


def make_venv(dest: Path, venv_py: Path) -> int:
    if venv_py.exists():
        print(f"[=] venv già presente: {dest/'venv'}"); return 0
    print(f"[+] creo venv: {dest/'venv'}")
    return run([sys.executable, "-m", "venv", str(dest / "venv")])


def pip(venv_py: Path, *args) -> int:
    return run([venv_py, "-m", "pip", *args])


def install_torch(venv_py: Path) -> int:
    if os.environ.get("SKIP_TORCH"):
        print("[=] SKIP_TORCH: salto torch"); return 0
    print("=== torch cu128 (Blackwell/RTX 5090) ===")
    return pip(venv_py, "install", *TORCH_PKGS, "--index-url", TORCH_INDEX)


def download_heart_weights(venv_py: Path) -> int:
    if os.environ.get("SKIP_WEIGHTS"):
        print("[=] SKIP_WEIGHTS: salto i pesi (scaricali poi a mano)"); return 0
    print("=== pesi HeartMuLa (diversi GB) ===")
    if pip(venv_py, "install", "huggingface_hub") != 0:
        return 1
    token = os.environ.get("HF_TOKEN", "")
    for repo, sub in HEART_WEIGHTS:
        local = HEARTLIB / sub
        local.mkdir(parents=True, exist_ok=True)
        code = (
            "from huggingface_hub import snapshot_download;"
            f"snapshot_download(repo_id={repo!r}, local_dir={str(local)!r},"
            f"token={token or None!r})"
        )
        print(f"[+] {repo} → {local}")
        if run([venv_py, "-c", code], cwd=HEARTLIB) != 0:
            print(f"[!] download fallito: {repo}"); return 1
    return 0


def install_heartmula() -> int:
    print("\n########## HeartMuLa ##########")
    if clone_or_update(HEARTLIB, "https://github.com/HeartMuLa/heartlib.git") != 0:
        return 1
    if make_venv(HEARTLIB, HEARTLIB_VENV_PY) != 0:
        return 1
    pip(HEARTLIB_VENV_PY, "install", "--upgrade", "pip", "wheel", "setuptools")
    if run([HEARTLIB_VENV_PY, "-m", "pip", "install", "-e", "."], cwd=HEARTLIB) != 0:
        return 1
    if install_torch(HEARTLIB_VENV_PY) != 0:        # cu128 ha l'ultima parola
        return 1
    return download_heart_weights(HEARTLIB_VENV_PY)


def install_mmaudio() -> int:
    print("\n########## MMAudio ##########")
    if clone_or_update(MMAUDIO, "https://github.com/hkchengrex/MMAudio.git") != 0:
        return 1
    if make_venv(MMAUDIO, MMAUDIO_VENV_PY) != 0:
        return 1
    pip(MMAUDIO_VENV_PY, "install", "--upgrade", "pip", "wheel", "setuptools")
    if run([MMAUDIO_VENV_PY, "-m", "pip", "install", "-e", "."], cwd=MMAUDIO) != 0:
        return 1
    return install_torch(MMAUDIO_VENV_PY)           # pesi MMAudio: auto al primo run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heartmula", action="store_true")
    ap.add_argument("--mmaudio", action="store_true")
    args = ap.parse_args()
    both = not (args.heartmula or args.mmaudio)
    print(f"Tools dir: {TOOLS}\ninstaller py: {sys.executable} ({sys.version.split()[0]})")
    rc = 0
    if both or args.heartmula:
        rc |= install_heartmula()
    if both or args.mmaudio:
        rc |= install_mmaudio()
    print("\n✔ Modulo musica installato." if rc == 0 else "\n[!] Installazione con errori: vedi log.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
