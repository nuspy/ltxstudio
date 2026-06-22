"""Installer del backend nativo (InferenceServer) — ENVIRONMENT-AWARE.

Rileva l'ambiente locale e sceglie i wheel adatti (importante per la DISTRIBUZIONE: gli utenti
hanno GPU/CUDA/Python/OS diversi, non solo RTX 5090):
  - Python ≥3.11 (richiesto da alcune pipeline)
  - GPU compute capability (sm) via nvidia-smi → decide se installare SageAttention
  - torch cu128 (Blackwell + schede precedenti); CPU-only se nessuna GPU CUDA
  - SageAttention 2++ + triton SOLO se sm ≥ 8.9 (40xx/50xx); altrimenti SDPA (fallback automatico)
  - wheel Sage scelto via API GitHub in base a CUDA tag + versione torch installata + OS

Uso: python scripts/install_inference_server.py
Env: SKIP_TORCH, SKIP_SAGE
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

TOOLS = Path(os.environ.get("AI_TOOLS_DIR", r"E:\AI\Tools"))
VENV = TOOLS / "inference_server_venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
DEPS = ["diffusers>=0.33.1", "transformers>=4.48.3,<5", "accelerate>=1.1.1", "safetensors",
        "sentencepiece", "protobuf", "pillow", "huggingface_hub", "peft", "fastapi", "uvicorn"]
SAGE_RELEASE = "https://api.github.com/repos/woct0rdho/SageAttention/releases/tags/v2.2.0-windows.post5"


def run(cmd: list) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode


# ── rilevamento ambiente ─────────────────────────────────────────────────────
def find_py() -> str:
    for v in ("Python312", "Python311"):
        c = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Python" / v / "python.exe"
        if c.exists():
            return str(c)
    for tag in ("-3.12", "-3.11"):
        try:
            r = subprocess.run(["py", tag, "-c", "import sys;print(sys.executable)"], capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:                               # noqa: BLE001
            pass
    return sys.executable


def gpu_compute_cap() -> float | None:
    """compute capability della GPU (es. 12.0 Blackwell, 8.9 Ada, 8.6 Ampere) via nvidia-smi."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=15)
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip().splitlines()[0].strip())
    except Exception:                                   # noqa: BLE001
        pass
    return None


def torch_version(py: Path) -> str:
    try:
        r = subprocess.run([str(py), "-c", "import torch;print(torch.__version__)"], capture_output=True, text=True)
        return r.stdout.strip().split("+")[0]
    except Exception:                                   # noqa: BLE001
        return ""


def pick_sage_url(cuda_tag: str, tver: str) -> str | None:
    """Sceglie il wheel SageAttention dalla release in base a CUDA + versione torch + OS Windows."""
    if os.name != "nt":
        return None                                     # (Linux: build/altri canali, non gestito qui)
    try:
        rel = json.load(urllib.request.urlopen(SAGE_RELEASE, timeout=20))
    except Exception as e:                              # noqa: BLE001
        print(f"[sage] impossibile leggere la release ({e})"); return None
    try:
        major_minor = float(".".join(tver.split(".")[:2]))
    except Exception:                                   # noqa: BLE001
        major_minor = 2.9
    tkey = "torch2.10.0andhigher" if major_minor >= 2.10 else "torch2.9.1"
    for a in rel.get("assets", []):
        n = a["name"]
        if n.endswith(".whl") and cuda_tag in n and tkey in n and "win_amd64" in n:
            return a["browser_download_url"]
    return None


def main() -> int:
    py = find_py()
    cc = gpu_compute_cap()
    print(f"== Ambiente ==  OS={os.name}  py={py}  GPU compute_cap={cc}")

    if not VENV_PY.exists():
        print(f"[+] creo venv: {VENV}")
        if run([py, "-m", "venv", str(VENV)]) != 0:
            return 1

    def pip(*a):
        return run([VENV_PY, "-m", "pip", *a])

    pip("install", "--upgrade", "pip", "wheel", "setuptools")
    if not os.environ.get("SKIP_TORCH"):
        if cc is None:
            print("[!] nessuna GPU CUDA rilevata → torch CPU (inferenza molto lenta).")
            if pip("install", "torch", "torchvision") != 0:
                return 1
        else:
            print("=== torch cu128 ===")
            if pip("install", "torch", "torchvision", "--index-url", TORCH_INDEX) != 0:
                return 1
    if pip("install", *DEPS) != 0:
        return 1

    # SageAttention: solo su GPU sm≥8.9 (40xx/50xx); altrove SDPA (fallback runtime automatico)
    if not os.environ.get("SKIP_SAGE") and cc is not None and cc >= 8.9:
        tver = torch_version(VENV_PY)
        url = pick_sage_url("cu128", tver)
        if url:
            print(f"=== SageAttention 2++ (sm{cc}, torch {tver}) ===")
            pip("install", "triton-windows")
            if pip("install", url) != 0:
                print("[!] Sage non installato: si userà SDPA.")
        else:
            print("[!] wheel Sage non trovato per questo ambiente: si userà SDPA.")
    elif cc is not None:
        print(f"[i] GPU sm{cc} < 8.9: SageAttention non supportato → SDPA (nessun problema).")

    print("\n✔ InferenceServer installato.")
    print(f"  Avvio: {VENV_PY} -m inference_server.server  (porta 8189)")
    print("  Attention: env NATIVE_ATTENTION=auto|sage|sdpa  (auto = Sage se disponibile, else SDPA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
