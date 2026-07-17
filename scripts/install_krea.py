"""Installer delle architetture Krea AI (alternative a FLUX/LTX) — Krea 2 (immagini) + Krea Realtime (video).

- **Krea 2** (`Krea2Pipeline`, diffusers 0.39, text-encoder Qwen3-VL, VAE Qwen-Image): stesse deps di LingBot
  → RIUSA `E:\\AI\\Tools\\lingbot_venv311` (py3.11 + torch 2.11 cu130 + FlashAttention-2 sm120). Qui si verifica
  solo che il venv esista e si clona/installa il pacchetto krea-2 SE la pipeline non carica via trust_remote_code.
- **Krea Realtime 14B** (`WanModularPipeline`, diffusers 0.36 MODULAR, componenti Wan 2.1 14B): venv DEDICATO
  `E:\\AI\\Tools\\krea_rt_venv` (py3.11 + torch 2.11 cu130 + diffusers 0.36 + Wan deps + wheel FA2 dal manifest).
  ⚠️ Licenza: usare la pipeline diffusers Apache-2.0 di HF, MAI il server GitHub CC-BY-NC-SA.

Uso:  python scripts/install_krea.py [--only krea2|krea_rt]
Env:  SKIP_TORCH, TORCH_INDEX (default cu130), SKIP_PKG
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

TOOLS = Path(os.environ.get("AI_TOOLS_DIR", r"E:\AI\Tools"))
LINGBOT_VENV = TOOLS / "lingbot_venv311"                     # riusato da Krea 2 (stesse deps)
KREA_RT_VENV = TOOLS / "krea_rt_venv"                        # dedicato a Krea Realtime (diffusers 0.36 modular)
KREA2_PKG = TOOLS / "krea-2"
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu130")
KREA2_REPO = os.environ.get("KREA2_REPO", "https://github.com/krea-ai/krea-2")
MANIFEST = Path(__file__).resolve().parent.parent / "data" / "precompiled_wheels.json"

# Krea Realtime: diffusers 0.36 (WanModularPipeline). Wan components + FA (dal wheel manifest).
RT_DEPS = [
    "diffusers==0.36.0", "transformers>=4.49", "accelerate", "safetensors>=0.4.5",
    "sentencepiece", "protobuf", "pillow", "huggingface_hub", "ftfy", "einops",
    "imageio", "imageio-ffmpeg", "ftfy",
]


def run(cmd: list) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode


def find_py311() -> str | None:
    c = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Python/Python311/python.exe"
    if c.exists():
        return str(c)
    try:
        r = subprocess.run(["py", "-3.11", "-c", "import sys;print(sys.executable)"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:                                       # noqa: BLE001
        pass
    return None


def find_fa_wheel() -> str | None:
    """Wheel FlashAttention-2 (cp311/torch211/cu130/sm120) dalle search-dir del manifest."""
    dirs = []
    try:
        dirs = json.loads(MANIFEST.read_text("utf-8")).get("search_dirs", [])
    except Exception:                                       # noqa: BLE001
        pass
    dirs = dirs or [r"E:\AI\Tools\precompiled_wheels", r"E:\AI\wheels"]
    for d in dirs:
        hits = glob.glob(str(Path(d) / "flash_attn-*cu130*sm120*cp311*win_amd64.whl"))
        if hits:
            return hits[0]
    return None


# ── Krea 2 (immagini) ─────────────────────────────────────────────────────────
def install_krea2() -> int:
    py = LINGBOT_VENV / "Scripts" / "python.exe"
    if not py.exists():
        print(f"[krea2] venv LingBot mancante ({LINGBOT_VENV}). Esegui prima scripts/install_lingbot.py "
              "(Krea 2 riusa lo stesso venv: diffusers 0.39 + Qwen3-VL + FA2).")
        return 1
    print(f"[krea2] uso il venv LingBot: {LINGBOT_VENV}")
    # La Krea2Pipeline è custom-code: prova il caricamento via trust_remote_code dallo snapshot HF (il worker
    # lo fa a runtime). Se serve il pacchetto krea-2 (mmdit/sampling), lo si clona + pip install --no-deps.
    if not os.environ.get("SKIP_PKG"):
        r = subprocess.run([str(py), "-c", "import importlib.util as u;print(bool(u.find_spec('krea2')))"],
                           capture_output=True, text=True)
        if "True" not in (r.stdout or ""):
            if not (KREA2_PKG.exists() and any(KREA2_PKG.iterdir())):
                print(f"[krea2] clono {KREA2_REPO} → {KREA2_PKG} (per la pipeline, se trust_remote_code non basta)")
                run(["git", "clone", "--depth", "1", KREA2_REPO, str(KREA2_PKG)])
            if KREA2_PKG.exists():
                run([str(py), "-m", "pip", "install", "-e", str(KREA2_PKG), "--no-deps"])
    print("[krea2] pronto. Il worker carica Krea2Pipeline (fp8) dallo snapshot krea/Krea-2-Turbo|Raw.")
    return 0


# ── Krea Realtime (video) ─────────────────────────────────────────────────────
def install_krea_rt() -> int:
    py311 = find_py311()
    if not py311:
        print("[krea_rt] Python 3.11 non trovato (richiesto dal wheel FA2 cp311)."); return 1
    vpy = KREA_RT_VENV / "Scripts" / "python.exe"
    if not vpy.exists():
        print(f"[+] creo venv: {KREA_RT_VENV}")
        if run([py311, "-m", "venv", str(KREA_RT_VENV)]) != 0:
            return 1

    def pip(*a):
        return run([vpy, "-m", "pip", *a])

    pip("install", "--upgrade", "pip", "wheel", "setuptools")
    if not os.environ.get("SKIP_TORCH"):
        print(f"=== torch 2.11 cu130 ({TORCH_INDEX}) ===")
        if pip("install", "torch==2.11.0", "torchvision", "--index-url", TORCH_INDEX) != 0:
            return 1
    fa = find_fa_wheel()
    if fa:
        print("=== FlashAttention-2 (sm120) ===")
        pip("install", fa)
    else:
        print("[krea_rt] !! wheel FlashAttention non trovato (data/precompiled_wheels.json) → "
              "Krea Realtime potrebbe girare male/errare senza flash-attn.")
    if pip("install", *RT_DEPS) != 0:
        return 1
    print("\n✔ Krea Realtime installato.")
    print(f"  venv: {KREA_RT_VENV}")
    print("  Pesi: krea/krea-realtime-video + base Wan-AI/Wan2.1-T2V-14B-Diffusers (wizard 'Pesi', modulo Video).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["krea2", "krea_rt"], default="")
    a = ap.parse_args()
    rc = 0
    if a.only in ("", "krea2"):
        rc |= install_krea2()
    if a.only in ("", "krea_rt"):
        rc |= install_krea_rt()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
