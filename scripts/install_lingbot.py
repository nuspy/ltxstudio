"""Installer del motore video LingBot-Video (2ª architettura) — RICETTA VALIDATA (17/07).

LingBot genera CORRETTAMENTE solo con FlashAttention (ramo attention "packed" via --batch_cfg) + caption
JSON ricca. Su Blackwell (RTX 5090 sm120) FA3/Hopper non esiste, ma FA2 sm120 SÌ. Serve un venv allineato
al wheel FA2 disponibile (cp311 / torch 2.11 cu130).

Cosa fa:
  1. crea `E:\\AI\\Tools\\lingbot_venv311` con **Python 3.11**
  2. torch **2.11.0 + cu130** (ABI del wheel FA2)
  3. installa il wheel **FlashAttention-2** (E:\\AI\\wheels\\flash_attn-2.8.4+cu130torch211...sm120...cp311...whl)
  4. transformers 5.8.1 / diffusers 0.39.0 / peft 0.19.1 + decord + json_repair + …
  5. `pip install -e lingbot-video`
  6. **PATCH** lingbot_video/transformer_lingbot_video.py: importa `flash_attn_varlen_func` da `flash_attn`
     (FA2) invece che da `flash_attn_interface` (FA3). Da riapplicare a OGNI reinstall del pacchetto.

Uso:  python scripts/install_lingbot.py
Env:  SKIP_TORCH, TORCH_INDEX (default cu130), LINGBOT_REPO, SKIP_PKG, FA_WHEEL (override path wheel)
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path

TOOLS = Path(os.environ.get("AI_TOOLS_DIR", r"E:\AI\Tools"))
VENV = TOOLS / "lingbot_venv311"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PKG_DIR = TOOLS / "lingbot-video"

TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu130")
LINGBOT_REPO = os.environ.get("LINGBOT_REPO", "https://github.com/Robbyant/lingbot-video")
WHEELS_DIR = Path(os.environ.get("AI_WHEELS_DIR", r"E:\AI\wheels"))

DEPS = [
    "transformers==5.8.1", "diffusers==0.39.0", "peft==0.19.1",
    "accelerate", "safetensors>=0.4.5", "sentencepiece", "protobuf", "pillow",
    "huggingface_hub", "decord>=0.6.0", "json_repair>=0.30", "einops", "ftfy", "imageio", "imageio-ffmpeg",
]


def run(cmd: list) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode


def find_py311() -> str | None:
    """Interprete Python 3.11 (richiesto dal wheel FA2 cp311)."""
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
    if os.environ.get("FA_WHEEL"):
        return os.environ["FA_WHEEL"]
    hits = glob.glob(str(WHEELS_DIR / "flash_attn-*cu130*sm120*cp311*win_amd64.whl"))
    return hits[0] if hits else None


def _clone_pkg() -> bool:
    if PKG_DIR.exists() and any(PKG_DIR.iterdir()):
        return True
    print(f"[+] clono {LINGBOT_REPO} → {PKG_DIR}")
    return run(["git", "clone", "--depth", "1", LINGBOT_REPO, str(PKG_DIR)]) == 0


def _patch_transformer() -> None:
    """FA3 (flash_attn_interface, Hopper sm90) non c'è su Blackwell → usa la varlen di FA2 (flash_attn).
    Idempotente. Da riapplicare dopo ogni reinstall del pacchetto."""
    f = PKG_DIR / "lingbot_video" / "transformer_lingbot_video.py"
    if not f.exists():
        print(f"[patch] {f} assente — salto"); return
    s = f.read_text(encoding="utf-8")
    old = "from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3"
    if old in s and "from flash_attn import flash_attn_varlen_func as flash_attn_varlen_func_v3" not in s:
        s = s.replace(
            "try:\n    " + old,
            "try:  # PATCH LTX Studio: FA2 (sm120) al posto di FA3/Hopper\n"
            "    from flash_attn import flash_attn_varlen_func as flash_attn_varlen_func_v3\n"
            "    _ = flash_attn_varlen_func_v3\n"
            "except Exception:\n    " + old, 1)
        f.write_text(s, encoding="utf-8")
        print("[patch] transformer: flash_attn_varlen_func → FA2")
    else:
        print("[patch] transformer già patchato (o formato diverso)")


def main() -> int:
    py = find_py311()
    if not py:
        print("[!] Python 3.11 non trovato. Installa Python 3.11 (winget install Python.Python.3.11) — "
              "richiesto dal wheel FlashAttention cp311.")
        return 1
    fa = find_fa_wheel()
    print(f"== LingBot install ==  py3.11={py}\n   FA wheel={fa or 'NON TROVATO in ' + str(WHEELS_DIR)}")

    if not VENV_PY.exists():
        print(f"[+] creo venv: {VENV}")
        if run([py, "-m", "venv", str(VENV)]) != 0:
            return 1

    def pip(*a):
        return run([VENV_PY, "-m", "pip", *a])

    pip("install", "--upgrade", "pip", "wheel", "setuptools")
    if not os.environ.get("SKIP_TORCH"):
        print(f"=== torch 2.11 cu130 ({TORCH_INDEX}) ===")
        if pip("install", "torch==2.11.0", "torchvision", "--index-url", TORCH_INDEX) != 0:
            return 1
    if fa:
        print("=== FlashAttention-2 (sm120) — INDISPENSABILE per la correttezza ===")
        if pip("install", fa) != 0:
            print("[!] wheel FA non installato: LingBot genererà GARBAGE senza flash-attn.")
    else:
        print("[!] wheel FlashAttention non trovato → LingBot genererà GARBAGE. "
              f"Metti il wheel cp311/torch211/cu130/sm120 in {WHEELS_DIR}.")
    if pip("install", *DEPS) != 0:
        return 1

    if not os.environ.get("SKIP_PKG"):
        if _clone_pkg():
            print("=== pip install -e lingbot-video ===")
            pip("install", "-e", str(PKG_DIR), "--no-deps")
            _patch_transformer()                            # DOPO l'install del pacchetto
        else:
            print("[!] clone lingbot-video fallito.")

    print("\n✔ LingBot-Video installato.")
    print(f"  venv:   {VENV}")
    print("  Ricetta corretta: runner ufficiale + --batch_cfg + caption JSON ricca + vae fp32 + FA2.")
    print("  Verifica: <venv>/python -c \"import flash_attn; print(flash_attn.__version__)\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
