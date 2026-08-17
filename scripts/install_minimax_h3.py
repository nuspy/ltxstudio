"""Installer del motore video MiniMax H3 (venv dedicato).

H3 richiede una combinazione che NON coesiste con gli altri venv video:
  · diffusers con le classi `MiniMaxH3*` (ModularPipeline, AutoencoderKLMiniMaxH3[Audio],
    MiniMaxH3Transformer3DModel, MiniMaxH3Scheduler) — supporto aggiunto dopo la 0.36
  · transformers con `Qwen3VLForConditionalGeneration` (il text encoder è un VLM da 32B)
Il worker (workers/minimax_h3_infer.py) costruisce il transformer dai config ufficiali e ci
carica i pesi dal single-file quantizzato, quindi NON servono i 124GB dello snapshot bf16.

Uso:  python scripts/install_minimax_h3.py
Env:  SKIP_TORCH=1 (salta torch), TORCH_INDEX (default cu130), DIFFUSERS_SPEC/TRANSFORMERS_SPEC
      per fissare versioni diverse (es. un branch git finché il supporto non è in una release).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TOOLS = Path(os.environ.get("AI_TOOLS_DIR", r"E:\AI\Tools"))
VENV = TOOLS / "minimax_h3_venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu130")
# diffusers: il supporto H3 è arrivato dopo la 0.36; se la release installata non ha le classi
# MiniMaxH3*, si passa il branch con DIFFUSERS_SPEC (il worker lo verifica e lo dice chiaro).
DIFFUSERS_SPEC = os.environ.get("DIFFUSERS_SPEC", "diffusers>=0.39.0")
TRANSFORMERS_SPEC = os.environ.get("TRANSFORMERS_SPEC", "transformers>=5.8.1")

DEPS = [
    DIFFUSERS_SPEC, TRANSFORMERS_SPEC, "accelerate", "safetensors>=0.4.5",
    "sentencepiece", "protobuf", "pillow", "huggingface_hub", "einops", "ftfy",
    "imageio", "imageio-ffmpeg", "soundfile", "av",
]


def run(cmd: list) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode


def ensure_venv() -> bool:
    if VENV_PY.exists():
        print(f"[=] venv già presente: {VENV}")
        return True
    VENV.parent.mkdir(parents=True, exist_ok=True)
    print(f"[+] creo il venv: {VENV}")
    return run([sys.executable, "-m", "venv", str(VENV)]) == 0


def check_classes() -> bool:
    """Verifica che diffusers ESPONGA davvero le classi H3: senza, il worker fallirebbe a
    generazione avviata invece che qui, con un messaggio comprensibile."""
    code = ("import diffusers, sys;"
            "missing=[n for n in ('MiniMaxH3Transformer3DModel','AutoencoderKLMiniMaxH3',"
            "'AutoencoderKLMiniMaxH3Audio','MiniMaxH3Scheduler','ModularPipeline') "
            "if not hasattr(diffusers,n)];"
            "print('MANCANTI:',missing) if missing else print('classi H3 OK, diffusers',diffusers.__version__);"
            "sys.exit(1 if missing else 0)")
    return run([str(VENV_PY), "-c", code]) == 0


def main() -> int:
    if not ensure_venv():
        print("[!] creazione venv fallita"); return 1
    run([str(VENV_PY), "-m", "pip", "install", "-U", "pip", "wheel"])
    if os.environ.get("SKIP_TORCH") != "1":
        print("[+] torch cu130")
        if run([str(VENV_PY), "-m", "pip", "install", "torch", "torchvision",
                "--index-url", TORCH_INDEX]) != 0:
            print("[!] installazione torch fallita"); return 1
    print("[+] dipendenze")
    if run([str(VENV_PY), "-m", "pip", "install", *DEPS]) != 0:
        print("[!] installazione dipendenze fallita"); return 1
    if not check_classes():
        print("\n[!] diffusers installato NON espone le classi MiniMaxH3*.\n"
              "    Rilancia fissando la sorgente giusta, es.:\n"
              '    set DIFFUSERS_SPEC=git+https://github.com/huggingface/diffusers@main\n'
              "    python scripts/install_minimax_h3.py")
        return 1
    print(f"\n[OK] motore MiniMax H3 pronto: {VENV_PY}\n"
          "     I pesi (~34GB con la variante nvfp4) si scaricano dal tab «Pesi modelli»\n"
          "     o al primo uso; la licenza MiniMax H3 Community esclude UE/UK/USA/Corea.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
