"""Installer del motore video LingBot-Video (2ª architettura, accanto a LTX) — ENVIRONMENT-AWARE.

LingBot-Video è un modello MoE diffusers (`LingBotVideoPipeline`) con pin di dipendenze INCOMPATIBILI
con `ltx2_venv`/`inference_server_venv` (torch 2.12 dev cu130, transformers 5.8.x, diffusers 0.39.x,
peft 0.19.x). Perciò gira in un VENV DEDICATO `E:\\AI\\Tools\\lingbot_venv`, lanciato dal worker
`workers/lingbot_infer.py` via subprocess (stesso pattern di `ltx2_infer.py`).

Cosa fa:
  - crea il venv dedicato
  - torch nightly cu130 (Blackwell/50xx e precedenti); override con TORCH_INDEX; CPU-only se no GPU
  - transformers/diffusers/peft ai pin di LingBot + decord + json_repair + safetensors + accelerate
  - clona/installa il pacchetto `lingbot-video` (pip install -e .) → fornisce la pipeline e lo scheduler
    custom (`scheduling_flow_unipc.py`); se il clone fallisce, si può comunque caricare via
    trust_remote_code dal repo HF (fallback runtime nel worker).

Uso:  python scripts/install_lingbot.py
Env:  SKIP_TORCH, TORCH_INDEX (default nightly cu130), LINGBOT_REPO (git url), SKIP_PKG (salta pip -e)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TOOLS = Path(os.environ.get("AI_TOOLS_DIR", r"E:\AI\Tools"))
VENV = TOOLS / "lingbot_venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PKG_DIR = TOOLS / "lingbot-video"                       # sorgente del pacchetto (git clone)

# torch: LingBot raccomanda 2.12.0.dev cu130. Nightly, quindi index dedicato; override con TORCH_INDEX.
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/nightly/cu130")
LINGBOT_REPO = os.environ.get("LINGBOT_REPO", "https://github.com/Robbyant/lingbot-video")

# Pin di runtime LingBot (dal README ufficiale). transformers 5.x è REQUISITO (non compatibile con <5).
DEPS = [
    "transformers==5.8.1", "diffusers==0.39.0", "peft==0.19.1",
    "accelerate", "safetensors>=0.4.5", "sentencepiece", "protobuf", "pillow",
    "huggingface_hub", "decord>=0.6.0", "json_repair>=0.30", "einops", "ftfy", "imageio", "imageio-ffmpeg",
]


def run(cmd: list) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode


def find_py() -> str:
    """Interprete base per creare il venv: Python 3.12/3.11 (LingBot richiede >=3.10)."""
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
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=15)
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip().splitlines()[0].strip())
    except Exception:                                   # noqa: BLE001
        pass
    return None


def _clone_pkg() -> bool:
    """Clona il repo lingbot-video (per la pipeline + scheduler custom). Idempotente."""
    if PKG_DIR.exists() and any(PKG_DIR.iterdir()):
        print(f"[i] pacchetto già presente: {PKG_DIR}")
        return True
    print(f"[+] clono {LINGBOT_REPO} → {PKG_DIR}")
    return run(["git", "clone", "--depth", "1", LINGBOT_REPO, str(PKG_DIR)]) == 0


def main() -> int:
    py = find_py()
    cc = gpu_compute_cap()
    print(f"== Ambiente LingBot ==  OS={os.name}  py={py}  GPU compute_cap={cc}")
    if cc is not None and cc >= 12.0:
        print("[i] GPU Blackwell (sm120): serve torch cu130 nightly per il supporto kernel completo.")

    if not VENV_PY.exists():
        print(f"[+] creo venv: {VENV}")
        if run([py, "-m", "venv", str(VENV)]) != 0:
            return 1

    def pip(*a):
        return run([VENV_PY, "-m", "pip", *a])

    pip("install", "--upgrade", "pip", "wheel", "setuptools")

    if not os.environ.get("SKIP_TORCH"):
        if cc is None:
            print("[!] nessuna GPU CUDA → torch CPU (inferenza LingBot molto lenta).")
            if pip("install", "--pre", "torch", "torchvision") != 0:
                return 1
        else:
            print(f"=== torch nightly ({TORCH_INDEX}) ===")
            # --pre: i build cu130 di torch 2.12 sono nightly/pre-release
            if pip("install", "--pre", "torch", "torchvision", "--index-url", TORCH_INDEX) != 0:
                print("[!] torch nightly cu130 fallito. Riprova impostando TORCH_INDEX su un canale valido "
                      "(es. .../whl/cu128) o installa manualmente il build raccomandato dal README LingBot.")
                return 1

    if pip("install", *DEPS) != 0:
        return 1

    # Pacchetto lingbot-video: fornisce LingBotVideoPipeline + scheduler custom. Non fatale se fallisce
    # (il worker può caricare la pipeline via trust_remote_code dal repo HF, più lento all'avvio).
    if not os.environ.get("SKIP_PKG"):
        if _clone_pkg():
            print("=== pip install -e lingbot-video ===")
            if pip("install", "-e", str(PKG_DIR), "--no-deps") != 0:
                print("[!] pip -e lingbot-video fallito: il worker userà trust_remote_code (fallback).")
        else:
            print("[!] clone lingbot-video fallito: il worker userà trust_remote_code (fallback).")

    print("\n✔ LingBot-Video installato.")
    print(f"  venv:   {VENV}")
    print(f"  worker: {VENV_PY} -m ... (lanciato da core/lingbot_video_job / workers/lingbot_infer.py)")
    print("  Pesi:   scaricali dal wizard 'Pesi' (modulo Video) o con scripts/download_weights.py --ids lingbot_dense,lingbot_moe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
