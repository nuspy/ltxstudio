"""Pre-scarica TUTTI i pesi di TRELLIS.2 nella cartella modelli di ComfyUI.

PERCHÉ: i nodi ComfyUI-TRELLIS2 girano in un ambiente isolato comfy-env con un
timeout IPC HARDCODED di 600s. Il nodo LoadTrellis2Models scarica i pesi (~25GB)
dentro quella chiamata: se il download supera i 10 minuti (HuggingFace lento) il
job va in timeout. Pre-scaricando i pesi a parte (qui, senza timeout), al primo
run il nodo li trova già presenti e parte subito.

Replica esattamente la logica di nodes/stages.py:_init_config().
Idempotente e ripartibile: hf_hub_download riprende i download parziali.

Uso (con il python del venv di ComfyUI):
  E:\\AI\\Comfy_Nexus\\ComfyUI\\venv\\Scripts\\python.exe scripts\\download_trellis2_weights.py
Variabili opzionali:
  TRELLIS2_MODELS_DIR  override cartella (default: <ComfyUI>/models/trellis2)
  HF_HUB_ENABLE_HF_TRANSFER=1  (se hf_transfer installato) download più veloce
"""
from __future__ import annotations

import json
import os
import sys

# Protocollo Xet (download paralleli a chunk dal CAS HuggingFace): attivo se hf_xet
# è installato. HF_XET_HIGH_PERFORMANCE sfrutta più thread/connessioni.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # fallback per file non-Xet

MODELS_DIR = os.environ.get(
    "TRELLIS2_MODELS_DIR",
    r"E:\AI\Comfy_Nexus\ComfyUI\models\trellis2",
)
BASE_REPO = "microsoft/TRELLIS.2-4B"


def main() -> int:
    from huggingface_hub import hf_hub_download

    os.makedirs(MODELS_DIR, exist_ok=True)

    def dl(repo: str, name: str):
        # .json + .safetensors via huggingface_hub (usa Xet se il repo è Xet-backed)
        for ext in (".json", ".safetensors"):
            hf_hub_download(repo, name + ext, local_dir=MODELS_DIR)

    # 1. pipeline.json (definisce la lista dei modelli)
    cfg_file = os.path.join(MODELS_DIR, "pipeline.json")
    if not os.path.exists(cfg_file):
        print(f"[TRELLIS2] scarico pipeline.json da {BASE_REPO}…", flush=True)
        hf_hub_download(BASE_REPO, "pipeline.json", local_dir=MODELS_DIR)
    cfg = json.load(open(cfg_file))["args"]

    models = cfg["models"]
    total = len(models) + 1
    print(f"[TRELLIS2] {total} modelli da garantire in {MODELS_DIR}", flush=True)

    # 2. ogni modello del pipeline
    for i, (key, model_path) in enumerate(models.items(), 1):
        parts = model_path.split("/")
        if len(parts) >= 3 and not model_path.startswith("ckpts/"):
            repo_id = f"{parts[0]}/{parts[1]}"
            name = "/".join(parts[2:])
        else:
            repo_id = BASE_REPO
            name = model_path
        print(f"[{i}/{total}] {key}: {name}  ({repo_id})", flush=True)
        dl(repo_id, name)

    # 3. encoder shape_slat (non in pipeline.json ma necessario)
    print(f"[{total}/{total}] shape_slat_encoder", flush=True)
    dl(BASE_REPO, "ckpts/shape_enc_next_dc_f16c32_fp16")

    print("[TRELLIS2] ✔ TUTTI i pesi presenti.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
