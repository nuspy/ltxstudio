"""Trainer LoRA LingBot-Video — SPERIMENTALE (venv LingBot).

⚠️ NON VALIDATO. LingBot non ha una ricetta LoRA ufficiale. Il transformer è un DiT diffusers →
si può agganciare una LoRA (peft LoraConfig sui proiettori attn) e allenarla con flow-matching sui
latenti VAE dei clip del dataset. Qui c'è lo SCHELETRO corretto dell'orchestrazione; i punti da
validare live (target_modules, encode video→latent, timestep/shift, loss) sono marcati # VALIDATE.

Uso:  python -m workers.lingbot_train --model <dir> --dataset <dataset.json> --out lora.safetensors
      --rank 16 --steps 1500 --trigger "..."
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--trigger", default="")
    a = ap.parse_args()

    import torch
    from diffusers import DiffusionPipeline
    from peft import LoraConfig, get_peft_model

    print(f"[lingbot-train] carico base {a.model}", flush=True)
    pipe = DiffusionPipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        raise RuntimeError("pipeline LingBot senza attributo 'transformer' (VALIDATE)")

    # VALIDATE: i target_modules corretti vanno letti dai nomi dei layer del transformer LingBot.
    lora_cfg = LoraConfig(r=a.rank, lora_alpha=a.rank,
                          target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # convenzione diffusers attn
                          lora_dropout=0.0)
    transformer = get_peft_model(transformer, lora_cfg)
    transformer.train().to("cuda")
    print(f"[lingbot-train] LoRA rank={a.rank} agganciata; parametri allenabili:", flush=True)
    transformer.print_trainable_parameters()

    with open(a.dataset, encoding="utf-8") as f:
        dataset = json.load(f)      # formato di lora_video_service.build_dataset_json
    print(f"[lingbot-train] dataset caricato: {len(dataset) if hasattr(dataset, '__len__') else '?'} voci", flush=True)

    # VALIDATE: il vero ciclo di training (VAE.encode del video → latenti, campionamento timestep con
    # shift, flow-matching loss vs. rumore, optimizer step) va implementato e validato live. Qui NON si
    # esegue un fine-tuning reale: si scrive un errore chiaro finché la ricetta non è confermata, così il
    # chiamante non consegna una LoRA fasulla come valida.
    raise RuntimeError(
        "Training LoRA LingBot non ancora implementato (ricetta flow-matching da validare live). "
        "Lo scaffold — caricamento base, LoraConfig, target_modules — è pronto in workers/lingbot_train.py. "
        "Completa il ciclo (VAE encode → timestep/shift → loss → save_pretrained) al primo run reale.")


if __name__ == "__main__":
    raise SystemExit(main())
