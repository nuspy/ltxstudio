"""Loader fp8-aware del transformer FLUX.2 (formato BFL fp8 mixed-scaled) — memory-bounded.

Il checkpoint `FLUX_2_Dev_FP8_Mixed_Scaled` è in formato BFL originale (double_blocks/single_blocks) con
128 layer **float8_e4m3fn** (weight F8_E4M3 + weight_scale F32 scalare + input_scale) e 43 layer bf16.
`Flux2Transformer2DModel.from_single_file` IGNORA le weight_scale (→ pesi fp8 raw senza scala = qualità
rotta) e materializza bf16 in RAM (~24GB) → su 32GB esplode.

Questo loader: (1) STREAMING dequant — per ogni peso fp8 applica `w.float()*weight_scale → bf16` (qualità
corretta), un tensore alla volta; (2) converte le chiavi BFL→diffusers col convertitore ufficiale; (3)
`init_empty_weights` (modello su meta, 0 RAM) + assign-and-free (svuota il dict mentre popola il modello).
Picco ~24GB invece di ~50GB. Per fascia 32GB il passo successivo è fp8-on-the-fly nel forward (task #63).
"""
from __future__ import annotations

import json
import os

import torch


def _dequant_flux2_bfl(path: str) -> dict:
    """Stream dal checkpoint BFL → dict {name: bf16} con le weight_scale fp8 APPLICATE (qualità corretta).
    Salta i tensori di sola-scala (weight_scale/input_scale/scale): vengono assorbiti nei pesi."""
    from safetensors import safe_open
    out: dict = {}
    with safe_open(path, framework="pt") as h:
        keys = set(h.keys())
        for k in keys:
            if k.endswith((".weight_scale", ".input_scale", ".scale")):
                continue
            base = k[:-len(".weight")] if k.endswith(".weight") else None
            if base and (base + ".weight_scale") in keys:          # peso fp8 scaled → dequant
                w = h.get_tensor(k)
                s = h.get_tensor(base + ".weight_scale").float()
                out[k] = (w.to(torch.float32) * s).to(torch.bfloat16)
            else:                                                  # bf16/plain → copia
                out[k] = h.get_tensor(k).to(torch.bfloat16)
    return out


def load_flux2_transformer(ckpt_path: str, config_dir: str):
    """Costruisce Flux2Transformer2DModel dal checkpoint fp8 BFL, fp8-aware e memory-bounded.
    config_dir: cartella con transformer/config.json (FLUX2_DIR/transformer)."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import Flux2Transformer2DModel
    from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers

    # GUARDIA RAM: il convertitore diffusers materializza il dict completo + .float() (picco ~70-90GB).
    # Finché non c'è il fp8-on-the-fly per-layer (task #63), evita il crash se la RAM non basta.
    try:
        import psutil
        free_gb = psutil.virtual_memory().available / 1024**3
        if free_gb < 80:
            raise MemoryError(
                f"FLUX.2 transformer: servono ~80GB RAM liberi per la conversione (disponibili {free_gb:.0f}GB). "
                "Il caricamento 32GB-safe (fp8-on-the-fly) è il task #63 — non ancora attivo.")
    except ImportError:
        pass
    sd_bfl = _dequant_flux2_bfl(ckpt_path)                          # scale applicate (qualità OK)
    sd = convert_flux2_transformer_checkpoint_to_diffusers(sd_bfl)  # BFL → diffusers
    del sd_bfl
    cfg = json.load(open(os.path.join(config_dir, "config.json")))
    cfg.pop("_class_name", None); cfg.pop("_diffusers_version", None)
    with init_empty_weights():                                     # modello su meta: 0 RAM all'init
        tr = Flux2Transformer2DModel.from_config(cfg)
    targets = {n for n, _ in tr.named_parameters()} | {n for n, _ in tr.named_buffers()}
    assigned = 0
    for k in list(sd.keys()):                                      # assign-and-free: svuota il dict
        v = sd.pop(k)
        if k in targets:
            set_module_tensor_to_device(tr, k, "cpu", value=v.to(torch.bfloat16))
            assigned += 1
        del v
    import logging
    logging.info("[flux2_transformer] assegnati %d/%d tensori (fp8-aware, streaming)", assigned, len(targets))
    return tr.to(torch.bfloat16)
