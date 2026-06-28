"""Loader fp8-aware + STREAMING del transformer FLUX.2 (BFL fp8 mixed-scaled) — memory-bounded.

Il checkpoint `FLUX_2_Dev_FP8_Mixed_Scaled` è formato BFL (double_blocks/single_blocks), 128 layer
float8_e4m3fn (weight F8_E4M3 + weight_scale F32 scalare + input_scale) + 43 layer bf16.
`from_single_file` IGNORA le weight_scale (qualità rotta) e materializza bf16 in RAM (~24GB); il
convertitore diffusers su dict-completo fa `.float()` su tutto (picco ~70-90GB) → crash su <128GB.

Qui: la conversione BFL→diffusers è fatta PER-TENSORE in streaming (verificato sul sorgente diffusers:
ogni operazione è locale — rename 1:1, split QKV di un singolo tensore, swap scale/shift; NESSUNA concat
cross-layer). Si dequantizza ogni peso (w*weight_scale→bf16), lo si converte e lo si assegna al modello
(init_empty_weights), liberandolo subito. Picco ~1 layer invece di ~70GB. Per i 32GB stretti il passo
ulteriore è tenere i pesi in fp8 nel forward (vedi nota in fondo).
"""
from __future__ import annotations

import json
import os

import torch

# Mapping replicati da diffusers.loaders.single_file_utils.convert_flux2_transformer_checkpoint_to_diffusers
_RENAME = {
    "img_in": "x_embedder", "txt_in": "context_embedder",
    "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
    "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
    "single_stream_modulation.lin": "single_stream_modulation.linear",
    "final_layer.linear": "proj_out",
}
_DOUBLE = {
    "img_attn.norm.query_norm": "attn.norm_q", "img_attn.norm.key_norm": "attn.norm_k",
    "img_attn.proj": "attn.to_out.0", "img_mlp.0": "ff.linear_in", "img_mlp.2": "ff.linear_out",
    "txt_attn.norm.query_norm": "attn.norm_added_q", "txt_attn.norm.key_norm": "attn.norm_added_k",
    "txt_attn.proj": "attn.to_add_out", "txt_mlp.0": "ff_context.linear_in", "txt_mlp.2": "ff_context.linear_out",
}
_SINGLE = {"linear1": "attn.to_qkv_mlp_proj", "linear2": "attn.to_out",
           "norm.query_norm": "attn.norm_q", "norm.key_norm": "attn.norm_k"}


def _convert_one(key: str, t: torch.Tensor):
    """BFL key + tensore (già dequantizzato) → lista [(diffusers_key, tensor)] (1 o 3 per lo split QKV)."""
    from diffusers.loaders.single_file_utils import swap_scale_shift
    parts = key.split(".")
    ptype = parts[-1]
    ptype = "weight" if ptype == "scale" else ptype           # norm.*.scale → .weight (RMSNorm)
    if "adaLN_modulation" in key and key.endswith(".weight"):
        return [("norm_out.linear.weight", swap_scale_shift(t, 0))]
    if "double_blocks." in key:
        idx = parts[1]; within = ".".join(parts[2:-1]); modality = parts[2]
        if "qkv" in within:                                   # split in q/k/v (chunk locale dim=0)
            q, k, v = torch.chunk(t, 3, dim=0)
            names = ("attn.to_q", "attn.to_k", "attn.to_v") if "img" in modality \
                else ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj")
            return [(f"transformer_blocks.{idx}.{n}.{ptype}", x) for n, x in zip(names, (q, k, v))]
        return [(f"transformer_blocks.{idx}.{_DOUBLE[within]}.{ptype}", t)]
    if "single_blocks." in key:
        idx = parts[1]; within = ".".join(parts[2:-1])
        return [(f"single_transformer_blocks.{idx}.{_SINGLE[within]}.{ptype}", t)]
    nk = key                                                  # rename 1:1 (substring replace)
    for r, rep in _RENAME.items():
        nk = nk.replace(r, rep)
    return [(nk, t)]


def _iter_flux2_diffusers_tensors(path: str):
    """Genera (diffusers_key, bf16_tensor) UN tensore alla volta: dequant fp8 (w*weight_scale) + convert
    locale. Salta SOLO le quant-scale (weight_scale/input_scale); il `.scale` dei norm è un peso vero."""
    from safetensors import safe_open
    with safe_open(path, framework="pt") as h:
        keys = set(h.keys())
        for k in keys:
            if k.endswith((".weight_scale", ".input_scale")):
                continue
            base = k[:-len(".weight")] if k.endswith(".weight") else None
            if base and (base + ".weight_scale") in keys:     # peso fp8 scaled → dequant
                w = h.get_tensor(k)
                s = h.get_tensor(base + ".weight_scale").float()
                t = (w.to(torch.float32) * s).to(torch.bfloat16)
            else:                                             # bf16/norm-scale → copia
                t = h.get_tensor(k).to(torch.bfloat16)
            for nk, nt in _convert_one(k, t):
                yield nk, nt.to(torch.bfloat16)


def load_flux2_transformer(ckpt_path: str, config_dir: str):
    """Costruisce Flux2Transformer2DModel fp8-aware in STREAMING (picco ~1 layer). config_dir: cartella
    con transformer/config.json."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import Flux2Transformer2DModel
    cfg = json.load(open(os.path.join(config_dir, "config.json")))
    cfg.pop("_class_name", None); cfg.pop("_diffusers_version", None)
    with init_empty_weights():
        tr = Flux2Transformer2DModel.from_config(cfg)
    targets = {n for n, _ in tr.named_parameters()} | {n for n, _ in tr.named_buffers()}
    assigned, missing = 0, []
    for nk, t in _iter_flux2_diffusers_tensors(ckpt_path):
        if nk in targets:
            set_module_tensor_to_device(tr, nk, "cpu", value=t)
            assigned += 1
        else:
            missing.append(nk)
        del t
    import logging
    logging.info("[flux2_transformer] streaming: assegnati %d/%d (non mappati: %d)",
                 assigned, len(targets), len(missing))
    if missing:
        logging.debug("[flux2_transformer] esempi non mappati: %s", missing[:5])
    return tr.to(torch.bfloat16)

# NB 32GB stretti: il modello finale resta bf16 (~24GB). Per i 32GB il passo successivo è una Fp8Linear
# che tiene i pesi in fp8 (~12GB) e dequantizza nel forward (task #63 step 2) + sequential_cpu_offload.
