"""Loader fp8-FORWARD del transformer FLUX.2 (~32B) — i pesi restano fp8, dequant on-the-fly nel forward.

FLUX.2-dev è ~32B: transformer bf16 = 66GB, TE = 30-46GB → ~100GB RAM in bf16 = IMPRATICABILE.
Soluzione (come ComfyUI su 32GB VRAM): tenere i pesi in **fp8** (transformer 33GB) e dequantizzare
on-the-fly per-layer durante il forward (`Fp8Linear`). RAM/VRAM ~metà del bf16; picco bf16 = 1 layer.

La conversione BFL→diffusers è per-tensore in streaming (operazioni LOCAL: rename, split QKV di un singolo
tensore, swap scale/shift). I 128 Linear quantizzati diventano `Fp8Linear` (weight fp8 + weight_scale
scalare); i 43 layer bf16 restano `nn.Linear`. Lo split QKV produce 3 Fp8Linear con la stessa scale.

Nota 32GB stretti: i pesi fp8 (33GB) + offload. Per RAM <33GB si aggiunge mmap da disco (step successivo).
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

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


class Fp8Linear(nn.Module):
    """Linear con peso in float8_e4m3fn + scala scalare: dequant on-the-fly nel forward (w.to(dtype)*scale).
    Tiene metà memoria del bf16; il bf16 esiste solo per il layer in esecuzione (1 alla volta)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn))
        self.register_buffer("weight_scale", torch.ones((), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


def _convert_key(key: str):
    """BFL tensor-key → lista di (diffusers_key, chunk_idx|None). chunk_idx per lo split QKV (0/1/2)."""
    parts = key.split("."); ptype = parts[-1]
    ptype = "weight" if ptype == "scale" else ptype
    if "adaLN_modulation" in key and key.endswith(".weight"):
        return [("norm_out.linear.weight", None, "swap")]
    if "double_blocks." in key:
        idx = parts[1]; within = ".".join(parts[2:-1]); modality = parts[2]
        if "qkv" in within:
            names = ("attn.to_q", "attn.to_k", "attn.to_v") if "img" in modality \
                else ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj")
            return [(f"transformer_blocks.{idx}.{n}.{ptype}", i, None) for i, n in enumerate(names)]
        return [(f"transformer_blocks.{idx}.{_DOUBLE[within]}.{ptype}", None, None)]
    if "single_blocks." in key:
        idx = parts[1]; within = ".".join(parts[2:-1])
        return [(f"single_transformer_blocks.{idx}.{_SINGLE[within]}.{ptype}", None, None)]
    nk = key
    for r, rep in _RENAME.items():
        nk = nk.replace(r, rep)
    return [(nk, None, None)]


def _module_and_attr(name: str):
    """'a.b.c.weight' → ('a.b.c', 'weight')."""
    return name.rsplit(".", 1)


def load_flux2_transformer(ckpt_path: str, config_dir: str):
    """Costruisce Flux2Transformer2DModel con i Linear quantizzati come Fp8Linear (pesi fp8, dequant
    on-the-fly). RAM ~33GB invece di ~66GB bf16; picco bf16 = 1 layer."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import Flux2Transformer2DModel
    from diffusers.loaders.single_file_utils import swap_scale_shift
    from safetensors import safe_open

    cfg = json.load(open(os.path.join(config_dir, "config.json")))
    cfg.pop("_class_name", None); cfg.pop("_diffusers_version", None)
    with init_empty_weights():
        tr = Flux2Transformer2DModel.from_config(cfg)

    # 1) quali MODULI diffusers vanno resi Fp8Linear (dai pesi fp8 del checkpoint = quelli con .weight_scale)
    with safe_open(ckpt_path, framework="pt") as h:
        keys = set(h.keys())
        fp8_modules: dict[str, torch.Tensor] = {}        # module_name → weight_scale (scalar)
        for k in keys:
            if not k.endswith(".weight"):
                continue
            base = k[:-len(".weight")]
            if (base + ".weight_scale") in keys:
                scale = h.get_tensor(base + ".weight_scale").float()
                for nk, _ci, _sp in _convert_key(k):
                    mod, _attr = _module_and_attr(nk)
                    fp8_modules[mod] = scale

        # 2) sostituisci quei moduli con Fp8Linear (stessa shape, su meta)
        modules = dict(tr.named_modules())
        for mod_name, scale in fp8_modules.items():
            old = modules.get(mod_name)
            if not isinstance(old, nn.Linear):
                continue
            new = Fp8Linear(old.in_features, old.out_features, bias=old.bias is not None)
            parent_name, child = _module_and_attr(mod_name)
            setattr(modules.get(parent_name, tr), child, new)
        modules = dict(tr.named_modules())               # rinfresca dopo le sostituzioni
        targets = {n for n, _ in tr.named_parameters()} | {n for n, _ in tr.named_buffers()}

        # 3) streaming: assegna i pesi (fp8 ai Fp8Linear, bf16 ai Linear normali) + scale + split QKV
        assigned = 0
        for k in keys:
            if k.endswith((".weight_scale", ".input_scale")):
                continue
            base = k[:-len(".weight")] if k.endswith(".weight") else None
            is_fp8 = bool(base and (base + ".weight_scale") in keys)
            raw = h.get_tensor(k)                         # fp8 (resta fp8!) o bf16
            scale = h.get_tensor(base + ".weight_scale").float() if is_fp8 else None
            for nk, ci, sp in _convert_key(k):
                t = raw if ci is None else torch.chunk(raw, 3, dim=0)[ci]
                if sp == "swap":
                    t = swap_scale_shift(t.to(torch.bfloat16), 0)
                mod, attr = _module_and_attr(nk)
                if nk not in targets:
                    continue
                if is_fp8 and isinstance(modules.get(mod), Fp8Linear) and attr == "weight":
                    set_module_tensor_to_device(tr, nk, "cpu", value=t.contiguous())       # resta fp8
                    set_module_tensor_to_device(tr, mod + ".weight_scale", "cpu", value=scale)
                else:
                    set_module_tensor_to_device(tr, nk, "cpu", value=t.to(torch.bfloat16))
                assigned += 1
    import logging
    logging.info("[flux2_transformer] fp8-forward: %d Fp8Linear, %d tensori assegnati", len(fp8_modules), assigned)
    return tr
