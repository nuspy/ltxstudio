"""FLUX.2 text encoder (Mistral Small 3, pruned) su 32GB — dequant NVFP4 → bf16 in PURO torch.

Perché: ComfyUI carica il TE FLUX.2 dal file fp4 locale `mistral_3_small_flux2.safetensors` (~12GB,
formato **nvfp4**: weight uint8 packed e2m1 + weight_scale fp8_e4m3 a blocchi di 16 in layout cublas
"blocked"/swizzled + weight_scale_2 global float32) usando kernel compilati `comfy_kitchen` (non
portabili). Alternativa percorribile SENZA download (no repo bf16 da 48GB) e senza kernel ComfyUI:
dequantizzare nvfp4→bf16 una volta in puro torch, poi caricare il Mistral text-only con
`enable_sequential_cpu_offload` (picco VRAM ~1 layer → entra nei 32GB).

⚠️ STATO: la math di dequant (LUT e2m1, scala-blocco, scala globale) è qui; resta da VALIDARE sulla
GPU Blackwell l'inversione esatta dello swizzle `to_blocked` (layout cublas dei block-scale) confrontando
con un'immagine di riferimento. Vedi Task "FLUX.2 nvfp4 TE — validazione live". Riferimento math:
ComfyUI `comfy/float.py` (stochastic_round_quantize_nvfp4_block, to_blocked) e `comfy/quant_ops.py`.

API: dequant_nvfp4_file(path) -> {clean_name: bf16_tensor}   (per scrivere un TE bf16 o caricarlo).
"""
from __future__ import annotations

import torch

# e2m1: 3 bit magnitudine (indice 0..7) + 1 bit segno. Valori rappresentabili nvfp4.
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
BLOCK = 16


def _decode_e2m1(nibbles: torch.Tensor) -> torch.Tensor:
    """nibble uint8 (0..15) → valore fp4 e2m1 con segno (bit 3)."""
    mag = _E2M1.to(nibbles.device)[(nibbles & 0x7).long()]
    sign = torch.where((nibbles & 0x8) > 0, -1.0, 1.0)
    return mag * sign


def _unswizzle_block_scales(scales_blocked: torch.Tensor, rows: int, n_blocks: int) -> torch.Tensor:
    """Inverte `to_blocked` (layout cublas) → scale per-blocco in layout naturale [rows, n_blocks].
    to_blocked: padded.view(n_row_blocks,128,n_col_blocks,4).permute(0,2,1,3) →
                .reshape(-1,4,32,4).transpose(1,2).reshape(-1,32,16).
    Qui si applica l'inverso esatto (da validare live su Blackwell)."""
    import math
    nrb = math.ceil(rows / 128)
    ncb = math.ceil(n_blocks / 4)
    flat = scales_blocked.reshape(-1)
    # inverso di: reshape(-1,32,16) <- transpose(1,2) <- reshape(-1,4,32,4)
    t = flat.reshape(-1, 32, 16).reshape(-1, 32, 4, 4).transpose(1, 2).reshape(nrb, ncb, 128, 4)
    nat = t.permute(0, 2, 1, 3).reshape(nrb * 128, ncb * 4)
    return nat[:rows, :n_blocks]


def dequant_nvfp4_weight(packed: torch.Tensor, block_scale_fp8: torch.Tensor,
                         global_scale: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """packed uint8 [out, in/2] → bf16 [out, in]. block_scale in layout blocked (da unswizzle)."""
    high = (packed >> 4) & 0xF                              # elemento pari (vedi packing: x[0::2]<<4)
    low = packed & 0xF                                     # elemento dispari
    nib = torch.stack([high, low], dim=-1).reshape(out_features, in_features)
    fp4 = _decode_e2m1(nib).float()
    nblk = in_features // BLOCK
    bs = _unswizzle_block_scales(block_scale_fp8.float(), out_features, nblk)   # [out, nblk]
    bs = bs.repeat_interleave(BLOCK, dim=1)[:, :in_features]
    w = fp4 * bs * float(global_scale)
    return w.to(torch.bfloat16)


FLUX2_TE_FILE = r"E:\AI\Models\TextEncoders\Flux\mistral_3_small_flux2.safetensors"


def load_flux2_text_encoder(config_dir: str, te_file: str = FLUX2_TE_FILE):
    """Costruisce il text encoder FLUX.2 (Mistral3) dal config + pesi nvfp4 dequantizzati.
    MEMORY-BOUNDED: init su meta (0 RAM) + dequant-and-assign per-tensore in STREAMING (un layer alla
    volta, liberato subito) → picco ~1 layer invece di ~50GB (il vecchio dict completo + init + copia
    saturava i 128GB). Ritorna il modello bf16 su CPU, pronto per sequential offload su GPU.
    Shim transformers<5: model_type 'mistral3'→'mistral', sliding_window=full (come Ernie)."""
    import json
    import os
    import torch
    import transformers
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers import Mistral3Model, Mistral3Config, MinistralForCausalLM
    if not hasattr(transformers, "Ministral3ForCausalLM"):
        transformers.Ministral3ForCausalLM = MinistralForCausalLM
    cfg = json.load(open(os.path.join(config_dir, "config.json")))
    tc = cfg.get("text_config", {})
    if tc.get("model_type") == "ministral3":
        tc["model_type"] = "ministral"
    if tc.get("sliding_window") is None:
        tc["sliding_window"] = tc.get("max_position_embeddings", 262144)
    with init_empty_weights():                              # modello su meta device: ZERO RAM all'init
        te = Mistral3Model(Mistral3Config(**cfg))
    targets = {n for n, _ in te.named_parameters()} | {n for n, _ in te.named_buffers()}

    def _target_name(k: str) -> str | None:
        if k in targets:
            return k
        rk = ("language_model." + k[6:]) if k.startswith("model.") else k   # remap come Ernie
        return rk if rk in targets else None

    assigned = 0
    for k, tensor in _iter_nvfp4_tensors(te_file):         # genera UN tensore bf16 alla volta
        name = _target_name(k)
        if name is not None:
            set_module_tensor_to_device(te, name, "cpu", value=tensor)      # assegna e libera
            assigned += 1
        del tensor
    import logging
    logging.info("[flux2_te] assegnati %d/%d tensori (streaming, memory-bounded)", assigned, len(targets))
    return te


def _iter_nvfp4_tensors(path: str):
    """Generatore: legge il file nvfp4 e produce (name, bf16_tensor) UN tensore alla volta, dequantizzando
    on-demand (peso nvfp4 → dequant; embed/norm plain → copia). Memory-bounded: non costruisce mai il dict
    completo (~24GB). Usato dal loader streaming; safe_open legge dal file su disco, non in RAM."""
    import json
    from safetensors import safe_open
    with safe_open(path, framework="pt") as h:
        keys = set(h.keys())
        quant_bases = {k[:-len(".comfy_quant")] for k in keys if k.endswith(".comfy_quant")}
        for k in keys:
            if k.endswith((".comfy_quant", ".weight_scale", ".weight_scale_2", ".tekken_model")):
                continue
            base = k[:-len(".weight")] if k.endswith(".weight") else None
            if base and base in quant_bases:                # peso quantizzato → dequant secondo il format
                fmt = json.loads(bytes(h.get_tensor(base + ".comfy_quant").tolist()).decode("utf-8")).get("format")
                w = h.get_tensor(base + ".weight")
                if fmt == "nvfp4":                          # U8 packed + block-scale fp8 + global-scale f32
                    bs = h.get_tensor(base + ".weight_scale")
                    g = h.get_tensor(base + ".weight_scale_2")
                    yield k, dequant_nvfp4_weight(w, bs, g, w.shape[0], w.shape[1] * 2)
                elif fmt == "float8_e4m3fn":                # fp8 + scala globale scalare → w*scale
                    s = h.get_tensor(base + ".weight_scale").float()
                    yield k, (w.to(torch.float32) * s).to(torch.bfloat16)
                else:                                       # formato ignoto → copia best-effort
                    yield k, w.to(torch.bfloat16)
            else:                                           # plain (embed/norm) → copia
                yield k, h.get_tensor(k).to(torch.bfloat16)


def dequant_nvfp4_file(path: str) -> dict:
    """Come _iter_nvfp4_tensors ma materializza tutto in un dict {name: bf16}. ⚠️ ~24GB in RAM — usare
    solo per scrivere un TE bf16 su disco, NON nel path caldo (il loader usa lo streaming)."""
    return {k: v for k, v in _iter_nvfp4_tensors(path)}
