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


class Nvfp4Linear(torch.nn.Module):
    """Linear con peso nvfp4 (packed uint8 + block_scale fp8 + global_scale f32): dequant on-the-fly nel
    forward. Il TE gira UNA volta per prompt → costo dequant trascurabile; RAM ~1/4 del bf16."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool):
        super().__init__()
        self.in_features = in_features; self.out_features = out_features
        self.register_buffer("packed", None); self.register_buffer("block_scale", None)
        self.register_buffer("global_scale", None)
        self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16)) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        # cast-on-forward (block-swap): packed/scale vivono su CPU, dequant su x.device 1 layer alla volta
        w = dequant_nvfp4_weight(self.packed.to(x.device), self.block_scale.to(x.device),
                                 self.global_scale.to(x.device), self.out_features, self.in_features)
        b = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w.to(x.dtype), b)


def _set_submodule(root, dotted: str, new):
    parent = root
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def load_flux2_text_encoder(config_dir: str, te_file: str = FLUX2_TE_FILE):
    """Text encoder FLUX.2 (Mistral3) fp8/nvfp4-FORWARD: i pesi restano quantizzati (nvfp4/fp8, ~11GB) e
    si dequantizzano on-the-fly nel forward (Nvfp4Linear/Fp8Linear). Niente bf16 intero (~30-46GB) che
    sommato al transformer fp8 saturava la RAM. init su meta + assign RAW (zero copie bf16).
    Shim transformers<5: model_type 'mistral3'→'mistral', sliding_window=full (come Ernie)."""
    import json
    import os
    import torch
    import transformers
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import Mistral3Model, Mistral3Config, MinistralForCausalLM
    from inference_server.backends.flux2_transformer import Fp8Linear
    if not hasattr(transformers, "Ministral3ForCausalLM"):
        transformers.Ministral3ForCausalLM = MinistralForCausalLM
    cfg = json.load(open(os.path.join(config_dir, "config.json")))
    tc = cfg.get("text_config", {})
    if tc.get("model_type") == "ministral3":
        tc["model_type"] = "ministral"
    if tc.get("sliding_window") is None:
        tc["sliding_window"] = tc.get("max_position_embeddings", 262144)
    with init_empty_weights():
        te = Mistral3Model(Mistral3Config(**cfg))
    mods = dict(te.named_modules())

    def _target(k: str) -> str | None:
        if k in mods:
            return k
        rk = ("language_model." + k[6:]) if k.startswith("model.") else k   # remap come Ernie
        return rk if rk in mods else None

    with safe_open(te_file, framework="pt") as h:
        keys = set(h.keys())
        qbases = {k[:-len(".comfy_quant")] for k in keys if k.endswith(".comfy_quant")}
        # 1) sostituisci i Linear quantizzati con Nvfp4Linear/Fp8Linear
        fmt_of: dict[str, str] = {}
        for base in qbases:
            mod_name = _target(base)
            old = mods.get(mod_name) if mod_name else None
            if not isinstance(old, torch.nn.Linear):
                continue
            fmt = json.loads(bytes(h.get_tensor(base + ".comfy_quant").tolist()).decode("utf-8")).get("format")
            fmt_of[mod_name] = fmt
            has_b = old.bias is not None
            new = (Nvfp4Linear(old.in_features, old.out_features, has_b) if fmt == "nvfp4"
                   else Fp8Linear(old.in_features, old.out_features, has_b))
            _set_submodule(te, mod_name, new)
        mods = dict(te.named_modules())
        params = {n for n, _ in te.named_parameters()} | {n for n, _ in te.named_buffers()}
        # 2) carica i pesi: quantizzati RAW nei layer custom; il resto bf16 via set_module_tensor
        assigned = 0
        for k in keys:
            if k.endswith((".comfy_quant", ".weight_scale", ".weight_scale_2", ".tekken_model", ".input_scale")):
                continue
            base = k[:-len(".weight")] if k.endswith(".weight") else None
            if base and base in qbases:
                mod_name = _target(base)
                m = mods.get(mod_name) if mod_name else None
                if isinstance(m, Nvfp4Linear):
                    m.packed = h.get_tensor(base + ".weight")
                    m.block_scale = h.get_tensor(base + ".weight_scale")
                    m.global_scale = h.get_tensor(base + ".weight_scale_2").float()
                    assigned += 1
                elif isinstance(m, Fp8Linear):
                    set_module_tensor_to_device(te, mod_name + ".weight", "cpu", value=h.get_tensor(k))
                    set_module_tensor_to_device(te, mod_name + ".weight_scale", "cpu",
                                                value=h.get_tensor(base + ".weight_scale").float())
                    assigned += 1
                continue
            name = _target(k)                              # plain (embed/norm) → bf16
            if name and name in params:
                set_module_tensor_to_device(te, name, "cpu", value=h.get_tensor(k).to(torch.bfloat16))
                assigned += 1
    # Il checkpoint TE FLUX.2 contiene SOLO il language_model: il vision_tower del Mistral3 (multimodale)
    # resta su meta e non serve per l'encoding del testo. Lo materializzo a zero (inerte) per consentire
    # .to()/sequential offload senza errori "copy from meta".
    _pd = dict(te.named_parameters()); _bd = dict(te.named_buffers())
    metas = [n for n, p in _pd.items() if p.is_meta] + [n for n, b in _bd.items() if b.is_meta]
    for name in metas:
        ref = _pd.get(name)
        if ref is None:
            ref = _bd.get(name)
        set_module_tensor_to_device(te, name, "cpu", value=torch.zeros(ref.shape, dtype=ref.dtype))
    import logging
    logging.info("[flux2_te] fp8/nvfp4-forward: %d layer quantizzati, %d assegnati, %d meta(vision)→zero",
                 len(fmt_of), assigned, len(metas))
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
