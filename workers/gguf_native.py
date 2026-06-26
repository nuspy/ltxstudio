"""GGUF on-the-fly dequant per il motore video NATIVO (ltx-core) — porta la tecnica di ComfyUI-GGUF.

Permette di caricare i checkpoint LTX-2.3 in formato **GGUF quantizzato** (es. Q4_K_M 13.5GB invece
del dev-fp8 27.8GB) così il 22B entra nei 32GB: i pesi restano QUANTIZZATI in VRAM e ogni Linear
dequantizza SOLO il proprio peso al volo nel forward, poi lo scarta.

Parti:
- dequant_* : funzioni di dequantizzazione per tipo GGML (Q4_K/Q6_K/…), portate da ComfyUI-GGUF
  (c) City96, Apache-2.0.
- GGMLTensor : tensor che conserva i dati quantizzati + tipo + shape originale.
- gguf_state_dict(path) : legge il .gguf → {nome: GGMLTensor}, rimappando i nomi ComfyUI→ltx-core.
- GGUFLinear : nn.Linear che dequantizza il peso nel forward (sostituisce nn.Linear).
- swap_linears_to_gguf(model, sd) : sostituisce gli nn.Linear del modello con GGUFLinear caricando i
  pesi quantizzati dal state_dict GGUF.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn

import gguf

QK_K = 256
K_SCALE_SIZE = 12
TORCH_COMPATIBLE = (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)


# ───────────────────────── dequant (port ComfyUI-GGUF, City96 Apache-2.0) ─────────────────────────
def _split(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def _get_scale_min(scales):
    n = scales.shape[0]
    scales = scales.view(torch.uint8).reshape((n, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape((n, 8)), mn.reshape((n, 8))


def _bf16(b, bs, ts, dtype=None):
    return (b.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


def _q8_0(b, bs, ts, dtype=None):
    d, x = _split(b, 2)
    return d.view(torch.float16).to(dtype) * x.view(torch.int8)


def _q6_k(b, bs, ts, dtype=None):
    n = b.shape[0]
    ql, qh, scales, d = _split(b, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n, QK_K // 16, 1))
    ql = ql.reshape((n, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n, -1, 32))
    qh = qh.reshape((n, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n, QK_K // 16, -1))
    return (d * q).reshape((n, QK_K))


def _q5_k(b, bs, ts, dtype=None):
    n = b.shape[0]
    d, dmin, scales, qh, qs = _split(b, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(dtype); dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n, -1, 1)); dm = (dmin * m).reshape((n, -1, 1))
    ql = qs.reshape((n, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n, -1, 1, 32)) >> torch.tensor(list(range(8)), device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n, -1, 32)); qh = (qh & 0x01).reshape((n, -1, 32))
    return (d * (ql | (qh << 4)) - dm).reshape((n, QK_K))


def _q4_k(b, bs, ts, dtype=None):
    n = b.shape[0]
    d, dmin, scales, qs = _split(b, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype); dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n, -1, 1)); dm = (dmin * m).reshape((n, -1, 1))
    qs = qs.reshape((n, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n, -1, 32))
    return (d * qs - dm).reshape((n, QK_K))


def _q3_k(b, bs, ts, dtype=None):
    n = b.shape[0]
    hmask, qs, scales, d = _split(b, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = (lscales.reshape((n, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))).reshape((n, 16))
    hscales = (hscales.reshape((n, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))).reshape((n, 16))
    scales = ((lscales & 0x0F) | ((hscales & 0x03) << 4)).to(torch.int8) - 32
    dl = (d * scales).reshape((n, 16, 1))
    ql = qs.reshape((n, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n, -1, 1, 32) >> torch.tensor(list(range(8)), device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n, 16, QK_K // 16)) & 1) ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)
    return (dl * q).reshape((n, QK_K))


_DEQUANT = {
    gguf.GGMLQuantizationType.BF16: _bf16, gguf.GGMLQuantizationType.Q8_0: _q8_0,
    gguf.GGMLQuantizationType.Q6_K: _q6_k, gguf.GGMLQuantizationType.Q5_K: _q5_k,
    gguf.GGMLQuantizationType.Q4_K: _q4_k, gguf.GGMLQuantizationType.Q3_K: _q3_k,
}


def dequantize(data, qtype, oshape, dtype):
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    out = _DEQUANT[qtype](blocks, block_size, type_size, dtype)
    return out.reshape(oshape)


class GGMLTensor(torch.Tensor):
    """Tensor che conserva i dati quantizzati grezzi + il tipo GGML + la shape originale."""
    def __new__(cls, data, tensor_type, tensor_shape):
        t = super().__new__(cls, data)
        return t

    def __init__(self, data, tensor_type, tensor_shape):
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape


def dequantize_tensor(t, dtype):
    qtype = getattr(t, "tensor_type", None)
    oshape = getattr(t, "tensor_shape", t.shape)
    if qtype in TORCH_COMPATIBLE:
        return torch.Tensor(t).to(dtype)
    if qtype in _DEQUANT:
        return dequantize(torch.Tensor(t).data, qtype, oshape, dtype).to(dtype)
    new = gguf.quants.dequantize(torch.Tensor(t).cpu().numpy(), qtype)   # fallback lento
    return torch.from_numpy(new).to(t.device, dtype=dtype)


# ───────────────────────── loader GGUF → state_dict quantizzato ─────────────────────────
def _orig_shape(reader, name):
    field = reader.get_field(f"comfy.gguf.orig_shape.{name}")
    if field is None:
        return None
    return torch.Size(tuple(int(field.parts[field.data[i]][0]) for i in range(1, len(field.data))))


def gguf_state_dict(path: str, rename_map: dict | None = None) -> dict:
    """Legge il .gguf → {nome_ltx: GGMLTensor}. Applica rename ComfyUI→ltx-core se fornito."""
    reader = gguf.GGUFReader(path)
    sd = {}
    for tensor in reader.tensors:
        name = tensor.name
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            data = torch.from_numpy(tensor.data)
        shape = _orig_shape(reader, name) or torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
        if tensor.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            data = data.view(*shape)
        gt = GGMLTensor(data, tensor_type=tensor.tensor_type, tensor_shape=shape)
        if len(shape) <= 1 and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            gt = dequantize_tensor(gt, torch.float32)
        if rename_map:
            name = rename_map.get(name, name)
        sd[name] = gt
    return sd


# ───────────────────────── GGUFLinear (dequant on-the-fly nel forward) ─────────────────────────
class GGUFLinear(nn.Module):
    """nn.Linear con peso GGML quantizzato: dequantizza il peso nel forward, poi lo scarta.
    Sostituisce nn.Linear nel modello ltx-core (pattern identico a FP8Linear)."""
    def __init__(self, in_features, out_features, bias=True, compute_dtype=torch.bfloat16):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.compute_dtype = compute_dtype
        self.weight = None          # GGMLTensor (quantizzato), assegnato dal loader
        self.bias = None

    def forward(self, x):
        w = dequantize_tensor(self.weight.to(x.device), self.compute_dtype)
        b = dequantize_tensor(self.bias.to(x.device), self.compute_dtype) if self.bias is not None else None
        out = torch.nn.functional.linear(x.to(self.compute_dtype), w, b)
        del w
        return out.to(x.dtype)
