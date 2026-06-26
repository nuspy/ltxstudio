"""GGUF on-the-fly dequant per il motore video NATIVO (ltx-core) — porta la tecnica di ComfyUI-GGUF.

Permette di caricare il transformer LTX-2.3 in **GGUF quantizzato** (Q4_K_M ~14GB invece del dev-fp8
27.8GB) così il 22B entra nei 32GB: i pesi restano QUANTIZZATI in VRAM e ogni Linear dequantizza SOLO
il proprio peso al volo nel forward, poi lo scarta. Dequant (c) City96, Apache-2.0.

Uso (dal worker, quando il checkpoint transformer è un .gguf):
  from workers import gguf_native as GG
  GG.patch_stage_to_gguf(pipe.stage_1, gguf_path, config_source_safetensors)
  GG.patch_stage_to_gguf(pipe.stage_2, gguf_path, config_source_safetensors)
Gli altri componenti (Gemma, vae, vocoder…) restano sul checkpoint fp8 con offload.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _dequantize(data, qtype, oshape, dtype):
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    return _DEQUANT[qtype](blocks, block_size, type_size, dtype).reshape(oshape)


def is_quantized(t):
    return getattr(t, "tensor_type", None) not in TORCH_COMPATIBLE


def dequantize_tensor(t, dtype):
    qtype = getattr(t, "tensor_type", None)
    # strip del subclass SENZA perdere device (torch.Tensor(t) ricreerebbe su CPU/meta)
    raw = t.as_subclass(torch.Tensor) if isinstance(t, GGMLTensor) else t
    if qtype in TORCH_COMPATIBLE:
        return raw.to(dtype)
    oshape = getattr(t, "tensor_shape", None) or raw.shape
    if qtype in _DEQUANT:
        return _dequantize(raw, qtype, oshape, dtype).to(dtype)
    new = gguf.quants.dequantize(raw.cpu().numpy(), qtype)
    return torch.from_numpy(new).to(raw.device, dtype=dtype)


# ───────────────────────── GGMLTensor (preserva tensor_type dopo .to) — port ComfyUI ─────────────────────────
class GGMLTensor(torch.Tensor):
    def __init__(self, *args, tensor_type, tensor_shape, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape

    def __new__(cls, *args, tensor_type, tensor_shape, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        return new

    def clone(self, *a, **k):
        return self

    def detach(self, *a, **k):
        return self

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape


# ───────────────────────── loader GGUF → state_dict quantizzato ─────────────────────────
def _orig_shape(reader, name):
    field = reader.get_field(f"comfy.gguf.orig_shape.{name}")
    if field is None:
        return None
    return torch.Size(tuple(int(field.parts[field.data[i]][0]) for i in range(1, len(field.data))))


def gguf_state_dict(path: str, rename_map: dict | None = None) -> dict:
    """Legge il .gguf → {nome: GGMLTensor}. Applica rename se fornito (ComfyUI↔ltx)."""
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


# ───────────────────────── dequant Linear in-place (forward GGUF) ─────────────────────────
def _gguf_linear_forward(self, x):
    """forward che dequantizza il peso GGML on-the-fly. Montato come metodo d'istanza sull'nn.Linear
    ESISTENTE (mutazione in-place) così TUTTI i riferimenti — incluso il TransformerArgsPreprocessor
    che cattura patchify_proj per riferimento al costruttore — usano lo stesso oggetto."""
    with torch.no_grad():
        w = dequantize_tensor(self.weight.to(x.device), self._gguf_dtype)
        bias = self.bias
        if bias is not None:
            bias = dequantize_tensor(bias.to(x.device), self._gguf_dtype).to(x.dtype)
        out = F.linear(x, w.to(x.dtype), bias)
        del w
    return out


def swap_and_assign_gguf(model: nn.Module, sd: dict, quantized_names: set) -> nn.Module:
    """MUTA in-place gli nn.Linear con peso quantizzato nel GGUF: assegna il GGMLTensor come peso e
    monta il forward-dequant sull'istanza (NON sostituisce il modulo → i riferimenti catturati, es.
    patchify_proj nel preprocessor, restano validi)."""
    import types as _types
    cnt = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or getattr(mod, "_gguf", False):
            continue
        wk = name + ".weight"
        if wk not in quantized_names:
            continue
        mod.weight = nn.Parameter(sd[wk], requires_grad=False)
        bk = name + ".bias"
        if mod.bias is not None and bk in sd:
            mod.bias = nn.Parameter(torch.Tensor(sd[bk]), requires_grad=False)
        mod._gguf = True
        mod._gguf_dtype = torch.bfloat16
        mod.forward = _types.MethodType(_gguf_linear_forward, mod)
        cnt += 1
    print(f"[gguf] Linear mutati in-place (GGUF dequant)={cnt}", flush=True)
    return model


# ───────────────────────── integrazione ltx-core ─────────────────────────
def patch_stage_to_gguf(stage, gguf_path: str, config_source: str) -> None:
    """Aggancia un DiffusionStage di ltx-core a caricare il transformer dal GGUF (quantizzato,
    dequant on-the-fly). `config_source` = un safetensors con la STESSA architettura (es. il dev-fp8)
    da cui leggere la config del modello. Forza offload_mode=NONE per lo stadio (il GGUF ~14GB entra
    in GPU senza streaming; Gemma/conditioner restano con l'offload della pipeline)."""
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
    from ltx_core.loader.primitives import StateDict
    from ltx_core.loader.module_ops import ModuleOps

    # Idempotente: su pipe in cache (chunking in-process) NON ri-patchare/ri-caricare il GGUF (14GB)
    # ad ogni chunk — altrimenti sd+closure si accumulano → OOM dopo N chunk.
    if getattr(stage, "_gguf_patched", False):
        return

    orig_builder = stage._transformer_builder
    configurator = orig_builder._model_class_configurator

    # rename ComfyUI→ltx (se serve): qui i nomi GGUF combaciano già coi nomi transformer.
    sd = gguf_state_dict(gguf_path)
    quantized = {k for k, v in sd.items() if is_quantized(v)}
    # chiavi gestite dallo swap (peso quantizzato + suo bias): escluse dal load_state_dict
    gguf_keys = set(quantized)
    for k in quantized:
        gguf_keys.add(k[: -len("weight")] + "bias")
    # state_dict NON-quantizzato per il load_state_dict normale (norm/scale_shift/embedding…), in bf16
    # per evitare il mismatch dtype (norm F32 vs input bf16) e dimezzare la memoria.
    nonquant_sd = {k: dequantize_tensor(v, torch.bfloat16)
                   for k, v in sd.items() if k not in gguf_keys}

    class _GGUFLoader:
        """StateDictLoader: config dal safetensors gemello, pesi NON-quantizzati dal GGUF
        (i quantizzati li assegna lo swap)."""
        def __init__(self):
            self._meta = SafetensorsModelStateDictLoader()

        def metadata(self, path):
            return self._meta.metadata(config_source)

        def load(self, path, sd_ops=None, device=None):
            size = sum(v.numel() * v.element_size() for v in nonquant_sd.values())
            dtypes = {v.dtype for v in nonquant_sd.values()}
            return StateDict(sd=nonquant_sd, device=torch.device("cpu"), size=size, dtype=dtypes)

    swap_op = ModuleOps(
        name="gguf_swap_linears",
        matcher=lambda m: True,
        mutator=lambda m: swap_and_assign_gguf(m, sd, quantized),
    )
    new_builder = SingleGPUModelBuilder(
        model_class_configurator=configurator,
        model_path=gguf_path,
        model_loader=_GGUFLoader(),
        module_ops=(swap_op,),
    )
    stage._transformer_builder = new_builder
    from ltx_pipelines.utils.types import OffloadMode
    stage._offload_mode = OffloadMode.NONE
    # niente torch.compile sul transformer GGUF: la compilation traccia con fake/meta tensor e il
    # GGMLTensor (dequant custom on-the-fly) non è tracciabile → "Tensor on device meta".
    if getattr(stage, "_compilation_config", None) is not None:
        stage._compilation_config = None
    # niente policy fp8: _build_transformer farebbe _chain_quantization che ri-swappa i GGUFLinear in
    # FP8Linear (GGUFLinear è sottoclasse di nn.Linear) → pesi persi/meta. Il GGUF gestisce già tutto.
    if getattr(stage, "_quantization", None) is not None:
        stage._quantization = None
    stage._gguf_patched = True
