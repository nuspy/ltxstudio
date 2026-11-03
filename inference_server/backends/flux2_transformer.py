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
        # cast-on-forward (block-swap stile ComfyUI): il peso fp8 vive su CPU, qui va su x.device 1 layer
        # alla volta (dequant→bf16), poi il temporaneo è liberato. Picco VRAM = 1 layer.
        if getattr(self, "exact_scale", False):
            # scala VERA del file (fp8 «scalato» ComfyUI): (fp8 -> fp32) * scala -> bf16, uguale bit per bit alla cache dequantizzata (provato su 4 pesi del
            # realDream: il prodotto tutto in bf16 differisce di ~1e-3)
            w = (self.weight.to(device=x.device).float() * self.weight_scale.to(device=x.device)).to(x.dtype)
        else:
            w = self.weight.to(device=x.device, dtype=x.dtype) * self.weight_scale.to(device=x.device, dtype=x.dtype)
        b = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


class Nvfp4Linear(nn.Module):
    """Linear con peso NVFP4 (U8 packed e2m1 + weight_scale a blocchi fp8 swizzled + weight_scale_2 F32):
    dequant on-the-fly nel forward via comfy_kitchen (block-swap come Fp8Linear). Formato ufficiale BFL
    FLUX.2-dev-NVFP4 (= comfy_kitchen/nunchaku). Modalità DEQUANT (matmul bf16 pieno, niente quant delle
    attivazioni): il più sicuro — riusa il block-swap e non introduce l'input_scale del file (kernel puro)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer("weight_scale", torch.empty(out_features, in_features // 16,
                                                        dtype=torch.float8_e4m3fn))   # swizzled as-stored
        self.register_buffer("weight_scale_2", torch.ones((), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import comfy_kitchen as ck
        import torch.nn.functional as F
        dev = x.device
        w = ck.dequantize_nvfp4(self.weight.to(dev), self.weight_scale_2.to(dev),
                                self.weight_scale.to(dev), output_type=torch.bfloat16)
        w = w[: self.out_features, : self.in_features]
        b = self.bias.to(device=dev, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x.to(torch.bfloat16), w, b).to(x.dtype)


_SWAP_LINEAR_MIN_BYTES = 16 * 2**20      # sotto i 16 MB un Linear e' "leggero" e sta stabilmente su GPU


def _swap_plain_linear(m, compute_device):
    """Linear NON quantizzato ma grosso (checkpoint FLUX.2-dev «nvfp4-mixed»: solo gli MLP sono nvfp4, attention e
    proiezioni sono bf16 = ~25 GB). Messi tutti su GPU come moduli «leggeri» riempivano i 32 GB (alloc 27 GB prima
    ancora di iniziare) e Windows spillava il resto nella RAM condivisa: commit +27 GB, app chiusa dal guardiano
    (giro 31/10, crea_flux_07). Qui peso su CPU e copiato su GPU solo per la durata del forward, come i Linear
    quantizzati; il bias (piccolo) resta su GPU."""
    import torch.nn.functional as F
    if m.bias is not None:
        m.bias.data = m.bias.data.to(compute_device)

    def forward(x, _m=m, _dev=compute_device):
        w = _m.weight.to(device=x.device, dtype=x.dtype, non_blocking=True)
        b = _m.bias.to(device=x.device, dtype=x.dtype) if _m.bias is not None else None
        return F.linear(x, w, b)
    m.forward = forward
    m._ls_swap_linear = True


def enable_blockswap(model, compute_device):
    """Block-swap stile ComfyUI (no accelerate): i pesi quantizzati (grossi) restano su CPU e il loro
    forward li sposta su GPU 1 layer alla volta; i moduli leggeri (norm/embed/modulation/bias) vanno su
    GPU stabilmente. Picco VRAM = moduli leggeri + 1 layer pesante. RAM tiene i pesi fp8."""
    for m in model.modules():
        is_q = isinstance(m, (Fp8Linear, Nvfp4Linear))
        if (not is_q and type(m) is nn.Linear and m.weight is not None and not m.weight.is_meta
                and m.weight.numel() * m.weight.element_size() >= _SWAP_LINEAR_MIN_BYTES):
            _swap_plain_linear(m, compute_device)           # checkpoint «mixed»: Linear bf16 grossi restano su CPU
            continue
        if is_q:
            if getattr(m, "bias", None) is not None:        # bias piccolo → GPU; weight/packed restano CPU
                m.bias.data = m.bias.data.to(compute_device)
        else:                                               # modulo leggero → param/buffer diretti su GPU
            for p in m.parameters(recurse=False):
                pd = p.data.to(compute_device)
                if pd.dtype == torch.float32:               # uniforma a bf16 (evita mat1≠mat2 nel forward)
                    pd = pd.to(torch.bfloat16)
                p.data = pd
            for bn, bv in list(m.named_buffers(recurse=False)):
                if bv is not None:
                    m.register_buffer(bn, bv.to(compute_device))
    return model


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

    # 1) quali MODULI diffusers vanno quantizzati. fp8: weight fp8 + weight_scale scalare. NVFP4 (BFL
    #    flux2-dev-nvfp4): weight U8 + weight_scale 2D (blocchi) + weight_scale_2 scalare (+ input_scale).
    with safe_open(ckpt_path, framework="pt") as h:
        keys = set(h.keys())
        fp8_modules: dict[str, torch.Tensor] = {}        # module_name → weight_scale (scalar)
        nvfp4_modules: set[str] = set()                  # module_name dei Linear NVFP4
        for k in keys:
            if not k.endswith(".weight"):
                continue
            base = k[:-len(".weight")]
            if (base + ".weight_scale_2") in keys:       # NVFP4 (ha la scala globale)
                for nk, _ci, _sp in _convert_key(k):
                    nvfp4_modules.add(_module_and_attr(nk)[0])
            elif (base + ".weight_scale") in keys:       # fp8 (solo scala scalare)
                scale = h.get_tensor(base + ".weight_scale").float()
                for nk, _ci, _sp in _convert_key(k):
                    fp8_modules[_module_and_attr(nk)[0]] = scale

        # 2) sostituisci i moduli con Fp8Linear / Nvfp4Linear (stessa shape logica, su meta)
        modules = dict(tr.named_modules())
        for mod_name, scale in fp8_modules.items():
            old = modules.get(mod_name)
            if isinstance(old, nn.Linear):
                new = Fp8Linear(old.in_features, old.out_features, bias=old.bias is not None)
                parent_name, child = _module_and_attr(mod_name)
                setattr(modules.get(parent_name, tr), child, new)
        for mod_name in nvfp4_modules:
            old = modules.get(mod_name)
            if isinstance(old, nn.Linear):
                new = Nvfp4Linear(old.in_features, old.out_features, bias=old.bias is not None)
                parent_name, child = _module_and_attr(mod_name)
                setattr(modules.get(parent_name, tr), child, new)
        modules = dict(tr.named_modules())               # rinfresca dopo le sostituzioni
        targets = {n for n, _ in tr.named_parameters()} | {n for n, _ in tr.named_buffers()}

        # 3) streaming: assegna i pesi + scale. NB: split QKV NON avviene sui NVFP4 di FLUX.2 (i qkv sono
        #    già Linear separati nel formato BFL nvfp4 → ci_idx sempre None per gli U8).
        assigned = 0
        for k in keys:
            if k.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
                continue
            base = k[:-len(".weight")] if k.endswith(".weight") else None
            is_nvfp4 = bool(base and (base + ".weight_scale_2") in keys)
            is_fp8 = bool(base and not is_nvfp4 and (base + ".weight_scale") in keys)
            raw = h.get_tensor(k)                         # U8 (nvfp4) / fp8 / bf16
            for nk, ci, sp in _convert_key(k):
                t = raw if ci is None else torch.chunk(raw, 3, dim=0)[ci]
                if sp == "swap":
                    t = swap_scale_shift(t.to(torch.bfloat16), 0)
                mod, attr = _module_and_attr(nk)
                if nk not in targets:
                    continue
                _m = modules.get(mod)
                if is_nvfp4 and isinstance(_m, Nvfp4Linear) and attr == "weight":
                    set_module_tensor_to_device(tr, nk, "cpu", value=t.contiguous())        # resta U8
                    set_module_tensor_to_device(tr, mod + ".weight_scale", "cpu",
                                                value=h.get_tensor(base + ".weight_scale"))
                    set_module_tensor_to_device(tr, mod + ".weight_scale_2", "cpu",
                                                value=h.get_tensor(base + ".weight_scale_2").float())
                elif is_fp8 and isinstance(_m, Fp8Linear) and attr == "weight":
                    set_module_tensor_to_device(tr, nk, "cpu", value=t.contiguous())         # resta fp8
                    set_module_tensor_to_device(tr, mod + ".weight_scale", "cpu",
                                                value=h.get_tensor(base + ".weight_scale").float())
                else:
                    # dtype=bf16 ESPLICITO: senza, il valore veniva ri-castato a float32 (dtype del modulo
                    # da from_config) → forward "mat1/mat2 BFloat16 != float".
                    set_module_tensor_to_device(tr, nk, "cpu", value=t.to(torch.bfloat16),
                                                dtype=torch.bfloat16)
                assigned += 1
    import logging
    logging.info("[flux2_transformer] %d Fp8Linear + %d Nvfp4Linear, %d tensori assegnati",
                 len(fp8_modules), len(nvfp4_modules), assigned)
    return tr


# ---------------------------------------------------------------------------
# FLUX.2 KLEIN (edit/inpaint) — variante DISTILLATA, fp8 PURO (no weight_scale),
# prefisso `model.diffusion_model.`, encoder Qwen3 (non Mistral). La config è
# derivata DINAMICAMENTE dai dim del checkpoint, così funziona sia per la 4B
# ufficiale (hidden 3072, TE qwen3_4b) sia per la miraclein 8B (hidden 4096,
# TE qwen3_8b) senza file di config dedicati.
# ---------------------------------------------------------------------------

def _strip_prefix(k: str) -> str:
    return k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k


def klein_config_from_header(ckpt_path: str) -> dict:
    """Ricava la config di Flux2Transformer2DModel misurando i tensori del checkpoint Klein.
    Robusto a 4B/8B e a futuri tagli: niente valori hardcoded sui dim."""
    from safetensors import safe_open
    shapes: dict[str, list[int]] = {}
    dbl, sgl = set(), set()
    import re
    with safe_open(ckpt_path, framework="pt") as h:
        for k in h.keys():
            b = _strip_prefix(k)
            shapes[b] = list(h.get_slice(k).get_shape())
            m = re.search(r"double_blocks\.(\d+)\.", b)
            if m:
                dbl.add(int(m.group(1)))
            m = re.search(r"single_blocks\.(\d+)\.", b)
            if m:
                sgl.add(int(m.group(1)))
    inner = shapes["img_in.weight"][0]                       # 4096 (8B) / 3072 (4B)
    in_ch = shapes["img_in.weight"][1]                       # 128
    jad = shapes["txt_in.weight"][1]                         # 12288 / 7680
    head_dim = shapes["double_blocks.0.img_attn.norm.query_norm.weight"][0]   # 128
    mlp_in = shapes["double_blocks.0.img_mlp.0.weight"][0]   # 24576 = 2*mlp_ratio*inner (GEGLU)
    mlp_ratio = round((mlp_in / 2) / inner, 4)               # → 3.0
    return {
        "patch_size": 1, "in_channels": in_ch, "out_channels": None,
        "num_layers": len(dbl), "num_single_layers": len(sgl),
        "attention_head_dim": head_dim, "num_attention_heads": inner // head_dim,
        "joint_attention_dim": jad, "timestep_guidance_channels": 256,
        "mlp_ratio": mlp_ratio, "axes_dims_rope": [32, 32, 32, 32],
        "rope_theta": 2000, "eps": 1e-6,
        "guidance_embeds": ("guidance_in.in_layer.weight" in shapes),
    }


def is_scaled_fp8_klein(ckpt_path: str) -> bool:
    """True per un Klein con pesi F8_E4M3 + `weight_scale` scalare (fp8 «scalato» ComfyUI, es. realDream_klein9B), che `load_flux2_klein_transformer`
    carica DIRETTAMENTE (Fp8Linear, ~9 GB) senza passare dalla dequantizzazione in bf16 (18 GB: transformer 17,8 GB + encoder 15,3 GB = 33 GB misurati,
    piano «offload» e processo a 60 GB: giro A3 del 03/11). Esclusi i Dev e i nvfp4 (pesi U8 / weight_scale_2)."""
    import json
    import struct
    try:
        with open(ckpt_path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        keys = [k for k in hdr if k != "__metadata__"]
        w8 = sum(1 for k in keys if k.endswith(".weight") and hdr[k].get("dtype") == "F8_E4M3")
        sc = sum(1 for k in keys if k.endswith(".weight_scale"))
        nv = any(k.endswith(".weight_scale_2") for k in keys)
        if not w8 or not sc or nv:
            return False
        cfg = klein_config_from_header(ckpt_path)
        return cfg["joint_attention_dim"] in (7680, 12288)               # Qwen3-4B / Qwen3-8B: Klein; i Dev (Mistral, 15360) no
    except Exception:                                                    # noqa: BLE001
        return False


def load_flux2_klein_transformer(ckpt_path: str, cfg: dict | None = None):
    """Carica il transformer Klein (fp8 puro) come Flux2Transformer2DModel con Fp8Linear (scale=1.0)
    per TUTTI i Linear fp8 → block-swap memory-bounded come per Dev. Config dedotta dai dim se non data."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import Flux2Transformer2DModel
    from diffusers.loaders.single_file_utils import swap_scale_shift
    from safetensors import safe_open

    if cfg is None:
        cfg = klein_config_from_header(ckpt_path)
    with init_empty_weights():
        tr = Flux2Transformer2DModel.from_config(cfg)

    with safe_open(ckpt_path, framework="pt") as h:
        raw_keys = list(h.keys())
        # mappa base(senza prefisso) → chiave reale nel file
        base2real = {_strip_prefix(k): k for k in raw_keys}
        # FP8 «SCALATO» (formato ComfyUI, es. realDream_klein9B): accanto a ogni `X.weight` fp8 ci sono `X.weight_scale` (scalare F32), `X.input_scale` e
        # `X.comfy_quant` (U8). Peso vero = fp8 * weight_scale. Prima quelle chiavi finivano nel mapping dei pesi e il caricamento moriva con «chunk
        # expects at least a 1-dimensional tensor» (qkv.weight_scale e scalare); la scala veniva comunque ignorata (1.0). Ora la scala va nel buffer
        # `weight_scale` di Fp8Linear (che la moltiplica nel forward); input_scale e comfy_quant non servono (compute bf16).
        file_scales: dict[str, float] = {}
        for b in [b for b in base2real if b.endswith(".weight_scale")]:
            file_scales[b[:-len(".weight_scale")]] = float(h.get_tensor(base2real[b]).float().item())
        for b in [b for b in base2real if b.endswith((".weight_scale", ".input_scale", ".comfy_quant"))]:
            del base2real[b]
        weight_bases = {b for b in base2real if b.endswith(".weight")}

        # 1) quali moduli diffusers diventano Fp8Linear: ogni weight in dtype fp8
        fp8_modules: set[str] = set()
        for b in weight_bases:
            t0 = h.get_slice(base2real[b])
            if "float8" in str(t0.get_dtype()).lower() or t0.get_dtype() in ("F8_E4M3", "F8_E5M2"):
                for nk, _ci, _sp in _convert_key(b):
                    fp8_modules.add(_module_and_attr(nk)[0])

        modules = dict(tr.named_modules())
        for mod_name in fp8_modules:
            old = modules.get(mod_name)
            if not isinstance(old, nn.Linear):
                continue
            # su 'meta': prima il costruttore allocava davvero un buffer fp8 vuoto da out*in byte per OGNI Linear (9,4 GB per la Klein 9B, poi sostituiti dai
            # pesi veri): commit del solo transformer 22,5 GB invece di ~10 (giro A del 03/11, Klein 9B a 59 GB di processo)
            with torch.device("meta"):
                new = Fp8Linear(old.in_features, old.out_features, bias=old.bias is not None)
            # moduli top-level (es. proj_out, anch'esso fp8 nella miraclein) non hanno parent → tr
            parent_name, child = (mod_name.rsplit(".", 1) if "." in mod_name else ("", mod_name))
            setattr(modules.get(parent_name, tr), child, new)
        modules = dict(tr.named_modules())
        targets = {n for n, _ in tr.named_parameters()} | {n for n, _ in tr.named_buffers()}

        # 2) streaming: pesi fp8 ai Fp8Linear (scale 1.0), il resto bf16
        assigned = 0
        for b, real in base2real.items():
            raw = h.get_tensor(real)
            for nk, ci, sp in _convert_key(b):
                t = raw if ci is None else torch.chunk(raw, 3, dim=0)[ci]
                if sp == "swap":
                    t = swap_scale_shift(t.to(torch.bfloat16), 0)
                mod, attr = _module_and_attr(nk)
                if nk not in targets:
                    continue
                if isinstance(modules.get(mod), Fp8Linear) and attr == "weight":
                    set_module_tensor_to_device(tr, nk, "cpu", value=t.contiguous())   # resta fp8
                    set_module_tensor_to_device(tr, mod + ".weight_scale", "cpu",
                                                value=torch.tensor(file_scales.get(b[:-len(".weight")], 1.0), dtype=torch.float32))
                    if b[:-len(".weight")] in file_scales:
                        modules[mod].exact_scale = True
                else:
                    set_module_tensor_to_device(tr, nk, "cpu", value=t.to(torch.bfloat16),
                                                dtype=torch.bfloat16)
                assigned += 1
    import logging
    logging.info("[flux2_klein] cfg=%s | %d Fp8Linear, %d tensori", cfg, len(fp8_modules), assigned)
    return tr
