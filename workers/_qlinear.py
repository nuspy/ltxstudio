"""Linear quantizzati generici per i worker a single-file (Qwen-Image 2.1, riusabili da altri motori).

Formati riconosciuti dalle chiavi del safetensors, modulo per modulo:
  · torchao Float8Tensor (unsloth «FP8»):  <m>._weight_qdata F8_E4M3 [out,in] + <m>._weight_scale F32 [out,1]
  · torchao Int8Tensor   (unsloth «INT8»): <m>._weight_qdata I8 + _weight_scale F32 [out,1] + _weight_zero_point
  · FP8 «pre-cast» senza scala (TE FP8):   <m>.weight F8_E4M3 (scala implicita 1)
  · NVFP4 ComfyUI/BFL («Prism native»):    <m>.weight U8 [out,in/2] + weight_scale F8 [out,in/16] + weight_scale_2
I pesi restano compressi in VRAM e si dequantizzano in bf16 un layer alla volta nel forward (matmul bf16:
qualità identica, niente quantizzazione delle attivazioni). Ogni layer accetta LoRA «al volo»
(y = base(x) + s·B(A(x))), perché PEFT non sa avvolgere moduli personalizzati.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LoraMixin:
    def _lora_init(self):
        self.loras: list = []                        # [(A [r,in], B [out,r], scale)]

    def add_lora(self, a: torch.Tensor, b: torch.Tensor, scale: float):
        self.loras.append((a, b, float(scale)))

    def _lora(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        for a, b, s in self.loras:
            y = y + s * F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype))
        return y


class RowFp8Linear(nn.Module, _LoraMixin):
    """Peso float8_e4m3 con scala per riga opzionale (assente = 1: TE «pre-cast»)."""

    def __init__(self, in_features: int, out_features: int, bias: bool, scaled: bool):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn))
        self.register_buffer("scale", torch.ones(out_features, 1, dtype=torch.float32) if scaled else None)
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16)) if bias else None
        self._lora_init()

    def forward(self, x):
        cd = torch.bfloat16 if x.dtype == torch.float32 else x.dtype
        w = self.weight.to(cd)
        if self.scale is not None:
            w = w * self.scale.to(cd)
        y = F.linear(x.to(cd), w, self.bias.to(cd) if self.bias is not None else None)
        return self._lora(x.to(cd), y).to(x.dtype)


class RowInt8Linear(nn.Module, _LoraMixin):
    """Peso int8 per riga: w = (q - zero_point) * scale."""

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features, 1, dtype=torch.float32))
        self.register_buffer("zero_point", torch.zeros(out_features, 1, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16)) if bias else None
        self._lora_init()

    def forward(self, x):
        cd = torch.bfloat16 if x.dtype == torch.float32 else x.dtype
        w = (self.weight.to(cd) - self.zero_point.to(cd)) * self.scale.to(cd)
        y = F.linear(x.to(cd), w, self.bias.to(cd) if self.bias is not None else None)
        return self._lora(x.to(cd), y).to(x.dtype)


class Nvfp4Linear(nn.Module, _LoraMixin):
    """NVFP4 (e2m1 impacchettato + scale a blocchi fp8 swizzled + scala globale), dequant via comfy_kitchen:
    stesso formato del Nvfp4Linear già validato su FLUX.2 (inference_server/backends/flux2_transformer)."""

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.empty(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer("weight_scale", torch.empty(out_features, in_features // 16, dtype=torch.float8_e4m3fn))
        self.register_buffer("weight_scale_2", torch.ones((), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16)) if bias else None
        self._lora_init()

    def forward(self, x):
        import comfy_kitchen as ck
        w = ck.dequantize_nvfp4(self.weight, self.weight_scale_2, self.weight_scale, output_type=torch.bfloat16)
        w = w[: self.out_features, : self.in_features]
        xb = x.to(torch.bfloat16)
        y = F.linear(xb, w, self.bias.to(torch.bfloat16) if self.bias is not None else None)
        return self._lora(xb, y).to(x.dtype)


QUANT_TYPES = (RowFp8Linear, RowInt8Linear, Nvfp4Linear)


def _set_module(root: nn.Module, name: str, new: nn.Module):
    *path, leaf = name.split(".")
    parent = root
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, leaf, new)


def quantize_from_state_dict(model: nn.Module, sd: dict) -> dict:
    """Sostituisce (su meta-device) ogni nn.Linear col tipo quantizzato indicato dalle chiavi di `sd` e ritorna
    lo state dict riscritto nei nomi dei buffer dei nostri moduli, pronto per load_state_dict(assign=True)."""
    out = dict(sd)
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        has_bias = mod.bias is not None
        inf, outf = mod.in_features, mod.out_features
        if f"{name}._weight_qdata" in sd:
            q = out.pop(f"{name}._weight_qdata")
            s = out.pop(f"{name}._weight_scale")
            zp = out.pop(f"{name}._weight_zero_point", None)
            with torch.device("meta"):
                new = RowInt8Linear(inf, outf, has_bias) if q.dtype == torch.int8 else RowFp8Linear(inf, outf, has_bias, True)
            out[f"{name}.weight"] = q
            out[f"{name}.scale"] = s.reshape(outf, 1).float()
            if isinstance(new, RowInt8Linear):
                out[f"{name}.zero_point"] = (zp if zp is not None else torch.zeros_like(s)).reshape(outf, 1).float()
        elif f"{name}.weight_scale" in sd and sd.get(f"{name}.weight") is not None and sd[f"{name}.weight"].dtype == torch.uint8:
            with torch.device("meta"):
                new = Nvfp4Linear(inf, outf, has_bias)
            out.pop(f"{name}.comfy_quant", None)
            out.pop(f"{name}.input_scale", None)
        elif sd.get(f"{name}.weight") is not None and sd[f"{name}.weight"].dtype == torch.float8_e4m3fn:
            with torch.device("meta"):
                new = RowFp8Linear(inf, outf, has_bias, False)
        else:
            continue
        _set_module(model, name, new)
    return out


def load_into(model: nn.Module, sd: dict, device: str = "cuda", dtype=torch.bfloat16) -> tuple[list, list]:
    """load_state_dict(assign) + parametri non quantizzati in `dtype` su `device`. Ritorna (missing, unexpected)."""
    fixed = {}
    for k, v in sd.items():
        if v.dtype in (torch.float32, torch.float16) and not k.endswith((".scale", ".zero_point", "weight_scale_2")):
            v = v.to(dtype)
        fixed[k] = v.to(device)
    missing, unexpected = model.load_state_dict(fixed, strict=False, assign=True)
    return missing, unexpected


def apply_lora(model: nn.Module, lora_sd: dict, strength: float = 1.0) -> int:
    """Applica una LoRA (formati peft/diffusers «lora_A/lora_B» o kohya «lora_down/lora_up» + alpha) ai moduli
    Linear/quantizzati del modello. Ritorna quanti moduli sono stati agganciati."""
    mods = dict(model.named_modules())
    groups: dict = {}
    for k, v in lora_sd.items():
        for tag, role in ((".lora_A.weight", "A"), (".lora_B.weight", "B"), (".lora_down.weight", "A"),
                          (".lora_up.weight", "B"), (".alpha", "alpha")):
            if k.endswith(tag):
                base = k[: -len(tag)]
                for pre in ("transformer.", "diffusion_model.", "model.diffusion_model.", "base_model.model."):
                    if base.startswith(pre):
                        base = base[len(pre):]
                groups.setdefault(base, {})[role] = v
                break
    n = 0
    for base, g in groups.items():
        m = mods.get(base)
        if m is None or "A" not in g or "B" not in g:
            continue
        a, b = g["A"], g["B"]
        rank = a.shape[0]
        alpha = float(g["alpha"]) if "alpha" in g else float(rank)
        dev = next((t.device for t in m.buffers() if t is not None), None) or \
            next((p.device for p in m.parameters()), torch.device("cuda"))
        a, b = a.to(dev, torch.bfloat16), b.to(dev, torch.bfloat16)
        s = strength * alpha / rank
        if isinstance(m, QUANT_TYPES):
            m.add_lora(a, b, s)
        elif isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight += (s * (b.float() @ a.float())).to(m.weight.dtype)
        else:
            continue
        n += 1
    return n
