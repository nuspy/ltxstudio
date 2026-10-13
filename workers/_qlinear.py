"""Linear quantizzati generici per i worker a single-file (Qwen-Image 2.1, riusabili da altri motori).

Formati riconosciuti dalle chiavi del safetensors, modulo per modulo:
  · torchao Float8Tensor (unsloth «FP8»):  <m>._weight_qdata F8_E4M3 [out,in] + <m>._weight_scale F32 [out,1]
  · torchao Int8Tensor   (unsloth «INT8»): <m>._weight_qdata I8 + _weight_scale F32 [out,1] + _weight_zero_point
  · FP8 «pre-cast» senza scala (TE FP8):   <m>.weight F8_E4M3 (scala implicita 1)
  · NVFP4 («Prism native» / nativo):       <m>.weight U8 [out,in/2] + weight_scale F8 [out,in/16] + weight_scale_2
                                           (+ pre_quant_scale AWQ, applicata all'ingresso)
  · FP8 nativo (core/native_convert):      <m>.weight F8_E4M3 + weight_scale F32 [out,1] (o scalare)
  · embedding int8 nativo:                 <m>.weight I8 + weight_scale F32 [num,1]
I pesi restano compressi in VRAM e si dequantizzano in bf16 un layer alla volta nel forward (matmul bf16:
qualità identica, niente quantizzazione delle attivazioni). Ogni layer accetta LoRA «al volo»
(y = base(x) + s·B(A(x))), perché PEFT non sa avvolgere moduli personalizzati.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def lokr_apply(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """x · kron(w1, w2)ᵀ senza materializzare il kron: con w1 [a1,a2], w2 [b1,b2] e l'ingresso visto come
    X [..., a2, b2], l'uscita è w1 · X · w2ᵀ → [..., a1, b1] (stesso ordine di vec del prodotto di Kronecker)."""
    a1, a2 = w1.shape
    b1, b2 = w2.shape
    xr = x.reshape(*x.shape[:-1], a2, b2)
    return torch.einsum("ik,...kl,jl->...ij", w1, xr, w2).reshape(*x.shape[:-1], a1 * b1)


class _LoraMixin:
    def _lora_init(self):
        self.loras: list = []                        # [(A [r,in], B [out,r], scale)] o ("lokr", w1, w2, scale)

    def add_lora(self, a: torch.Tensor, b: torch.Tensor, scale: float):
        self.loras.append((a, b, float(scale)))

    def add_lokr(self, w1: torch.Tensor, w2: torch.Tensor, scale: float):
        self.loras.append(("lokr", w1, w2, float(scale)))

    def _lora(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        for lo in self.loras:
            if lo[0] == "lokr":
                _, w1, w2, s = lo
                y = y + s * lokr_apply(x, w1.to(x.dtype), w2.to(x.dtype))
            else:
                a, b, s = lo
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
    """Peso int8 per riga: w = (q - zero_point) * scale. `convrot` > 0: il peso è ruotato (W·H a gruppi di
    Hadamard, formato «int8_convrot») → si ruota anche l'ingresso: (x·H)(W·H)ᵀ = x·Wᵀ."""

    def __init__(self, in_features: int, out_features: int, bias: bool, convrot: int = 0):
        super().__init__()
        self.in_features, self.out_features, self.convrot = in_features, out_features, int(convrot)
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features, 1, dtype=torch.float32))
        self.register_buffer("zero_point", torch.zeros(out_features, 1, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16)) if bias else None
        self._lora_init()

    def forward(self, x):
        cd = torch.bfloat16 if x.dtype == torch.float32 else x.dtype
        w = (self.weight.to(cd) - self.zero_point.to(cd)) * self.scale.to(cd)
        xin = x.to(cd)
        if self.convrot:
            from core.native_convert import rotate
            xin = rotate(xin, self.convrot)
        y = F.linear(xin, w, self.bias.to(cd) if self.bias is not None else None)
        return self._lora(x.to(cd), y).to(x.dtype)       # la LoRA legge l'ingresso NON ruotato


class Nvfp4Linear(nn.Module, _LoraMixin):
    """NVFP4 (e2m1 impacchettato + scale a blocchi fp8 swizzled + scala globale), dequant via comfy_kitchen:
    stesso formato del Nvfp4Linear già validato su FLUX.2 (inference_server/backends/flux2_transformer)."""

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.empty(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer("weight_scale", torch.empty(out_features, in_features // 16, dtype=torch.float8_e4m3fn))
        self.register_buffer("weight_scale_2", torch.ones((), dtype=torch.float32))
        self.register_buffer("pre_quant_scale", None)   # AWQ (ModelOpt): x * pre_quant_scale prima del matmul
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16)) if bias else None
        self._lora_init()

    def forward(self, x):
        import comfy_kitchen as ck
        w = ck.dequantize_nvfp4(self.weight, self.weight_scale_2, self.weight_scale, output_type=torch.bfloat16)
        w = w[: self.out_features, : self.in_features]
        xb = x.to(torch.bfloat16)
        xq = xb * self.pre_quant_scale.to(torch.bfloat16) if self.pre_quant_scale is not None else xb
        y = F.linear(xq, w, self.bias.to(torch.bfloat16) if self.bias is not None else None)
        return self._lora(xb, y).to(x.dtype)            # la LoRA vede l'ingresso ORIGINALE (non scalato AWQ)


class Int8RowEmbedding(nn.Module):
    """Embedding int8 con scala per riga («int8_tensorwise» ComfyUI): dequantizza SOLO le righe cercate."""

    def __init__(self, num: int, dim: int, padding_idx=None):
        super().__init__()
        self.num_embeddings, self.embedding_dim, self.padding_idx = num, dim, padding_idx
        self.register_buffer("weight", torch.empty(num, dim, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(num, 1, dtype=torch.float32))

    def forward(self, ids):
        return (self.weight[ids].to(torch.bfloat16) * self.scale[ids].to(torch.bfloat16))


QUANT_TYPES = (RowFp8Linear, RowInt8Linear, Nvfp4Linear)

_ST_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}


def read_safetensors(path: str, device: str = "cuda", rename=None, cast=None, skip=()) -> dict:
    """Legge un .safetensors UN TENSORE ALLA VOLTA (seek + read dall'header) e lo sposta subito su `device`.
    Niente mmap (su Windows la mappatura copy-on-write impegna il commit per l'intero file: crash «invalid
    python storage» quando il pagefile è pieno) e niente lettura dell'intero file in RAM: il picco di memoria
    di sistema è UN tensore. `rename(key) -> key` opzionale; `cast` converte (sul device) i float non-fp8."""
    import json
    import struct
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        base = 8 + n
        on_gpu = str(device).startswith("cuda")
        chunk = 64 << 20
        stage = torch.empty(chunk, dtype=torch.uint8) if on_gpu else None      # UNICO buffer di transito
        for key, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            if key in skip:
                continue
            a, b = info["data_offsets"]
            n = b - a
            dt = _ST_DTYPES[info["dtype"]]
            f.seek(base + a)
            if on_gpu:
                # a blocchi da 64 MB direttamente nel tensore GPU: il commit di sistema resta 64 MB anche per
                # tensori da GB (con f.read + bytearray serviva 2× il tensore: MemoryError a commit quasi pieno)
                raw = torch.empty(n, dtype=torch.uint8, device=device)
                off = 0
                while off < n:
                    k = min(chunk, n - off)
                    got = f.readinto(memoryview(stage.numpy())[:k])
                    if got != k:
                        raise IOError(f"{path}: lettura troncata su {key}")
                    raw[off:off + k].copy_(stage[:k])
                    off += k
                t = raw.view(dt).reshape(info["shape"]) if n else torch.empty(info["shape"], dtype=dt, device=device)
            else:
                t = torch.empty(n, dtype=torch.uint8)                         # una sola copia in RAM
                if n and f.readinto(memoryview(t.numpy())) != n:
                    raise IOError(f"{path}: lettura troncata su {key}")
                t = t.view(dt).reshape(info["shape"]).to(device) if n else torch.empty(info["shape"], dtype=dt)
            if cast is not None and t.is_floating_point() and t.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
                t = t.to(cast)
            out[rename(key) if rename else key] = t
            del t
    return out


def _set_module(root: nn.Module, name: str, new: nn.Module):
    *path, leaf = name.split(".")
    parent = root
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, leaf, new)


class FusedGateUpSwiGLU(nn.Module):
    """SwiGLU con le proiezioni gate/up FUSE in un solo Linear (formato ComfyUI «gate_up»): prima metà delle
    righe = gate_layer, seconda = proj (verificato sui pesi diffusers di Qwen-Image 2.1: coseno 0.995)."""

    def __init__(self, gate_up: nn.Module, out: nn.Module):
        super().__init__()
        self.gate_up, self.out = gate_up, out

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.out(F.silu(g) * u)


def _fuse_gate_up(model: nn.Module, sd: dict) -> None:
    """Se il file ha `<mlp>.gate_up.weight` dove il modello ha `<mlp>.gate_layer` + `<mlp>.proj`, sostituisce il
    MLP con FusedGateUpSwiGLU (gate_up = nn.Linear segnaposto su meta, quantizzato poi come gli altri)."""
    mods = dict(model.named_modules())
    for k in [k for k in sd if k.endswith(".gate_up.weight")]:
        parent = k[: -len(".gate_up.weight")]
        mlp = mods.get(parent)
        if mlp is None or not (hasattr(mlp, "gate_layer") and hasattr(mlp, "proj") and hasattr(mlp, "out")):
            continue
        g, pj = mlp.gate_layer, mlp.proj
        with torch.device("meta"):
            gu = nn.Linear(g.in_features, g.out_features + pj.out_features, bias=False)
        _set_module(model, parent, FusedGateUpSwiGLU(gu, mlp.out))


def quantize_from_state_dict(model: nn.Module, sd: dict) -> dict:
    """Sostituisce (su meta-device) ogni nn.Linear col tipo quantizzato indicato dalle chiavi di `sd` e ritorna
    lo state dict riscritto nei nomi dei buffer dei nostri moduli, pronto per load_state_dict(assign=True)."""
    _fuse_gate_up(model, sd)
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
        elif sd.get(f"{name}.weight") is not None and sd[f"{name}.weight"].dtype == torch.int8                 and f"{name}.weight_scale" in sd:
            # int8 nativo (per riga o per tensore), eventualmente ruotato ConvRot
            g = out.pop(f"{name}.convrot_group", None)
            with torch.device("meta"):
                new = RowInt8Linear(inf, outf, has_bias, int(g) if g is not None else 0)
            ws = out.pop(f"{name}.weight_scale").float()
            out[f"{name}.scale"] = ws.reshape(-1, 1).expand(outf, 1).contiguous()
            out[f"{name}.zero_point"] = torch.zeros(outf, 1, dtype=torch.float32, device=ws.device)
        elif f"{name}.weight_scale" in sd and sd.get(f"{name}.weight") is not None and sd[f"{name}.weight"].dtype == torch.uint8:
            with torch.device("meta"):
                new = Nvfp4Linear(inf, outf, has_bias)
            out.pop(f"{name}.comfy_quant", None)
            out.pop(f"{name}.input_scale", None)
            if f"{name}.pre_quant_scale" in sd:
                new.pre_quant_scale = torch.empty(inf, dtype=torch.bfloat16, device="meta")
        elif (sd.get(f"{name}.weight") is not None and sd[f"{name}.weight"].dtype == torch.float8_e4m3fn
              and f"{name}.weight_scale" in sd and sd[f"{name}.weight_scale"].numel() in (1, outf)):
            # fp8 con scala per riga (formato nativo) o per tensore (scalare, espanso per riga)
            with torch.device("meta"):
                new = RowFp8Linear(inf, outf, has_bias, True)
            ws = out.pop(f"{name}.weight_scale").float()
            out[f"{name}.scale"] = ws.reshape(-1, 1).expand(outf, 1).contiguous()
            out.pop(f"{name}.comfy_quant", None)
            out.pop(f"{name}.input_scale", None)
        elif sd.get(f"{name}.weight") is not None and sd[f"{name}.weight"].dtype == torch.float8_e4m3fn:
            with torch.device("meta"):
                new = RowFp8Linear(inf, outf, has_bias, False)
        else:
            continue
        _set_module(model, name, new)
    for name, mod in list(model.named_modules()):
        w = sd.get(f"{name}.weight")
        if isinstance(mod, nn.Embedding) and w is not None and w.dtype == torch.int8:
            with torch.device("meta"):
                new = Int8RowEmbedding(mod.num_embeddings, mod.embedding_dim, mod.padding_idx)
            out[f"{name}.scale"] = out.pop(f"{name}.weight_scale").float().reshape(-1, 1)
            out.pop(f"{name}.comfy_quant", None)
            _set_module(model, name, new)
    return out


def load_into(model: nn.Module, sd: dict, device: str = "cuda", dtype=torch.bfloat16,
              keep_fp32: tuple = ()) -> tuple[list, list]:
    """load_state_dict(assign) + parametri non quantizzati in `dtype` su `device`. Ritorna (missing, unexpected).
    `keep_fp32`: sottostringhe dei nomi che restano float32 (es. tabelle/teste in fp32 del modello)."""
    fixed = {}
    for k, v in sd.items():
        if any(p in k for p in keep_fp32) and v.is_floating_point() and v.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            v = v.float()
        elif v.dtype in (torch.float32, torch.float16) and not k.endswith((".scale", ".zero_point", "weight_scale_2")):
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
        for tag, role in ((".lokr_w1", "w1"), (".lokr_w2", "w2"), (".lokr_w1_a", "w1a"), (".lokr_w1_b", "w1b"),
                          (".lokr_w2_a", "w2a"), (".lokr_w2_b", "w2b"), (".lora_A.weight", "A"), (".lora_B.weight", "B"), (".lora_A.default.weight", "A"),
                          (".lora_B.default.weight", "B"), (".lora_down.weight", "A"),
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
        if any(r in g for r in ("w1", "w1a", "w2", "w2a")):          # LoKr (LyCORIS)
            n += _lokr_into(m, g, strength, base) if m is not None else 0
            continue
        if "A" not in g or "B" not in g:
            continue
        a, b = g["A"], g["B"]
        if m is None and base.endswith((".gate_layer", ".proj")):   # MLP fuso (gate_up): B nella metà giusta
            fused = mods.get(base.rsplit(".", 1)[0] + ".gate_up")
            if fused is not None:
                half = b.shape[0]
                full = torch.zeros(2 * half, b.shape[1], dtype=b.dtype, device=b.device)
                full[(0 if base.endswith(".gate_layer") else half):][:half] = b
                m, b = fused, full
        rank = a.shape[0]
        s = strength * (float(g["alpha"]) if "alpha" in g else float(rank)) / rank
        if m is None and base.endswith(".gate_up"):           # LoRA sul MLP fuso, modello coi due rami separati
            stem = base.rsplit(".", 1)[0]
            g_m, p_m = mods.get(stem + ".gate_layer"), mods.get(stem + ".proj")
            if g_m is not None and p_m is not None:
                half = b.shape[0] // 2
                n += _lora_into(g_m, a, b[:half], s) + _lora_into(p_m, a, b[half:], s)
            continue
        if m is not None:
            n += _lora_into(m, a, b, s)
    return n


def _lokr_into(m: nn.Module, g: dict, strength: float, name: str = "") -> int:
    """LoKr: ΔW = s · kron(w1, w2), con w1/w2 pieni o a basso rango (w = a @ b). Scala come LyCORIS: entrambi
    pieni → 1 (alpha ignorato); altrimenti alpha / rango del fattore decomposto."""
    w1 = g["w1"] if "w1" in g else g["w1a"] @ g["w1b"]
    w2 = g["w2"] if "w2" in g else g["w2a"] @ g["w2b"]
    if "w1" in g and "w2" in g:
        s = strength
    else:
        rank = (g["w2b"] if "w2b" in g else g["w1b"]).shape[0]
        s = strength * (float(g["alpha"]) if "alpha" in g else float(rank)) / rank
    out_f, in_f = w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1]
    if (getattr(m, "out_features", None), getattr(m, "in_features", None)) != (out_f, in_f):
        print(f"[lokr] {name}: forma {out_f}x{in_f} ≠ modulo {getattr(m, 'out_features', '?')}x"
              f"{getattr(m, 'in_features', '?')} → saltato", flush=True)
        return 0
    dev = next((t.device for t in m.buffers() if t is not None), None) or         next((p.device for p in m.parameters()), torch.device("cuda"))
    w1, w2 = w1.to(dev, torch.bfloat16), w2.to(dev, torch.bfloat16)
    if isinstance(m, QUANT_TYPES):
        m.add_lokr(w1, w2, s)
    elif isinstance(m, nn.Linear):
        with torch.no_grad():
            m.weight += (s * torch.kron(w1.float(), w2.float())).to(m.weight.dtype)
    else:
        return 0
    return 1


def _lora_into(m: nn.Module, a: torch.Tensor, b: torch.Tensor, s: float) -> int:
    """Aggancia una coppia A/B con scala `s` a un modulo: 1 se agganciata, 0 se il tipo non è supportato."""
    dev = next((t.device for t in m.buffers() if t is not None), None) or \
        next((p.device for p in m.parameters()), torch.device("cuda"))
    a, b = a.to(dev, torch.bfloat16), b.to(dev, torch.bfloat16)
    if isinstance(m, QUANT_TYPES):
        m.add_lora(a, b, s)
    elif isinstance(m, nn.Linear):
        with torch.no_grad():
            m.weight += (s * (b.float() @ a.float())).to(m.weight.dtype)
    else:
        return 0
    return 1
