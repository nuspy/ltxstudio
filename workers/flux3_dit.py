"""DiT di FLUX.3 Image Klein (5,55 B) in PyTorch puro — carica DIRETTAMENTE i file nativi BFL (bf16 o fp8r per riga).

Perché nativo e non `diffusers.Flux3Transformer2DModel`: il file `variants/fp8r/model.safetensors` ha le chiavi del
codice ufficiale (`content_mode_blocks.image.N.q_proj`, `txt_mode_blocks`, `emb_in.image`…), non quelle di diffusers; e
così il worker gira nel venv già esistente (nessun diffusers ≥ 15/10 da ripinnare, nessun rischio per Qwen-Image 2.1 e
MiniMax H3 che condividono `diffusers_next_venv`). La matematica è quella del riferimento ufficiale
(github.com/black-forest-labs/flux-3-image-klein, src/flux3/{dit,fp8}.py — Apache-2.0, vedi docs/THIRD_PARTY_NOTICES_FLUX3.md):
riscrittura adattata a batch=1/due campioni, config parametrica (per i test su CPU con un modello minuscolo) e
caricamento senza mmap (core del progetto: commit di sistema minimo).

Struttura: per ogni modalità (image, image_cond, text) 5 blocchi PROPRI, poi 28 blocchi single-stream condivisi sulla
sequenza concatenata dei token di ogni campione. Il distillato è un «flow map»: `ref_time_in` (tempo di arrivo) e
nessuna CFG; il Base usa CFG su un pacchetto di due campioni [incondizionato, condizionato].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MODES = ("image", "image_cond", "text")        # ordine dei token dentro un campione


@dataclass(frozen=True)
class Config:
    hidden: int = 3072
    heads: int = 24
    depth: int = 5
    depth_single: int = 28
    mlp_ratio: float = 3.0
    axes_dim: tuple = (32, 32, 32, 32)
    theta: int = 10000
    in_channels: dict = field(default_factory=lambda: {"image": 128, "image_cond": 128, "text": 20480})
    flow_map: bool = True


@dataclass
class Packed:
    """Pacchetto per modalità: mode_indices[m][s] = (inizio, fine) del campione s dentro data[m]."""
    data: dict
    ids: dict
    timesteps: dict
    mode_indices: dict

    @property
    def num_samples(self) -> int:
        return len(self.mode_indices["image"])


# ── primitive ────────────────────────────────────────────────────────────────────────────────────────────
def timestep_embedding(t: Tensor, dim: int = 256, max_period: int = 10000) -> Tensor:
    t = 1000.0 * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(t)


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    out = torch.einsum("...n,d->...nd", pos.double(), 1.0 / (theta ** scale))
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def rope_embed(ids: Tensor, axes_dim, theta: int) -> Tensor:
    """ids (B, L, 4) → matrici di rotazione (B, 1, L, head_dim//2, 2, 2)."""
    with torch.autocast(device_type=ids.device.type, enabled=False):
        emb = torch.cat([rope(ids[..., i], axes_dim[i], theta) for i in range(len(axes_dim))], dim=-3)
    return emb.unsqueeze(1)


def rmsnorm_rope(x: Tensor, scale: Tensor, freqs: Tensor, eps: float = 1e-6) -> Tensor:
    with torch.autocast(device_type=x.device.type, enabled=False):
        b, h, l, d = x.shape
        xf = x.float()
        xn = xf * xf.pow(2).mean(-1, keepdim=True).add(eps).rsqrt() * scale.float()
        xr, freqs = xn.reshape(b, h, l, d // 2, 1, 2), freqs.float()
        rotated = freqs[..., 0] * xr[..., 0] + freqs[..., 1] * xr[..., 1]
        return rotated.reshape(b, h, l, d).to(x.dtype)


def layernorm_modulation(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    return (1.0 + scale) * F.layer_norm(x, (x.shape[-1],), eps=1e-6) + shift


def _sdpa(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Attenzione fusa: su GPU cuDNN con ripiego su flash/efficient/math (dipende dalla build di torch)."""
    if q.is_cuda:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        try:
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                              SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], set_priority=True):
                return F.scaled_dot_product_attention(q, k, v)
        except TypeError:                                   # torch senza set_priority
            pass
    return F.scaled_dot_product_attention(q, k, v)


def packed_attention(q: Tensor, k: Tensor, v: Tensor, bounds) -> Tensor:
    """Attenzione per campione su (1, H, L, D): i campioni non si guardano fra loro."""
    _, heads, length, dim = q.shape
    out = torch.zeros(length, heads * dim, device=q.device, dtype=q.dtype)
    for start, end in bounds:
        if start == end:
            continue
        chunk = _sdpa(q[:, :, start:end], k[:, :, start:end], v[:, :, start:end])
        out[start:end] = chunk[0].transpose(0, 1).reshape(end - start, heads * dim)
    return out


# ── moduli ───────────────────────────────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q, k, freqs):
        return rmsnorm_rope(q, self.query_norm.scale, freqs), rmsnorm_rope(k, self.key_norm.scale, freqs)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden, bias=False)
        self.out_layer = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.out_layer(F.silu(self.in_layer(x)))


class ModeModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, vec):
        return tuple(self.lin(F.silu(vec)).chunk(3, dim=-1))        # shift, scale, gate


class LastLayer(nn.Module):
    def __init__(self, hidden: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(hidden, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=False))

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear(layernorm_modulation(x, scale, shift))


class Block(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: float, axes_dim, theta: int):
        super().__init__()
        self.heads = heads
        mh = int(hidden * mlp_ratio)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp_in = nn.Linear(hidden, mh * 2, bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.mlp_out = nn.Linear(mh, hidden, bias=False)
        self.norm = QKNorm(hidden // heads)

    def forward(self, x, freqs, bounds, mod):
        shift, scale, gate = mod
        xm = layernorm_modulation(x, scale, shift)
        split = lambda t: t.reshape(1, t.shape[0], self.heads, -1).transpose(1, 2)     # L (H D) → 1 H L D
        q, k = self.norm(split(self.q_proj(xm)), split(self.k_proj(xm)), freqs)
        v = split(self.v_proj(xm))
        attn = packed_attention(q, k, v.to(q.dtype), bounds)
        g, up = self.mlp_in(xm).chunk(2, dim=-1)
        return x + gate * (self.attn_out(attn) + self.mlp_out(F.silu(g) * up))


class DiT(nn.Module):
    def __init__(self, cfg: Config | None = None):
        super().__init__()
        c = self.cfg = cfg or Config()
        self.flow_map = c.flow_map
        self.emb_in = nn.ModuleDict({m: nn.Linear(c.in_channels[m], c.hidden, bias=False)
                                     for m in ("image", "image_cond")})
        self.txt_in = nn.Linear(c.in_channels["text"], c.hidden, bias=False)
        self.time_in = MLPEmbedder(256, c.hidden)
        if c.flow_map:
            self.ref_time_in = MLPEmbedder(256, c.hidden)
        mk = lambda: nn.ModuleDict({k: ModeModulation(c.hidden) for k in ("image", "image_cond", "txt")})
        self.early_stream_modulations = mk()
        self.single_stream_modulations = mk()
        blk = lambda: Block(c.hidden, c.heads, c.mlp_ratio, c.axes_dim, c.theta)
        self.content_mode_blocks = nn.ModuleDict({m: nn.ModuleList([blk() for _ in range(c.depth)])
                                                  for m in ("image", "image_cond")})
        self.txt_mode_blocks = nn.ModuleList([blk() for _ in range(c.depth)])
        self.single_blocks = nn.ModuleList([blk() for _ in range(c.depth_single)])
        self.final_layer = nn.ModuleDict({"image": LastLayer(c.hidden, c.in_channels["image"])})

    def forward(self, packed: Packed, timesteps_ref: Tensor | None = None) -> Tensor:
        """Velocità per i token immagine, nell'ordine di packed.data["image"]. timesteps_ref: solo flow map, il tempo
        a cui salta ogni campione."""
        c = self.cfg
        if (timesteps_ref is not None) != self.flow_map:
            raise ValueError("timesteps_ref serve a un DiT flow-map e SOLO a quello")
        data, ids, ts = dict(packed.data), dict(packed.ids), dict(packed.timesteps)
        for m in MODES:        # modalità vuota: un token fantasma perché le proiezioni girino (non viene mai guardato)
            if data[m].shape[0] == 0:
                data[m] = data[m].new_zeros(1, data[m].shape[1])
                ids[m] = ids[m].new_zeros(1, 4)
                ts[m] = ts[m].new_zeros(1)
        mk = lambda m: "txt" if m == "text" else m
        vec = {m: self.time_in(timestep_embedding(ts[m])) for m in MODES}
        if timesteps_ref is not None:
            ref = self.ref_time_in(timestep_embedding(timesteps_ref.to(ts["image"].dtype)))
            for m in MODES:
                parts = [ref[s].expand(b - a, -1) for s, (a, b) in enumerate(packed.mode_indices[m]) if b > a]
                if parts:
                    vec[m] = vec[m] + torch.cat(parts)
        early = {m: self.early_stream_modulations[mk(m)](vec[m]) for m in MODES}
        single = {m: self.single_stream_modulations[mk(m)](vec[m]) for m in MODES}
        x = {m: (self.txt_in if m == "text" else self.emb_in[m])(data[m]) for m in MODES}
        freqs = {m: rope_embed(ids[m][None], c.axes_dim, c.theta) for m in MODES}
        for d in range(c.depth):
            for m in ("image", "image_cond"):
                x[m] = self.content_mode_blocks[m][d](x[m], freqs[m], packed.mode_indices[m], early[m])
            x["text"] = self.txt_mode_blocks[d](x["text"], freqs["text"], packed.mode_indices["text"], early["text"])

        chunks, id_chunks, mods, bounds, placement, offset = [], [], ([], [], []), [], [], 0
        for s in range(packed.num_samples):
            start = offset
            for m in MODES:
                a, b = packed.mode_indices[m][s]
                if b > a:
                    chunks.append(x[m][a:b])
                    id_chunks.append(ids[m][a:b])
                    for i in range(3):
                        mods[i].append(single[m][i][a:b])
                    placement.append((m, a, b, offset))
                    offset += b - a
            bounds.append((start, offset))
        h = torch.cat(chunks)
        fq = rope_embed(torch.cat(id_chunks)[None], c.axes_dim, c.theta)
        mods = tuple(torch.cat(t) for t in mods)
        for block in self.single_blocks:
            h = block(h, fq, bounds, mods)
        image = torch.empty_like(x["image"])
        for m, a, b, start in placement:
            if m == "image":
                image[a:b] = h[start:start + b - a]
        return self.final_layer["image"](image, vec["image"])


# ── fp8 per riga ─────────────────────────────────────────────────────────────────────────────────────────
FP8 = getattr(torch, "float8_e4m3fn", None)
FP8_MAX = 448.0
SCALE_SUFFIX = ".weight_scale"
_GEMM_ALIGN, _ROW_ALIGN = 256, 128


def _ceil(v: int, a: int) -> int:
    return (v + a - 1) // a * a


def quantize_rows(value: Tensor):
    rows = value.reshape(-1, value.shape[-1]).float()
    scale = (rows.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)
    return (rows / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8).reshape_as(value), scale


class FP8Linear(nn.Module):
    """Peso E4M3 con scala fp32 per riga (`weight * weight_scale[:, None]`); attivazione quantizzata per riga a ogni
    chiamata, prodotto con `torch._scaled_mm` (serve compute capability ≥ 8.9: Ada/Blackwell)."""

    def __init__(self, in_features: int, out_features: int, device=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=FP8, device=device))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32, device=device))

    def forward(self, x: Tensor) -> Tensor:
        leading = tuple(x.shape[:-1])
        rows = math.prod(leading)
        flat = x.to(torch.bfloat16).reshape(rows, self.in_features)
        if rows == 0:
            return x.new_zeros(*leading, self.out_features, dtype=torch.bfloat16)
        n, k, m = _ceil(self.out_features, _GEMM_ALIGN), _ceil(self.in_features, _GEMM_ALIGN), _ceil(rows, _ROW_ALIGN)
        weight, scale = self.weight, self.weight_scale
        if (n, k) != (self.out_features, self.in_features):
            weight = F.pad(weight.view(torch.uint8), (0, k - self.in_features, 0, n - self.out_features)).view(FP8)
            scale = F.pad(scale, (0, n - self.out_features), value=1.0)
        if (m, k) != (rows, self.in_features):
            flat = F.pad(flat, (0, k - self.in_features, 0, m - rows))
        q, qs = quantize_rows(flat.contiguous())
        out = torch._scaled_mm(q, weight.T, qs[:, None].contiguous(), scale[None, :].contiguous(),
                               out_dtype=torch.bfloat16, use_fast_accum=True)
        return out[:rows, :self.out_features].reshape(*leading, self.out_features)


def swap_fp8_linears(module: nn.Module, keys) -> int:
    """Ogni nn.Linear con `<nome>.weight_scale` nel file diventa FP8Linear (sul posto). Ritorna quante."""
    names = [n for n, m in module.named_modules() if isinstance(m, nn.Linear) and n + SCALE_SUFFIX in keys]
    for name in names:
        lin = module.get_submodule(name)
        parent, _, child = name.rpartition(".")
        setattr(module.get_submodule(parent) if parent else module, child,
                FP8Linear(lin.in_features, lin.out_features, lin.weight.device))
    return len(names)


def is_flow_map_keys(keys) -> bool:
    """Il distillato ha `ref_time_in.*` (tempo di arrivo): è ciò che lo distingue dal Base (checkpoint.py ufficiale)."""
    return "ref_time_in.in_layer.weight" in keys


def build_meta(cfg: Config | None = None) -> DiT:
    with torch.device("meta"):
        return DiT(cfg)


def load_state(model: DiT, state: dict, device: str = "cuda", dtype=torch.bfloat16) -> DiT:
    """Carica `state` (chiavi native, già sul device) in un DiT su meta: swap fp8, controllo esatto delle chiavi,
    float → bf16 (scale e pesi fp8 restano come sono), `assign=True` (nessuna copia)."""
    n_fp8 = swap_fp8_linears(model, state.keys()) if FP8 is not None else 0
    expected = set(model.state_dict().keys())
    missing, unexpected = sorted(expected - set(state)), sorted(set(state) - expected)
    if missing or unexpected:
        raise KeyError(f"DiT FLUX.3: chiavi mancanti {missing[:5]} / inattese {unexpected[:5]}")
    keep = lambda k, v: k.endswith(SCALE_SUFFIX) or v.dtype == FP8 or not v.is_floating_point()
    state = {k: (v if keep(k, v) else v.to(dtype)) for k, v in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    model._n_fp8 = n_fp8
    return model.to(device).eval()


def make_packed(samples, device) -> Packed:
    """samples[s] = [testo, immagine_cond|None, immagine], ciascuno (token, ids). Come `pack` del codice ufficiale."""
    in_ch = None
    data = {m: [] for m in MODES}
    ids = {m: [] for m in MODES}
    idx = {m: [] for m in MODES}
    counts = dict.fromkeys(MODES, 0)
    for text, cond, image in samples:
        for mode, item in (("text", text), ("image_cond", cond), ("image", image)):
            start = counts[mode]
            if item is not None:
                data[mode].append(item[0])
                ids[mode].append(item[1])
                counts[mode] += item[0].shape[0]
            idx[mode].append((start, counts[mode]))
    widths = {"text": None, "image_cond": None, "image": None}
    for m in MODES:
        if data[m]:
            widths[m] = data[m][0].shape[1]
    # modalità vuote: serve la larghezza dei canali (dal primo campione che ne ha una, altrimenti dal config)
    cat = lambda parts, width, dt: torch.cat(parts) if parts else torch.zeros((0, width), device=device, dtype=dt)
    ch = {"text": widths["text"] or 20480, "image_cond": widths["image_cond"] or 128, "image": widths["image"] or 128}
    d = {m: cat(data[m], ch[m], torch.bfloat16) for m in MODES}
    i = {m: cat(ids[m], 4, torch.float32) for m in MODES}
    return Packed(data=d, ids=i,
                  timesteps={m: torch.zeros(d[m].shape[0], device=device, dtype=torch.bfloat16) for m in MODES},
                  mode_indices=idx)


def schedule(steps: int, alpha: float) -> Tensor:
    """t = α·s/(1+(α−1)·s) con s da 1 a 0 in `steps` passi (steps+1 valori)."""
    t = torch.linspace(1.0, 0.0, steps + 1)
    return alpha * t / (1.0 + (alpha - 1.0) * t)


@torch.inference_mode()
def denoise_flow_map(dit: DiT, packed: Packed, sched: Tensor, on_step=None) -> Tensor:
    """Distillato: Euler sul flow map, un campione, nessuna CFG."""
    for i in range(len(sched) - 1):
        t, dt = sched[i].item(), (sched[i + 1] - sched[i]).item()
        packed.timesteps["image"].fill_(t)
        target = sched[i + 1].reshape(1).to(packed.data["image"].device)
        with torch.autocast(device_type=packed.data["image"].device.type, dtype=torch.bfloat16,
                            enabled=packed.data["image"].is_cuda):
            v = dit(packed, timesteps_ref=target)
        packed.data["image"] = packed.data["image"] + dt * v
        if on_step:
            on_step(i + 1, len(sched) - 1)
    return packed.data["image"]


@torch.inference_mode()
def denoise_cfg(dit: DiT, packed: Packed, sched: Tensor, guidance: float, on_step=None) -> Tensor:
    """Base: CFG su due campioni [incondizionato, condizionato]; ritorna i token del condizionato."""
    for i in range(len(sched) - 1):
        t, dt = sched[i].item(), (sched[i + 1] - sched[i]).item()
        packed.timesteps["image"].fill_(t)
        with torch.autocast(device_type=packed.data["image"].device.type, dtype=torch.bfloat16,
                            enabled=packed.data["image"].is_cuda):
            u, c = dit(packed).chunk(2)
        v = u + guidance * (c - u)
        x = packed.data["image"].chunk(2)[0] + dt * v
        packed.data["image"] = torch.cat((x, x))
        if on_step:
            on_step(i + 1, len(sched) - 1)
    return packed.data["image"].chunk(2)[0]
