"""Wan2.1 I2V 14B + adattatore MultiTalk (UN parlante) — riscrittura compatta, solo PyTorch.

Derivata da `wan/modules/multitalk_model.py` (MeiGen-AI/MultiTalk, Apache-2.0, Alibaba Wan Team) ma:
  · niente xfuser / xformers / flash-attn / diffusers.ModelMixin: attenzione = SageAttention se disponibile e la
    scheda la regge (sm80+), altrimenti PyTorch SDPA (ripiego automatico, funziona su ogni scheda);
  · pesi in fp8 «pre-cast» (Kijai `Wan2_1-I2V-14B-480P_fp8_e4m3fn`) con dequantizzazione bf16 un layer alla
    volta (`_qlinear.RowFp8Linear`): su sm<89 non servono kernel fp8, il matmul e' sempre bf16;
  · LoRA «al volo» (lightx2v distill + LoRA Wan dell'utente) senza toccare i pesi fp8;
  · block-swap dinamico: i blocchi oltre `resident` restano in RAM e scorrono sulla GPU (VRAM tarata dal chiamante);
  · RoPE reale (cos/sin) invece di complessi float64: stesso risultato, ~3x meno memoria.
Solo un parlante: la parte multi-persona dell'originale (ref_target_masks, rope 1D per classe) non e' portata.
Chiavi dei pesi = quelle originali Wan (`blocks.N.self_attn.q.weight`, `audio_proj.*`, `blocks.N.audio_cross_attn.*`).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:                                     # SageAttention (opzionale): solo schede sm80+, kernel gia' nel venv
    from sageattention import sageattn as _sageattn
except Exception:                        # noqa: BLE001
    _sageattn = None

_ATTN = {"mode": "auto", "sage_ok": None}


def set_attention_mode(mode: str) -> str:
    """'auto' (sage se possibile) | 'sdpa' (sempre SDPA) | 'sage'. Ritorna il backend effettivo."""
    _ATTN["mode"] = mode if mode in ("auto", "sdpa", "sage") else "auto"
    return effective_attention()


def effective_attention() -> str:
    if _ATTN["mode"] == "sdpa" or _sageattn is None or not torch.cuda.is_available():
        return "sdpa"
    if _ATTN["sage_ok"] is None:
        try:
            _ATTN["sage_ok"] = torch.cuda.get_device_capability()[0] >= 8
        except Exception:                # noqa: BLE001
            _ATTN["sage_ok"] = False
    return "sage" if _ATTN["sage_ok"] else "sdpa"


def attention(q, k, v):
    """q,k,v: [B, S, H, D] (layout NHD) -> [B, S, H, D]. Sage con ripiego SDPA se il kernel fallisce."""
    if effective_attention() == "sage" and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        try:
            return _sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
        except Exception:                # noqa: BLE001 — forma/dtype non gestiti: SDPA per sempre
            _ATTN["sage_ok"] = False
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return o.transpose(1, 2)


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(torch.arange(max_seq_len),
                        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    return freqs                          # angoli [max_seq_len, dim/2] (float64): cos/sin si calcolano a runtime


def build_rope(head_dim: int, f: int, h: int, w: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin float32 [f*h*w, 1, head_dim/2] con la stessa partizione degli assi dell'originale (t, h, w)."""
    d = head_dim
    ang = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)),
                     rope_params(1024, 2 * (d // 6))], dim=1)               # [1024, d/2]
    c = d // 2
    a_t, a_h, a_w = ang.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    ang3 = torch.cat([a_t[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                      a_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                      a_w[:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1)
    ang3 = ang3.to(device)
    return torch.cos(ang3).float(), torch.sin(ang3).float()


def apply_rope(x, cos, sin):
    """x [B,S,N,D]: coppie adiacenti (pari, dispari) ruotate — equivalente al prodotto complesso dell'originale."""
    xf = x.float()
    x1, x2 = xf[..., 0::2], xf[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack([o1, o2], dim=-1).flatten(-2).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class LayerNorm32(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return F.layer_norm(x.float(), self.normalized_shape,
                            None if self.weight is None else self.weight.float(),
                            None if self.bias is None else self.bias.float(), self.eps).to(x.dtype)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.norm_q, self.norm_k = RMSNorm(dim, eps), RMSNorm(dim, eps)

    def forward(self, x, rope):
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        return self.o(attention(q, k, v).flatten(2))


class I2VCrossAttention(nn.Module):
    """Testo (512 token, padding incluso come nell'originale) + immagine CLIP (257 token) in due attenzioni."""

    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.norm_q, self.norm_k = RMSNorm(dim, eps), RMSNorm(dim, eps)
        self.k_img, self.v_img = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.norm_k_img = RMSNorm(dim, eps)

    def forward(self, x, context):
        ci, ct = context[:, :257], context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(ct)).view(b, -1, n, d)
        v = self.v(ct).view(b, -1, n, d)
        ki = self.norm_k_img(self.k_img(ci)).view(b, -1, n, d)
        vi = self.v_img(ci).view(b, -1, n, d)
        return self.o((attention(q, ki, vi) + attention(q, k, v)).flatten(2))


class AudioCrossAttention(nn.Module):
    """Attenzione audio per FOTOGRAMMA latente (un solo parlante): ogni frame latente guarda solo i suoi 32 token
    audio. Chiavi: q_linear / kv_linear / proj (qkv_bias=True, niente qk-norm) come l'adattatore MultiTalk."""

    def __init__(self, dim, enc_dim, num_heads):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.q_linear = nn.Linear(dim, dim, bias=True)
        self.kv_linear = nn.Linear(enc_dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, enc, n_t):
        """x [1, N_t*S, C]; enc [N_t, M, enc_dim] -> [1, N_t*S, C]."""
        _, ns, c = x.shape
        s = ns // n_t
        n, d = self.num_heads, self.head_dim
        q = self.q_linear(x.view(n_t, s, c)).view(n_t, s, n, d)
        kv = self.kv_linear(enc).view(n_t, enc.shape[1], 2, n, d)
        o = attention(q, kv[:, :, 0], kv[:, :, 1])                          # [N_t, S, H, D]
        return self.proj(o.reshape(n_t, s, c)).view(1, ns, c)


class AttentionBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, eps=1e-6, audio_dim=768):
        super().__init__()
        self.dim = dim
        self.norm1 = LayerNorm32(dim, eps)
        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.norm3 = LayerNorm32(dim, eps, elementwise_affine=True)
        self.cross_attn = I2VCrossAttention(dim, num_heads, eps)
        self.norm2 = LayerNorm32(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
        self.audio_cross_attn = AudioCrossAttention(dim, audio_dim, num_heads)
        self.norm_x = LayerNorm32(dim, eps, elementwise_affine=True)

    def forward(self, x, e, rope, context, audio, n_t, audio_scale):
        dtype = x.dtype
        e = (self.modulation.float() + e).chunk(6, dim=1)
        y = self.self_attn((self.norm1(x).float() * (1 + e[1]) + e[0]).type_as(x), rope)
        x = (x.float() + y.float() * e[2]).to(dtype)
        x = x + self.cross_attn(self.norm3(x), context)
        if audio is not None and audio_scale:
            x = x + self.audio_cross_attn(self.norm_x(x), audio, n_t) * audio_scale
        y = self.ffn((self.norm2(x).float() * (1 + e[4]) + e[3]).to(dtype))
        return (x.float() + y.float() * e[5]).to(dtype)


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.norm = LayerNorm32(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        e = (self.modulation.float() + e.unsqueeze(1)).chunk(2, dim=1)
        return self.head((self.norm(x).float() * (1 + e[1]) + e[0]).to(x.dtype))


class MLPProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim), nn.GELU(),
                                  nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x):
        return self.proj(x)


class AudioProjModel(nn.Module):
    """wav2vec (5 finestre x 12 strati x 768) -> 32 token audio da 768 per frame latente."""

    def __init__(self, seq_len=5, seq_len_vf=8, blocks=12, channels=768, intermediate_dim=512, output_dim=768,
                 context_tokens=32, norm_output_audio=True):
        super().__init__()
        self.context_tokens, self.output_dim = context_tokens, output_dim
        self.proj1 = nn.Linear(seq_len * blocks * channels, intermediate_dim)
        self.proj1_vf = nn.Linear(seq_len_vf * blocks * channels, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, first, latter):
        """first [B,1,w,b,c]; latter [B,N_t-1,w',b,c] -> [B, N_t, 32, 768]."""
        b = first.shape[0]
        f1 = first.reshape(first.shape[0] * first.shape[1], -1)
        f2 = latter.reshape(latter.shape[0] * latter.shape[1], -1)
        f1 = torch.relu(self.proj1(f1)).view(b, -1, self.proj1.out_features)
        f2 = torch.relu(self.proj1_vf(f2)).view(b, -1, self.proj1.out_features)
        h = torch.cat([f1, f2], dim=1)
        nt = h.shape[1]
        h = torch.relu(self.proj2(h.reshape(b * nt, -1)))
        out = self.proj3(h).reshape(b * nt, self.context_tokens, self.output_dim)
        return self.norm(out).view(b, nt, self.context_tokens, self.output_dim)


class WanMTModel(nn.Module):
    """Wan2.1 I2V 14B (dim 5120, 40 blocchi) con adattatore audio MultiTalk."""

    def __init__(self, in_dim=36, dim=5120, ffn_dim=13824, freq_dim=256, text_dim=4096, out_dim=16,
                 num_heads=40, num_layers=40, patch_size=(1, 2, 2), text_len=512, eps=1e-6,
                 audio_window=5, vae_scale=4, audio_dim=768, context_tokens=32):
        super().__init__()
        self.dim, self.freq_dim, self.out_dim, self.patch_size = dim, freq_dim, out_dim, patch_size
        self.num_heads, self.text_len, self.audio_window, self.vae_scale = num_heads, text_len, audio_window, vae_scale
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([AttentionBlock(dim, ffn_dim, num_heads, eps, audio_dim) for _ in range(num_layers)])
        self.head = Head(dim, out_dim, patch_size, eps)
        self.img_emb = MLPProj(1280, dim)
        self.audio_proj = AudioProjModel(seq_len=audio_window, seq_len_vf=audio_window + vae_scale - 1,
                                         output_dim=audio_dim, context_tokens=context_tokens)
        # block-swap: oltre `resident` i blocchi vivono in RAM e passano alla GPU uno alla volta
        self.swap = None

    # ---- audio: finestre per frame latente (stessa logica dell'originale) ----
    def audio_tokens(self, audio, dtype):
        """audio [1, F, window, blocks, C] (F = frame video) -> [N_t, 32, 768] (un parlante)."""
        a = audio.to(dtype)
        first = a[:, :1]
        latter = a[:, 1:]
        latter = latter.reshape(1, -1, self.vae_scale, *latter.shape[2:])       # [1, n_t, n, w, s, c]
        mid = self.audio_window // 2
        lf = latter[:, :, :1, :mid + 1].flatten(2, 3)
        ll = latter[:, :, -1:, mid:].flatten(2, 3)
        lm = latter[:, :, 1:-1, mid:mid + 1].flatten(2, 3)
        latter_s = torch.cat([lf, lm, ll], dim=2)
        return self.audio_proj(first, latter_s)[0]

    def forward(self, x, t, context, clip_fea, y, audio=None, audio_scale=1.0):
        """x [16,T,H,W] latente rumoroso; y [20,T,H,W] (maschera 4 + latente immagine 16); t [1];
        context [L,4096] testo; clip_fea [1,257,1280]; audio [1,F,5,12,768] o None. -> [16,T,H,W] float32."""
        dev = next(self.patch_embedding.parameters()).device
        dt = self.patch_embedding.weight.dtype                                   # bf16 col modello reale, fp32 nei test CPU
        xin = torch.cat([x, y], dim=0).unsqueeze(0).to(dev, dt)
        xp = self.patch_embedding(xin)
        gf, gh, gw = xp.shape[2:]
        x = xp.flatten(2).transpose(1, 2)                                       # [1, S, C]
        rope = build_rope(self.dim // self.num_heads, gf, gh, gw, dev)

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.to(dev)).float().to(self.time_embedding[0].weight.dtype))
        e0 = self.time_projection(e).float().unflatten(1, (6, self.dim))
        e = e.float()

        ctx = context.to(dev, dt)
        ctx = torch.cat([ctx, ctx.new_zeros(self.text_len - ctx.shape[0], ctx.shape[1])])[None]
        ctx = self.text_embedding(ctx)
        ctx = torch.cat([self.img_emb(clip_fea.to(dev, dt)), ctx], dim=1).to(dt)

        a_tok = None
        if audio is not None and audio_scale:
            a_tok = self.audio_tokens(audio.to(dev), dt)                        # [N_t, 32, 768]
            assert a_tok.shape[0] == gf, f"frame audio {a_tok.shape[0]} != frame latenti {gf}"

        x = x.to(dt)
        for i, blk in enumerate(self.blocks):
            if self.swap is not None:
                self.swap.before(i)
            x = blk(x, e0, rope, ctx, a_tok, gf, audio_scale)
            if self.swap is not None:
                self.swap.after(i)
        x = self.head(x, e)
        # unpatchify
        c, (pf, ph, pw) = self.out_dim, self.patch_size
        u = x[0].view(gf, gh, gw, pf, ph, pw, c)
        u = torch.einsum("fhwpqrc->cfphqwr", u).reshape(c, gf * pf, gh * ph, gw * pw)
        return u.float()


class BlockSwap:
    """Block-swap con master in RAM: i blocchi con indice >= `resident` stanno su CPU e vengono copiati sulla GPU
    solo mentre servono (poi la copia GPU viene scartata). Niente pinned memory (col pagefile piccolo la pinned
    fa segfault, vedi memoria progetto): copie sincrone, semplici e robuste. `prefetch=True` carica il blocco
    successivo su uno stream CUDA separato mentre il corrente calcola."""

    def __init__(self, blocks: nn.ModuleList, resident: int, device, prefetch: bool = True):
        self.blocks, self.resident, self.device = blocks, resident, torch.device(device)
        self.masters: dict[int, list] = {}
        self.prefetch = prefetch and self.device.type == "cuda"
        self._stream = torch.cuda.Stream(self.device) if self.prefetch else None
        self._pending: dict[int, list] = {}
        for i, blk in enumerate(blocks):
            if i >= resident:
                self.masters[i] = self._slots(blk)

    @staticmethod
    def _slots(blk):
        """[(contenitore, chiave, tensore)] di parametri, buffer e LoRA del blocco (la lista LoRA e' di tuple)."""
        out = []
        for m in blk.modules():
            for k, p in m._parameters.items():
                if p is not None:
                    out.append((m._parameters, k, p))
            for k, b in m._buffers.items():
                if b is not None:
                    out.append((m._buffers, k, b))
            lo = getattr(m, "loras", None)
            if lo:
                for j, ent in enumerate(lo):
                    out.append((lo, j, ent))
        return out

    @staticmethod
    def _to(ent, dev, non_blocking=False):
        if isinstance(ent, tuple):
            return tuple(t.to(dev, non_blocking=non_blocking) if torch.is_tensor(t) else t for t in ent)
        if isinstance(ent, nn.Parameter):
            return nn.Parameter(ent.data.to(dev, non_blocking=non_blocking), requires_grad=False)
        return ent.to(dev, non_blocking=non_blocking)

    def _load(self, i):
        for cont, k, ent in self.masters[i]:
            cont[k] = self._to(ent, self.device)

    def _load_async(self, i):
        with torch.cuda.stream(self._stream):
            self._pending[i] = [(cont, k, self._to(ent, self.device, True)) for cont, k, ent in self.masters[i]]

    def before(self, i):
        if i < self.resident:
            return
        if i in self._pending:
            torch.cuda.current_stream(self.device).wait_stream(self._stream)
            cur = torch.cuda.current_stream(self.device)
            for cont, k, t in self._pending.pop(i):
                for tt in (t if isinstance(t, tuple) else (t,)):
                    if torch.is_tensor(tt):
                        tt.record_stream(cur)               # la memoria non torna al pool finche' il calcolo l'usa
                cont[k] = t
        else:
            self._load(i)
        if self.prefetch and i + 1 < len(self.blocks):
            self._load_async(i + 1)

    def after(self, i):
        if i < self.resident:
            return
        for cont, k, ent in self.masters[i]:
            cont[k] = ent                                   # torna al master CPU: la copia GPU viene liberata

    def offload_all(self):
        for i in self.masters:
            self.after(i)
