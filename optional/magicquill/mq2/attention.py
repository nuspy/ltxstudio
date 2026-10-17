"""Blocchi FLUX con condizione di controllo MagicQuill V2, per il transformer diffusers 0.38 GIÀ caricato.

Derivato da MagicQuillV2 `src/layers_cache.py` e `src/transformer_flux.py` (CC BY-NC 4.0, (c) Zichen Liu et al.; da
EasyControl). Vedi `NOTICE.md` (cartella del componente) per attribuzione e avvertenza di licenza (NON
commerciale). Riscritto (non copiato) per la struttura di diffusers 0.38: layout (B, S, H, D), `FluxAttention`, blocco
singolo con txt separato, `FluxTransformer2DModel.forward` STANDARD (non sostituito).

COME SI AGGANCIA
    La sequenza "immagine" passata al transformer è [rumore | immagine Kontext | condizione]: la CONDIZIONE viaggia in
    coda (`MQState.n_cond` token). Il forward standard del transformer la embedda (x_embedder) e la trascina fra i
    blocchi; qui si sostituisce SOLO `block.forward` (per istanza, ripristinato a fine job) con una versione che:
      * SCRITTURA (primo passaggio, blocco senza K/V in banca): la coda è la condizione. Si modula con `cond_temb`
        (timestep 0), q/k/v della condizione = pesi base + LoRA di controllo (solo sui suoi token), la condizione
        attende SOLO sé stessa, e il suo K/V (dopo norm_k + RoPE) va nella banca. Le righe immagine/testo attendono
        anche le chiavi della condizione, con bias log(gamma) sulle colonne-condizione (come la maschera upstream).
      * LETTURA (passaggi successivi): la condizione viene SCARTATA (le sue righe, a K/V cachati, non influenzano
        nessuna riga non-condizione: lo si verifica nei test) e le righe immagine/testo leggono K/V dalla banca.
    Il passaggio di warm-up upstream (timestep[0], output scartato) è ASSORBITO nel primo step reale: i token
    non-condizione producono lo stesso risultato in scrittura e in lettura (test_mq2_attention).

FEDELTÀ ALL'ORIGINALE (anche dove sembra una svista upstream; da NON "correggere" senza decidere):
    * righe con bias gamma: blocchi doppi [T, N+K) dove N+K = n. token immagine (NON T+N+K: l'ultima fetta di T righe
      Kontext non riceve il bias); blocchi singoli [T, T+N+K). `DOUBLE_ROWS_UPSTREAM=False` usa [T, T+N+K) in entrambi.
    * cond_temb = time_text_embed(0, guidance*1000, pooled): stessa guidance del passaggio normale.
"""
from __future__ import annotations

import contextlib
import math
from typing import Any

import torch
import torch.nn.functional as F

# True = range righe bias dei blocchi DOPPI identico a upstream (vedi sopra); False = [T, T+N+K) come nei singoli.
DOUBLE_ROWS_UPSTREAM = True


class MQUnsupported(RuntimeError):
    """Il transformer caricato non è un FluxTransformer2DModel diffusers standard (es. nunchaku/svdq)."""


class MQState:
    """Stato di UN job: condizioni, banca K/V per blocco, cond_temb. Vive solo durante `patch_transformer`."""

    def __init__(self, controls: list):
        self.controls = list(controls)
        self.cond_sizes = [c.n_tokens for c in self.controls]
        self.offsets = [sum(self.cond_sizes[:i]) for i in range(len(self.controls))]
        self.n_cond = sum(self.cond_sizes)
        self.n_base: int | None = None          # token rumore+Kontext (lo imposta la pipeline in prepare_latents)
        self.bank: dict[tuple, tuple] = {}      # (kind, idx) -> (K_cond [B,C,H,D] post norm/RoPE, V_cond)
        self.tte_module = None
        self.tte_args: tuple | None = None
        self.cond_temb = None
        self._busy = False
        self.writes = 0
        self.reads = 0

    # ---- bias gamma -------------------------------------------------------------------------------------------
    @property
    def has_bias(self) -> bool:
        return any(abs(float(c.gamma) - 1.0) > 1e-12 for c in self.controls)

    def bias_row(self, n_keys: int, dtype, device):
        """Bias additivo per le colonne-condizione (le ULTIME `n_cond` chiavi): log(gamma_i), 0 altrove."""
        b = torch.zeros(1, 1, 1, n_keys, dtype=torch.float32, device=device)
        base = n_keys - self.n_cond
        for c, off, n in zip(self.controls, self.offsets, self.cond_sizes):
            g = float(c.gamma)
            b[..., base + off: base + off + n] = math.log(g) if g > 0 else float("-inf")
        return b.to(dtype)

    # ---- cond_temb --------------------------------------------------------------------------------------------
    def get_cond_temb(self):
        if self.cond_temb is None:
            if self.tte_args is None:
                raise RuntimeError("MQState: time_text_embed non ancora chiamato (hook di cattura assente?)")
            args, kwargs = self.tte_args
            new_args = (torch.zeros_like(args[0]),) + tuple(args[1:])
            self._busy = True
            try:
                self.cond_temb = self.tte_module(*new_args, **kwargs)
            finally:
                self._busy = False
        return self.cond_temb

    def split_tail(self, hidden_states, write: bool):
        """(token non-condizione, condizione|None). In scrittura la coda È la condizione; in lettura la si scarta."""
        S = hidden_states.shape[1]
        C = self.n_cond
        if write:
            if S < C:
                raise RuntimeError(f"MQ: sequenza ({S}) più corta dei token condizione ({C})")
            return hidden_states[:, :S - C], hidden_states[:, S - C:]
        if self.n_base is not None and S == self.n_base + C:
            return hidden_states[:, :self.n_base], None
        return hidden_states, None

    def clear(self):
        self.bank.clear()
        self.cond_temb = None
        self.tte_args = None


# ---- primitive --------------------------------------------------------------------------------------------------
def _sdpa(q, k, v, mask=None):
    """SDPA su layout (B, S, H, D)."""
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    return o.transpose(1, 2)


def _attend(st: MQState, q, k, v, row_lo: int, row_hi: int):
    """Attenzione delle righe non-condizione su tutte le chiavi; bias log(gamma) sulle colonne-condizione per le
    righe [row_lo, row_hi). Le righe fuori range usano SDPA senza maschera (path veloce)."""
    Lq = q.shape[1]
    row_hi = min(row_hi, Lq)
    if not st.has_bias or st.n_cond == 0 or row_hi <= row_lo:
        return _sdpa(q, k, v)
    bias = st.bias_row(k.shape[1], q.dtype, q.device)
    parts = []
    if row_lo > 0:
        parts.append(_sdpa(q[:, :row_lo], k, v))
    parts.append(_sdpa(q[:, row_lo:row_hi], k, v, bias))
    if row_hi < Lq:
        parts.append(_sdpa(q[:, row_hi:], k, v))
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


def _rope(x, emb, a: int, b: int):
    from diffusers.models.embeddings import apply_rotary_emb
    return apply_rotary_emb(x, (emb[0][a:b], emb[1][a:b]), sequence_dim=1)


def _qkv(attn, x):
    H = attn.heads
    return (attn.to_q(x).unflatten(-1, (H, -1)), attn.to_k(x).unflatten(-1, (H, -1)),
            attn.to_v(x).unflatten(-1, (H, -1)))


def _cond_stream_attention(st: MQState, attn, kind: str, idx: int, norm_c, emb, base_len: int):
    """Q/K/V della condizione (base + LoRA sui SUOI token), norm + RoPE, attenzione cond->cond. Ritorna
    (output [B,C,H*D], K_cond, V_cond)."""
    H = attn.heads
    outs, ks, vs = [], [], []
    for ctl, off, n in zip(st.controls, st.offsets, st.cond_sizes):
        xi = norm_c[:, off:off + n]
        w = float(ctl.lora_weight)
        q = attn.to_q(xi) + w * ctl.lora.delta(kind, idx, "q", xi)
        k = attn.to_k(xi) + w * ctl.lora.delta(kind, idx, "k", xi)
        v = attn.to_v(xi) + w * ctl.lora.delta(kind, idx, "v", xi)
        q = attn.norm_q(q.unflatten(-1, (H, -1)))
        k = attn.norm_k(k.unflatten(-1, (H, -1)))
        v = v.unflatten(-1, (H, -1))
        q = _rope(q, emb, base_len + off, base_len + off + n)
        k = _rope(k, emb, base_len + off, base_len + off + n)
        o = _sdpa(q, k, v).flatten(2, 3).to(q.dtype)          # la condizione attende SOLO sé stessa
        outs.append(o)
        ks.append(k)
        vs.append(v)
    return torch.cat(outs, 1), torch.cat(ks, 1), torch.cat(vs, 1)


def _clip16(x):
    return x.clip(-65504, 65504) if x.dtype == torch.float16 else x


# ---- blocchi ----------------------------------------------------------------------------------------------------
def make_double_forward(block, idx: int, st: MQState):
    """Sostituto di `FluxTransformerBlock.forward` (diffusers 0.38). Con st.n_cond == 0 è identico al forward stock."""
    attn = block.attn

    def forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        key = ("d", idx)
        T = encoder_hidden_states.shape[1]
        write = st.n_cond > 0 and key not in st.bank
        if st.n_cond > 0:
            hs, cond = st.split_tail(hidden_states, write)
        else:
            hs, cond = hidden_states, None
        Lb = hs.shape[1]
        emb = image_rotary_emb

        norm_h, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(hs, emb=temb)
        norm_e, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = block.norm1_context(encoder_hidden_states, emb=temb)

        q, k, v = _qkv(attn, norm_h)
        q, k = attn.norm_q(q), attn.norm_k(k)
        H = attn.heads
        eq = attn.norm_added_q(attn.add_q_proj(norm_e).unflatten(-1, (H, -1)))
        ek = attn.norm_added_k(attn.add_k_proj(norm_e).unflatten(-1, (H, -1)))
        ev = attn.add_v_proj(norm_e).unflatten(-1, (H, -1))
        Q = _rope(torch.cat([eq, q], 1), emb, 0, T + Lb)
        K = _rope(torch.cat([ek, k], 1), emb, 0, T + Lb)
        V = torch.cat([ev, v], 1)

        cond_out = None
        if write:
            cond_temb = st.get_cond_temb()
            norm_c, cg_msa, cg_shift, cg_scale, cg_gate_mlp = block.norm1(cond, emb=cond_temb)
            oc, kc, vc = _cond_stream_attention(st, attn, "d", idx, norm_c, emb, T + Lb)
            st.bank[key] = (kc, vc)
            st.writes += 1
            oc = attn.to_out[0](oc)
            parts = []
            for ctl, off, n in zip(st.controls, st.offsets, st.cond_sizes):
                sl = oc[:, off:off + n]
                parts.append(sl + float(ctl.lora_weight) * ctl.lora.delta("d", idx, "proj", sl))
            cond_out = attn.to_out[1](torch.cat(parts, 1))
        elif st.n_cond > 0:
            st.reads += 1
        if st.n_cond > 0:
            kc, vc = st.bank[key]
            K = torch.cat([K, kc], 1)
            V = torch.cat([V, vc], 1)

        row_hi = Lb if DOUBLE_ROWS_UPSTREAM else T + Lb
        o = _attend(st, Q, K, V, T, row_hi).flatten(2, 3).to(Q.dtype)
        enc_o, img_o = o[:, :T], o[:, T:]
        img_o = attn.to_out[1](attn.to_out[0](img_o))
        enc_o = attn.to_add_out(enc_o)

        hs = hs + gate_msa.unsqueeze(1) * img_o
        n2 = block.norm2(hs)
        n2 = n2 * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hs = hs + gate_mlp.unsqueeze(1) * block.ff(n2)

        enc = encoder_hidden_states + c_gate_msa.unsqueeze(1) * enc_o
        ne = block.norm2_context(enc)
        ne = ne * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        enc = _clip16(enc + c_gate_mlp.unsqueeze(1) * block.ff_context(ne))

        if write:
            cond = cond + cg_msa.unsqueeze(1) * cond_out
            nc = block.norm2(cond)
            nc = nc * (1 + cg_scale[:, None]) + cg_shift[:, None]
            cond = cond + cg_gate_mlp.unsqueeze(1) * block.ff(nc)
            hs = torch.cat([hs, cond], 1)
        return enc, hs

    return forward


def make_single_forward(block, idx: int, st: MQState):
    """Sostituto di `FluxSingleTransformerBlock.forward` (diffusers 0.38)."""
    attn = block.attn

    def forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        key = ("s", idx)
        T = encoder_hidden_states.shape[1]
        write = st.n_cond > 0 and key not in st.bank
        if st.n_cond > 0:
            hs, cond = st.split_tail(hidden_states, write)
        else:
            hs, cond = hidden_states, None
        Lb = hs.shape[1]
        emb = image_rotary_emb

        joint = torch.cat([encoder_hidden_states, hs], 1)
        residual = joint
        norm_j, gate = block.norm(joint, emb=temb)
        mlp = block.act_mlp(block.proj_mlp(norm_j))
        q, k, v = _qkv(attn, norm_j)
        q, k = attn.norm_q(q), attn.norm_k(k)
        Q = _rope(q, emb, 0, T + Lb)
        K = _rope(k, emb, 0, T + Lb)
        V = v

        if write:
            cond_temb = st.get_cond_temb()
            norm_c, cgate = block.norm(cond, emb=cond_temb)
            mlp_c = block.act_mlp(block.proj_mlp(norm_c))
            oc, kc, vc = _cond_stream_attention(st, attn, "s", idx, norm_c, emb, T + Lb)
            st.bank[key] = (kc, vc)
            st.writes += 1
        elif st.n_cond > 0:
            st.reads += 1
        if st.n_cond > 0:
            kc, vc = st.bank[key]
            K = torch.cat([K, kc], 1)
            V = torch.cat([V, vc], 1)

        o = _attend(st, Q, K, V, T, T + Lb).flatten(2, 3).to(Q.dtype)
        out = torch.cat([o, mlp], 2)
        out = residual + gate.unsqueeze(1) * block.proj_out(out)
        out = _clip16(out)
        enc, hs_out = out[:, :T], out[:, T:]
        if write:
            c_out = cond + cgate.unsqueeze(1) * block.proj_out(torch.cat([oc, mlp_c], 2))
            hs_out = torch.cat([hs_out, _clip16(c_out)], 1)
        return enc, hs_out

    return forward


# ---- aggancio / sgancio -----------------------------------------------------------------------------------------
def validate_transformer(tr) -> None:
    """Il port richiede il FluxTransformer2DModel diffusers (blocchi con FluxAttention non fuso)."""
    name = type(tr).__name__
    if "nunchaku" in type(tr).__module__.lower() or "Nunchaku" in name:
        raise MQUnsupported("MagicQuill V2 non supporta i checkpoint nunchaku/SVDQ (kernel FP4 custom): usa un "
                            "FLUX.1 Kontext fp8/bf16 o GGUF")
    for attr in ("transformer_blocks", "single_transformer_blocks", "time_text_embed"):
        if not hasattr(tr, attr):
            raise MQUnsupported(f"transformer '{name}' senza '{attr}': non è un FluxTransformer2DModel diffusers")
    b0 = tr.transformer_blocks[0]
    for attr in ("norm1", "norm1_context", "norm2", "norm2_context", "ff", "ff_context", "attn"):
        if not hasattr(b0, attr):
            raise MQUnsupported(f"blocco doppio senza '{attr}': struttura diffusers diversa da 0.38")
    for a in ("to_q", "to_k", "to_v", "norm_q", "norm_k", "add_q_proj", "norm_added_q", "to_add_out"):
        if not hasattr(b0.attn, a):
            raise MQUnsupported(f"attention senza '{a}': struttura diffusers diversa da 0.38")
    if getattr(b0.attn, "fused_projections", False):
        raise MQUnsupported("proiezioni qkv fuse (fuse_qkv_projections): non supportate dal port MagicQuill")


@contextlib.contextmanager
def patch_transformer(tr, st: MQState):
    """Sostituisce temporaneamente `forward` di TUTTI i blocchi e aggancia la cattura degli argomenti di
    `time_text_embed` (per cond_temb). Ripristina SEMPRE (anche su errore) e svuota la banca."""
    validate_transformer(tr)
    saved: list[tuple[Any, Any]] = []
    st.tte_module = tr.time_text_embed

    def _hook(module, args, kwargs):
        if not st._busy and st.cond_temb is None:
            st.tte_args = (args, kwargs)

    handle = tr.time_text_embed.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        for i, b in enumerate(tr.transformer_blocks):
            saved.append((b, b.__dict__.get("forward")))
            b.forward = make_double_forward(b, i, st)
        for i, b in enumerate(tr.single_transformer_blocks):
            saved.append((b, b.__dict__.get("forward")))
            b.forward = make_single_forward(b, i, st)
        yield st
    finally:
        handle.remove()
        for b, old in saved:
            if old is None:
                b.__dict__.pop("forward", None)
            else:
                b.__dict__["forward"] = old
        st.clear()
        st.tte_module = None
