"""Lettore GGUF → pesi Hugging Face di Qwen3.5 (solo modello di linguaggio) per l'upsampler FLUX.3 di terzi (Sakura, abliterato).

transformers 5.x NON carica GGUF dell'architettura `qwen35` (ggml.py ha solo il tokenizer), quindi si legge il file da qui:
parser GGUF v3 (nessuna dipendenza), dequantizzazione Q8_0/F16/BF16/F32 (un tensore alla volta, senza mmap), poi le
trasformazioni INVERSE di llama.cpp (convert_hf_to_gguf, classe Qwen3.5) per tornare ai nomi/ordini del checkpoint HF:
  · norme `(1 + w)` (input/post-attention/q/k/finale): llama.cpp salva w+1 → si sottrae 1; la norma gated di ssm NO;
  · A_log = log(-ssm_a); conv1d (C, K) → (C, 1, K);
  · teste V dell'attenzione lineare riordinate da «raggruppate per testa K» (HF) a «a piastrelle» (ggml): permutazione
    `perm = arange(nv).reshape(nv // nk, nk).T.reshape(-1)` (la testa HF i è la testa GGUF perm[i]) su: righe V di in_proj_qkv,
    righe di in_proj_z/a/b, A_log, dt_bias, canali V di conv1d, COLONNE di out_proj.
VERIFICATO il 18/10/2026 su tensori piccoli del file Sakura Q8_0 contro l'upsampler ufficiale BFL (letti via range HTTP, ~10 MB in
tutto: norme, A_log, dt_bias, conv1d, in_proj_a/b, righe di in_proj_qkv/z, out_proj, q/k/v/o, ffn): scarto massimo = errore di
quantizzazione Q8_0 (≤ 1e-3 su pesi di scala 1e-2), esatto sui tensori non quantizzati. La prova sta in
`orch03/task13/work/gguf_validate.py`; il test `tests/test_flux3_gguf.py` copre parser e trasformazioni su un file sintetico.
Non letti dal file (non servono): blocco 32 (testa MTP) e torre visiva (il proiettore visivo è un GGUF a parte).
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

GGML_F32, GGML_F16, GGML_Q8_0, GGML_BF16 = 0, 1, 8, 30
_NP = {0: 4, 1: 2, 30: 2}                        # byte per elemento dei tipi non quantizzati
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class _R:
    def __init__(self, f):
        self.f = f

    def u(self, fmt: str):
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))

    def s(self) -> str:
        (n,) = self.u("<Q")
        return self.f.read(n).decode("utf-8", "replace")

    def val(self, t: int):
        if t == 8:
            return self.s()
        if t == 9:
            (et,) = self.u("<I")
            (n,) = self.u("<Q")
            if et == 8:
                for _ in range(n):
                    self.s()
            else:
                self.f.read(struct.calcsize(_SCALARS[et]) * n)
            return None                             # gli array (tokenizer, ruoli) non servono qui
        return self.u(_SCALARS[t])[0]


def read_header(path: str) -> tuple[dict, list, int]:
    """(metadati scalari, [(nome, dims ggml, tipo, offset)], inizio dei dati)."""
    with open(path, "rb") as f:
        r = _R(f)
        if f.read(4) != b"GGUF":
            raise RuntimeError(f"{path}: non è un file GGUF")
        (ver,) = r.u("<I")
        if ver < 2:
            raise RuntimeError(f"GGUF v{ver} non supportato")
        (nt,) = r.u("<Q")
        (nkv,) = r.u("<Q")
        kv = {}
        for _ in range(nkv):
            k = r.s()
            (t,) = r.u("<I")
            kv[k] = r.val(t)
        tens = []
        for _ in range(nt):
            name = r.s()
            (nd,) = r.u("<I")
            dims = r.u("<" + "Q" * nd)
            (ty,) = r.u("<I")
            (off,) = r.u("<Q")
            tens.append((name, tuple(dims), ty, off))
        align = int(kv.get("general.alignment", 32) or 32)
        pos = f.tell()
        return kv, tens, pos + (-pos) % align


def _read_dequant(f, base: int, dims: tuple, ty: int, off: int, device: str):
    """Un tensore → torch bf16 sul device, con la forma numpy (dims invertite)."""
    import torch
    n = 1
    for d in dims:
        n *= d
    shape = tuple(reversed(dims))
    f.seek(base + off)
    if ty == GGML_Q8_0:
        nb = n // 32
        raw = torch.frombuffer(bytearray(f.read(nb * 34)), dtype=torch.uint8).to(device).reshape(nb, 34)
        sc = raw[:, :2].contiguous().view(torch.float16).float()               # (nb, 1)
        q = raw[:, 2:].contiguous().view(torch.int8).float()
        return (q * sc).reshape(shape).to(torch.bfloat16)
    if ty in _NP:
        raw = torch.frombuffer(bytearray(f.read(n * _NP[ty])), dtype=torch.uint8).to(device)
        dt = {GGML_F32: torch.float32, GGML_F16: torch.float16, GGML_BF16: torch.bfloat16}[ty]
        return raw.view(dt).reshape(shape).to(torch.bfloat16)
    raise RuntimeError(f"tipo GGML {ty} non supportato (servono Q8_0, F16, BF16 o F32): scegli il file Q8_0 dell'upsampler")


def v_perm(n_v_heads: int = 32, n_k_heads: int = 16):
    """Testa HF i ← testa GGUF perm[i] (HF: raggruppate per testa K; ggml: a piastrelle)."""
    import torch
    return torch.arange(n_v_heads).reshape(n_v_heads // n_k_heads, n_k_heads).T.reshape(-1)


_BLK = re.compile(r"^blk\.(\d+)\.(.+)$")
_SIMPLE = {                                       # nome GGUF (dopo blk.N.) → (nome HF, sottrai 1 [norma (1+w)])
    "attn_norm.weight": ("input_layernorm.weight", True),
    "post_attention_norm.weight": ("post_attention_layernorm.weight", True),
    "ffn_gate.weight": ("mlp.gate_proj.weight", False),
    "ffn_up.weight": ("mlp.up_proj.weight", False),
    "ffn_down.weight": ("mlp.down_proj.weight", False),
    "attn_q.weight": ("self_attn.q_proj.weight", False),
    "attn_k.weight": ("self_attn.k_proj.weight", False),
    "attn_v.weight": ("self_attn.v_proj.weight", False),
    "attn_output.weight": ("self_attn.o_proj.weight", False),
    "attn_q_norm.weight": ("self_attn.q_norm.weight", True),
    "attn_k_norm.weight": ("self_attn.k_norm.weight", True),
    "ssm_norm.weight": ("linear_attn.norm.weight", False),     # norma gated: NON (1+w)
    "ssm_dt.bias": ("linear_attn.dt_bias", False),             # + perm
    "ssm_a": ("linear_attn.A_log", False),                     # log(-a) + perm
    "ssm_alpha.weight": ("linear_attn.in_proj_a.weight", False),   # + perm righe
    "ssm_beta.weight": ("linear_attn.in_proj_b.weight", False),    # + perm righe
    "attn_gate.weight": ("linear_attn.in_proj_z.weight", False),   # + perm righe a blocchi di testa
    "attn_qkv.weight": ("linear_attn.in_proj_qkv.weight", False),  # + perm righe V
    "ssm_conv1d.weight": ("linear_attn.conv1d.weight", False),     # + perm canali V + unsqueeze
    "ssm_out.weight": ("linear_attn.out_proj.weight", False),      # + perm COLONNE
}
_TOP = {"token_embd.weight": ("model.language_model.embed_tokens.weight", False),
        "output.weight": ("lm_head.weight", False),
        "output_norm.weight": ("model.language_model.norm.weight", True)}


def convert(name: str, w, perm, head_dim: int, key_dim: int):
    """(nome HF, tensore HF) da (nome GGUF, tensore dequantizzato). None se il tensore non serve."""
    import torch
    if name in _TOP:
        hf, minus = _TOP[name]
        return hf, (w.float() - 1.0).to(torch.bfloat16) if minus else w
    m = _BLK.match(name)
    if not m:
        raise RuntimeError(f"tensore GGUF inatteso: {name}")
    i, rest = int(m.group(1)), m.group(2)
    if rest not in _SIMPLE:
        raise RuntimeError(f"tensore GGUF inatteso: {name}")
    hf, minus = _SIMPLE[rest]
    pre = f"model.language_model.layers.{i}."
    p = perm.to(w.device)
    nv = len(perm)
    if minus:
        w = (w.float() - 1.0).to(torch.bfloat16)
    elif rest == "ssm_a":
        w = torch.log(-w.float())[p].to(torch.bfloat16)
    elif rest in ("ssm_dt.bias", "ssm_alpha.weight", "ssm_beta.weight"):
        w = w[p]
    elif rest == "attn_gate.weight":
        w = w.reshape(nv, head_dim, -1)[p].reshape(nv * head_dim, -1)
    elif rest == "attn_qkv.weight":
        v0 = 2 * key_dim
        w = torch.cat([w[:v0], w[v0:].reshape(nv, head_dim, -1)[p].reshape(nv * head_dim, -1)], 0)
    elif rest == "ssm_conv1d.weight":
        v0 = 2 * key_dim
        w = torch.cat([w[:v0], w[v0:].reshape(nv, head_dim, -1)[p].reshape(nv * head_dim, -1)], 0).unsqueeze(1)
    elif rest == "ssm_out.weight":
        w = w.reshape(w.shape[0], nv, head_dim)[:, p].reshape(w.shape[0], nv * head_dim)
    return pre + hf, w.contiguous()


def read_state(path: str, device: str = "cuda") -> dict:
    """State dict HF (bf16, sul device) del solo modello di linguaggio. Salta la testa MTP (blocco oltre l'ultimo livello)."""
    kv, tens, base = read_header(str(path))
    arch = kv.get("general.architecture")
    if arch != "qwen35":
        raise RuntimeError(f"{Path(path).name}: architettura GGUF «{arch}», attesa qwen35 (upsampler FLUX.3 = Qwen3.5-9B)")
    n_layers = int(kv["qwen35.block_count"]) - int(kv.get("qwen35.nextn_predict_layers", 0) or 0)
    nv = int(kv["qwen35.ssm.time_step_rank"])
    nk = int(kv["qwen35.ssm.group_count"])
    head_dim = int(kv["qwen35.ssm.state_size"])
    key_dim = nk * head_dim
    perm = v_perm(nv, nk)
    out = {}
    with open(path, "rb") as f:
        for name, dims, ty, off in tens:
            m = _BLK.match(name)
            if m and int(m.group(1)) >= n_layers:
                continue
            w = _read_dequant(f, base, dims, ty, off, device)
            hf, t = convert(name, w, perm, head_dim, key_dim)
            out[hf] = t
    return out
