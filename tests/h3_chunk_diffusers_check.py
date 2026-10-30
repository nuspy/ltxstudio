"""Controllo del blocco H3 a token-chunk contro il MiniMaxH3TransformerBlock REALE di diffusers (CPU, pesi casuali).

Va lanciato col venv di H3 (diffusers recente col modello MiniMax):
    E:\\AI\\Tools\\diffusers_next_venv\\Scripts\\python.exe tests\\h3_chunk_diffusers_check.py
Lo lancia anche tests/test_h3_vram_oom.py (salta se il venv non c'e'). Stampa «OK» e esce con 0 se il blocco a
blocchi coincide con quello originale (fused e non fuso, fp32 e bf16, blocchi che non dividono N, OOM simulato).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "workers"))
os.environ["H3_BLOCK_CHUNK"] = "7"
os.environ["H3_ATTN_QCHUNK"] = "9"

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from diffusers.models.transformers.transformer_minimax_h3 import (  # noqa: E402
    MiniMaxH3RotaryPosEmbed, MiniMaxH3TransformerBlock)

import h3_chunk as K  # noqa: E402

HID, HEADS, HD, FFN, TDIM, N = 48, 4, 16, 64, 24, 53


def make(dtype, fused: bool):
    torch.manual_seed(0)
    blk = MiniMaxH3TransformerBlock(HID, HEADS, HD, FFN, TDIM, 1e-6, 1e-6).to(dtype).eval()
    if fused:
        a = blk.attn
        w = torch.cat([a.to_q.weight, a.to_k.weight, a.to_v.weight], 0)
        a.to_qkv = nn.Linear(HID, w.shape[0], bias=False).to(dtype)
        a.to_qkv.weight.data = w.clone()
        del a.to_q, a.to_k, a.to_v
        a.fused_projections = True
    return blk


def inputs(dtype):
    torch.manual_seed(1)
    hs = torch.randn(1, N, HID, dtype=dtype)
    temb = torch.randn(2, TDIM, dtype=dtype)                     # 2 istanti x 3 modalita'
    idx = (torch.randint(0, 2, (N,)) * 3 + torch.randint(0, 3, (N,)))
    pos = torch.stack([torch.arange(N), torch.arange(N) % 5, torch.arange(N) % 7], -1).float()
    cos, sin = MiniMaxH3RotaryPosEmbed(rope_freq_dim=2)(pos)       # (N, 12) <= head_dim 16
    return hs, temb, idx, (cos, sin)


def check(dtype, fused, tol):
    import diffusers.models.attention_dispatch as AD
    calls = []
    orig_d = AD.dispatch_attention_fn
    AD.dispatch_attention_fn = lambda q, k, v, **kw: (calls.append(q.shape[1]), orig_d(q, k, v, **kw))[1]
    blk = make(dtype, fused)
    hs, temb, idx, rope = inputs(dtype)
    with torch.no_grad():
        ref = blk(hs.clone(), temb, idx, rope)
        K.enable_chunked_blocks(type("T", (), {"transformer_blocks": [blk]})())
        got = blk(hs.clone(), temb, idx, rope)
    AD.dispatch_attention_fn = orig_d
    if fused:   # attenzione a blocchi di QUERY: ceil(53/9) = 6 chiamate, K/V pieni
        assert calls == [9, 9, 9, 9, 9, 8], calls
    err = (got.float() - ref.float()).abs().max().item()
    assert got.shape == ref.shape and got.dtype == ref.dtype, (got.shape, got.dtype, ref.dtype)
    assert err <= tol, f"dtype={dtype} fused={fused}: errore massimo {err}"
    print(f"  {str(dtype):15s} fused={fused!s:5s} err={err:.2e}")


def check_oom():
    """Un OOM simulato sui blocchi > 20 token deve essere assorbito dimezzando, con lo stesso risultato."""
    blk = make(torch.float32, True)
    hs, temb, idx, rope = inputs(torch.float32)
    K.MIN_CHUNK = 4
    os.environ["H3_BLOCK_CHUNK"] = "40"
    with torch.no_grad():
        ref = blk(hs.clone(), temb, idx, rope)
        K.enable_chunked_blocks(type("T", (), {"transformer_blocks": [blk]})())
        orig = blk.attn.to_qkv.forward

        def fragile(x):
            if x.shape[1] > 20:
                raise torch.OutOfMemoryError("CUDA out of memory finto")
            return orig(x)

        blk.attn.to_qkv.forward = fragile
        got = blk(hs.clone(), temb, idx, rope)
    err = (got - ref).abs().max().item()
    assert err < 1e-5, err
    print(f"  OOM simulato assorbito, err={err:.2e}")


if __name__ == "__main__":
    for fused in (False, True):
        check(torch.float32, fused, 1e-5)
        check(torch.bfloat16, fused, 6e-2)
    check_oom()
    print("OK")
