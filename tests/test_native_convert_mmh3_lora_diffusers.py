"""core/native_convert: converter «mmh3_lora_diffusers» (LoRA H3 lightx2v in formato diffusers/peft → nativo): q/k/v fusi, alpha/rank cotto in B, gate_up/out."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")
from core import native_convert as nc  # noqa: E402


def _write(path: Path, tensors: dict, meta: dict):
    safetensors.save_file(tensors, str(path), metadata=meta)


def _src(tmp_path, alpha="2", dtype=torch.float32):
    g = torch.Generator().manual_seed(0)
    r, din, dq, dff = 4, 6, 5, 7
    t = {}
    for blk in ("transformer_blocks.0", "token_refiner.refiner_blocks.1"):
        for n in ("to_q", "to_k", "to_v"):
            t[f"{blk}.attn.{n}.lora_A.default.weight"] = torch.randn(r, din, generator=g).to(dtype)
            t[f"{blk}.attn.{n}.lora_B.default.weight"] = torch.randn(dq, r, generator=g).to(dtype)
        t[f"{blk}.attn.to_out.0.lora_A.default.weight"] = torch.randn(r, dq, generator=g).to(dtype)
        t[f"{blk}.attn.to_out.0.lora_B.default.weight"] = torch.randn(din, r, generator=g).to(dtype)
        t[f"{blk}.ff.net.0.proj.lora_A.default.weight"] = torch.randn(r, din, generator=g).to(dtype)
        t[f"{blk}.ff.net.0.proj.lora_B.default.weight"] = torch.randn(2 * dff, r, generator=g).to(dtype)
        t[f"{blk}.ff.net.2.lora_A.default.weight"] = torch.randn(r, dff, generator=g).to(dtype)
        t[f"{blk}.ff.net.2.lora_B.default.weight"] = torch.randn(din, r, generator=g).to(dtype)
    src = tmp_path / "turbo_diffusers.safetensors"
    _write(src, t, {"key_format": "minimax-h3-diffusers", "alpha": alpha, "format": "pt"})
    return src, t, float(alpha) / r


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_diffusers_lora_becomes_native_with_fused_qkv_and_baked_alpha(tmp_path, dtype):
    src, t, scale = _src(tmp_path, dtype=dtype)
    dst = tmp_path / "turbo-native.safetensors"
    out = nc.convert("mmh3_lora_diffusers", src, dst, delete_source=False, log=lambda m: None)
    assert out == dst and nc.is_native(dst) and src.exists()                       # l'originale NON si cancella (decide l'utente)
    from safetensors.torch import load_file
    n = load_file(str(dst))
    for blk in ("transformer_blocks.0", "token_refiner.refiner_blocks.1"):
        keys = {k for k in n if k.startswith(blk + ".")}
        assert keys == {f"{blk}.attn.to_qkv.lora_A.weight", f"{blk}.attn.to_qkv.lora_B.weight", f"{blk}.attn.to_out.0.lora_A.weight",
                        f"{blk}.attn.to_out.0.lora_B.weight", f"{blk}.ff.gate_up.lora_A.weight", f"{blk}.ff.gate_up.lora_B.weight",
                        f"{blk}.ff.out.lora_A.weight", f"{blk}.ff.out.lora_B.weight"}, keys
        a, b = n[f"{blk}.attn.to_qkv.lora_A.weight"].float(), n[f"{blk}.attn.to_qkv.lora_B.weight"].float()
        assert a.shape == (12, 6) and b.shape == (15, 12)                        # rank 3x4, uscita 3x5
        delta = b @ a                                                           # il delta fuso = i tre delta separati impilati, con scala alpha/rank
        for i, nm in enumerate(("to_q", "to_k", "to_v")):
            ref = t[f"{blk}.attn.{nm}.lora_B.default.weight"].float() @ t[f"{blk}.attn.{nm}.lora_A.default.weight"].float() * scale
            tol = 2e-2 if dtype == torch.bfloat16 else 1e-5
            assert torch.allclose(delta[i * 5:(i + 1) * 5], ref, atol=tol, rtol=tol), (blk, nm)
        ref = t[f"{blk}.ff.net.0.proj.lora_B.default.weight"].float() @ t[f"{blk}.ff.net.0.proj.lora_A.default.weight"].float() * scale
        got = n[f"{blk}.ff.gate_up.lora_B.weight"].float() @ n[f"{blk}.ff.gate_up.lora_A.weight"].float()
        assert torch.allclose(got, ref, atol=2e-2, rtol=2e-2)
    assert not any(k.endswith(".alpha") for k in n)                              # niente alpha: il nativo applica scala 1


def test_native_lora_loads_like_the_other_native_h3_loras(tmp_path):
    """apply_lora del worker legge i nomi nativi (to_qkv, ff.gate_up/out): stesso formato delle LoRA gia convertite."""
    src, t, _ = _src(tmp_path)
    dst = tmp_path / "x-native.safetensors"
    nc.convert("mmh3_lora_diffusers", src, dst, delete_source=False, log=lambda m: None)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))
    import _qlinear as Q
    import torch.nn as nn

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_qkv = nn.Linear(6, 15, bias=False)
            self.to_out = nn.ModuleList([nn.Linear(5, 6, bias=False)])

    class FF(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up = nn.Linear(6, 14, bias=False)
            self.out = nn.Linear(7, 6, bias=False)

    class Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attn()
            self.ff = FF()

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList([Blk()])
    m = M()
    from safetensors.torch import load_file
    sd = {k: v for k, v in load_file(str(dst)).items() if k.startswith("transformer_blocks.")}
    assert Q.apply_lora(m, sd, 1.0) == 4                                         # qkv, to_out, gate_up, out
