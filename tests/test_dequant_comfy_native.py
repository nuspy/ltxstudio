"""Dequant nativo comfy_quant → bf16 (core/dequant_comfy sopra native_convert «comfy_bf16»): fp8 con scala, nvfp4
con scale swizzled cublas, int8 + ConvRot, formato sconosciuto → errore. File sintetici, niente rete/GPU."""
import json
import os
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import dequant_comfy as dq  # noqa: E402
from core import native_convert as nc  # noqa: E402


def _cq(conf: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)


def _to_blocked(m: torch.Tensor) -> torch.Tensor:
    """Layout scale cublas (come comfy/float.py::to_blocked, flatten=False)."""
    rows, cols = m.shape
    nrb, ncb = -(-rows // 128), -(-cols // 4)
    pad = torch.zeros(nrb * 128, ncb * 4, dtype=m.dtype)
    pad[:rows, :cols] = m
    blocks = pad.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)


def test_tre_formati(tmp_path, monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(dq, "CACHE", tmp_path / "cache")
    # fp8 per-tensore
    w8 = (torch.randn(16, 32) * 0.1).to(torch.float8_e4m3fn)
    s8 = torch.tensor(0.5)
    # nvfp4: valori della LUT × scala di blocco (e4m3) × scala globale
    lut = torch.tensor(nc._FP4_VALUES)
    idx = torch.randint(0, 16, (128, 64))
    packed = ((idx[:, 0::2] << 4) | idx[:, 1::2]).to(torch.uint8)
    bscale = (torch.rand(128, 4) + 0.5).to(torch.float8_e4m3fn)
    g = torch.tensor(0.25)
    ref_fp4 = (lut[idx].reshape(128, 4, 16) * (bscale.float() * g).unsqueeze(-1)).reshape(128, 64)
    # int8 + ConvRot
    W = torch.randn(8, 256) * 0.02
    Wr = nc.rotate(W, 256)
    si = Wr.abs().amax(1, keepdim=True) / 127
    qi = torch.round(Wr / si).to(torch.int8)
    src = tmp_path / "m.safetensors"
    save_file({"a.weight": w8, "a.weight_scale": s8, "a.comfy_quant": _cq({"format": "float8_e4m3fn"}),
               "b.weight": packed, "b.weight_scale": _to_blocked(bscale).reshape(128, 4),
               "b.weight_scale_2": g, "b.comfy_quant": _cq({"format": "nvfp4"}),
               "c.weight": qi, "c.weight_scale": si, "c.comfy_quant": _cq({"format": "int8_tensorwise",
                                                                         "convrot": True, "convrot_groupsize": 256}),
               "norm.weight": torch.ones(4, dtype=torch.bfloat16)}, str(src), metadata={"config": "{}"})
    out = dq.ensure_dequantized(str(src))
    assert out != str(src) and src.exists()                          # cache, sorgente intatto
    with safe_open(out, "pt") as f:
        assert set(f.keys()) == {"a.weight", "b.weight", "c.weight", "norm.weight"}
        assert f.metadata()["config"] == "{}"
        assert torch.allclose(f.get_tensor("a.weight").float(), (w8.float() * 0.5).to(torch.bfloat16).float())
        assert torch.allclose(f.get_tensor("b.weight").float(), ref_fp4.to(torch.bfloat16).float())
        cos = torch.nn.functional.cosine_similarity(f.get_tensor("c.weight").float().flatten(), W.flatten(), 0)
        assert cos > 0.999
    assert dq.ensure_dequantized(str(src)) == out                     # seconda volta: dalla cache


def test_formato_sconosciuto(tmp_path, monkeypatch):
    monkeypatch.setattr(dq, "CACHE", tmp_path / "cache")
    src = tmp_path / "x.safetensors"
    save_file({"a.weight": torch.zeros(4, 4, dtype=torch.float8_e4m3fn), "a.weight_scale": torch.ones(4, 1, dtype=torch.uint8),
               "a.comfy_quant": _cq({"format": "mxfp8"})}, str(src))
    with pytest.raises(ValueError, match="mxfp8"):
        dq.ensure_dequantized(str(src))
