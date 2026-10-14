"""DiT H3 W4A8/W6A8 (comfy-kitchen, «DaSiWa» int4/int6): il convertitore nativo lascia invariati i pesi impacchettati
e aggiunge il marcatore `.w4a8`; `_qlinear` li monta in W4A8Linear con la stessa uscita del riferimento comfy-kitchen."""
import json
import os
import sys

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "workers"))

from core import native_convert as NC  # noqa: E402
import _qlinear as Q  # noqa: E402

N, K = 64, 512


def _fake_w4a8(bits=4):
    gs = 16 if bits == 4 else 32
    sd = {"weight": torch.randint(-128, 127, (N, K * bits // 8), dtype=torch.int8),
          "weight_s_rel": torch.rand(N, K // gs).to(torch.float8_e4m3fn),
          "weight_s_channel": torch.rand(N)}
    if bits == 4:
        sd["weight_codebook"] = torch.linspace(-1, 1, 16)
    return sd, gs


@pytest.mark.parametrize("bits", [4, 6])
def test_convert_passthrough_e_marcatore(tmp_path, bits):
    sd, gs = _fake_w4a8(bits)
    src = {f"blocks.0.attn.out_proj.{k}": v for k, v in sd.items()}
    src["blocks.0.norm1.weight"] = torch.ones(K, dtype=torch.bfloat16)
    fmt = "asym_w4a8_int8" if bits == 4 else "w6a8_int8"
    if bits == 4:                       # metadata globale (file int4 DaSiWa)
        md = {"_quantization_metadata": json.dumps({"layers": {"blocks.0.attn.out_proj": {
            "format": fmt, "group_size": gs, "convrot": True, "convrot_groupsize": 256}}})}
    else:                               # tensore comfy_quant per layer (file int6 DaSiWa)
        md = {}
        cq = json.dumps({"format": fmt, "group_size": gs, "convrot": True, "convrot_groupsize": 256}).encode()
        src["blocks.0.attn.out_proj.comfy_quant"] = torch.tensor(list(cq), dtype=torch.uint8)
    save_file(src, str(tmp_path / "src.safetensors"), metadata=md)
    NC.convert("mmh3_dit", tmp_path / "src.safetensors", tmp_path / "dst.safetensors", delete_source=False,
               log=lambda *_: None)
    out = load_file(str(tmp_path / "dst.safetensors"))
    m = "transformer_blocks.0.attn.to_out.0"
    assert out[f"{m}.w4a8"].tolist() == [gs, 256]
    for k, v in sd.items():
        assert torch.equal(out[f"{m}.{k}"].view(torch.uint8), v.view(torch.uint8)), k
    assert f"{m}.comfy_quant" not in out and f"{m}.convrot_group" not in out


def test_loader_monta_w4a8linear():
    sd, gs = _fake_w4a8(4)
    model = nn.Sequential(nn.Linear(K, N, bias=True))
    st = {f"0.{k}": v for k, v in sd.items()}
    st["0.bias"] = torch.zeros(N)
    st["0.w4a8"] = torch.tensor([gs, 256], dtype=torch.int32)
    out = Q.quantize_from_state_dict(model, st)
    assert isinstance(model[0], Q.W4A8Linear) and model[0].group == gs and model[0].convrot == 256
    miss, unex = Q.load_into(model, out, device="cpu")
    assert not miss and not unex
    assert model[0].s_channel.dtype == torch.float32 and model[0].codebook.dtype == torch.float32


def _ck():
    try:
        import comfy_kitchen as ck
        return ck if hasattr(ck, "w4a8_int8_linear") and torch.cuda.is_available() else None
    except ImportError:
        return None


@pytest.mark.skipif(_ck() is None, reason="serve comfy-kitchen ≥ 0.2.37 e CUDA (venv diffusers_next)")
@pytest.mark.parametrize("bits", [4, 6])
def test_forward_come_riferimento_e_lora(bits):
    ck = _ck()
    torch.manual_seed(0)
    gs = 16 if bits == 4 else 32
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    q, s_rel, s_ch, corr, cb = ck.quantize_w4a8_int8_weight(w, group_size=gs, codebook=bits == 4, bits=bits)
    st = {"0.weight": q, "0.weight_s_rel": s_rel, "0.weight_s_channel": s_ch,
          "0.w4a8": torch.tensor([gs, 256], dtype=torch.int32)}
    if cb is not None:
        st["0.weight_codebook"] = cb
    if corr is not None:
        st["0.weight_correction"] = corr
    model = nn.Sequential(nn.Linear(K, N, bias=False))
    Q.load_into(model, Q.quantize_from_state_dict(model, st), device="cuda")
    x = torch.randn(2, 40, K, device="cuda", dtype=torch.bfloat16)
    ref = ck.w4a8_int8_linear(x, q, s_rel, s_ch, cb, corr, group_size=gs)
    y = model(x)
    assert torch.allclose(y.float(), ref.float(), atol=1e-3, rtol=1e-3)
    # LoRA al volo sul modulo int4: y += s·B·A·x con l'ingresso NON ruotato
    a = torch.randn(4, K, device="cuda") * 0.01
    b = torch.randn(N, 4, device="cuda") * 0.01
    assert Q.apply_lora(model, {"0.lora_A.weight": a, "0.lora_B.weight": b}) == 1
    exp = ref.float() + 1.0 * (x.float() @ a.t() @ b.t())
    assert torch.allclose(model(x).float(), exp, atol=2e-3, rtol=2e-2)
