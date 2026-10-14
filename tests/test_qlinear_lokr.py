"""LoKr (LyCORIS) in workers/_qlinear: applicata al volo sui Linear quantizzati (senza materializzare il kron) e
fusa nei pesi dei nn.Linear normali — deve coincidere con ΔW = s·kron(w1, w2) denso."""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workers"))

import _qlinear as Q  # noqa: E402


def test_lokr_apply_uguale_al_kron_denso():
    torch.manual_seed(0)
    w1, w2 = torch.randn(4, 3), torch.randn(5, 6)
    x = torch.randn(2, 7, 3 * 6)
    ref = x @ torch.kron(w1, w2).T
    assert torch.allclose(Q.lokr_apply(x, w1, w2), ref, atol=1e-4)


def test_lokr_su_linear_quantizzato_e_normale():
    torch.manual_seed(1)
    w1, w2 = torch.randn(2, 2), torch.randn(8, 4) * 0.1
    base = torch.randn(16, 8) * 0.05
    lin = nn.Linear(8, 16, bias=False)
    lin.weight.data.copy_(base)
    q = Q.RowFp8Linear(8, 16, False, False)
    q.weight = base.to(torch.float8_e4m3fn)
    model = nn.Module()
    model.a, model.b = lin, q
    sd = {"a.lokr_w1": w1, "a.lokr_w2": w2, "a.alpha": torch.tensor(99.0),     # pieni: alpha ignorato (scala 1)
          "b.lokr_w1": w1, "b.lokr_w2": w2}
    assert Q.apply_lora(model, sd, 0.5) == 2
    dw = 0.5 * torch.kron(w1, w2)
    x = torch.randn(3, 8)
    assert torch.allclose(lin.weight.float(), base + dw, atol=2e-2)            # fuso (bf16 → tolleranza)
    y = q(x.to(torch.bfloat16)).float()
    ref = x @ (base.to(torch.float8_e4m3fn).float() + dw).T
    assert torch.allclose(y, ref, atol=5e-2), (y - ref).abs().max()


def test_lokr_forma_sbagliata_saltata():
    m = nn.Module()
    m.a = nn.Linear(8, 16, bias=False)
    assert Q.apply_lora(m, {"a.lokr_w1": torch.randn(3, 3), "a.lokr_w2": torch.randn(4, 4)}, 1.0) == 0


def test_diff_su_linear_e_folded_bias():
    """Delta pieni `.diff`/`.diff_b`: peso del Linear e folded_bias del modulo AdaLN padre (H3 pruned)."""
    m = nn.Module()
    m.adaln = nn.Module()
    m.adaln.linear = nn.Linear(8, 6, bias=False)
    m.adaln.register_buffer("folded_bias", torch.zeros(6))
    w0 = m.adaln.linear.weight.detach().clone()
    d, db = torch.randn(6, 8), torch.randn(6)
    assert Q.apply_lora(m, {"adaln.linear.diff": d, "adaln.linear.diff_b": db}, 0.5) == 2
    assert torch.allclose(m.adaln.linear.weight, w0 + 0.5 * d, atol=1e-6)
    assert torch.allclose(m.adaln.folded_bias, 0.5 * db, atol=1e-6)
