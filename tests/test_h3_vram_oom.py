"""Task 22 — MiniMax H3 non va più in OOM con le LoRA a 10 s (campagna del 29/10: 51 test su 51 senza output).

Tutto su CPU con tensori finti: nessuna GPU. Cosa si prova:
  · LoRA sommata IN PLACE a blocchi di token == percorso denso, senza copie intere di x;
  · se un blocco va in OOM viene dimezzato e il risultato resta identico (nessuna LoRA applicata due volte);
  · pianificazione VRAM: la 5090 regge 10 s a 1376×736, una scheda da 24 GB usa l'offload, una da 12 GB riceve
    un messaggio chiaro con la durata che ci starebbe;
  · offload a blocchi: stesso risultato, pesi di nuovo in RAM a fine forward;
  · un OOM arriva alla GUI come «memoria GPU esaurita», non come «nessun output».
"""
import importlib.util
import os
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "workers"))

import _qlinear as Q  # noqa: E402
import h3_offload as O  # noqa: E402
from core import h3_vram_plan as HP  # noqa: E402


def _layer(inf=24, outf=40, seed=0, rank=4, n_lora=2, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    m = Q.RowInt8Linear(inf, outf, bias=False)
    m.weight = (torch.randn(outf, inf, generator=g) * 20).round().clamp(-100, 100).to(torch.int8)
    m.scale = torch.full((outf, 1), 0.01)
    m.zero_point = torch.zeros(outf, 1)
    for k in range(n_lora):
        a = (torch.randn(rank, inf, generator=g) * 0.2).to(dtype)
        b = (torch.randn(outf, rank, generator=g) * 0.2).to(dtype)
        m.add_lora(a, b, 0.5 + 0.25 * k)
    return m


def _dense_ref(m, x):
    """Riferimento: il vecchio percorso `y + s·B(A(x))` con la copia intera di x."""
    cd = torch.bfloat16 if x.dtype == torch.float32 else x.dtype
    w = (m.weight.to(cd) - m.zero_point.to(cd)) * m.scale.to(cd)
    y = torch.nn.functional.linear(x.to(cd), w)
    return m._lora_dense(x.to(cd), y)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk", ["1", "7", "64", "100000"])
def test_lora_a_blocchi_uguale_al_denso(monkeypatch, dtype, chunk):
    monkeypatch.setenv("QLINEAR_LORA_CHUNK", chunk)
    m = _layer(dtype=dtype)
    x = torch.randn(2, 53, 24, dtype=dtype)                  # [batch, token, canali]
    with torch.no_grad():
        got = m(x)
        ref = _dense_ref(m, x).to(x.dtype)
    assert got.shape == x.shape[:-1] + (40,)
    assert torch.allclose(got.float(), ref.float(), atol=2e-2, rtol=2e-2), (got - ref).abs().max()


def test_lora_in_place_e_cast_per_blocco(monkeypatch):
    """x in fp32 e y in bf16: nessun x.to(bf16) intero; i blocchi visti da _lora_term sono ≤ chunk."""
    monkeypatch.setenv("QLINEAR_LORA_CHUNK", "10")
    seen = []
    orig = Q._lora_term

    def spy(xc, lo):
        seen.append((xc.shape[0], xc.dtype))
        return orig(xc, lo)

    monkeypatch.setattr(Q, "_lora_term", spy)
    m = _layer()
    x = torch.randn(95, 24)                                  # fp32
    y = torch.zeros(95, 40, dtype=torch.bfloat16)
    ptr = y.data_ptr()
    with torch.no_grad():
        out = m._lora(x, y)
    assert out.data_ptr() == ptr, "la somma deve essere in place su y"
    assert seen and max(n for n, _ in seen) <= 10
    assert all(dt == torch.bfloat16 for _, dt in seen), "il cast è per blocco, al dtype di y"


def test_lora_lokr_a_blocchi(monkeypatch):
    monkeypatch.setenv("QLINEAR_LORA_CHUNK", "9")
    torch.manual_seed(3)
    m = _layer(n_lora=0)
    w1, w2 = torch.randn(2, 3).bfloat16(), (torch.randn(20, 8) * 0.1).bfloat16()      # kron → 40 × 24
    m.add_lokr(w1, w2, 0.7)
    m.add_lora((torch.randn(4, 24) * 0.2).bfloat16(), (torch.randn(40, 4) * 0.2).bfloat16(), 0.4)
    x = torch.randn(41, 24, dtype=torch.bfloat16)
    with torch.no_grad():
        got, ref = m(x), _dense_ref(m, x)
    assert torch.allclose(got.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_oom_dimezza_il_blocco_senza_doppie_somme(monkeypatch):
    """Il term va in OOM per blocchi > 20 righe: il risultato deve restare quello dell'esecuzione senza OOM."""
    monkeypatch.setenv("QLINEAR_LORA_CHUNK", "64")
    monkeypatch.setattr(Q, "LORA_CHUNK_MIN", 4)
    m = _layer(n_lora=3)
    x = torch.randn(100, 24, dtype=torch.bfloat16)
    with torch.no_grad():
        ok = m(x).clone()
    orig = Q._lora_term
    chiamate = []

    def fragile(xc, lo):
        chiamate.append(xc.shape[0])
        if xc.shape[0] > 20:
            raise torch.OutOfMemoryError("CUDA out of memory finto")
        return orig(xc, lo)

    monkeypatch.setattr(Q, "_lora_term", fragile)
    with torch.no_grad():
        dopo = m(x)
    assert torch.equal(dopo, ok), "una LoRA è stata applicata due volte o saltata"
    assert max(chiamate) > 20 and min(chiamate) <= 20


def test_oom_al_minimo_solleva(monkeypatch):
    monkeypatch.setenv("QLINEAR_LORA_CHUNK", "300")
    m = _layer()

    def sempre(xc, lo):
        raise torch.OutOfMemoryError("CUDA out of memory finto")

    monkeypatch.setattr(Q, "_lora_term", sempre)
    with torch.no_grad(), pytest.raises(torch.OutOfMemoryError):
        m(torch.randn(900, 24, dtype=torch.bfloat16))


def test_con_autograd_resta_il_percorso_denso():
    m = _layer()
    x = torch.randn(5, 24, dtype=torch.bfloat16, requires_grad=True)
    y = m(x)
    assert y.requires_grad


# ── pianificazione VRAM ─────────────────────────────────────────────────────────────────────────
def test_token_del_caso_della_campagna():
    """1376×736, 243 fotogrammi (10 s): ~72k token, come l'allocazione da 2,89 GiB del qkv nel log."""
    t = HP.sequence_tokens(243, 736, 1376)
    assert 70_000 < t < 78_000
    qkv_gib = t * 3 * 7168 * 2 / 2**30
    assert 2.8 < qkv_gib < 3.2
    assert HP.latent_frames(243) == 72
    assert HP.sequence_tokens(243, 544, 960) < t / 1.8      # la tela 544p ha meno della metà dei token


def test_5090_regge_10s_a_768p_con_lora():
    t = HP.sequence_tokens(243, 736, 1376)
    p = HP.plan(29.4, t, weights_gb=13.6, lora_gb=0.6)
    assert p["fits"] and p["offload_gb"] == 0


def test_scheda_da_24gb_usa_l_offload_e_da_12_gb_riceve_un_messaggio_chiaro():
    t = HP.sequence_tokens(243, 736, 1376)
    p24 = HP.plan(22.5, t, weights_gb=13.6, lora_gb=0.6)
    assert p24["fits"] and 0 < p24["offload_gb"] < 0.85 * 14.2
    p12 = HP.plan(11.0, t, weights_gb=13.6, lora_gb=0.6)
    assert not p12["fits"]                                   # nemmeno i pesi stanno: niente OOM a metà
    msg = HP.explain_failure(p12, height=736, width=1376, num_frames=243, free_gb=11.0,
                             canvas_544=(544, 960))
    assert "VRAM insufficiente" in msg and "11.0" in msg and "Prova a" in msg


def test_non_ci_sono_soglie_fisse_la_stima_scala_con_la_durata():
    short = HP.plan(20, HP.sequence_tokens(101, 736, 1376), weights_gb=13.6, lora_gb=0.6)
    long_ = HP.plan(20, HP.sequence_tokens(365, 736, 1376), weights_gb=13.6, lora_gb=0.6)
    assert short["need_gb"] < long_["need_gb"]
    assert short["offload_gb"] < long_["offload_gb"]
    assert HP.max_frames_that_fit(20, 736, 1376, weights_gb=13.6, lora_gb=0.6) >= 101


def test_dims_dal_config():
    d = HP.dims_from_config({"hidden_size": 100, "num_attention_heads": 4, "attention_head_dim": 8, "ffn_dim": 7})
    assert d == {"hidden": 100, "inner": 32, "ffn": 7}
    assert HP.dims_from_config(None) == HP.DEFAULT_DIMS


# ── offload a blocchi ───────────────────────────────────────────────────────────────────────────
class _Blocco(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.lin = _layer(seed=seed)
        self.norm = nn.LayerNorm(40)

    def forward(self, x):
        return self.norm(self.lin(x))


def test_offload_stesso_risultato_e_pesi_di_nuovo_in_ram(monkeypatch):
    monkeypatch.setenv("QLINEAR_LORA_CHUNK", "16")
    blocchi = nn.ModuleList([_Blocco(s) for s in range(4)])
    for b in blocchi:                                         # ingresso/uscita 40: le dimensioni devono combaciare
        b.lin = _layer(inf=40, outf=40, seed=1)
    x = torch.randn(30, 40, dtype=torch.bfloat16)
    with torch.no_grad():
        ref = x
        for b in blocchi:
            ref = b(ref)
        assert O.block_bytes(blocchi[0]) > 0
        n = O.blocks_for_deficit(blocchi, deficit_gb=1e-9)
        assert n == 1
        assert O.blocks_for_deficit(blocchi, deficit_gb=1e9) == int(4 * 0.85)
        assert O.enable_block_offload(blocchi, 2, "cpu") == 2
        got = x
        for b in blocchi:
            got = b(got)
    assert torch.equal(got, ref)
    # a fine forward i tensori sono quelli di RAM e le LoRA ci sono ancora
    assert blocchi[3].lin.weight.device.type == "cpu" and len(blocchi[3].lin.loras) == 2


# ── errore chiaro verso la GUI ──────────────────────────────────────────────────────────────────
def _worker():
    spec = importlib.util.spec_from_file_location("mmh3w22", os.path.join(ROOT, "workers", "minimax_h3_infer.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_oom_diventa_un_messaggio_per_la_gui():
    w = _worker()
    msg = w.friendly_error(torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.89 GiB."))
    assert "memoria GPU" in msg and "VRAM" in msg and "2.89" in msg
    assert w.friendly_error(RuntimeError("altro")) == "altro"
