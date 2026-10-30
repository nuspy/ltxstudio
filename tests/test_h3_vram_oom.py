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


def test_5090_regge_10s_a_768p_con_lora_con_poco_offload_e_544p_senza():
    """Dopo la prova del 30/10 (attivazioni reali > stima) il piano e' prudente: a 768p/10 s tiene in RAM qualche
    blocco anche su 29 GB liberi; a 544p sta tutto residente."""
    t = HP.sequence_tokens(243, 736, 1376)
    p = HP.plan(29.4, t, weights_gb=13.7, lora_gb=0.5)
    assert p["fits"] and 0 <= p["offload_gb"] < 3.0
    t544 = HP.sequence_tokens(243, 544, 960)
    p2 = HP.plan(29.4, t544, weights_gb=13.7, lora_gb=0.5)
    assert p2["fits"] and p2["offload_gb"] == 0


def test_caso_vivo_30_10_non_e_piu_tutto_residente():
    """Dopo il caricamento (DiT + 2 LoRA = 13,7 GB) restavano 13,5 GB liberi: il vecchio piano diceva «servono
    ~12,7: tutto residente» e il denoise finiva in OOM. Ora il fabbisogno supera i liberi -> offload."""
    t = HP.sequence_tokens(243, 736, 1376)
    act = HP.activation_gb(t, HP.DEFAULT_DIMS)
    assert act + HP.RESERVE_GB >= (11.7 + 1.0) + 3.0, "fabbisogno: almeno +3 GB sul vecchio piano (12,7 GB)"
    assert act + HP.RESERVE_GB > 13.5 - 0.4, "con 13,1 GB liberi non si puo' dire tutto residente"
    p = HP.plan(13.1, t, weights_gb=0.0, lora_gb=0.0)           # come _fit_after_load: pesi gia' in VRAM
    assert p["offload_gb"] > 0


def test_percorso_non_chunked_e_piu_pesante():
    t = HP.sequence_tokens(243, 736, 1376)
    assert HP.activation_gb(t, HP.DEFAULT_DIMS, chunked=False) > HP.activation_gb(t, HP.DEFAULT_DIMS)


def test_scheda_da_24gb_usa_l_offload_e_da_12_gb_riceve_un_messaggio_chiaro():
    t = HP.sequence_tokens(243, 736, 1376)
    p24 = HP.plan(22.5, t, weights_gb=13.6, lora_gb=0.6)
    assert p24["fits"] and 0 < p24["offload_gb"] < 0.85 * 14.2
    p12 = HP.plan(11.0, t, weights_gb=13.6, lora_gb=0.6)
    assert not p12["fits"]                                   # nemmeno i pesi stanno: niente OOM a meta'
    msg = HP.explain_failure(p12, height=736, width=1376, num_frames=243, free_gb=11.0,
                             canvas_544=(544, 960))
    assert "VRAM insufficiente" in msg and "11.0" in msg and "Prova a" in msg


def test_non_ci_sono_soglie_fisse_la_stima_scala_con_la_durata():
    short = HP.plan(24, HP.sequence_tokens(101, 736, 1376), weights_gb=13.6, lora_gb=0.6)
    long_ = HP.plan(24, HP.sequence_tokens(365, 736, 1376), weights_gb=13.6, lora_gb=0.6)
    assert short["need_gb"] < long_["need_gb"]
    assert short["offload_gb"] < long_["offload_gb"]
    assert HP.max_frames_that_fit(24, 736, 1376, weights_gb=13.6, lora_gb=0.6) >= 101


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


# ── MLP e blocco a token-chunk ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ffn_swiglu_a_blocchi_identico(monkeypatch, dtype):
    torch.manual_seed(5)
    gu = nn.Linear(24, 2 * 40, bias=False).to(dtype)
    out = nn.Linear(40, 24, bias=False).to(dtype)
    ff = Q.FusedGateUpSwiGLU(gu, out)
    x = torch.randn(2, 61, 24, dtype=dtype)
    with torch.no_grad():
        monkeypatch.setenv("QLINEAR_FFN_CHUNK", "100000")
        ref = ff(x)
        for ch in ("1", "9", "60"):
            monkeypatch.setenv("QLINEAR_FFN_CHUNK", ch)
            got = ff(x)
            assert got.shape == ref.shape and got.dtype == ref.dtype
            assert torch.allclose(got.float(), ref.float(), atol=1e-2 if dtype == torch.bfloat16 else 1e-5)


class _FakeAttn(nn.Module):
    fused_projections = False

    def __init__(self, c):
        super().__init__()
        self.lin = nn.Linear(c, c, bias=False)
        self.processor = object()

    def forward(self, nh, rope, mask):
        return self.lin(nh.float().cumsum(1) / 10).to(nh.dtype)        # dipende dai token precedenti: sequenza intera


class _FakeBlock(nn.Module):
    def __init__(self, c=24):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(c), nn.RMSNorm(c)
        self.attn = _FakeAttn(c)
        self.ff = Q.FusedGateUpSwiGLU(nn.Linear(c, 2 * 40, bias=False), nn.Linear(40, c, bias=False))
        self.adaln_proj = lambda temb: temb.chunk(6, dim=-1)

    def forward(self, hs, temb, idx, rope, mask=None):                  # riferimento: come diffusers
        sm, cm, gm, sf, cf, gf = self.adaln_proj(temb)
        a = self.attn(self.norm1(hs) * (1.0 + cm.index_select(0, idx)) + sm.index_select(0, idx), rope, mask)
        hs = hs + gm.index_select(0, idx) * a
        f = self.ff(self.norm2(hs) * (1.0 + cf.index_select(0, idx)) + sf.index_select(0, idx))
        return hs + gf.index_select(0, idx) * f


def test_blocco_a_token_chunk_ramo_processore_originale(monkeypatch):
    """Anche con un processore non standard (Veda) o proiezioni non fuse il blocco resta identico."""
    import functools
    import h3_chunk as K
    monkeypatch.setenv("H3_BLOCK_CHUNK", "8")
    torch.manual_seed(2)
    blk = _FakeBlock()
    hs = torch.randn(1, 37, 24)
    temb = torch.randn(6, 144)                                     # 6 righe di tabella: 2 istanti x 3 modalita'
    idx = torch.randint(0, 6, (37,))
    keep = hs.clone()
    with torch.no_grad():
        ref = blk(hs, temb, idx, None)
        tr = type("T", (), {"transformer_blocks": [blk]})()
        assert K.enable_chunked_blocks(tr) == 1
        assert isinstance(blk.forward, functools.partial)
        got = blk(hs, temb, idx, None)
    assert torch.allclose(got, ref, atol=1e-5)
    assert torch.equal(hs, keep), "l'ingresso del blocco non va modificato"


def test_blocco_h3_chunk_disattivabile(monkeypatch):
    import h3_chunk as K
    monkeypatch.setenv("H3_BLOCK_CHUNK", "0")
    assert K.enable_chunked_blocks(type("T", (), {"transformer_blocks": [_FakeBlock()]})()) == 0


VENV_PY = r"E:\AI\Tools\diffusers_next_venv\Scripts\python.exe"


@pytest.mark.skipif(not os.path.exists(VENV_PY), reason="venv di H3 (diffusers col modello MiniMax) assente")
def test_blocco_a_token_chunk_contro_il_blocco_reale_di_diffusers():
    import subprocess
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([VENV_PY, "-I", os.path.join(ROOT, "tests", "h3_chunk_diffusers_check.py")],
                       capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stdout[-800:] + r.stderr[-800:]


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


def test_fit_after_load_attiva_chunk_e_offload_quando_il_margine_manca(monkeypatch):
    """Flusso del worker dopo il caricamento (CPU, VRAM finta): 13,1 GB liberi non coprono ~16,6 -> blocchi in RAM."""
    w = _worker()
    monkeypatch.setattr(O, "free_gb_now", lambda: 13.5)              # -0,4 di margine -> 13,1
    chiamate = []
    monkeypatch.setattr(O, "enable_block_offload", lambda blocks, n, dev, log=None: (chiamate.append(n), n)[1])
    monkeypatch.setattr(O, "blocks_for_deficit", lambda blocks, d, max_frac=0.85: (chiamate.append(("d", round(d, 1))), 12)[1])
    blocchi = [nn.Linear(2, 2) for _ in range(50)]
    tr = type("T", (), {"transformer_blocks": blocchi})()
    t = HP.sequence_tokens(243, 736, 1376)
    plan0 = HP.plan(29.9, t, weights_gb=11.7, lora_gb=2.0)
    with pytest.raises(RuntimeError):                                # free2 resta 13,5: la finta offload non libera nulla
        w._fit_after_load(tr, HP.DEFAULT_DIMS, plan0, {"height": 736, "width": 1376}, 243)
    assert chiamate[0][0] == "d" and chiamate[0][1] > 3.0 and chiamate[1] == 12
    assert all(hasattr(b, "forward") for b in blocchi)
    # con VRAM sufficiente: tutto residente, nessun offload
    chiamate.clear()
    monkeypatch.setattr(O, "free_gb_now", lambda: 30.0)
    w._fit_after_load(tr, HP.DEFAULT_DIMS, plan0, {"height": 736, "width": 1376}, 243)
    assert chiamate == []
