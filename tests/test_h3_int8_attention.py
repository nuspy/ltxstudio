"""Attenzione int8 di comfy_kitchen (casella della GUI, spenta di default): installazione, verifica sul primo risultato, scarti scritti nel log."""
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
sys.path.insert(0, str(ROOT))
import _qlinear as qlinear  # noqa: E402
import h3_int8_attention as attention  # noqa: E402

VENV_PY = r"E:\AI\Tools\diffusers_next_venv\Scripts\python.exe"


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    qlinear._LOGGED_ONCE.clear()
    monkeypatch.setattr(attention.INT8_ATTENTION, "reason", "")                 # «utilizzabile»: la parte CUDA e provata dal turno GPU
    monkeypatch.setattr(attention.INT8_ATTENTION, "verified", False)


def fake_kitchen(monkeypatch, kernel, available=True):
    module = types.SimpleNamespace(int8_attention=kernel, int8_attention_is_available=lambda device: available)
    monkeypatch.setitem(sys.modules, "comfy_kitchen", module)


def qkv():
    torch.manual_seed(1)
    return tuple(torch.randn(1, 128, 4, 64) for _ in range(3))                  # [batch, sequenza, teste, dimensione]


def sdpa_heads_first(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def test_good_kernel_is_verified_once_and_used(capsys, monkeypatch):
    calls = []

    def kernel(q, k, v, scale=None, attn_mask=None):
        calls.append(1)
        return sdpa_heads_first(q, k, v) * 1.001                               # quasi uguale: errore 0,1 %
    fake_kitchen(monkeypatch, kernel)
    q, k, v = qkv()
    out = attention.int8_attention_backend(q, k, v)
    assert len(calls) == 1 and attention.INT8_ATTENTION.verified and out.shape == q.shape
    assert "GEMM attenzione int8 attivo" in capsys.readouterr().out
    attention.int8_attention_backend(q, k, v)
    assert len(calls) == 2 and "attivo" not in capsys.readouterr().out         # verificato una volta sola


def test_wrong_kernel_is_discarded_with_a_log_line_and_the_reference_is_returned(capsys, monkeypatch):
    fake_kitchen(monkeypatch, lambda q, k, v, scale=None, attn_mask=None: torch.randn_like(q))
    q, k, v = qkv()
    out = attention.int8_attention_backend(q, k, v)
    assert torch.equal(out, attention.reference_attention(q, k, v))
    assert "non usato" in capsys.readouterr().out and attention.INT8_ATTENTION.reason.startswith("errore relativo")
    q2, k2, v2 = qkv()
    assert torch.equal(attention.int8_attention_backend(q2, k2, v2), attention.reference_attention(q2, k2, v2))      # da qui in poi sempre il riferimento


def test_kernel_error_and_missing_gpu_kernel_are_logged_not_swallowed(capsys, monkeypatch):
    def broken(q, k, v, scale=None, attn_mask=None):
        raise RuntimeError("nessun kernel per sm_120")
    fake_kitchen(monkeypatch, broken)
    q, k, v = qkv()
    assert torch.equal(attention.int8_attention_backend(q, k, v), attention.reference_attention(q, k, v))
    out = capsys.readouterr().out
    assert "nessun kernel per sm_120" in out and "int8_attention_backend" in out
    monkeypatch.setattr(attention.INT8_ATTENTION, "reason", "")
    fake_kitchen(monkeypatch, broken, available=False)
    attention.int8_attention_backend(q, k, v)
    assert "non disponibile per questa GPU" in capsys.readouterr().out


def test_cases_the_kernel_does_not_cover_use_the_reference_without_calling_it(monkeypatch):
    def kernel(*args, **kwargs):
        raise AssertionError("non va chiamato")
    fake_kitchen(monkeypatch, kernel)
    q, k, v = qkv()
    causal = attention.int8_attention_backend(q, k, v, is_causal=True)
    masked = attention.int8_attention_backend(q, k, v, attn_mask=torch.ones(128, 128, dtype=torch.bool))
    assert causal.shape == q.shape and masked.shape == q.shape


def test_flag_reaches_params_and_worker():
    from core import video_edit_h3 as vh3
    clip = Path(os.environ.get("TEMP", ".")) / "int8_attn_probe.mp4"
    clip.write_bytes(b"x")
    try:
        cfg = {"engine": "h3_edit", "mode": "change_background", "video": str(clip), "object": "person", "prompt": "a refuge", "h3_profile": "nvfp4"}
        assert "h3_int8_attention" not in vh3.build_params(cfg, "o.mp4", present=lambda item: True)               # default: spenta
        assert vh3.build_params({**cfg, "h3_int8_attention": True}, "o.mp4", present=lambda item: True)["h3_int8_attention"] is True
    finally:
        clip.unlink()
    worker = (ROOT / "workers" / "minimax_h3_infer.py").read_text(encoding="utf-8")
    assert 'if p.get("h3_int8_attention"):' in worker and "h3_int8_attention.install()" in worker


@pytest.mark.skipif(not os.path.isfile(VENV_PY), reason="serve il venv di H3")
def test_install_replaces_the_native_backend_and_keeps_cpu_results_identical():
    script = (
        "import sys, torch; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s')\n"
        "from diffusers.models import attention_dispatch as d\n"
        "import h3_int8_attention as a\n"
        "q, k, v = (torch.randn(1, 64, 2, 64) for _ in range(3))\n"
        "before = d.dispatch_attention_fn(q, k, v)\n"
        "a.install(); a.install()\n"
        "assert d._AttentionBackendRegistry._backends[d.AttentionBackendName.NATIVE] is a.int8_attention_backend\n"
        "after = d.dispatch_attention_fn(q, k, v)\n"
        "assert torch.allclose(before, after, atol=1e-6), (before - after).abs().max()\n"
        "print('INT8 ATTENTION INSTALL OK')\n" % (ROOT / "workers", ROOT))
    result = subprocess.run([VENV_PY, "-c", script], capture_output=True, text=True, timeout=300)
    assert "INT8 ATTENTION INSTALL OK" in result.stdout, result.stdout[-600:] + result.stderr[-900:]
