"""workers/h3_te_attn.py — attenzione del TE a blocchi di query: risultato identico a quella intera (CPU; serve transformers)."""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))
try:
    from transformers.integrations.sdpa_attention import sdpa_attention_forward as FULL
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except Exception:                                            # noqa: BLE001
    FULL = ALL_ATTENTION_FUNCTIONS = None
import h3_te_attn as TA  # noqa: E402

pytestmark = pytest.mark.skipif(FULL is None, reason="transformers con sdpa_attention non disponibile")


def _qkv(lq=70, lk=70, h=8, hk=4, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, h, lq, d, generator=g), torch.randn(1, hk, lk, d, generator=g), torch.randn(1, hk, lk, d, generator=g))


MOD = types.SimpleNamespace(num_key_value_groups=2, is_causal=True)


def _chunked(q, k, v, mask, causal=None):
    fn = TA._chunked_sdpa(FULL, max_bytes=2048)               # soglia minuscola: forza molti blocchi
    return fn(MOD, q, k, v, mask, dropout=0.0, scaling=None, is_causal=causal)[0]


def test_causal_without_mask_is_identical():
    q, k, v = _qkv()
    full = FULL(MOD, q, k, v, None, dropout=0.0, scaling=None, is_causal=None)[0]
    assert torch.allclose(_chunked(q, k, v, None), full, atol=1e-5)


def test_explicit_boolean_mask_with_bidirectional_block_is_identical():
    q, k, v = _qkv(60, 60)
    m = torch.tril(torch.ones(60, 60, dtype=torch.bool))
    m[10:30, 10:30] = True                                    # blocco bidirezionale (come i token di visione)
    m = m[None, None]
    full = FULL(MOD, q, k, v, m, dropout=0.0, scaling=None, is_causal=False)[0]
    assert torch.allclose(_chunked(q, k, v, m, causal=False), full, atol=1e-5)


def test_float_mask_is_identical():
    q, k, v = _qkv(50, 50)
    m = torch.zeros(1, 1, 50, 50)
    m[..., 25:, :5] = float("-inf")
    full = FULL(MOD, q, k, v, m, dropout=0.0, scaling=None, is_causal=False)[0]
    assert torch.allclose(_chunked(q, k, v, m, causal=False), full, atol=1e-5)


def test_below_threshold_calls_original_untouched():
    q, k, v = _qkv(8, 8)
    called = []
    orig = lambda *a, **k2: (called.append(1) or FULL(*a, **k2))      # noqa: E731
    fn = TA._chunked_sdpa(orig, max_bytes=10 ** 9)
    fn(MOD, q, k, v, None, dropout=0.0, scaling=None, is_causal=None)
    assert called == [1]


def test_context_manager_installs_and_restores():
    before = ALL_ATTENTION_FUNCTIONS["sdpa"]
    with TA.query_chunked_attention(2048) as ok:
        assert ok and ALL_ATTENTION_FUNCTIONS["sdpa"] is not before
        q, k, v = _qkv()
        out = ALL_ATTENTION_FUNCTIONS["sdpa"](MOD, q, k, v, None, dropout=0.0, scaling=None, is_causal=None)[0]
        assert torch.allclose(out, FULL(MOD, q, k, v, None, dropout=0.0, scaling=None, is_causal=None)[0], atol=1e-5)
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is before
