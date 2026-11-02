"""workers/h3_chunk.alloc_qkv — allocazione atomica dei buffer Q/K/V: un OOM a meta non lascia la lista a pezzi (IndexError del 02/11)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))
import h3_chunk as K  # noqa: E402

# forme del log del 02/11: 22k token, 56 teste x 128, chunk di qualche migliaio di token (qui ridotte per la CPU, stessa struttura)
B, N, HEADS, D = 1, 2048, 4, 16


def _qkv_chunk(i, j):
    t = torch.arange(i, j, dtype=torch.float32).view(1, j - i, 1, 1).expand(1, j - i, HEADS, D).clone()
    return t, t + 0.5, t + 1.0


def test_alloc_is_atomic_when_the_second_allocation_fails(monkeypatch):
    real_empty, calls = torch.empty, {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:                                      # la 2a empty muore (come nel log: 316 MiB, VRAM al tetto)
            raise torch.OutOfMemoryError("finto")
        return real_empty(*a, **k)
    monkeypatch.setattr(K.torch, "empty", flaky)
    q, k, v = _qkv_chunk(0, 8)
    qkv: list = []
    with pytest.raises(torch.OutOfMemoryError):
        K.alloc_qkv(qkv, B, N, q, k, v, "cpu")
    assert qkv == [], "lista rimasta a pezzi: il riprovo salterebbe l'allocazione e darebbe IndexError"
    K.alloc_qkv(qkv, B, N, q, k, v, "cpu")                        # riprovo: ora alloca tutto
    assert len(qkv) == 3 and all(t.shape == (B, N, HEADS, D) for t in qkv)
    before = [id(t) for t in qkv]
    K.alloc_qkv(qkv, B, N, q, k, v, "cpu")                        # chiamate successive: nessuna riallocazione
    assert [id(t) for t in qkv] == before


def test_each_chunk_halving_recovers_and_fills_every_row(monkeypatch):
    """Lo stage_qkv vero in miniatura dentro each_chunk: OOM alla 2a allocazione nel primo blocco -> dimezzamento -> tutto riempito, nessun IndexError."""
    real_empty, state = torch.empty, {"n": 0}

    def flaky(*a, **k):
        state["n"] += 1
        if state["n"] == 2:
            raise torch.OutOfMemoryError("finto")
        return real_empty(*a, **k)
    monkeypatch.setattr(K.torch, "empty", flaky)
    qkv: list = []

    def stage_qkv(i, j):
        q, k, v = _qkv_chunk(i, j)
        K.alloc_qkv(qkv, B, N, q, k, v, "cpu")
        qkv[0][:, i:j] = q
        qkv[1][:, i:j] = k
        qkv[2][:, i:j] = v

    K.each_chunk(N, 512, stage_qkv, None)
    q, k, v = qkv
    want = torch.arange(N, dtype=torch.float32)[None]
    assert torch.equal(q[:, :, 0, 0], want) and torch.equal(k[:, :, 0, 0], want + 0.5) and torch.equal(v[:, :, 0, 0], want + 1.0)
    assert torch.equal(q[:, :, 1:, :], q[:, :, :1, :].expand_as(q[:, :, 1:, :]))           # tutte le teste riempite
