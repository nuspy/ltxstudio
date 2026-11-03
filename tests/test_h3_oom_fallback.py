"""H3: ripiego automatico dopo un OOM (stesso clip rifatto con lo streaming completo del DiT) e ciclo di riferimenti di each_chunk (giro 03/11).

Giro 03/11: 27,1 GB allocati all'OOM, 19 GB = 50 flussi residui [1, 37929, 5376] bf16 (uno per blocco del DiT) tenuti in vita dalla `go` ricorsiva di each_chunk
(funzione <-> cella della chiusura = ciclo di riferimenti: li libera solo il GC ciclico, che la memoria GPU non fa scattare). Qui: il ciclo non c'e piu, e se un clip
va comunque in OOM il worker lo rifa da capo con piu blocchi in RAM."""
import gc
import sys
import weakref
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h3_chunk as K  # noqa: E402

ER = pytest.importorskip("h3_edit_run")


class FakeW:
    def __init__(self):
        self.lines = []

    def _log(self, m):
        self.lines.append(m)


def test_each_chunk_leaves_no_reference_cycle_so_block_buffers_are_freed_at_once():
    class Buf:                                        # il flusso residuo / buffer di un blocco
        pass
    gc.disable()
    try:
        buf = Buf()
        ref = weakref.ref(buf)
        out = []
        K.each_chunk(100, 16, lambda i, j, b=buf: out.append((i, j)))
        del buf
        assert ref() is None, "each_chunk tiene in vita la chiusura (ciclo di riferimenti): niente gc ciclico = i buffer restano"
        assert out[0] == (0, 16) and out[-1] == (96, 100) and len(out) == 7
    finally:
        gc.enable()


def test_each_chunk_still_halves_on_oom_and_covers_every_row_once():
    covered = [0] * 64
    calls = {"n": 0}

    def fn(i, j):
        calls["n"] += 1
        if j - i > 8:
            raise torch.OutOfMemoryError("CUDA out of memory (finto)")
        for r in range(i, j):
            covered[r] += 1
    K.MIN_CHUNK = 4
    K.each_chunk(64, 32, fn)
    assert covered == [1] * 64 and calls["n"] > 2


def test_oom_in_a_clip_reruns_the_same_clip_with_full_streaming_and_logs_it():
    w, calls = FakeW(), []

    def runner(W, jd, stream_all=False):
        calls.append(stream_all)
        if not stream_all:
            raise RuntimeError("il clip non e stato generato (codice 1):\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 390.00 MiB")

    assert ER.run_clip_with_fallback(w, runner, "clip_00") is True
    assert calls == [False, True]                                           # stesso clip, secondo giro con streaming completo
    assert any("memoria insufficiente, riprovo con piu pezzi del modello in RAM (piu lento)" in l for l in w.lines)


def test_no_fallback_when_the_clip_succeeds_or_the_error_is_not_memory():
    w, calls = FakeW(), []
    assert ER.run_clip_with_fallback(w, lambda W, jd, **k: calls.append(k), "c") is False and calls == [{}]
    with pytest.raises(RuntimeError, match="ffmpeg"):
        ER.run_clip_with_fallback(w, lambda W, jd, **k: (_ for _ in ()).throw(RuntimeError("ffmpeg non trovato")), "c")
    assert not w.lines


def test_second_oom_after_the_fallback_is_a_real_error():
    w, calls = FakeW(), []

    def runner(W, jd, stream_all=False):
        calls.append(stream_all)
        raise RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        ER.run_clip_with_fallback(w, runner, "c")
    assert calls == [False, True]                                           # un solo ripiego, poi l'errore vero


def test_fallback_child_gets_streaming_and_small_chunks_through_the_environment(monkeypatch, tmp_path):
    import subprocess
    seen = {}

    class P:
        stdout = iter(["[h3] ok\n"])

        def wait(self):
            return 0
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: (seen.update(env=kw.get("env"), cmd=cmd), P())[1])
    W = type("W", (), {"__file__": str(tmp_path / "minimax_h3_infer.py")})
    ER._subprocess_clip_runner(W, str(tmp_path))
    assert seen["env"] is None                                              # primo tentativo: ambiente invariato
    ER._subprocess_clip_runner(W, str(tmp_path), stream_all=True)
    assert seen["env"]["H3_STREAM_ALL"] == "1" and int(seen["env"]["H3_BLOCK_CHUNK"]) <= 2048 and int(seen["env"]["H3_ATTN_QCHUNK"]) <= 2048
