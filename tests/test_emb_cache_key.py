"""Invarianti della cache embedding (core/emb_cache) — nessuna GPU, nessuna GUI.

Il test autoritativo resta quello dalla GUI (tests/gui_regression/live_emb_cache.py): qui si coprono
le parti che si possono verificare a freddo e che, se si rompono, producono il bug peggiore di questa
feature — servire il tensore di un altro modello senza che nessuno se ne accorga.
"""
from __future__ import annotations

import os
import sys
import zlib

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from core import emb_cache, st_direct                      # noqa: E402


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LS_EMB_CACHE_DIR", str(tmp_path / "emb"))
    monkeypatch.setenv("LS_EMB_CACHE", "1")
    yield


def _trio(seed: int = 0):
    """Una terna come quella dell'EmbeddingsProcessor: video bf16, audio bf16, mask int64."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 16, 32, generator=g).to(torch.bfloat16),
            torch.randn(1, 16, 8, generator=g).to(torch.bfloat16),
            torch.ones(1, 16, dtype=torch.int64))


# ─────────────────────────── serializzazione ───────────────────────────

def test_save_load_bit_esatto(tmp_path):
    """save_file/load_file devono essere bit-esatti su bf16 e int64 — i dtype che numpy non conosce
    e per cui l'implementazione passa da view(uint8)."""
    src = {"a": _trio(1)[0], "b": _trio(1)[2], "c": torch.randn(4, 4)}
    p = str(tmp_path / "t.safetensors")
    st_direct.save_file(p, src, {"hello": "mondo"})
    back = st_direct.load_file(p)
    assert set(back) == set(src)
    for k in src:
        assert back[k].dtype == src[k].dtype
        assert back[k].shape == src[k].shape
        assert torch.equal(back[k], src[k]), f"{k} non è bit-esatto"
    assert st_direct.metadata(p)["hello"] == "mondo"


def test_save_file_tensore_vuoto(tmp_path):
    p = str(tmp_path / "v.safetensors")
    st_direct.save_file(p, {"empty": torch.zeros(0, dtype=torch.bfloat16)}, None)
    assert st_direct.load_file(p)["empty"].numel() == 0


# ─────────────────────────── chiave ───────────────────────────

def test_chiave_stabile_e_sensibile():
    base = {"prompts": ["un gatto astronauta"], "te": [["g.safetensors", 10, 1, 2, 3]],
            "ep": [["ck.safetensors", 20, 4, 5, 6]], "dtype": "torch.bfloat16", "release": "2.3"}
    k = emb_cache.key_for(base)
    assert k == emb_cache.key_for(dict(base)), "stessa materia → stessa chiave"
    assert len(k) == 32
    for campo, nuovo in (("prompts", ["altro prompt"]),
                         ("te", [["g.safetensors", 11, 1, 2, 3]]),      # dimensione diversa
                         ("ep", [["ck.safetensors", 20, 4, 5, 99]]),    # crc di coda diverso
                         ("release", "2.5"),
                         ("dtype", "torch.float32")):
        mut = dict(base)
        mut[campo] = nuovo
        assert emb_cache.key_for(mut) != k, f"cambiare {campo} DEVE cambiare la chiave"


def test_ordine_prompt_conta():
    """La cache è per-chiamata: [pos, neg] e [neg, pos] sono batch diversi, chiavi diverse."""
    a = emb_cache.key_for({"prompts": ["A", "B"]})
    b = emb_cache.key_for({"prompts": ["B", "A"]})
    assert a != b


def test_impronta_file_cattura_la_sostituzione(tmp_path):
    """Un file sostituito con stessa dimensione ma contenuto diverso deve cambiare l'impronta:
    è il caso che renderebbe stantio un tensore cachato."""
    f = tmp_path / "w.safetensors"
    f.write_bytes(b"A" * 4096)
    fp1 = emb_cache.file_fp(str(f))
    f.write_bytes(b"B" * 4096)
    assert emb_cache.file_fp(str(f)) != fp1


# ─────────────────────────── get / put ───────────────────────────

def test_round_trip_get_put():
    out = [_trio(1), _trio(2)]
    assert emb_cache.put("k" * 32, out, ["pos", "neg"], {"encode_s": "13.4"})
    back = emb_cache.get("k" * 32)
    assert back is not None and len(back) == 2
    for orig, got in zip(out, back):
        for t0, t1 in zip(orig, got):
            assert torch.equal(t0, t1), "un hit DEVE restituire lo stesso identico tensore"


def test_audio_none_sopravvive():
    assert emb_cache.put("a" * 32, [(_trio()[0], None, _trio()[2])], ["p"], None)
    back = emb_cache.get("a" * 32)
    assert back is not None and back[0][1] is None


def test_miss_su_file_troncato():
    emb_cache.put("b" * 32, [_trio()], ["p"], None)
    p = emb_cache.path_for("b" * 32)
    with open(p, "r+b") as fh:
        fh.truncate(1000)
    assert emb_cache.get("b" * 32) is None, "un file troncato deve essere un MISS, non un errore"
    assert not p.exists(), "la voce illeggibile va rimossa"


def test_miss_su_crc_alterato():
    """Il caso subdolo: dimensione giusta, byte sbagliati. Senza il crc passerebbe come hit."""
    emb_cache.put("c" * 32, [_trio()], ["p"], None)
    p = emb_cache.path_for("c" * 32)
    _hdr, base = emb_cache._header(p)
    with open(p, "r+b") as fh:
        fh.seek(base)
        fh.write(b"\xff\xff")
    assert emb_cache.get("c" * 32) is None


def test_get_assente_non_esplode():
    assert emb_cache.get("d" * 32) is None


def test_stats_e_clear():
    emb_cache.put("e" * 32, [_trio()], ["p"], None)
    assert emb_cache.stats()["n"] == 1
    assert emb_cache.entries()[0]["prompt"] == "p"
    assert emb_cache.clear() == 1
    assert emb_cache.stats()["n"] == 0


def test_lru_rispetta_il_budget(monkeypatch):
    """Con budget minuscolo la GC deve sfrattare le voci meno usate, non tutte."""
    for i in range(4):
        emb_cache.put(f"{i}" * 32, [_trio(i)], ["p"], None)
    mb = emb_cache.stats()["mb"]
    monkeypatch.setenv("LS_EMB_CACHE_MB", str(max(0.001, mb * 0.6)))
    emb_cache.gc()
    assert 0 < emb_cache.stats()["n"] < 4


def test_disabilitata_da_env(monkeypatch):
    monkeypatch.setenv("LS_EMB_CACHE", "0")
    assert emb_cache.enabled() is False
