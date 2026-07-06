"""Unit test ComponentStore (FASE A/B review) — CPU-only, moduli finti nn.Linear."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn  # noqa: E402

from inference_server.component_store import ComponentStore, _measure_mb  # noqa: E402


def _mk():
    return nn.Linear(4, 4)


def test_loader_called_once():
    s = ComponentStore()
    calls = []
    for _ in range(3):
        s.acquire("k", lambda: (calls.append(1), _mk())[1])
    assert len(calls) == 1
    assert s._items["k"].refcount == 3


def test_refcount_acquire_release_retain():
    s = ComponentStore()
    s.acquire("k", _mk)
    s.retain("k")
    assert s._items["k"].refcount == 2
    s.release("k"); s.release("k"); s.release("k")       # clamp a 0, mai negativo
    assert s._items["k"].refcount == 0
    assert "k" in s._items                                # release NON distrugge mai


def test_evict_only_orphans_lru(monkeypatch):
    s = ComponentStore()
    a = s.acquire("a", _mk); s._items["a"].device = "cuda"; s._items["a"].last_used = 1.0
    s.acquire("b", _mk); s._items["b"].device = "cuda"; s._items["b"].last_used = 2.0
    s.release("b")                                        # b orfano (refcount 0), a vivo (1)
    monkeypatch.setattr(s, "_ram_ok", lambda mb: False)   # RAM "scarsa" → destroy, non demote
    s.evict_unused(999999)
    assert "a" in s._items and "b" not in s._items        # mai toccare i referenziati
    assert a is not None


def test_evict_demote_to_cpu(monkeypatch):
    s = ComponentStore()
    s.acquire("t5", _mk); s._items["t5"].device = "cuda"
    s.release("t5")
    monkeypatch.setattr(s, "_ram_ok", lambda mb: True)    # RAM ok → demote
    e0 = s.placement_epoch
    s.evict_unused(1)
    assert s._items["t5"].device == "cpu"                 # demoted, non distrutto
    assert s.placement_epoch > e0                         # epoch bumpato → rebind delle pipe


def test_acquire_rollback_on_loader_error():
    s = ComponentStore()

    def boom():
        raise RuntimeError("x")
    try:
        s.acquire("bad", boom)
    except RuntimeError:
        pass
    assert "bad" not in s._items                          # niente entry fantasma


def test_resident_of_and_measure():
    s = ComponentStore()
    s.acquire("k", _mk)
    s._items["k"].device = "cuda"
    assert s.resident_of(("k",), "cuda") == s._items["k"].size_mb
    assert s.resident_of(("k",), "cpu") == 0
    assert _measure_mb(_mk()) > 0


def test_shrink_cpu_lru(monkeypatch):
    s = ComponentStore()
    monkeypatch.setenv("NATIVE_CPU_CACHE_MB", "0")        # budget zero → ogni cpu-cached è di troppo
    s.acquire("old", _mk); s.release("old")
    s._items["old"].device = "cpu"; s._items["old"].last_used = 1.0
    s._shrink_cpu_lru(1.0)
    assert "old" not in s._items
