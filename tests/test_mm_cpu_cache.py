"""Unit test cache CPU-RAM del ModelManager (FASE B review) — oggetti dummy con .to() registrante."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_server.model_manager import ModelManager  # noqa: E402
from inference_server import component_store as cs  # noqa: E402


class _Mono:
    """Pipeline monolitica finta: registra i .to() e i remove_all_hooks."""
    def __init__(self):
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        return self

    def remove_all_hooks(self):
        self.moves.append("strip")


def test_unload_demote_and_restore(monkeypatch):
    monkeypatch.setattr(cs.STORE, "_ram_ok", lambda mb: True)
    mm = ModelManager()
    calls = []

    def loader():
        calls.append(1)
        return _Mono(), 100
    mm.register("zimage:x", loader)
    obj = mm.ensure("zimage:x")
    assert len(calls) == 1
    mm.unload("zimage:x")                                 # demote → RAM, non distrutto
    assert "zimage:x" in mm._cpu_cache
    assert "cpu" in obj.moves
    obj2 = mm.ensure("zimage:x")                          # restore da RAM: loader NON richiamato
    assert obj2 is obj
    assert len(calls) == 1
    assert "zimage:x" in mm._cache and "zimage:x" not in mm._cpu_cache


def test_unload_destroy_when_ram_low(monkeypatch):
    monkeypatch.setattr(cs.STORE, "_ram_ok", lambda mb: False)
    mm = ModelManager()
    mm.register("k", lambda: (_Mono(), 100))
    mm.ensure("k")
    mm.unload("k")                                        # RAM "scarsa" → destroy (comportamento storico)
    assert "k" not in mm._cpu_cache and "k" not in mm._cache


def test_clear_empties_cpu_cache_and_store(monkeypatch):
    monkeypatch.setattr(cs.STORE, "_ram_ok", lambda mb: True)
    import torch.nn as nn
    cs.STORE.acquire("comp", lambda: nn.Linear(2, 2))
    mm = ModelManager()
    mm.register("k", lambda: (_Mono(), 100))
    mm.ensure("k")
    mm.unload("k")                                        # → _cpu_cache
    mm.clear()                                            # /free: svuota TUTTO
    assert not mm._cpu_cache and not mm._cache
    assert "comp" not in cs.STORE._items


def test_shell_never_cpu_cached(monkeypatch):
    monkeypatch.setattr(cs.STORE, "_ram_ok", lambda mb: True)
    import torch.nn as nn
    store = cs.ComponentStore()
    monkeypatch.setattr(cs, "STORE", store)
    store.acquire("t5", lambda: nn.Linear(2, 2))

    class _Shell(_Mono):
        _ls_component_keys = ("t5",)
        _ls_shared = True
    mm = ModelManager()
    mm.register("flux:x", lambda: (_Shell(), 50))
    mm.ensure("flux:x")
    mm.unload("flux:x")
    assert "flux:x" not in mm._cpu_cache                  # shell distrutta (release, mai demote)
    assert store._items["t5"].refcount == 0
