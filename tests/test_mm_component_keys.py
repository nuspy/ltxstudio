"""Unit test ModelManager × ComponentStore (FASE A) — le shell rilasciano i componenti all'unload."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn  # noqa: E402

from inference_server.model_manager import ModelManager  # noqa: E402
from inference_server import component_store as cs  # noqa: E402


class _Shell:
    """Pipeline finta: shell che referenzia componenti nello store globale."""
    def __init__(self, keys):
        self._ls_component_keys = keys
        self._ls_shared = True


def test_unload_releases_components(monkeypatch):
    store = cs.ComponentStore()
    monkeypatch.setattr(cs, "STORE", store)               # store pulito per il test
    store.acquire("t5", lambda: nn.Linear(2, 2))          # refcount 1 (acquisito dalla shell)
    mm = ModelManager()
    mm.register("flux:x", lambda: (_Shell(("t5",)), 50))
    mm.ensure("flux:x")
    assert mm._cache["flux:x"].component_keys == ("t5",)
    assert store._items["t5"].refcount == 1
    mm.unload("flux:x")                                   # shell via, componente SOPRAVVIVE (refcount 0)
    assert "flux:x" not in mm._cache
    assert store._items["t5"].refcount == 0
    assert "t5" in store._items


def test_ensure_same_key_no_reload():
    mm = ModelManager()
    calls = []
    mm.register("k", lambda: (calls.append(1), object())[1] and (object(), 10) or (object(), 10))
    # loader semplice e deterministico
    mm._loaders["k"] = lambda: (calls.append(1), (object(), 10))[1]
    mm.ensure("k"); mm.ensure("k"); mm.ensure("k")
    assert len(calls) == 1                                # stesso modello → zero reload
