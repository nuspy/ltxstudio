"""Pesi <-> matrice dell'edizione: menu (`menu_allowed` = regola di `wanted`) e dipendenze del downloader
(`plan_ids` / `resolve_dep` rispettano le righe `model.<id>`; «a|b» sceglie la consentita; nessuna = errore)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import downloader as dl  # noqa: E402
from core import features  # noqa: E402
from core import weights_catalog as wc  # noqa: E402


def _edition(monkeypatch, feats: dict | None):
    """Manifest finto: {fid: bool}. None = dev senza manifest (fail-open)."""
    if feats is None:
        man = {}
    else:
        man = {"features": {k: {"enabled": v, "params": {}} for k, v in feats.items()},
               "allows_adult": False}
    monkeypatch.setattr(features, "_MANIFEST", man)
    monkeypatch.setattr(features, "_FEATURES", man.get("features", {}))


def E(i, **kw):
    return {"id": i, "label": i, "module": "Video", "used_by": ["cap.video"], **kw}


def test_menu_allowed_cella_e_capability(monkeypatch):
    e = E("m1")
    _edition(monkeypatch, {"model.m1": True, "cap.video": True})
    assert wc.menu_allowed(e)
    _edition(monkeypatch, {"model.m1": False, "cap.video": True})
    assert not wc.menu_allowed(e) and not wc.model_allowed(e)
    # cella accesa MA funzione spenta: prima il menu lo mostrava (solo model_allowed), ora no (come wanted)
    _edition(monkeypatch, {"model.m1": True, "cap.video": False})
    assert wc.model_allowed(e) and not wc.menu_allowed(e)


def test_menu_allowed_dev_fail_open(monkeypatch):
    _edition(monkeypatch, None)
    assert wc.menu_allowed(E("m1"))


def test_menu_allowed_adult(monkeypatch):
    _edition(monkeypatch, {"model.m1": True, "cap.video": True})
    assert not wc.menu_allowed(E("m1", sensitivity="adult"))   # edizione senza allows_adult


BY = {"te_a": {"id": "te_a"}, "te_b": {"id": "te_b"}, "dit": {"id": "dit"}}


def test_resolve_dep_sceglie_la_consentita(monkeypatch):
    monkeypatch.setattr(wc, "is_present", lambda e, base=None: False)
    _edition(monkeypatch, {"model.te_a": False, "model.te_b": True})
    assert wc.resolve_dep("te_a|te_b", BY, strict=True, owner="dit") == "te_b"
    _edition(monkeypatch, {"model.te_a": True, "model.te_b": True})
    assert wc.resolve_dep("te_a|te_b", BY, strict=True) == "te_a"          # nessuna presente: la prima
    monkeypatch.setattr(wc, "is_present", lambda e, base=None: e["id"] == "te_b")
    assert wc.resolve_dep("te_a|te_b", BY, strict=True) == "te_b"          # la presente fra le consentite


def test_resolve_dep_nessuna_consentita(monkeypatch):
    _edition(monkeypatch, {"model.te_a": False, "model.te_b": False})
    with pytest.raises(wc.DepNotAllowedError) as ei:
        wc.resolve_dep("te_a|te_b", BY, strict=True, owner="dit")
    assert ei.value.owner == "dit" and ei.value.alternatives == ["te_a", "te_b"]
    assert "dit" in str(ei.value) and "te_a" in str(ei.value)
    assert wc.resolve_dep("te_a|te_b", BY) == "te_a"                       # non strict: comportamento storico


def test_plan_ids_rispetta_la_matrice(monkeypatch):
    cat = [{"id": "dit", "deps": ["te_a|te_b", "vae"]}, {"id": "te_a"}, {"id": "te_b"}, {"id": "vae"}]
    monkeypatch.setattr(wc, "load", lambda: cat)
    _edition(monkeypatch, {"model.te_a": False, "model.te_b": True, "model.dit": True})
    assert [e["id"] for e in dl.plan_ids(["dit"])] == ["te_b", "vae", "dit"]   # vae: nessuna riga = consentita
    _edition(monkeypatch, {"model.te_a": False, "model.te_b": False})
    with pytest.raises(wc.DepNotAllowedError):
        dl.plan_ids(["dit"])
    assert "dit" in [e["id"] for e in dl.plan_ids(["dit"], enforce_matrix=False)]
    _edition(monkeypatch, {"model.vae": False, "model.te_a": True})           # dipendenza semplice spenta
    with pytest.raises(wc.DepNotAllowedError):
        dl.plan_ids(["dit"])


def test_missing_ids_with_deps_strict(monkeypatch):
    cat = [{"id": "dit", "deps": ["te_a"]}, {"id": "te_a"}]
    monkeypatch.setattr(wc, "load", lambda: cat)
    monkeypatch.setattr(wc, "is_present", lambda e, base=None: False)
    _edition(monkeypatch, {"model.te_a": False})
    with pytest.raises(wc.DepNotAllowedError):
        wc.missing_ids_with_deps(cat[0], strict=True)
    assert wc.missing_ids_with_deps(cat[0]) == ["te_a", "dit"]               # menu/stato: non solleva
