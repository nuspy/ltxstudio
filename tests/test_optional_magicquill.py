"""MagicQuill V2: se un'edizione ESCLUDE i file (cap spenta) il motore immagini non si rompe: errore chiaro, nessun crash."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import optional_components as oc  # noqa: E402


def test_module_present_in_normal_location():
    assert (ROOT / "inference_server" / "backends" / "magicquill_v2.py").is_file()
    assert oc.magicquill_installed()


def test_missing_component_gives_clear_error_not_crash(monkeypatch):
    monkeypatch.setenv(oc.ENV_DISABLE, "magicquill")
    assert not oc.magicquill_installed() and oc.load_magicquill() is None
    res = oc.image_mq_handler()({"mode": "local"}, None, "")
    assert res["status"] == "error" and "non è inclusa" in res["error"]


def test_really_absent_file_is_handled(monkeypatch):
    """Il modulo non c'è DAVVERO (come in un'edizione senza il file): find_spec None → handler «non inclusa»."""
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == oc.MODULE else real(name, *a, **k))
    assert oc.load_magicquill() is None
    assert oc.image_mq_handler() is oc.missing_run


def test_job_runner_survives_without_component(tmp_path):
    """Un job image_mq nel job_runner VERO senza il componente: risultato d'errore leggibile, il runner non muore."""
    pj, rj = tmp_path / "p.json", tmp_path / "r.json"
    pj.write_text(json.dumps({"module": "image_mq", "params": {"mode": "local"}}), "utf-8")
    env = dict(os.environ, **{oc.ENV_DISABLE: "magicquill"})
    p = subprocess.run([sys.executable, "-m", "inference_server.job_runner", str(pj), str(rj)], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=300)
    assert rj.is_file(), p.stdout[-800:] + p.stderr[-800:]
    res = json.loads(rj.read_text("utf-8"))
    assert res["status"] == "error" and "non è inclusa" in res["error"], res
