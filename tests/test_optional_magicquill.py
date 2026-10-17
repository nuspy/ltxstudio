"""MagicQuill V2 = componente OPZIONALE: se manca (build/edizione) nessun crash, errore chiaro «non installato»;
il codice derivato (CC BY-NC) non entra MAI nelle build."""
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


def test_pack_found_in_repo_and_override(monkeypatch, tmp_path):
    assert oc.pack_dir() == ROOT / "optional" / "magicquill"
    monkeypatch.setenv(oc.ENV_DISABLE, "magicquill")
    assert oc.pack_dir() is None and not oc.magicquill_installed()
    monkeypatch.delenv(oc.ENV_DISABLE)
    # cartella installata dall'utente: stessa struttura, trovata quando il repo non ce l'ha
    (tmp_path / "mq2").mkdir()
    (tmp_path / "magicquill_v2.py").write_text("def run(p, mm, o):\n    return {'status': 'success', 'output': []}\n")
    (tmp_path / "mq2" / "__init__.py").write_text("")
    monkeypatch.setattr(oc, "repo_dir", lambda: tmp_path / "assente")
    monkeypatch.setenv(oc.ENV, str(tmp_path))
    assert oc.pack_dir() == tmp_path
    assert oc.load_magicquill().run({}, None, "") == {"status": "success", "output": []}


def test_missing_component_gives_clear_error_not_crash(monkeypatch):
    monkeypatch.setenv(oc.ENV_DISABLE, "magicquill")
    assert oc.load_magicquill() is None
    res = oc.image_mq_handler()({"mode": "local"}, None, "")
    assert res["status"] == "error" and "non installato" in res["error"] and "mq2-code" in res["error"]
    with pytest.raises(ModuleNotFoundError):
        oc.require_magicquill()


def test_job_runner_survives_without_component(tmp_path):
    """Un job image_mq nel job_runner VERO senza il componente: risultato d'errore leggibile, il runner non muore."""
    pj, rj = tmp_path / "p.json", tmp_path / "r.json"
    pj.write_text(json.dumps({"module": "image_mq", "params": {"mode": "local"}}), "utf-8")
    env = dict(os.environ, **{oc.ENV_DISABLE: "magicquill"})
    p = subprocess.run([sys.executable, "-m", "inference_server.job_runner", str(pj), str(rj)], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=300)
    assert rj.is_file(), p.stdout[-800:] + p.stderr[-800:]
    res = json.loads(rj.read_text("utf-8"))
    assert res["status"] == "error" and "non installato" in res["error"], res
