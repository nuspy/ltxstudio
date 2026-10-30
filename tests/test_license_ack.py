"""Pesi con licenza non commerciale (MagicQuill V2): mai scaricati in silenzio."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from core import license_ack as la, weights_catalog as wc

ROOT = Path(__file__).resolve().parents[1]
MQ = ["mq2-local-lora", "mq2-removal-lora", "mq2-edge-lora", "mq2-color-lora", "mq2-puzzle-lora"]


def test_catalog_marks_magicquill_and_never_base():
    ents = {e["id"]: e for e in wc.load()}
    for i in MQ:
        e = ents[i]
        assert e["license_ack"] == "cc_by_nc_sa" and "CC BY-NC-SA" in e["license"]
        assert wc.install_role(e) != "base", "i pesi MagicQuill non devono entrare nell'installer"


def test_needing_ack_only_when_missing(monkeypatch):
    assert len(la.needing_ack(MQ)) == 0 or all(e["id"] in MQ for e in la.needing_ack(MQ))
    monkeypatch.setattr(wc, "is_present", lambda e, base=None: False)
    assert {e["id"] for e in la.needing_ack(["mq2-color-lora"])} == {"mq2-color-lora"}
    assert la.needing_ack(["flux-redux"]) == []              # licenza FLUX dev: nessuna conferma extra qui


def test_args_for():
    assert la.args_for([]) == [] and la.args_for(["a", "b"]) == ["--accept-license", "a,b"]


def test_script_refuses_without_accept(tmp_path):
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "download_weights.py"), "--ids", "mq2-local-lora",
                        "--base-dir", str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert p.returncode == 1, p.stdout + p.stderr
    assert "licenza" in p.stdout and "non confermata" in p.stdout and "↓" not in p.stdout
