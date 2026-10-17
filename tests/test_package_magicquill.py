"""scripts/package_magicquill_code.py: lo zip contiene codice + NOTICE + licenze e, estratto dove l'utente lo installa,
si carica con core/optional_components (stessa struttura cercata in E:\AI\Models\LoRA\MagicQuill\_code)."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from core import optional_components as oc  # noqa: E402
import package_magicquill_code as pk  # noqa: E402


def test_zip_contents_and_loadable_when_extracted(tmp_path, monkeypatch):
    pack, z = pk.build(tmp_path / "out")
    names = set(zipfile.ZipFile(z).namelist())
    for need in ("magicquill_v2.py", "mq2/__init__.py", "mq2/attention.py", "NOTICE.md", "LICENSE-upstream-CC-BY-NC-4.0.txt",
                 "LICENSE-EasyControl-Apache-2.0.txt", "INSTALL.txt", "VERSION"):
        assert need in names, need
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)
    notice = zipfile.ZipFile(z).read("NOTICE.md").decode("utf-8")
    assert "CC BY-NC 4.0" in notice and "Zichen Liu" in notice and "EasyControl" in notice and "Apache-2.0" in notice
    dst = tmp_path / "installed" / "_code"
    zipfile.ZipFile(z).extractall(dst)
    monkeypatch.setenv(oc.ENV, str(dst))
    monkeypatch.setattr(oc, "repo_dir", lambda: tmp_path / "nessun_repo")
    try:
        assert oc.pack_dir() == dst
        mod = oc.load_magicquill()
        if mod is None:
            pytest.skip(oc.last_error())                      # senza torch/diffusers nel python di test
        assert callable(mod.run) and Path(mod.__file__).parent == dst
    finally:
        for k in [k for k in sys.modules if k == oc.PKG or k.startswith(oc.PKG + ".")]:
            del sys.modules[k]


def test_catalog_entry_for_code_package():
    from core import weights_catalog as wc
    e = next(x for x in wc.load() if x["id"] == "mq2-code")
    assert e["license_ack"] and "CC BY-NC" in e["license"] and e.get("no_remote") and not wc.downloadable(e)
    assert wc.install_role(e) != "base"
