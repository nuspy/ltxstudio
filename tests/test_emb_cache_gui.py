"""Widget della cache embedding — costruzione offscreen, nessuna GPU, nessun torch.

Copre la parte che il test live (tests/gui_regression/live_emb_cache.py) non può verificare senza
occupare la GPU: che i controlli esistano davvero nella finestra reale, che scrivano la preferenza
letta dal worker, e che il pulsante di svuotamento funzioni. Verifica anche che importare i widget
NON tiri dentro torch nel processo GUI, che è la regola [[app-no-cuda-context]].
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication                # noqa: E402

from core import emb_cache, user_prefs                    # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LS_EMB_CACHE_DIR", str(tmp_path / "emb"))
    yield


class _FakeEngine:
    def loader_nodes(self):
        return []


class _FakeDriver:
    def is_up(self):
        return False

    def models_in(self, _folder):
        return []


def test_emb_cache_import_non_carica_torch():
    """core.emb_cache è importato dal processo GUI: se trascinasse torch, la GUI creerebbe un
    contesto CUDA e i subprocess pesanti crasherebbero."""
    import importlib
    import subprocess
    r = subprocess.run([sys.executable, "-c",
                        "import sys; from core import emb_cache; "
                        "print('torch' in sys.modules)"],
                       capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert r.stdout.strip() == "False", f"emb_cache ha importato torch: {r.stdout} {r.stderr}"
    importlib.reload(emb_cache)


def test_settings_dialog_ha_i_controlli(app, monkeypatch):
    from ui.settings_dialog import SettingsDialog
    monkeypatch.setattr(user_prefs, "get", lambda k, d=None: {"emb_cache.enabled": True,
                                                              "emb_cache.mb": 8192}.get(k, d))
    d = SettingsDialog(_FakeEngine(), _FakeDriver())
    assert d.cb_embcache.isChecked() is True
    assert d.sp_embcache.value() == 8              # 8192 MB → 8 GB
    assert d.cb_embcache.toolTip(), "il tooltip spiega cosa fa: non deve essere vuoto"
    d.deleteLater()


def test_settings_dialog_persiste(app, monkeypatch):
    from ui.settings_dialog import SettingsDialog
    scritti: dict = {}
    monkeypatch.setattr(user_prefs, "set", lambda k, v: scritti.__setitem__(k, v))
    monkeypatch.setattr(user_prefs, "get", lambda k, d=None: d)
    d = SettingsDialog(_FakeEngine(), _FakeDriver())
    d.cb_embcache.setChecked(False)
    d.sp_embcache.setValue(2)
    d._on_accept()
    assert scritti["emb_cache.enabled"] is False
    assert scritti["emb_cache.mb"] == 2048
    d.deleteLater()


def test_memory_monitor_mostra_e_svuota(app):
    from ui.memory_monitor import MemoryMonitor
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    emb_cache.put("f" * 32, [(torch.randn(1, 8, 16, generator=g).to(torch.bfloat16), None,
                              torch.ones(1, 8, dtype=torch.int64))], ["prova"], None)
    m = MemoryMonitor()
    m._refresh_emb_cache()
    assert "1" in m._emb_lbl.text(), m._emb_lbl.text()
    assert m._emb_btn.isEnabled()
    m._clear_emb_cache()
    assert emb_cache.stats()["n"] == 0
    assert not m._emb_btn.isEnabled(), "a cache vuota il pulsante non ha nulla da fare"
    m.deleteLater()


def test_params_del_job_portano_la_preferenza(monkeypatch):
    """La preferenza deve viaggiare nei params: il server persistente non erediterebbe una variabile
    d'ambiente impostata dopo il suo avvio."""
    from core import ltx_video_job
    monkeypatch.setattr(user_prefs, "get", lambda k, d=None: False if k == "emb_cache.enabled" else d)
    p = ltx_video_job.build_params(output="o.mp4", prompt="x", width=512, height=320, num_frames=25)
    assert p["emb_cache"] is False
    monkeypatch.setattr(user_prefs, "get", lambda k, d=None: True if k == "emb_cache.enabled" else d)
    p = ltx_video_job.build_params(output="o.mp4", prompt="x", width=512, height=320, num_frames=25)
    assert p["emb_cache"] is True
