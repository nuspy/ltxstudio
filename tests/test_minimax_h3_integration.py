"""Integrazione MiniMax H3 (4ª architettura video) — offscreen, senza pesi né GPU.

Qui si verifica TUTTO ciò che non richiede i 34GB di pesi: catalogo e catene, dispatch verso il
venv/worker giusti, capability, presenza nel selettore, e che il worker dia un errore CHIARO
(con gli id di catalogo) quando i pesi mancano. La generazione vera resta da validare live.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from core import downloader as dl  # noqa: E402
from core import ltx_video_job as lvj  # noqa: E402
from core import model_paths as mp  # noqa: E402
from core import weights_catalog as wc  # noqa: E402
from core.model_catalog import video_caps  # noqa: E402


def test_catalogo_h3_completo():
    ids = {e["id"] for e in wc.load() if e.get("arch") == "minimax_h3"}
    for need in ("mmh3-configs", "mmh3-fl2va-nvfp4", "mmh3-ref2va-nvfp4", "mmh3-te-nvfp4",
                 "mmh3-video-vae", "mmh3-audio-vae", "mmh3-turbo-lora-fl2v"):
        assert need in ids, f"manca la voce {need}"
    for e in wc.load():
        if e.get("arch") == "minimax_h3":
            assert wc.downloadable(e), f"{e['id']} non scaricabile"
            assert e.get("used_by") == ["cap.video_minimax_h3"], f"{e['id']}: gating errato"


def test_catene_per_modalita():
    """Il checkpoint dei riferimenti e quello t2v tirano ENTRAMBI config+TE+VAE, deps-first."""
    for ckpt in ("mmh3-fl2va-nvfp4", "mmh3-ref2va-nvfp4"):
        plan = [x["id"] for x in dl.plan_ids([ckpt])]
        assert plan[-1] == ckpt, f"{ckpt} deve venire dopo le sue dipendenze"
        for dep in ("mmh3-configs", "mmh3-te-nvfp4", "mmh3-video-vae", "mmh3-audio-vae"):
            assert dep in plan, f"{ckpt}: manca {dep}"
    gb = sum(float(mp.entry(i).get("approx_gb", 0))
             for i in [x["id"] for x in dl.plan_ids(["mmh3-fl2va-nvfp4"])])
    assert 30 < gb < 40, f"scarico minimo inatteso: {gb:.1f} GB"


def test_configs_sono_solo_json():
    """La voce dei config NON deve tirarsi dietro i pesi bf16 (124GB) dal repo ufficiale."""
    e = mp.entry("mmh3-configs")
    assert e["kind"] == "snapshot" and e["repo_id"] == "MiniMaxAI/MiniMax-H3"
    assert e["allow_patterns"], "senza allow_patterns scaricherebbe l'intero repo"
    assert all(p.endswith(".json") for p in e["allow_patterns"]), e["allow_patterns"]
    assert float(e["approx_gb"]) < 0.1


def test_dispatch_arch():
    py, worker = lvj.runner_for({"arch": "minimax_h3"})
    assert worker.endswith("minimax_h3_infer.py")
    assert "minimax_h3_venv" in py
    assert os.path.exists(worker), "worker inesistente"


def test_capability():
    caps = video_caps("minimax_h3")
    assert caps["audio"] and caps["multiref"] and caps["t2v"] and caps["i2v"]
    # roba della pipeline LTX che H3 non ha: deve risultare spenta (la GUI disabilita i pannelli)
    assert not caps["controlnet"] and not caps["v2v"] and not caps["storyboard"]
    assert video_caps("ltx")["multiref"] is False


def test_selettore_arch(qtbot):
    """NB: cambiare architettura PERSISTE la preferenza dell'utente → va ripristinata,
    altrimenti il test lascia l'app impostata su H3 (e inquina i test successivi)."""
    from core import user_prefs
    from ui.tabs.video_tab import VideoTab
    prev = user_prefs.get("video_arch", "ltx")
    try:
        t = VideoTab(); qtbot.addWidget(t)
        assert t.video_arch.findData("minimax_h3") >= 0, "H3 assente dal selettore architettura"
        t.video_arch.setCurrentIndex(t.video_arch.findData("minimax_h3"))
        assert t.reference_panel.isEnabled(), "con H3 il pannello riferimenti deve essere attivo"
        assert t.reference_panel._lim == {"image": 9, "video": 3, "audio": 3}
        assert not t.storyboard_panel.isEnabled(), "H3 non ha lo storyboard a keyframe latenti"
    finally:
        user_prefs.set("video_arch", prev)


def test_worker_errore_chiaro_senza_pesi(tmp_path):
    """Senza pesi il worker deve dire QUALI id di catalogo mancano, non esplodere a metà."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mmh3w", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "workers", "minimax_h3_infer.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    with pytest.raises(RuntimeError) as ei:
        m.resolve_weights({"prompt": "x"})
    msg = str(ei.value)
    assert "mancanti" in msg.lower() and "mmh3-" in msg, msg
    # e il modo giusto di rimediare è indicato
    assert "Pesi modelli" in msg


def test_worker_sceglie_ref2va_con_i_riferimenti(monkeypatch):
    """Con i riferimenti serve il checkpoint SEPARATO ref2va: è una scelta di modalità, non di modello."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mmh3w2", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "workers", "minimax_h3_infer.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    seen = {}

    def fake_first(ids):
        seen.setdefault("ordini", []).append(ids)
        return "C:/finto/" + ids[0] + ".safetensors"

    monkeypatch.setattr(m, "_first_present", fake_first)
    monkeypatch.setattr(m.os.path, "isdir", lambda p: True)
    w = m.resolve_weights({"prompt": "x", "references": [{"path": "a.png", "kind": "image"}]})
    assert w["mode"] == "ref2va"
    assert "ref2va" in w["ckpt"], w["ckpt"]
    w2 = m.resolve_weights({"prompt": "x"})
    assert w2["mode"] == "fl2va" and "fl2va" in w2["ckpt"]
