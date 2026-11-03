"""core/segmentation_engine: scelta SAM2 / SAM 3.1, stato dei file (dimensioni esatte, .part), nessun ripiego silenzioso."""
from pathlib import Path

import pytest

from core import segmentation_engine as S


def _make(tmp_path, complete=True):
    d = tmp_path / "sam3.1"
    d.mkdir()
    for n, size in S.SAM31_FILES.items():
        p = d / n
        with open(p, "wb") as f:
            f.truncate(size if (complete or n != S.SAM31_CHECKPOINT) else 1000)      # file sparsi: nessun I/O vero
    return d


def test_default_is_sam2_and_choice_is_remembered(monkeypatch):
    store = {}
    from core import user_prefs
    monkeypatch.setattr(user_prefs, "get", lambda k, d=None: store.get(k, d))
    monkeypatch.setattr(user_prefs, "set", lambda k, v: store.__setitem__(k, v))
    assert S.current() == S.SAM2
    S.set_current(S.SAM31)
    assert S.current() == S.SAM31 and store[S.PREF_KEY] == "sam31"
    store[S.PREF_KEY] = "boh"
    assert S.current() == S.SAM2
    with pytest.raises(ValueError):
        S.set_current("boh")


def test_missing_partial_complete_states(tmp_path):
    assert S.sam31_status(tmp_path / "non_esiste")[0] == "missing"
    d = _make(tmp_path, complete=False)                                              # il .pt e di 1000 byte: download a pezzi
    st, bad = S.sam31_status(d)
    assert st == "partial" and bad == [S.SAM31_CHECKPOINT]
    (d / (S.SAM31_CHECKPOINT)).unlink()
    (d / (S.SAM31_CHECKPOINT + ".part")).write_bytes(b"x")
    assert S.sam31_status(d)[0] == "partial"
    d2 = _make(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path)
    assert S.sam31_status(d2)[0] == "ok"


def test_menu_marks_not_installed_and_blocks_selection(tmp_path):
    e = {x["id"]: x for x in S.menu_entries(tmp_path / "vuoto")}
    assert e["sam2"]["selectable"] and not e["sam31"]["selectable"] and e["sam31"]["state"] == "missing"
    assert "non installato" in e["sam31"]["label"]
    d = _make(tmp_path)
    e = {x["id"]: x for x in S.menu_entries(d)}
    assert e["sam31"]["installed"] and e["sam31"]["state"] == ("ok" if S.LOADER_READY else "loader")


def test_resolve_for_job_never_falls_back_silently(tmp_path, monkeypatch):
    assert S.resolve_for_job("sam2") == ("sam2", "")
    eng, err = S.resolve_for_job("sam31", tmp_path / "vuoto")
    assert eng == "sam31" and "non e installato" in err and "huggingface.co/facebook/sam3.1" in err and "SAM2" in err
    d = _make(tmp_path)
    monkeypatch.setattr(S, "LOADER_READY", False)
    assert "caricatore nativo" in S.resolve_for_job("sam31", d)[1]
    monkeypatch.setattr(S, "LOADER_READY", True)
    assert S.resolve_for_job("sam31", d) == ("sam31", "")


def test_403_means_pending_approval():
    assert "attesa di approvazione da Meta" in S.describe_error("403 Client Error: Forbidden for url facebook/sam3.1")
    assert "token" in S.describe_error("401 Unauthorized")


@pytest.mark.skipif(not (Path(r"E:\AI\Models\Segmentation\sam3.1") / S.SAM31_CHECKPOINT).exists(), reason="SAM 3.1 non presente")
def test_real_folder_is_complete():
    assert S.sam31_status(Path(r"E:\AI\Models\Segmentation\sam3.1")) == ("ok", [])


def test_backends_refuse_unusable_sam31_without_falling_back(tmp_path, monkeypatch):
    """image_segment.run e video_segment.track_object: SAM 3.1 scelto ma non utilizzabile = errore chiaro (mai SAM2 di nascosto)."""
    monkeypatch.setattr(S, "models_dir", lambda: tmp_path / "vuoto")
    from inference_server.backends import image_segment, video_segment
    r = image_segment.run({"seg_engine": "sam31", "image_path": "x.png"}, None, str(tmp_path))
    assert r["status"] == "error" and "SAM 3.1" in r["error"] and "SAM2" in r["error"]
    with pytest.raises(RuntimeError, match="SAM 3.1"):
        video_segment.track_object("v.mp4", point=(1, 1), engine="sam31")
