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
            f.truncate((size or 1_000_000) if (complete or n != S.SAM31_CHECKPOINT) else 1000)      # file sparsi: nessun I/O vero
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
    d = _make(tmp_path, complete=False)                                              # il modello e di 1000 byte: troncato
    st, bad = S.sam31_status(d)
    if S.SAM31_FILES[S.SAM31_CHECKPOINT]:                                            # con la misura esatta un file troncato non conta come installato
        assert st == "partial" and bad == [S.SAM31_CHECKPOINT]
    (d / S.SAM31_CHECKPOINT).unlink()
    (d / (S.SAM31_CHECKPOINT + ".part")).write_bytes(b"x")                           # download a pezzi
    assert S.sam31_status(d)[0] == "partial"
    (tmp_path / "b").mkdir()
    d2 = _make(tmp_path / "b")
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
    assert eng == "sam31" and "non e installato" in err and "Pesi modelli" in err and "SAM2" in err
    d = _make(tmp_path)
    monkeypatch.setattr(S, "LOADER_READY", False)
    assert "caricatore nativo" in S.resolve_for_job("sam31", d)[1]
    monkeypatch.setattr(S, "LOADER_READY", True)
    assert S.resolve_for_job("sam31", d) == ("sam31", "")


def test_access_texts_say_hours_not_days_and_point_to_the_alternatives():
    err = S.resolve_for_job("sam31", Path("Z:/non/esiste"))[1]
    assert "alcune ore" in err and "giorni" not in err and "Comfy-Org" in err and "SAM2" in err and "huggingface.co/facebook/sam3.1" in err
    assert "alcune ore" in S.describe_error("403 Client Error") and "token" in S.describe_error("401 Unauthorized")
    assert S.SAM31_REPO == "facebook/sam3.1" and S.SAM31_CATALOG_ID == "sam3.1-multiplex"


@pytest.mark.skipif(not (Path(r"E:\AI\Models\Segmentation\sam3.1") / S.SAM31_CHECKPOINT).exists(), reason="SAM 3.1 non presente")
def test_real_folder_is_complete():
    assert S.sam31_status(Path(r"E:\AI\Models\Segmentation\sam3.1")) == ("ok", [])


def test_backends_refuse_unusable_sam31_without_falling_back(tmp_path, monkeypatch):
    """image_segment.run e video_segment.track_object: SAM 3.1 scelto ma non utilizzabile = errore chiaro (mai SAM2 di nascosto)."""
    monkeypatch.setattr(S, "models_dir", lambda: tmp_path / "vuoto")
    monkeypatch.setattr(S, "comfy_dir", lambda: tmp_path / "vuoto_comfy")
    from inference_server.backends import image_segment, video_segment
    r = image_segment.run({"seg_engine": "sam31", "image_path": "x.png"}, None, str(tmp_path))
    assert r["status"] == "error" and "SAM 3.1" in r["error"] and "SAM2" in r["error"]
    with pytest.raises(RuntimeError, match="SAM 3.1"):
        video_segment.track_object("v.mp4", point=(1, 1), engine="sam31")


def test_video_segment_runs_as_a_script_without_the_repo_on_the_path(tmp_path):
    """Giro C 03/11: `python video_segment.py ...` (CLI del tracking) non trovava `core`: ModuleNotFoundError. Il file deve aggiungere da se la radice del repo."""
    import subprocess
    import sys
    seg = Path(__file__).resolve().parents[1] / "inference_server" / "backends" / "video_segment.py"
    code = ("import sys, runpy; sys.path = [p for p in sys.path if 'ltxstudio' not in p.lower()];"
            f"m = runpy.run_path(r'{seg}', run_name='x');"
            "\ntry:\n    m['track_object']('v.mp4', point=(1, 1), engine='sam2')\nexcept Exception as e:\n    print('OK:' + type(e).__name__)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path))
    assert "OK:" in r.stdout and "ModuleNotFoundError" not in r.stdout, r.stdout + r.stderr[-400:]
