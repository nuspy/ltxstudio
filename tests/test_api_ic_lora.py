"""POST /video/ic_lora: validazione, gate d'edizione, avvisi sul checkpoint e parita' con la GUI (senza GPU)."""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.app import app  # noqa: E402

client = TestClient(app)


@pytest.fixture()
def video(tmp_path):
    p = tmp_path / "in.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42")      # basta come path esistente: il job NON viene eseguito
    return str(p)


@pytest.fixture()
def no_run(monkeypatch):
    """Intercetta la sottomissione del job (niente GPU) e ignora la presenza reale dei pesi."""
    from api.handlers import video as h
    from core import video_edit_ic as vic
    captured = {}
    monkeypatch.setattr(h.MANAGER, "submit", lambda kind, fn: captured.update(kind=kind, fn=fn) or "job-test")
    monkeypatch.setattr(vic, "missing_weight", lambda spec: "")
    return captured


def test_invalid_kind_422(video):
    r = client.post("/video/ic_lora", json={"kind": "nope", "video_path": video})
    assert r.status_code == 422
    assert "refocus" in r.json()["detail"]["valid"]
    assert "lipdub" in r.json()["detail"]["valid"]
    # canny/union non sono modalita' di questo endpoint; hdr e' una fase futura (disabilitata)
    for k in ("canny", "union", "hdr", "motion-track"):
        assert client.post("/video/ic_lora", json={"kind": k, "video_path": video}).status_code == 422, k


def test_gate_off_403(video, monkeypatch):
    from api import gate
    monkeypatch.setattr(gate, "is_on", lambda fid: fid != "cap.video_restore_iclora")
    r = client.post("/video/ic_lora", json={"kind": "refocus", "video_path": video})
    assert r.status_code == 403
    assert r.json()["detail"]["capability"] == "cap.video_restore_iclora"
    # un'altra cap resta accessibile (non 403)
    r2 = client.post("/video/ic_lora", json={"kind": "outpaint", "video_path": video, "canvas": "bad"})
    assert r2.status_code == 422


def test_validation_per_kind(video):
    assert client.post("/video/ic_lora", json={"kind": "restyle", "video_path": video}).status_code == 422
    r = client.post("/video/ic_lora", json={"kind": "outpaint", "video_path": video, "canvas": "5:7"})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "invalid_canvas"
    r = client.post("/video/ic_lora", json={"kind": "outpaint", "video_path": video, "canvas": "custom"})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "margins_required"
    r = client.post("/video/ic_lora", json={"kind": "refocus", "video_path": "Z:/nope.mp4"})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "video_not_found"
    r = client.post("/video/ic_lora", json={"kind": "refocus", "video_path": video, "attention": 5})
    assert r.status_code == 422


def test_release25_rejected(video):
    r = client.post("/video/ic_lora", json={"kind": "refocus", "video_path": video, "ltx_release": "2.5"})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "release_unsupported"


def test_dev_checkpoint_warns_never_replaces(video, no_run):
    r = client.post("/video/ic_lora", json={"kind": "uncompress", "video_path": video,
                                            "model": "E:/m/ltx-2.3-22b-dev-fp8.safetensors"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == "job-test" and body["warnings"], body
    assert no_run["kind"] == "video.ic_lora"
    # checkpoint distilled: nessun avviso
    r = client.post("/video/ic_lora", json={"kind": "uncompress", "video_path": video,
                                            "model": "E:/m/ltx-2.3-22b-distilled-1.1_fp8.safetensors"})
    assert r.status_code == 200 and r.json()["warnings"] == []
    # nessun modello indicato: avviso (default del worker = dev), nessuna scelta automatica
    r = client.post("/video/ic_lora", json={"kind": "refocus", "video_path": video})
    assert r.status_code == 200 and r.json()["warnings"]


def test_weight_missing_409(video, monkeypatch):
    from core import video_edit_ic as vic
    monkeypatch.setattr(vic, "missing_weight", lambda spec: "Peso mancante X")
    r = client.post("/video/ic_lora", json={"kind": "refocus", "video_path": video})
    assert r.status_code == 409 and r.json()["detail"]["error"] == "weight_missing"


def test_known_caps_include_new():
    from api import gate
    for c in ("cap.video_restore_iclora", "cap.video_outpaint", "cap.video_restyle", "cap.video_camera_ic"):
        assert c in gate.KNOWN_CAPS


def test_outpaint_feather_and_min_bar_validation_and_passthrough(video, no_run, monkeypatch):
    base = {"kind": "outpaint", "video_path": video}
    assert client.post("/video/ic_lora", json={**base, "feather_px": -1}).status_code == 422
    assert client.post("/video/ic_lora", json={**base, "feather_px": 999}).status_code == 422
    assert client.post("/video/ic_lora", json={**base, "min_bar": 0.9}).status_code == 422
    r = client.post("/video/ic_lora", json={**base, "canvas": "4:3", "anchor": "bottom", "feather_px": 40,
                                            "min_bar": 0.15, "model": "E:/m/ltx-2.3-22b-distilled-1.1_fp8.safetensors"})
    assert r.status_code == 200, r.text
    from core import video_edit_ic as vic
    seen = {}

    def fake_build(cfg, out_dir, stamp, output, log=None):
        seen.update(cfg)
        raise RuntimeError("stop")
    monkeypatch.setattr(vic, "build_job", fake_build)
    with pytest.raises(RuntimeError, match="stop"):
        no_run["fn"](type("J", (), {"log": staticmethod(lambda m: None)})())
    assert seen["out_feather"] == 40 and seen["out_min_bar"] == 0.15
    assert seen["out_format"] == "4:3" and seen["out_anchor"] == "bottom" and seen["out_paste"] is True


def test_lipdub_validation_and_passthrough(video, no_run, monkeypatch):
    from core import video_ic_prep as prep
    monkeypatch.setattr(prep, "has_audio", lambda p: False)
    r = client.post("/video/ic_lora", json={"kind": "lipdub", "video_path": video, "line": "ciao"})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "video_no_audio"
    monkeypatch.setattr(prep, "has_audio", lambda p: True)
    r = client.post("/video/ic_lora", json={"kind": "lipdub", "video_path": video})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "line_required"
    r = client.post("/video/ic_lora", json={"kind": "lipdub", "video_path": video, "line": "Hola amigos",
                                            "language": "es", "scene": "a chef",
                                            "model": "E:/m/ltx-2.3-22b-distilled-1.1_fp8.safetensors"})
    assert r.status_code == 200 and r.json()["warnings"] == [], r.text
    from core import video_edit_ic as vic
    seen = {}

    def fake_build(cfg, out_dir, stamp, output, log=None):
        seen.update(cfg)
        raise RuntimeError("stop")
    monkeypatch.setattr(vic, "build_job", fake_build)
    with pytest.raises(RuntimeError, match="stop"):
        no_run["fn"](type("J", (), {"log": staticmethod(lambda m: None)})())
    assert seen["mode"] == "lipdub" and seen["lipdub_line"] == "Hola amigos" and seen["lipdub_lang"] == "es"
    assert seen["lipdub_scene"] == "a chef"


def test_lipdub_gate_off_403(video, monkeypatch):
    from api import gate
    monkeypatch.setattr(gate, "is_on", lambda fid: fid != "cap.video_lipdub")
    r = client.post("/video/ic_lora", json={"kind": "lipdub", "video_path": video, "line": "x"})
    assert r.status_code == 403 and r.json()["detail"]["capability"] == "cap.video_lipdub"


def test_lipdub_translate_and_language_warning(video, no_run, monkeypatch):
    from core import video_ic_prep as prep
    monkeypatch.setattr(prep, "has_audio", lambda p: True)
    base = {"kind": "lipdub", "video_path": video, "model": "E:/m/ltx-2.3-22b-distilled-1.1_fp8.safetensors"}
    r = client.post("/video/ic_lora", json={**base, "translate_to": "klingon"})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "invalid_translate_to" and "it" in r.json()["detail"]["valid"]
    r = client.post("/video/ic_lora", json={**base, "translate_to": "en"})            # senza line: ok, traduce nel job
    assert r.status_code == 200 and r.json()["warnings"] == [], r.text
    r = client.post("/video/ic_lora", json={**base, "line": "x", "language": "ar"})   # lingua senza prove: avviso, non blocco
    assert r.status_code == 200 and any("senza prove" in w for w in r.json()["warnings"])
    from core import video_edit_ic as vic
    from core import video_dub_translate as vdt
    seen = {}
    monkeypatch.setattr(vdt, "fill_cfg", lambda cfg, log=None: seen.update(called=dict(cfg)) or {})
    monkeypatch.setattr(vic, "build_job", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))
    client.post("/video/ic_lora", json={**base, "translate_to": "en", "source_text": "Ciao"})
    with pytest.raises(RuntimeError, match="stop"):
        no_run["fn"](type("J", (), {"log": staticmethod(lambda m: None)})())
    assert seen["called"]["translate_to"] == "en" and seen["called"]["source_text"] == "Ciao"


def test_lipdub_audio_api(video, no_run, monkeypatch, tmp_path):
    from core import video_ic_prep as prep
    aud = tmp_path / "a.wav"
    aud.write_bytes(b"RIFF")
    base = {"kind": "lipdub_audio", "video_path": video, "model": "E:/m/ltx-2.3-22b-distilled-1.1_fp8.safetensors"}
    monkeypatch.setattr(prep, "has_audio", lambda p: True)
    assert client.post("/video/ic_lora", json=base).json()["detail"]["error"] == "audio_not_found"
    r = client.post("/video/ic_lora", json={**base, "audio_path": str(aud), "audio_reference": "mix"})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "invalid_audio_reference"
    monkeypatch.setattr(prep, "has_audio", lambda p: p != str(aud))
    r = client.post("/video/ic_lora", json={**base, "audio_path": str(aud)})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "audio_no_stream"
    monkeypatch.setattr(prep, "has_audio", lambda p: p == str(aud))                   # video muto
    r = client.post("/video/ic_lora", json={**base, "audio_path": str(aud), "audio_reference": "original"})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "video_no_audio"
    r = client.post("/video/ic_lora", json={**base, "audio_path": str(aud)})
    assert r.status_code == 200 and any("SPERIMENTALE" in w for w in r.json()["warnings"]), r.text
    from api import gate
    monkeypatch.setattr(gate, "is_on", lambda fid: fid != "cap.video_lipdub")
    assert client.post("/video/ic_lora", json={**base, "audio_path": str(aud)}).status_code == 403
