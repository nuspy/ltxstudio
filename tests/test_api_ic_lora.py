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
    assert "hdr" in r.json()["detail"]["valid"]
    # canny/union/motion-track non sono modalita' di questo endpoint
    for k in ("canny", "union", "motion-track"):
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
    for c in ("cap.video_restore_iclora", "cap.video_outpaint", "cap.video_restyle", "cap.video_camera_ic", "cap.video_motion_track"):
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
    assert seen["out_fill"] is True                                      # «garantisci riempimento»: default SI (17/10)
    assert seen["out_format"] == "4:3" and seen["out_anchor"] == "bottom" and seen["out_paste"] is True


def test_outpaint_fill_coverage_passthrough(video, no_run, monkeypatch):
    from core import video_edit_ic as vic
    seen = {}

    def fake_build(cfg, out_dir, stamp, output, log=None):
        seen.update(cfg)
        raise RuntimeError("stop")
    monkeypatch.setattr(vic, "build_job", fake_build)
    r = client.post("/video/ic_lora", json={"kind": "outpaint", "video_path": video, "canvas": "1:1", "fill_coverage": False,
                                            "model": "E:/m/ltx-2.3-22b-distilled-1.1_fp8.safetensors"})
    assert r.status_code == 200, r.text
    with pytest.raises(RuntimeError, match="stop"):
        no_run["fn"](type("J", (), {"log": staticmethod(lambda m: None)})())
    assert seen["out_fill"] is False


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


def test_hdr_validation_passthrough_and_outputs(video, no_run, monkeypatch, tmp_path):
    base = {"kind": "hdr", "video_path": video, "model": "E:/m/ltx-2.3-22b-distilled-1.1_fp8.safetensors"}
    r = client.post("/video/ic_lora", json={**base, "hdr_format": "dpx"})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "invalid_hdr_format" and "pq" in r.json()["detail"]["valid"]
    r = client.post("/video/ic_lora", json={**base, "hdr_exposure_ev": 9})
    assert r.status_code == 422                                   # fuori da -6..6
    r = client.post("/video/ic_lora", json={**base, "hdr_format": "both", "hdr_exposure_ev": 1.5, "hdr_exr_half": False,
                                            "hdr_hq": True})
    assert r.status_code == 200 and r.json()["warnings"] == [], r.text
    from api.handlers import video as h
    from core import video_edit_ic as vic
    seen = {}
    prev = tmp_path / "o.mp4"
    prev.write_bytes(b"x")
    pq = tmp_path / "o_HDR_PQ.mp4"
    pq.write_bytes(b"x")
    exr = tmp_path / "o_EXR"
    exr.mkdir()
    spec = vic.vem.get_spec("hdr")

    class _IJ:
        params = {"output": str(prev)}

        def outputs(self, result):
            return [result, str(pq), str(exr)]

    def fake_build(cfg, out_dir, stamp, output, log=None):
        seen.update(cfg)
        return _IJ()
    monkeypatch.setattr(vic, "build_job", fake_build)
    monkeypatch.setattr(h, "run_ltx2", lambda params, job: [str(prev)])
    monkeypatch.setattr(vic, "finalize", lambda ij, out, log=None: out)
    logs = []
    outs = no_run["fn"](type("J", (), {"log": staticmethod(logs.append)})())
    assert seen["mode"] == "hdr" and seen["hdr_format"] == "both" and seen["hdr_exposure_ev"] == 1.5
    assert seen["hdr_exr_half"] is False and seen["hdr_hq"] is True
    assert outs == [str(prev), str(pq)], "la cartella EXR non e' un file scaricabile: solo anteprima e HEVC"
    assert any(str(exr) in m for m in logs)
    assert spec.cap == "cap.video_hdr_ic" and spec.kind == "hdr"


def test_hdr_gate_off_403(video, monkeypatch):
    from api import gate
    monkeypatch.setattr(gate, "is_on", lambda fid: fid != "cap.video_hdr_ic")
    r = client.post("/video/ic_lora", json={"kind": "hdr", "video_path": video})
    assert r.status_code == 403 and r.json()["detail"]["capability"] == "cap.video_hdr_ic"


# ── motion_track ────────────────────────────────────────────────────────────────────────────────────
_CTRL = [[{"x": 100, "y": 100}, {"x": 300, "y": 180}, {"x": 520, "y": 120}], [{"x": 320, "y": 300}]]


@pytest.fixture()
def image(tmp_path):
    from PIL import Image
    p = tmp_path / "start.png"
    Image.new("RGB", (640, 360), (90, 120, 160)).save(p)
    return str(p)


def test_motion_track_is_a_valid_kind(video):
    r = client.post("/video/ic_lora", json={"kind": "nope", "video_path": video})
    assert "motion_track" in r.json()["detail"]["valid"]


def test_motion_track_validation(image, no_run):
    base = {"kind": "motion_track", "image_path": image, "tracks": _CTRL}
    ok = client.post("/video/ic_lora", json=base)                              # nessun video_path richiesto
    assert ok.status_code == 200, ok.text
    for body, err in (({**base, "image_path": None}, "image_required"),
                      ({**base, "image_path": "Z:/nope.png"}, "image_required"),
                      ({**base, "tracks": None}, "tracks_required"),
                      ({**base, "tracks": []}, "tracks_required"),
                      ({**base, "tracks": "[]"}, "invalid_tracks"),
                      ({**base, "tracks": "non json"}, "invalid_tracks"),
                      ({**base, "tracks_format": "boh"}, "invalid_tracks_format"),
                      ({**base, "tracks": None, "tracks_video_path": "Z:/nope.mp4"}, "tracks_video_not_found"),
                      ({**base, "fps": 2}, "invalid_fps"),
                      ({**base, "seconds": 12}, "too_long"),
                      ({**base, "num_frames": 400}, "too_long")):
        r = client.post("/video/ic_lora", json=body)
        assert r.status_code == 422 and r.json()["detail"]["error"] == err, (err, r.status_code, r.text)
    r = client.post("/video/ic_lora", json={**base, "seconds": 10, "fps": 24})
    assert r.status_code == 200, r.text                                         # 10 s a 24 fps = 241 frame, dentro il tetto


def test_motion_track_gate_and_weight(image, monkeypatch):
    from api import gate
    from core import video_edit_ic as vic
    monkeypatch.setattr(gate, "is_on", lambda fid: fid != "cap.video_motion_track")
    r = client.post("/video/ic_lora", json={"kind": "motion_track", "image_path": image, "tracks": _CTRL})
    assert r.status_code == 403 and r.json()["detail"]["capability"] == "cap.video_motion_track"
    monkeypatch.setattr(gate, "is_on", lambda fid: True)
    monkeypatch.setattr(vic, "missing_weight", lambda spec: "Peso mancante X")
    r = client.post("/video/ic_lora", json={"kind": "motion_track", "image_path": image, "tracks": _CTRL})
    assert r.status_code == 409 and r.json()["detail"]["error"] == "weight_missing"


def test_motion_track_cfg_passthrough(image, no_run, monkeypatch):
    from core import video_edit_ic as vic
    seen = {}

    def fake_build(cfg, out_dir, stamp, output, log=None):
        seen.update(cfg)
        raise RuntimeError("stop")
    monkeypatch.setattr(vic, "build_job", fake_build)
    r = client.post("/video/ic_lora", json={"kind": "motion_track", "image_path": image, "tracks": _CTRL, "prompt": "p",
                                            "seed": 9, "model": "E:/m/ltx-2.3-22b-distilled-1.1_fp8.safetensors"})
    assert r.status_code == 200 and r.json()["warnings"] == []                  # distilled: nessun avviso
    with pytest.raises(RuntimeError, match="stop"):
        no_run["fn"](type("J", (), {"log": staticmethod(lambda m: None)})())
    assert seen["mode"] == "motion_track" and seen["track_image"] == image and seen["tracks"] == _CTRL
    assert seen["fps"] == 24 and seen["image_strength"] == 1.0 and seen["seed"] == 9 and seen["prompt"] == "p"
    assert "width" not in seen and "height" not in seen, "senza valori espliciti: dall'aspetto dell'immagine"
    assert seen["model"].endswith("distilled-1.1_fp8.safetensors") and seen["keep_audio"] is False
    assert seen["native_audio"] is True
    # valori espliciti passano invariati
    seen.clear()
    r = client.post("/video/ic_lora", json={"kind": "motion_track", "image_path": image, "tracks": _CTRL, "width": 768,
                                            "height": 448, "image_strength": 0.8, "num_frames": 49, "fps": 25,
                                            "tracks_size": [768, 448]})
    assert r.status_code == 200
    with pytest.raises(RuntimeError, match="stop"):
        no_run["fn"](type("J", (), {"log": staticmethod(lambda m: None)})())
    assert (seen["width"], seen["height"], seen["image_strength"], seen["num_frames"], seen["fps"], seen["tracks_size"]) == \
           (768, 448, 0.8, 49, 25, [768, 448])


def test_motion_track_job_runs_through_build_and_returns_video_plus_tracks(image, no_run, monkeypatch, tmp_path):
    """Parita' con la GUI: stessa build_job; il worker e' finto (nessuna GPU). Output scaricabili: video + video delle tracce."""
    import os
    from api.handlers import video as h
    seen = {}

    def fake_run(p, job):
        seen.update(p)
        open(p["output"], "wb").write(b"x")
        return [p["output"]]
    monkeypatch.setattr(h, "run_ltx2", fake_run)
    monkeypatch.setattr(h, "out_path", lambda name, ext="mp4": str(tmp_path / f"{name}.{ext}"))
    r = client.post("/video/ic_lora", json={"kind": "motion_track", "image_path": image, "tracks": _CTRL, "seconds": 2,
                                            "width": 512, "height": 320})
    assert r.status_code == 200, r.text
    outs = no_run["fn"](type("J", (), {"log": staticmethod(lambda m: None)})())
    assert len(outs) == 2 and outs[0].endswith("video_ic_motion_track.mp4") and outs[1].endswith("_tracks.mp4")
    assert all(os.path.isfile(o) for o in outs)
    assert seen["control_type"] == "motion-track" and seen["num_frames"] == 49 and (seen["width"], seen["height"]) == (512, 320)
