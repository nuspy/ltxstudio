"""Inpaint VIDEO con foto di riferimento (task 8, LoRA r2v): core condiviso, worker (controllo statico), API. Senza GPU."""
from __future__ import annotations

import ast
import os
import types
from pathlib import Path

import pytest

from core import video_inpaint as vi

ROOT = Path(__file__).resolve().parents[1]
WORKER = (ROOT / "workers" / "ltx2_infer.py").read_text(encoding="utf-8")


@pytest.fixture()
def photo(tmp_path):
    from PIL import Image
    p = tmp_path / "ref.png"
    Image.new("RGB", (640, 640), (10, 20, 30)).save(p)       # 1:1, fondo scuro uniforme
    return str(p)


# ── LoRA: scelta esplicita (foto sì/no), mai automatica ───────────────────────────────────────────
def test_senza_foto_resta_la_lora_t2v():
    cfg = {"prompt": "x", "edit_lora": vi.INPAINT_T2V_LORA}
    assert not vi.has_reference(cfg)
    assert vi.lora_for(cfg) == vi.INPAINT_T2V_LORA
    p = vi.inpaint_params(cfg, "out.mp4")
    assert "t2v" in p["edit_lora"] and "image_path" not in p and "r2v_mode" not in p


def test_con_foto_si_usa_la_lora_r2v(photo):
    cfg = {"prompt": "x", "edit_lora": vi.INPAINT_T2V_LORA, "ref_image": photo, "ref_strength": 0.65}
    p = vi.inpaint_params(cfg, "out.mp4")
    assert "r2v_rank32" in p["edit_lora"]
    assert p["image_path"] == photo and p["ref_strength"] == 0.65
    assert p["r2v_every_segment"] is True and p["r2v_mode"] == "frame0"


def test_checkpoint_come_scelto_mai_sostituito(photo):
    p = vi.inpaint_params({"model": "E:/m/ltx-2.3-22b-dev-fp8.safetensors", "ltx_release": "2.3"}, "o.mp4")
    assert p["model"].endswith("dev-fp8.safetensors") and p["ltx_release"] == "2.3"
    assert "model" not in vi.inpaint_params({}, "o.mp4")


def test_parametri_inpaint_propagati():
    p = vi.inpaint_params({"guide": 0.7, "cfg_scale": 8.5, "mask_dilate": 40, "mask_blur": 3.0, "seed": 7}, "o.mp4")
    assert (p["control_strength"], p["cfg_scale"], p["mask_dilate"], p["mask_blur"], p["seed"]) == (0.7, 8.5, 40, 3.0, 7)


def test_peso_mancante_e_foto_mancante(monkeypatch, tmp_path):
    monkeypatch.setattr(vi, "LORA_DIR", str(tmp_path))
    msg = vi.missing_weight({"ref_image": str(tmp_path / "nope.png")})
    assert "r2v" in msg and vi.R2V_CATALOG_ID in msg and "nope.png" in msg
    assert "mancante" in vi.missing_weight({})              # anche la t2v, se manca, e' un errore chiaro (non degrado)
    # peso presente + foto presente: nessun messaggio
    for rel in (vi.INPAINT_R2V_LORA, vi.INPAINT_T2V_LORA):
        f = tmp_path / rel.replace("/", os.sep)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
    ph = tmp_path / "ok.png"
    ph.write_bytes(b"x")
    assert vi.missing_weight({"ref_image": str(ph)}) == "" and vi.missing_weight({}) == ""


# ── foto adattata al quadro: oggetto intero (pad) invece del ritaglio centrale ───────────────────────
def test_fit_reference_pad_tiene_l_oggetto_intero(photo, tmp_path):
    from PIL import Image
    p = {"image_path": photo, "width": 1024, "height": 576}
    note = vi.fit_reference(p, str(tmp_path), 1)
    assert note and p["image_path"] != photo
    im = Image.open(p["image_path"])
    assert im.size == (1024, 576)
    assert im.getpixel((5, 5)) == (10, 20, 30)                # lati = colore mediano del bordo


def test_fit_reference_conserva_i_dettagli_non_taglia(tmp_path):
    from PIL import Image, ImageDraw
    src = tmp_path / "tall.png"
    im = Image.new("RGB", (400, 800), (0, 0, 0))
    ImageDraw.Draw(im).rectangle([0, 0, 399, 20], fill=(255, 0, 0))                     # banda in CIMA
    im.save(src)
    p = {"image_path": str(src), "width": 1024, "height": 576}
    vi.fit_reference(p, str(tmp_path), "t")
    out = Image.open(p["image_path"])
    assert out.getpixel((512, 3)) == (255, 0, 0), "la parte alta della foto deve restare nel quadro (con crop andrebbe persa)"


def test_fit_reference_nulla_da_fare(photo, tmp_path):
    p = {"image_path": photo, "width": 1024, "height": 1024}   # stesso aspetto
    assert vi.fit_reference(p, str(tmp_path), 1) == "" and p["image_path"] == photo
    p = {"image_path": photo, "width": 1024, "height": 576}
    assert vi.fit_reference(p, str(tmp_path), 1, mode="crop") == "" and p["image_path"] == photo
    assert vi.fit_reference({"width": 1, "height": 1}, str(tmp_path), 1) == ""   # senza foto (t2v)


def test_r2v_frame_da_env(photo, monkeypatch):
    assert vi.inpaint_params({"ref_image": photo}, "o.mp4")["r2v_frame"] == 0
    monkeypatch.setenv("LTX_R2V_FRAME", "-1")
    assert vi.inpaint_params({"ref_image": photo}, "o.mp4")["r2v_frame"] == -1
    assert "r2v_frame" not in vi.inpaint_params({}, "o.mp4")


# ── video lunghi: segmenti, la foto vale per ogni segmento ────────────────────────────────────────
def test_clip_corto_non_si_spezza_clip_lungo_si():
    assert "chunk_frames" not in vi.inpaint_params({"num_frames": 97}, "o.mp4")
    assert vi.segments_info(97) == 1
    p = vi.inpaint_params({"num_frames": 249}, "o.mp4")           # ~10 s a 25 fps
    assert p["chunk_frames"] == vi.SEGMENT_FRAMES and p["overlap_frames"] == vi.SEGMENT_OVERLAP
    assert vi.segments_info(249) == 3


def test_finestre_maschere_dei_segmenti_coprono_il_clip():
    from core import video_ic_prep as prep
    total = 249
    segs = prep.plan_segments(total, vi.SEGMENT_FRAMES, vi.SEGMENT_OVERLAP)
    assert len(segs) == vi.segments_info(total)
    assert segs[0].start == 0 and segs[-1].end == total
    for a, b in zip(segs, segs[1:]):
        assert a.end - b.start >= vi.SEGMENT_OVERLAP            # overlap sufficiente per il crossfade
    assert all((s.count - 1) % 8 == 0 for s in segs)


def _fn_source(name: str):
    tree = ast.parse(WORKER)
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_worker_ic_chunkable_accetta_inpaint_ma_non_attention_mask_o_retake():
    node = _fn_source("_ic_chunkable")
    ns: dict = {}
    exec(compile(ast.Module([node], []), "w", "exec"), ns)
    f = ns["_ic_chunkable"]
    assert f({"control_video": "v.mp4", "inpaint_masks_dir": "m"}) is True       # task 8
    assert f({"control_video": "v.mp4"}) is True
    assert f({"control_video": "v.mp4", "attention_mask_dir": "m"}) is False
    assert f({"control_video": "v.mp4", "retake": True}) is False
    assert f({"control_video": "v.mp4", "references": [1]}) is False
    assert f({"inpaint_masks_dir": "m"}) is False                                # senza guida non si segmenta


def test_worker_passa_foto_e_finestra_maschere():
    # ramo vinpaint: foto -> images=, maschere dal mask_offset; _run_chunked_ic: offset per segmento e foto su ogni segmento
    assert "images=imgs" in WORKER and 'p.get("mask_offset"' in WORKER and 'p.get("r2v_frame"' in WORKER
    assert 'cp["mask_offset"] = int(sg.start)' in WORKER
    assert 'not p.get("r2v_every_segment")' in WORKER
    assert "images=list(images or [])" in (ROOT / "workers" / "ltx_inpaint_pipeline.py").read_text(encoding="utf-8")


# ── tracking SAM2: agganci ai params ──────────────────────────────────────────────────────────────
def test_attach_tracking_aggancia_maschere_e_guida(tmp_path):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(stdout="GUIDE:x.mp4\n", stderr="")

    src = str(tmp_path / "clip.mp4")
    p = vi.attach_tracking({}, src, "umbrella", "g.mp4", 24, run=fake_run)
    assert p["control_video"] == src
    assert p["inpaint_masks_dir"].endswith(os.path.join("clip_track", "masks"))
    assert calls[0][-4:] == [src, "umbrella", "g.mp4", "24"]


def test_attach_tracking_errore_chiaro():
    fake = lambda cmd, **kw: types.SimpleNamespace(stdout="", stderr="boom")      # noqa: E731
    with pytest.raises(RuntimeError, match="tracking SAM2 fallito"):
        vi.attach_tracking({}, "c.mp4", "x", "g.mp4", 0, run=fake)


# ── API ───────────────────────────────────────────────────────────────────────────────────────────
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.app import app  # noqa: E402

client = TestClient(app)


@pytest.fixture()
def video(tmp_path):
    p = tmp_path / "in.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    return str(p)


@pytest.fixture()
def no_run(monkeypatch):
    from api.handlers import video as h
    captured = {}
    monkeypatch.setattr(h.MANAGER, "submit", lambda kind, fn: captured.update(kind=kind, fn=fn) or "job-test")
    monkeypatch.setattr(vi, "missing_weight", lambda cfg: "")
    return captured


def _body(video, **kw):
    return {"video_path": video, "object": "umbrella", "prompt": "a glowing jellyfish umbrella", **kw}


def test_api_validazione(video, photo):
    r = client.post("/video/inpaint", json=_body("Z:/nope.mp4"))
    assert r.status_code == 422 and r.json()["detail"]["error"] == "video_not_found"
    r = client.post("/video/inpaint", json=_body(video, reference_image="Z:/nope.png"))
    assert r.status_code == 422 and r.json()["detail"]["error"] == "reference_not_found"
    r = client.post("/video/inpaint", json=_body(video, ltx_release="2.5"))
    assert r.status_code == 422 and r.json()["detail"]["error"] == "release_unsupported"
    assert client.post("/video/inpaint", json={**_body(video), "object": " "}).status_code == 422
    assert client.post("/video/inpaint", json=_body(video, reference_strength=3)).status_code == 422


def test_api_gate_r2v_solo_con_foto(video, photo, no_run, monkeypatch):
    from api import gate
    monkeypatch.setattr(gate, "is_on", lambda fid: fid != "cap.video_inpaint_r2v")
    r = client.post("/video/inpaint", json=_body(video, reference_image=photo))
    assert r.status_code == 403 and r.json()["detail"]["capability"] == "cap.video_inpaint_r2v"
    assert client.post("/video/inpaint", json=_body(video)).status_code == 200           # senza foto: nessun gate r2v


def test_api_peso_mancante_409(video, photo, monkeypatch):
    monkeypatch.setattr(vi, "missing_weight", lambda cfg: "Peso r2v mancante")
    r = client.post("/video/inpaint", json=_body(video, reference_image=photo))
    assert r.status_code == 409 and r.json()["detail"]["error"] == "weight_missing"


def test_api_ok_avvisa_dev_mai_sostituisce(video, photo, no_run):
    r = client.post("/video/inpaint", json=_body(video, reference_image=photo,
                                                 model="E:/m/ltx-2.3-22b-dev-fp8.safetensors"))
    assert r.status_code == 200, r.text
    assert r.json()["job_id"] == "job-test" and r.json()["warnings"] and no_run["kind"] == "video.inpaint"
    r = client.post("/video/inpaint", json=_body(video, model="E:/m/ltx-2.3-22b-distilled-1.1_fp8.safetensors"))
    assert r.status_code == 200 and r.json()["warnings"] == []


def test_api_il_job_costruisce_i_params_come_la_gui(video, photo, no_run, monkeypatch, tmp_path):
    """Esegue `fn` del job con tracking e worker finti: i params che arrivano a run_ltx2 sono quelli della GUI."""
    from api.handlers import video as h
    from core import video_ic_prep as prep
    seen = {}
    monkeypatch.setattr(prep, "probe", lambda p, count_frames=False: types.SimpleNamespace(
        width=1280, height=704, fps=25.0, nb_frames=250))
    monkeypatch.setattr(vi, "attach_tracking", lambda params, src, obj, g, d, log=None: params.update(
        control_video=src, inpaint_masks_dir="M") or params)
    monkeypatch.setattr(h, "run_ltx2", lambda params, job=None: seen.update(params) or ["out.mp4"])
    monkeypatch.setattr(h, "out_path", lambda prefix, ext="mp4": str(tmp_path / f"{prefix}.{ext}"))
    client.post("/video/inpaint", json=_body(video, reference_image=photo, reference_strength=0.7, seed=5))
    outs = no_run["fn"](types.SimpleNamespace(log=lambda *a, **k: None))
    assert outs == ["out.mp4"]
    assert "r2v_rank32" in seen["edit_lora"] and seen["ref_strength"] == 0.7
    assert seen["image_path"] != photo and "_r2vref_" in seen["image_path"] and os.path.isfile(seen["image_path"])   # foto 1:1 su 16:9: bordi
    assert (seen["width"], seen["height"], seen["fps"]) == (1280, 704, 25)
    assert seen["num_frames"] == 249 and seen["chunk_frames"] == vi.SEGMENT_FRAMES and seen["seed"] == 5
    assert seen["inpaint_masks_dir"] == "M" and seen["control_video"] == video


def test_cap_nota_al_gate():
    from api import gate
    assert "cap.video_inpaint_r2v" in gate.KNOWN_CAPS
