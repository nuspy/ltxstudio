"""Task 21 — i metadati di generazione stanno nel file FINALE per ogni percorso di salvataggio dell'app:
sidecar GUI (MetadataSaver), native_driver (immagini/Film Studio/avatar/Krea2), video (ltx2_service, run_clip),
REST (job + ?embed_metadata=false), master Film Studio / videoclip. Nessuna GPU, nessun Qt: i motori sono finti
e scrivono un file con firma invisibile e watermark SIMULATI prima di restituire il controllo."""
import json
import os
import shutil
import subprocess
import time

import numpy as np
import pytest
from PIL import Image

from core import gen_metadata as G
from core import stego

_FF = shutil.which("ffmpeg")
needs_ff = pytest.mark.skipif(not _FF, reason="ffmpeg non disponibile")

PARAMS = {"prompt": "a glass pyramid on a salt flat", "negative": "blurry", "steps": 16, "guidance": 1.0, "seed": 7,
          "width": 320, "height": 320, "arch": "flux",
          "model": "E:\\AI\\Models\\diffusion_models\\Flux\\FLUX_Krea_Dev.safetensors",
          "loras": [{"name": "Krea2\\x.safetensors", "strength": 0.7}],
          "init_image": "C:\\Users\\atain\\Desktop\\in.png", "_t_button": 1.0}


def _engine_writes_final_image(path, rgba=False):
    """Simula image_flux.run: salva, firma (stego, riscrive il file), 'marchia' (altra riscrittura)."""
    y, x = np.mgrid[0:320, 0:320]
    arr = np.stack([x * 255 // 320, y * 255 // 320, (x + y) * 255 // 640], -1).astype(np.uint8)
    arr = np.clip(arr.astype(int) + np.random.default_rng(1).integers(-12, 12, arr.shape), 0, 255).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)
    stego.sign_file(path)
    im = Image.open(path).convert("RGB")                     # watermark visibile simulato: ricodifica PIL
    im.save(path)


def _assert_final(path):
    r = G.read(path)
    assert r is not None, "metadati assenti nel file finale"
    j = r[G.KEY_JSON]
    assert j["seed"] == 7 and j["steps"] == 16
    blob = open(path, "rb").read()
    assert b"C:\\Users" not in blob and b"atain" not in blob and b"E:\\AI" not in blob and b"_t_button" not in blob
    assert "Model: FLUX_Krea_Dev" in r[G.KEY_PARAMS] and "<lora:x:0.7>" in r[G.KEY_PARAMS]


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.delenv("LTX_EMBED_METADATA", raising=False)
    monkeypatch.setattr(G, "enabled", lambda: getattr(G._TL, "override", None) is not False)


def test_native_driver_funnel_after_stego_and_watermark(tmp_path, monkeypatch):
    from core import native_driver as nd
    out = str(tmp_path / "flux3_x.png")

    def fake_submit(module, params):
        return "pid1"

    def fake_history(pid):
        _engine_writes_final_image(out)
        return {"status": "success", "output": [out]}
    monkeypatch.setattr(nd, "submit", fake_submit)
    monkeypatch.setattr(nd, "history", fake_history)
    res = nd.run_and_wait("image", dict(PARAMS), poll=0.01, timeout=10)
    assert res["status"] == "success", res
    _assert_final(out)
    assert stego.detect_file(out)[0]                         # la firma è sopravvissuta (pixel intatti)


def test_native_driver_segment_not_embedded(tmp_path, monkeypatch):
    from core import native_driver as nd
    out = str(tmp_path / "mask.png")
    Image.new("L", (16, 16)).save(out)
    monkeypatch.setattr(nd, "submit", lambda m, p: "p")
    monkeypatch.setattr(nd, "history", lambda pid: {"status": "success", "output": [out]})
    nd.run_and_wait("segment", {"text": "cat"}, poll=0.01, timeout=5)
    assert G.read(out) is None


def test_native_driver_error_result_untouched(tmp_path, monkeypatch):
    from core import native_driver as nd
    monkeypatch.setattr(nd, "submit", lambda m, p: "p")
    monkeypatch.setattr(nd, "history", lambda pid: {"status": "error", "error": "boom"})
    assert nd.run_and_wait("image", dict(PARAMS), poll=0.01, timeout=5)["status"] == "error"


def test_sidecar_save_embeds_in_file(tmp_path, monkeypatch):
    """GUI: _auto_sidecar → MetadataSaver.save_sidecar. File finale (post stego+watermark) → metadati dentro."""
    from core import metadata_saver as ms
    monkeypatch.setattr(ms, "SIDECAR_DIR", tmp_path / "sidecars")
    out = str(tmp_path / "gui.png")
    _engine_writes_final_image(out)
    assert G.read(out) is None
    p = {k: v for k, v in PARAMS.items() if k != "_t_button"}
    ms.MetadataSaver.save_sidecar(out, {"label": "Immagine", "output_kind": "png", **p})
    _assert_final(out)
    assert (tmp_path / "sidecars" / "gui.png.json").exists()
    # il sidecar in locale resta com'era (percorsi inclusi: serve alla Galleria); i metadati nel file no
    assert "C:\\\\Users" in (tmp_path / "sidecars" / "gui.png.json").read_text(encoding="utf-8")


def test_sidecar_fallback_reads_embedded(tmp_path, monkeypatch):
    from core import metadata_saver as ms
    monkeypatch.setattr(ms, "SIDECAR_DIR", tmp_path / "sidecars")
    out = str(tmp_path / "copied.png")
    _engine_writes_final_image(out)
    G.embed(out, PARAMS, respect_pref=False)
    meta = ms.MetadataSaver.load_sidecar(out)
    assert meta and meta["seed"] == 7 and meta["prompt"].startswith("a glass pyramid")


def test_pref_off_no_metadata_in_gui_path(tmp_path, monkeypatch):
    from core import metadata_saver as ms
    monkeypatch.setattr(ms, "SIDECAR_DIR", tmp_path / "sidecars")
    monkeypatch.setattr(G, "enabled", lambda: False)
    out = str(tmp_path / "off.png")
    _engine_writes_final_image(out)
    ms.MetadataSaver.save_sidecar(out, {"label": "x", **PARAMS})
    assert G.read(out) is None
    raw = open(out, "rb").read()
    assert b"glass pyramid" not in raw


def _fake_video(path):
    cmd = [_FF, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=128x96:rate=24:duration=1",
           "-f", "lavfi", "-i", "sine=duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
           "-shortest", path]
    subprocess.run(cmd, check=True)
    stego.sign_file(path)                                    # firma + mux audio come ltx2_infer


@needs_ff
def test_ltx2_service_generate_embeds_final_video(tmp_path, monkeypatch):
    from core import ltx2_service as svc
    out = str(tmp_path / "ltx2_x.mp4")

    def fake_impl(params, timeout=3600):
        _fake_video(params["output"])
        return {"status": "success", "output": [params["output"]]}
    monkeypatch.setattr(svc, "_generate_impl", fake_impl)
    res = svc.generate(dict(PARAMS, output=out, num_frames=24, fps=24))
    assert res["status"] == "success", res
    _assert_final(out)
    r = subprocess.run([_FF, "-v", "error", "-i", out, "-map", "0:a:0", "-f", "null", "-"], capture_output=True)
    assert r.returncode == 0                                 # audio intatto


@needs_ff
def test_run_clip_embeds_final_video(tmp_path, monkeypatch):
    from core import ltx_video_job as lvj
    out = str(tmp_path / "clip.mp4")

    class R:
        stdout = "OK"
        stderr = ""
    monkeypatch.setattr(lvj, "free_video_vram", lambda **k: None)
    monkeypatch.setattr(lvj, "runner_for", lambda p: ("python", "worker.py"))

    def fake_run(cmd, **kw):
        _fake_video(out)
        return R()
    import types
    monkeypatch.setattr(lvj, "subprocess", types.SimpleNamespace(run=fake_run))
    got = lvj.run_clip(dict(PARAMS, output=out))
    assert got == out
    _assert_final(out)


# ── REST ─────────────────────────────────────────────────────────────────────────────────────────────────────


def _rest_client(monkeypatch, tmp_path, out):
    from fastapi.testclient import TestClient
    from core import native_driver as nd
    import api.app as appmod
    monkeypatch.setattr(nd, "is_up", lambda: True)
    monkeypatch.setattr(nd, "submit", lambda m, p: "pid")

    def fake_history(pid):
        _engine_writes_final_image(out)
        return {"status": "success", "output": [out]}
    monkeypatch.setattr(nd, "history", fake_history)
    monkeypatch.setattr(nd, "ensure_up", lambda **k: True)
    return TestClient(appmod.app)


def _wait_job(c, jid):
    for _ in range(300):
        j = c.get(f"/jobs/{jid}").json()
        if j["status"] in ("success", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError("job non terminato")


def test_rest_job_embeds_and_per_request_off(tmp_path, monkeypatch):
    out = str(tmp_path / "rest_a.png")
    c = _rest_client(monkeypatch, tmp_path, out)
    body = {"prompt": "a glass pyramid on a salt flat", "model": "flux1-dev.safetensors", "arch": "flux",
            "steps": 16, "guidance": 1.0, "seed": 7, "width": 320, "height": 320,
            "negative": "blurry"}
    r = c.post("/image/generate", json=body)
    assert r.status_code == 200, r.text
    j = _wait_job(c, r.json()["job_id"])
    assert j["status"] == "success", j
    got = G.read(out)
    assert got and got[G.KEY_JSON]["seed"] == 7 and got[G.KEY_JSON]["prompt"].startswith("a glass pyramid")
    assert "Seed: 7" in got[G.KEY_PARAMS]
    assert stego.detect_file(out)[0]

    # per-richiesta: ?embed_metadata=false → il file resta senza dati di generazione
    r = c.post("/image/generate?embed_metadata=false", json=body)
    j = _wait_job(c, r.json()["job_id"])
    assert j["status"] == "success", j
    assert G.read(out) is None
    assert b"glass pyramid" not in open(out, "rb").read()


def test_rest_final_safety_net_after_later_rewrite(tmp_path, monkeypatch):
    """Un passo successivo del job (es. stego sul file) riscrive l'output: la rete finale rimette i metadati."""
    from api.jobs import JobManager
    out = str(tmp_path / "rest_b.png")
    mgr = JobManager()

    def fn(job):
        _engine_writes_final_image(out)
        G.embed(out, PARAMS, respect_pref=False)             # passo 1 (come run_image)
        stego.sign_file(out)                                 # passo 2 riscrive e perde i metadati
        return [out]
    jid = mgr.submit("video.test", fn)
    for _ in range(200):
        if mgr.get(jid).status in ("success", "error"):
            break
        time.sleep(0.05)
    assert mgr.get(jid).status == "success"
    _assert_final(out)


# ── master Film Studio / videoclip ──────────────────────────────────────────────────────────────────────────


@needs_ff
def test_film_master_and_musicvideo_final_helpers(tmp_path):
    from core.film_studio import otio_export as oe
    from core import music_video_service as mv
    for fn, name in ((oe._embed_master_metadata, "m.mp4"), (mv._embed_final_metadata, "c.mp4")):
        p = str(tmp_path / name)
        _fake_video(p)
        fn(p)
        assert G.has_metadata(p)
