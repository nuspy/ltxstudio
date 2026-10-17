"""API immagini del task 9: /image/controlnet, /image/redux, /image/magicquill, inpaint con controlnet_inpaint.
Valida richieste e parametri verso il backend (nessun job eseguito: _submit/run_image sono finti)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.handlers import images  # noqa: E402

M = r"E:\AI\Models\diffusion_models"


@pytest.fixture()
def client(monkeypatch, tmp_path):
    sent = []
    monkeypatch.setattr(images.gate, "require", lambda *a, **k: None)
    monkeypatch.setattr(images, "_submit", lambda kind, fn: (sent.append((kind, fn)) or {"job_id": "x", "status": "pending"}))
    monkeypatch.setattr(images, "run_image", lambda p, **k: sent.append(("run", dict(p))) or [])
    import core.ensure_weights as ew
    monkeypatch.setattr(ew, "require", lambda ids, feature="": None)
    app = FastAPI()
    images.register(app)
    c = TestClient(app)
    c.sent = sent
    c.tmp = tmp_path
    return c


def _run(c):
    kind, fn = c.sent[-1]
    fn(None)
    return c.sent[-1][1]


def _png(tmp, name):
    from PIL import Image
    p = tmp / name
    Image.new("RGB", (64, 64), (10, 20, 30)).save(p)
    return str(p)


def _flux1():
    p = os.path.join(M, "Flux", "FLUX_Dev.safetensors")
    if not os.path.isfile(p):
        pytest.skip("FLUX_Dev assente")
    return r"Flux\FLUX_Dev.safetensors"


def test_controlnet_flux_params(client):
    g = _png(client.tmp, "g.png")
    r = client.post("/image/controlnet", json={"prompt": "una gondola-astronave", "model": _flux1(), "control_image": g,
                                              "control_preprocess": "pose", "seed": 5})
    assert r.status_code == 200, r.text
    p = _run(client)
    assert p["_module"] == "image_control"
    assert p["mode"] == "flux_cn" and p["arch"] == "flux" and p["control_preprocess"] == "pose" and p["seed"] == 5


def test_controlnet_rejects_unknown_kind_and_missing_file(client):
    g = _png(client.tmp, "g.png")
    assert client.post("/image/controlnet", json={"prompt": "x", "model": "m", "control_image": g,
                                                  "control_preprocess": "hed"}).status_code == 422
    assert client.post("/image/controlnet", json={"prompt": "x", "model": "m",
                                                  "control_image": str(client.tmp / "no.png")}).status_code == 422
    assert client.post("/image/controlnet", json={"prompt": "x", "arch": "krea2", "control_image": g}).status_code == 422


def test_controlnet_rejects_flux2_model(client):
    from core import arch_filter as af
    g = _png(client.tmp, "g.png")
    f2 = next((n for n in os.listdir(os.path.join(M, "Flux")) if af.is_flux2(r"Flux\\" + n)), None) if os.path.isdir(os.path.join(M, "Flux")) else None
    if not f2:
        pytest.skip("nessun FLUX.2 sul disco")
    r = client.post("/image/controlnet", json={"prompt": "x", "model": "Flux\\" + f2, "control_image": g})
    assert r.status_code == 422 and "FLUX.1" in r.text


def test_weights_missing_is_409(client, monkeypatch):
    import core.ensure_weights as ew

    def boom(ids, feature=""):
        raise ew.WeightsMissingError([{"id": i, "approx_gb": 4.3} for i in ids], feature)
    monkeypatch.setattr(ew, "require", boom)
    g = _png(client.tmp, "g.png")
    r = client.post("/image/controlnet", json={"prompt": "x", "model": _flux1(), "control_image": g})
    assert r.status_code == 409 and "flux-cn-union-pro2" in r.text


def test_redux_and_magicquill_validation(client):
    g = _png(client.tmp, "g.png")
    r = client.post("/image/redux", json={"model": _flux1(), "ref_images": [g], "prompt": "tramonto"})
    assert r.status_code == 200, r.text
    p = _run(client)
    assert p["mode"] == "flux_redux" and p["arch"] == "flux" and p["ref_images"] == [g]
    assert client.post("/image/redux", json={"model": _flux1(), "ref_images": [str(client.tmp / 'no.png')]}).status_code == 422
    # MagicQuill: serve un Kontext; un FLUX dev normale e' 422
    assert client.post("/image/magicquill", json={"mode": "local", "model": _flux1(), "image_path": g}).status_code == 422
    assert client.post("/image/magicquill", json={"mode": "boh", "model": "x", "image_path": g}).status_code == 422
    k = os.path.join(M, "Flux", "FLUX_Kontext_Dev.safetensors")
    if os.path.isfile(k):
        r = client.post("/image/magicquill", json={"mode": "removal", "model": r"Flux\FLUX_Kontext_Dev.safetensors",
                                                   "image_path": g, "mask_path": g})
        assert r.status_code == 200, r.text
        p = _run(client)
        assert p["arch"] == "kontext" and p["mode"] == "removal"


def test_inpaint_with_controlnet_only_on_flux1_base(client):
    g = _png(client.tmp, "g.png")
    fill = os.path.join(M, "Flux", "FLUX_DEV_Fill.safetensors")
    if os.path.isfile(fill):
        r = client.post("/image/inpaint", json={"prompt": "x", "image_path": g, "mask_path": g,
                                                "model": r"Flux\FLUX_DEV_Fill.safetensors", "controlnet_inpaint": True})
        assert r.status_code == 422 and "BASE" in r.text
    base = os.path.join(M, "Flux", "FLUX_Krea_Dev.safetensors")
    if os.path.isfile(base):
        r = client.post("/image/outpaint", json={"prompt": "x", "image_path": g, "mask_path": g,
                                                 "model": r"Flux\FLUX_Krea_Dev.safetensors", "controlnet_inpaint": True,
                                                 "control_scale": 0.8})
        assert r.status_code == 200, r.text
        p = _run(client)
        assert p["mode"] == "flux_cn_inpaint" and p["outpaint"] is True and abs(p["control_scale"] - 0.8) < 1e-9
    # senza il flag nulla cambia e control_scale non arriva al backend normale
    r = client.post("/image/inpaint", json={"prompt": "x", "image_path": g, "mask_path": g,
                                            "model": r"Flux\FLUX_DEV_Fill.safetensors"})
    assert r.status_code == 200
    p = _run(client)
    assert "control_scale" not in p and "mode" not in p
