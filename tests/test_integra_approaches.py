"""Integra, approcci alternativi (prove del 03/11): `MainWindow._start_integrate_approach` costruisce i parametri giusti per ogni motore, e senza approccio
(canvas.integrate_approach == "") non fa nulla. Senza app e senza GPU: la finestra e sostituita da uno stub che registra le chiamate.
Piu: il worker Qwen rispetta `mask_ref`, il backend FLUX ha le tre chiavi opt-in, il worker LTX applica la LoRA MSR 2.5."""
from __future__ import annotations

import os
import re
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _img(p, size, color):
    Image.new("RGB", size, color).save(p)
    return str(p)


def _exp(tmp_path):
    pm = Image.new("L", (400, 300), 0)
    pm.paste(255, (260, 60, 360, 280))                            # soggetto a destra
    pm.save(tmp_path / "pos.png")
    return {"image": _img(tmp_path / "comp.png", (400, 300), (90, 80, 70)), "mask": _img(tmp_path / "ctx.png", (400, 300), (255, 255, 255)),
            "position_mask": str(tmp_path / "pos.png"), "base": _img(tmp_path / "base.png", (400, 300), (60, 70, 80)),
            "layer_refs": [_img(tmp_path / "layer1.png", (200, 300), (128, 128, 128))], "has_region": True}


@pytest.fixture()
def stub(monkeypatch):
    mw = pytest.importorskip("ui.main_window")
    calls = []

    class W:
        _QWEN21_INTEGRATE_PROMPT = "blend"

        def __init__(self, approach="", strength=None):
            self.frame1_tab = SimpleNamespace(canvas=SimpleNamespace(integrate_approach=approach, integrate_strength=strength))

        def _start_native_qwen21_edit(self, dit, prompt, **kw):
            calls.append(("qwen", dit, prompt, kw))

        def _start_native_kontext(self, exp, model, prompt, negative, steps, seed, guidance, pulid=None, target_wh=None):
            calls.append(("kontext", exp, model, prompt, dict(steps=steps, guidance=guidance, wh=target_wh)))

        def _log(self, *_a):
            pass

        def _start_native_preview(self, *_a):
            pass

        def _enqueue(self, label, worker, cb):
            calls.append(("enqueue", worker.params))

        _on_native_inpaint_done = None

    class FakeWorker:
        def __init__(self, params):
            self.params = params
    monkeypatch.setattr(mw, "_NativeImageWorker", FakeWorker)
    monkeypatch.setattr(mw, "OUTPUT_DIR", mw.Path(os.environ.get("TEMP", ".")))
    return mw, W, calls


def test_qwen_armonizza_senza_riferimento(stub, tmp_path):
    mw, W, calls = stub
    w = W("armonizza", 0.45)
    ok = mw.MainWindow._start_integrate_approach(w, "armonizza", "qwen_image21", _exp(tmp_path), {"integrate_qwen21_dit": "qwen21-turbo-dit-nvfp4"}, "put her at the table",
                                                 None, "neg", 5, 8, 1.0)
    assert ok
    kind, dit, prompt, kw = calls[0]
    assert kind == "qwen" and dit == "qwen21-turbo-dit-nvfp4" and prompt == "put her at the table"
    assert kw["strength"] == 0.45 and kw["refs"] == [] and kw["extra"] == {"mask_ref": False}
    assert kw["init_image"].endswith("comp.png") and kw["mask"].endswith("ctx.png")           # composito come punto di partenza, non come riferimento


def test_klein_riferimento_e_armonizza(stub, tmp_path):
    mw, W, calls = stub
    w = W()
    mw.MainWindow._start_integrate_approach(w, "klein_ref_inpaint", "flux2", _exp(tmp_path), {}, "match the light", "Flux\\realDream_klein9BV2.safetensors", "", 7, 12, 4.0)
    p = calls[-1][1]
    assert p["arch"] == "flux2_inpaint" and p["strength"] == 1.0 and p["image_path"].endswith("base.png")          # parte dalla SCENA senza la foto
    assert p["ref_images"] and p["ref_images"][0].endswith("layer1.png") and "on the right side" in p["prompt"]
    assert p["model"].endswith("realDream_klein9BV2.safetensors") and os.path.isabs(p["model"])
    w2 = W("klein_armonizza", 0.5)
    mw.MainWindow._start_integrate_approach(w2, "klein_armonizza", "flux2", _exp(tmp_path), {}, "match the light", "Flux\\realDream_klein9BV2.safetensors", "", 7, 12, 4.0)
    p2 = calls[-1][1]
    assert p2["strength"] == 0.5 and p2["image_path"].endswith("comp.png") and "ref_images" not in p2                # composito, nessun riferimento


def test_kontext_stitch_una_immagine(stub, tmp_path):
    mw, W, calls = stub
    w = W()
    ok = mw.MainWindow._start_integrate_approach(w, "kontext_stitch", "kontext", _exp(tmp_path), {}, "", "Flux\\FLUX_Kontext_Dev.safetensors", "", 3, 28, 2.5)
    assert ok
    kind, exp, model, prompt, extra = calls[-1]
    assert kind == "kontext" and "mask" not in exp and extra["wh"] == {"width": 400, "height": 300}                  # nessuna maschera: edit per istruzione
    w_strip = Image.open(exp["image"]).size
    assert w_strip[1] == 300 and w_strip[0] > 400                                                                      # scena | soggetto affiancati
    assert "strip of 2 pictures" in prompt and "on the right side" in prompt


def test_approccio_sconosciuto_o_motore_diverso_non_fa_nulla(stub, tmp_path):
    mw, W, calls = stub
    assert mw.MainWindow._start_integrate_approach(W(), "boh", "qwen_image21", _exp(tmp_path), {}, "", None, "", 1, 8, 1.0) is False
    assert mw.MainWindow._start_integrate_approach(W(), "armonizza", "kontext", _exp(tmp_path), {}, "", "m", "", 1, 8, 1.0) is False
    assert calls == []


# ── worker e backend: le chiavi opt-in ESISTONO e il default e invariato ────────────────────────────────────────────────
def _src(rel):
    return open(os.path.join(ROOT, rel), encoding="utf-8").read()


def test_worker_qwen_mask_ref_default_invariato():
    s = _src("workers/qwen_image21_infer.py")
    assert 'if init_img is not None and mask_path and p.get("mask_ref", True):' in s


def test_backend_flux_chiavi_opt_in():
    s = _src("inference_server/backends/image_flux.py")
    assert 'params.get("padding_mask_crop") and not params.get("outpaint")' in s
    assert 'kw["image_reference"]' in s and 'params.get("strength") is not None' in s


def test_worker_ltx_applica_la_lora_msr_25():
    s = _src("workers/ltx2_infer.py")
    m = re.search(r'elif _msr_strategy == "slot_embedding" and p.get\("msr_apply_lora", True\):\n(.*?)\n    ckpt = _resolve_ckpt', s, re.S)
    assert m and "LoraPathStrengthAndSDOps(p[\"msr_lora\"]" in m.group(1)
    # il file della LoRA 2.5 porta davvero i pesi del DiT (480 coppie) oltre agli slot: senza applicarli i riferimenti sono ignorati
    lora = r"E:\ai\Models\LoRA\LTX-2.5\LTX-2.5-Licon-MSR-V1.safetensors"
    if os.path.isfile(lora):
        import json
        import struct
        with open(lora, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            keys = [k for k in json.loads(fh.read(n)) if k != "__metadata__"]
        assert sum("lora_A" in k for k in keys) == 480 and sum("reference_slot" in k for k in keys) == 5
