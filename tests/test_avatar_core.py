"""Test unitari della logica condivisa dell'avatar parlante (core/avatar) e dell'API /avatar/*.

Niente GPU: i motori sono sostituiti da finti che registrano i parametri. Coprono il calcolo dei segmenti,
il copione a due voci, l'adattamento della durata, l'istruzione di editing C/D, la mappatura della richiesta
API sui parametri del worker LongCat e i quattro modi immagine.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.avatar import audio_chain as A          # noqa: E402
from core.avatar import image_step as IS          # noqa: E402
from core.avatar import timing                    # noqa: E402


def _wav(p: Path, s: float, sr: int = 16000) -> str:
    t = np.linspace(0, s, int(sr * s), endpoint=False)
    sf.write(str(p), (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    return str(p)


@pytest.mark.parametrize("d,n", [(0.5, 1), (3.72, 1), (3.73, 2), (6.92, 2), (6.93, 3), (60, 19)])
def test_segments_for(d, n):
    assert timing.segments_for(d) == n
    assert timing.covered_seconds(n) >= d


def test_effective_duration_trim_and_limit():
    assert timing.effective_duration(20, trim_start=2, trim_end=8) == 6
    assert timing.effective_duration(20, max_s=5) == 5
    assert timing.effective_duration(20, trim_start=25) == 0


def test_parse_script_names_and_continuations():
    lines = A.parse_script("A: ciao\nB: ehi\ncome va?\nMarta: bene", ["A", "Marta"])
    assert lines == [(0, "ciao"), (1, "ehi come va?"), (1, "bene")]


def test_build_speech_two_aligned_tracks(tmp_path, monkeypatch):
    seq = iter([1.0, 2.0, 0.5])
    monkeypatch.setattr(A, "OUT_DIR", tmp_path)
    monkeypatch.setattr(A, "synth_line", lambda sp, text: _wav(tmp_path / f"l{text[:3]}.wav", next(seq), A.SR))
    r = A.build_speech("A: uno\nB: due\nA: tre", [A.Speaker(name="A"), A.Speaker(name="B")], gap_ms=0)
    d1, d2 = sf.info(r.tracks["person1"]).duration, sf.info(r.tracks["person2"]).duration
    assert abs(d1 - d2) < 1e-3 and abs(d1 - 3.5) < 0.01 and r.audio_type == "para"
    y2, _ = sf.read(r.tracks["person2"])
    assert np.abs(y2[: int(0.9 * A.SR)]).max() < 1e-6, "la persona 2 deve tacere mentre parla la 1"


def test_fit_duration_pad_and_error(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "OUT_DIR", tmp_path)
    src = _wav(tmp_path / "v.wav", 2.0, A.SR)
    assert abs(sf.info(A.fit_duration(src, 3.0, "pad")).duration - 3.0) < 0.01
    with pytest.raises(ValueError):
        A.fit_duration(src, 3.0, "error")


def test_instruction_keeps_identity_and_background():
    s = IS.build_instruction({"hair": "short platinum bob", "outfit": "red leather jacket"}, True, 1)
    assert "Keep the identity" in s and "short platinum bob" in s and "Keep the original background" in s
    s2 = IS.build_instruction({"background": "neon-lit Tokyo street"}, True, 3)
    assert "neon-lit Tokyo street" in s2 and "original background" not in s2 and "reference images" in s2


def test_make_image_modes(tmp_path, monkeypatch):
    img = tmp_path / "face.png"
    from PIL import Image
    Image.new("RGB", (64, 64)).save(img)
    calls = []
    monkeypatch.setattr(IS, "_run_image", lambda params, log=None: calls.append(params) or str(img))
    monkeypatch.setattr(IS, "_check_weights", lambda r: None)
    assert IS.make_image({"mode": "image", "image_path": str(img)}, "o.png") == str(img)
    assert IS.make_image({"mode": "none"}, "o.png") == ""
    IS.make_image({"mode": "text", "prompt": "a jazz saxophonist", "profile": "nvfp4", "aspect": "9:16"}, "o.png")
    assert calls[-1]["arch"] == "qwen_image21" and calls[-1]["qwen21_profile"] == "nvfp4"
    assert calls[-1]["aspect_2k"] == "9:16" and "jazz saxophonist" in calls[-1]["prompt"]
    IS.make_image({"mode": "character_multi", "character": {"ref_images": [str(img)] * 12,
                                                           "changes": {"hair": "curly"}}}, "o.png")
    assert len(calls[-1]["ref_images"]) == 10, "Qwen-Image 2.1 accetta al massimo 10 riferimenti"
    IS.make_image({"mode": "character", "engine": "krea2_edit",
                   "character": {"ref_images": [str(img), str(img)]}}, "o.png")
    assert calls[-1]["arch"] == "krea2_edit" and calls[-1]["image_path"] == str(img)
    with pytest.raises(ValueError):
        IS.make_image({"mode": "character", "engine": "flux", "character": {"ref_images": [str(img)]}}, "o.png")


def test_pipeline_maps_request_to_longcat(tmp_path, monkeypatch):
    from core.avatar import pipeline as P
    from core import longcat_service as lc
    img = tmp_path / "face.png"
    from PIL import Image
    Image.new("RGB", (64, 64)).save(img)
    voice = _wav(tmp_path / "voice.wav", 5.0)
    monkeypatch.setattr(P, "OUT", tmp_path)
    seen = {}

    class _Job:
        def __init__(self, params):
            self.p = params

        def run(self):
            Path(self.p["output"]).write_bytes(b"x")
            return 0
    monkeypatch.setattr(lc, "preflight", lambda p: [])
    monkeypatch.setattr(lc, "make_job", lambda p, **k: seen.setdefault("p", p) and _Job(p))
    res = P.run({"engine": "longcat",
                 "image": {"mode": "image", "image_path": str(img)},
                 "audio": {"mode": "file", "audio_path": voice, "vocal_separation": False},
                 "video": {"prompt": "a lighthouse keeper tells a ghost story", "resolution": "720p",
                           "precision_profile": "fp8", "seed": 7, "ref_img_index": 20}})
    p = seen["p"]
    assert p["mode"] == "ai2v" and p["resolution"] == "720p" and p["precision"] == "fp8" and p["seed"] == 7
    assert p["audio_path"] == voice and p["vocal_separation"] is False and p["ref_img_index"] == 20
    assert res["video"] == p["output"] and res["image"] == str(img)


def test_api_avatar_options_and_validation():
    from fastapi.testclient import TestClient
    from api.app import app
    c = TestClient(app)
    r = c.get("/avatar/options")
    assert r.status_code == 200 and "longcat" in r.json()["engines"]
    r = c.post("/avatar/generate", json={"video": {"ref_img_index": 99}})
    assert r.status_code == 422, "ref_img_index fuori da 0–30 va rifiutato"
