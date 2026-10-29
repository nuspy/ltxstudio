"""API immagini: @image_N nel prompt = k-esimo elemento di ref_images (stessa convenzione della GUI)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from api.handlers import images  # noqa: E402


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.setattr(images.gate, "require", lambda *a, **k: None)


def test_token_tradotto_in_image_n():
    p = images._qwen21({"prompt": "la tartaruga in @image_1 mangia la lattuga in @image_2", "arch": "qwen_image21",
                        "ref_images": ["a.png", "b.png"]})
    assert p["prompt"] == "la tartaruga in image 1 mangia la lattuga in image 2"


def test_image_n_scritto_a_mano_resta():
    p = images._qwen21({"prompt": "image 1 al mare", "arch": "qwen_image21", "ref_images": ["a.png"]})
    assert p["prompt"] == "image 1 al mare"


def test_token_inesistente_e_422():
    with pytest.raises(HTTPException) as e:
        images._qwen21({"prompt": "@image_1 con @image_3", "arch": "qwen_image21", "ref_images": ["a.png", "b.png"]})
    assert e.value.status_code == 422 and "@image_3" in str(e.value.detail)
    with pytest.raises(HTTPException):
        images._qwen21({"prompt": "@image_1", "arch": "qwen_image21"})


def test_foto_non_citata_resta_nella_lista():
    p = images._qwen21({"prompt": "solo @image_2", "arch": "qwen_image21", "ref_images": ["a.png", "b.png", "c.png"]})
    assert p["ref_images"] == ["a.png", "b.png", "c.png"] and p["prompt"] == "solo image 2"


def test_guidance_kontext_default_e_range():
    """Kontext: guidance default 2.5, ammessa 1.0-4.0 (valori bassi attenuano contrasto/saturazione eccessivi)."""
    p = {"arch": "kontext"}
    images._kontext_guidance(p)
    assert p["guidance"] == 2.5
    p = {"arch": "kontext", "guidance": 1.0}
    images._kontext_guidance(p)
    assert p["guidance"] == 1.0
    for bad in (0.5, 4.5, 30.0):
        with pytest.raises(HTTPException) as e:
            images._kontext_guidance({"arch": "kontext", "guidance": bad})
        assert e.value.status_code == 422
    q = {"arch": "flux2_kontext", "guidance": 4.0}                       # altri motori: invariati
    images._kontext_guidance(q)
    assert q["guidance"] == 4.0
