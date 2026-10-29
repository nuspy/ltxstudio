"""core/arch_filter: classificazione dall'header, cosa sa fare davvero l'inpaint, default per modello, is_flux2 unico."""
from __future__ import annotations

import json
import struct

import pytest

from core import arch_filter as af

F16 = "F16"


def _st(path, tensors, meta=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    head = {k: {"dtype": F16, "shape": list(v), "data_offsets": [0, 0]} for k, v in tensors.items()}
    if meta:
        head["__metadata__"] = meta
    raw = json.dumps(head).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


FLUX1 = {"double_blocks.0.a.weight": (4, 4), "single_blocks.0.b.weight": (4, 4), "vector_in.w": (4, 4),
         "guidance_in.w": (4, 4), "img_in.weight": (3072, 64)}


@pytest.fixture()
def root(tmp_path, monkeypatch):
    from core import model_detect, offline_inventory
    monkeypatch.setattr(offline_inventory, "MODELS_ROOT", tmp_path)
    monkeypatch.setattr(model_detect, "_MODELS_ROOT", tmp_path)
    monkeypatch.setattr(model_detect, "CACHE_PATH", tmp_path / "md.json")
    monkeypatch.setitem(model_detect._MEM, "cache", None)
    monkeypatch.setitem(model_detect._MEM, "mtime", None)
    af.clear_memo()
    d = tmp_path / "diffusion_models"
    _st(d / "Flux" / "base.safetensors", FLUX1)
    _st(d / "Flux" / "fill.safetensors", dict(FLUX1, **{"img_in.weight": (3072, 384)}))
    _st(d / "Flux" / "klein.safetensors", {"double_blocks.0.a.weight": (4, 4), "single_blocks.0.b.weight": (4, 4),
                                          "img_in.weight": (4096, 128)})
    _st(d / "Flux" / "dev2.safetensors", {"double_blocks.0.a.weight": (4, 4), "single_blocks.0.b.weight": (4, 4),
                                         "guidance_in.w": (4, 4), "img_in.weight": (6144, 128)})
    _st(d / "Flux" / "nunchaku" / "svdq-fp4_r32-FLUX.2-klein.safetensors",
        {"x.qweight": (4, 4), "img_in.weight": (4096, 128)})
    _st(d / "Z-Image" / "nunchaku" / "svdq-fp4_r128-z-image-turbo.safetensors", {"x.qweight": (4, 4)})
    _st(d / "Krea2" / "k.safetensors", {"blocks.0.attn.wq.weight": (4, 4)})
    (d / "Flux" / "g.gguf").write_bytes(b"GGUF")
    (d / "Flux" / "fill-hyp.gguf").write_bytes(b"GGUF")
    yield tmp_path
    af.clear_memo()


def test_inpaint_kind_from_header(root):
    k = af.inpaint_kind
    assert k("Flux\\fill.safetensors") == "fill"
    assert k("Flux\\base.safetensors") == "base"
    assert k("Flux\\klein.safetensors") == "klein"
    assert k("Flux\\dev2.safetensors") is None                       # FLUX.2 Dev: guidance_in → nessuna pipeline inpaint
    assert k("Flux\\nunchaku\\svdq-fp4_r32-FLUX.2-klein.safetensors") is None
    assert k("Krea2\\k.safetensors") is None
    assert k("Flux\\g.gguf") is None and k("Flux\\fill-hyp.gguf") == "fill"
    assert k("inesistente.safetensors") is None


def test_is_flux2_unica_definizione(root):
    assert af.is_flux2("Flux\\klein.safetensors") and af.is_flux2("Flux\\dev2.safetensors")
    assert not af.is_flux2("Flux\\base.safetensors")
    # nome non su disco: niente falso positivo «flux_2» in «Flux_20»
    assert not af.is_flux2("PornFlux_20_x.safetensors")
    assert af.is_flux2("FLUX_2_Dev.safetensors") and af.is_flux2("flux2-klein.safetensors")
    from core import gen_pipeline, inpaint_pipeline
    assert gen_pipeline.is_flux2("Flux\\klein.safetensors") and inpaint_pipeline.is_flux2("Flux\\klein.safetensors")
    assert not gen_pipeline.is_flux2("PornFlux_20_x.safetensors")


def test_nunchaku_klein_non_e_flux1(root):
    c = af.classify("Flux\\nunchaku\\svdq-fp4_r32-FLUX.2-klein.safetensors")
    assert c["family"] == "flux2" and c["packed"] == "nunchaku"


def test_nunchaku_zimage_non_e_flux1(root):
    """task 17: lo svdq di Z-Image finiva fra i FLUX.1 (ramo packed cadeva su flux1)."""
    c = af.classify("Z-Image" + chr(92) + "nunchaku" + chr(92) + "svdq-fp4_r128-z-image-turbo.safetensors")
    assert c["family"] == "zimage" and c["packed"] == "nunchaku"


def test_default_del_proprio_modello_coincidono_col_backend():
    from inference_server.backends.image_flux import _auto_settings
    assert af.inpaint_defaults("fill", "FLUX_DEV_Fill") == (40, 30.0)
    assert af.inpaint_defaults("klein", "k") == (28, 4.0)
    assert af.inpaint_defaults("base", "x") == (28, 3.5)
    for kind, arch in (("fill", "inpaint"), ("base", "flux"), ("klein", "flux2_inpaint")):
        for name in ("model", "x-hyper-8", "turbo"):
            s, g, _h = _auto_settings(arch, name)
            assert af.inpaint_defaults(kind, name) == (s, g)


def test_copia_cache_normalized_fuori_da_tutti_i_menu(tmp_path, monkeypatch):
    """`X.safetensors.normalized.safetensors` e' la cache di image_flux, non un modello: l'inventario la nasconde."""
    from core import offline_inventory as oi
    monkeypatch.setattr(oi, "MODELS_ROOT", tmp_path)
    d = tmp_path / "diffusion_models" / "Flux"
    d.mkdir(parents=True)
    (d / "fux.safetensors").write_bytes(b"x")
    (d / "fux.safetensors.normalized.safetensors").write_bytes(b"x")
    (d / "altro.safetensors").write_bytes(b"x")
    assert [n.split(chr(92))[-1] for n in oi.list_models("diffusion_models")] == ["altro.safetensors", "fux.safetensors"]
    assert oi.is_cache_copy("a.safetensors.normalized.safetensors") and not oi.is_cache_copy("a.safetensors")
