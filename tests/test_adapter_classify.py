"""Classificazione degli adattatori (task 9): ControlNet e LoRA di controllo NON sono modelli t2i; il filtro per
architettura li esclude dai menu di generazione. Usa i file veri se presenti (altrimenti salta)."""
from __future__ import annotations

import os

import pytest

from core import arch_filter as af, model_detect as md

M = r"E:\AI\Models"
CASES = [
    (rf"{M}\ControlNet\Flux\shakker_union_pro_v2.safetensors", "flux1", "controlnet"),
    (rf"{M}\ControlNet\Flux\alimama_flux_inpainting.safetensors", "flux1", "controlnet"),
    (rf"{M}\ControlNet\Z-Image\Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors", "zimage", "controlnet"),
    (rf"{M}\LoRA\MagicQuill\color_lora.safetensors", "flux1", "control_lora"),
    (rf"{M}\LoRA\MagicQuill\edge_lora.safetensors", "flux1", "control_lora"),
]


@pytest.mark.parametrize("path,fam,role", CASES)
def test_adapter_role_from_keys(path, fam, role):
    if not os.path.isfile(path):
        pytest.skip("file assente")
    d = md.detect_one(path)
    assert (d["family"], d["role"]) == (fam, role)
    assert d["role"] != "t2i"


@pytest.mark.parametrize("path,fam,role", CASES)
def test_adapters_never_in_t2i_menus(path, fam, role):
    if not os.path.isfile(path):
        pytest.skip("file assente")
    for arch in ("FLUX", "Z-Image", "Ernie"):
        assert not af.is_t2i_for_arch(arch, path)


def test_cache_version_invalidates_old_entries(tmp_path, monkeypatch):
    """Una voce in cache senza `_dv` corrente (classificata dal rilevatore vecchio) viene rifatta."""
    p = rf"{M}\ControlNet\Flux\shakker_union_pro_v2.safetensors"
    if not os.path.isfile(p):
        pytest.skip("file assente")
    st = os.stat(p)
    old = {"path": p, "family": "flux1", "role": "t2i", "_sig": [st.st_mtime_ns, st.st_size]}
    monkeypatch.setattr(md, "CACHE_PATH", tmp_path / "md.json")
    md._MEM.update({"cache": None, "mtime": None})
    (tmp_path / "md.json").write_text(__import__("json").dumps({p: old}), "utf-8")
    assert md.get(p)["role"] == "controlnet"
    md._MEM.update({"cache": None, "mtime": None})


def test_image_function_backend_matrix():
    """ControlNet/Redux: solo FLUX.1 e Z-Image (Redux solo FLUX.1); mai FLUX.2, Krea 2, Qwen, Ernie."""
    f = af.image_function_backend
    assert f("controlnet", "Z-Image") == "zimage_cn" and f("redux", "Z-Image") == ""
    assert f("controlnet", "FLUX") == "flux_cn" and f("redux", "FLUX") == "flux_redux"       # nessun modello: FLUX.1 di default
    for arch in ("Krea 2", "Qwen-Image 2.1", "Ernie"):
        assert f("controlnet", arch) == "" and f("redux", arch) == ""
    assert f("boh", "FLUX") == ""
    p = rf"{M}\Flux\FLUX_2_Dev_FP8_Mixed_Scaled.safetensors"
    if os.path.isfile(p):
        assert f("controlnet", "FLUX", r"Flux\FLUX_2_Dev_FP8_Mixed_Scaled.safetensors") == ""
    q = rf"{M}\Flux\FLUX_Dev.safetensors"
    if os.path.isfile(q):
        assert f("controlnet", "FLUX", r"Flux\FLUX_Dev.safetensors") == "flux_cn"
