"""Krea 2: _load_transformer_fp8_direct (device_map + cast fp8 sul device) = stessi pesi di from_pretrained + enable_layerwise_casting, su un transformer
minuscolo (stesso codice, device cpu). Servono diffusers con Krea2Transformer2DModel (venv lingbot): altrimenti si salta."""
import hashlib
import os

import pytest
import torch

try:
    from diffusers import Krea2Transformer2DModel
except Exception:                                    # noqa: BLE001
    Krea2Transformer2DModel = None


def _fp(m):
    return {n: (str(p.dtype), tuple(p.shape), hashlib.sha1(p.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
            for n, p in list(m.named_parameters()) + list(m.named_buffers())}


@pytest.mark.skipif(Krea2Transformer2DModel is None, reason="diffusers senza Krea2")
def test_direct_fp8_load_matches_cpu_path(tmp_path):
    from workers import krea2_infer as W
    cfg = dict(in_channels=16, num_layers=2, attention_head_dim=16, num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
               timestep_embed_dim=32, text_hidden_dim=32, num_text_layers=3, text_num_attention_heads=2, text_num_key_value_heads=2,
               text_intermediate_size=64, num_layerwise_text_blocks=1, num_refiner_text_blocks=1, axes_dims_rope=(4, 6, 6), rope_theta=1000.0,
               norm_eps=1e-5)
    torch.manual_seed(0)
    m = Krea2Transformer2DModel(**cfg).to(torch.bfloat16)
    d = tmp_path / "model"
    os.makedirs(d / "transformer")
    m.save_pretrained(str(d / "transformer"))
    old = Krea2Transformer2DModel.from_pretrained(str(d / "transformer"), torch_dtype=torch.bfloat16)
    old.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    new = W._load_transformer_fp8_direct(str(d), (), "cpu")
    fo, fn = _fp(old), _fp(new)
    assert set(fo) == set(fn) and all(fo[k] == fn[k] for k in fo)
    assert any(v[0] == "torch.float8_e4m3fn" for v in fn.values())
