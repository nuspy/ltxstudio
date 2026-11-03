"""Klein 9B (realDream) = fp8 SCALATO in formato ComfyUI: la scala va applicata (peso = fp8 * weight_scale), le chiavi weight_scale/input_scale/comfy_quant
non devono finire nel mapping dei pesi (prima: «chunk expects at least a 1-dimensional tensor»), nessun tensore resta su 'meta'."""
from pathlib import Path

import pytest
import torch

REAL = Path(r"E:\AI\Models\diffusion_models\Flux\realDream_klein9BV2.safetensors")


@pytest.mark.skipif(not REAL.exists(), reason="serve il checkpoint realDream locale")
def test_scaled_fp8_klein_loads_with_scales(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import save_file
    from inference_server.backends.flux2_transformer import klein_config_from_header, load_flux2_klein_transformer
    keep = {}
    with safe_open(str(REAL), framework="pt") as h:
        for k in h.keys():
            parts = k.split(".")
            if "double_blocks" in parts or "single_blocks" in parts:
                if int(parts[parts.index("double_blocks" if "double_blocks" in parts else "single_blocks") + 1]) > 1:
                    continue
            keep[k] = h.get_tensor(k)
    mini = tmp_path / "mini.safetensors"
    save_file(keep, str(mini), metadata={"format": "pt"})
    cfg = klein_config_from_header(str(mini))
    assert cfg["num_layers"] == 2 and cfg["num_single_layers"] == 2
    tr = load_flux2_klein_transformer(str(mini), cfg)
    assert not [n for n, p in list(tr.named_parameters()) + list(tr.named_buffers()) if p.is_meta]
    key = "model.diffusion_model.double_blocks.0.img_attn.qkv"
    scale = keep[key + ".weight_scale"].item()
    attn = tr.transformer_blocks[0].attn
    assert abs(float(attn.to_q.weight_scale) - scale) < 1e-12 and abs(float(attn.to_v.weight_scale) - scale) < 1e-12
    assert torch.equal(attn.to_k.weight.view(torch.uint8), torch.chunk(keep[key + ".weight"], 3, 0)[1].view(torch.uint8))
