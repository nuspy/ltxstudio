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


MIRA = Path(r"E:\AI\Models\diffusion_models\Flux\miraclein_flux2klein_nsfw_v2_fp8.safetensors")
CACHE = Path(r"E:\AI\Models\_dequant_cache\realDream_klein9BV2.none.0a43e0536962.safetensors")


@pytest.mark.skipif(not REAL.exists() or not MIRA.exists(), reason="servono i checkpoint locali")
def test_scaled_fp8_klein_is_recognised_and_plain_fp8_is_not():
    from inference_server.backends.flux2_transformer import is_scaled_fp8_klein
    assert is_scaled_fp8_klein(str(REAL)) is True
    assert is_scaled_fp8_klein(str(MIRA)) is False                       # fp8 senza weight_scale: percorso di sempre


@pytest.mark.skipif(not REAL.exists() or not CACHE.exists(), reason="serve la cache dequantizzata del realDream")
def test_exact_scale_forward_equals_the_dequantized_cache_bit_for_bit():
    """Il Klein fp8 scalato non si dequantizza piu: il forward di Fp8Linear con la scala del file deve dare lo STESSO peso bf16 della cache."""
    from safetensors import safe_open
    from inference_server.backends.flux2_transformer import Fp8Linear
    pref = "model.diffusion_model."
    with safe_open(str(REAL), framework="pt") as ho, safe_open(str(CACHE), framework="pt") as hc:
        for base in ("double_blocks.0.img_attn.qkv", "double_blocks.3.img_mlp.0", "single_blocks.5.linear1"):
            w = ho.get_tensor(pref + base + ".weight")
            sc = ho.get_tensor(pref + base + ".weight_scale").float()
            ref = hc.get_tensor(pref + base + ".weight") if pref + base + ".weight" in hc.keys() else hc.get_tensor(base + ".weight")
            lin = Fp8Linear(w.shape[1], w.shape[0], bias=False)
            lin.weight = w
            lin.weight_scale = sc
            lin.exact_scale = True
            x = torch.eye(w.shape[1], dtype=torch.bfloat16)[:8]               # y = x @ W^T: righe di W^T selezionate
            y = lin(x)
            assert torch.equal(y, ref.t()[:8].to(torch.bfloat16)) or torch.equal(y.float(), (x.float() @ ref.float().t())), base
