"""Unit test detection fp8 E4M3 dall'header (FASE C review) — safetensors sintetici."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


def _detect(path):
    from inference_server.backends.image_flux import _flux_fp8_e4m3
    return _flux_fp8_e4m3(path)


def test_detect_fp8_true(tmp_path):
    p = str(tmp_path / "fp8.safetensors")
    t = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
    save_file({"double_blocks.0.img_mod.lin.weight": t.clone(),
               "double_blocks.1.img_mod.lin.weight": t.clone(),
               "single_blocks.0.linear1.weight": t.clone()}, p)
    assert _detect(p) is True


def test_detect_bf16_false(tmp_path):
    p = str(tmp_path / "bf16.safetensors")
    t = torch.zeros(4, 4, dtype=torch.bfloat16)
    save_file({"double_blocks.0.img_mod.lin.weight": t.clone(),
               "single_blocks.0.linear1.weight": t.clone()}, p)
    assert _detect(p) is False


def test_detect_mixed_majority(tmp_path):
    # pochi tensori fp8 su un file bf16 (misto) → maggioranza bf16 → False (fallback sicuro)
    p = str(tmp_path / "mixed.safetensors")
    f8 = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
    bf = torch.zeros(4, 4, dtype=torch.bfloat16)
    save_file({"double_blocks.0.img_mod.lin.weight": f8.clone(),
               "double_blocks.1.img_mod.lin.weight": bf.clone(),
               "double_blocks.2.img_mod.lin.weight": bf.clone()}, p)
    assert _detect(p) is False


def test_detect_missing_file():
    assert _detect(r"X:\non\esiste.safetensors") is False
