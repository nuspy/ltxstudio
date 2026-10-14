"""Convertitore «ltx25_te»: TE LTX-2.5 int8 per-tensore ComfyUI → stesso file con gli int8 intatti (il fold a bf16
lo fa ltx-core al caricamento) + asset/metadata innestati in coda (gancio `extras`). File sintetico, niente rete."""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import native_convert as nc  # noqa: E402


def test_ltx25_te_int8_intatto_e_asset(tmp_path, monkeypatch):
    torch.manual_seed(0)
    q = torch.randint(-127, 128, (64, 32), dtype=torch.int8)
    sc = torch.tensor([0.01], dtype=torch.float32)
    src = tmp_path / "te_int8.safetensors"
    save_file({"model.layers.0.mlp.up_proj.weight": q, "model.layers.0.mlp.up_proj.weight_scale": sc,
               "model.norm.weight": torch.ones(32, dtype=torch.bfloat16),
               "tokenizer_json": torch.tensor(list(b'{"t":1}'), dtype=torch.uint8)}, str(src),
              metadata={"quantization": "int8_tensorwise"})
    asset = torch.tensor(list(b"{{ chat }}"), dtype=torch.uint8)
    monkeypatch.setattr(nc, "_ltx25_te_assets", lambda: ({"gemma_config": json.dumps({"a": 1})},
                                                         {"hf_asset__chat_template.jinja": asset,
                                                          "tokenizer_json": torch.zeros(3, dtype=torch.uint8)}))
    dst = tmp_path / "te_int8_native.safetensors"
    nc.convert("ltx25_te", src, dst, log=lambda *_: None)
    assert nc.is_native(dst) and not src.exists()
    with safe_open(str(dst), "pt") as f:
        md = f.metadata()
        assert torch.equal(f.get_tensor("model.layers.0.mlp.up_proj.weight"), q)
        assert torch.equal(f.get_tensor("model.layers.0.mlp.up_proj.weight_scale"), sc)
        assert bytes(f.get_tensor("hf_asset__chat_template.jinja").tolist()) == b"{{ chat }}"
        assert bytes(f.get_tensor("tokenizer_json").tolist()) == b'{"t":1}'        # quello del sorgente vince
    assert json.loads(md["gemma_config"]) == {"a": 1}
