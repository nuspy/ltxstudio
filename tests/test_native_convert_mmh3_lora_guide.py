"""Convertitore «mmh3_lora_guide»: la LoRA h3upscale (nomi ComfyUI diffusion_model.blocks.N.…lora_A/lora_B, senza
alpha) → DiT H3 nativo, e l'intestazione con la geometria della guida SOPRAVVIVE (con mmh3_lora andrebbe persa e la
LoRA non sarebbe più riconoscibile come LoRA di guida dopo la cancellazione dell'originale).
Chiavi e intestazione replicano quelle di sandpies/ComfyUI-H3-Video-Upsampler lora/h3upscale_v2_x2.safetensors
(header letto via HTTP range il 17/10/2026: 416 tensori, rank 64, metadata di guida)."""
import json
import os
import struct
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import native_convert as nc  # noqa: E402
from core import h3_upsample_plan as P  # noqa: E402

META = {
    "reference_downscale_factor": "2", "minimax_h3_guide_spatial_version": "target_crop_v2",
    "minimax_h3_guide_position_version": "target_grid_stride_v1", "guide_latent_only": "true",
    "minimax_h3_reference_rope": json.dumps({"guide_rope_layout": "overlap", "reference_rope_layout": "native"}),
    "ss_base_model_version": "minimax_h3_ref2va", "name": "h3upscale_v2_x2",
}


def _synthetic():
    sd = {}
    r = 4
    for i in range(3):
        for mod in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
            sd[f"diffusion_model.blocks.{i}.{mod}.lora_A.weight"] = torch.randn(r, 16)
            sd[f"diffusion_model.blocks.{i}.{mod}.lora_B.weight"] = torch.randn(16, r)
    for mod in ("attn.qkv_proj", "mlp.fc1"):
        sd[f"diffusion_model.token_refiner.blocks.0.{mod}.lora_A.weight"] = torch.randn(r, 16)
        sd[f"diffusion_model.token_refiner.blocks.0.{mod}.lora_B.weight"] = torch.randn(16, r)
    return sd


def _meta(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).get("__metadata__") or {}


def test_metadata_di_guida_sopravvive(tmp_path):
    src, dst = tmp_path / "h3upscale.safetensors", tmp_path / "h3upscale-native.safetensors"
    save_file(_synthetic(), str(src), metadata=META)
    nc.convert("mmh3_lora_guide", src, dst, delete_source=False, log=lambda *_: None)
    md = _meta(dst)
    assert md["reference_downscale_factor"] == "2"
    assert md["ltx_native"] == nc.FORMAT
    assert P.guide_factor_from_meta(md) == 2                      # riconosciuta come LoRA di guida dopo la conversione
    out = load_file(str(dst))
    assert "transformer_blocks.0.attn.to_qkv.lora_A.weight" in out or any("to_qkv" in k for k in out), list(out)[:6]
    assert any(".ff.gate_up." in k for k in out), "fc1 deve diventare ff.gate_up (MLP fuso)"
    assert len(out) == len(_synthetic())


def test_mmh3_lora_normale_perde_il_metadata(tmp_path):
    """Documenta il motivo del convertitore dedicato."""
    src, dst = tmp_path / "x.safetensors", tmp_path / "x-native.safetensors"
    save_file(_synthetic(), str(src), metadata=META)
    nc.convert("mmh3_lora", src, dst, delete_source=False, log=lambda *_: None)
    assert "reference_downscale_factor" not in _meta(dst)
