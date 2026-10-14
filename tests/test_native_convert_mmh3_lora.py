"""Convertitore «mmh3_lora»: nomi ComfyUI/kohya/nudi → DiT H3 nativo; con PIÙ LoRA sullo stesso modulo (file Civitai
«Nipple Sucking») vince la stessa convenzione che applica ComfyUI: nomi nudi > diffusion_model. > lora_unet_."""
import os
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import native_convert as nc  # noqa: E402

T = "transformer_blocks.{}.attn.to_out.0.{}"


def _lora(prefix_fmt, block, val):
    down, up = torch.full((4, 8), float(val)), torch.full((8, 4), float(val))
    k = prefix_fmt.format(block)
    return {f"{k}.lora_down.weight": down, f"{k}.lora_up.weight": up, f"{k}.alpha": torch.tensor(4.0)}


def test_precedenza_come_comfyui(tmp_path):
    sd = {}
    sd.update(_lora("lora_unet_blocks_{}_attn_out_proj", 0, 1))           # kohya: blocchi 0 e 1
    sd.update(_lora("lora_unet_blocks_{}_attn_out_proj", 1, 1))
    sd.update(_lora("diffusion_model.blocks.{}.attn.out_proj", 0, 2))     # ComfyUI: blocchi 0 e 1
    sd.update(_lora("diffusion_model.blocks.{}.attn.out_proj", 1, 2))
    sd.update(_lora("blocks.{}.attn.out_proj", 1, 3))                     # nudi: solo blocco 1
    sd.update(_lora("lora_unet_blocks_{}_attn_out_proj", 2, 1))           # solo kohya: blocco 2
    src = tmp_path / "multi.safetensors"
    save_file(sd, str(src))
    dst = tmp_path / "multi-native.safetensors"
    nc.convert("mmh3_lora", src, dst, log=lambda *_: None)
    out = load_file(str(dst))
    assert out[T.format(0, "lora_down.weight")][0, 0] == 2               # diffusion_model. batte kohya
    assert out[T.format(1, "lora_down.weight")][0, 0] == 3               # i nudi battono tutti
    assert out[T.format(2, "lora_down.weight")][0, 0] == 1               # unica fonte: resta
    assert len(out) == 9                                                  # 3 moduli × (down, up, alpha)
