"""Mappa chiavi Krea 2 single-file ComfyUI → Krea2Transformer2DModel (verificata sull'header di KreaNoct V4 contro
lo snapshot ufficiale Krea-2-Turbo, 13/10/2026: 0 chiavi mancanti/inattese)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workers"))

import krea2_custom as K  # noqa: E402


def test_blocchi():
    assert K.dit_key("blocks.3.attn.wq.weight") == "transformer_blocks.3.attn.to_q.weight"
    assert K.dit_key("blocks.3.attn.wo.weight_scale") == "transformer_blocks.3.attn.to_out.0.weight_scale"
    assert K.dit_key("blocks.3.attn.gate.convrot_group") == "transformer_blocks.3.attn.to_gate.convrot_group"
    assert K.dit_key("blocks.3.attn.qknorm.knorm.scale") == "transformer_blocks.3.attn.norm_k.weight"
    assert K.dit_key("blocks.3.mlp.gate.weight") == "transformer_blocks.3.ff.gate.weight"
    assert K.dit_key("blocks.3.prenorm.scale") == "transformer_blocks.3.norm1.weight"
    assert K.dit_key("blocks.3.mod.lin") == "transformer_blocks.3.scale_shift_table"
    assert K.is_block_modulation("blocks.3.mod.lin") and not K.is_block_modulation("last.modulation.lin")
    assert K.is_block_modulation("model.diffusion_model.blocks.3.mod.lin")         # prefisso ComfyUI (SickRealism)


def test_testa_coda_e_text_fusion():
    assert K.dit_key("first.weight") == "img_in.weight"
    assert K.dit_key("last.modulation.lin") == "final_layer.scale_shift_table"
    assert K.dit_key("last.norm.scale") == "final_layer.norm.weight"
    assert K.dit_key("tmlp.2.bias") == "time_embed.linear_2.bias"
    assert K.dit_key("tproj.1.weight") == "time_mod_proj.weight"
    assert K.dit_key("txtmlp.0.scale") == "txt_in.norm.weight"
    assert K.dit_key("txtmlp.3.weight") == "txt_in.linear_2.weight"
    assert K.dit_key("txtfusion.refiner_blocks.1.mlp.up.weight") == "text_fusion.refiner_blocks.1.ff.up.weight"
    assert K.dit_key("txtfusion.projector.weight") == "text_fusion.projector.weight"


def test_lora_e_idempotenza():
    assert K.lora_key("diffusion_model.blocks.0.attn.wq.lora_A.weight") == "transformer_blocks.0.attn.to_q.lora_A.weight"
    assert K.lora_key("transformer.txtfusion.layerwise_blocks.0.attn.wk.lora_B.weight") == \
        "text_fusion.layerwise_blocks.0.attn.to_k.lora_B.weight"
    d = "transformer_blocks.5.attn.to_out.0.weight"                       # nomi diffusers: invariati
    assert K.dit_key(d) == d


def test_lora_kohya():
    assert K.lora_key("lora_unet_blocks_12_attn_gate.lora_down.weight") == "transformer_blocks.12.attn.to_gate.lora_down.weight"
    assert K.lora_key("lora_unet_blocks_3_mlp_up.alpha") == "transformer_blocks.3.ff.up.alpha"
    assert K.lora_key("lora_unet_txtfusion_refiner_blocks_0_attn_wq.lora_up.weight") == \
        "text_fusion.refiner_blocks.0.attn.to_q.lora_up.weight"
