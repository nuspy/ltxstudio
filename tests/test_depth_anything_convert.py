"""Conversione Depth-Anything V2 .pth -> HF (core/depth_anything_convert): mappa chiavi, senza GPU."""
import os
import pytest

from core import depth_anything_convert as dc


def test_rename_key_casi_chiave():
    assert dc.rename_key("pretrained.pos_embed") == [("backbone.embeddings.position_embeddings", None)]
    q = dc.rename_key("pretrained.blocks.3.attn.qkv.weight")
    assert [n for n, _ in q] == [f"backbone.encoder.layer.3.attention.attention.{x}.weight" for x in ("query", "key", "value")]
    assert [p for _, p in q] == [(0, 3), (1, 3), (2, 3)]
    assert dc.rename_key("pretrained.blocks.0.ls1.gamma")[0][0] == "backbone.encoder.layer.0.layer_scale1.lambda1"
    assert dc.rename_key("depth_head.scratch.refinenet4.resConfUnit1.conv2.bias")[0][0] == \
        "neck.fusion_stage.layers.0.residual_layer1.convolution2.bias"
    assert dc.rename_key("depth_head.scratch.refinenet1.out_conv.weight")[0][0] == "neck.fusion_stage.layers.3.projection.weight"
    assert dc.rename_key("depth_head.scratch.layer1_rn.weight")[0][0] == "neck.convs.0.weight"
    assert dc.rename_key("depth_head.scratch.output_conv2.2.bias")[0][0] == "head.conv3.bias"
    with pytest.raises(KeyError):
        dc.rename_key("depth_head.scratch.qualcosa_di_nuovo.weight")


@pytest.mark.skipif(not os.path.isfile(dc.DEFAULT_SRC), reason="serve il .pth originale")
def test_mappa_completa_sul_pth_reale():
    """Tutte le 407 chiavi originali coprono ESATTAMENTE le 503 del modello HF (header, nessun tensore caricato in RAM)."""
    import torch
    from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation
    cfg = DepthAnythingConfig(**{k: v for k, v in dc.LARGE_CONFIG.items() if k not in ("architectures", "torch_dtype")})
    with torch.device("meta"):
        want = set(DepthAnythingForDepthEstimation(cfg).state_dict())
    # lettura con mmap: nessun commit significativo
    sd = torch.load(dc.DEFAULT_SRC, map_location="cpu", weights_only=True, mmap=True)
    got = {nk for k in sd for nk, _ in dc.rename_key(k)}
    assert len(sd) == dc.EXPECTED_KEYS
    assert got == want
