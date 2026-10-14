"""LightVAE H3 (workers/h3_lightvae): encoder Lite potato sulla struttura diffusers reale, decoder a 12 blocchi e
sostituzione delle sole chiavi decoder./encoder. del VAE ufficiale. Serve diffusers con AutoencoderKLMiniMaxH3
(venv diffusers_next); con i pesi scaricati confronta anche le chiavi con i file Lite."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "workers"))

import h3_lightvae as LV  # noqa: E402

CFG = r"E:\AI\Models\configs\minimax-h3\vae"
KEPT_ENC = [0, 2, 4, 6, 8, 11]                  # kept_residual_indices del file ufficiale lightvae-lite-h3-encoder


def _vae(layers=None):
    try:
        from diffusers import AutoencoderKLMiniMaxH3
    except ImportError:
        pytest.skip("diffusers senza AutoencoderKLMiniMaxH3 (venv diffusers_next)")
    if not os.path.isdir(CFG):
        pytest.skip("config MiniMax H3 assente")
    from accelerate import init_empty_weights
    cfg = AutoencoderKLMiniMaxH3.load_config(CFG)
    if layers:
        cfg = {**cfg, "decoder_num_layers": layers}
    with init_empty_weights():
        return AutoencoderKLMiniMaxH3.from_config(cfg)


def test_merge_state_sostituisce_solo_il_componente():
    base = {"decoder.a": 1, "decoder.b": 2, "encoder.c": 3, "quant_conv.w": 4, "post_quant_conv.w": 6}
    # il file Lite reale porta anche latents_mean/std: nel VAE diffusers sono config → ignorati
    out = LV.merge_state(base, dec={"decoder.x": 9, "post_quant_conv.w": 7, "latents_mean": 0, "latents_std": 0})
    assert out == {"decoder.x": 9, "encoder.c": 3, "quant_conv.w": 4, "post_quant_conv.w": 7}
    out = LV.merge_state(base, enc={"encoder.y": 8, "latents_std": 0})
    assert "encoder.c" not in out and out["encoder.y"] == 8 and out["decoder.a"] == 1 and "latents_std" not in out
    with pytest.raises(ValueError):
        LV.merge_state(base, dec={"encoder.z": 1})                     # file dell'altro componente


def test_prune_encoder_e_decoder_ridotto():
    vae = _vae(layers=12)
    assert len(vae.decoder.transformer_blocks) == 12
    assert LV.prune_encoder(vae, KEPT_ENC) == 12 - len(KEPT_ENC)
    flat = [b for st in vae.encoder.down_blocks for b in st.resnets]
    for i, b in enumerate(flat):
        assert isinstance(b, LV.ResidualShortcut) == (i not in KEPT_ENC)
    for b in flat:                                  # dei blocchi potati resta solo la scorciatoia
        if isinstance(b, LV.ResidualShortcut):
            assert {k.split(".")[0] for k in b.state_dict()} <= {"conv_shortcut"}
    with pytest.raises(ValueError):
        LV.prune_encoder(_vae(), [5, 1])


@pytest.mark.parametrize("part", ["encoder", "decoder"])
def test_chiavi_come_il_file_lite(part):
    path = os.path.join(r"E:\AI\Models\vae", "lightvae-lite-h3-" + part + ".safetensors")
    found = [os.path.join(d, f) for d, _, fs in os.walk(r"E:\AI\Models\vae") for f in fs
             if f == os.path.basename(path)]
    if not found:
        pytest.skip("pesi LightVAE non scaricati")
    from safetensors import safe_open
    with safe_open(found[0], "pt") as f:
        file_keys = {k for k in f.keys() if k.startswith(part + ".")}
    if part == "decoder":
        vae = _vae(layers=LV.decoder_layers(found[0]))
    else:
        vae = _vae()
        LV.prune_encoder(vae, LV.kept_residuals(found[0]))
    model_keys = {k for k in vae.state_dict() if k.startswith(part + ".")}
    assert file_keys == model_keys, (sorted(file_keys - model_keys)[:5], sorted(model_keys - file_keys)[:5])
