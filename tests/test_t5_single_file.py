"""T5 single-file fp8 per FLUX.1 (inference_server/backends/t5_single_file): sorgente e layerwise upcast, su un T5 minuscolo."""
import os
import pytest
import torch

from inference_server.backends import t5_single_file as t5

transformers = pytest.importorskip("transformers")
safetensors = pytest.importorskip("safetensors.torch")


def _tiny(tmp_path):
    cfg = transformers.T5Config(vocab_size=64, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4,
                                feed_forward_proj="gated-gelu", relative_attention_num_buckets=8,
                                tie_word_embeddings=False)
    te = tmp_path / "_te" / "t5xxl"
    te.mkdir(parents=True)
    cfg.save_pretrained(te)
    torch.manual_seed(0)
    m = transformers.T5EncoderModel(cfg).eval()
    sd = {k: v.detach().to(torch.float8_e4m3fn) for k, v in m.state_dict().items()}
    f = tmp_path / "TE" / "t5xxl_enconly.safetensors"
    f.parent.mkdir()
    safetensors.save_file(sd, str(f))
    return cfg, te, f, m


def test_layerwise_uguale_al_denso_e_wo_non_castato_a_fp8(tmp_path):
    cfg, te, f, _ = _tiny(tmp_path)
    ids = torch.randint(1, 60, (2, 7))
    m8 = t5.load_t5_single_file(str(f), str(te), torch.bfloat16, keep_fp8=True)
    md = t5.load_t5_single_file(str(f), str(te), torch.bfloat16, keep_fp8=False)
    assert m8.encoder.block[0].layer[1].DenseReluDense.wo.w8.dtype == torch.float8_e4m3fn
    with torch.no_grad():
        a, b = m8(ids).last_hidden_state, md(ids).last_hidden_state
    assert a.dtype == torch.bfloat16 and torch.isfinite(a.float()).all()
    assert torch.equal(a, b)                     # stesso calcolo: upcast per layer = pesi bf16 densi


def test_pick_weights_precedenze(tmp_path, monkeypatch):
    monkeypatch.delenv("NATIVE_T5_WEIGHTS", raising=False)
    cfg, te, f, _ = _tiny(tmp_path)
    te_dir = str(te.parent)
    kind, p = t5.pick_weights(te_dir, str(f.parent))
    assert kind == "single" and p.endswith("t5xxl_enconly.safetensors")     # cartella senza pesi -> single-file
    (te / "model.safetensors").write_bytes(b"x")
    assert t5.pick_weights(te_dir, str(f.parent))[0] == "folder"             # la cartella HF vince sempre (storico)
    monkeypatch.setenv("NATIVE_T5_WEIGHTS", str(f))
    assert t5.pick_weights(te_dir, str(f.parent)) == ("single", str(f))      # override esplicito (test)
    monkeypatch.setenv("NATIVE_T5_WEIGHTS", str(tmp_path / "manca.safetensors"))
    with pytest.raises(FileNotFoundError):
        t5.pick_weights(te_dir)


def test_tensori_mancanti_errore_chiaro(tmp_path):
    cfg, te, f, m = _tiny(tmp_path)
    sd = {k: v.detach().to(torch.float8_e4m3fn) for k, v in m.state_dict().items() if "block.1." not in k}
    g = tmp_path / "TE" / "rotto.safetensors"
    safetensors.save_file(sd, str(g))
    with pytest.raises(ValueError, match="mancano"):
        t5.load_t5_single_file(str(g), str(te))
