"""VAE H3 solo encoder per le codifiche e pulizia dei pesi del testo (turno 40 del 04/11: OOM a 736p nella codifica del VAE)."""
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
sys.path.insert(0, str(ROOT))
VENV_PY = r"E:\AI\Tools\diffusers_next_venv\Scripts\python.exe"
VAE_FILE = r"E:\AI\Models\VAE\minimax_h3_video_vae_fp16-native.safetensors"


def write_safetensors(path, tensors):
    import json
    header, blobs, offset = {}, [], 0
    for name, tensor in tensors.items():
        data = tensor.numpy().tobytes()
        header[name] = {"dtype": "F16", "shape": list(tensor.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def test_header_keys_and_skip_leave_the_decoder_out(tmp_path):
    import minimax_h3_infer as worker
    import _qlinear
    path = tmp_path / "vae.safetensors"
    write_safetensors(path, {"encoder.conv_in.weight": torch.ones(2, 2, dtype=torch.float16), "decoder.conv_out.weight": torch.ones(3, 3, dtype=torch.float16),
                             "post_quant_conv.weight": torch.ones(1, dtype=torch.float16), "quant_conv.weight": torch.ones(1, dtype=torch.float16)})
    keys = worker._header_keys(str(path))
    assert sorted(keys) == ["decoder.conv_out.weight", "encoder.conv_in.weight", "post_quant_conv.weight", "quant_conv.weight"]
    skip = frozenset(key for key in keys if key.startswith(worker.VAE_DECODER_PREFIXES))
    kept = _qlinear.read_safetensors(str(path), "cpu", skip=skip)
    assert sorted(kept) == ["encoder.conv_in.weight", "quant_conv.weight"]


def test_encoder_only_is_used_where_the_vae_only_encodes_and_the_text_leftovers_are_purged():
    worker_source = (ROOT / "workers" / "minimax_h3_infer.py").read_text(encoding="utf-8")
    assert 'comps = {"vae": load_vae(W, encoder_only=True)}' in worker_source
    assert "def load_vae(W: dict, encoder_only: bool = False, device: str = \"cuda\")" in worker_source
    after_text = worker_source.split("_drop_module_weights(te)", 1)[1].split("# 2) VAE encoder", 1)[0]
    assert "purge_big_quantized_tensors(log=_log)" in after_text                       # subito dopo il testo e prima della codifica del VAE
    assert "load_vae(W)" not in worker_source.split("# 2) VAE encoder", 1)[1].split("# 3) denoise", 1)[0]
    edit_source = (ROOT / "workers" / "h3_edit_run.py").read_text(encoding="utf-8")
    assert "vae = W.load_vae(Wd, encoder_only=True)" in edit_source
    decode_source = worker_source.split("def _decode_and_free", 1)[1].split("\ndef ", 1)[0]
    assert "encoder_only" not in decode_source                                          # il decode usa il VAE intero


@pytest.mark.skipif(not (os.path.isfile(VENV_PY) and os.path.isfile(VAE_FILE)), reason="serve il venv di H3 e il VAE sul disco")
def test_encoder_only_vae_encodes_exactly_like_the_full_vae_on_cpu():
    result = subprocess.run([VENV_PY, str(ROOT / "tests" / "h3_vae_encoder_only_check.py")], capture_output=True, text=True, timeout=900)
    assert "VAE ENCODER-ONLY OK" in result.stdout, result.stdout[-800:] + result.stderr[-800:]
