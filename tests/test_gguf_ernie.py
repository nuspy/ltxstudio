"""GGUF non quantizzato -> safetensors (core/gguf_to_safetensors) e prompt enhancer ERNIE (inference_server/backends/
ernie_pe): casi piccoli, senza GPU e senza pesi reali."""
import json
import struct

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from core import gguf_to_safetensors as g2s
from inference_server.backends import ernie_pe


def _write_gguf(path, tensors):
    w = gguf.GGUFWriter(str(path), "wan")
    for name, arr in tensors.items():
        w.add_tensor(name, arr)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def test_convert_f32_f16_shape_e_byte(tmp_path):
    rng = np.random.default_rng(0)
    t = {"a.weight": rng.standard_normal((3, 5)).astype(np.float32),
         "b.bias": rng.standard_normal((7,)).astype(np.float16)}
    src, dst = tmp_path / "m.gguf", tmp_path / "m.safetensors"
    _write_gguf(src, t)
    g2s.convert(str(src), str(dst), log=lambda *_: None)
    with open(dst, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        base = 8 + n
        assert h["a.weight"]["shape"] == [3, 5] and h["a.weight"]["dtype"] == "F32"
        assert h["b.bias"]["dtype"] == "F16"
        a, b = h["a.weight"]["data_offsets"]
        f.seek(base + a)
        assert np.array_equal(np.frombuffer(f.read(b - a), np.float32).reshape(3, 5), t["a.weight"])
    assert src.exists()                                           # l'originale non si tocca
    with pytest.raises(FileExistsError):
        g2s.convert(str(src), str(dst))


def test_rifiuta_quantizzati(tmp_path):
    src = tmp_path / "q.gguf"
    w = gguf.GGUFWriter(str(src), "wan")
    data = np.zeros((2, 34), np.uint8)                            # 2 righe da 1 blocco Q8_0 (34 byte)
    w.add_tensor("q.weight", data, raw_shape=(2, 34), raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    with pytest.raises(ValueError, match="quantizzati"):
        g2s.convert(str(src), str(tmp_path / "q.safetensors"))


def test_pe_non_disponibile_e_fallback_prompt_originale(tmp_path):
    assert not ernie_pe.available(str(tmp_path))
    logs = []
    out = ernie_pe.enhance("un gatto", 512, 512, str(tmp_path), device="cpu", log=logs.append)
    assert out == "un gatto" and any("non riuscito" in l for l in logs)
