"""Scrittore GGUF minimo per i test (file sintetici): nessuna dipendenza oltre numpy."""
import struct
from pathlib import Path


def gguf_write(path: Path, kv: dict, tensors: list):
    """tensors = [(nome, array float32 numpy, tipo ggml 0|8)]; dims ggml = shape numpy invertita."""
    import numpy as np

    def s(x: str):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(tensors), len(kv))
    for k, v in kv.items():
        head += s(k)
        if isinstance(v, str):
            head += struct.pack("<I", 8) + s(v)
        else:
            head += struct.pack("<I", 4) + struct.pack("<I", int(v))
    blobs, infos, off = [], b"", 0
    for name, arr, ty in tensors:
        if ty == 8:
            flat = arr.astype("float32").reshape(-1, 32)
            sc = (np.abs(flat).max(1) / 127.0).astype("float16")
            q = np.round(flat / np.maximum(sc.astype("float32"), 1e-9)[:, None]).clip(-127, 127).astype("int8")
            raw = b"".join(sc[i:i + 1].tobytes() + q[i].tobytes() for i in range(len(flat)))
        else:
            raw = arr.astype("float32").tobytes()
        pad = (-off) % 32
        blobs.append(b"\0" * pad + raw)
        off += pad
        dims = tuple(reversed(arr.shape))
        infos += s(name) + struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", ty, off)
        off += len(raw)
    blob = head + infos
    blob += b"\0" * ((-len(blob)) % 32) + b"".join(blobs)
    path.write_bytes(blob)
