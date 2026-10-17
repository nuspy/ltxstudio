"""GGUF NON quantizzato (BF16/F16/F32) -> safetensors, senza perdita e a flusso (un tensore alla volta).

Serve per i checkpoint distribuiti come «GGUF BF16» (es. ERNIE-Image-Turbo): il GGUF mette le dimensioni in ordine
inverso ma i byte sono già row-major della shape safetensors, quindi si copiano i byte e si invertono solo le
dimensioni. Rifiuta i tipi quantizzati (Q4/Q8…: lì serve il loader GGUF nativo di workers/gguf_native). L'ORIGINALE
NON SI CANCELLA. Verifica finale: confronto byte per byte di ogni tensore col GGUF.

Uso:  python -m core.gguf_to_safetensors SRC.gguf DST.safetensors
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

_DT = {"BF16": "BF16", "F16": "F16", "F32": "F32"}          # nome GGMLQuantizationType -> dtype safetensors
_CHUNK = 64 * 2**20


def read_plan(src: str) -> list[dict]:
    """[{name, shape, dtype, nbytes, tensor}] — solleva ValueError se c'è un tipo quantizzato."""
    import gguf
    reader = gguf.GGUFReader(src)
    plan, bad = [], []
    for t in reader.tensors:
        dt = _DT.get(t.tensor_type.name)
        if dt is None:
            bad.append(f"{t.name}:{t.tensor_type.name}")
            continue
        plan.append({"name": t.name, "shape": [int(x) for x in reversed(t.shape)], "dtype": dt,
                     "nbytes": int(t.n_bytes), "tensor": t})
    if bad:
        raise ValueError(f"{os.path.basename(src)} contiene tipi quantizzati ({bad[:3]}…): conversione lossless impossibile")
    return plan


def _bytes_of(t) -> memoryview:
    import numpy as np
    return memoryview(np.ascontiguousarray(t.data).reshape(-1).view(np.uint8))


def convert(src: str, dst: str, log=print) -> Path:
    dst_p = Path(dst)
    if dst_p.exists():
        raise FileExistsError(f"{dst} esiste già: non lo sovrascrivo")
    plan = read_plan(src)
    header, off = {}, 0
    for p in sorted(plan, key=lambda p: p["name"]):
        header[p["name"]] = {"dtype": p["dtype"], "shape": p["shape"], "data_offsets": [off, off + p["nbytes"]]}
        off += p["nbytes"]
    header["__metadata__"] = {"format": "pt", "converted_from": os.path.basename(src)}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    by_name = {p["name"]: p for p in plan}
    from core import resources
    resources.require(commit_gb=2.0, what="conversione GGUF -> safetensors")
    tmp = dst_p.with_name(dst_p.name + ".partial")
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for name in sorted(by_name):
            mv = _bytes_of(by_name[name]["tensor"])
            if len(mv) != by_name[name]["nbytes"]:
                raise ValueError(f"{name}: {len(mv)} byte letti, attesi {by_name[name]['nbytes']}")
            for i in range(0, len(mv), _CHUNK):
                f.write(mv[i:i + _CHUNK])
    verify(src, str(tmp), plan)
    os.replace(tmp, dst_p)
    log(f"[gguf] scritto {dst_p} ({len(plan)} tensori, {off / 2**30:.2f} GiB); originale {src} NON toccato")
    return dst_p


def verify(src: str, out: str, plan: list[dict] | None = None) -> None:
    plan = plan or read_plan(src)
    with open(out, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        hdr.pop("__metadata__", None)
        if set(hdr) != {p["name"] for p in plan}:
            raise ValueError("nomi tensore diversi dopo la conversione")
        for p in plan:
            h = hdr[p["name"]]
            if h["dtype"] != p["dtype"] or h["shape"] != p["shape"]:
                raise ValueError(f"{p['name']}: dtype/shape diversi")
            mv = _bytes_of(p["tensor"])
            a, _b = h["data_offsets"]
            f.seek(base + a)
            for i in range(0, len(mv), _CHUNK):
                if f.read(min(_CHUNK, len(mv) - i)) != bytes(mv[i:i + _CHUNK]):
                    raise ValueError(f"contenuto diverso dopo la conversione: {p['name']}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    convert(sys.argv[1], sys.argv[2])
