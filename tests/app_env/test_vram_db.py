"""Test del DB canonico footprint (core/vram_estimate, F1 unified-model-manager) — senza GPU.
Valida: round-trip con variant, compat voci vecchie, transformer_mb su safetensors sintetico,
migrazione legacy idempotente, scritture concorrenti da 4 processi (file mai corrotto)."""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_REPO = str(Path(__file__).resolve().parents[2])


def _make_st(path: str, tensors: dict) -> None:
    """Scrive un safetensors sintetico: {nome: (dtype, shape)} con dati zero."""
    hdr, off = {}, 0
    for name, (dt, shape) in tensors.items():
        n = 1
        for s in shape:
            n *= s
        nb = n * {"F8_E4M3": 1, "BF16": 2, "F32": 4}[dt]
        hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + nb]}
        off += nb
    hj = json.dumps(hdr).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hj)))
        fh.write(hj)
        fh.write(b"\0" * off)


def run() -> int:
    fails = []

    def check(name, cond):
        print(("OK  " if cond else "FAIL") + " " + name, flush=True)
        if not cond:
            fails.append(name)

    tmp = tempfile.mkdtemp(prefix="vram_db_")
    db_path = os.path.join(tmp, "model_vram_cache.json")
    os.environ["LTX_VRAM_DB"] = db_path
    from core import vram_estimate as V
    V._CACHE = db_path                                       # il modulo può essere già importato
    V._LEGACY_FOOTPRINT = os.path.join(tmp, "model_footprint.json")

    # modello sintetico: transformer fp8 1024x1024 (1MB) + text_encoder bf16 512x512 (0.5MB)
    ckpt = os.path.join(tmp, "fake_model.safetensors")
    _make_st(ckpt, {"transformer.blocks.0.w": ("F8_E4M3", (1024, 1024)),
                    "text_encoder.emb.w": ("BF16", (512, 512))})

    # 1) header_mb: include TE → NIENTE +9700; fp8_storage=True → F8 conta 1B
    hm = V.header_mb(ckpt, fp8_storage=True)
    check("header_mb ~1.5MB (fp8 1B, te 2B, no +9700)", hm == int((1024 * 1024 + 512 * 512 * 2) / 1024**2))

    # 2) transformer_mb: SOLO il transformer (esclude il TE, niente +9700)
    tm = V.transformer_mb(ckpt, fp8_storage=True)
    check("transformer_mb = solo transformer", tm == int(1024 * 1024 / 1024**2))

    # 3) round-trip cache con variant fp8 + metadati
    V.cache_put(ckpt, 1234, variant="fp8", source="measured", category="video",
                transformer_mb=1000, components={"transformer": 1000, "vae": 200})
    check("cache_get variant", V.cache_get(ckpt, "fp8") == 1234)
    check("cache_get senza variant → fallback voce base assente", V.cache_get(ckpt) == 0)
    check("transformer_mb preferisce il misurato", V.transformer_mb(ckpt, fp8_storage=True) == 1000)
    e = V._entry(ckpt, "fp8")
    check("metadati persistiti", e.get("source") == "measured" and e.get("category") == "video")

    # 4) compat: voce VECCHIA (solo footprint_mb/name) letta senza errori
    old_key = V._key(ckpt)
    V._atomic_update(lambda db: db.__setitem__(old_key, {"footprint_mb": 777, "name": "x"}))
    check("compat voce vecchia", V.cache_get(ckpt) == 777)
    mb, src = V.estimate_mb(ckpt, working_mb=0)
    check("estimate_mb usa la cache", (mb, src) == (777, "cache"))

    # 5) put senza metadati NON cancella i metadati esistenti (merge per-campo)
    V.cache_put(ckpt, 1300, variant="fp8")
    e2 = V._entry(ckpt, "fp8")
    check("merge per-campo", e2.get("footprint_mb") == 1300 and e2.get("category") == "video")

    # 6) comp_* + migrazione legacy idempotente
    with open(V._LEGACY_FOOTPRINT, "w", encoding="utf-8") as fh:
        json.dump({"t5xxl": {"vram_mb": 9100.0, "dtype": "bf16", "category": "aux"}}, fh)
    n1 = V.migrate_legacy()
    n2 = V.migrate_legacy()                                  # file già rinominato → 0
    check("migrate_legacy importa", n1 == 1 and V.comp_get("t5xxl") == 9100)
    check("migrate_legacy idempotente", n2 == 0)

    # 7) CONCORRENZA: 4 processi × 25 put ciascuno → JSON valido e completo
    code = (
        "import os, sys; sys.path.insert(0, r'%s'); os.environ['LTX_VRAM_DB']=r'%s'\n"
        "from core import vram_estimate as V; V._CACHE=r'%s'\n"
        "wid = int(sys.argv[1])\n"
        "for i in range(25): V.comp_put(f'w{wid}_{i}', 10 + i, category='aux')\n"
        % (_REPO, db_path, db_path))
    procs = [subprocess.Popen([sys.executable, "-c", code, str(w)]) for w in range(4)]
    for p in procs:
        p.wait(timeout=120)
    with open(db_path, encoding="utf-8") as fh:
        final = json.load(fh)                                # non solleva = mai corrotto
    got = sum(1 for k in final if k.startswith("comp:w"))
    check(f"concorrenza: JSON valido, {got}/100 voci (≥90 con last-writer-wins)", got >= 90)

    print(f"\n{'TUTTO OK' if not fails else 'FALLITI: ' + ', '.join(fails)}")
    return len(fails)


if __name__ == "__main__":
    raise SystemExit(run())
