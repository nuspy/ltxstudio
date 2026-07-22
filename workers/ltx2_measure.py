"""ltx2_measure — misura REALE del footprint di un checkpoint VIDEO LTX (F7 unified-model-manager).

CLI (venv ltx2):  python workers/ltx2_measure.py <ckpt.safetensors>

Materializza i tensori di OGNI componente del bundle col loader VERO (no-mmap, stessi dtype che
finiscono in VRAM: fp8 resta 1 byte con scaled-mm, bf16 2 byte) e ne somma i byte reali — un
componente alla volta, poi liberato (picco commit = un componente, mai il bundle intero). I byte
materializzati sono identici a quelli che occuperebbero la VRAM: è il "peso reale definitivo".
Scrive nel DB canonico: footprint totale + transformer_mb + breakdown, source="measured".
Exit: 0 = misurato; 2 = fallito.
"""
from __future__ import annotations

import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# raggruppa le chiavi top-level del bundle nei componenti del planner
_COMP_OF = {"model": "transformer", "vae": "vae", "video_vae": "vae", "audio_vae": "audio_vae",
            "vocoder": "vocoder", "text_embedding_projection": "text_projection"}


def main() -> int:
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        print("uso: ltx2_measure <ckpt.safetensors>", flush=True)
        return 2
    path = sys.argv[1]
    try:
        import faulthandler
        faulthandler.enable()
    except Exception:                                       # noqa: BLE001
        pass
    import torch  # noqa: F401 — serve a frombuffer dtype fp8
    from core import st_direct, vram_estimate

    try:
        hdr, base = st_direct.read_header(path)
    except Exception as e:                                  # noqa: BLE001
        print(f"[measure] header illeggibile ({e})", flush=True)
        return 2
    dt = st_direct._dtypes()                                # noqa: SLF001
    comps: dict = {}
    fp8 = False
    try:
        with open(path, "rb") as fh:
            for name, info in hdr.items():
                if name == "__metadata__":
                    continue
                comp = _COMP_OF.get(name.split(".")[0], name.split(".")[0])
                t = st_direct.read_tensor(fh, base, info, dt)   # materializza DAVVERO i byte
                comps[comp] = comps.get(comp, 0.0) + t.nbytes / 2**20
                fp8 = fp8 or str(info.get("dtype", "")).startswith("F8")
                del t                                            # picco commit = un tensore
        gc.collect()
    except (MemoryError, OSError) as e:                     # transitorio (commit basso) → exit 3 = RIPROVA
        print(f"[measure] memoria insufficiente ora ({type(e).__name__}) → riprovare più tardi", flush=True)
        return 3
    total = int(sum(comps.values()))
    tr = int(comps.get("transformer", 0))
    # categoria: bundle con transformer → video (LTX); file senza transformer (VAE/TE/adapter/shard
    # misurati qui come materializzatore CPU GENERICO) → aux. I checkpoint IMMAGINE veri passano
    # dal measure_worker del server (misura la pipeline caricata, con upcast/quant reali).
    _cat = "video" if tr > 0 else "aux"
    vram_estimate.cache_put(path, total, variant=("fp8" if fp8 else ""), source="measured",
                            category=_cat, dtype=("fp8" if fp8 else ""),
                            transformer_mb=tr, components={k: int(v) for k, v in comps.items()})
    if _cat == "video":
        for k, v in sorted(comps.items(), key=lambda x: -x[1]):
            vram_estimate.comp_put(f"ltx2:{k}", v, category=("video" if k == "transformer" else "aux"))
    print(f"[measure] OK: {os.path.basename(path)} = {total}MB (transformer {tr}MB) → DB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
