"""measure_worker — misura REALE del footprint di un checkpoint IMMAGINE (F7 unified-model-manager).

CLI (venv inference_server):  python -m inference_server.measure_worker <model_path> [arch]

Carica la pipeline col loader VERO di image_flux (stessi dtype/quantizzazioni/upcast del runtime):
il percorso di finalize misura i moduli (byte reali dtype-aware) e scrive il footprint nel DB
canonico (core/vram_estimate, source="measured", breakdown per-componente). Poi esce: la VRAM
torna all'OS. Load commit-safe (safetensors no-mmap via core/st_direct).
Exit code: 0 = misurato; 2 = modello non caricabile (l'orchestratore lo marca e non ritenta).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    if len(sys.argv) < 2:
        print("uso: measure_worker <model_path> [arch]", flush=True)
        return 2
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"[measure] file inesistente: {path}", flush=True)
        return 2
    try:
        import faulthandler
        faulthandler.enable()
    except Exception:                                       # noqa: BLE001
        pass
    try:
        from core import st_direct
        if st_direct.patch_safetensors_torch():
            print("[measure] safetensors → I/O diretto (commit-safe)", flush=True)
    except Exception:                                       # noqa: BLE001
        pass
    from inference_server.backends import image_flux
    arch = sys.argv[2] if len(sys.argv) > 2 else ""
    if not arch:
        # SENZA arch esplicita: chiedi al DETECTOR (famiglia canonica), non a _arch_of — che da un
        # path nudo ricade su "flux" e caricava Z-Image/Ernie col loader FLUX (IndexError, exit 2).
        try:
            from core import model_detect
            fam = str((model_detect.get(path) or {}).get("family") or "")
            arch = {"flux1": "flux", "flux2": "flux2", "zimage": "zimage", "ernie": "ernie"}.get(fam, "")
        except Exception:                                   # noqa: BLE001
            arch = ""
    if not arch:
        try:
            arch = image_flux._arch_of({"model": path})     # noqa: SLF001
        except Exception:                                   # noqa: BLE001
            arch = "flux"
    print(f"[measure] carico {os.path.basename(path)} (arch={arch}) per la misura reale…", flush=True)
    try:
        pipe, vram_mb = image_flux._load(path, arch)        # noqa: SLF001 — finalize misura e scrive il DB
        print(f"[measure] OK: {os.path.basename(path)} = {vram_mb}MB (source=measured nel DB)", flush=True)
        del pipe
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                   # noqa: BLE001
            pass
        return 0
    except (MemoryError, OSError) as e:                     # transitorio (commit/RAM bassi) → exit 3 = RIPROVA
        print(f"[measure] memoria insufficiente ora ({type(e).__name__}: {e}) → riprovare più tardi",
              flush=True)
        return 3
    except Exception as e:                                  # noqa: BLE001 — strutturale (arch/pipeline) → skip
        _msg = str(e)
        if "out of memory" in _msg.lower() or "paging file" in _msg.lower() or "1455" in _msg:
            print(f"[measure] memoria insufficiente ora ({type(e).__name__}) → riprovare più tardi", flush=True)
            return 3
        print(f"[measure] FALLITO ({type(e).__name__}: {_msg[:300]})", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
