"""job_runner — esegue UN job immagine/upscale/segment/video-postprocess in un PROCESSO ISOLATO e muore.

Perché: PyTorch + safetensors NON sono thread-safe dentro un server uvicorn di lunga durata su Windows.
Caricare un checkpoint (safetensors) mentre uvicorn serve richieste → access violation; caricare un
SECONDO modello nello stesso processo dopo il primo → segfault/MemoryError. Isolando ogni job in un
processo figlio che nasce, carica UN modello, genera e MUORE, entrambi i problemi spariscono alla radice:
niente uvicorn nel processo di calcolo, e ogni processo vede un solo caricamento. (È lo stesso modello del
worker VIDEO, che gira già come subprocess e non ha mai avuto questi crash.)

Uso:  python -m inference_server.job_runner <params.json> <result.json>
  - params.json: {"module": "image"|"upscale"|"segment"|..., "params": {...}}
  - result.json: vi scrive {"status": ..., "output": [...], "error": ...}
  - LTX2_PREVIEW_DIR (env): se impostato, preview_bus scrive preview.png/preview.json là (il server li serve).
"""
from __future__ import annotations

import json
import sys
import traceback


def main() -> int:
    params_path, result_path = sys.argv[1], sys.argv[2]
    with open(params_path, encoding="utf-8") as f:
        job = json.load(f)
    module = job.get("module", "image")
    params = job.get("params", {})

    # faulthandler: se anche il PROCESSO FIGLIO dovesse morire di crash nativo, lascia lo stack nel suo
    # stderr (raccolto dal server nel file di log del job) invece del silenzio.
    try:
        import faulthandler
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:                                       # noqa: BLE001
        pass

    # gli handler importano torch: import QUI (nel figlio), MAI nel server HTTP.
    import time as _t
    _tbtn = float(params.get("_t_button") or 0)
    if _tbtn:
        print(f"[step-child] +{_t.time()-_tbtn:.1f}s dal bottone | figlio AVVIATO (spawn+attesa coda inclusi)", flush=True)
    _ti = _t.perf_counter()
    from inference_server.model_manager import ModelManager
    from inference_server.backends import image_flux, image_upscale, image_segment, video_postprocess
    from core import optional_components as _oc                  # MagicQuill V2 = componente OPZIONALE (CC BY-NC, fuori dalle build)
    image_mq_run = _oc.image_mq_handler()                        # se manca: risponde «non installato», mai un crash
    from inference_server.backends import image_control
    print(f"[step-child] import torch/diffusers/backends = {_t.perf_counter()-_ti:.1f}s "
          f"(overhead figlio fresco per-job)", flush=True)
    handlers = {
        "image": image_flux.run,
        "image_mq": image_mq_run,
        "image_control": image_control.run,
        "upscale": image_upscale.run,
        "segment": image_segment.run,
        "video_upscale": video_postprocess.run,
        "video_face": video_postprocess.run_face,
        "video_fps": video_postprocess.run_fps,
    }
    output_dir = r"E:\AI\ltxstudio\output"
    mm = ModelManager()                                     # cache locale: viene distrutta all'uscita del processo

    try:
        handler = handlers.get(module)
        if handler is None:
            res = {"status": "error", "error": f"modulo non supportato: {module}"}
        else:
            res = handler(params, mm, output_dir)
    except Exception as e:                                  # noqa: BLE001
        res = {"status": "error", "error": f"{e}\n{traceback.format_exc()[-1500:]}"}

    tmp = result_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f)
    import os
    os.replace(tmp, result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
