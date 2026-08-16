"""Server VIDEO PERSISTENTE Krea Realtime 14B — gira nel venv `E:\\AI\\Tools\\krea_rt_venv`.

Tiene la WanModularPipeline residente + il `torch.compile(max-autotune)` di flex_attention SCALDATO tra i
clip: il primo clip paga il warmup, dal secondo in poi è REALTIME (≈1s video in ~2s). È il modo per avere
la perf realtime di krea_rt, che un subprocess-per-clip non può dare (ripaga il warmup ogni volta).

Modellato su `ltx2_server` (:8190), che gira stabile da mesi con lo stesso pattern (i server VIDEO
persistono bene, a differenza del figlio immagine fp8). API: submit asincrono + polling history. Porta 8192.

Avvio:  <krea_rt_venv>/python -m workers.krea_rt_server   (cwd = repo)
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # per `import krea_rt_infer`

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import krea_rt_infer  # porta _PIPE (cache pipeline) + run_generation  # noqa: E402

app = FastAPI(title="LTX Studio — Native Video Server (Krea Realtime 14B)")
_jobs: dict[str, dict] = {}
_q: "queue.Queue[str]" = queue.Queue()


class SubmitReq(BaseModel):
    params: dict


def _worker() -> None:
    while True:
        jid = _q.get()
        job = _jobs.get(jid)
        if not job:
            _q.task_done()
            continue
        job["status"] = "running"
        try:
            out = krea_rt_infer.run_generation(job["params"])
            job["output"] = [out]
            job["status"] = "success"
        except Exception as e:                              # noqa: BLE001
            job["status"] = "error"
            job["error"] = f"{e}\n{traceback.format_exc()[-1500:]}"
        finally:
            _q.task_done()


threading.Thread(target=_worker, daemon=True).start()


@app.post("/submit")
def submit(req: SubmitReq):
    jid = uuid.uuid4().hex
    _jobs[jid] = {"status": "pending", "params": req.params, "output": []}
    _q.put(jid)
    return {"prompt_id": jid}


@app.get("/history/{jid}")
def history(jid: str):
    return _jobs.get(jid, {"status": "unknown"})


class FreeReq(BaseModel):
    mode: str = "demote"


@app.post("/free")
def free(req: FreeReq | None = None):
    """Libera la VRAM del pipe krea_rt (chiamato dal registro quando un'ALTRA categoria/modello la richiede).
    NB unified-model-manager: qui il campo `mode` è accettato per uniformità ma demote = destroy —
    il pipe autoregressivo tiene KV-cache/stato che non si sposta in modo pulito su CPU, e il modello
    è piccolo (1.3B, reload ~5s dall'SSD): il parcheggio RAM non vale il rischio."""
    import gc
    import torch
    mode = (req.mode if req else "demote").lower()
    n = len(krea_rt_infer._PIPE)
    try:
        krea_rt_infer._PIPE.clear()
    except Exception:                                       # noqa: BLE001
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[krea_rt] /free (mode={mode}, sempre destroy): {n} pipeline scaricate", flush=True)
    return {"freed_pipelines": n, "mode": mode}


@app.get("/system_stats")
def system_stats():
    import torch
    free, total = (torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0))
    # VRAM PROPRIA (allocator di questo processo), non total-free: quel numero includeva gli altri
    # motori e falsava il projected_free del planner.
    own_mb = int(torch.cuda.memory_reserved() / 2**20) if torch.cuda.is_available() else 0
    model_key = "video:krea_rt" if krea_rt_infer._PIPE else "-"   # per il gpu_registry unico
    return {"up": True, "cached_pipelines": len(krea_rt_infer._PIPE), "model_key": model_key,
            "vram_used_mb": own_mb, "model_vram_mb": own_mb,
            "vram_free_mb": free // 1024 // 1024, "vram_total_mb": total // 1024 // 1024}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8192)
