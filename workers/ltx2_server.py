"""Server VIDEO PERSISTENTE LTX-2.3 — gira nel venv `E:\\AI\\Tools\\ltx2_venv` (torch 2.9.1).

Tiene la pipeline LTX-2 in RAM (offload_mode=CPU → cache CPU dei pesi): la prima generazione
legge da disco, le successive riusano la cache → NESSUN ricaricamento da disco a ogni video.
API stile inference_server immagini (submit asincrono + polling). Porta 8190.

Avvio:  <ltx2_venv>/python -m workers.ltx2_server   (cwd = repo)
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # per `import ltx2_infer`

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import ltx2_infer  # applica i fix meta (mask + buffer) e la cache pipeline  # noqa: E402

app = FastAPI(title="LTX Studio — Native Video Server (LTX-2.3)")
_jobs: dict[str, dict] = {}
_q: "queue.Queue[str]" = queue.Queue()


class SubmitReq(BaseModel):
    params: dict


def _worker() -> None:
    while True:
        jid = _q.get()
        job = _jobs.get(jid)
        if not job:
            continue
        job["status"] = "running"
        try:
            out = ltx2_infer.run_generation(job["params"])
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


@app.post("/free")
def free():
    """M2 — libera la VRAM/RAM tenuta dalla pipeline video (chiamato quando la prossima pipeline è di un'ALTRA
    categoria: immagine/3D/mondi/musica). Prima MANCAVA → dopo un video la VRAM restava occupata a tempo
    indefinito e i job immagine successivi finivano in offload."""
    import gc
    import torch
    n = len(ltx2_infer._PIPE_CACHE)
    try:
        ltx2_infer._PIPE_CACHE.clear()
    except Exception:                                       # noqa: BLE001
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[ltx2] /free: {n} pipeline video scaricate", flush=True)
    return {"freed_pipelines": n}


@app.get("/system_stats")
def system_stats():
    import torch
    free, total = (torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0))
    return {"up": True, "cached_pipelines": len(ltx2_infer._PIPE_CACHE),
            "vram_free_mb": free // 1024 // 1024, "vram_total_mb": total // 1024 // 1024}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8190)
