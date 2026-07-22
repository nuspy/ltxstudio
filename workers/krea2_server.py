"""Server IMMAGINI PERSISTENTE Krea 2 — gira nel venv `E:\\AI\\Tools\\lingbot_venv311` (torch 2.11, diffusers 0.39).

Tiene la pipeline Krea 2 (t2i + Krea 2 Edit) residente in `krea2_infer._PIPE_CACHE`: la PRIMA immagine legge
da disco e casta fp8, le successive riusano la pipe calda → NESSUN ricaricamento a ogni shot. È l'analogo di
`ltx2_server` (video) per l'immagine: senza, `img_gen_job` lancia un subprocess nuovo per OGNI foto → un
film/videomusicale da decine di shot = ore di reload. API stile inference_server (submit asincrono + polling),
porta 8191. Libera la VRAM solo su `/free` (quando un'altra pipeline la richiede).

Avvio:  <lingbot_venv311>/python -m workers.krea2_server   (cwd = repo)
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # per `import krea2_infer`

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import krea2_infer  # porta _PIPE_CACHE + run_edit/run_generation  # noqa: E402

app = FastAPI(title="LTX Studio — Native Image Server (Krea 2 / Krea 2 Edit)")
_jobs: dict[str, dict] = {}
_q: "queue.Queue[str]" = queue.Queue()


class SubmitReq(BaseModel):
    params: dict


def _is_edit(p: dict) -> bool:
    return str(p.get("arch", "")).lower() == "krea2_edit" or str(p.get("mode", "")).lower() == "edit"


def _worker() -> None:
    while True:
        jid = _q.get()
        job = _jobs.get(jid)
        if not job:
            _q.task_done()
            continue
        job["status"] = "running"
        try:
            p = job["params"]
            out = krea2_infer.run_edit(p) if _is_edit(p) else krea2_infer.run_generation(p)
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
    mode: str = "demote"                                    # "demote" (parcheggia in RAM) | "destroy"


@app.post("/free")
def free(req: FreeReq | None = None):
    """Libera la VRAM per un'altra pipeline (via mem_coordinator). Due modalità (unified-model-manager):
    - demote (default): i pipe vengono PARCHEGGIATI su CPU-RAM (rientro senza SSD se si torna a Krea2);
    - destroy: distrutti (VRAM e RAM all'OS — categorie effimere o «Libera VRAM»)."""
    import gc
    import torch
    mode = (req.mode if req else "demote").lower()
    n = len(krea2_infer._PIPE_CACHE)
    parked_mb = 0.0
    if mode == "destroy":
        try:
            krea2_infer._PIPE_CACHE.clear()
        except Exception:                                   # noqa: BLE001
            pass
    else:
        parked_mb = krea2_infer._demote_pipes()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[krea2] /free mode={mode}: {n} pipe, parcheggio RAM {parked_mb:.0f}MB "
          f"{'DISTRUTTO' if mode == 'destroy' else 'mantenuto'}", flush=True)
    return {"freed_pipelines": n if mode == "destroy" else 0, "mode": mode,
            "parked_ram_mb": round(parked_mb)}


@app.get("/system_stats")
def system_stats():
    import torch
    free, total = (torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0))
    return {"up": True, "cached_pipelines": len(krea2_infer._PIPE_CACHE),
            "vram_free_mb": free // 1024 // 1024, "vram_total_mb": total // 1024 // 1024}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8191)
