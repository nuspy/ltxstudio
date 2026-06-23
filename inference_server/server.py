"""InferenceServer — backend nativo persistente (sostituto di ComfyUI).

Processo separato (proprio venv torch+diffusers) avviato in background. Espone un'API HTTP
minima con la stessa semantica del driver ComfyUI (submit asincrono + polling history), una
coda SERIALE (un job alla volta) e un ModelManager condiviso per la VRAM.

Avvio:  <venv>/python -m inference_server.server   (porta 8189)
"""
from __future__ import annotations

import threading
import queue
import traceback
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from inference_server.model_manager import ModelManager
from inference_server.backends import image_flux
from inference_server.backends import video_ltx
from inference_server.backends import image_upscale
from inference_server.backends import image_segment
from inference_server.backends import video_postprocess

app = FastAPI(title="LTX Studio — Native Inference Server")
MM = ModelManager()
OUTPUT_DIR = r"E:\AI\ltxstudio\output"

_jobs: dict[str, dict] = {}
_q: "queue.Queue[str]" = queue.Queue()

# dispatch modulo → handler.run(params, mm, output_dir) → {status, output}
HANDLERS = {"image": image_flux.run, "video": video_ltx.run,
            "upscale": image_upscale.run, "segment": image_segment.run,
            "video_upscale": video_postprocess.run,
            "video_face": video_postprocess.run_face}


class SubmitReq(BaseModel):
    module: str
    params: dict


def _worker():
    while True:
        jid = _q.get()
        job = _jobs.get(jid)
        if not job:
            continue
        job["status"] = "running"
        try:
            handler = HANDLERS[job["module"]]
            res = handler(job["params"], MM, OUTPUT_DIR)
            job.update(res)
            job["status"] = res.get("status", "success")
        except Exception as e:                          # noqa: BLE001
            job["status"] = "error"
            job["error"] = f"{e}\n{traceback.format_exc()[-1500:]}"
        finally:
            _q.task_done()


threading.Thread(target=_worker, daemon=True).start()


@app.post("/submit")
def submit(req: SubmitReq):
    if req.module not in HANDLERS:
        return {"error": f"modulo non supportato: {req.module}"}
    jid = uuid.uuid4().hex
    _jobs[jid] = {"status": "pending", "module": req.module, "params": req.params, "output": []}
    _q.put(jid)
    return {"prompt_id": jid}


@app.get("/history/{jid}")
def history(jid: str):
    return _jobs.get(jid, {"status": "unknown"})


@app.get("/system_stats")
def system_stats():
    return MM.status()


@app.post("/free")
def free():
    MM.clear()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8189)
