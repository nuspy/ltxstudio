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

from fastapi import FastAPI, Response
from pydantic import BaseModel

from inference_server import preview_bus
from inference_server.model_manager import ModelManager
from inference_server.backends import image_flux
from inference_server.backends import image_upscale
from inference_server.backends import image_segment
from inference_server.backends import video_postprocess
# NB: il VIDEO gira via workers/ltx2_infer.py (ltx-core ufficiale, subprocess), NON da qui.
# Il vecchio path diffusers `video_ltx` (modulo "video"/"video_edit"/"video_long") era codice morto
# (non caricava il fp8 scaled) → rimosso. Qui restano solo i post-process video.

app = FastAPI(title="LTX Studio — Native Inference Server")
MM = ModelManager()
OUTPUT_DIR = r"E:\AI\ltxstudio\output"

_jobs: dict[str, dict] = {}
_q: "queue.Queue[str]" = queue.Queue()

# dispatch modulo → handler.run(params, mm, output_dir) → {status, output}
HANDLERS = {"image": image_flux.run,
            "upscale": image_upscale.run, "segment": image_segment.run,
            "video_upscale": video_postprocess.run,
            "video_face": video_postprocess.run_face,
            "video_fps": video_postprocess.run_fps}


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
        preview_bus.reset()                                 # azzera l'anteprima del job precedente
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


@app.get("/preview")
def preview():
    """Progresso + ultima anteprima per-step del job in corso. Header X-Preview-Step/Total (SEMPRE,
    barra credibile) + X-Preview-PngSeq (cresce quando c'è un nuovo PNG). Body = PNG (vuoto se non
    ancora disponibile ma il progresso è comunque negli header → 200 con body vuoto)."""
    p = preview_bus.get_preview()
    return Response(content=p.get("png") or b"", media_type="image/png",
                    headers={"X-Preview-PngSeq": str(p["png_seq"]), "X-Preview-Step": str(p["step"]),
                             "X-Preview-Total": str(p["total"])})


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
