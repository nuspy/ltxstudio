"""InferenceServer — backend nativo persistente (sostituto di ComfyUI).

Processo separato (proprio venv torch+diffusers) avviato in background. Espone un'API HTTP
minima con la stessa semantica del driver ComfyUI (submit asincrono + polling history), una
coda SERIALE (un job alla volta) e un ModelManager condiviso per la VRAM.

Avvio:  <venv>/python -m inference_server.server   (porta 8189)
"""
from __future__ import annotations

import os
import sys
import threading
import queue
import traceback
import uuid


def _tee_log() -> None:
    """Duplica stdout/stderr del server su output/server.log (oltre alla console) → c'è SEMPRE un log da
    consultare anche quando il server gira in background/detached e l'utente non vede la console."""
    try:
        logdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        os.makedirs(logdir, exist_ok=True)
        f = open(os.path.join(logdir, "server.log"), "a", buffering=1, encoding="utf-8", errors="replace")

        class _Tee:
            def __init__(self, *streams): self._s = [s for s in streams if s]
            def write(self, d):
                for s in self._s:
                    try: s.write(d)
                    except Exception: pass
            def flush(self):
                for s in self._s:
                    try: s.flush()
                    except Exception: pass
            def isatty(self):                              # uvicorn chiama sys.stdout.isatty() → NON è un tty
                return False
            def fileno(self):
                for s in self._s:
                    try: return s.fileno()
                    except Exception: pass
                raise OSError("no fileno")
            def __getattr__(self, name):                   # delega encoding/errors/… al primo stream reale
                for s in self._s:
                    if hasattr(s, name): return getattr(s, name)
                raise AttributeError(name)
        sys.stdout = _Tee(sys.__stdout__, f)
        sys.stderr = _Tee(sys.__stderr__, f)
        print(f"[server] log su {os.path.join(logdir, 'server.log')}", flush=True)
    except Exception:                                       # noqa: BLE001
        pass


_tee_log()

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
        preview_bus.set_message("Caricamento componenti…")  # fase iniziale (finché il backend non aggiorna)
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
    import urllib.parse
    p = preview_bus.get_preview()
    return Response(content=p.get("png") or b"", media_type="image/png",
                    headers={"X-Preview-PngSeq": str(p["png_seq"]), "X-Preview-Step": str(p["step"]),
                             "X-Preview-Total": str(p["total"]),
                             "X-Preview-Msg": urllib.parse.quote(p.get("msg", ""))})


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
