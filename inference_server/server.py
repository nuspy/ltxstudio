"""InferenceServer — backend nativo (sostituto di ComfyUI).

Server HTTP leggero (uvicorn) con la stessa semantica del driver ComfyUI (submit asincrono + polling
history) e una coda SERIALE. **Non tocca MAI torch**: ogni job viene eseguito in un PROCESSO FIGLIO isolato
(`job_runner`) che carica il modello, genera e muore.

Perché l'isolamento: PyTorch + safetensors non sono thread-safe dentro un server uvicorn di lunga durata su
Windows. Caricare un checkpoint mentre uvicorn serve richieste → access violation; caricarne un SECONDO
nello stesso processo (cambio modello) → segfault/MemoryError. Con un processo per job — che nasce, carica
UN modello e muore — entrambi i problemi spariscono alla radice (è come già gira il worker VIDEO). Costo:
il modello si ricarica ad ogni job (niente cache tra job); in cambio, zero crash.

Avvio:  <venv>/python -m inference_server.server   (porta 8189)
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import uuid

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = r"E:\AI\ltxstudio\output"
_JOB_DIR = os.path.join(PROJECT, "output", ".jobs")          # params/result/log per-job (IPC via file)
_PREVIEW_DIR = os.path.join(PROJECT, "output", ".preview")   # preview.png/preview.json scritti dal figlio
os.makedirs(_JOB_DIR, exist_ok=True)
os.makedirs(_PREVIEW_DIR, exist_ok=True)


def _tee_log() -> None:
    """Duplica stdout/stderr del server su output/server.log (oltre alla console)."""
    try:
        logdir = os.path.join(PROJECT, "output")
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
            def isatty(self):
                return False
            def fileno(self):
                for s in self._s:
                    try: return s.fileno()
                    except Exception: pass
                raise OSError("no fileno")
            def __getattr__(self, name):
                for s in self._s:
                    if hasattr(s, name): return getattr(s, name)
                raise AttributeError(name)
        sys.stdout = _Tee(sys.__stdout__, f)
        sys.stderr = _Tee(sys.__stderr__, f)
        print(f"[server] log su {os.path.join(logdir, 'server.log')}", flush=True)
    except Exception:                                       # noqa: BLE001
        pass


_tee_log()

# il server LEGGE la preview dai file scritti dal processo figlio (get_preview usa questa env)
os.environ["LTX2_SERVER_PREVIEW_DIR"] = _PREVIEW_DIR

from fastapi import FastAPI, Response
from pydantic import BaseModel

from inference_server import preview_bus                    # leggero: nessun import torch

app = FastAPI(title="LTX Studio — Native Inference Server")

# moduli validi (l'esecuzione avviene in job_runner, non qui — nessun handler/torch importato nel server)
_MODULES = {"image", "upscale", "segment", "video_upscale", "video_face", "video_fps"}
_jobs: dict[str, dict] = {}
_q: "queue.Queue[str]" = queue.Queue()


class SubmitReq(BaseModel):
    module: str
    params: dict


def _tail(path: str, n: int = 2000) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()[-n:]
    except Exception:                                       # noqa: BLE001
        return ""


def _worker():
    """Coda seriale: per ogni job lancia un PROCESSO FIGLIO isolato (job_runner) e ne raccoglie il risultato.
    Il server non importa torch → non può crashare per il caricamento; se il figlio muore di crash nativo,
    lo rileviamo (result mancante + exit code) e restituiamo un errore chiaro col tail del suo log."""
    while True:
        jid = _q.get()
        job = _jobs.get(jid)
        if not job:
            _q.task_done()
            continue
        job["status"] = "running"
        preview_bus.reset()                                 # azzera l'anteprima del job precedente
        pf = os.path.join(_JOB_DIR, f"job_{jid}.json")
        rf = os.path.join(_JOB_DIR, f"res_{jid}.json")
        lf = os.path.join(_JOB_DIR, f"log_{jid}.txt")
        try:
            with open(pf, "w", encoding="utf-8") as f:
                json.dump({"module": job["module"], "params": job["params"]}, f)
            env = dict(os.environ)
            env["LTX2_PREVIEW_DIR"] = _PREVIEW_DIR          # il figlio scrive preview.png/json qui
            env.pop("LTX2_SERVER_PREVIEW_DIR", None)        # nel figlio preview_bus SCRIVE, non legge
            with open(lf, "w", encoding="utf-8", errors="replace") as logf:
                proc = subprocess.Popen(
                    [sys.executable, "-u", "-m", "inference_server.job_runner", pf, rf],
                    cwd=PROJECT, env=env, stdout=logf, stderr=logf)
                job["_pid"] = proc.pid
                proc.wait()
            if os.path.exists(rf):
                with open(rf, encoding="utf-8") as f:
                    res = json.load(f)
            else:                                           # figlio morto senza risultato = crash nativo
                res = {"status": "error",
                       "error": f"job '{job['module']}' terminato senza risultato (exit {proc.returncode}). "
                                f"Log:\n{_tail(lf)}"}
            job.update(res)
            job["status"] = res.get("status", "success")
        except Exception as e:                              # noqa: BLE001
            job["status"] = "error"
            job["error"] = str(e)
        finally:
            for p in (pf, rf):
                try: os.remove(p)
                except Exception: pass                      # noqa: BLE001
            _q.task_done()


threading.Thread(target=_worker, daemon=True).start()


@app.post("/submit")
def submit(req: SubmitReq):
    if req.module not in _MODULES:
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
    """Progresso + ultima anteprima per-step (letti dai file scritti dal processo figlio)."""
    import urllib.parse
    p = preview_bus.get_preview()
    return Response(content=p.get("png") or b"", media_type="image/png",
                    headers={"X-Preview-PngSeq": str(p["png_seq"]), "X-Preview-Step": str(p["step"]),
                             "X-Preview-Total": str(p["total"]),
                             "X-Preview-Msg": urllib.parse.quote(p.get("msg", ""))})


@app.get("/system_stats")
def system_stats():
    """VRAM via NVML/nvidia-smi — NIENTE torch.cuda nel server (nessun contesto CUDA, nessun crash)."""
    try:
        from core.gpu_mem import free_vram_mb
        free = free_vram_mb()
    except Exception:                                       # noqa: BLE001
        free = 0
    return {"vram_free_mb": free, "loaded": [], "isolated": True}


@app.post("/free")
def free():
    # niente da liberare: ogni job gira in un processo che muore da sé (VRAM/RAM rilasciate all'uscita).
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8189)
