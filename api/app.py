"""LTX Studio — REST API pubblica (FastAPI).

Espone come JOB ASINCRONI tutte le funzioni native: immagini, video, film, musica, 3D/rig/anim.
Ogni POST crea un job e ritorna `{job_id}`; `GET /jobs/{id}` dà stato/progress e, a fine job, i
file con i link di download. Coda seriale (un job GPU alla volta). 100% locale.

Avvio:  <venv>/python -m api.app     (porta 8200)
Swagger: http://127.0.0.1:8200/docs
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from api.config import HOST, PORT, is_allowed_path
from api.jobs import MANAGER
from api.download import files_payload

app = FastAPI(
    title="LTX Studio API",
    version="1.0.0",
    description="API locale per generazione/modifica di immagini, video, film, musica, modelli 3D, "
                "rig/skinning e animazioni. Job asincroni + link di download.",
)

# ── Auto-shutdown per inattività: se per IDLE_TIMEOUT non arrivano richieste E nessun job è in
#    coda/esecuzione, il server si spegne da solo (lo skill lo riavvia al bisogno). ──
IDLE_TIMEOUT = float(os.environ.get("LTX_API_IDLE_TIMEOUT", "1800"))   # 30 min
_last_activity = time.time()
_activity_lock = threading.Lock()


def _touch_activity():
    global _last_activity
    with _activity_lock:
        _last_activity = time.time()


@app.middleware("http")
async def _activity_mw(request: Request, call_next):
    _touch_activity()
    return await call_next(request)


def _idle_watchdog():
    interval = max(2.0, min(30.0, IDLE_TIMEOUT / 4.0)) if IDLE_TIMEOUT > 0 else 30.0
    while True:
        time.sleep(interval)
        if IDLE_TIMEOUT <= 0:
            continue
        with _activity_lock:
            idle = time.time() - _last_activity
        if idle >= IDLE_TIMEOUT and not MANAGER.busy():
            print(f"[api] inattività da {idle:.0f}s e nessun job → auto-shutdown.", flush=True)
            os._exit(0)


threading.Thread(target=_idle_watchdog, daemon=True).start()


# ─────────────────────────── infra: health / jobs / download ───────────────────────────
@app.get("/health", tags=["infra"])
def health():
    """Stato dell'API e del backend nativo (:8189)."""
    from core import native_driver
    return {"api": "ok", "native_backend_up": native_driver.is_up()}


@app.get("/jobs/{job_id}", tags=["infra"])
def job_status(job_id: str):
    """Stato di un job. A `success` include `files[]` con i link di download."""
    job = MANAGER.get(job_id)
    if not job:
        raise HTTPException(404, f"job sconosciuto: {job_id}")
    out = {
        "job_id": job.id, "kind": job.kind, "status": job.status,
        "progress": round(job.progress, 3), "message": job.message,
        "queue_position": MANAGER.queue_position(job_id) if job.status == "pending" else 0,
    }
    if job.status == "success":
        out["files"] = files_payload(job.outputs)
    if job.status == "error":
        out["error"] = job.error
    return out


@app.get("/download", tags=["infra"])
def download(path: str = Query(..., description="path locale del file da scaricare")):
    """Scarica un file prodotto (solo dentro le cartelle output/upload)."""
    if not is_allowed_path(path) or not Path(path).is_file():
        raise HTTPException(404, "file non trovato o non consentito")
    return FileResponse(path, filename=Path(path).name)


@app.post("/files/upload", tags=["infra"])
async def files_upload(file: UploadFile):
    """Carica un file (immagine/video/audio/mesh) sul server; ritorna il path da riusare nei job."""
    from api.download import save_upload, file_entry
    return file_entry(save_upload(file))


# ─────────────────────────── capability routers ───────────────────────────
# Ogni gruppo registra i propri endpoint POST (creano job) su questo `app`.
from api.handlers import images, video, music, threed, film   # noqa: E402
for _mod in (images, video, music, threed, film):
    if hasattr(_mod, "register"):
        _mod.register(app)


def main():
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
