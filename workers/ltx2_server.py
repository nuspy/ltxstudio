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
_last_activity = [__import__("time").monotonic()]       # auto-shutdown per inattività (vedi _idle_watch)


def _idle_watch() -> None:
    """AUTO-SPEGNIMENTO per inattività (richiesta utente: il contesto CUDA del server ~0.6GB non
    deve restare in VRAM per sempre). Dopo LTX2_SERVER_IDLE_S secondi senza job (default 30 min,
    0 = mai) il processo esce: VRAM e parcheggio RAM tornano all'OS. La GUI lo rilancia da sola
    al prossimo video (ensure_server): si paga solo ~15s di riavvio dopo lunghe pause."""
    import time as _t
    idle_s = float(os.environ.get("LTX2_SERVER_IDLE_S", "1800") or 0)
    if idle_s <= 0:
        return
    while True:
        _t.sleep(60)
        busy = any(j.get("status") in ("pending", "running") for j in _jobs.values())
        if not busy and _t.monotonic() - _last_activity[0] > idle_s:
            print(f"[ltx2] inattivo da {idle_s/60:.0f} min → auto-spegnimento (VRAM/RAM all'OS)",
                  flush=True)
            os._exit(0)


class SubmitReq(BaseModel):
    params: dict


def _worker() -> None:
    # ⚠ CRITICO: `torch.set_grad_enabled(False)` di ltx2_infer (import-time) vale SOLO per il thread
    # che importa (il main di uvicorn). Il grad-mode di torch è PER-THREAD: senza questa riga, qui
    # l'autograd era ATTIVO → l'encode Gemma + il denoise trattenevano il grafo (~17GB di attivazioni
    # live) → 40GB "allocati" su 32 → spill in shared WDDM → sistema saturo (visto live 22/07: OOM
    # a 40.4GiB e crash a cascata). Stessa malattia documentata in testa a ltx2_infer, per-thread.
    import torch
    torch.set_grad_enabled(False)
    while True:
        jid = _q.get()
        job = _jobs.get(jid)
        if not job:
            continue
        if job.get("status") == "cancelled":                # annullato mentre era in coda → salta
            _q.task_done()
            continue
        job["status"] = "running"
        ltx2_infer.clear_cancel()                           # top-level: nessun cancel residuo dal job prima
        try:
            from core import preview_file
            preview_file.reset()                            # preview pulita per il nuovo job
        except Exception:                                   # noqa: BLE001
            pass
        try:
            with torch.no_grad():                       # cintura+bretelle col set_grad_enabled sopra
                out = ltx2_infer.run_generation(job["params"])
            job["output"] = [out]
            job["status"] = "success"
        except ltx2_infer.GenerationCancelled:              # cancel cooperativo (POST /cancel)
            job["status"] = "cancelled"
            print(f"[ltx2] job {jid[:8]} ANNULLATO dall'utente", flush=True)
        except Exception as e:                              # noqa: BLE001
            job["status"] = "error"
            job["error"] = f"{e}\n{traceback.format_exc()[-1500:]}"
        finally:
            # consistenza post-job (coda SERIALE): i latenti intermedi vanno via, _PIPE_CACHE e il
            # parcheggio RAM (_PARK) restano — il job successivo riusa la pipeline senza reload.
            try:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:                               # noqa: BLE001
                pass
            _last_activity[0] = __import__("time").monotonic()
            _q.task_done()


threading.Thread(target=_worker, daemon=True).start()
threading.Thread(target=_idle_watch, daemon=True).start()


@app.post("/submit")
def submit(req: SubmitReq):
    jid = uuid.uuid4().hex
    _jobs[jid] = {"status": "pending", "params": req.params, "output": []}
    _last_activity[0] = __import__("time").monotonic()
    _q.put(jid)
    return {"prompt_id": jid}


@app.get("/history/{jid}")
def history(jid: str):
    d = dict(_jobs.get(jid, {"status": "unknown"}))
    d.pop("params", None)                                   # non ritornare i params (grossi/inutili al client)
    if d.get("status") == "running":
        try:                                                # progress: il server è LOCALE → embedda il canale
            from core import preview_file                   # a file di ltx2_infer (msg + step/total)
            d["progress"] = preview_file.read() or {}
        except Exception:                                   # noqa: BLE001
            pass
    return d


@app.post("/cancel/{jid}")
def cancel(jid: str):
    """Annulla un job: in coda → marcato 'cancelled' (il worker lo salta); in esecuzione →
    cancel COOPERATIVO (ltx2_infer controlla tra gli step del denoise e ai confini di fase;
    NON è istantaneo durante il decode VAE). Il server e le sue cache restano vivi e consistenti."""
    job = _jobs.get(jid)
    if job is None:
        return {"status": "not_found"}
    st = job.get("status")
    if st == "pending":
        job["status"] = "cancelled"
        return {"status": "cancelled"}
    if st == "running":
        ltx2_infer.request_cancel()
        return {"status": "cancelling"}
    return {"status": "done"}


class FreeReq(BaseModel):
    mode: str = "demote"                                    # "demote" (parcheggia in RAM) | "destroy"


@app.post("/free")
def free(req: FreeReq | None = None):
    """M2/M-park — libera la VRAM per un'altra pipeline. Due modalità (unified-model-manager):
    - demote (default): svuota la VRAM ma TIENE il parcheggio RAM (_PARK) e la _PIPE_CACHE →
      se la pipeline successiva torna al video, i pesi rientrano dalla RAM (niente SSD).
    - destroy: come prima — distrugge pipeline E parcheggio (chi chiama vuole anche la RAM:
      categorie effimere 3D/mondi/musica, o il bottone «Libera VRAM»)."""
    import gc
    import torch
    mode = (req.mode if req else "demote").lower()
    n = len(ltx2_infer._PIPE_CACHE)
    parked = 0.0
    if mode == "destroy":
        try:
            ltx2_infer._PIPE_CACHE.clear()
            ltx2_infer._PARK.clear()
        except Exception:                                   # noqa: BLE001
            pass
    else:                                                   # demote: VRAM sì, RAM no
        try:
            parked = sum(i["mb"] for i in ltx2_infer._PARK._info.values())
        except Exception:                                   # noqa: BLE001
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[ltx2] /free mode={mode}: {n} pipeline, parcheggio RAM {parked:.0f}MB "
          f"{'DISTRUTTO' if mode == 'destroy' else 'mantenuto'}", flush=True)
    return {"freed_pipelines": n if mode == "destroy" else 0, "mode": mode,
            "parked_ram_mb": round(parked)}


@app.get("/system_stats")
def system_stats():
    import os
    import torch
    free, total = (torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0))
    # model_key per il gpu_registry unico: video:basename(ckpt) del pipe residente più recente (la chiave
    # cache è (ckpt, qmode, offload, loras, stages, ptype)). '-' se nessuno.
    model_key = "-"
    try:
        cache = ltx2_infer._PIPE_CACHE
        if cache:
            ckpt = list(cache.keys())[-1][0]
            model_key = f"video:{os.path.basename(str(ckpt)) or '-'}"
    except Exception:                                       # noqa: BLE001
        pass
    used_mb = (total - free) // 1024 // 1024
    try:                                                    # M-park: lista modelli parcheggiati in RAM
        parked = ltx2_infer._PARK.stats()
    except Exception:                                       # noqa: BLE001
        parked = []
    return {"up": True, "cached_pipelines": len(ltx2_infer._PIPE_CACHE), "model_key": model_key,
            "vram_used_mb": used_mb, "model_vram_mb": used_mb,
            "vram_free_mb": free // 1024 // 1024, "vram_total_mb": total // 1024 // 1024,
            "parked": parked, "parked_ram_mb": round(sum(p["mb"] for p in parked))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8190)
