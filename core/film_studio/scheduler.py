"""Job Scheduler dell'AI Film Studio (spec §13) — coda con priorità/stato/retry/log, non blocca la UI.

Le generazioni pesanti (immagini, video, training LoRA, audio) passano da qui. Persistenza su DB
(tabella job). Esecuzione in un thread worker di background; multi-GPU = predisposto (round-robin su
N worker), per ora seriale sul server nativo (single-queue). Stima costo/tempo prima del lancio.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from queue import PriorityQueue

from core.film_studio import db

# stime grezze (s) per il preventivo "Genera Tutti"
EST_SECONDS = {"image": 25, "video": 90, "lora_train": 1800, "audio_voice": 20,
               "audio_music": 120, "audio_sfx": 30, "lipsync": 60}

_q: "PriorityQueue" = PriorityQueue()
_handlers: dict = {}            # type -> callable(params) -> file
_thread: threading.Thread | None = None
_listeners: list = []          # callback(event: dict)
_running = False


def register(job_type: str, fn) -> None:
    _handlers[job_type] = fn


def on_event(cb) -> None:
    _listeners.append(cb)


def _emit(ev: dict) -> None:
    for cb in list(_listeners):
        try:
            cb(ev)
        except Exception:                                   # noqa: BLE001
            pass


def estimate(jobs: list[dict]) -> dict:
    """Preventivo per una lista di job: {count, seconds, by_type}."""
    by = {}
    for j in jobs:
        by[j["type"]] = by.get(j["type"], 0) + EST_SECONDS.get(j["type"], 60)
    return {"count": len(jobs), "seconds": sum(by.values()), "by_type": by}


def enqueue(job_type: str, params: dict, *, priority: int = 5, owner_kind: str = "", owner_id: int = 0) -> int:
    p = dict(params)
    p["_owner_kind"], p["_owner_id"] = owner_kind, owner_id   # tracciati per lo snapshot dei take a fine job
    jid = db.insert("job", type=job_type, status="pending", priority=priority, gpu=0,
                    params=json.dumps(p, ensure_ascii=False), est=json.dumps({}),
                    artifact_id=0, log="", updated=time.time())
    _q.put((priority, jid))
    _emit({"event": "enqueued", "job_id": jid, "type": job_type})
    _ensure_worker()
    return jid


def _ensure_worker() -> None:
    global _thread, _running
    if _thread and _thread.is_alive():
        return
    _running = True
    _thread = threading.Thread(target=_loop, daemon=True)
    _thread.start()


def _loop() -> None:
    global _running
    while _running:
        try:
            _, jid = _q.get(timeout=1.0)
        except Exception:                                   # noqa: BLE001
            if _q.empty():
                continue
            continue
        job = db.get("job", jid)
        if not job or job["status"] not in ("pending", "retry"):
            continue
        db.update("job", jid, status="running", updated=time.time())
        _emit({"event": "started", "job_id": jid, "type": job["type"]})
        try:
            fn = _handlers.get(job["type"])
            if not fn:
                raise RuntimeError(f"nessun handler per job '{job['type']}'")
            out = fn(json.loads(job["params"]))
            db.update("job", jid, status="done", log=str(out)[:500], updated=time.time())
            _emit({"event": "done", "job_id": jid, "type": job["type"], "output": out})
        except Exception as e:                              # noqa: BLE001
            db.update("job", jid, status="error", log=(str(e) + "\n" + traceback.format_exc())[:1000],
                      updated=time.time())
            _emit({"event": "error", "job_id": jid, "type": job["type"], "error": str(e)})


def retry(jid: int) -> None:
    job = db.get("job", jid)
    if job:
        db.update("job", jid, status="retry", updated=time.time())
        _q.put((job["priority"], jid))
        _ensure_worker()


def recover() -> int:
    """RECOVERY post-crash (B10): i job rimasti 'running' (processo morto durante l'esecuzione) vengono
    ri-accodati come 'pending'; i 'pending'/'retry' persistiti vengono ri-messi in coda. Da chiamare all'avvio
    del Film Studio. Ritorna il numero di job recuperati."""
    n = 0
    for j in db.query("job", "status IN ('running','pending','retry')", order="priority"):
        db.update("job", j["id"], status="pending", updated=time.time())
        _q.put((j.get("priority", 5), j["id"]))
        n += 1
    if n:
        _emit({"event": "recovered", "count": n})
        _ensure_worker()
    return n


def pending() -> list[dict]:
    return db.query("job", "status IN ('pending','running','retry')", order="priority")


def stop() -> None:
    global _running
    _running = False
