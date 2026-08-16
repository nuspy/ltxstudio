"""Lock GPU inter-processo — «un solo job GPU pesante alla volta», per DAVVERO.

La coda seriale della GUI copre solo i job passati da MainWindow._enqueue: training, mondi 3D,
musica, i tab del Film Studio e l'API REST (:8200, JobManager separato) partivano IN PARALLELO a
un job pesante — due load concorrenti hanno già saturato RAM/VRAM e congelato il sistema (regola
ferrea in [[dynamic-vram-tuning]]). Questo modulo è il punto unico: un file-lock in data/gpu.lock
con pid+tag, condiviso da TUTTI i processi (GUI, API, servizi).

Proprietà:
  - RIENTRANTE per processo (refcount): un job in coda che internamente passa da un altro punto
    coordinato non si auto-blocca.
  - STALE-SAFE: se il pid proprietario è morto il lock viene rimosso e riacquisito (nessun lock
    orfano dopo un crash).
  - Best-effort ONESTO: acquire() blocca (poll) finché il lock non è libero; `on_wait` riceve il
    proprietario corrente per mostrare all'utente CHI tiene la GPU (statusbar/log).

API:  acquire(tag, timeout_s=None, on_wait=None) -> bool | release() | holder() | hold(tag) ctx.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

_LOCK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "gpu.lock")
_STATE = {"count": 0, "tag": ""}                        # rientranza per-processo


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:                                   # noqa: BLE001 — senza psutil: prudente, vivo
        return True


def holder() -> dict | None:
    """{pid, tag, ts} del proprietario corrente, None se libero/illeggibile."""
    try:
        with open(_LOCK_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:                                   # noqa: BLE001
        return None


def _try_take(tag: str) -> bool:
    os.makedirs(os.path.dirname(_LOCK_PATH), exist_ok=True)
    try:
        fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        h = holder()
        if h and not _pid_alive(int(h.get("pid", 0) or 0)):
            try:                                        # proprietario MORTO → lock orfano, rimuovi
                os.remove(_LOCK_PATH)
                print(f"[gpu-lock] lock orfano di pid {h.get('pid')} ({h.get('tag')}) rimosso",
                      flush=True)
            except OSError:
                pass
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "tag": tag, "ts": time.time()}, f)
    return True


def acquire(tag: str, *, timeout_s: float | None = None, poll_s: float = 1.0,
            on_wait=None, abort=None) -> bool:
    """Prende il lock GPU (bloccante). `on_wait(holder_dict)` viene chiamato ~ogni 5s di attesa
    (per statusbar/log); `abort()` -> True interrompe l'attesa (annullamento utente).
    timeout_s=None = attesa illimitata. Ritorna False su timeout/abort."""
    if _STATE["count"] > 0:                             # rientrante nello stesso processo
        _STATE["count"] += 1
        return True
    t0 = time.monotonic()
    last_cb = 0.0
    while not _try_take(tag):
        if timeout_s is not None and time.monotonic() - t0 >= timeout_s:
            return False
        if abort is not None:
            try:
                if abort():
                    return False
            except Exception:                           # noqa: BLE001
                pass
        now = time.monotonic()
        if on_wait and now - last_cb >= 5.0:
            last_cb = now
            try:
                on_wait(holder() or {})
            except Exception:                           # noqa: BLE001
                pass
        time.sleep(poll_s)
    _STATE["count"] = 1
    _STATE["tag"] = tag
    return True


def release() -> None:
    """Rilascia il lock (solo se posseduto da questo processo; gestisce la rientranza)."""
    if _STATE["count"] > 1:
        _STATE["count"] -= 1
        return
    if _STATE["count"] == 1:
        _STATE["count"] = 0
        h = holder()
        if h and int(h.get("pid", 0) or 0) == os.getpid():
            try:
                os.remove(_LOCK_PATH)
            except OSError:
                pass


@contextmanager
def hold(tag: str, *, timeout_s: float | None = None, on_wait=None):
    """Context manager: `with gpu_lock.hold("video"): …` — release garantito anche su eccezione."""
    ok = acquire(tag, timeout_s=timeout_s, on_wait=on_wait)
    try:
        yield ok
    finally:
        if ok:
            release()
