"""JobManager — coda SERIALE di job asincroni per la REST API.

Un solo job pesante alla volta (rispetta il vincolo GPU: job concorrenti congelano il sistema,
vedi [[dynamic-vram-tuning]]). Ogni handler è una funzione `fn(job) -> list[str]` che ritorna i
path dei file prodotti; può aggiornare `job.progress`/`job.message` durante l'esecuzione.

Stato del job esposto via API: pending → running → success | error.
"""
from __future__ import annotations

import threading
import queue
import time
import traceback
import contextvars
import uuid
from dataclasses import dataclass, field
from typing import Callable


# Task 21: ?embed_metadata=false su una richiesta spegne i metadati di generazione nei file di QUEL job
# (impostato dal middleware di api/app.py, letto da JobManager.submit nello stesso contesto della richiesta).
REQUEST_EMBED: "contextvars.ContextVar[bool | None]" = contextvars.ContextVar("ltx_request_embed", default=None)


@dataclass
class Job:
    id: str
    kind: str                                   # es. "image.generate", "video.generate"
    params: dict = field(default_factory=dict)
    status: str = "pending"                     # pending | running | success | error
    progress: float = 0.0                       # 0.0–1.0 (best-effort)
    message: str = ""
    outputs: list[str] = field(default_factory=list)   # path locali dei file prodotti
    error: str = ""
    warnings: list[str] = field(default_factory=list)  # avvisi non bloccanti (es. lingua testo ≠ lingua voce)
    info: dict = field(default_factory=dict)           # dettagli del risultato (es. lingue usate per speaker)
    embed_metadata: bool | None = None                 # None = preferenza utente; False = per-richiesta "no metadati"
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def touch(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.updated = time.time()

    def log(self, msg: str, progress: float | None = None):
        self.message = msg
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        self.updated = time.time()


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._q: "queue.Queue[str]" = queue.Queue()
        self._fns: dict[str, Callable[[Job], list[str]]] = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, kind: str, fn: "Callable[[Job], list[str]]", params: dict | None = None) -> str:
        jid = uuid.uuid4().hex
        with self._lock:
            self._jobs[jid] = Job(id=jid, kind=kind, params=params or {}, embed_metadata=REQUEST_EMBED.get())
            self._fns[jid] = fn
        self._q.put(jid)
        return jid

    def get(self, jid: str) -> Job | None:
        return self._jobs.get(jid)

    def busy(self) -> bool:
        """True se c'è almeno un job in coda o in esecuzione (blocca l'auto-shutdown idle)."""
        return any(j.status in ("pending", "running") for j in self._jobs.values())

    def queue_position(self, jid: str) -> int:
        """Quanti job pending precedono questo (0 = prossimo/in esecuzione)."""
        pend = [j for j in self._jobs.values() if j.status == "pending"]
        pend.sort(key=lambda j: j.created)
        ids = [j.id for j in pend]
        return ids.index(jid) if jid in ids else 0

    def _worker(self):
        while True:
            jid = self._q.get()
            job = self._jobs.get(jid)
            fn = self._fns.get(jid)
            if not job or not fn:
                self._q.task_done(); continue
            # LOCK GPU INTER-PROCESSO: questa coda è seriale ma è INDIPENDENTE dalla coda GUI —
            # un job API e un job GUI potevano girare in parallelo (due load pesanti = freeze di
            # sistema già visto). Attesa bloccante nel worker thread (mai nell'handler HTTP).
            _locked = False
            try:
                from core import gpu_lock
                _locked = gpu_lock.acquire(
                    f"api:{job.kind}",
                    on_wait=lambda h: job.log(f"in attesa: GPU occupata da {h.get('tag', '?')}"))
            except Exception:                               # noqa: BLE001 — senza lock: come prima
                pass
            job.touch(status="running", progress=0.01, message="avvio…")
            try:
                from core import gen_metadata
                with gen_metadata.override(job.embed_metadata):
                    gen_metadata.reset_last()
                    outs = fn(job) or []
                    # metadati di generazione nel file FINALE: dopo qualunque passo che riscriva il file
                    gen_metadata.finalize_outputs([str(o) for o in outs if o], job.kind)
                job.outputs = [str(o) for o in outs if o]
                if job.outputs:
                    job.touch(status="success", progress=1.0, message="completato")
                else:
                    job.touch(status="error", error="nessun file prodotto", message="errore")
            except Exception as e:                          # noqa: BLE001
                job.touch(status="error", error=f"{e}\n{traceback.format_exc()[-1500:]}", message="errore")
            finally:
                if _locked:
                    try:
                        from core import gpu_lock
                        gpu_lock.release()
                    except Exception:                       # noqa: BLE001
                        pass
                self._fns.pop(jid, None)
                self._q.task_done()


# singleton condiviso dall'app
MANAGER = JobManager()
