"""Salute del contesto CUDA di un processo figlio persistente.

Dopo un «CUDA error: out of memory» (o un device-side assert, un illegal memory access…) il contesto CUDA del
processo è AVVELENATO: ogni chiamata successiva fallisce con lo stesso errore e, peggio, i ripieghi «prudenti»
(ricarica, riassembla) lavorano su uno stato inconsistente. Incidente del 17/10/2026 (task 12): il primo OOM è
stato inghiottito da un try/except, il ripiego ha continuato nel processo avvelenato fino a ~90 GB di memoria
virtuale e al crash di sistema. Regola: un errore CUDA fatale NON si inghiotte e NON si ripara in-process: il
figlio si dichiara non sano ed esce; il server lo rigenera (server.py → `_PersistentChild`).

Solo stdlib: importabile da server, runner e backend.
"""
from __future__ import annotations

# frammenti (minuscoli) che indicano un contesto CUDA compromesso
_FATAL_MARKERS = (
    "cuda error",                 # «CUDA error: out of memory», «CUDA error: an illegal memory access…»
    "cuda out of memory",         # torch.cuda.OutOfMemoryError: «CUDA out of memory. Tried to allocate…»
    "device-side assert",
    "illegal memory access",
    "cublas_status_alloc_failed",
    "cudnn_status_not_initialized",
    "cuda driver error",
)


def is_cuda_fatal(exc: BaseException | str) -> bool:
    """True se l'eccezione (o il testo) indica un errore CUDA che compromette il processo.
    Un MemoryError di sistema / errore 1455 (commit) NON è CUDA: il processo resta sano."""
    if isinstance(exc, BaseException):
        if type(exc).__name__ == "OutOfMemoryError":        # torch.OutOfMemoryError / torch.cuda.OutOfMemoryError
            return True
        text = f"{type(exc).__name__}: {exc}"
    else:
        text = str(exc)
    t = text.lower()
    return any(m in t for m in _FATAL_MARKERS)


def reraise_if_fatal(exc: BaseException) -> None:
    """Da chiamare in OGNI `except Exception` che circonda codice CUDA: se l'errore è fatale lo rilancia
    (il figlio esce), altrimenti ritorna e il chiamante prosegue col suo ripiego."""
    if is_cuda_fatal(exc):
        raise exc
