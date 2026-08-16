"""Lettura VRAM SENZA creare un contesto CUDA.

CRITICO: nel processo dell'APP (GUI) non si deve MAI chiamare `torch.cuda.*` — anche solo
`torch.cuda.is_available()` o `mem_get_info()` inizializza un CONTESTO CUDA che resta vivo per tutta la
sessione. Quel contesto convive con il subprocess CUDA della generazione 3D (TRELLIS) e su RTX 5090/Blackwell
manda in crash nativo il subprocess (il backend DA SOLO funziva, dall'app moriva). Qui leggiamo la memoria via
NVML (pynvml) o, in fallback, nvidia-smi: entrambi NON creano contesti CUDA.
"""
from __future__ import annotations

import subprocess

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def free_vram_mb() -> int:
    """VRAM libera (MB), 0 se non rilevabile. Nessun contesto CUDA."""
    # 1) pynvml se presente (più veloce, niente processo)
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            return int(pynvml.nvmlDeviceGetMemoryInfo(h).free / (1024 * 1024))
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        pass
    # 2) nvidia-smi (sempre disponibile col driver) — finestra nascosta
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, creationflags=_CREATE_NO_WINDOW)
        return int(float(out.stdout.strip().splitlines()[0]))
    except Exception:  # noqa: BLE001
        return 0


def free_vram_gb() -> float:
    """VRAM libera (GB), 0.0 se non rilevabile. Nessun contesto CUDA."""
    return free_vram_mb() / 1024.0


def wait_free_vram_mb(min_free_mb: float, timeout_s: int = 40, poll_s: float = 1.0,
                      tag: str = "") -> int:
    """Attende il rilascio EFFETTIVO della VRAM: poll NVML/nvidia-smi finché `free ≥ min_free_mb`
    o scade il timeout. È il tassello che mancava tra "/free inviato" e "carico il modello": il
    rilascio driver è ASINCRONO rispetto alla risposta HTTP — se il load parte nel buco, CUDA
    alloca in shared-RAM (WDDM) e ci resta (10-20× più lento). Nessun contesto CUDA (mai torch
    nel processo app). La soglia è clampata alla VRAM fisica −2GB: mai irraggiungibile per design.
    Ritorna la VRAM libera all'uscita (il chiamante può loggare/decidere)."""
    import time as _t
    total = total_vram_mb()
    if total:
        min_free_mb = min(float(min_free_mb), max(total - 2048, 1024))
    free = free_vram_mb()
    t0 = _t.monotonic()
    while free < min_free_mb and _t.monotonic() - t0 < timeout_s:
        _t.sleep(poll_s)
        free = free_vram_mb()
    ok = free >= min_free_mb
    print(f"[vram-wait{('/' + tag) if tag else ''}] libera {free}MB / soglia {int(min_free_mb)}MB "
          f"in {_t.monotonic() - t0:.1f}s ({'ok' if ok else 'ATTENZIONE: scaduto → rischio shared'})",
          flush=True)
    return free


def total_vram_mb() -> int:
    """VRAM totale della GPU (MB), 0 se non rilevabile. Nessun contesto CUDA.
    Serve per rendere DINAMICHE le soglie di attesa (mai chiedere più VRAM di quanta la scheda ne abbia:
    su una 12GB una soglia fissa da 24GB non si raggiunge mai e l'attesa degenera in uno sleep pieno)."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            return int(pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024 * 1024))
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, creationflags=_CREATE_NO_WINDOW)
        return int(float(out.stdout.strip().splitlines()[0]))
    except Exception:  # noqa: BLE001
        return 0
