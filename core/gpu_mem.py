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
