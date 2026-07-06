"""Monitor memoria di sistema: RAM + VRAM DEDICATA + VRAM SHARED (i tre valori di Task Manager di Windows).

- RAM: psutil.
- VRAM dedicata/shared: contatori Windows "GPU Adapter Memory / Dedicated|Shared Usage" via win32pdh, nomi
  ENGLISH (locale-independent), sommati su tutti gli adapter. La 'shared' è la RAM usata come VRAM (lo spill
  che rallenta) — nessuna altra API Python la espone semplicemente.
- Capacità VRAM dedicata (per la %): nvidia-smi una-tantum (cache).
"""
from __future__ import annotations

import subprocess

try:
    import psutil
except Exception:                                          # noqa: BLE001
    psutil = None
try:
    import win32pdh
except Exception:                                          # noqa: BLE001
    win32pdh = None

_VRAM_TOTAL: int | None = None    # capacità dedicata (byte), cache


def _gpu_total_bytes() -> int:
    global _VRAM_TOTAL
    if _VRAM_TOTAL is not None:
        return _VRAM_TOTAL
    v = 0
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=6,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        v = int(out.stdout.strip().splitlines()[0]) * 1024 * 1024
    except Exception:                                      # noqa: BLE001
        v = 0
    _VRAM_TOTAL = v
    return v


def _gpu_util() -> int:
    """% utilizzo GPU (core CUDA/compute), 0-100. Via nvidia-smi (una chiamata/sec, finestra nascosta)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=4,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return int(out.stdout.strip().splitlines()[0])
    except Exception:                                      # noqa: BLE001
        return 0


def _pdh_sum(counter: str) -> int:
    """Somma un contatore GPU (tutti gli adapter). 0 se non disponibile."""
    if win32pdh is None:
        return 0
    hq = None
    try:
        hq = win32pdh.OpenQuery()
        h = win32pdh.AddEnglishCounter(hq, counter)
        win32pdh.CollectQueryData(hq)
        arr = win32pdh.GetFormattedCounterArray(h, win32pdh.PDH_FMT_LARGE)   # {istanza: valore}
        return int(sum(v for v in arr.values() if isinstance(v, (int, float)) and v > 0))
    except Exception:                                      # noqa: BLE001
        return 0
    finally:
        if hq is not None:
            try:
                win32pdh.CloseQuery(hq)
            except Exception:                              # noqa: BLE001
                pass


def sample() -> dict:
    """Snapshot memoria corrente (byte). Chiamabile ~1/s."""
    ram_used = ram_total = ram_pct = 0
    if psutil is not None:
        vm = psutil.virtual_memory()
        ram_used, ram_total, ram_pct = vm.used, vm.total, vm.percent
    ded = _pdh_sum(r"\GPU Adapter Memory(*)\Dedicated Usage")
    sha = _pdh_sum(r"\GPU Adapter Memory(*)\Shared Usage")
    return {
        "ram_used": ram_used, "ram_total": ram_total, "ram_pct": ram_pct,
        "vram_ded_used": ded, "vram_ded_total": _gpu_total_bytes(),
        "vram_shared_used": sha, "vram_shared_total": ram_total // 2,   # limite shared Windows ≈ metà RAM
        "gpu_util": _gpu_util(),                                          # % core CUDA (0-100)
    }
