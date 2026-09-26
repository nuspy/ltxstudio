"""Controllo delle risorse PRIMA di avviare un lavoro pesante (worker GPU, conversioni grandi).

Sul PC dell'utente girano spesso altri programmi che occupano gran parte del commit di sistema: lanciare un
worker da 15–30 GB senza margine produce MemoryError o "CUDA error: unknown error" a metà caricamento. Qui si
misura (senza torch né CUDA: si può chiamare anche dal processo GUI) e, se non basta, si rifiuta con i numeri.
Non si chiude MAI nessun altro processo.
"""
from __future__ import annotations

import ctypes
import os
import subprocess


class NotEnoughResources(RuntimeError):
    pass


def commit_free_gb() -> float:
    """Commit di sistema ancora disponibile (GB). Windows: GlobalMemoryStatusEx.ullAvailPageFile."""
    if os.name != "nt":
        try:
            with open("/proc/meminfo") as f:
                kv = {l.split(":")[0]: int(l.split()[1]) for l in f}
            return (kv.get("MemAvailable", 0) + kv.get("SwapFree", 0)) / 2**20
        except OSError:
            return float("inf")

    class _MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    ms = _MS()
    ms.dwLength = ctypes.sizeof(_MS)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
        return float("inf")
    return ms.ullAvailPageFile / 2**30


def vram_free_gb() -> float:
    """VRAM libera sulla GPU 0 via nvidia-smi (niente contesto CUDA nel chiamante)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip().splitlines()[0]) / 1024
    except Exception:                                        # noqa: BLE001
        return float("inf")


def require(*, commit_gb: float, vram_gb: float = 0.0, what: str = "questo lavoro") -> None:
    """Solleva NotEnoughResources se commit o VRAM liberi sono sotto il necessario (margine incluso)."""
    c = commit_free_gb()
    v = vram_free_gb() if vram_gb else float("inf")
    probs = []
    if c < commit_gb:
        probs.append(f"memoria di sistema (commit) libera {c:.1f} GB, ne servono almeno {commit_gb:.0f}")
    if v < vram_gb:
        probs.append(f"VRAM libera {v:.1f} GB, ne servono almeno {vram_gb:.0f}")
    if probs:
        raise NotEnoughResources(
            f"Risorse insufficienti per {what}: " + "; ".join(probs) +
            ". Altri programmi stanno usando la memoria: chiudili o attendi che finiscano, poi riprova.")
