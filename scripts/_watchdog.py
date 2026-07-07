"""Watchdog anti-orfano per i subprocess GPU (TRELLIS 3D, Rig, Mondo).

Stdlib-only (i venv isolati non hanno `core`). Se l'APP (processo padre) muore — ANCHE per crash — il subprocess
si autodistrugge, così non resta orfano a occupare GPU/VRAM (evita che una nuova generazione trovi DUE istanze
che competono). Su Windows: apre un handle SYNCHRONIZE al padre e blocca finché è vivo, poi `os._exit`.
"""
from __future__ import annotations

import os
import sys
import threading


def start_watchdog(parent_pid: int) -> None:
    if not parent_pid or sys.platform != "win32":
        return

    def _watch(ppid: int) -> None:
        import ctypes
        SYNCHRONIZE = 0x00100000
        h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, ppid)
        if not h:
            return                                           # padre già morto o non accessibile
        ctypes.windll.kernel32.WaitForSingleObject(h, 0xFFFFFFFF)   # blocca finché il padre è vivo
        print("[watchdog] APP padre terminata → esco (anti-orfano)", flush=True)
        os._exit(1)

    threading.Thread(target=_watch, args=(parent_pid,), daemon=True).start()
