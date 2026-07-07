"""Utility processi (lato APP) — kill di subprocess GPU orfani.

Un subprocess GPU (TRELLIS 3D / Rig / Mondo) può sopravvivere a un crash dell'app e restare ORFANO a occupare
GPU/VRAM: una nuova generazione troverebbe DUE istanze che competono. Le code di questi moduli sono SERIALI
(una generazione per volta), quindi prima di lanciarne una nuova non deve esistere nessun subprocess di quel
tipo → quelli che si trovano sono orfani e vanno terminati. Best-effort, Windows, stdlib-only.
"""
from __future__ import annotations

import os
import subprocess


def kill_orphans(script_pattern: str) -> int:
    """Termina i python.exe la cui command line contiene `script_pattern` (es. 'trellis_generate',
    'rig_native_skintoken', 'worldgen_generate'). Ritorna quanti ne ha uccisi (best-effort).
    SICUREZZA: esclude SEMPRE il processo chiamante (l'app) → mai auto-kill anche se per qualche motivo la sua
    command line contenesse il pattern."""
    try:
        me = int(os.getpid())
        cmd = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
               f"Where-Object {{ $_.CommandLine -like '*{script_pattern}*' -and $_.ProcessId -ne {me} }} | "
               "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; 'k' }")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                             capture_output=True, text=True, timeout=15,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return out.stdout.count("k")
    except Exception:  # noqa: BLE001
        return 0
