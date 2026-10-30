"""Risoluzione del ComfyUI LEGACY locale (Comfy_Nexus*) — SOLO stdlib, importabile ovunque.

NON è un backend di generazione (quello è nativo): serve ai 2 usi residui che riusano il
venv ComfyUI come LIBRERIA: rig SkinToken (core/rig_service.py) e dequant dei checkpoint `.comfy_quant`
(core/dequant_comfy.py). MultiTalk NON lo usa piu' (task 20: motore nativo workers/wan_mt).
La cartella è stata rinominata più volte (Comfy_Nexus → .bak → .bak.bau, 17/08) rompendo i
path hardcoded: qui c'è UNA lista di candidati + override env, così un rinomino futuro si
sistema in un punto solo. Un candidato è valido solo se ha il venv (python.exe).
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "LTX_LEGACY_COMFY_ROOT"

CANDIDATES = [
    r"E:\AI\Comfy_Nexus.bak.bau\ComfyUI",
    r"E:\AI\Comfy_Nexus.bak\ComfyUI",
    r"E:\AI\Comfy_Nexus\ComfyUI",
]


def _venv_py(root: str | Path) -> Path:
    return Path(root) / "venv" / "Scripts" / "python.exe"


def comfy_root() -> str | None:
    """Radice ComfyUI legacy (env ha priorità; poi il primo candidato COL venv). None se assente."""
    env = os.environ.get(ENV_VAR)
    if env and _venv_py(env).exists():
        return env
    for c in CANDIDATES:
        if _venv_py(c).exists():
            return c
    return None


def venv_python() -> str | None:
    """python.exe del venv ComfyUI legacy, o None se non installato."""
    root = comfy_root()
    return str(_venv_py(root)) if root else None
