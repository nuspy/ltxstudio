"""Rigging SkinTokens: mesh 3D → scheletro + skinning (rig) — STANDALONE, niente ComfyUI.

Sostituisce il custom node ComfyUI-SkinToken. Gira nel venv `E:\\AI\\Tools\\SkinTokens\\venv`
(repo VAST-AI/SkinTokens / TokenRig) via subprocess (scripts/rig_generate.py), con Blender headless
(riuso del Blender dell'app) per applicare lo skinning ed esportare il GLB/FBX rigged.
"""
from __future__ import annotations

import re
from pathlib import Path

from core.music_service import Job

TOOLS = Path(r"E:\AI\Tools")
SK_DIR = TOOLS / "SkinTokens"
SK_VENV_PY = SK_DIR / "venv" / "Scripts" / "python.exe"
GEN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rig_generate.py"
OUT_DIR = Path(r"E:\AI\ltxstudio\output\3d")


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", (name or "rig").strip()) or "rig"


def is_installed() -> bool:
    return SK_VENV_PY.exists() and (SK_DIR / ".ltx_ready").exists()


def make_job(mesh_path: str, name: str = "rig", **cbs) -> Job:
    """mesh GLB/OBJ → GLB rigged (scheletro + skin weights)."""
    out = OUT_DIR / _safe(name)
    out.mkdir(parents=True, exist_ok=True)
    rigged = out / "rigged.glb"
    cmd = [SK_VENV_PY, "-u", str(GEN_SCRIPT), "--mesh", mesh_path, "--output", str(rigged)]
    return Job(cmd, SK_DIR, **cbs)


def preflight(mesh_path: str) -> list[str]:
    errs = []
    if not is_installed():
        errs.append("SkinTokens non installato: usa l'installer (scripts/install_skintokens.py).")
    if not mesh_path or not Path(mesh_path).is_file():
        errs.append(f"Mesh inesistente: {mesh_path}")
    return errs
