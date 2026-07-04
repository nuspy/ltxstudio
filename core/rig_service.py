"""Rigging SkinTokens: mesh 3D → scheletro + skinning (rig) — niente ComfyUI server.

VALIDATO 30/06: il repo standalone VAST-AI/TokenRig è morto (404) e il porting `skin_tokens` non c'era.
Soluzione: riusiamo il node ComfyUI-SkinToken come LIBRERIA (no server) via `scripts/rig_native_skintoken.py`,
eseguito nel venv ComfyUI (Comfy_Nexus.bak) che ha torch+deps+skintokens; Blender 5.1 per l'export armatura.
Validato su mesh TRELLIS → REAL_rig_puppy.glb (armatura Mixamo, skin weights). Vedi [[trellis2-3d-integration]].
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from core.music_service import Job

# venv ComfyUI (ha le deps skintokens) — override via SKINTOKEN_COMFY_ROOT
COMFY_ROOT = Path(os.environ.get("SKINTOKEN_COMFY_ROOT", r"E:\AI\Comfy_Nexus.bak\ComfyUI"))
COMFY_VENV_PY = COMFY_ROOT / "venv" / "Scripts" / "python.exe"
SK_NODE_DIR = COMFY_ROOT / "custom_nodes" / "ComfyUI-SkinToken"
GEN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rig_native_skintoken.py"
OUT_DIR = Path(r"E:\AI\ltxstudio\output\3d")


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", (name or "rig").strip()) or "rig"


def is_installed() -> bool:
    # pronto se il venv ComfyUI + il node SkinToken (con codice vendorizzato) ci sono
    return COMFY_VENV_PY.exists() and (SK_NODE_DIR / "sktn_nodes.py").exists()


def make_job(mesh_path: str, name: str = "rig", *, template: str = "Mixamo",
             file_format: str = "glb", top_k: int = 50, top_p: float = 0.95,
             temperature: float = 1.0, repetition_penalty: float = 1.0,
             num_beams: int = 1, group_per_vertex: int = 4, **cbs) -> Job:
    """mesh GLB/OBJ → GLB/FBX rigged (scheletro + skin weights) via SkinToken-as-library."""
    out = OUT_DIR / _safe(name)
    out.mkdir(parents=True, exist_ok=True)
    rigged = out / f"rigged.{file_format}"
    cmd = [COMFY_VENV_PY, "-u", str(GEN_SCRIPT), "--mesh", mesh_path, "--output", str(rigged),
           "--template", template, "--format", file_format,
           "--top_k", str(int(top_k)), "--top_p", str(float(top_p)),
           "--temperature", str(float(temperature)), "--repetition_penalty", str(float(repetition_penalty)),
           "--num_beams", str(int(num_beams)), "--group_per_vertex", str(int(group_per_vertex))]
    return Job(cmd, GEN_SCRIPT.parent, **cbs)


def preflight(mesh_path: str) -> list[str]:
    errs = []
    if not is_installed():
        errs.append("SkinToken non disponibile: serve il venv ComfyUI + custom_nodes/ComfyUI-SkinToken.")
    if not mesh_path or not Path(mesh_path).is_file():
        errs.append(f"Mesh inesistente: {mesh_path}")
    return errs
