"""Generazione di SCENE/MONDI 3D da una foto (o da testo) con WorldGen — standalone, non ComfyUI.

WorldGen (Apache-2.0) produce, da un'immagine o un prompt:
  - MESH  → GLB (apribile in Blender/engine/model-viewer; esportabile FBX/OBJ)
  - SPLAT → 3D Gaussian Splatting (.ply), reso fotorealistico navigabile

Gira nel proprio venv isolato (vedi scripts/install_world.py), pilotato via subprocess
attraverso scripts/worldgen_generate.py (headless, senza il server viser). I pesi (incl.
FLUX.1-dev, gated) si scaricano da HuggingFace al primo uso → serve un HF token.

L'export FBX/OBJ dal GLB usa il Blender headless già presente nell'app (tab Convertitore).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from core.music_service import Job          # job subprocess generico (riuso)

# ── Percorsi (layout E:\AI\Tools come gli altri moduli) ──────────────────────
TOOLS = Path(r"E:\AI\Tools")
WORLDGEN_DIR = TOOLS / "WorldGen"
WORLDGEN_VENV_PY = WORLDGEN_DIR / "venv" / "Scripts" / "python.exe"
GEN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "worldgen_generate.py"
WORLD_OUTPUT_DIR = Path(r"E:\AI\ltxstudio\output\worlds")
PLAYER_DIR = Path(__file__).resolve().parents[1] / "assets" / "world_player"


@dataclass
class WorldParams:
    name: str = "world"
    image_path: str = ""                    # vuoto ⇒ text-to-scene
    prompt: str = ""
    resolution: int = 1600
    low_vram: bool = False
    use_sharp: bool = False

    def project_dir(self) -> Path:
        return WORLD_OUTPUT_DIR / _safe(self.name)

    def glb(self) -> Path:
        return self.project_dir() / "mesh.glb"

    def splat(self) -> Path:
        return self.project_dir() / "splat.ply"


def _safe(name: str) -> str:
    import re
    return re.sub(r"[^\w.-]+", "_", (name or "world").strip()) or "world"


def is_installed() -> bool:
    # Marker scritto dall'installer SOLO dopo un import test riuscito (`import worldgen`).
    # NB: WorldGen è Linux-first (deps nunchaku wheel-linux, pytorch3d, py>=3.11) → su Windows
    # nativo l'installazione non è banale; il marker evita falsi "installato".
    return WORLDGEN_VENV_PY.exists() and (WORLDGEN_DIR / ".ltx_ready").exists()


def make_job(p: WorldParams, *, mesh: bool, hf_token: str | None = None, **cbs) -> Job:
    """Crea il Job di generazione (mesh=True → GLB, mesh=False → splat .ply)."""
    out = p.project_dir(); out.mkdir(parents=True, exist_ok=True)
    cmd = [WORLDGEN_VENV_PY, "-u", GEN_SCRIPT, "--output_dir", out,
           "--prompt", p.prompt or "", "--resolution", int(p.resolution)]
    if p.image_path:
        cmd += ["--image", p.image_path]
    if mesh:
        cmd += ["--mesh"]
    if p.low_vram:
        cmd += ["--low_vram"]
    if p.use_sharp:
        cmd += ["--use_sharp"]
    env = {"HF_TOKEN": hf_token, "HUGGING_FACE_HUB_TOKEN": hf_token} if hf_token else None
    return Job(cmd, WORLDGEN_DIR, env=env, **cbs)


def preflight(p: WorldParams) -> list[str]:
    errs = []
    if not is_installed():
        errs.append("WorldGen non installato: usa il pulsante d'installazione.")
    if p.image_path and not Path(p.image_path).is_file():
        errs.append(f"Immagine inesistente: {p.image_path}")
    if not p.image_path and not (p.prompt or "").strip():
        errs.append("Indica un'immagine di partenza, oppure un prompt (text-to-scene).")
    return errs


# ── Export FBX/OBJ dal GLB via Blender headless (riuso del Blender dell'app) ──
def _find_blender() -> str | None:
    try:
        from core.blender_installer import find_blender
        return find_blender()
    except Exception:                                   # noqa: BLE001
        return os.environ.get("SKINTOKEN_BLENDER_BIN") or shutil.which("blender")


def export_mesh(glb_path: Path, fmt: str = "fbx") -> tuple[Path | None, str]:
    """Converte un GLB in FBX/OBJ con Blender headless. Ritorna (out_path, errore)."""
    glb_path = Path(glb_path)
    if not glb_path.exists():
        return None, f"GLB inesistente: {glb_path}"
    blender = _find_blender()
    if not blender:
        return None, "Blender non trovato (installa Blender o imposta SKINTOKEN_BLENDER_BIN)."
    out = glb_path.with_suffix("." + fmt)
    exporter = {"fbx": "export_scene.fbx", "obj": "wm.obj_export"}.get(fmt, "export_scene.fbx")
    expr = (
        "import bpy;"
        "bpy.ops.wm.read_factory_settings(use_empty=True);"
        f"bpy.ops.import_scene.gltf(filepath=r'{glb_path}');"
        f"bpy.ops.{exporter}(filepath=r'{out}')"
    )
    try:
        r = subprocess.run([blender, "--background", "--python-expr", expr],
                           capture_output=True, text=True, timeout=600)
    except Exception as e:                              # noqa: BLE001
        return None, f"Blender fallito: {e}"
    if out.exists():
        return out, ""
    return None, (r.stderr or r.stdout or "conversione fallita")[-500:]
