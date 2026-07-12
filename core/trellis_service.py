"""TRELLIS.2: immagine → modello 3D (mesh GLB con PBR) — STANDALONE, niente ComfyUI.

Gira nel proprio venv isolato `E:\\AI\\Tools\\TRELLIS2\\venv` (repo microsoft/TRELLIS), pilotato
via subprocess attraverso scripts/trellis_generate.py (headless). Sostituisce il custom node
ComfyUI-TRELLIS2. Pesi (microsoft/TRELLIS-image-large) scaricati da HF al primo uso.
Export FBX/OBJ dal GLB col Blender dell'app (riuso world_service.export_mesh).
"""
from __future__ import annotations

import re
from pathlib import Path

from core.music_service import Job          # job subprocess generico (riuso)

TOOLS = Path(r"E:\AI\Tools")


def kill_orphans() -> int:
    """Killa eventuali subprocess trellis_generate.py orfani (crash precedente dell'app). Vedi core.proc_utils."""
    from core.proc_utils import kill_orphans as _ko
    return _ko("trellis_generate")
TRELLIS_DIR = TOOLS / "TRELLIS2"
TRELLIS_VENV_PY = TRELLIS_DIR / "venv" / "Scripts" / "python.exe"
GEN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trellis_generate.py"
OUT_DIR = Path(r"E:\AI\ltxstudio\output\3d")


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", (name or "model").strip()) or "model"


def is_installed() -> bool:
    # marker scritto dall'installer solo dopo `import trellis` riuscito (deps CUDA difficili su Windows)
    return TRELLIS_VENV_PY.exists() and (TRELLIS_DIR / ".ltx_ready").exists()


def glb_path(name: str) -> Path:
    return OUT_DIR / _safe(name) / "model.glb"


def make_job(image_path: str, name: str = "model", *, seed: int = 1,
             ss_steps: int = 12, slat_steps: int = 12, simplify: float = 0.95,
             texture_size: int = 1024, target_faces: int | None = None,
             hf_token: str | None = None,
             resolution: str | None = None, max_tokens: int | None = None,
             guidance: float | None = None, geo_steps: int | None = None,
             tex_guidance: float | None = None, tex_steps: int | None = None,
             texture: bool = True, remove_bg: bool = True,
             attn: str | None = None, precision: str | None = None,
             bg_color: str | None = None, **cbs) -> Job:
    out = OUT_DIR / _safe(name)
    out.mkdir(parents=True, exist_ok=True)
    # trellis_generate.py cabla ORA all'API ufficiale TRELLIS.2 i parametri della GUI:
    # resolution→--pipeline_type, max_tokens→--max_tokens, guidance/geo_steps→sampler geo (ss+shape),
    # tex_*→sampler texture, texture→--texture (texture_extraction), remove_bg→--remove_bg (preprocess),
    # attn→ATTN_BACKEND, precision→autocast, bg_color→composito soggetto. target_faces→--decimation.
    import os as _os
    kill_orphans()                                               # niente DUE istanze: pulisci orfani da crash precedenti
    # MULTI-VISTA: image_path può essere una LISTA di path (più foto dello stesso oggetto) →
    # --image ripetuto; lo script concatena i token delle viste nel condizionatore.
    _imgs = image_path if isinstance(image_path, (list, tuple)) else [image_path]
    cmd = [TRELLIS_VENV_PY, "-u", str(GEN_SCRIPT)]
    for _im in _imgs:
        cmd += ["--image", str(_im)]
    cmd += ["--output_dir", str(out),
            "--seed", int(seed), "--texture_size", int(texture_size),
            "--texture", 1 if texture else 0, "--remove_bg", 1 if remove_bg else 0,
            "--parent-pid", int(_os.getpid())]                   # watchdog: se l'app muore, il subprocess muore
    if target_faces:
        cmd += ["--decimation", int(target_faces)]
    if resolution:
        cmd += ["--pipeline_type", str(resolution)]
    if max_tokens:
        cmd += ["--max_tokens", int(max_tokens)]
    if guidance is not None and guidance >= 0:
        cmd += ["--guidance", float(guidance)]
    if geo_steps:
        cmd += ["--steps", int(geo_steps)]
    if tex_guidance is not None and tex_guidance >= 0:
        cmd += ["--tex_guidance", float(tex_guidance)]
    if tex_steps:
        cmd += ["--tex_steps", int(tex_steps)]
    if attn and attn != "auto":
        cmd += ["--attn", str(attn)]
    if precision and precision != "auto":
        cmd += ["--precision", str(precision)]
    if bg_color:
        cmd += ["--bg_color", str(bg_color)]
    # ISOLAMENTO dal venv PADRE (app = ComfyUI venv): il subprocess gira nel venv TRELLIS ma erediterebbe
    # l'ambiente dell'app → può caricare torch/CUDA DLL SBAGLIATE = crash nativo intermittente (il backend da
    # solo funziona, dall'app muore). Rimuoviamo PYTHONPATH/PYTHONHOME/PYTHONSTARTUP e togliamo dal PATH le
    # cartelle di ComfyUI, così TRELLIS usa ESCLUSIVAMENTE le proprie librerie. (Job: valore None = rimuovi.)
    import os as _os
    _clean_path = _os.pathsep.join(
        p for p in _os.environ.get("PATH", "").split(_os.pathsep)
        if p and "comfy" not in p.lower())
    env: dict = {"PYTHONPATH": None, "PYTHONHOME": None, "PYTHONSTARTUP": None, "PATH": _clean_path}
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    return Job(cmd, TRELLIS_DIR, env=env, **cbs)


def preflight(image_path: str) -> list[str]:
    errs = []
    if not is_installed():
        errs.append("TRELLIS non installato: usa l'installer (scripts/install_trellis.py).")
    if not image_path or not Path(image_path).is_file():
        errs.append(f"Immagine inesistente: {image_path}")
    return errs
