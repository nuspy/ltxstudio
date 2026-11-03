"""Ponte fra i backend di segmentazione e SAM 3.1 di Meta (pacchetto `sam3` revisionato in E:/AI/Tools/sam3_src): testo -> riquadri / maschere, al posto di GroundingDINO.

Il codice di Meta gira SOLO in un processo figlio con le stesse garanzie del lanciatore E:/AI/Tools/sam3_run.sh: venv diffusers_next_venv, `-P -s`
(la cartella dello script non entra in sys.path, niente site-packages dell'utente), PYTHONPATH su sam3_src e sam3_deps, script scelti per nome dalla cartella
workers/sam3_jobs del repo (copia di lavoro in E:/AI/Tools/sam3_jobs). Inferenza senza rete: il checkpoint e locale. I punti non passano di qui: clic e lazo
restano di SAM2.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

PY = r"E:\AI\Tools\diffusers_next_venv\Scripts\python.exe"
SAM3_SRC = r"E:\AI\Tools\sam3_src"
SAM3_DEPS = r"E:\AI\Tools\sam3_deps"
JOBS_DIR = str(Path(__file__).resolve().parents[1] / "workers" / "sam3_jobs")


def job_command(name: str, args: list[str], python: str = "", jobs_dir: str = "") -> tuple[list[str], dict]:
    """(argv, env) per lanciare lo script `name` di sam3_jobs. Solo nomi di file (niente percorsi), come il lanciatore."""
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"solo nomi di file di sam3_jobs: {name!r}")
    script = os.path.join(jobs_dir or JOBS_DIR, name)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([SAM3_SRC, SAM3_DEPS])
    env["PYTHONIOENCODING"] = "utf-8"
    return [python or PY, "-P", "-s", "-u", script] + [str(a) for a in args], env


def _run(name: str, args: list[str], marker: str, runner=None, python: str = "") -> dict:
    cmd, env = job_command(name, args, python)
    if runner is None:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        rc, so, se = r.returncode, r.stdout or "", r.stderr or ""
    else:
        rc, so, se = runner(cmd, env)
    line = next((ln for ln in so.splitlines() if ln.startswith(marker)), "")
    if not line:
        raise RuntimeError("SAM 3.1 non ha risposto" + (f" (codice {rc}): {(se or so).strip()[-400:]}" if (se or so).strip() else f" (codice {rc})"))
    return json.loads(line[len(marker):])


def default_checkpoint() -> str:
    """Il checkpoint di Meta se c'e, altrimenti la copia Comfy-Org (solo se e stata scelta come equivalente e scaricata)."""
    from core import segmentation_engine as SE
    return str(SE.checkpoint_path())


def detect(image_path: str, text: str, device: str = "", threshold: float = 0.5, out_dir: str = "", checkpoint: str = "", runner=None, python: str = "") -> dict:
    """{'boxes': [[x0,y0,x1,y1]], 'scores': [...], 'masks': [png...], 'union': png|'' , 'presence', 'size', 'device'}; liste vuote se non trova nulla.
    `runner(cmd, env) -> (returncode, stdout, stderr)` iniettabile nei test."""
    out_dir = out_dir or tempfile.mkdtemp(prefix="sam31_")
    args = ["--image", image_path, "--text", text.strip().rstrip("."), "--out-dir", out_dir, "--threshold", threshold,
            "--checkpoint", checkpoint or default_checkpoint()]
    if device:
        args += ["--device", device]
    return _run("sam31_image_text.py", args, "SAM31:", runner, python)


def boxes(image_path: str, text: str, device: str = "", threshold: float = 0.5, **kw) -> list[list[float]]:
    """Riquadri [x0, y0, x1, y1, punteggio] ordinati per punteggio decrescente (stesso formato di video_segment.detect_boxes)."""
    r = detect(image_path, text, device, threshold, **kw)
    out = [[float(v) for v in b[:4]] + [float(s)] for b, s in zip(r["boxes"], r["scores"])]
    return sorted(out, key=lambda b: -b[4])


def track(video_path: str, text: str, out_dir: str, device: str = "", max_frames: int = 0, checkpoint: str = "", runner=None, python: str = "") -> dict:
    """Tracking multiplex da testo (tutti gli oggetti, tutti i fotogrammi): out_dir/obj_<id>/<fotogramma>.png. Richiede il tracker multiplex di Meta."""
    args = ["--video", video_path, "--text", text.strip().rstrip("."), "--out-dir", out_dir, "--checkpoint", checkpoint or default_checkpoint()]
    if device:
        args += ["--device", device]
    if max_frames:
        args += ["--max-frames", max_frames]
    return _run("sam31_video_text.py", args, "SAM31V:", runner, python)
