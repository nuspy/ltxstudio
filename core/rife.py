"""RIFE frame-interpolation post-process: fluidifica i video (anti-judder) e aumenta gli fps.

Interpola frame intermedi tra coppie consecutive → video più morbido. `factor=2` raddoppia i
fotogrammi (e l'fps) mantenendo la DURATA (quindi corregge gli scatti, non la velocità). Per
correggere anche la velocità di Wan (16fps): generare a 16fps e poi RIFE a 24/30 → naturale e fluido.

Riusa Practical-RIFE (codice + pesi flownet.pkl già su disco). Dal processo GUI/API gira SEMPRE
come SUBPROCESS nel venv inference_server (scripts/rife_run.py): in-process creava un contesto
CUDA + un modello in _CACHE mai liberati nel processo app (regola no-CUDA-in-app). Il ramo
in-process resta solo per il runner (RIFE_INPROC=1). Ri-muxa l'audio originale.

API:
  available() -> bool
  interpolate_video(in, out, factor=2, target_fps=None, progress=) -> str
"""
from __future__ import annotations

import os
import sys
import subprocess

# Codice di Practical-RIFE incluso nel progetto (workers/rife_pr, MIT): prima stava dentro la cartella FramePack,
# cancellata il 14/10/2026 → l'interpolazione si rompeva. I pesi sono la voce di catalogo «rife-flownet-v4».
RIFE_REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workers", "rife_pr")


def weights_dir() -> str:
    """Cartella con flownet.pkl (voce di catalogo rife-flownet-v4); "" se il catalogo non la risolve."""
    try:
        from core import model_paths as _mp
        p = _mp.p("rife-flownet-v4", "")
        return os.path.dirname(p) if p else ""
    except Exception:                                    # noqa: BLE001
        return ""
_INF_PY = r"E:\AI\Tools\inference_server_venv\Scripts\python.exe"
_CACHE: dict = {}


def available() -> bool:
    w = weights_dir()
    return (os.path.exists(os.path.join(RIFE_REPO, "train_log", "RIFE_HDv3.py"))
            and bool(w) and os.path.isfile(os.path.join(w, "flownet.pkl")))


def _load():
    if "m" in _CACHE:
        return _CACHE["m"]
    import torch
    if RIFE_REPO not in sys.path:
        sys.path.insert(0, RIFE_REPO)
    from train_log.RIFE_HDv3 import Model    # architettura + loader di Practical-RIFE
    m = Model()
    m.load_model(weights_dir(), -1)
    m.eval()
    m.device()
    _CACHE["m"] = m
    return m


def _delegate(in_path: str, out_path: str, factor: int, target_fps: float | None, progress) -> str:
    """Esegue RIFE nel venv inference_server via scripts/rife_run.py; inoltra il progress
    (righe 'PROG:i/n') e ritorna il path da 'OK:<path>'."""
    runner = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "scripts", "rife_run.py")
    cmd = [_INF_PY, "-u", runner, in_path, out_path, str(int(factor)), str(float(target_fps or 0))]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    ok = ""
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("PROG:") and progress:
            try:
                i, n = line[5:].split("/")
                progress(int(i), int(n))
            except Exception:                               # noqa: BLE001
                pass
        elif line.startswith("OK:"):
            ok = line[3:].strip()
    proc.wait()
    if not ok:
        raise RuntimeError(f"RIFE (subprocess inference_server_venv) fallito: exit {proc.returncode}")
    return ok


def interpolate_video(in_path: str, out_path: str, factor: int = 2,
                      target_fps: float | None = None, progress=None) -> str:
    """factor=2 → ×2 frame/fps (anti-judder). target_fps opzionale forza l'fps di uscita.
    Dal processo app SEMPRE subprocess (no-CUDA-in-app); in-process solo nel runner dedicato."""
    if not os.environ.get("RIFE_INPROC") and os.path.exists(_INF_PY):
        return _delegate(in_path, out_path, factor, target_fps, progress)
    import cv2
    import torch
    import numpy as np
    import torch.nn.functional as F
    m = _load()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise FileNotFoundError(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out_fps = float(target_fps) if target_fps else fps * factor
    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64

    def to_t(fr):
        t = torch.from_numpy(fr[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float().to(dev) / 255.0
        return F.pad(t, (0, pw - w, 0, ph - h))

    def to_img(t):
        a = (t[0, :, :h, :w].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255)
        return a.astype(np.uint8)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".noaudio.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("video vuoto")
    i = 0
    with torch.inference_mode():
        while True:
            ok, nxt = cap.read()
            vw.write(prev)
            if not ok:
                break
            I0, I1 = to_t(prev), to_t(nxt)
            for k in range(1, factor):
                mid = m.inference(I0, I1, k / factor)
                vw.write(to_img(mid))
            prev = nxt
            i += 1
            if progress and i % 20 == 0:
                progress(i, total)
    cap.release(); vw.release()
    # ri-muxa l'audio originale (durata invariata → resta in sync)
    subprocess.run(["ffmpeg", "-y", "-i", tmp, "-i", in_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0?", "-shortest", out_path], check=False)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return out_path
