"""RIFE frame-interpolation post-process: fluidifica i video (anti-judder) e aumenta gli fps.

Interpola frame intermedi tra coppie consecutive → video più morbido. `factor=2` raddoppia i
fotogrammi (e l'fps) mantenendo la DURATA (quindi corregge gli scatti, non la velocità). Per
correggere anche la velocità di Wan (16fps): generare a 16fps e poi RIFE a 24/30 → naturale e fluido.

Riusa Practical-RIFE (codice + pesi flownet.pkl già su disco), in-process con torch. Nessun
servizio esterno. Ri-muxa l'audio originale (stessa durata → resta sincronizzato).

API:
  available() -> bool
  interpolate_video(in, out, factor=2, target_fps=None, progress=) -> str
"""
from __future__ import annotations

import os
import sys
import subprocess

RIFE_REPO = r"E:\AI\Framepack\FramePack\Practical-RIFE"
_CACHE: dict = {}


def available() -> bool:
    return os.path.exists(os.path.join(RIFE_REPO, "train_log", "RIFE_HDv3.py"))


def _load():
    if "m" in _CACHE:
        return _CACHE["m"]
    import torch
    if RIFE_REPO not in sys.path:
        sys.path.insert(0, RIFE_REPO)
    from train_log.RIFE_HDv3 import Model    # architettura + loader di Practical-RIFE
    m = Model()
    m.load_model(os.path.join(RIFE_REPO, "train_log"), -1)
    m.eval()
    m.device()
    _CACHE["m"] = m
    return m


def interpolate_video(in_path: str, out_path: str, factor: int = 2,
                      target_fps: float | None = None, progress=None) -> str:
    """factor=2 → ×2 frame/fps (anti-judder). target_fps opzionale forza l'fps di uscita."""
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
