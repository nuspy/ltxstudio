"""Preprocessing ControlNet nativo (canny / depth / pose) — sostituisce i custom_node controlnet_aux.

Estrae la "guida" di controllo da un'immagine o dai frame di un video, per i ControlNet nativi del
motore video LTX-2 (#24) e per le immagini. Canny è puro OpenCV (nessun peso, validabile a freddo);
depth e pose richiedono modelli (DepthAnything / DWpose) → scaffold con import lazy, da validare su GPU.

API:
  canny(image_path, out, low=100, high=200) -> str         # puro cv2
  depth(image_path, out) -> str                            # DepthAnything-V2 (transformers)
  pose(image_path, out) -> str                             # DWpose/openpose (stub)
  preprocess(image_path, kind, out) -> str                 # dispatch per 'canny'|'depth'|'pose'
  preprocess_video(video_path, kind, out_video, fps=None) -> str  # frame-by-frame → video di controllo
"""
from __future__ import annotations

import os

DEPTH_DIR = r"E:\AI\Models\Segmentation\hf\depth-anything-v2-large"   # da scaricare per depth
POSE_DIR = r"E:\AI\Models\Segmentation\hf\dwpose"                     # da scaricare per pose

KINDS = ("canny", "depth", "pose")


def canny(image_path: str, out_path: str, low: int = 100, high: int = 200) -> str:
    """Edge map Canny (puro OpenCV). Nessun peso necessario."""
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    edges = cv2.Canny(img, int(low), int(high))
    rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)        # 3 canali (i ControlNet vogliono RGB)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, rgb)
    return out_path


def depth(image_path: str, out_path: str) -> str:
    """Depth map via DepthAnything-V2 (transformers pipeline). Richiede i pesi in DEPTH_DIR.
    Scaffold: da validare su GPU dopo il download dei pesi."""
    import numpy as np
    import cv2
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline(task="depth-estimation", model=DEPTH_DIR)
    from PIL import Image
    res = pipe(Image.open(image_path).convert("RGB"))
    d = np.asarray(res["depth"], dtype="float32")
    d = (d - d.min()) / (d.ptp() + 1e-6) * 255.0
    rgb = cv2.cvtColor(d.astype("uint8"), cv2.COLOR_GRAY2BGR)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, rgb)
    return out_path


def pose(image_path: str, out_path: str) -> str:
    """Pose skeleton (DWpose/openpose). Stub: richiede i pesi in POSE_DIR + il preprocessor.
    Da implementare/validare su GPU (fase successiva)."""
    raise NotImplementedError(
        "pose preprocessing non ancora implementato: scaricare DWpose in POSE_DIR e collegare il detector.")


def preprocess(image_path: str, kind: str, out_path: str, **kw) -> str:
    kind = (kind or "canny").lower()
    if kind == "canny":
        return canny(image_path, out_path, **kw)
    if kind == "depth":
        return depth(image_path, out_path)
    if kind == "pose":
        return pose(image_path, out_path)
    raise ValueError(f"controlnet kind sconosciuto: {kind} (validi: {KINDS})")


def preprocess_video(video_path: str, kind: str, out_video: str, fps: float | None = None) -> str:
    """Applica il preprocess ad OGNI frame del video → un video di controllo (per ControlNet video).
    Canny è in-process (cv2); depth/pose richiedono i rispettivi modelli."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = fps or (cap.get(cv2.CAP_PROP_FPS) or 24.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    vw = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if kind.lower() != "canny":
        cap.release(); vw.release()
        raise NotImplementedError(f"preprocess_video supporta solo 'canny' in-process; '{kind}' è per-frame su modello.")
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        e = cv2.Canny(fr, 100, 200)
        vw.write(cv2.cvtColor(e, cv2.COLOR_GRAY2BGR))
    cap.release(); vw.release()
    return out_video
