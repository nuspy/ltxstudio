"""Preprocessing ControlNet nativo (canny / depth / pose) — sostituisce i custom_node controlnet_aux.

Estrae la "guida" di controllo da un'immagine o dai frame di un video, per i ControlNet nativi del
motore video LTX-2 (#24) e per le immagini. Canny è puro OpenCV (nessun peso, validabile a freddo);
depth e pose richiedono modelli (DepthAnything / OpenPose body CMU) con import lazy.

API:
  canny(image_path, out, low=100, high=200) -> str         # puro cv2
  depth(image_path, out) -> str                            # DepthAnything-V2 (transformers)
  pose(image_path, out) -> str                             # OpenPose body nativo (core.openpose_body)
  preprocess(image_path, kind, out) -> str                 # dispatch per 'canny'|'depth'|'pose'
  preprocess_video(video_path, kind, out_video, fps=None) -> str  # frame-by-frame → video di controllo
"""
from __future__ import annotations

import os

# FASE F weights-unification: dir DAL registro unico (id depth-anything-v2), fallback al letterale.
try:
    from core import model_paths as _mp
    DEPTH_DIR = str(_mp.path_of("depth-anything-v2") or r"E:\AI\Models\Segmentation\hf\depth-anything-v2-large")
except Exception:                                        # noqa: BLE001
    DEPTH_DIR = r"E:\AI\Models\Segmentation\hf\depth-anything-v2-large"
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


def depth(image_path: str, out_path: str, device: int = -1) -> str:
    """Depth map via DepthAnything-V2 (transformers pipeline). Richiede i pesi in DEPTH_DIR.
    `device=-1` = CPU: il processo GUI non apre MAI un contesto CUDA (regola app-no-cuda-context); un worker
    può passare device=0. Pesi: cartella HF; l'originale .pth si converte con core.depth_anything_convert."""
    import numpy as np
    import cv2
    from transformers import pipeline as hf_pipeline
    # weights-unification C: path assente → errore CHIARO (prima: HFValidationError criptica).
    if not os.path.isdir(DEPTH_DIR):
        raise RuntimeError(f"DepthAnything-V2 assente in {DEPTH_DIR}: scaricalo dal tab "
                           "«Pesi modelli» (voce depth-anything-v2, ~1.3GB).")
    pipe = hf_pipeline(task="depth-estimation", model=DEPTH_DIR, device=device)
    from PIL import Image
    res = pipe(Image.open(image_path).convert("RGB"))
    d = np.asarray(res["depth"], dtype="float32")
    d = (d - d.min()) / (float(np.ptp(d)) + 1e-6) * 255.0   # np.ptp: ndarray.ptp non esiste in NumPy 2
    rgb = cv2.cvtColor(d.astype("uint8"), cv2.COLOR_GRAY2BGR)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, rgb)
    return out_path


def pose(image_path: str, out_path: str, device: int = -1) -> str:
    """Posa OpenPose CORPO (18 keypoint CMU, scheletro colorato su fondo nero) — nativa, senza controlnet_aux.
    Pesi: body_pose_model.pth (core.openpose_body.WEIGHTS_PATH). `device=-1` = CPU: il processo GUI non apre MAI
    un contesto CUDA (regola app-no-cuda-context); un worker puo' passare device=0. Nessuna persona -> immagine nera."""
    import cv2
    from core import openpose_body as ob
    ob.check_weights()                                   # errore CHIARO (nome file + dimensione) se mancano i pesi
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    canvas, _res = ob.pose_image(bgr[:, :, ::-1], device="cpu" if device < 0 else f"cuda:{int(device)}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, canvas[:, :, ::-1])            # canvas RGB -> cv2 BGR (stessa convenzione di depth())
    return out_path


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
    # GUARDIE anti silent-failure: un ref-video illeggibile/corrotto produceva un out_video da 1 BYTE
    # senza errori → il worker LTX crashava molto a valle (exit 1 criptico). Errore chiaro QUI.
    if w <= 0 or h <= 0:
        cap.release()
        raise ValueError(f"video di riferimento illeggibile (dimensioni {w}x{h}): {video_path}")
    if kind.lower() != "canny":
        cap.release()
        raise NotImplementedError(f"preprocess_video supporta solo 'canny' in-process; '{kind}' è per-frame su modello.")
    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    vw = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        cap.release()
        raise RuntimeError(f"VideoWriter non apribile per {out_video} (codec mp4v, {w}x{h}@{fps:g}fps)")
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        e = cv2.Canny(fr, 100, 200)
        vw.write(cv2.cvtColor(e, cv2.COLOR_GRAY2BGR))
        n += 1
    cap.release(); vw.release()
    if n == 0 or not os.path.exists(out_video) or os.path.getsize(out_video) < 1024:
        raise RuntimeError(f"preprocess {kind}: nessun frame estratto da {video_path} (frame={n}) — "
                           "video di riferimento non valido.")
    return out_video
