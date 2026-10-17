"""Metriche oggettive per i test dal vivo del task 9 (non sono test pytest: le chiamano la campagna live e test_task9_metrics).

  edge_iou(src, out)            IoU dei bordi Canny (dilatati di `tol` px) fra la guida e il risultato: «la sagoma e' rispettata»
  pose_agreement(src, out)      keypoint OpenPose del risultato vicini a quelli della guida: (n_corrispondenti, n_guida, errore_medio_px)
  psnr_outside_mask(orig, out, mask)   PSNR sui pixel con maschera NERA (fuori dall'area modificata): infinito = invariato
  report(...)                   riga di testo per il rapporto
Tutto su CPU (OpenCV/numpy; la posa usa core.openpose_body)."""
from __future__ import annotations

import math

import numpy as np


def _load(path, size=None):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    return im.resize(size, Image.LANCZOS) if size and im.size != tuple(size) else im


def edge_map(img, lo=100, hi=200):
    import cv2
    return cv2.Canny(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY), lo, hi) > 0


def edge_iou(src, out, tol: int = 3) -> float:
    """IoU simmetrica tollerante: i bordi di una mappa contano se c'e' un bordo dell'altra entro `tol` px."""
    import cv2
    a_img = _load(src)
    a = edge_map(a_img)
    b = edge_map(_load(out, a_img.size))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    ad = cv2.dilate(a.astype("uint8"), k) > 0
    bd = cv2.dilate(b.astype("uint8"), k) > 0
    inter = (a & bd).sum() + (b & ad).sum()
    union = a.sum() + b.sum()
    return float(inter / union) if union else 1.0


def _keypoints(img):
    from core import openpose_body as ob
    res = ob.detect(np.asarray(img))
    cand = np.asarray(res["candidate"]) if len(res["candidate"]) else np.zeros((0, 4))
    pts = {}
    for person in res["subset"]:
        for i in range(18):
            idx = int(person[i])
            if idx >= 0:
                pts.setdefault(i, []).append(cand[idx][:2])
    return {i: v[0] for i, v in pts.items()}                  # prima persona trovata per ogni punto


def pose_agreement(src, out, max_dist_frac: float = 0.06):
    """(punti del risultato entro max_dist_frac·lato lungo dal punto omologo della guida, punti della guida, errore medio px)."""
    a_img = _load(src)
    b_img = _load(out, a_img.size)
    ka, kb = _keypoints(a_img), _keypoints(b_img)
    lim = max_dist_frac * max(a_img.size)
    errs = [float(np.linalg.norm(ka[i] - kb[i])) for i in ka if i in kb]
    ok = sum(1 for e in errs if e <= lim)
    return ok, len(ka), (float(np.mean(errs)) if errs else float("nan"))


def psnr_outside_mask(orig, out, mask, thr: int = 127) -> float:
    """PSNR (dB) sui pixel dove la maschera e' NERA (non modificare). inf = identico."""
    o = _load(orig)
    r = _load(out, o.size)
    m = np.asarray(_load(mask, o.size).convert("L")) <= thr
    if not m.any():
        return float("nan")
    d = (np.asarray(o).astype("float64") - np.asarray(r).astype("float64"))[m]
    mse = float((d ** 2).mean())
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def report(name: str, **vals) -> str:
    return f"{name}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in vals.items())
