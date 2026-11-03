"""Misura AUTOMATICA del «doppione» della persona (cambia-sfondo, giudizio 03/11): invece di far guardare 6 video uguali.

Per ogni fotogramma campionato di un video H3 (o di un gen.npy) e le maschere SAM2 vere della persona misura:
  · quota di PERSONA FUORI SAGOMA: pixel salienti «persona» (u2net, CPU) fuori dalla sagoma vera dilatata di `tol` px / pixel salienti totali;
  · FEDELTA DELLA TESTA (con --source): la parte alta della persona nell'uscita contro la sorgente (~0 = e l'originale, > 0,05 = ridisegnata dal modello);
  · VOLTI «FANTASMA»: volti (Haar) nel vicinato della persona a piu di 0,25 larghezze dalla testa vera (i volti dello sfondo si scartano).
Aggrega per finestre di tempo (clip 1 = primi 6 s contro il resto): se il doppione e nel clip 1 si vede subito. Nessuna GPU (onnxruntime CPU).

Uso:
  python tests/app_live/ghost_metric.py <video.mp4> <cartella_maschere> --source <sorgente.mp4> [--secs 0:6,6:12] [--step 6] [--tol 10] [--json out.json]
  python tests/app_live/ghost_metric.py --gen <gen.npy> <cartella_maschere> [--start 0]
Le maschere sono i png del tracking (una per frame sorgente a 24 fps, bianco = persona), indicizzate come i frame."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

U2NET = Path.home() / ".u2net" / "u2net.onnx"


# ── funzioni pure (testabili senza modelli) ─────────────────────────────────────────────────────
def head_center(mask: np.ndarray, top_frac: float = 0.18):
    """(cx, cy, larghezza persona, altezza persona) del centro della TESTA = baricentro della parte alta (top_frac) della sagoma; None se vuota."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 200:
        return None
    y0, y1 = ys.min(), ys.max()
    h = y1 - y0 + 1
    sel = ys <= y0 + max(8, int(h * top_frac))
    return float(xs[sel].mean()), float(ys[sel].mean()), float(xs.max() - xs.min() + 1), float(h)


def out_share(salient: np.ndarray, mask: np.ndarray, tol_px: int = 10) -> float:
    """Quota dei pixel salienti (persona vista dal segmentatore) che cadono FUORI dalla sagoma vera dilatata di tol_px. 0 = niente persona fuori sagoma."""
    import cv2
    s = salient.astype(bool)
    if s.sum() < 50:
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1))
    zone = cv2.dilate(mask.astype(np.uint8), k) > 0
    return float((s & ~zone).sum()) / float(s.sum())


def faces_near_person(faces: list[tuple[float, float]], head: tuple, person_w: float, person_h: float):
    """Volti NEL VICINATO della persona (a +-0,9 larghezze dal centro della testa in orizzontale, tra 0,5 altezze sopra e 0,6 sotto): scarta i volti di sfondo."""
    return [(fx, fy) for fx, fy in faces if abs(fx - head[0]) <= 0.9 * person_w and -0.5 * person_h <= fy - head[1] <= 0.6 * person_h]


def face_offset(faces: list[tuple[float, float]], head: tuple, person_w: float, person_h: float = 1e9):
    """(volti vicini, distanza px del piu vicino alla testa vera, distanza / larghezza persona, volti «fantasma» = vicini ma a piu di 0,25 larghezze dalla testa vera)."""
    if head is None:
        return 0, None, None, 0
    near = faces_near_person(faces, head, person_w, person_h)
    if not near:
        return 0, None, None, 0
    d = [float(np.hypot(fx - head[0], fy - head[1])) for fx, fy in near]
    return len(near), min(d), min(d) / max(1.0, person_w), sum(1 for x in d if x > 0.25 * person_w)


def head_fidelity(out_bgr: np.ndarray, src_bgr: np.ndarray, mask: np.ndarray, top_frac: float = 0.35):
    """FEDELTA della testa/spalle: differenza quadratica normalizzata (0 = identica) fra la parte alta della persona nel SORGENTE (solo pixel della sagoma) e lo stesso
    punto del video in uscita. Nel cambia-sfondo la persona e tenuta dall'originale: ~0 quando e davvero l'originale, alta (>0,05) quando nelle celle di bordo il modello
    ha ridisegnato la testa (doppione/volto sfasato). None se la sagoma e vuota."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 200:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    ty1 = y0 + max(8, int((y1 - y0 + 1) * top_frac))
    a = out_bgr[y0:ty1, x0:x1 + 1].astype(np.float32)
    b = src_bgr[y0:ty1, x0:x1 + 1].astype(np.float32)
    m = mask[y0:ty1, x0:x1 + 1]
    if m.sum() < 100:
        return None
    d = ((a - b) ** 2).sum(-1)[m].sum()
    n = ((a ** 2).sum(-1)[m].sum() * (b ** 2).sum(-1)[m].sum()) ** 0.5
    return float(d / max(n, 1e-6))


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    fid = [r["head_fid"] for r in rows if r.get("head_fid") is not None]
    return {"n": len(rows), "fedelta_testa": float(np.mean(fid)) if fid else None,
            "quota_fuori_sagoma": float(np.mean([r["out_share"] for r in rows])),
            "frame_con_volto_fantasma": float(np.mean([1.0 if r["faces_ghost"] > 0 else 0.0 for r in rows])),
            "volti_vicini_medi": float(np.mean([r["faces"] for r in rows]))}


# ── modelli (solo CPU) ──────────────────────────────────────────────────────────────────────────
_SESS = {}


def saliency(frame_rgb: np.ndarray) -> np.ndarray:
    """Maschera booleana «persona/oggetto saliente» con u2net su CPU, alla risoluzione del frame."""
    import cv2
    import onnxruntime as ort
    if "s" not in _SESS:
        _SESS["s"] = ort.InferenceSession(str(U2NET), providers=["CPUExecutionProvider"])
    h, w = frame_rgb.shape[:2]
    x = cv2.resize(frame_rgb, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    y = _SESS["s"].run(None, {_SESS["s"].get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0, 0]
    y = (y - y.min()) / max(1e-6, float(y.max() - y.min()))
    return cv2.resize(y, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5


def detect_faces(frame_rgb: np.ndarray) -> list[tuple[float, float]]:
    import cv2
    if "f" not in _SESS:
        _SESS["f"] = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    g = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    fs = _SESS["f"].detectMultiScale(g, scaleFactor=1.1, minNeighbors=5, minSize=(max(20, g.shape[1] // 40), max(20, g.shape[1] // 40)))
    return [(float(x + w / 2), float(y + h / 2)) for x, y, w, h in fs]


def measure_frame(frame_rgb: np.ndarray, mask: np.ndarray, tol_px: int = 10, src_rgb: np.ndarray | None = None) -> dict | None:
    import cv2
    head = head_center(mask)
    if head is None:
        return None
    sal = saliency(frame_rgb)
    n, dpx, dw, fghost = face_offset(detect_faces(frame_rgb), head, head[2], head[3])
    r = {"out_share": out_share(sal, mask, tol_px), "faces": n, "face_dist_px": dpx, "face_dist_w": dw, "faces_ghost": fghost, "head_fid": None}
    if src_rgb is not None:
        s = cv2.resize(src_rgb, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_AREA)
        r["head_fid"] = head_fidelity(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), cv2.cvtColor(s, cv2.COLOR_RGB2BGR), mask)
    return r


def _load_masks(mdir: str, n: int, size):
    import cv2
    from PIL import Image
    files = sorted(glob.glob(os.path.join(mdir, "*.png")))
    out = []
    for i in range(min(n, len(files))):
        out.append(cv2.resize(np.array(Image.open(files[i]).convert("L")), size, interpolation=cv2.INTER_NEAREST) > 127)
    return out


def measure_video(video: str, mdir: str, secs=((0, 6), (6, 12)), step: int = 6, tol_px: int = 10, gen: str | None = None, start: int = 0, source: str | None = None) -> dict:
    import cv2
    if gen:
        frames = np.load(gen)
        get = lambda i: frames[i]                                       # noqa: E731
        n = frames.shape[0]
        h, w = frames.shape[1:3]
        fps = 24.0
    else:
        cap = cv2.VideoCapture(video)
        fl = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            fl.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        get = lambda i: fl[i]                                           # noqa: E731
        n = len(fl)
        h, w = fl[0].shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    masks = _load_masks(mdir, start + n, (w, h))
    sf = None
    if source:
        cs = cv2.VideoCapture(source)
        sf = []
        while True:
            ok, f = cs.read()
            if not ok:
                break
            sf.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    res = {}
    for a, b in secs:
        rows = []
        for t in range(int(a * fps), min(n, int(b * fps)), step):
            if start + t >= len(masks):
                break
            r = measure_frame(get(t), masks[start + t], tol_px, sf[start + t] if sf is not None and start + t < len(sf) else None)
            if r:
                rows.append(r)
        res[f"{a}-{b}s"] = aggregate(rows)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?")
    ap.add_argument("masks")
    ap.add_argument("--gen")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--secs", default="0:6,6:12")
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--tol", type=int, default=10)
    ap.add_argument("--source", help="video sorgente (per la fedelta della testa)")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if not U2NET.exists():
        print(f"manca {U2NET}", file=sys.stderr)
        return 2
    secs = [tuple(int(x) for x in s.split(":")) for s in a.secs.split(",")]
    r = measure_video(a.video or "", a.masks, secs, a.step, a.tol, a.gen, a.start, a.source)
    for k, v in r.items():
        fid = v.get("fedelta_testa")
        print(f"{k:8s}  n={v.get('n')}  fedelta testa {'n/d' if fid is None else format(fid, '.3f')}  persona fuori sagoma {100 * v.get('quota_fuori_sagoma', 0):5.1f} %  "
              f"frame con volto fantasma {100 * v.get('frame_con_volto_fantasma', 0):5.1f} %  volti vicini {v.get('volti_vicini_medi')}")
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
