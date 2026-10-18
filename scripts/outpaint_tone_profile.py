"""Profilo di tono lungo il confine foto/estensione di un outpaint (CPU, nessuna GPU).

Per capire se c'è l'«effetto cornice»: a bande di 32 px, dal confine verso l'esterno (estensione) e verso l'interno
(foto), mediana di L*, chroma C* e a*, b* in LAB. Una cornice = salto fra la banda 0 dentro e la banda 0 fuori,
oppure deriva di L*/C* via via che ci si allontana. Metriche di sintesi:
  step_L / step_C  = |banda 0 fuori − banda 0 dentro|  (salto al confine)
  drift_L / drift_C = |banda lontana − banda 0 fuori|    (deriva dentro l'estensione)

Uso (su un risultato reale + il frame che il canvas ha esportato e la sua maschera):
    python scripts/outpaint_tone_profile.py RISULTATO.png MASCHERA.png [--json out.json]
La maschera è quella di mask_canvas._border_mask (bianco = estensione generata, nero = foto).
Per il confronto prima/dopo (stesso seme) si lancia due volte: con NATIVE_OUTPAINT_TONEMATCH=0 e =1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.tone_match import _srgb_to_lab  # noqa: E402

BAND = 32


def _dist_outside_inside(ext: np.ndarray):
    """Distanza (px) dal confine: positiva dentro l'estensione, negativa dentro la foto."""
    import cv2
    d_ext = cv2.distanceTransform(ext.astype(np.uint8), cv2.DIST_L2, 3)
    d_pho = cv2.distanceTransform((~ext).astype(np.uint8), cv2.DIST_L2, 3)
    return np.where(ext, d_ext, -d_pho)


def profile(img: Image.Image, ext_mask: np.ndarray, n_out: int = 6, n_in: int = 3) -> dict:
    lab = _srgb_to_lab(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    d = _dist_outside_inside(ext_mask)
    rows = []
    for k in list(range(-n_in, 0)) + list(range(0, n_out)):
        lo, hi = k * BAND, (k + 1) * BAND
        sel = (d > lo) & (d <= hi) if k >= 0 else (d > lo) & (d <= hi) & (~ext_mask)
        if sel.sum() < 200:
            continue
        rows.append({"band": k, "lo_px": lo, "hi_px": hi, "n": int(sel.sum()),
                     "L": float(np.median(lab[..., 0][sel])), "C": float(np.median(chroma[sel])),
                     "a": float(np.median(lab[..., 1][sel])), "b": float(np.median(lab[..., 2][sel])),
                     "L_std": float(lab[..., 0][sel].std())})
    by = {r["band"]: r for r in rows}
    out = {"bands": rows}
    if 0 in by and -1 in by:
        out["step_L"] = abs(by[0]["L"] - by[-1]["L"]); out["step_C"] = abs(by[0]["C"] - by[-1]["C"])
    far = [r for r in rows if r["band"] >= 2]
    if 0 in by and far:
        f = far[-1]
        out["drift_L"] = abs(f["L"] - by[0]["L"]); out["drift_C"] = abs(f["C"] - by[0]["C"])
        out["far_band"] = f["band"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result"); ap.add_argument("mask"); ap.add_argument("--json", default="")
    a = ap.parse_args()
    img = Image.open(a.result).convert("RGB")
    m = np.asarray(Image.open(a.mask).convert("L").resize(img.size, Image.NEAREST)) > 127
    p = profile(img, m)
    for r in p["bands"]:
        print(f"banda {r['band']:+d} ({r['lo_px']:+d}..{r['hi_px']:+d} px, n={r['n']}): "
              f"L {r['L']:.1f}  C {r['C']:.1f}  a {r['a']:+.1f}  b {r['b']:+.1f}")
    print({k: round(v, 2) for k, v in p.items() if k.startswith(("step", "drift"))})
    if a.json:
        Path(a.json).write_text(json.dumps(p, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
