"""Sfondo VECCHIO residuo nella fascia attorno alla sagoma (cambia sfondo H3 sul video nuovo). Per ogni fotogramma: fascia = dilatazione della persona di BAND px
(a 864x480) meno la persona; «vecchio» = pixel della fascia dove l'uscita e quasi uguale alla sorgente (diff media sui 3 canali < THR). Controllo: la stessa misura
in una zona lontana dalla persona (sfondo che deve essere tutto cambiato)."""
import sys, glob, os
import numpy as np
import cv2
from PIL import Image

OUT, SRC, MASKS = sys.argv[1], sys.argv[2], sys.argv[3]
BAND, THR = 24, 14

def frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out

of = frames(OUT)
sf = frames(SRC)
mf = sorted(glob.glob(os.path.join(MASKS, "*.png")))
n = min(len(of), len(sf), len(mf))
h, w = of[0].shape[:2]
res = []
for t in range(0, n, 6):
    o = of[t].astype(np.int16)
    s = cv2.resize(sf[t], (w, h), interpolation=cv2.INTER_AREA).astype(np.int16)
    m = cv2.resize(np.array(Image.open(mf[t]).convert("L")), (w, h), interpolation=cv2.INTER_NEAREST) > 127
    if m.sum() < 500:
        continue
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BAND + 1, 2 * BAND + 1))
    big = cv2.dilate(m.astype(np.uint8), k) > 0
    band = big & ~m
    far = ~cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * BAND + 1, 4 * BAND + 1))).astype(bool)
    d = np.abs(o - s).mean(-1)
    # solo pixel con struttura (non uniformi): un muro liscio uguale prima e dopo non conta come «vecchio» ma come ambiguo, lo teniamo comunque nel totale
    res.append((t, float((d[band] < THR).mean()), float((d[far] < THR).mean()) if far.any() else float("nan")))
a = np.array(res)
sec = a[:, 0] / 24.0
def seg(lo, hi):
    k = (sec >= lo) & (sec < hi)
    return (float(np.nanmean(a[k, 1])) * 100, float(np.nanmean(a[k, 2])) * 100) if k.any() else (float("nan"), float("nan"))
print(f"fotogrammi misurati: {len(a)} su {n} (uno ogni 6), fascia {BAND}px, soglia diff<{THR}/255")
for lo, hi, nm in ((0, 2, "primi 2 s"), (2, 6, "2-6 s"), (6, 12, "6-12 s")):
    b, f = seg(lo, hi)
    print(f"  {nm:10s}: fascia attorno alla sagoma ~ sorgente {b:5.1f} %   zona lontana ~ sorgente {f:5.1f} %")
b, f = float(np.nanmean(a[:, 1])) * 100, float(np.nanmean(a[:, 2])) * 100
print(f"  TOTALE    : fascia {b:5.1f} %   lontano {f:5.1f} %")
