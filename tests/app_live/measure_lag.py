"""Ritardo TEMPORALE k(t) della persona RIDISEGNATA nel clip 1 di un edit H3 (cambia-sfondo): per ogni frame t del generato (gen.npy), lo scarto k in [-24, 24]
per cui la persona sorgente al frame t+k combacia col generato nella zona dove c'e' la persona sfasata ma NON la persona vera (fuori dalla sagoma al frame t).
Uso: python measure_lag.py <cartella clip_00 con gen.npy e clip_src.npy> <cartella maschere sorgente>.  Stampa k(t) e, in fondo, mediana/media di k
sui frame «FANTASMA» (diff migliore < 0,75 x mediana) e la quota di frame col fantasma. k vicino a 0 = persona ridisegnata sincrona."""
import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image


def lag_series(gen, src, masks, step: int = 4):
    n, h, w = gen.shape[:3]
    M = np.stack([cv2.resize(np.array(Image.open(masks[i]).convert("L")), (w, h), interpolation=cv2.INTER_NEAREST) > 127 for i in range(min(n, len(masks)))])
    n = M.shape[0]
    out = []
    for t in range(4, n - 4, step):
        true = cv2.dilate(M[t].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        row = {}
        for k in range(-24, 25):
            s = t + k
            if s < 0 or s >= n:
                continue
            zone = M[s] & ~true
            if zone.sum() < 400:
                continue
            row[k] = float(np.abs(gen[t].astype(np.int16) - src[s].astype(np.int16)).mean(-1)[zone].mean())
        if not row:
            continue
        kb = min(row, key=row.get)
        med = float(np.median(list(row.values())))
        out.append((t, kb, row[kb], med, row[kb] < 0.75 * med))
    return out


def summary(series):
    ks = [k for _, k, _, _, g in series if g]
    return {"n": len(series), "n_fantasma": len(ks), "k_mediana": float(np.median(ks)) if ks else None, "k_media": float(np.mean(ks)) if ks else None}


def main(argv=None) -> int:
    a = argv or sys.argv[1:]
    gen = np.load(os.path.join(a[0], "gen.npy"), mmap_mode="r")
    src = np.load(os.path.join(a[0], "clip_src.npy"), mmap_mode="r")
    masks = sorted(glob.glob(os.path.join(a[1], "*.png")))
    ser = lag_series(gen, src, masks)
    for t, k, d, med, g in ser:
        print(f"t={t:3d} ({t / 24:4.1f}s)  k={k:+3d}  diff={d:5.1f}  mediana={med:5.1f}  {'FANTASMA' if g else ''}")
    s = summary(ser)
    print(f"RIEPILOGO k(t): frame {s['n']}, col fantasma {s['n_fantasma']}, k mediana {s['k_mediana']}, k media {s['k_media']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
