import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "app_live"))
import measure_lag as L  # noqa: E402


def test_lag_series_finds_the_known_delay(tmp_path):
    n, h, w = 60, 60, 160
    rng = np.random.default_rng(0)
    src = (rng.random((n, h, w, 3)) * 255).astype(np.uint8)
    masks = []
    for i in range(n):                                 # persona (quadrato) che scorre di 2 px per frame
        m = np.zeros((h, w), np.uint8); x = 5 + 2 * i; m[10:40, x:x + 24] = 255
        p = tmp_path / f"{i:04d}.png"; Image.fromarray(m).save(p); masks.append(str(p))
    gen = (rng.random((n, h, w, 3)) * 255).astype(np.uint8)               # sfondo nuovo, diverso dalla sorgente
    for t in range(n):                                  # generato = sorgente con il fantasma in ritardo di 10 frame (fuori dalla persona vera)
        s = max(0, t - 10)
        mm = np.array(Image.open(masks[s]).convert("L")) > 127
        gen[t][mm] = src[s][mm]
    ser = L.lag_series(gen, src, masks, step=4)
    sm = L.summary(ser)
    assert sm["n_fantasma"] > 3 and sm["k_mediana"] == -10, (sm, ser[:4])
