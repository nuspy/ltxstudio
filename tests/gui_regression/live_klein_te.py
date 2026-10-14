"""Test LIVE dalla GUI dell'encoder di FLUX.2 Klein 9B (pannello Inpaint, menu «Encoder Klein»): stesso inpaint, stesso
seme, una volta con il Qwen3-8B ufficiale e una con l'uncensored (ponpoke). Widget veri, segnale vero del canvas,
generazione reale su GPU. Copia i risultati in output/DIAG_klein_te_<encoder>.png e stampa la differenza fra i due.

Uso: QT_QPA_PLATFORM=offscreen python tests/gui_regression/live_klein_te.py [--only official,uncensored]
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

import harness as H                                           # noqa: E402
from live_qwen21_edit import MAN, _arg, _hook, _wait           # noqa: E402
sys.stdout.reconfigure(line_buffering=True)

PROMPT = "a dark red velvet smoking jacket with gold embroidered dragons"
ENCODERS = {"official": "", "uncensored": "klein9b-te-uncensored"}


def run(w, key: str) -> str:
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QColor, QPainter
    f = w.frame1_tab
    f.sub.setCurrentIndex(1)
    f.canvas.load_image(str(MAN), fit_mode="canvas")
    i = next((k for k in range(f.inp_model.count()) if "miraclein" in f.inp_model.itemText(k).lower()), -1)
    assert i >= 0, "Klein 9B (miraclein) assente dal combo Inpaint"
    f.inp_model.setCurrentIndex(i)
    assert not f.inp_klein_te.isHidden(), "menu «Encoder Klein» nascosto con Klein 9B"
    f.inp_klein_te.setCurrentIndex(f.inp_klein_te.findData(ENCODERS[key]))
    f.inpaint_prompt.setPlainText(PROMPT)
    f.inp_seed_rand.setChecked(False); f.inp_seed.setValue(20261014)
    cv = f.canvas
    cv.mask_image.fill(QColor(0, 0, 0))
    pm = QPainter(cv.mask_image)                              # maschera: fascia del busto
    pm.fillRect(QRect(cv.aspect_w // 5, cv.aspect_h * 3 // 5, cv.aspect_w * 3 // 5, cv.aspect_h * 2 // 5 - 4),
                QColor(255, 255, 255))
    pm.end(); cv._refresh_mask()
    got = _hook(w); t0 = time.time()
    cv.request_inpaint.emit()                                 # = bottone «Inpaint»
    ok = _wait(lambda: "res" in got, 1800)
    r = got.get("res") or {}
    print(f"[live] {key}: {time.time() - t0:.0f}s ok={ok} stato={r.get('status')} out={got.get('out')}")
    if r.get("status") != "success":
        print(f"[live] {key} ERRORE: " + str(r.get("error", ""))[-3000:])
        return ""
    out = got.get("out")
    if not out or not os.path.isfile(out):
        return ""
    dst = ROOT / "output" / f"DIAG_klein_te_{key}.png"
    shutil.copyfile(out, dst)
    return str(dst)


def main() -> int:
    only = [k for k in _arg("--only", "official,uncensored").split(",") if k]
    w = H.build_window()
    w.tabs.setCurrentWidget(w.frame1_tab)
    res = {k: run(w, k) for k in only}
    print("RISULTATI:", res, flush=True)
    if all(res.values()) and len(res) == 2:
        from PIL import Image
        import numpy as np
        a, b = (np.asarray(Image.open(p).convert("RGB")).astype(float) for p in res.values())
        if a.shape == b.shape:
            print(f"[live] differenza media ufficiale/uncensored: {np.abs(a - b).mean():.2f} (0 = encoder ignorato)")
    return 0 if all(res.values()) else 1


if __name__ == "__main__":
    os._exit(main())
