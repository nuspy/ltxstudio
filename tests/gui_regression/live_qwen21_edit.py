"""Test LIVE dalla GUI delle funzioni di editing di Qwen-Image 2.1 (widget veri, segnali dei bottoni veri,
generazione reale su GPU col DiT NVFP4 già installato):

  A. img2img dal sotto-tab Genera (immagine iniziale + forza)
  B. inpaint con maschera dal sotto-tab Modifica (voce Qwen nel combo modelli): l'esterno della maschera
     deve restare IDENTICO all'originale e il risultato avere la risoluzione del ritaglio
  C. Componi con 2 riferimenti (motore Qwen-Image 2.1)
  D. Cambia colore (motore Qwen-Image 2.1)

Copia i risultati in output/DIAG_qwen21_<prova>.png.
Uso: QT_QPA_PLATFORM=offscreen python tests/gui_regression/live_qwen21_edit.py [--only A,B,C,D] [--dit qwen21-dit-nvfp4]
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
sys.stdout.reconfigure(line_buffering=True)

CAST = ROOT / "output" / "film_studio" / "project_22" / "cast"
MAN, WOMAN = CAST / "Dorian.png", CAST / "Kaira.png"
I2I = "The same scene as a Japanese ukiyo-e woodblock print, bold outlines, flat indigo and vermilion colors"
INP = "a bright yellow Hawaiian shirt covered in tiny pineapples"
COMP = ("The man in image 1 and the woman in image 2 dance an elegant tango in a rainy Buenos Aires street "
        "at night, neon reflections on the wet cobblestones")


def _arg(name: str, default: str = "") -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def _wait(pred, timeout_s: float) -> bool:
    app = H.app()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        app.processEvents()
        if pred():
            return True
        time.sleep(0.3)
    return False


def _hook(w):
    """Cattura l'esito dei due percorsi di completamento (generazione / modifica) senza dialog."""
    got = {}
    w._show_inpaint_result = lambda out, *a, **k: got.setdefault("out", out)
    for name in ("_on_native_image_done", "_on_native_inpaint_done"):
        orig = getattr(type(w), name)

        def _done(res, _w=w, _orig=orig):
            got["res"] = res
            if res.get("status") == "success" and res.get("output"):
                got.setdefault("out", res["output"][0])
            _orig(_w, res)
        setattr(w, name, _done)
    return got


def _finish(label: str, got: dict, t0: float) -> str:
    ok = _wait(lambda: "res" in got, 1800)
    r = got.get("res") or {}
    print(f"[live] {label}: {time.time() - t0:.0f}s ok={ok} stato={r.get('status')} out={got.get('out')}")
    if r.get("status") != "success":
        print(f"[live] {label} ERRORE: " + str(r.get("error", ""))[-3000:])
        return ""
    out = got.get("out")
    if not out or not os.path.isfile(out):
        return ""
    dst = ROOT / "output" / f"DIAG_qwen21_{label}.png"
    shutil.copyfile(out, dst)
    return str(dst)


def test_img2img(w, dit):
    f = w.frame1_tab
    f.sub.setCurrentIndex(0)
    f.gen_arch.setCurrentText("Qwen-Image 2.1")
    f.gen_model.setCurrentIndex(f.gen_model.findData(dit))
    f.q21_refs.enable.setChecked(False)
    f.q21_i2i.setChecked(True)
    f.q21_i2i_photo["edit"].setText(str(MAN))
    f.q21_strength.setValue(int(_arg("--strength", "60")))
    f.gen_prompt.setPlainText(I2I)
    f.gen_seed_rand.setChecked(False); f.gen_seed.setValue(20261012)
    got = _hook(w); t0 = time.time()
    f.gen_btn.click()                                          # bottone vero
    out = _finish(f"img2img_f{_arg('--strength', '60')}", got, t0)
    f.q21_i2i.setChecked(False)
    return out


def test_inpaint(w, dit):
    from PIL import Image, ImageChops
    import numpy as np
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QColor, QPainter
    f = w.frame1_tab
    f.sub.setCurrentIndex(1)
    f.canvas.load_image(str(MAN), fit_mode="canvas")
    label = next(l for l, d in f._qwen21_inp.items() if d == dit)
    f.inp_model.setCurrentIndex(f.inp_model.findText(label))
    f.inpaint_prompt.setPlainText(INP)
    f.inp_seed_rand.setChecked(False); f.inp_seed.setValue(20261012)
    cv = f.canvas
    pm = QPainter(cv.mask_image)                              # maschera: fascia del busto
    pm.fillRect(QRect(cv.aspect_w // 5, cv.aspect_h * 3 // 5, cv.aspect_w * 3 // 5, cv.aspect_h * 2 // 5 - 4),
                QColor(255, 255, 255))
    pm.end(); cv._refresh_mask()
    got = _hook(w); t0 = time.time()
    cv.request_inpaint.emit()                                 # = bottone «Inpaint»
    out = _finish("inpaint", got, t0)
    if out:
        exp_img = Image.open(ROOT / "output" / "frame1.png").convert("RGB")
        res = Image.open(out).convert("RGB")
        mask = Image.open(ROOT / "output" / "frame1_mask.png").convert("L").resize(res.size)
        far = np.asarray(mask.filter(__import__("PIL.ImageFilter", fromlist=["x"]).MaxFilter(61))) < 8
        diff = np.asarray(ImageChops.difference(exp_img.resize(res.size), res)).max(axis=2)
        print(f"[live] inpaint: dimensioni {res.size} vs ritaglio {exp_img.size}; "
              f"max diff lontano dalla maschera = {int(diff[far].max()) if far.any() else -1}")
    return out


def test_compose(w, dit):
    f = w.frame1_tab
    f.sub.setCurrentIndex(1)
    f.integrate_engine.setCurrentIndex(3)
    f.integrate_model.setCurrentIndex(f.integrate_model.findData(dit))
    f.compose_list.clear()
    f.compose_list.addItem(str(MAN)); f.compose_list.addItem(str(WOMAN))
    f.compose_prompt.setPlainText(COMP)
    got = _hook(w); t0 = time.time()
    f.request_compose.emit()                                  # = bottone «Componi»
    return _finish("componi", got, t0)


def test_recolor(w, dit):
    f = w.frame1_tab
    f.sub.setCurrentIndex(2)
    f.canvas.load_image(str(MAN), fit_mode="canvas")
    f.color_engine.setCurrentIndex(3)
    f.color_model.setCurrentIndex(f.color_model.findData(dit))
    f.color_target.setText("emerald green"); f.color_subject.setText("his jacket")
    got = _hook(w); t0 = time.time()
    f.color_btn.click()                                       # bottone vero
    return _finish("colore", got, t0)


def main() -> int:
    dit = _arg("--dit", "qwen21-dit-nvfp4")
    only = set((_arg("--only", "A,B,C,D")).split(","))
    w = H.build_window()
    w.tabs.setCurrentWidget(w.frame1_tab)
    res = {}
    for key, fn in (("A", test_img2img), ("B", test_inpaint), ("C", test_compose), ("D", test_recolor)):
        if key in only:
            res[key] = fn(w, dit)
    print("RISULTATI:", res)
    return 0 if all(res.values()) else 1


if __name__ == "__main__":
    os._exit(main())
