"""VALIDAZIONE LIVE dell'unificazione pesi DALL'APP COMPLETA (branch feature/weights-unification).

Verifica che la migrazione dei path al REGISTRO UNICO non cambi nulla a runtime:
  1. l'app boota e il tab «Pesi modelli» v2 è popolato (categorie→arch, >90 voci);
  2. generazione IMMAGINE reale (FLUX via :8189) → i path migrati di image_flux
     (AE_FLUX/FLUX2_VAE/KLEIN dirs dal catalogo) reggono il job vero;
  3. generazione VIDEO breve reale (LTX via :8190) → i path migrati di ltx2_infer
     (CHECKPOINT/GEMMA/DISTILLED_LORA/IC_LORA dal catalogo) reggono il job vero.

Da lanciare A GPU LIBERA:  <gui_python> tests/app_env/test_weights_gui_live.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["NATIVE_PERSISTENT_CHILD"] = "1"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness as H  # noqa: E402

OUT = r"E:\ai\ltxstudio\output"


def main() -> int:
    print("=== UNIFICAZIONE PESI — LIVE DALL'APP COMPLETA ===", flush=True)
    w = H.boot_full_app()
    fails = []

    def check(name, cond):
        print(("OK  " if cond else "FAIL") + " " + name, flush=True)
        if not cond:
            fails.append(name)

    # 1) tab Pesi v2 dentro l'app VERA
    t = w.weights_tab
    check("tab Pesi v2 popolato (>90 voci)", len(t._checks) > 90)
    t.filter_txt.setText("ltx23")
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()
    check("filtro live funziona (ltx23)", 5 <= len(t._checks) <= 40)
    t.filter_txt.setText("")
    QApplication.processEvents()

    # 2) IMMAGINE reale (path image_flux dal registro)
    from core import native_driver
    p = {"arch": "flux", "model": "FLUX_Dev.safetensors",
         "prompt": "a giant mechanical whale surfacing through desert dunes at golden hour, cinematic",
         "negative": "blurry", "width": 768, "height": 768, "steps": 8, "guidance": 3.5,
         "seed": 4242, "output": os.path.join(OUT, "DIAG_weights_img.png")}
    t0 = time.time()
    r = native_driver.run_and_wait("image", p, timeout=600)
    check("immagine FLUX reale (path registro)", r.get("status") == "success")
    print(f"[img] {r.get('status')} in {time.time()-t0:.0f}s", flush=True)

    # 3) VIDEO breve reale (path ltx2_infer dal registro)
    from core import ltx2_service
    os.environ["LTX_ENGINE"] = "lightricks"
    pv = {"prompt": "a jazz-playing octopus in a smoky underwater club, lanterns swaying, whimsical",
          "negative": "worst quality, blurry", "width": 512, "height": 320, "num_frames": 33,
          "fps": 25, "steps": 8, "guidance": 3.0, "seed": 7, "arch": "ltx",
          "output": os.path.join(OUT, "DIAG_weights_vid.mp4")}
    t0 = time.time()
    rv = ltx2_service.generate(pv, timeout=900)
    check("video LTX reale (path registro)", rv.get("status") == "success"
          and rv.get("output") and os.path.exists(rv["output"][0]))
    print(f"[vid] {rv.get('status')} in {time.time()-t0:.0f}s", flush=True)

    H.teardown(w)
    ok = not fails
    print(("=== WEIGHTS GUI LIVE OK ===" if ok else f"=== FAIL: {fails} ==="), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
