"""Test LIVE dalla GUI di Qwen-Image 2.1 (tab Immagini): widget veri, bottone vero, generazione reale su GPU.

Prova 1: testo → immagine (profilo scelto, turbo opzionale).  Prova 2 (--edit): editing con un riferimento
(la foto del cast) → «stesso volto, capelli e vestiti diversi». Prova 3 (--rgba): adesivo con sfondo trasparente.
Copia i risultati in output/DIAG_qwen21_*.png.

Uso: QT_QPA_PLATFORM=offscreen python tests/gui_regression/live_qwen21.py [--profile nvfp4] [--turbo 6] [--edit] [--rgba]
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

FACE = ROOT / "output" / "film_studio" / "project_22" / "cast" / "Dorian.png"
T2I = ("A retired pirate captain with a grey braided beard teaches a parrot to play chess on the deck of a "
       "ship at golden hour, cinematic photo, detailed wood textures")
EDIT = ("The man in image 1 as a 1970s disco singer: big curly afro hair, shiny gold sequin jacket, "
        "on a stage with colorful spotlights. Keep his face and glasses exactly the same.")
RGBA = "A cute cartoon cactus wearing sunglasses and holding a tiny surfboard, sticker style"


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


def run_one(w, label: str, prompt: str, *, refs=(), rgba=False, profile="nvfp4", turbo="") -> str:
    f = w.frame1_tab
    f.gen_arch.setCurrentText("Qwen-Image 2.1")
    f.gen_model.setCurrentIndex(f.gen_model.findData(f"qwen21-dit-{profile}"))
    f.q21_turbo.setCurrentIndex(f.q21_turbo.findData(turbo))
    f.q21_aspect.setCurrentIndex(f.q21_aspect.findData(""))
    f.q21_rgba.setChecked(rgba)
    f.q21_refs.enable.setChecked(bool(refs))
    for p in refs:
        f.q21_refs.add_path(str(p))
    f.gen_prompt.setPlainText(prompt)
    f.gen_seed_rand.setChecked(False); f.gen_seed.setValue(20260926)
    got = {}
    w._show_inpaint_result = lambda out, *a, **k: got.setdefault("out", out)
    orig_done = type(w)._on_native_image_done

    def _done(res, _w=w):
        got["res"] = res
        orig_done(_w, res)
    w._on_native_image_done = _done
    t0 = time.time()
    w.generate_image()
    ok = _wait(lambda: "res" in got, 1800)
    r = got.get("res") or {}
    print(f"[live] {label}: {time.time() - t0:.0f}s ok={ok} stato={r.get('status')} out={got.get('out')}", flush=True)
    if r.get("status") != "success":
        print(f"[live] {label} ERRORE: " + str(r.get("error", ""))[-3000:], flush=True)
    out = got.get("out")
    if not out or not os.path.isfile(out):
        return ""
    dst = ROOT / "output" / f"DIAG_qwen21_{label}_{profile}{'_t' + turbo if turbo else ''}.png"
    shutil.copyfile(out, dst)
    return str(dst)


def main() -> int:
    profile, turbo = _arg("--profile", "nvfp4"), _arg("--turbo", "")
    w = H.build_window()
    w.tabs.setCurrentWidget(w.frame1_tab)
    # risoluzione del tab Video (usata come dimensioni se il formato 2K non è scelto)
    try:
        w.video_tab.width.setValue(1024); w.video_tab.height.setValue(1024)
    except Exception:                                        # noqa: BLE001
        pass
    res = []
    res.append(run_one(w, "t2i", T2I, profile=profile, turbo=turbo))
    if "--edit" in sys.argv:
        res.append(run_one(w, "edit", EDIT, refs=[FACE], profile=profile, turbo=turbo))
    if "--rgba" in sys.argv:
        res.append(run_one(w, "rgba", RGBA, rgba=True, profile=profile, turbo=turbo))
    print("RISULTATI:", res, flush=True)
    return 0 if all(res) else 1


if __name__ == "__main__":
    os._exit(main())
