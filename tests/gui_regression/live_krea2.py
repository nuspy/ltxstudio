"""Test LIVE dalla GUI di Krea 2 (tab Immagini): widget veri, bottone Genera vero, generazione reale su GPU.

Per ogni modello scelto nel dropdown Krea 2 (snapshot Turbo/Raw o single-file come Krea Noct V4 / Kroma) genera
un'immagine col prompt di prova; passi e guidance li imposta la GUI (Turbo e derivati: 8 / 0). Con --te usa il text
encoder alternativo (es. krea2-te-uncensored). Le voci 18+ richiedono la modalità NSFW, che il test attiva da sé.
Copia i risultati in output/DIAG_krea2_<modello>[_<te>].png.

Uso: QT_QPA_PLATFORM=offscreen python tests/gui_regression/live_krea2.py --models krea2-noct-v4,krea2-kroma-v031
     [--te krea2-te-uncensored]
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
sys.stdout.reconfigure(line_buffering=True)                  # os._exit scarta i buffer: righe subito

PROMPT = ("An amateur photograph of a woman in her thirties with short silver hair, sitting on the roof of a "
          "vintage camper van in a desert at sunrise, playing a ukulele to a curious roadrunner. Low warm sun from "
          "the left, long shadows, dust in the air. Full body from a low angle, phone camera look, natural skin.")


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


def run_one(w, model: str, te: str) -> str:
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("NSFW")                 # le voci 18+ (Noct, TE uncensored) compaiono solo in NSFW
    f.gen_arch.setCurrentText("Krea 2")
    i = f.gen_model.findData(model)
    if i < 0:
        print(f"[live] {model}: assente dal dropdown Krea 2", flush=True)
        return ""
    f.gen_model.setCurrentIndex(i)
    if te:
        j = f.k2_te.findData(te)
        assert j >= 0 and not f._k2_box.isHidden(), "encoder Krea 2 alternativo non selezionabile"
        f.k2_te.setCurrentIndex(j)
    f.gen_prompt.setPlainText(PROMPT)
    f.gen_seed_rand.setChecked(False); f.gen_seed.setValue(20261014)
    print(f"[live] {model}: passi={f.gen_steps.value()} guidance={f.gen_cfg.value()} te={te or 'ufficiale'}",
          flush=True)
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
    print(f"[live] {model}: {time.time() - t0:.0f}s ok={ok} stato={r.get('status')} out={got.get('out')}", flush=True)
    if r.get("status") != "success":
        print(f"[live] {model} ERRORE: " + str(r.get("error", ""))[-3000:], flush=True)
    out = got.get("out") or r.get("output") or ""
    if not out or not os.path.isfile(out):
        return ""
    dst = ROOT / "output" / f"DIAG_krea2_{model}{'_' + te if te else ''}.png"
    shutil.copyfile(out, dst)
    return str(dst)


def main() -> int:
    models = [m for m in _arg("--models", "krea2-noct-v4").split(",") if m]
    te = _arg("--te", "")
    w = H.build_window()
    w.tabs.setCurrentWidget(w.frame1_tab)
    try:                                              # 1024x1536 (verticale): formato consigliato dai finetune
        w.video_tab.width.setValue(1024); w.video_tab.height.setValue(1536)
    except Exception:                                        # noqa: BLE001
        pass
    res = [run_one(w, m, te) for m in models]
    print("RISULTATI:", res, flush=True)
    return 0 if all(res) else 1


if __name__ == "__main__":
    os._exit(main())
