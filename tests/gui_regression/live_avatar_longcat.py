"""Test LIVE dalla GUI del tab Avatar (LongCat 1.5): widget veri, bottoni veri, generazione reale su GPU.

1. «Genera voce» nel tab (Kokoro, voce im_nicola) → la voce finisce nel player della persona 1;
2. Sfoglia → foto del volto (caricata anche nel canvas);
3. «Genera avatar parlante» → coda reale → worker LongCat nel suo venv → video nel pannello anteprima.
Controlla: video prodotto, durata ≈ durata della voce. Copia il risultato in output/DIAG_avatar_longcat_*.mp4.

Uso:  QT_QPA_PLATFORM=offscreen python tests/gui_regression/live_avatar_longcat.py [--res 480p] [--prec int8]
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

import harness as H                                           # noqa: E402
sys.stdout.reconfigure(line_buffering=True)                  # os._exit scarta i buffer: righe subito

FACE = ROOT / "output" / "film_studio" / "project_22" / "cast" / "Dorian.png"
VOICE_TEXT = ("Signore e signori, confesso tutto: ho insegnato al mio gatto a usare il telecomando. "
              "Adesso guarda solo documentari sui pesci e mi giudica quando cambio canale.")
PROMPT = ("A man with messy brown hair and round glasses talks to the camera with a dry, amused expression, "
          "black t-shirt, plain light background, natural head movements and subtle eyebrow raises, "
          "soft studio light, realistic")


def _arg(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def _wait(pred, timeout_s: float, step_ms: int = 200) -> bool:
    app = H.app()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        app.processEvents()
        if pred():
            return True
        time.sleep(step_ms / 1000.0)
    return False


def _duration(path: str) -> float:
    ff = shutil.which("ffprobe")
    if not ff:
        return 0.0
    r = subprocess.run([ff, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def main() -> int:
    res, prec = _arg("--res", "480p"), _arg("--prec", "int8")
    w = H.build_window()
    t = w.avatar_tab
    w.tabs.setCurrentWidget(t)
    H.app().processEvents()

    # 1) voce dal pannello del tab
    t.people.setCurrentIndex(t.people.findData(1))
    t.voice_group.setChecked(True)
    t.voice_text.setPlainText(VOICE_TEXT)
    t.spk1.engine.setCurrentIndex(t.spk1.engine.findData("kokoro"))
    t.spk1.voice.setCurrentText("im_nicola")
    t.btn_voice.click()
    if not _wait(lambda: t.btn_voice.isEnabled() and t.audio1.value(), 600):
        print("FAIL voce non generata:", t.log.toPlainText()[-800:]); return 1
    voice = t.audio1.value()
    vdur = _duration(voice)
    print(f"[live] voce: {voice} ({vdur:.2f}s) — {t.lbl_duration.text()}", flush=True)

    # 2) foto con Sfoglia (va anche nel canvas)
    from PySide6.QtWidgets import QFileDialog
    orig = QFileDialog.getOpenFileName
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(FACE), ""))
    try:
        t.btn_img_browse.click()
    finally:
        QFileDialog.getOpenFileName = orig
    H.app().processEvents()
    assert t._image_path == str(FACE) and w.frame1_tab.image_path == str(FACE), "foto non caricata"

    # 3) genera
    t.prompt.setPlainText(PROMPT)
    t.resolution.setCurrentIndex(t.resolution.findData(res))
    t.precision.setCurrentIndex(t.precision.findData(prec))
    t.seed_random.setChecked(False); t.seed.setValue(20260926)
    t0 = time.time()
    t.gen_btn.click()
    H.app().processEvents()
    ok = _wait(lambda: t.gen_btn.isEnabled() and ("✓" in t.log.toPlainText() or "✗" in t.log.toPlainText()),
               3600, 500)
    log = t.log.toPlainText()
    (ROOT / "output" / "DIAG_live_avatar_longcat_tablog.txt").write_text(log, "utf-8")
    out = getattr(t, "_last_video", "")
    print(f"[live] fine in {time.time() - t0:.0f}s ok={ok}", flush=True)
    print("[live] log tab (coda):\n" + log[-2500:], flush=True)
    if not (ok and out and os.path.isfile(out)):
        print("FAIL nessun video"); return 1
    dst = ROOT / "output" / f"DIAG_avatar_longcat_dorian_{res}_{prec}.mp4"
    shutil.copyfile(out, dst)
    d = _duration(str(dst))
    print(f"PASS video {dst} durata {d:.2f}s (voce {vdur:.2f}s, scarto {abs(d - vdur):.2f}s)")
    return 0 if abs(d - vdur) < 0.5 else 2


if __name__ == "__main__":
    rc = main()
    os._exit(rc)
