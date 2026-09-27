"""Test LIVE dalla GUI di MiniMax H3 (tab Video): widget veri, bottone Genera vero, generazione reale su GPU.

Architettura «MiniMax H3», profilo nvfp4, turbo 8 passi (default), 5 s, una battuta in italiano con
emozione + suoni d'ambiente. Controlla: mp4 prodotto CON traccia audio, durata ≈ fotogrammi/24.
Copia il risultato in output/DIAG_h3_*.mp4 e il log GUI in output/DIAG_live_h3_guilog.txt.

Uso:  QT_QPA_PLATFORM=offscreen python tests/gui_regression/live_h3_video.py [--turbo 8|4|""] [--secs 5]
      [--image PATH] (i2v: foto come primo fotogramma)
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
sys.stdout.reconfigure(line_buffering=True)

PROMPT = ("Live-action, cinematic. A grumpy medieval knight in dented armour sits in a modern laundromat at "
          "night, watching his chainmail spin in a washing machine, fluorescent light, handheld medium shot.")
LINE = "[emotion:angry]Tre cicli e questa cotta di maglia puzza ancora di drago![/emotion]"


def _arg(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def _spin(secs: float):
    app = H.app()
    t0 = time.time()
    while time.time() - t0 < secs:
        app.processEvents()
        time.sleep(0.05)


def _probe(path: str) -> str:
    ff = shutil.which("ffprobe")
    if not ff:
        return "?"
    r = subprocess.run([ff, "-v", "error", "-show_entries", "stream=codec_type,width,height,sample_rate,channels:"
                        "format=duration", "-of", "compact", path], capture_output=True, text=True)
    return r.stdout.strip().replace("\n", " | ")


def main() -> int:
    turbo, secs, image = _arg("--turbo", "8"), float(_arg("--secs", "5")), _arg("--image", "")
    prompt, line = _arg("--prompt", PROMPT), _arg("--line", LINE)
    speaker, label = _arg("--speaker", "The knight"), _arg("--label", "knight")
    H.patch_dialogs()
    w = H.build_window()
    vt = w.video_tab
    w.tabs.setCurrentWidget(vt)
    _spin(5.0)                       # il ripristino differito dello stato GUI rimette l'arch salvata: aspetto
    vt.video_arch.setCurrentIndex(vt.video_arch.findData("minimax_h3"))
    _spin(1.0)
    i = next((j for j in range(vt.model.count()) if (vt.model.itemData(j) or {}).get("profile") == "nvfp4"), -1)
    assert i >= 0, "profilo nvfp4 assente dal dropdown"
    vt.model.setCurrentIndex(i)
    assert (vt.model.itemData(i) or {}).get("present"), "pesi nvfp4 non presenti"
    vt.h3_turbo.setCurrentIndex(vt.h3_turbo.findData(turbo))
    vt.h3_dialogue.clear()
    vt.h3_dialogue.add_row(speaker, "Italian", line)
    vt.h3_soundscape.setText("washing machines humming and thumping, fluorescent lights buzzing")
    vt.prompt.setPlainText(prompt)
    try:
        vt.res_preset.setCurrentText("Personalizzato")
    except Exception:                                        # noqa: BLE001
        pass
    vt.width.setValue(1344); vt.height.setValue(768)
    vt.length.setValue(secs)
    vt.native_audio.setChecked(True)
    if image:                                   # i2v: la foto nel canvas del tab Immagini = primo fotogramma
        H.load_canvas_image(w, image)
        H.app().processEvents()
        assert w.frame1_tab.image_path == image, "foto non caricata nel canvas"
    t0 = time.time()
    w.on_generate()
    _spin(3.0)
    while time.time() - t0 < 3600:
        if not w.jobs.is_busy():
            _spin(2.0)
            if not w.jobs.is_busy():
                break
        _spin(1.0)
    log = w.log.toPlainText()
    (ROOT / "output" / "DIAG_live_h3_guilog.txt").write_text(log, "utf-8")
    outs = sorted((p for p in (ROOT / "output").rglob("*.mp4")
                   if p.stat().st_mtime > t0 and "DIAG_" not in p.name), key=lambda p: p.stat().st_mtime)
    print(f"[live] fine in {time.time() - t0:.0f}s — log GUI (coda):\n{log[-3000:]}", flush=True)
    if not outs:
        print("FAIL nessun video")
        return 1
    dst = ROOT / "output" / f"DIAG_h3_{label}_{'t' + turbo if turbo else 'base'}_{int(secs)}s.mp4"
    shutil.copyfile(outs[-1], dst)
    info = _probe(str(dst))
    print(f"[live] {dst}\n[live] ffprobe: {info}")
    ok = "codec_type=audio" in info
    print("PASS" if ok else "FAIL: il video non ha la traccia audio")
    return 0 if ok else 2


if __name__ == "__main__":
    os._exit(main())
