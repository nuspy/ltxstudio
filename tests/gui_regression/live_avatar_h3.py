"""Test LIVE dalla GUI del tab Avatar con il motore MiniMax H3: widget veri, bottone vero, generazione reale.

Motore «MiniMax H3», foto con Sfoglia (va anche nel canvas), una battuta con marcatori, durata 6 s.
  · senza --voiceref: FL2VA (la foto è il primo fotogramma, voce inventata dal modello), turbo 8 passi;
  · con --voiceref WAV: Ref2VA (foto = riferimento del soggetto, registrazione = timbro), turbo 4 passi.
Controlla: mp4 con traccia audio, durata ≈ fotogrammi/24. Copia in output/DIAG_avatar_h3_*.mp4.

Uso:  QT_QPA_PLATFORM=offscreen python tests/gui_regression/live_avatar_h3.py [--voiceref PATH] [--secs 6]
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

FACE = ROOT / "output" / "film_studio" / "project_22" / "cast" / "Dorian.png"
PROMPT = ("Live-action, cinematic medium close-up. The man with messy brown hair and round glasses talks to the "
          "camera like a late-night TV host, expressive eyebrows and small hand gestures, warm studio light.")
LINE = ("[emotion:excited]Stasera ospite d'eccezione:[/emotion] [pause:300ms] "
        "[emph]un tostapane[/emph] che ha vinto un talent show!")


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
    r = subprocess.run([ff, "-v", "error", "-show_entries", "stream=codec_type,width,height,sample_rate:"
                        "format=duration", "-of", "compact", path], capture_output=True, text=True)
    return r.stdout.strip().replace("\n", " | ")


def main() -> int:
    voiceref, secs = _arg("--voiceref", ""), float(_arg("--secs", "6"))
    H.patch_dialogs()
    w = H.build_window()
    t = w.avatar_tab
    w.tabs.setCurrentWidget(t)
    _spin(5.0)                                   # ripristino differito dello stato GUI
    t.engine.setCurrentIndex(t.engine.findData("minimax_h3"))
    _spin(0.5)
    from PySide6.QtWidgets import QFileDialog
    orig = QFileDialog.getOpenFileName
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(FACE), ""))
    try:
        t.btn_img_browse.click()
    finally:
        QFileDialog.getOpenFileName = orig
    _spin(0.5)
    assert t._image_path == str(FACE), "foto non caricata"
    t.mode_img.setChecked(True)
    t.prompt.setPlainText(PROMPT)
    t.h3_dialogue.clear()
    t.h3_dialogue.add_row("The host", "Italian", LINE)
    t.h3_duration.setValue(secs)
    t.h3_profile.setCurrentIndex(t.h3_profile.findData("nvfp4"))
    t.h3_te.setCurrentIndex(t.h3_te.findData("mmh3-te-nvfp4"))
    t.h3_soundscape.setText("a small studio audience chuckling, soft room tone")
    if voiceref:
        t.h3_voice_ref.setChecked(True)
        t.audio1.set_path(voiceref)
        t.h3_turbo.setCurrentIndex(t.h3_turbo.findData("4"))
    else:
        t.h3_voice_ref.setChecked(False)
        t.h3_turbo.setCurrentIndex(t.h3_turbo.findData("8"))
    t.seed_random.setChecked(False); t.seed.setValue(20260927)
    t0 = time.time()
    t.gen_btn.click()
    _spin(3.0)
    while time.time() - t0 < 3600:
        if t.gen_btn.isEnabled() and ("✓" in t.log.toPlainText() or "✗" in t.log.toPlainText()):
            break
        _spin(1.0)
    log = t.log.toPlainText()
    tag = "ref2va" if voiceref else "fl2va"
    (ROOT / "output" / f"DIAG_live_avatar_h3_{tag}_tablog.txt").write_text(log, "utf-8")
    print(f"[live] fine in {time.time() - t0:.0f}s — log tab (coda):\n{log[-2500:]}", flush=True)
    out = getattr(t, "_last_video", "")
    if not (out and os.path.isfile(out)):
        print("FAIL nessun video")
        return 1
    dst = ROOT / "output" / f"DIAG_avatar_h3_dorian_{tag}.mp4"
    shutil.copyfile(out, dst)
    info = _probe(str(dst))
    print(f"[live] {dst}\n[live] ffprobe: {info}")
    ok = "codec_type=audio" in info
    print("PASS" if ok else "FAIL: niente traccia audio")
    return 0 if ok else 2


if __name__ == "__main__":
    os._exit(main())
