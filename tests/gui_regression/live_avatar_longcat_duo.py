"""Test LIVE dalla GUI del tab Avatar, LongCat 1.5 con DUE persone: widget veri, bottoni veri, generazione reale.

1. «Persone: due» + copione «Dorian: … / Kaira: …» nel pannello «Genera voce» (Kokoro im_nicola e if_sara,
   lingua del testo it-IT) → due tracce allineate caricate nei due player, modalità «insieme» (para);
2. Sfoglia → foto con i due volti affiancati (riquadri automatici sinistra/destra);
3. «Genera avatar parlante» → worker LongCat multi-persona.
Controlla: video prodotto, durata ≈ voce. Copia in output/DIAG_avatar_longcat_duo_*.mp4.

Uso:  QT_QPA_PLATFORM=offscreen python tests/gui_regression/live_avatar_longcat_duo.py [--res 480p]
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

DUO = ROOT / "output" / "DIAG_duo_dorian_kaira.png"
SCRIPT = ("Dorian: [emotion:excited]Ho comprato un drone per portare a spasso il cane![/emotion]\n"
          "Kaira: [emotion:surprised]Ma noi non abbiamo un cane.[/emotion]\n"
          "Dorian: [pause:300ms] [whisper]Adesso sì: l'ha trovato il drone.[/whisper]")
PROMPT = ("Two people sit side by side facing the camera and talk to each other with lively expressions, the man "
          "with round glasses on the left, the woman with dark hair tied back on the right, plain light background")


def _arg(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def _wait(pred, timeout_s: float, step_ms: int = 300) -> bool:
    app = H.app()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        app.processEvents()
        if pred():
            return True
        time.sleep(step_ms / 1000.0)
    return False


def _duration(path: str) -> float:
    r = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of",
                        "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def _set_speaker(box, name: str, engine: str, voice: str, lang: str):
    box.name.setText(name)
    box.engine.setCurrentIndex(box.engine.findData(engine))
    box.voice.setCurrentText(voice)
    box.language.setCurrentText(lang)


def main() -> int:
    res = _arg("--res", "480p")
    H.patch_dialogs()
    w = H.build_window()
    t = w.avatar_tab
    w.tabs.setCurrentWidget(t)
    _wait(lambda: False, 5)                               # ripristino differito dello stato GUI
    t.engine.setCurrentIndex(t.engine.findData("longcat"))
    t.mode_img.setChecked(True)
    t.people.setCurrentIndex(t.people.findData(2))
    H.app().processEvents()
    assert not t.audio2.isHidden(), "due persone: il secondo audio deve comparire"
    # 1) voce: copione a due
    t.voice_group.setChecked(True)
    t.voice_text.setPlainText(SCRIPT)
    _set_speaker(t.spk1, "Dorian", "kokoro", "im_nicola", "it-IT")
    _set_speaker(t.spk2, "Kaira", "kokoro", "if_sara", "it-IT")
    t.btn_voice.click()
    _wait(lambda: not t.btn_voice.isEnabled(), 10, 100)          # partito
    _wait(lambda: t.btn_voice.isEnabled(), 900)                     # finito (bene o male)
    if not t.audio2.value():
        print("FAIL voce non generata:", t.log.toPlainText()[-1200:]); return 1
    a1, a2 = t.audio1.value(), t.audio2.value()
    print(f"[live] tracce: {a1} ({_duration(a1):.2f}s) · {a2} ({_duration(a2):.2f}s) · modo {t.turns.currentData()}",
          flush=True)
    # 2) foto
    from PySide6.QtWidgets import QFileDialog
    orig = QFileDialog.getOpenFileName
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(DUO), ""))
    try:
        t.btn_img_browse.click()
    finally:
        QFileDialog.getOpenFileName = orig
    H.app().processEvents()
    assert t._image_path == str(DUO), "foto non caricata"
    # 3) genera
    t.prompt.setPlainText(PROMPT)
    t.resolution.setCurrentIndex(t.resolution.findData(res))
    t.seed_random.setChecked(False); t.seed.setValue(20260927)
    t0 = time.time()
    t.gen_btn.click()
    ok = _wait(lambda: t.gen_btn.isEnabled() and ("✓" in t.log.toPlainText() or "✗" in t.log.toPlainText()),
               3600, 1000)
    log = t.log.toPlainText()
    (ROOT / "output" / "DIAG_live_avatar_longcat_duo_tablog.txt").write_text(log, "utf-8")
    print(f"[live] fine in {time.time() - t0:.0f}s ok={ok}\n[live] log tab (coda):\n{log[-2500:]}", flush=True)
    out = getattr(t, "_last_video", "")
    if not (ok and out and os.path.isfile(out)):
        print("FAIL nessun video")
        return 1
    dst = ROOT / "output" / f"DIAG_avatar_longcat_duo_{res}.mp4"
    shutil.copyfile(out, dst)
    d, v = _duration(str(dst)), max(_duration(a1), _duration(a2))
    print(f"PASS video {dst} durata {d:.2f}s (voce {v:.2f}s)")
    return 0 if abs(d - v) < 0.6 else 2


if __name__ == "__main__":
    os._exit(main())
