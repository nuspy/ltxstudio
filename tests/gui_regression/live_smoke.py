"""Smoke LIVE — generazione VERA pilotata dalla GUI (backend reale, GPU), un modello alla volta.

A differenza di run_all.py (mockato, verifica l'ISTRUZIONE), qui NON si mocka il backend: si pilota la
GUI, parte la generazione reale e si ASPETTA il file prodotto in output/. Serve a vedere output reali
(immagini/video) per ogni modello. VRAM-safe: seriale (una generazione alla volta).

Uso:
  QT_QPA_PLATFORM=offscreen <venv_gui_python> tests/gui_regression/live_smoke.py [--kind image|video] [--limit N]
Richiede l'InferenceServer nativo attivo (:8189).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "output"
MANIFEST = OUT / "tests" / "LIVE_MANIFEST.txt"

from PySide6.QtWidgets import QApplication, QMessageBox, QDialog, QFileDialog   # noqa: E402


def _patch_dialogs():
    QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    QMessageBox.exec = lambda self, *a, **k: QMessageBox.Ok
    QDialog.exec = lambda self, *a, **k: 0          # preview/conferma → non bloccare
    QDialog.exec_ = lambda self, *a, **k: 0
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: ("", ""))


def _free_vram():
    """Libera la VRAM sul server prima di ogni generazione (test 1-modello-alla-volta, VRAM-safe)."""
    import urllib.request
    try:
        urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:8189/free", data=b"{}", headers={"Content-Type": "application/json"}), timeout=30).read()
    except Exception:                                         # noqa: BLE001
        pass


def _snapshot(patterns):
    s = set()
    for p in patterns:
        s |= set(str(x) for x in OUT.glob(p))
    return s


def _wait_new_file(app, patterns, before, timeout):
    """Spinna l'event loop finché compare un nuovo file (rispetto a `before`) o scade il timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.4)
        new = _snapshot(patterns) - before
        # considera completo solo se il file non sta più crescendo
        done = [f for f in new if os.path.exists(f) and os.path.getsize(f) > 1024]
        if done:
            time.sleep(0.6)
            return sorted(done, key=os.path.getmtime)[-1], time.time() - t0
    return None, time.time() - t0


def _concrete_image_models(combo):
    out = []
    for i in range(combo.count()):
        d = combo.itemData(i); txt = combo.itemText(i)
        if d and "auto" not in str(txt).lower() and "nessun" not in str(txt).lower():
            out.append((i, txt))
    return out


def main() -> int:
    kind = sys.argv[sys.argv.index("--kind") + 1] if "--kind" in sys.argv else "image"
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 999
    app = QApplication.instance() or QApplication([])
    _patch_dialogs()
    from ui.main_window import MainWindow
    MainWindow._maybe_autostart_native = lambda self: None        # server già attivo
    MainWindow._maybe_autostart_comfyui = lambda self: None
    w = MainWindow(); w.show()
    app.processEvents()

    results = []
    if kind == "image":
        f = w.frame1_tab
        f.prompt.setPlainText("a photorealistic red apple on a wooden table, studio light")
        # FLUX.2 Dev ORA nativo (block-swap) → incluso; solo Klein/miraclein è escluso (manca Qwen3-8B).
        allm = [(i, t) for i, t in _concrete_image_models(f.gen_model)
                if "klein" not in t.lower() and "miraclein" not in t.lower()]
        f2 = [(i, t) for i, t in allm if "flux_2" in t.lower() or "flux2" in t.lower()]
        rest = [(i, t) for i, t in allm if (i, t) not in f2]
        models = (rest[:max(0, limit - len(f2))] + f2)[:limit] if limit < 999 else allm
        for idx, txt in models:
            f.gen_model.setCurrentIndex(idx)
            app.processEvents()
            _free_vram()                                      # VRAM pulita per ogni modello
            before = _snapshot(["*.png"])
            t0 = time.time()
            try:
                w.generate_image()
            except Exception as e:                                # noqa: BLE001
                results.append((txt, "ERR-submit", str(e)[:120], 0)); continue
            # FLUX.2 Dev (33GB in block-swap su GPU 32GB) ~22s/step ad alta risoluzione → ~12min/immagine.
            # Timeout ampio per non dare falsi TIMEOUT (l'immagine ESCE, solo lenta). Gli altri finiscono prima.
            _to = 1200 if ("flux_2" in txt.lower() or "flux2" in txt.lower()) else 240
            fpath, dt = _wait_new_file(app, ["*.png"], before, timeout=_to)
            if fpath:
                mb = os.path.getsize(fpath) / 1e6
                results.append((txt, "OK", f"{Path(fpath).name} ({mb:.2f}MB)", dt))
            else:
                results.append((txt, "TIMEOUT", "nessun file in 240s", dt))
            print(f"[{results[-1][1]}] {txt} → {results[-1][2]} ({results[-1][3]:.0f}s)", flush=True)

    lines = [f"SMOKE LIVE ({kind}) — {sum(1 for r in results if r[1]=='OK')}/{len(results)} OK", ""]
    for txt, st, det, dt in results:
        lines.append(f"{st:8} {txt}  →  {det}  [{dt:.0f}s]")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:1]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
