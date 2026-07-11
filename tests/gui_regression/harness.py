"""Harness REGRESSIVO GUI-driven — verifica che la GUI mandi ISTRUZIONI CORRETTE al backend.

Problema da intercettare: anche se la pipeline funziona, dalla GUI possono partire istruzioni errate
(param mancanti/sbagliati, crash nell'handler). Qui ogni test:
  1) costruisce la MainWindow offscreen (o un singolo tab),
  2) imposta i campi del tab (modello, opzioni, comandi),
  3) invoca l'AZIONE (es. on_generate) — il backend è INTERCETTATO (spy): cattura (module, params)
     SENZA eseguire (niente GPU, veloce, parallelizzabile),
  4) verifica: nessuna eccezione + la richiesta catturata ha i campi richiesti e sensati.

I dialog (QMessageBox/QFileDialog/QInputDialog) sono auto-chiusi per non bloccare.
Risultati salvati in output/tests/<nome>.txt con: cosa-dovrebbe-fare + esito + dettaglio.

NON sostituisce gli smoke LIVE (generazione reale, 1 alla volta per la VRAM): quelli stanno in
tests/gui_regression/live_*.py.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "output" / "tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_APP = None


def app():
    global _APP
    from PySide6.QtWidgets import QApplication
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def patch_dialogs():
    """Rende non-bloccanti i dialog modali (altrimenti i test si piantano in offscreen)."""
    from PySide6.QtWidgets import QMessageBox, QFileDialog, QInputDialog, QDialog
    QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    QMessageBox.exec = lambda self, *a, **k: QMessageBox.Ok
    QMessageBox.exec_ = lambda self, *a, **k: QMessageBox.Ok
    QInputDialog.getText = staticmethod(lambda *a, **k: ("test", True))
    QInputDialog.getInt = staticmethod(lambda *a, **k: (1, True))
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: ("", ""))
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: ("", ""))
    QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: "")
    QDialog.exec = lambda self, *a, **k: 0          # i dialog (preview/conferma) → "annullato"
    QDialog.exec_ = lambda self, *a, **k: 0


class Spy:
    """Cattura le richieste GUI→backend senza eseguirle. Patcha _enqueue, ComfyDriver.submit,
    native_driver.run_and_wait/submit. `calls` = lista di dict {via, label, module, params}."""

    def __init__(self, window):
        self.window = window
        self.calls: list[dict] = []
        self._orig = {}

    def __enter__(self):
        w = self.window
        from core import native_driver

        def fake_enqueue(label, worker, on_done=None, on_progress=None):
            self.calls.append({"via": "enqueue", "label": label,
                               "module": getattr(worker, "module", None),
                               "params": dict(getattr(worker, "params", {}) or {})})
            return 0
        self._orig["_enqueue"] = w._enqueue
        w._enqueue = fake_enqueue

        if hasattr(w, "driver") and hasattr(w.driver, "submit"):
            self._orig["driver.submit"] = w.driver.submit
            def fake_submit(prompt, *a, **k):
                self.calls.append({"via": "comfy", "label": "driver.submit", "module": "comfy",
                                   "params": {"nodes": len(prompt) if hasattr(prompt, "__len__") else "?"}})
                return "spy-pid"
            w.driver.submit = fake_submit

        self._orig["nd.run_and_wait"] = native_driver.run_and_wait
        self._orig["nd.submit"] = native_driver.submit
        native_driver.run_and_wait = lambda module, params, **k: (
            self.calls.append({"via": "native_driver", "label": "run_and_wait",
                               "module": module, "params": dict(params or {})}) or {"status": "success", "output": []})
        native_driver.submit = lambda module, params, **k: (
            self.calls.append({"via": "native_driver", "label": "submit",
                               "module": module, "params": dict(params or {})}) or "spy-pid")
        return self

    def __exit__(self, *a):
        from core import native_driver
        w = self.window
        if "_enqueue" in self._orig:
            w._enqueue = self._orig["_enqueue"]
        if "driver.submit" in self._orig:
            w.driver.submit = self._orig["driver.submit"]
        native_driver.run_and_wait = self._orig["nd.run_and_wait"]
        native_driver.submit = self._orig["nd.submit"]
        return False


def build_window():
    """Costruisce la MainWindow offscreen (dialog patchati, autostart server DISABILITATO nei test)."""
    app()
    patch_dialogs()
    from ui.main_window import MainWindow
    MainWindow._maybe_autostart_native = lambda self: None    # no spawn server per-test (era lento/blocco)
    MainWindow._maybe_autostart_comfyui = lambda self: None
    # neutralizza i check che fanno RETE (download encoder/pesi) → i test verificano l'ISTRUZIONE, non i download
    MainWindow._ensure_flux2_encoder = lambda self, unet=None: True
    if hasattr(MainWindow, "_feature_gate"):
        MainWindow._feature_gate = lambda self, *a, **k: True
    # evita chiamate nvidia-smi ripetute nei combo precisione (VRAM costante nel test)
    try:
        from core import model_precision as _mp
        _mp.gpu_free_gb = lambda: 32.0
    except Exception:                                          # noqa: BLE001
        pass
    w = MainWindow()
    w.show()        # offscreen: necessario perché isVisible()/isVisibleTo() abbiano senso nei test UI
    return w


# ── helper per preparare precondizioni comuni ───────────────────────────────────
def make_test_image(path: str, w: int = 768, h: int = 512):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), (120, 150, 180))
    ImageDraw.Draw(im).ellipse([w//4, h//4, 3*w//4, 3*h//4], fill=(200, 120, 90))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return path


def load_canvas_image(window, path: str | None = None):
    """Carica una foto di test nel canvas (precondizione per inpaint/outpaint/integra/upscale)."""
    path = path or str(OUT_DIR / "_gui_test_input.png")
    if not os.path.exists(path):
        make_test_image(path)
    cv = window.frame1_tab.canvas
    # fit_mode esplicito: il default 'ask' legge user_prefs REALI (canvas_fit_mode ricordato) → test
    # non ermetico (canvas che resta es. 704×1280 invece di adottare le proporzioni della foto).
    cv.load_image(path, fit_mode="canvas")                # imposta canvas._image_path (→ frame1_tab.image_path)
    return path


# ── esecuzione + report ─────────────────────────────────────────────────────────
def write_result(name: str, should: str, ok: bool, detail: str = ""):
    """Salva l'esito in output/tests/<name>.txt (nome descrittivo + cosa dovrebbe fare + esito)."""
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)
    p = OUT_DIR / f"{safe}.txt"
    status = "PASS ✅" if ok else "FAIL ❌"
    p.write_text(
        f"TEST: {name}\nSTATO: {status}\n\nCOSA DOVREBBE FARE:\n{should}\n\nESITO/DETTAGLIO:\n{detail}\n",
        encoding="utf-8")
    return ok


def run_case(name: str, should: str, fn) -> dict:
    """Esegue un caso: fn(window, spy) deve impostare GUI + invocare l'azione. Verifica no-crash +
    almeno una richiesta catturata (o l'assert dentro fn). Ritorna dict esito."""
    res = {"name": name, "ok": False, "detail": ""}
    try:
        w = build_window()
        with Spy(w) as spy:
            fn(w, spy)
        # default: ok se non ha lanciato eccezioni (i casi possono aggiungere assert propri)
        res["ok"] = True
        res["detail"] = f"calls={spy.calls}"
    except AssertionError as e:
        res["detail"] = f"ASSERT: {e}\n{traceback.format_exc()[-1200:]}"
    except Exception as e:                                    # noqa: BLE001
        res["detail"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    write_result(name, should, res["ok"], res["detail"])
    return res
