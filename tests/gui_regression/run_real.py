"""Runner REALE — esegue i casi della suite col BACKEND VERO (no mock), uno alla volta.

Differenza da run_all.py (mockato): qui `_enqueue` NON registra soltanto, ma ESEGUE davvero il worker
(generazione reale, GPU) e aspetta l'esito. Prima di ogni test libera la VRAM e logga RAM/VRAM, così
non si satura la memoria. Per ogni caso registra: esito della GUI (assert) + esito REALE (success/error)
+ file prodotto + RAM/VRAM.

Uso:
  QT_QPA_PLATFORM=offscreen <venv_gui_python> tests/gui_regression/run_real.py [--filter substr] [--start N] [--limit N]
Richiede l'InferenceServer nativo attivo (:8189).
"""
from __future__ import annotations

import os
import sys
import time
import traceback
import urllib.request
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
OUTDIR = ROOT / "output" / "tests" / "REAL"
OUTDIR.mkdir(parents=True, exist_ok=True)

import harness as H                       # noqa: E402  (riusa build helpers/dialoghi)
H.REAL_MODE = True                        # i casi comfy-mock (rig ecc.) skippano in run reale


def _ram_vram() -> str:
    import shutil  # noqa: F401
    s = ""
    try:
        import psutil
        vm = psutil.virtual_memory()
        s += f"RAM {vm.available/1e9:.1f}/{vm.total/1e9:.1f}GB liberi"
    except Exception:                       # noqa: BLE001
        s += "RAM ?"
    try:
        st = _server_stats()
        s += f" | VRAM {st.get('vram_free_mb',0)}/{st.get('vram_total_mb',0)}MB liberi tracked={st.get('tracked_mb',0)}"
    except Exception:                       # noqa: BLE001
        s += " | VRAM ?"
    return s


def _server_stats() -> dict:
    import json
    return json.load(urllib.request.urlopen("http://127.0.0.1:8189/system_stats", timeout=5))


def _free_vram():
    try:
        urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:8189/free", data=b"{}", headers={"Content-Type": "application/json"}), timeout=60).read()
    except Exception:                       # noqa: BLE001
        pass


def _build_real_window():
    """Come harness.build_window ma SENZA mockare il backend: solo dialoghi + no-autostart (server già su)."""
    H.app(); H.patch_dialogs()
    from ui.main_window import MainWindow
    MainWindow._maybe_autostart_native = lambda self: None
    MainWindow._maybe_autostart_comfyui = lambda self: None
    try:
        from core import model_precision as _mp
        _mp.gpu_free_gb = lambda: 30.0
    except Exception:                       # noqa: BLE001
        pass
    # NIENTE processEvents qui: altrimenti parte l'inventario reale dell'app (lista filtrata) e
    # scavalca l'inventario completo iniettato dai casi (_populate_inventory) → modelli "assenti".
    w = MainWindow()
    return w


class RealSpy:
    """Sostituisce _enqueue: registra la chiamata (così gli assert dei casi passano) E ESEGUE il worker
    davvero (sincrono), liberando la VRAM prima. Conserva l'esito reale in self.results."""

    def __init__(self, window):
        self.w = window
        self.calls = []
        self.results = []          # esiti reali dei worker eseguiti
        self._orig = None

    def __enter__(self):
        self._orig = self.w._enqueue
        def real_enqueue(label, worker, on_done=None, on_progress=None):
            self.calls.append({"via": "enqueue", "label": label,
                               "module": getattr(worker, "module", None),
                               "params": dict(getattr(worker, "params", {}) or {})})
            _free_vram()
            res = {}
            try:
                holder = {}
                if hasattr(worker, "done"):
                    try: worker.done.connect(lambda r: holder.update({"r": r}))
                    except Exception: pass   # noqa: BLE001,E701
                if hasattr(worker, "run"):
                    worker.run()             # SINCRONO: esegue davvero (genera) e emette done
                elif hasattr(worker, "start"):
                    worker.start(); worker.wait() if hasattr(worker, "wait") else None
                res = holder.get("r", {}) or {}
            except Exception as e:           # noqa: BLE001
                res = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            self.results.append({"label": label, "res": res})
            if on_done:
                try: on_done(res)
                except Exception: pass        # noqa: BLE001,E701
            return 0
        self.w._enqueue = real_enqueue
        return self

    def __exit__(self, *a):
        self.w._enqueue = self._orig
        return False


def _verdict(spy: RealSpy):
    """Esito complessivo: gli assert del caso (già verificati dal chiamante) + l'esito REALE dei worker.
    Ritorna (ok, detail)."""
    if not spy.results:
        return True, "nessuna generazione (caso UI/guardia) — ok se nessuna eccezione"
    errs = [r for r in spy.results if (r["res"] or {}).get("status") not in ("success", None)]
    outs = []
    for r in spy.results:
        o = (r["res"] or {}).get("output")
        if o: outs += (o if isinstance(o, list) else [o])
    if errs:
        return False, f"ERRORE REALE: {errs[0]['res'].get('error','')[:400]}"
    return True, f"output reale: {outs}"


def run_one(name, should, fn):
    pre = _ram_vram()
    _free_vram()
    rec = {"name": name, "ok": False, "detail": "", "ram": pre}
    try:
        w = _build_real_window()
        with RealSpy(w) as spy:
            fn(w, spy)                    # esegue setup + azione (con esecuzione REALE dentro _enqueue)
        ok, det = _verdict(spy)
        rec["ok"] = ok; rec["detail"] = det
    except AssertionError as e:
        rec["detail"] = f"ASSERT GUI: {e}"
    except Exception as e:                # noqa: BLE001
        rec["detail"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}"
    post = _ram_vram()
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)
    (OUTDIR / f"{safe}.txt").write_text(
        f"TEST: {name}\nSTATO: {'PASS ✅' if rec['ok'] else 'FAIL ❌'}\n\nCOSA DOVREBBE FARE:\n{should}\n\n"
        f"RAM/VRAM prima: {pre}\nRAM/VRAM dopo:  {post}\n\nESITO REALE:\n{rec['detail']}\n", encoding="utf-8")
    return rec


def main() -> int:
    import importlib
    cases = []
    for mod in sorted(p.stem for p in HERE.glob("cases*.py")):
        try:
            m = importlib.import_module(mod)
            cases += list(getattr(m, "CASES", []))
        except Exception as e:            # noqa: BLE001
            print(f"[warn] import {mod}: {e}", flush=True)
    if "--filter" in sys.argv:
        sub = sys.argv[sys.argv.index("--filter") + 1]
        cases = [c for c in cases if sub in c[0]]
    start = int(sys.argv[sys.argv.index("--start") + 1]) if "--start" in sys.argv else 0
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else len(cases)
    cases = cases[start:start + limit]

    results = []
    for i, (name, should, fn) in enumerate(cases):
        r = run_one(name, should, fn)
        results.append(r)
        print(f"[{('PASS' if r['ok'] else 'FAIL')}] {start+i+1}/{start+len(cases)} {name}  | {r['detail'].splitlines()[0][:80]}", flush=True)
    n_ok = sum(1 for r in results if r["ok"])
    summ = [f"RUN REALE — {n_ok}/{len(results)} PASS", ""]
    summ += [f"{'PASS' if r['ok'] else 'FAIL'}  {r['name']}  | {r['detail'].splitlines()[0][:70]}" for r in results]
    (OUTDIR / "REAL_SUMMARY.txt").write_text("\n".join(summ) + "\n", encoding="utf-8")
    print(f"\n== REALE {n_ok}/{len(results)} PASS ==", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
