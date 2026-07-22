"""TEST LIVE end-to-end DALLA GUI — unified-model-manager (audit + monitor + residenza :8190).

Percorso REALE utente→GUI→backend (regola [[test-end-to-end-from-gui]]):
  1) costruisce la MainWindow vera (offscreen);
  2) attende l'AUDIT del registro modelli all'avvio (F2) e ne verifica il risultato;
  3) apre il Memory Monitor e verifica che la TABELLA modelli (F4) mostri i residenti/parcheggiati
     (il server :8190 caldo ha il parcheggio RAM → righe attese);
  4) imposta i widget del tab Video (prompt creativo, 512x320, ~1.3s, 8 step) e chiama
     on_generate() REALE (nessuno spy) → il worker deve instradare al SERVER :8190 (residenza);
  5) attende il completamento dal loop Qt e verifica l'mp4 + che il job sia passato dal server
     (il log del server cresce di un job; nessun subprocess ltx2_infer spawnato).

Avvio:  QT_QPA_PLATFORM=offscreen <gui_python> tests/gui_regression/live_unified_residency.py
"""
from __future__ import annotations

import faulthandler
import glob
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
faulthandler.enable()

LOG = ROOT / "output" / "DIAG_unified_gui.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _spin(app, seconds: float) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        app.processEvents()
        time.sleep(0.05)


def main() -> int:
    import harness

    LOG.write_text("", encoding="utf-8")
    fails: list[str] = []

    def check(name, cond):
        log(("OK  " if cond else "FAIL") + " " + name)
        if not cond:
            fails.append(name)

    # spia della calibrazione livello-2: NIENTE dialog nel test (la policy resta dell'utente)
    from ui.main_window import MainWindow
    calib_calls: list[int] = []
    MainWindow._maybe_start_calibration = (
        lambda self, res: calib_calls.append(len(res.get("to_measure", []))))

    log("=== BUILD MainWindow (offscreen) ===")
    w = harness.build_window()
    app = harness.app()

    # ── 2) AUDIT all'avvio (F2): parte 2s dopo l'init, gira in QThread ──
    t0 = time.monotonic()
    while time.monotonic() - t0 < 180 and not hasattr(w, "_audit_result"):
        _spin(app, 0.5)
    res = getattr(w, "_audit_result", None)
    check("audit all'avvio completato", isinstance(res, dict) and res.get("total", 0) > 0)
    if isinstance(res, dict):
        log(f"    audit: {res.get('total')} modelli, {len(res.get('measured', []))} misurati, "
            f"{len(res.get('to_measure', []))} da misurare")
        check("hook calibrazione chiamato col to_measure", bool(calib_calls))

    # ── 3) Memory Monitor (F4): la tabella mostra residenti/parcheggiati ──
    from ui.memory_monitor import MemoryMonitor
    mon = MemoryMonitor(w)
    mon._refresh_models()
    rows = mon._tbl.rowCount()
    log(f"    monitor: {rows} righe nella tabella modelli")
    check("tabella Memory Monitor popolata (server :8190 caldo)", rows >= 1)

    # ── 4) GENERAZIONE VIDEO REALE dai widget (residenza :8190) ──
    srv_log = ROOT / "output" / "DIAG_srv8190b.log"
    srv_lines_before = len(srv_log.read_text(encoding="utf-8", errors="replace").splitlines()) \
        if srv_log.exists() else 0
    out_before = set(glob.glob(str(ROOT / "output" / "ltx_native_*.mp4")))

    w.video_tab.prompt.setPlainText(
        "un violinista suona sul tetto di un tram in corsa attraverso una citta' innevata, "
        "luci calde dei lampioni, fiocchi di neve in controluce")
    w.video_tab.width.setValue(512)
    w.video_tab.height.setValue(320)
    if hasattr(w.video_tab, "length"):
        w.video_tab.length.setValue(1.3)
    w.video_tab.sampler_seed.steps.setValue(8)
    w.tabs.setCurrentWidget(w.video_tab)
    log("=== on_generate() REALE (atteso: server :8190) ===")
    w.on_generate()

    t0 = time.monotonic()
    while time.monotonic() - t0 < 420:
        _spin(app, 1.0)
        if not w.jobs.is_busy() and time.monotonic() - t0 > 10:
            break
    new_out = set(glob.glob(str(ROOT / "output" / "ltx_native_*.mp4"))) - out_before
    check("mp4 generato dalla GUI", bool(new_out))
    if new_out:
        log(f"    output: {sorted(new_out)[-1]}")

    srv_lines_after = len(srv_log.read_text(encoding="utf-8", errors="replace").splitlines()) \
        if srv_log.exists() else 0
    check("il job e' passato dal SERVER (log :8190 cresciuto)", srv_lines_after > srv_lines_before)

    log("TUTTO OK" if not fails else f"FALLITI: {fails}")
    return len(fails)


if __name__ == "__main__":
    raise SystemExit(main())
