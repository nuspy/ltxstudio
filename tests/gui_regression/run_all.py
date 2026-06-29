"""Runner regressione GUI — ISOLAMENTO per-test (ogni caso in un subprocess con timeout).

Perché subprocess: costruire la MainWindow per ogni caso è pesante e un singolo handler che si
impianta bloccherebbe tutta la suite. Isolando, un test che si blocca viene ucciso dal suo timeout
e marcato FAIL('timeout'); gli altri proseguono. Inoltre i test live (GPU) restano 1-alla-volta.

Uso:
  <py> tests/gui_regression/run_all.py              # orchestratore: lancia ogni caso isolato
  <py> tests/gui_regression/run_all.py --case NAME  # esegue UN caso (usato dall'orchestratore)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

PER_TEST_TIMEOUT = 90          # s per test (mock, no GPU); i live hanno il loro runner


def _all_cases():
    """Aggrega i CASES da cases.py + ogni cases_*.py (un file per tab, generati dallo swarm)."""
    import importlib
    out = []
    for mod in sorted(p.stem for p in HERE.glob("cases*.py")):
        try:
            m = importlib.import_module(mod)
            out += list(getattr(m, "CASES", []))
        except Exception as e:                               # noqa: BLE001
            print(f"[warn] import {mod}: {e}", flush=True)
    return out


def _run_single(name: str) -> int:
    import harness as H
    CASES = _all_cases()
    case = next((c for c in CASES if c[0] == name), None)
    if case is None:
        print(f"NOCASE {name}", flush=True); return 2
    r = H.run_case(case[0], case[1], case[2])
    print(("PASS " if r["ok"] else "FAIL ") + name, flush=True)
    return 0 if r["ok"] else 1


def main() -> int:
    if "--case" in sys.argv:
        return _run_single(sys.argv[sys.argv.index("--case") + 1])

    import harness as H
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    CASES = _all_cases()
    only = None
    if "--filter" in sys.argv:
        only = sys.argv[sys.argv.index("--filter") + 1]
        CASES = [c for c in CASES if only in c[0]]
    workers = 6
    if "--jobs" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--jobs") + 1])
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")

    def _one(item):
        name, should, _ = item
        try:
            p = subprocess.run([sys.executable, "-u", str(HERE / "run_all.py"), "--case", name],
                               capture_output=True, text=True, timeout=PER_TEST_TIMEOUT, env=env)
            return name, p.returncode == 0, ((p.stdout or "")[-200:] + (p.stderr or "")[-200:])
        except subprocess.TimeoutExpired:
            H.write_result(name, should, False, f"TIMEOUT >{PER_TEST_TIMEOUT}s")
            return name, False, f"TIMEOUT >{PER_TEST_TIMEOUT}s"

    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, c): c[0] for c in CASES}
        for fut in as_completed(futs):
            name, ok, detail = fut.result()
            done += 1
            results.append((name, ok))
            if not ok:
                print(f"[FAIL {done}/{len(CASES)}] {name}  ({(detail.strip().splitlines() or ['?'])[-1][:90]})", flush=True)
    results.sort()

    n_ok = sum(1 for _, ok in results if ok)
    lines = [f"REGRESSIONE GUI — {n_ok}/{len(results)} PASS", ""]
    lines += [f"{'PASS' if ok else 'FAIL'}  {n}" for n, ok in results]
    (H.OUT_DIR / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n== {n_ok}/{len(results)} PASS == (dettagli in output/tests/)", flush=True)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
