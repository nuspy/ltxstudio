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
    CASES = _all_cases()
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    results = []
    for name, should, _ in CASES:
        try:
            p = subprocess.run([sys.executable, "-u", str(HERE / "run_all.py"), "--case", name],
                               capture_output=True, text=True, timeout=PER_TEST_TIMEOUT, env=env)
            ok = p.returncode == 0
            detail = (p.stdout or "")[-300:] + (p.stderr or "")[-300:]
        except subprocess.TimeoutExpired:
            ok = False
            detail = f"TIMEOUT >{PER_TEST_TIMEOUT}s (handler bloccato)"
            H.write_result(name, should, False, detail)
        results.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"  ({detail.splitlines()[-1] if detail.strip() else '?'})"), flush=True)

    n_ok = sum(1 for _, ok in results if ok)
    lines = [f"REGRESSIONE GUI — {n_ok}/{len(results)} PASS", ""]
    lines += [f"{'PASS' if ok else 'FAIL'}  {n}" for n, ok in results]
    (H.OUT_DIR / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n== {n_ok}/{len(results)} PASS == (dettagli in output/tests/)", flush=True)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
