"""Runner della suite di regressione GUI-driven. Esegue tutti i CASES, salva ogni esito in
output/tests/<nome>.txt e un riepilogo SUMMARY.txt. Uso:
  <venv_gui_python> tests/gui_regression/run_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # per import flat di harness/cases

import harness as H                                    # noqa: E402
from cases import CASES                                # noqa: E402


def main() -> int:
    results = []
    for name, should, fn in CASES:
        r = H.run_case(name, should, fn)
        flag = "PASS" if r["ok"] else "FAIL"
        print(f"[{flag}] {name}", flush=True)
        if not r["ok"]:
            print("       " + r["detail"].splitlines()[0], flush=True)
        results.append(r)
    n_ok = sum(1 for r in results if r["ok"])
    summary = [f"REGRESSIONE GUI — {n_ok}/{len(results)} PASS", ""]
    for r in results:
        summary.append(f"{'PASS' if r['ok'] else 'FAIL'}  {r['name']}")
        if not r["ok"]:
            summary.append("      " + r["detail"].splitlines()[0])
    (H.OUT_DIR / "SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"\n== {n_ok}/{len(results)} PASS ==  (dettagli in output/tests/)", flush=True)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
