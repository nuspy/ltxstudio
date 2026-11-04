"""Smoke test dell'ambiente app-completa: boota la MainWindow VERA, verifica che l'InferenceServer :8189
(ComponentStore) sia SU e leggibile, stampa lo snapshot memoria, poi teardown pulito. Valida la harness
stessa prima di usarla per i test delle feature.

Uso:  <gui_python> tests/app_env/smoke.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))       # per `import app_env_harness`

import app_env_harness as H  # noqa: E402


def main() -> int:
    t0 = time.time()
    print("=== SMOKE app_env: boot full-app ===", flush=True)
    before = H.snapshot("boot-pre")
    H.print_snapshot(before)

    w = H.boot_full_app()
    up = H.snapshot("app-up")
    H.print_snapshot(up)

    # verifica chiave: l'InferenceServer :8189 (ComponentStore) è SU (i meccanismi di memoria sono reali)
    stats = H.server_stats(8189)
    ok_native = stats is not None
    print(f"[smoke] InferenceServer :8189 (ComponentStore) reachable: {ok_native} → {stats}", flush=True)

    final = H.teardown(w)
    H.print_snapshot(final)
    print(f"=== SMOKE {'OK' if ok_native else 'FAIL(:8189 non su)'} in {int(time.time()-t0)}s ===", flush=True)
    return 0 if ok_native else 2


if __name__ == "__main__":
    raise SystemExit(main())
