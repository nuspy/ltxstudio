"""Test di plan_krea2_placement (unificazione, commit 3) — funzione pura, senza torch/GPU."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.vram_calibration import plan_krea2_placement as P  # noqa: E402


def run() -> int:
    fails = []

    def check(name, cond):
        print(("OK  " if cond else "FAIL") + " " + name, flush=True)
        if not cond:
            fails.append(name)

    # DB muto → legacy (soglie storiche, comportamento invariato alla prima run)
    check("DB vuoto → legacy", P(31000, 0, 0) == "legacy")
    # bf16 entra → qualità piena (26GB bf16, 30GB liberi)
    check("bf16 entra → cuda_bf16", P(30000, 26000, 13000) == "cuda_bf16")
    # bf16 non entra, fp8 sì (26GB bf16, 13GB fp8, 17GB liberi)
    check("solo fp8 entra → cuda_fp8", P(17000, 26000, 13000) == "cuda_fp8")
    # niente entra → offload (10GB liberi)
    check("niente entra → offload_fp8", P(10000, 26000, 13000) == "offload_fp8")
    # solo fp8 noto e entra
    check("solo fp8 noto → cuda_fp8", P(17000, 0, 13000) == "cuda_fp8")
    # solo fp8 noto e NON entra
    check("solo fp8 noto, non entra → offload_fp8", P(9000, 0, 13000) == "offload_fp8")
    # working di default 3000: 13000+3000=16000 > 15999 → offload
    check("margine working rispettato", P(15999, 0, 13000) == "offload_fp8")

    print(f"\n{'TUTTO OK' if not fails else 'FALLITI: ' + ', '.join(fails)}")
    return len(fails)


if __name__ == "__main__":
    raise SystemExit(run())
