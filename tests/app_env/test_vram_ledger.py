"""Test del ledger VRAM live (core/vram_ledger, F3 unified-model-manager) — senza GPU.
Valida: register/unregister/set_state, snapshot che scarta e CANCELLA i PID morti, voci stantie,
concorrenza multi-processo (sharding per-PID: zero contese)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_REPO = str(Path(__file__).resolve().parents[2])


def run() -> int:
    fails = []

    def check(name, cond):
        print(("OK  " if cond else "FAIL") + " " + name, flush=True)
        if not cond:
            fails.append(name)

    tmp = tempfile.mkdtemp(prefix="vram_ledger_")
    os.environ["LTX_VRAM_LEDGER_DIR"] = tmp
    from core import vram_ledger as L
    L._DIR = tmp                                            # il modulo può essere già importato

    # 1) register/snapshot/set_state/unregister
    L.register("flux_tr:test", 14000, category="image", state="vram", path="E:/fake.safetensors")
    L.register("t5xxl", 9100, category="aux", state="vram")
    snap = L.snapshot()
    check("register+snapshot (2 voci)", len(snap) == 2 and snap[0]["mb"] == 14000)
    check("total_mb vram", L.total_mb() == 23100)
    L.set_state("t5xxl", "ram_parked")
    check("set_state → ram_parked", L.total_mb(state="ram_parked") == 9100 and L.total_mb() == 14000)
    L.unregister("flux_tr:test")
    check("unregister", L.total_mb() == 0)
    L.clear_own()
    check("clear_own → file rimosso", not os.path.exists(L._own_path()))

    # 2) PID morto → voce scartata E file cancellato dallo snapshot
    dead = os.path.join(tmp, "999999.json")
    with open(dead, "w", encoding="utf-8") as fh:
        json.dump({"pid": 999999, "venv": "x", "ts": int(time.time()),
                   "models": {"zombie": {"mb": 5000, "category": "image", "state": "vram"}}}, fh)
    snap = L.snapshot()
    check("PID morto scartato", all(e["pid"] != 999999 for e in snap))
    check("file del PID morto cancellato", not os.path.exists(dead))

    # 3) voce STANTIA (ts vecchio, pid VIVO — riciclo pid) → scartata
    stale = os.path.join(tmp, f"{os.getpid()}.json")
    with open(stale, "w", encoding="utf-8") as fh:
        json.dump({"pid": os.getpid(), "venv": "x", "ts": int(time.time()) - 100000,
                   "models": {"old": {"mb": 123, "category": "image", "state": "vram"}}}, fh)
    check("voce stantia scartata", all(e["key"] != "old" for e in L.snapshot()))
    try:
        os.remove(stale)
    except OSError:
        pass

    # 4) concorrenza: 3 processi registrano e MUOIONO → snapshot li ripulisce; uno resta VIVO
    code = ("import os, sys, time; sys.path.insert(0, r'%s'); os.environ['LTX_VRAM_LEDGER_DIR']=r'%s'\n"
            "from core import vram_ledger as L; L._DIR=r'%s'\n"
            "L.register('w%%d' %% os.getpid(), 1000, category='video')\n"
            "time.sleep(float(sys.argv[1]))\n" % (_REPO, tmp, tmp))
    dead_procs = [subprocess.Popen([sys.executable, "-c", code, "0"]) for _ in range(3)]
    live = subprocess.Popen([sys.executable, "-c", code, "20"])
    for p in dead_procs:
        p.wait(timeout=60)
    time.sleep(1.5)                                          # il vivo ha certamente scritto
    snap = L.snapshot()
    live_keys = [e for e in snap if e["key"].startswith("w")]
    check(f"concorrenza: solo il processo VIVO resta ({len(live_keys)} voce)", len(live_keys) == 1)
    live.kill()
    live.wait(timeout=30)
    # NB: atexit non gira su kill() — è il caso "worker crashato": lo snapshot deve ripulire
    check("dopo il kill: snapshot ripulisce il crashato",
          all(not e["key"].startswith("w") for e in L.snapshot()))

    print(f"\n{'TUTTO OK' if not fails else 'FALLITI: ' + ', '.join(fails)}")
    return len(fails)


if __name__ == "__main__":
    raise SystemExit(run())
