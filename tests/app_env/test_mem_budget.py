"""Test di core/mem_budget (politica RAM unica, commit 2) — senza GPU.
Valida: default, catena di fallback env legacy, predicato fits, LRU."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_ENVS = ("LS_RAM_CACHE_MB", "NATIVE_CPU_CACHE_MB", "LTX_RAM_PARK_MB", "LS_RAM_FLOOR_MB",
         "NATIVE_RAM_FLOOR_MB", "LS_COMMIT_FLOOR_MB", "NATIVE_COMMIT_FLOOR_MB",
         "LTX_COMMIT_FLOOR_MB", "LS_RAM_CACHE", "LTX_RAM_PARK", "LS_PARK_VRAM_PEAK_MB")


def _clean():
    for e in _ENVS:
        os.environ.pop(e, None)


def run() -> int:
    fails = []

    def check(name, cond):
        print(("OK  " if cond else "FAIL") + " " + name, flush=True)
        if not cond:
            fails.append(name)

    from core import mem_budget as M

    # 1) default
    _clean()
    check("default budget 49152", M.ram_cache_mb() == 49152)
    check("default floor 16384", M.ram_floor_mb() == 16384)
    check("default commit floor 20480", M.commit_floor_mb() == 20480)
    check("default enabled", M.enabled() is True)

    # 2) catena di fallback: legacy vince sul default, LS_* vince su tutto
    os.environ["NATIVE_CPU_CACHE_MB"] = "40960"
    check("legacy NATIVE_CPU_CACHE_MB vince sul default", M.ram_cache_mb() == 40960)
    os.environ["LS_RAM_CACHE_MB"] = "12345"
    check("LS_RAM_CACHE_MB vince su tutto", M.ram_cache_mb() == 12345)
    _clean()
    os.environ["LTX_COMMIT_FLOOR_MB"] = "30000"
    check("legacy LTX_COMMIT_FLOOR_MB letto", M.commit_floor_mb() == 30000)
    _clean()
    os.environ["LTX_RAM_PARK"] = "0"
    check("legacy LTX_RAM_PARK=0 disabilita", M.enabled() is False)
    _clean()
    os.environ["LS_RAM_CACHE"] = "0"
    check("LS_RAM_CACHE=0 disabilita", M.enabled() is False)
    _clean()

    # 3) fits: monkeypatch di RAM/commit
    _oa, _oc = M.avail_ram_mb, M.avail_commit_mb
    try:
        M.avail_ram_mb = lambda: 30000.0
        M.avail_commit_mb = lambda: 40000.0
        check("fits ok dentro i vincoli", M.fits(5000, 10000) is True)
        check("fits nega sotto floor RAM", M.fits(14000, 0) is False)          # 30000-14000=16000 !> 16384
        M.avail_ram_mb = lambda: 100000.0
        check("fits nega sotto floor commit", M.fits(20000, 0) is False)       # 40000-20000 !> 20480
        # 17/10: riserva di commit per la VRAM che il worker impegnera' ancora (WDDM): crash 0xC0000005 dev+IC-LoRA
        M.avail_commit_mb = lambda: 76000.0
        check("fits senza riserva: 24GB con 76GB liberi ok", M.fits(24000, 0) is True)
        check("fits con riserva VRAM 30GB: 76-24-30=22GB ok (soglia 20GB)", M.fits(24000, 0, reserve_mb=30000) is True)
        M.avail_commit_mb = lambda: 60000.0
        check("fits con riserva VRAM: 60-24-30=6GB NEGA (era il caso del crash)", M.fits(24000, 0, reserve_mb=30000) is False)
        check("commit_reserve_mb: picco 30GB meno VRAM gia' allocata", M.commit_reserve_mb(0) == 30720 and M.commit_reserve_mb(26000) == 4720 and M.commit_reserve_mb(99999) == 0)
        os.environ["LS_PARK_VRAM_PEAK_MB"] = "10000"
        check("LS_PARK_VRAM_PEAK_MB letto", M.vram_peak_mb() == 10000)
        _clean()
        M.avail_commit_mb = lambda: 100000.0
        check("fits nega oltre il budget", M.fits(10000, 45000) is False)      # 55000 > 49152
        os.environ["LS_RAM_CACHE"] = "0"
        check("fits nega se disabilitato", M.fits(1, 0) is False)
        _clean()
    finally:
        M.avail_ram_mb, M.avail_commit_mb = _oa, _oc

    # 4) LRU: più vecchi prima, si ferma al need
    v = M.pick_lru_victims([("a", 100, 3), ("b", 200, 1), ("c", 300, 2)], 250)
    check("LRU: vecchi prima, stop al need", v == ["b", "c"])
    check("LRU: need<=0 → nessuna vittima", M.pick_lru_victims([("a", 1, 1)], 0) == [])

    print(f"\n{'TUTTO OK' if not fails else 'FALLITI: ' + ', '.join(fails)}")
    return len(fails)


if __name__ == "__main__":
    raise SystemExit(run())
