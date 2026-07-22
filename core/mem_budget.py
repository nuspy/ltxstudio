"""mem_budget — POLITICA RAM UNICA dei motori (unificazione pipeline, commit 2).

Prima questa politica viveva in TRE copie divergenti: ModelManager._trim_cpu_cache e
ComponentStore._ram_ok (immagini, env NATIVE_*, budget 40960) e _CpuParkRegistry (video,
env LTX_*, budget 49152, senza floor RAM fisica). Qui: un solo set di env NUOVI (LS_*) con
gli env legacy sempre letti come fallback, un solo predicato "ci sta?" (RAM fisica + COMMIT
+ budget) e un solo helper LRU. stdlib-only: importabile da ogni venv.

Env (in ordine di precedenza, primo trovato vince):
  LS_RAM_CACHE_MB   → NATIVE_CPU_CACHE_MB → LTX_RAM_PARK_MB   → 49152  (budget cache/park RAM)
  LS_RAM_FLOOR_MB   → NATIVE_RAM_FLOOR_MB                     → 16384  (RAM fisica da non intaccare)
  LS_COMMIT_FLOOR_MB→ NATIVE_COMMIT_FLOOR_MB → LTX_COMMIT_FLOOR_MB → 20480 (commit: il limite VERO
                      su Windows — sforarlo = access violation, vedi pagefile-commit-mmap-crash)
  LS_RAM_CACHE=0    → disabilita il parcheggio/cache RAM (legacy: LTX_RAM_PARK=0)
"""
from __future__ import annotations

import os


def _env_chain(default: float, *names: str) -> float:
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            try:
                return float(v)
            except ValueError:
                pass
    return default


def ram_cache_mb() -> float:
    """Budget massimo (MB) della cache/parcheggio RAM dei modelli."""
    return _env_chain(49152.0, "LS_RAM_CACHE_MB", "NATIVE_CPU_CACHE_MB", "LTX_RAM_PARK_MB")


def ram_floor_mb() -> float:
    """RAM fisica libera (MB) da non intaccare MAI."""
    return _env_chain(16384.0, "LS_RAM_FLOOR_MB", "NATIVE_RAM_FLOOR_MB")


def commit_floor_mb() -> float:
    """Commit disponibile (RAM+pagefile, MB) da non intaccare MAI."""
    return _env_chain(20480.0, "LS_COMMIT_FLOOR_MB", "NATIVE_COMMIT_FLOOR_MB", "LTX_COMMIT_FLOOR_MB")


def enabled() -> bool:
    """Parcheggio/cache RAM abilitati? (LS_RAM_CACHE=0 o legacy LTX_RAM_PARK=0 → off)."""
    if os.environ.get("LS_RAM_CACHE", "").strip() == "0":
        return False
    return os.environ.get("LTX_RAM_PARK", "1") != "0"


def avail_ram_mb() -> float:
    """RAM fisica disponibile (psutil; 1e9 se psutil assente = nessun vincolo)."""
    try:
        import psutil
        return psutil.virtual_memory().available / 2**20
    except Exception:                                   # noqa: BLE001
        return 1e9


def avail_commit_mb() -> float:
    try:
        from core import st_direct
        return st_direct.commit_avail_mb()
    except Exception:                                   # noqa: BLE001
        return 1e9


def fits(size_mb: float, resident_mb: float, *, require_psutil: bool = False) -> bool:
    """Il predicato UNICO "questi size_mb ci stanno in cache RAM?":
    (RAM fisica − size > floor) AND (commit − size > commit_floor) AND (residente + size ≤ budget).
    require_psutil=True → psutil assente = False (semantica storica di ComponentStore._ram_ok);
    False → psutil assente = vincolo RAM saltato (semantica storica di _trim/_CpuParkRegistry)."""
    if not enabled():
        return False
    if require_psutil:
        try:
            import psutil                               # noqa: F401
        except Exception:                               # noqa: BLE001
            return False
    return ((avail_ram_mb() - size_mb > ram_floor_mb())
            and (avail_commit_mb() - size_mb > commit_floor_mb())
            and (resident_mb + size_mb <= ram_cache_mb()))


def pick_lru_victims(entries, need_mb: float) -> list:
    """Sceglie le vittime LRU per liberare `need_mb` dalla cache RAM.
    entries: iterable di (key, size_mb, last_used). Ritorna le key da sfrattare (più vecchie
    prima) finché la somma copre need_mb (lista vuota se need_mb <= 0)."""
    if need_mb <= 0:
        return []
    out, freed = [], 0.0
    for key, size_mb, _used in sorted(entries, key=lambda e: e[2]):
        out.append(key)
        freed += float(size_mb)
        if freed >= need_mb:
            break
    return out
