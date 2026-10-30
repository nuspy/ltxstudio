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
  LS_PARK_VRAM_PEAK_MB → VRAM totale GPU via NVML (fallback 30720; picco di VRAM che il worker video porterà ANCORA sul commit: su Windows/WDDM la
                      VRAM allocata è addebitata al commit, vedi `commit_reserve_mb`)
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


def vram_peak_mb() -> float:
    """Picco di VRAM (MB) che una generazione video puo' raggiungere — addebitato anche al commit su WDDM.
    DINAMICO (regola «VRAM tarata per PC»): la VRAM totale della GPU letta via NVML/nvidia-smi
    (core.gpu_mem, nessun contesto CUDA); 30720 solo se la lettura fallisce. Override: LS_PARK_VRAM_PEAK_MB."""
    v = _env_chain(0.0, "LS_PARK_VRAM_PEAK_MB")
    if v > 0:
        return v
    try:
        from core import gpu_mem
        t = float(gpu_mem.total_vram_mb())
        if t > 0:
            return t
    except Exception:                                   # noqa: BLE001
        pass
    return 30720.0


def commit_reserve_mb(vram_charged_mb: float = 0.0) -> float:
    """Commit (MB) che il worker deve ancora impegnare quando la VRAM sale al picco: picco atteso meno
    la VRAM già allocata dal processo (già dentro il commit libero). Il parcheggio RAM la deve lasciare
    libera, altrimenti il load successivo (LoRA, stage 2) esaurisce il commit: 0xC0000005 (17/10)."""
    return max(0.0, vram_peak_mb() - max(0.0, float(vram_charged_mb)))


def emb_cache_mb() -> float:
    """Budget massimo (MB) della cache SU DISCO degli embedding di prompt (core/emb_cache).
    È un budget di spazio, non di RAM: le voci sono 24 MB l'una e non toccano il commit."""
    return _env_chain(4096.0, "LS_EMB_CACHE_MB")


def disk_free_mb(path: str) -> float:
    """Spazio libero (MB) sul volume che ospita `path`; 1e9 se non determinabile."""
    try:
        import shutil
        return shutil.disk_usage(path).free / 2**20
    except Exception:                                   # noqa: BLE001
        return 1e9


def skip_park_streaming_blocks(partition_name: str, streaming: bool) -> bool:
    """True se la partizione `partition_name` (nome dello SDOps passato al registry) NON va parcheggiata.

    Con l'offload a blocchi (streaming) il builder di ltx-core copia ogni blocco in buffer pinned e poi
    scarta lo state dict letto da disco: parcheggiarlo in RAM lo terrebbe vivo, cioe' una SECONDA copia
    dello stesso transformer (misurato 29/10 su LTX-2.3 fp8: ~17,7 GB pinned + ~17,7 GB parcheggiati,
    processo a 64-72 GB). La partizione dei blocchi (`*__blocks`) quindi non si parcheggia; le partizioni
    piccole (non-block, VAE, vocoder, connector) si'. Costo: il prossimo stadio/job rilegge i blocchi da
    disco. LTX_PARK_BLOCKS=1 ripristina il comportamento precedente."""
    if not streaming:
        return False
    if os.environ.get("LTX_PARK_BLOCKS", "").strip() == "1":
        return False
    return str(partition_name or "").endswith("__blocks")


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


def fits(size_mb: float, resident_mb: float, *, require_psutil: bool = False,
         reserve_mb: float = 0.0) -> bool:
    """Il predicato UNICO "questi size_mb ci stanno in cache RAM?":
    (RAM fisica − size > floor) AND (commit − size − reserve > commit_floor) AND (residente + size ≤ budget).
    reserve_mb = commit che il chiamante impegnerà ancora dopo (es. VRAM del worker, vedi `commit_reserve_mb`).
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
            and (avail_commit_mb() - size_mb - reserve_mb > commit_floor_mb())
            and (resident_mb + size_mb <= ram_cache_mb()))


class LoadRefused(RuntimeError):
    """Il caricamento previsto non ci sta nel commit: rifiutato PRIMA di iniziare (errore chiaro, nessun crash)."""


def check_load(load_mb: float, what: str = "modello") -> None:
    """GUARDIA di caricamento (task 12, incidente 17/10): se commit libero − carico previsto < pavimento di
    commit → solleva `LoadRefused` invece di procedere (superare il commit su Windows = access violation / crash
    di sistema, non un OOM pulito). `load_mb` = byte che il caricamento impegnerà ANCORA (0 se già residente).
    Pavimento: `commit_floor_mb()` (env LS_COMMIT_FLOOR_MB…)."""
    if load_mb <= 0:
        return
    avail = avail_commit_mb()
    floor = commit_floor_mb()
    if avail - float(load_mb) < floor:
        raise LoadRefused(
            f"Caricamento di {what} rifiutato: servono ~{load_mb / 1024:.1f} GB di memoria, il commit libero è "
            f"{avail / 1024:.1f} GB e il pavimento di sicurezza {floor / 1024:.0f} GB. Chiudi altri programmi o "
            f"scarica un modello dalla memoria («Libera VRAM») e riprova.")


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
