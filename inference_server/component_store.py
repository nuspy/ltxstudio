"""ComponentStore — cache con refcount per i COMPONENTI condivisi tra pipeline (T5-XXL, CLIP-L, VAE,
transformer FLUX.1).

Perché: T5-XXL (~9.1GB), CLIP-L (~0.25GB) e la VAE flux (~0.17GB) sono IDENTICI per tutte le pipeline
FLUX.1 (base/Fill/Kontext/Krea/…). Prima di questo store, _assemble_flux li ricaricava da disco per OGNI
pipeline e lo sfratto LRU li distruggeva insieme al transformer → 10-20GB di I/O inutile per workflow.
Ora: ogni componente vive UNA volta nello store; le pipeline sono "shell" leggere che li referenziano
(refcount). Lo sfratto di una shell NON distrugge i componenti; make_room li tocca solo come ULTIMA
risorsa (evict_unused, solo refcount==0, LRU) — e in quel caso li DEMOTA su CPU-RAM se il budget lo
consente (restore 10× più veloce del reload da SSD), altrimenti li distrugge.

placement_epoch: contatore globale dei cambi di stato device/hook. Ogni pipeline salva l'epoch del suo
ultimo placement (`_ls_place_epoch`); a inizio job, se l'epoch dello store è avanzato (qualcun altro ha
mosso i componenti o attaccato hook), la pipeline si ri-posiziona (_ensure_placement in image_flux).
Coda del server SERIALE → un solo job attivo → niente locking.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable

try:
    import torch
except Exception:                                       # pragma: no cover — solo nel venv server
    torch = None


def _now() -> float:
    return time.monotonic()


def _measure_mb(obj) -> float:
    """MB reali di parametri+buffer di un nn.Module (element_size onora fp8/bf16/quantizzati)."""
    try:
        total = 0
        for p in obj.parameters(recurse=True):
            total += p.numel() * p.element_size()
        for b in obj.buffers(recurse=True):
            total += b.numel() * b.element_size()
        return total / 2**20
    except Exception:                                   # noqa: BLE001 — oggetti non-Module
        return 0.0


@dataclass
class _CEntry:
    obj: object
    size_mb: float
    refcount: int = 0
    last_used: float = 0.0
    device: str = "cuda"                                # "cuda" | "cpu" (dove stanno i PESI ora)


@dataclass
class ComponentStore:
    placement_epoch: int = 0
    _items: dict[str, _CEntry] = field(default_factory=dict)

    # ── ciclo di vita ────────────────────────────────────────────────────────
    def acquire(self, key: str, loader: Callable[[], object]) -> object:
        """Ritorna il componente (refcount++), caricandolo col loader al primo uso. Un componente in
        CPU-cache viene ritornato COSÌ COM'È: i device-move li fa il placement a inizio job (rebind)."""
        e = self._items.get(key)
        if e is None:
            obj = loader()
            _sz = _measure_mb(obj)
            e = _CEntry(obj=obj, size_mb=_sz, refcount=0, last_used=_now(),
                        device="cpu")                    # from_pretrained/from_single_file → pesi su CPU
            self._items[key] = e
            try:                                         # SMART-MEMORY: auto-popola il registro footprint (dtype-aware)
                from core import vram_estimate           # F1: DB canonico unico (era model_registry, write-only)
                if not vram_estimate.comp_get(key):
                    _cat = "image" if ("transformer" in key or ":tr" in key or "_tr" in key) else "aux"
                    try:
                        _dt = str(next(obj.parameters()).dtype).replace("torch.", "")
                    except Exception:                    # noqa: BLE001
                        _dt = ""
                    vram_estimate.comp_put(key, _sz, category=_cat, dtype=_dt)
            except Exception:                            # noqa: BLE001 — best-effort, mai fatale
                pass
            try:                                         # F3: ledger live. Alla creazione i pesi sono su
                from core import vram_ledger             # CPU (from_pretrained) → "ram_parked"; sarà
                vram_ledger.register(key, _sz, category="image", state="ram_parked")  # note_device()
            except Exception:                            # noqa: BLE001 — a metterlo "vram" dopo il place
                pass
        e.refcount += 1
        e.last_used = _now()
        return e.obj

    def retain(self, key: str) -> object:
        """refcount++ su una entry ESISTENTE (per from_pipe: la nuova shell referenzia gli stessi moduli)."""
        e = self._items[key]
        e.refcount += 1
        e.last_used = _now()
        return e.obj

    def release(self, key: str) -> None:
        """refcount-- (clamp a 0). NON distrugge MAI: i componenti orfani restano riusabili finché
        make_room non ha davvero bisogno del loro spazio (evict_unused)."""
        e = self._items.get(key)
        if e is not None and e.refcount > 0:
            e.refcount -= 1

    def note_device(self, key: str, device: str) -> None:
        """Annota dove stanno i pesi ora (chiamata dal placement dopo to('cuda')/offload)."""
        e = self._items.get(key)
        if e is not None:
            e.device = device
            try:                                        # F3: lo stato del ledger segue il device reale
                from core import vram_ledger
                vram_ledger.set_state(key, "vram" if device == "cuda" else "ram_parked")
            except Exception:                           # noqa: BLE001
                pass

    # ── memoria ──────────────────────────────────────────────────────────────
    def resident_mb(self, device: str = "cuda") -> float:
        return sum(e.size_mb for e in self._items.values() if e.device == device)

    def clear(self) -> int:
        """HARD-free: distrugge TUTTI i componenti e libera la VRAM (empty_cache). Semantica di /free =
        "un'altra pipeline ha bisogno della VRAM ORA" (cambio-pipeline). Ritorna quanti ne ha distrutti.
        Da chiamare solo TRA i job (mai a job in corso)."""
        import gc
        n = len(self._items)
        for k in list(self._items.keys()):
            e = self._items.pop(k, None)
            try:
                del e.obj
            except Exception:                               # noqa: BLE001
                pass
        try:
            from core import vram_ledger
            vram_ledger.clear_own()
        except Exception:                                   # noqa: BLE001
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()                    # WDDM rilascia lazy: barriera prima del report
                torch.cuda.empty_cache()
        except Exception:                                   # noqa: BLE001
            pass
        return n

    def stats(self) -> dict:
        """Stato OSSERVABILE del ComponentStore (per i test app-completa e il monitor memoria): quali
        componenti sono residenti, su quale device, refcount e MB. Serve a VERIFICARE la gestione memoria
        allo switch di pipeline (condivisi tenuti, sfratto solo se serve). Vedi tests/app_env."""
        comps = [{"key": k, "device": e.device, "refcount": e.refcount, "size_mb": round(e.size_mb, 1),
                  "last_used": round(e.last_used, 2)} for k, e in self._items.items()]
        comps.sort(key=lambda c: (c["device"] != "cuda", -c["size_mb"]))
        return {"components": comps, "n": len(comps),
                "vram_mb": round(self.resident_mb("cuda"), 1), "cpu_mb": round(self.resident_mb("cpu"), 1),
                "epoch": self.placement_epoch}

    def resident_of(self, keys, device: str = "cuda") -> float:
        """MB dei componenti in `keys` GIÀ residenti su `device` — sottratti dalla stima di _place
        (lo spazio è già pagato, la pipeline li riusa)."""
        return sum(e.size_mb for k in (keys or ()) if (e := self._items.get(k)) and e.device == device)

    def _ram_ok(self, size_mb: float) -> bool:
        """La CPU-cache può ospitare altri size_mb? Predicato UNIFICATO (commit 2): RAM fisica +
        COMMIT (il limite VERO su Windows — vedi pagefile-commit-mmap-crash) + budget, da
        core/mem_budget (env LS_* con fallback legacy NATIVE_*/LTX_*)."""
        try:
            from core import mem_budget
            return mem_budget.fits(size_mb, self.resident_mb("cpu"), require_psutil=True)
        except Exception:                               # noqa: BLE001
            return False

    def _shrink_cpu_lru(self, need_mb: float) -> None:
        """Fa posto nel budget CPU distruggendo i cpu-cached più vecchi (refcount==0). LRU unificata."""
        from core import mem_budget
        over = self.resident_mb("cpu") + need_mb - mem_budget.ram_cache_mb()
        victims = mem_budget.pick_lru_victims(
            ((k, e.size_mb, e.last_used) for k, e in self._items.items()
             if e.refcount == 0 and e.device == "cpu"), over)
        for k in victims:
            e = self._items.pop(k)
            try:
                del e.obj
            except Exception:                           # noqa: BLE001
                pass

    def evict_unused(self, need_mb: float) -> float:
        """ULTIMA risorsa di make_room: libera VRAM dai componenti orfani (refcount==0, device=cuda),
        LRU. Se il budget RAM lo consente → DEMOTE su CPU (restore veloce al prossimo acquire); se no →
        destroy. Ritorna i MB (stimati) liberati; make_room resta self-correcting su mem_get_info."""
        freed = 0.0
        # SIZE-AWARE (direttiva utente): scegliere QUALE sfrattare per liberare `need_mb` col MINOR spreco/costo
        # di reload — "più economico scaricare 2GB che 10GB se il buco è piccolo". Non LRU cieco:
        #   1) se UN singolo orfano copre da solo il deficit → sfratta il PIÙ PICCOLO che basta (un solo reload,
        #      niente over-eviction), tie-break LRU;
        #   2) altrimenti servono più modelli → sfratta dal PIÙ GRANDE (copre con meno evizioni), tie-break LRU.
        _orph = [(k, e.size_mb, e.last_used) for k, e in self._items.items()
                 if e.refcount == 0 and e.device == "cuda"]
        _cover = sorted((o for o in _orph if o[1] >= need_mb), key=lambda o: (o[1], o[2]))
        if _cover:
            _order = [_cover[0][0]]                          # il più piccolo che copre da solo
        else:
            _order = [o[0] for o in sorted(_orph, key=lambda o: (-o[1], o[2]))]   # grande→piccolo, tie LRU
        for k in _order:
            e = self._items[k]
            self._shrink_cpu_lru(e.size_mb)
            demoted = False
            if self._ram_ok(e.size_mb) and hasattr(e.obj, "to"):
                try:
                    try:                                 # difensivo: refcount==0 ⇒ nessuna pipe viva,
                        from accelerate.hooks import remove_hook_from_module   # ma strip comunque
                        remove_hook_from_module(e.obj, recurse=True)
                    except Exception:                    # noqa: BLE001
                        pass
                    e.obj.to("cpu")
                    e.device = "cpu"
                    demoted = True
                    print(f"[store] demote→cpu {k} ({e.size_mb:.0f}MB)", flush=True)
                    try:
                        from core import vram_ledger
                        vram_ledger.set_state(k, "ram_parked")
                    except Exception:                    # noqa: BLE001
                        pass
                except Exception as _e:                  # noqa: BLE001
                    print(f"[store] demote {k} fallita ({_e}) → destroy", flush=True)
            if not demoted:
                self._items.pop(k, None)
                try:
                    del e.obj
                except Exception:                        # noqa: BLE001
                    pass
                print(f"[store] destroy {k} ({e.size_mb:.0f}MB)", flush=True)
                try:
                    from core import vram_ledger
                    vram_ledger.unregister(k)
                except Exception:                        # noqa: BLE001
                    pass
            freed += e.size_mb
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if freed >= need_mb:
                break
        if freed:
            self.bump_epoch()                            # i device sono cambiati → rebind delle pipe
        return freed

    def demote_all(self) -> int:
        """FREE_DEMOTE (unified-model-manager): parcheggia in RAM TUTTI i componenti residenti in
        VRAM — la pipeline successiva è di un'altra categoria (o ignota), ma questi modelli possono
        servire di nuovo: invece di distruggerli, .to('cpu') se il budget/commit lo consente,
        altrimenti destroy. Ritorna quanti componenti sono stati demotati."""
        n = 0
        for k in list(self._items.keys()):
            e = self._items.get(k)
            if e is None or e.device != "cuda":
                continue
            demoted = False
            if self._ram_ok(e.size_mb) and hasattr(e.obj, "to"):
                try:
                    try:
                        from accelerate.hooks import remove_hook_from_module
                        remove_hook_from_module(e.obj, recurse=True)
                    except Exception:                    # noqa: BLE001
                        pass
                    e.obj.to("cpu")
                    e.device = "cpu"
                    demoted = True
                    n += 1
                    print(f"[store] demote_all→cpu {k} ({e.size_mb:.0f}MB)", flush=True)
                    try:
                        from core import vram_ledger
                        vram_ledger.set_state(k, "ram_parked")
                    except Exception:                    # noqa: BLE001
                        pass
                except Exception as _e:                  # noqa: BLE001
                    print(f"[store] demote_all {k} fallita ({_e}) → destroy", flush=True)
            if not demoted:
                self._items.pop(k, None)
                try:
                    del e.obj
                except Exception:                        # noqa: BLE001
                    pass
                try:
                    from core import vram_ledger
                    vram_ledger.unregister(k)
                except Exception:                        # noqa: BLE001
                    pass
        import gc
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()                     # WDDM rilascia lazy: barriera prima del "FREED"
            torch.cuda.empty_cache()
        self.bump_epoch()                                # device cambiati → rebind al prossimo job
        return n

    # ── epoch / stato ────────────────────────────────────────────────────────
    def bump_epoch(self) -> int:
        self.placement_epoch += 1
        return self.placement_epoch

    def status(self) -> dict:
        return {"epoch": self.placement_epoch,
                "components": {k: {"mb": round(e.size_mb), "refcount": e.refcount, "device": e.device}
                               for k, e in self._items.items()}}


STORE = ComponentStore()
