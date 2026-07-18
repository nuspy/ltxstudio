"""ModelManager — gestione VRAM del backend nativo (sostituto della smart-memory di ComfyUI).

Responsabilità (vedi docs/comfyui-migration-analysis.md §F):
- registry di "loader" per chiave (es. "flux:dev", "vae:flux2", "lora:<path>")
- caricamento LAZY on-demand, con stima/misura della VRAM occupata
- unload LRU quando il budget VRAM non basta (scenario foto→video / cambio modello/LoRA)
- offload diffusers (model_cpu_offload / sequential) e VAE tiling per i casi pesanti
- LoRA hot-swap senza ricaricare il modello base

Gira DENTRO l'InferenceServer (processo con torch+diffusers). NON importato dall'app PyQt.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Callable

try:
    import torch
except Exception:                                       # pragma: no cover - solo nel venv server
    torch = None


@dataclass
class _Entry:
    obj: object
    vram_mb: float
    last_used: float
    sticky: bool = False                                # non scaricare con LRU (es. VAE condivisa)
    component_keys: tuple = ()                          # componenti nel ComponentStore referenziati (refcount)


@dataclass
class ModelManager:
    budget_frac: float = 0.90                           # quota di VRAM totale utilizzabile
    _cache: dict[str, _Entry] = field(default_factory=dict)
    _loaders: dict[str, Callable[[], tuple[object, float]]] = field(default_factory=dict)
    # CPU-RAM cache dei modelli MONOLITICI sfrattati (zimage/ernie/checkpoint completi): invece di
    # distruggerli, .to("cpu") → il prossimo ensure li ripristina da RAM (10× più veloce dell'SSD).
    # I componenti CONDIVISI (flux family) hanno la loro cpu-cache DENTRO il ComponentStore.
    _cpu_cache: dict[str, _Entry] = field(default_factory=dict)

    # ── registrazione loader ────────────────────────────────────────────────
    def register(self, key: str, loader: Callable[[], tuple[object, float]]) -> None:
        """`loader()` deve ritornare (oggetto, vram_mb_stimati)."""
        self._loaders[key] = loader

    # ── VRAM ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _vram_total_free_mb() -> tuple[float, float]:
        if torch is None or not torch.cuda.is_available():
            return (0.0, 0.0)
        free, total = torch.cuda.mem_get_info()
        return (total / 1024**2, free / 1024**2)

    def _budget_mb(self) -> float:
        total, _ = self._vram_total_free_mb()
        return total * self.budget_frac

    def _used_mb(self) -> float:
        return sum(e.vram_mb for e in self._cache.values())

    # ── accesso modelli ───────────────────────────────────────────────────────
    def ensure(self, key: str, *, need_mb: float | None = None, sticky: bool = False) -> object:
        """Ritorna l'oggetto per `key`, caricandolo (e liberando spazio LRU) se serve."""
        e = self._cache.get(key)
        if e is not None:
            e.last_used = _now()
            return e.obj
        # RESTORE da CPU-RAM cache (monolitiche sfrattate con demote): niente loader/SSD; il device-move
        # su cuda lo fa il placement per-job (_ensure_placement: l'epoch è stale dopo il demote).
        ce = self._cpu_cache.pop(key, None)
        if ce is not None:
            if need_mb:
                self.make_room(need_mb)
            ce.last_used = _now()
            self._cache[key] = ce
            print(f"[mm] restore da RAM: {key}", flush=True)
            return ce.obj
        if key not in self._loaders:
            raise KeyError(f"nessun loader registrato per '{key}'")
        # libera spazio se necessario PRIMA di caricare un modello DIVERSO: sfratta i meno usati (LRU) finché
        # c'è spazio → il nuovo modello entra in VRAM (GPU-resident, veloce) invece di finire in offload su RAM
        # (era 249s/step!). Lo STESSO modello NON arriva qui (ensure ritorna la cache sopra) → nessun reload.
        if need_mb:
            self.make_room(need_mb)
        obj, vram = self._loaders[key]()
        # Le pipeline "shell" (componenti condivisi nel ComponentStore) dichiarano le chiavi referenziate
        # in _ls_component_keys: l'unload della shell fa release() (refcount--) SENZA distruggere T5/VAE/…
        ckeys = tuple(getattr(obj, "_ls_component_keys", ()) or ())
        self._cache[key] = _Entry(obj=obj, vram_mb=float(vram or 0.0), last_used=_now(), sticky=sticky,
                                  component_keys=ckeys)
        return obj

    def make_room(self, need_mb: float, keep: set[str] | None = None) -> None:
        """Scarica i modelli meno usati (LRU, non sticky, non in `keep`) finché c'è spazio."""
        keep = keep or set()
        _, free = self._vram_total_free_mb()
        if free >= need_mb:
            return
        victims = sorted(
            (k for k, e in self._cache.items() if not e.sticky and k not in keep),
            key=lambda k: self._cache[k].last_used,
        )
        for k in victims:
            self.unload(k)
            _, free = self._vram_total_free_mb()
            if free >= need_mb:
                return
        # ULTIMA risorsa: dopo le shell, tocca i COMPONENTI orfani del ComponentStore (refcount==0, LRU).
        # Demote su CPU se il budget RAM lo consente, altrimenti destroy (vedi component_store.evict_unused).
        try:
            from inference_server.component_store import STORE
            _, free = self._vram_total_free_mb()
            if free < need_mb:
                STORE.evict_unused(need_mb - free)
                self.flush(force=True)                   # spazio richiesto ORA → svuota davvero
        except Exception:                               # noqa: BLE001
            pass

    def unload(self, key: str, *, allow_cpu: bool = True) -> None:
        e = self._cache.pop(key, None)
        if e is None:
            return
        # Shell con componenti condivisi: release (refcount--) PRIMA di distruggere l'oggetto — così
        # T5/CLIP/VAE/transformer sopravvivono nello store (riusabili dalla prossima pipeline o demote/evict
        # da make_room). La shell in sé è leggera (~50MB): si distrugge, ricrearla costa ~1s.
        if e.component_keys:
            try:
                from inference_server.component_store import STORE
                for ck in e.component_keys:
                    STORE.release(ck)
            except Exception:                           # noqa: BLE001
                pass
        elif allow_cpu and hasattr(e.obj, "to") and not getattr(e.obj, "_ls_no_rebind", False):
            # MONOLITICA (zimage/ernie/checkpoint completo) → demote su CPU se il budget RAM lo consente:
            # il prossimo ensure la ripristina da RAM (10× più veloce dell'SSD). Esclusa la flux2 blockswap
            # (_ls_no_rebind: pesi già su CPU by design → si distrugge come prima). Il .to("cpu") copia
            # param-per-param (picco RAM ≈ size, picco VRAM nullo) — ok con 128GB + floor psutil.
            try:
                from inference_server.component_store import STORE
                if STORE._ram_ok(e.vram_mb):
                    try:
                        e.obj.remove_all_hooks()
                    except Exception:                   # noqa: BLE001
                        pass
                    e.obj.to("cpu")
                    e.last_used = _now()
                    self._cpu_cache[key] = e
                    STORE.bump_epoch()                  # device cambiati → rebind al restore
                    print(f"[mm] demote→RAM: {key} ({e.vram_mb:.0f}MB)", flush=True)
                    self._trim_cpu_cache()              # LRU: non accumulare oltre budget/floor RAM
                    self.flush()
                    return
            except Exception:                           # noqa: BLE001 — fallback: distruggi
                pass
        try:
            del e.obj
        except Exception:                               # noqa: BLE001
            pass
        del e
        self.flush()

    def clear_all(self, *, allow_cpu: bool = False) -> int:
        """HARD-free: scarica TUTTE le shell/pipeline in cache (release dei componenti condivisi) e svuota la
        cpu-cache. Usato dal /free del figlio persistente (cambio-pipeline: un altro motore vuole la VRAM).
        allow_cpu=False → distrugge (VRAM subito libera) invece di demotare su RAM. Ritorna quante scaricate."""
        n = len(self._cache)
        for k in list(self._cache.keys()):
            self.unload(k, allow_cpu=allow_cpu)
        self._cpu_cache.clear()
        return n

    def evict_others(self, keep_key: str) -> int:
        """AUTO-EVICT al cambio modello: scarica TUTTE le shell tranne keep_key (non-sticky) → rilascia i loro
        componenti. Il transformer del modello PRECEDENTE diventa orfano (refcount 0) e verrà liberato/demotato
        dallo STORE (evict_unused); i componenti CONDIVISI usati da keep_key restano (refcount>0). Ritorna il
        numero di shell scaricate. Così cambiando modello la VRAM del precedente si libera SUBITO (non attende
        la pressione di make_room). Disattivabile con NATIVE_AUTOEVICT=0."""
        others = [k for k, e in self._cache.items() if k != keep_key and not e.sticky]
        for k in others:
            self.unload(k, allow_cpu=False)             # la shell è leggera; l'orfano lo gestisce lo STORE
        if others:
            print(f"[mm] auto-evict: {len(others)} shell del modello precedente scaricate (keep={keep_key})",
                  flush=True)
        return len(others)

    def _trim_cpu_cache(self) -> None:
        """LRU della cache-RAM monolitica: distrugge i modelli MENO recenti finché il totale rientra nel budget
        (NATIVE_CPU_CACHE_MB) E resta il floor di RAM libera (NATIVE_RAM_FLOOR_MB). Senza questo la RAM cresceva
        fino al tetto cambiando modello di continuo, senza mai scendere."""
        import os
        budget = float(os.environ.get("NATIVE_CPU_CACHE_MB", "40960"))
        floor = float(os.environ.get("NATIVE_RAM_FLOOR_MB", "16384"))

        def _avail_mb() -> float:
            try:
                import psutil
                return psutil.virtual_memory().available / 2**20
            except Exception:                           # noqa: BLE001
                return 1e9

        def _total_mb() -> float:
            return sum(getattr(e, "vram_mb", 0) for e in self._cpu_cache.values())

        while self._cpu_cache and (_total_mb() > budget or _avail_mb() < floor):
            k = min(self._cpu_cache, key=lambda kk: self._cpu_cache[kk].last_used)
            e = self._cpu_cache.pop(k)
            try:
                del e.obj
            except Exception:                           # noqa: BLE001
                pass
            print(f"[mm] LRU cache-RAM: distrutto {k} ({getattr(e, 'vram_mb', 0):.0f}MB) "
                  f"— RAM libera {_avail_mb():.0f}MB", flush=True)
        self.flush()

    def clear(self, include_sticky: bool = False) -> None:
        """Svuota TUTTO (endpoint /free, chiamato dal worker video prima di partire): shell, CPU-cache e
        componenti dello store — il video ha bisogno di VRAM *e* RAM (block-streaming Q4 legge dalla RAM)."""
        for k in list(self._cache):
            if include_sticky or not self._cache[k].sticky:
                self.unload(k, allow_cpu=False)         # /free = libera davvero, niente demote
        for k, e in list(self._cpu_cache.items()):
            try:
                del e.obj
            except Exception:                           # noqa: BLE001
                pass
            self._cpu_cache.pop(k, None)
        try:
            from inference_server.component_store import STORE
            for k in list(STORE._items):
                ent = STORE._items.pop(k)
                try:
                    del ent.obj
                except Exception:                       # noqa: BLE001
                    pass
            STORE.bump_epoch()
        except Exception:                               # noqa: BLE001
            pass
        try:                                            # PuLID: cache modulo FUORI dallo store (eva-clip/
            from core import pulid_adapter              # insightface) — senza questo /free lasciava GB
            pulid_adapter.clear_cache()                 # pinnati in VRAM (OOM a cascata, run 11/07)
        except Exception:                               # noqa: BLE001
            pass
        self.flush(force=True)                          # /free = libera davvero tutto

    @staticmethod
    def flush(force: bool = False) -> None:
        """M3 — `empty_cache()` EURISTICO (come `soft_empty_cache` di ComfyUI): svuotare il caching allocator
        ad ogni unload è costoso e spesso inutile. Lo facciamo solo se il reserved-ma-libero (memoria che il
        caching allocator tiene senza usarla) supera il 25% del free reale — oppure se `force=True` (/free,
        make_room: lì lo spazio serve ORA)."""
        gc.collect()
        if torch is None or not torch.cuda.is_available():
            return
        if force:
            torch.cuda.empty_cache()
            return
        try:
            free, _ = torch.cuda.mem_get_info()
            cached_unused = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            if cached_unused > free * 0.25:
                torch.cuda.empty_cache()
        except Exception:                                   # noqa: BLE001
            torch.cuda.empty_cache()

    @staticmethod
    def place(pipe, vram_mb: float) -> str:
        """Posiziona un pipeline: SU GPU se c'è spazio (resta caricato tra le run → niente offload/reload
        ripetuto), altrimenti enable_model_cpu_offload. L'eviction quando serve spazio per un ALTRO modello
        è gestita dalla cache LRU (make_room/unload). Anti system-memory-fallback Windows: GPU solo se
        free ≥ vram·margine + floor. Override: NATIVE_FORCE_OFFLOAD / NATIVE_GPU_MARGIN / NATIVE_GPU_FLOOR_MB."""
        import os
        if os.environ.get("NATIVE_FORCE_OFFLOAD") == "1" or torch is None or not torch.cuda.is_available():
            pipe.enable_model_cpu_offload()
            return "offload"
        free_mb = torch.cuda.mem_get_info()[0] / 1024**2
        margin = float(os.environ.get("NATIVE_GPU_MARGIN", "1.35"))
        floor_mb = float(os.environ.get("NATIVE_GPU_FLOOR_MB", "1500"))
        if free_mb >= float(vram_mb) * margin + floor_mb:
            pipe.to("cuda")
            return "gpu"
        pipe.enable_model_cpu_offload()
        return "offload"

    # ── LoRA hot-swap (su un pipeline diffusers) ─────────────────────────────
    @staticmethod
    def apply_loras(pipe, loras: list[dict] | None) -> None:
        """loras = [{'name': path, 'strength': w}]. Carica/azzera senza ricaricare la base."""
        try:
            pipe.unload_lora_weights()
        except Exception:                               # noqa: BLE001
            pass
        # Filtra le voci placeholder/vuote che la GUI può passare (nessuna LoRA realmente selezionata):
        # {'name': ''}, 'None', 'nessuna', ecc. Se non resta nulla → nessuna LoRA da applicare, esci.
        def _real(lo):
            n = (lo.get("name") if isinstance(lo, dict) else lo) or ""
            return str(n).strip() and str(n).strip().lower() not in ("none", "nessuna", "nessuno", "-")
        loras = [lo for lo in (loras or []) if _real(lo)]
        if not loras:
            return
        # Base con Linear quantizzati CUSTOM (FLUX.2/Klein fp8: Fp8Linear/Nvfp4Linear): PEFT non sa
        # iniettare LoRA su questi moduli (→ "Target module Fp8Linear() is not supported"). Invece di
        # crashare, fallisci con un messaggio chiaro/azionabile SOLO se c'è davvero una LoRA da applicare.
        # Il supporto LoRA nativo su base fp8 (iniezione manuale nel forward dequant) è dedicato — #17/#78.
        tr = getattr(pipe, "transformer", None)
        if tr is not None and any(type(m).__name__ in ("Fp8Linear", "Nvfp4Linear")
                                  for m in tr.modules()):
            raise RuntimeError(
                "LoRA non supportata su base FLUX.2/Klein in fp8 (moduli quantizzati custom): "
                "usa un modello base bf16 per applicare la LoRA, oppure fondi la LoRA nel checkpoint. "
                "Il supporto LoRA nativo su fp8 è pianificato (task #17/#78).")
        import os
        names, weights = [], []
        for i, lo in enumerate(loras):
            adapter = f"a{i}"
            name = lo["name"]
            # la GUI passa il nome RELATIVO stile ComfyUI (es. 'Flux\\x.safetensors') → risolvi il path
            if name and not os.path.isabs(name) and not os.path.exists(name):
                cand = os.path.join(r"E:\AI\Models\LoRA", name)
                if os.path.exists(cand):
                    name = cand
            try:
                pipe.load_lora_weights(name, adapter_name=adapter)
            except Exception as _e:                     # noqa: BLE001
                if getattr(getattr(pipe, "transformer", None), "_ls_fp8_storage", False):
                    raise RuntimeError(
                        "LoRA non caricabile sul base con storage fp8 (layerwise casting): riavvia il "
                        f"server con NATIVE_FLUX1_FP8=0 per usare questa LoRA in bf16. ({_e})") from _e
                raise
            names.append(adapter); weights.append(float(lo.get("strength", 1.0)))
        try:
            pipe.set_adapters(names, adapter_weights=weights)
        except Exception:                               # noqa: BLE001
            pass

    def status(self) -> dict:
        total, free = self._vram_total_free_mb()
        st = {"vram_total_mb": round(total), "vram_free_mb": round(free),
              "budget_mb": round(self._budget_mb()), "tracked_mb": round(self._used_mb()),
              "loaded": list(self._cache)}
        try:
            from inference_server.component_store import STORE
            st["store"] = STORE.status()
        except Exception:                               # noqa: BLE001
            pass
        return st


def _now() -> float:
    return time.monotonic()
