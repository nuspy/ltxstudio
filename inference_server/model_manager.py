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


@dataclass
class ModelManager:
    budget_frac: float = 0.90                           # quota di VRAM totale utilizzabile
    _cache: dict[str, _Entry] = field(default_factory=dict)
    _loaders: dict[str, Callable[[], tuple[object, float]]] = field(default_factory=dict)

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
        if key not in self._loaders:
            raise KeyError(f"nessun loader registrato per '{key}'")
        # libera spazio se necessario PRIMA di caricare
        if need_mb:
            self.make_room(need_mb, keep=set(self._cache))   # non scaricare nulla di già usato ora
        obj, vram = self._loaders[key]()
        self._cache[key] = _Entry(obj=obj, vram_mb=float(vram or 0.0), last_used=_now(), sticky=sticky)
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
                break

    def unload(self, key: str) -> None:
        e = self._cache.pop(key, None)
        if e is None:
            return
        # NIENTE .to("cpu"): pullerebbe i pesi (22GB) in RAM prima di cancellarli = picco RAM inutile
        # e, con due processi attivi, congela il sistema. Cancello e basta → empty_cache libera la VRAM
        # (i tensori GPU vengono rilasciati col del); gc.collect() libera la RAM.
        try:
            del e.obj
        except Exception:                               # noqa: BLE001
            pass
        del e
        self.flush()

    def clear(self, include_sticky: bool = False) -> None:
        for k in list(self._cache):
            if include_sticky or not self._cache[k].sticky:
                self.unload(k)

    @staticmethod
    def flush() -> None:
        gc.collect()
        if torch is not None and torch.cuda.is_available():
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
        if not loras:
            return
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
            pipe.load_lora_weights(name, adapter_name=adapter)
            names.append(adapter); weights.append(float(lo.get("strength", 1.0)))
        try:
            pipe.set_adapters(names, adapter_weights=weights)
        except Exception:                               # noqa: BLE001
            pass

    def status(self) -> dict:
        total, free = self._vram_total_free_mb()
        return {"vram_total_mb": round(total), "vram_free_mb": round(free),
                "budget_mb": round(self._budget_mb()), "tracked_mb": round(self._used_mb()),
                "loaded": list(self._cache)}


def _now() -> float:
    return time.monotonic()
