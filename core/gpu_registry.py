"""Registro GPU UNICO — gestore centrale di load/riuso/sfratto dei modelli generativi (immagini, video,
Krea2, LTX, LingBot, krea_rt), sopra i server persistenti e i worker.

Problema che risolve: oggi ogni motore gestisce la memoria per conto suo (ComponentStore dentro :8189 ma
per-job-isolato, ltx2_server con la sua cache, subprocess che ricaricano). Non c'è un posto unico che sappia
COSA è residente sulla GPU e decida: *riusa se è già caricato, tieni i componenti condivisi, scarica solo se
cambia modello o serve spazio*. Questo modulo è quel posto.

Principio (richiesto dall'utente):
  - Se la pipeline richiesta usa un modello GIÀ residente → **riusa** (nessun reload).
  - Se condivide componenti con un residente (es. text-encoder T5) → **riusa quei componenti**.
  - Un modello NON viene scaricato se (a) è ancora in uso, oppure (b) serve alla prossima pipeline, oppure
    (c) non serve liberare spazio. Si scarica solo su **cambio-modello** o **pressione VRAM**.

Il registro è cross-processo (i motori sono server/subprocess separati): lo stato di verità è ciò che i
server persistenti riportano in `/system_stats` (chiave modello + VRAM), più il coordinamento dei `/free` via
`mem_coordinator`. Solo stdlib. Nessun `torch.cuda` (VRAM via `core/gpu_mem`, NVML/nvidia-smi).
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field

from core import gpu_mem, mem_coordinator

# porta → categoria del server persistente (ciò che il registro coordina). Allineato a mem_coordinator.
SERVER_PORTS = {
    8189: "image",      # InferenceServer FLUX (ComponentStore)
    8191: "image",      # Krea 2 (immagini/edit)
    8200: "image",      # API pubblica (gen in-process)
    8190: "video",      # LTX-2 persistente
}

# margine di sicurezza: sotto questa VRAM libera (MB) consideriamo "serve spazio" e sfrattiamo prima di caricare.
DEFAULT_HEADROOM_MB = 2000


@dataclass
class ModelKey:
    """Identità di un modello ai fini del riuso: (arch, checkpoint, dtype). Due job con la stessa chiave
    riusano il residente; chiavi diverse = cambio-modello (candidato allo sfratto se serve spazio)."""
    arch: str = ""
    model: str = ""
    dtype: str = ""

    def norm(self) -> tuple[str, str, str]:
        return (str(self.arch or "").lower(), str(self.model or "").lower(), str(self.dtype or "").lower())

    def __str__(self) -> str:
        a, m, d = self.norm()
        return f"{a}:{m or '-'}:{d or '-'}"


@dataclass
class Resident:
    """Ciò che un server riporta come residente: chiave modello + VRAM + componenti + refcount."""
    port: int
    category: str
    key: str = ""
    vram_mb: float = 0.0
    components: list = field(default_factory=list)          # chiavi dei componenti (per il riuso condiviso)
    busy: bool = False


def _stats(port: int, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=timeout) as r:
            return json.load(r)
    except Exception:                                       # noqa: BLE001
        return None


def residents() -> list[Resident]:
    """Fotografia dei modelli residenti dichiarati dai server persistenti attivi. I server riportano
    `model_key`/`components` in `/system_stats` (chi non lo fa ancora → chiave vuota, contribuisce solo VRAM)."""
    out: list[Resident] = []
    for port, cat in SERVER_PORTS.items():
        s = _stats(port)
        if s is None:
            continue
        out.append(Resident(
            port=port, category=cat,
            key=str(s.get("model_key", "")),
            vram_mb=float(s.get("vram_used_mb", s.get("model_vram_mb", 0)) or 0),
            components=list(s.get("components", []) or []),
            busy=bool(s.get("busy", False)),
        ))
    return out


def plan(want: ModelKey, *, category: str, need_mb: float = 0.0,
         headroom_mb: int = DEFAULT_HEADROOM_MB) -> dict:
    """Decide COSA fare prima di eseguire una pipeline che vuole `want` (categoria image/video).
    Ritorna un piano: {action, reuse_port?, free_ports?, reason, shared[]}.

      action:
        'reuse'         → il modello è GIÀ residente su un server: nessun reload.
        'load'          → non residente ma c'è spazio: carica senza sfrattare.
        'evict_load'    → serve spazio (o cambio-categoria): libera prima i server indicati, poi carica.

    Regole:
      - stessa chiave residente → reuse (mai reload).
      - categoria diversa da un residente pesante → quella va liberata (cambio-pipeline).
      - stessa categoria, chiave diversa → si tiene finché c'è spazio; si sfratta solo se manca VRAM.
      - i componenti condivisi (stesso nome) NON contano come da-liberare (li riusa il nuovo motore).
    """
    want_key = str(want)
    res = residents()
    free_mb = gpu_mem.free_vram_mb()

    # 1) già residente con la stessa chiave → riusa
    for r in res:
        if r.key and r.key == want_key:
            return {"action": "reuse", "reuse_port": r.port, "free_ports": [],
                    "reason": f"modello già residente su :{r.port}", "shared": r.components}

    # 2) server di ALTRA categoria residenti → vanno liberati (cambio pipeline foto↔video)
    other_cat = [r for r in res if r.category != category and r.vram_mb > 0]
    # 3) stessa categoria, chiave diversa → libera solo se manca spazio
    same_cat_diff = [r for r in res if r.category == category and r.key and r.key != want_key and r.vram_mb > 0]

    free_ports: list[int] = [r.port for r in other_cat]
    reason = []
    if other_cat:
        reason.append(f"cambio categoria → libero {[r.port for r in other_cat]}")

    projected_free = free_mb + sum(r.vram_mb for r in other_cat)
    if projected_free < need_mb + headroom_mb and same_cat_diff:
        # serve ancora spazio: sfratta i modelli stessa-categoria a chiave diversa (LRU-ish: tutti), NON i condivisi
        free_ports += [r.port for r in same_cat_diff]
        reason.append(f"serve spazio ({int(need_mb)}+{headroom_mb}MB) → libero {[r.port for r in same_cat_diff]}")

    action = "evict_load" if free_ports else "load"
    if not reason:
        reason.append("spazio sufficiente, nessuno sfratto")
    return {"action": action, "reuse_port": None, "free_ports": free_ports,
            "reason": "; ".join(reason), "shared": []}


def apply(planned: dict, *, timeout: float = 8.0) -> list[int]:
    """Esegue il piano: libera i server indicati (via HTTP /free, coordinato). Ritorna le porte liberate.
    `reuse`/`load` senza free_ports → no-op (niente da scaricare)."""
    ports = list(planned.get("free_ports") or [])
    if not ports:
        return []
    freed = mem_coordinator._free_ports(ports, timeout)     # riusa la meccanica /free esistente
    return freed


def acquire(want: ModelKey, *, category: str, need_mb: float = 0.0) -> dict:
    """Comodità: plan + apply in un colpo. Da chiamare PRIMA di eseguire la pipeline. Ritorna il piano
    (con 'freed' effettivo). Il caricamento vero lo fa il motore/server, che poi riporterà `model_key`."""
    p = plan(want, category=category, need_mb=need_mb)
    p["freed"] = apply(p) if p["action"] == "evict_load" else []
    return p


def before_job(category: str, *, arch: str = "", model: str = "", dtype: str = "",
               need_mb: float = 0.0, log=None) -> dict:
    """PUNTO DI CABLAGGIO nel dispatch (rimpiazza il coarse mem_coordinator.free_for con la decisione
    model-aware): chiamato PRIMA di eseguire un job. Se il modello richiesto è già residente → NON libera
    nulla (riuso). Al cambio-categoria (foto↔video) o mancanza spazio → libera i server giusti. Best-effort:
    non solleva mai (in caso d'errore ripiega sul comportamento neutro). Ritorna il piano applicato."""
    try:
        want = ModelKey(arch=arch or category, model=model, dtype=dtype)
        p = acquire(want, category=category, need_mb=need_mb)
        if log:
            try:
                log(f"[gpu_registry] {category}/{want}: {p['action']} — {p['reason']}"
                    + (f" (liberati {p['freed']})" if p.get("freed") else ""))
            except Exception:                               # noqa: BLE001
                pass
        return p
    except Exception:                                       # noqa: BLE001 — mai bloccare il job per il registro
        # fallback neutro: coordina come prima (libera le altre categorie)
        try:
            mem_coordinator.free_for(category)
        except Exception:                                   # noqa: BLE001
            pass
        return {"action": "fallback", "free_ports": [], "reason": "registro non disponibile"}
