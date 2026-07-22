"""Coordinatore memoria cross-server — unload PIPELINE-AWARE (M1).

Su GPU singola un solo tipo pesante di pipeline alla volta. Prima di un job di categoria C liberiamo i server
delle ALTRE categorie (è il "cambio pipeline": immagine→video→3D→mondi→musica), mentre i job della STESSA
categoria NON liberano nulla → il ModelManager/epoch riusa ciò che è già caricato (rigenerazioni successive
con lo stesso modello restano gratis).

Categorie e porte dei server persistenti:
- image  → :8189 (InferenceServer immagini) + :8200 (API pubblica) + :8191 (server Krea 2 / Krea 2 Edit)
- video  → :8190 (server video persistente LTX-2.3)
- 3d/world/music → subprocess EFFIMERI (GPU pulita a fine run) → prima di partire liberano TUTTO.

Solo stdlib (urllib): importabile anche da chiamanti leggeri. Best-effort, timeout corto, silenzioso.
"""
from __future__ import annotations

import json
import urllib.request

_IMAGE_PORTS = (8189, 8200, 8191)   # 8191 = server Krea 2 persistente (immagini/edit)
_VIDEO_PORTS = (8190,)

# Ricorda l'ultima categoria servita in QUESTO processo: due job consecutivi della stessa categoria non
# rifanno il /free (che sarebbe un no-op costoso in round-trip). Il cambio categoria lo forza.
_last: dict[str, str | None] = {"cat": None}


def _free_ports(ports, timeout: float, mode: str = "demote") -> list[int]:
    """POST /free ai server. `mode` (unified-model-manager):
    - "demote": libera la VRAM ma PARCHEGGIA i modelli in RAM (la pipeline successiva potrebbe
      riusarli, o essere ancora ignota) — i server che non conoscono il campo lo ignorano (=hard).
    - "destroy": distruggi anche il parcheggio RAM (il chiamante vuole VRAM *e* RAM)."""
    body = json.dumps({"mode": mode}).encode("utf-8")
    hit = []
    for p in ports:
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"http://127.0.0.1:{p}/free", data=body,
                                       headers={"Content-Type": "application/json"}),
                timeout=timeout).read()
            hit.append(p)
        except Exception:                                   # noqa: BLE001 — server assente/occupato: ignora
            pass
    return hit


def auto_mode() -> str:
    """«Anche la RAM va svuotata dai modelli caricati, se non servono più» (richiesta utente 22/07):
    il parcheggio RAM ha senso solo finché la RAM abbonda. Se la macchina è GIÀ corta (RAM fisica
    disponibile sotto il floor, o commit sotto il floor), il demote sposterebbe solo il problema
    dalla VRAM alla RAM → si passa a destroy: i modelli dell'altra categoria escono ANCHE dalla RAM
    (si rileggeranno da SSD quando torneranno a servire). Con memoria abbondante resta demote:
    il riuso a caldo image⇄video è il guadagno misurato della residenza (run 2 −28%)."""
    try:
        import psutil
        from core import mem_budget, st_direct
        if psutil.virtual_memory().available / 2**20 < mem_budget.ram_floor_mb():
            return "destroy"
        if st_direct.commit_avail_mb() < mem_budget.commit_floor_mb():
            return "destroy"
    except Exception:                                       # noqa: BLE001 — senza misure: comportamento storico
        pass
    return "demote"


def free_for(category: str, *, timeout: float = 8.0, force: bool = False) -> list[int]:
    """Libera i server delle categorie DIVERSE da `category`. Ritorna le porte effettivamente liberate.
    `force=True` bypassa il guard "stessa categoria consecutiva" (es. il pulsante GUI «Libera VRAM»).
    Cambio image⇄video → demote (i modelli dell'altra categoria restano parcheggiati in RAM, pronti
    al rientro), MA se RAM/commit sono sotto i floor si passa a destroy (vedi auto_mode);
    categorie EFFIMERE (3d/world/music) → destroy (vogliono anche la RAM)."""
    cat = (category or "").lower()
    if not force and _last["cat"] == cat:
        return []                                           # stessa categoria di fila → niente da cambiare
    _last["cat"] = cat
    if cat == "image":
        return _free_ports(_VIDEO_PORTS, timeout, mode=auto_mode())
    if cat == "video":
        return _free_ports(_IMAGE_PORTS, timeout, mode=auto_mode())
    # 3d / world / music / sconosciute → effimere: liberano TUTTO, RAM compresa
    return _free_ports(_IMAGE_PORTS + _VIDEO_PORTS, timeout, mode="destroy")


def free_all(*, timeout: float = 8.0) -> list[int]:
    """Libera TUTTI i server GPU persistenti (pulsante GUI «Libera VRAM») — hard: anche i parcheggi RAM.
    Resetta il guard di categoria."""
    _last["cat"] = None
    return _free_ports(_IMAGE_PORTS + _VIDEO_PORTS, timeout, mode="destroy")


def reset() -> None:
    """Azzera il guard di categoria: dopo un /free manuale esterno, il prossimo job rifà il /free giusto."""
    _last["cat"] = None
