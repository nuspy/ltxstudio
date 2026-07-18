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

import urllib.request

_IMAGE_PORTS = (8189, 8200, 8191)   # 8191 = server Krea 2 persistente (immagini/edit)
_VIDEO_PORTS = (8190,)

# Ricorda l'ultima categoria servita in QUESTO processo: due job consecutivi della stessa categoria non
# rifanno il /free (che sarebbe un no-op costoso in round-trip). Il cambio categoria lo forza.
_last: dict[str, str | None] = {"cat": None}


def _free_ports(ports, timeout: float) -> list[int]:
    hit = []
    for p in ports:
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"http://127.0.0.1:{p}/free", data=b"{}",
                                       headers={"Content-Type": "application/json"}),
                timeout=timeout).read()
            hit.append(p)
        except Exception:                                   # noqa: BLE001 — server assente/occupato: ignora
            pass
    return hit


def free_for(category: str, *, timeout: float = 8.0, force: bool = False) -> list[int]:
    """Libera i server delle categorie DIVERSE da `category`. Ritorna le porte effettivamente liberate.
    `force=True` bypassa il guard "stessa categoria consecutiva" (es. il pulsante GUI «Libera VRAM»)."""
    cat = (category or "").lower()
    if not force and _last["cat"] == cat:
        return []                                           # stessa categoria di fila → niente da cambiare
    _last["cat"] = cat
    if cat == "image":
        return _free_ports(_VIDEO_PORTS, timeout)
    if cat == "video":
        return _free_ports(_IMAGE_PORTS, timeout)
    # 3d / world / music / sconosciute → effimere: libera TUTTI i server persistenti
    return _free_ports(_IMAGE_PORTS + _VIDEO_PORTS, timeout)


def free_all(*, timeout: float = 8.0) -> list[int]:
    """Libera TUTTI i server GPU persistenti (pulsante GUI «Libera VRAM»). Resetta il guard di categoria."""
    _last["cat"] = None
    return _free_ports(_IMAGE_PORTS + _VIDEO_PORTS, timeout)


def reset() -> None:
    """Azzera il guard di categoria: dopo un /free manuale esterno, il prossimo job rifà il /free giusto."""
    _last["cat"] = None
