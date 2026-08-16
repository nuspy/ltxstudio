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
import urllib.error
import urllib.request

_IMAGE_PORTS = (8189, 8191)         # 8191 = server Krea 2 persistente (immagini/edit).
                                    # 8200 RIMOSSO: l'API non ha /free né /system_stats (POST garantiti a vuoto).
_VIDEO_PORTS = (8190, 8192)         # 8192 = krea_rt/Self-Forcing (prima ORFANO: mai liberato da nessuno).

# NOTA STORICA: qui c'era un guard "stessa categoria di fila → skip /free" (_last["cat"]). RIMOSSO:
# era per-processo (GUI e API si ingannavano a vicenda) e gpu_registry.apply() lo bypassava senza
# aggiornarlo → la sequenza video→immagine→video NON liberava il server immagini e il transformer
# 22B caricava con FLUX ancora residente (spill in shared WDDM). Il risparmio era qualche POST
# no-op sub-secondo per job: non vale la classe di bug. `_last`/`reset()` restano per compatibilità.
_last: dict[str, str | None] = {"cat": None}


def _free_ports(ports, timeout: float, mode: str = "demote") -> dict[int, str]:
    """POST /free ai server. Ritorna l'ESITO per porta:
      'freed'   → il server ha liberato (risposta OK);
      'busy'    → il server ha RIFIUTATO (job in corso: la sua VRAM resta occupata!);
      'absent'  → server non attivo (connessione rifiutata: non tiene VRAM*);
      'timeout' → server vivo ma lento: lo scarico potrebbe essere ANCORA IN CORSO;
      'error'   → endpoint rotto/risposta anomala.
    Prima gli errori erano inghiottiti (`except: pass`) e un timeout era indistinguibile da un
    server spento: il chiamante procedeva a caricare DURANTE lo scarico → spill in shared WDDM.
    (*il processo può essere vivo ma il server HTTP no: caso raro, coperto dal wait su VRAM reale.)
    `mode` (unified-model-manager): "demote" = VRAM libera + modelli parcheggiati in RAM;
    "destroy" = anche il parcheggio RAM va all'OS."""
    body = json.dumps({"mode": mode}).encode("utf-8")
    out: dict[int, str] = {}
    for p in ports:
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(f"http://127.0.0.1:{p}/free", data=body,
                                           headers={"Content-Type": "application/json"}),
                    timeout=timeout) as r:
                resp = json.loads(r.read() or b"{}")
            out[p] = "busy" if resp.get("busy") else "freed"
        except urllib.error.HTTPError:
            out[p] = "error"                                # endpoint esiste ma risponde male
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", None)
            out[p] = "timeout" if isinstance(reason, TimeoutError) else "absent"
        except TimeoutError:                                # socket.timeout è alias di TimeoutError (py3.10+)
            out[p] = "timeout"
        except Exception:                                   # noqa: BLE001
            out[p] = "error"
    bad = {p: o for p, o in out.items() if o not in ("freed", "absent")}
    if bad:
        print(f"[mem] /free mode={mode}: esiti anomali {bad} — la VRAM di quelle porte "
              "potrebbe essere ANCORA occupata", flush=True)
    return out


def _freed(out: dict[int, str]) -> list[int]:
    return [p for p, o in out.items() if o == "freed"]


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
    Si libera SEMPRE (il vecchio guard "stessa categoria consecutiva" è stato rimosso, vedi sopra):
    per un server senza nulla da scaricare il /free è un no-op sub-secondo. `force` resta per
    compatibilità di firma.
    Cambio image⇄video → demote (i modelli dell'altra categoria restano parcheggiati in RAM, pronti
    al rientro), MA se RAM/commit sono sotto i floor si passa a destroy (vedi auto_mode);
    categorie EFFIMERE (3d/world/music) → destroy (vogliono anche la RAM)."""
    cat = (category or "").lower()
    _last["cat"] = cat
    if cat == "image":
        return _freed(_free_ports(_VIDEO_PORTS, timeout, mode=auto_mode()))
    if cat == "video":
        return _freed(_free_ports(_IMAGE_PORTS, timeout, mode=auto_mode()))
    # 3d / world / music / llm / sconosciute → effimere: liberano TUTTO, RAM compresa
    return _freed(_free_ports(_IMAGE_PORTS + _VIDEO_PORTS, timeout, mode="destroy"))


def free_all(*, timeout: float = 8.0) -> list[int]:
    """Libera TUTTI i server GPU persistenti (pulsante GUI «Libera VRAM») — hard: anche i parcheggi RAM."""
    _last["cat"] = None
    return _freed(_free_ports(_IMAGE_PORTS + _VIDEO_PORTS, timeout, mode="destroy"))


def reset() -> None:
    """Compat storica (il guard di categoria non esiste più): azzera solo la nota informativa."""
    _last["cat"] = None
