"""Client dell'InferenceServer nativo (lato app PyQt) — stessa semantica di comfy_driver.

L'app NON importa torch/diffusers: parla col server nativo via HTTP (come fa con ComfyUI),
così la GUI resta leggera. Usato dall'astrazione `core/backend.py`.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.parse    # NB: DEVE stare qui in cima. Se importato DENTRO preview(), `urllib` diventa una
                       # variabile LOCALE dell'intera funzione → la 1ª riga urllib.request.Request(...) dà
                       # UnboundLocalError → preview() tornava sempre None (progress/anteprima mai aggiornati).

BASE = "http://127.0.0.1:8189"


def _post(path: str, payload: dict, timeout: int = 30) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def _get(path: str, timeout: int = 15) -> dict:
    return json.load(urllib.request.urlopen(BASE + path, timeout=timeout))


def is_up() -> bool:
    try:
        _get("/system_stats", timeout=1.5); return True
    except Exception:                                   # noqa: BLE001
        return False


def submit(module: str, params: dict) -> str:
    # M1 — prima di un job IMMAGINE libera i server di categoria diversa (video :8190): su single-GPU la
    # pipeline video residua terrebbe la VRAM. Pipeline-aware: immagine→immagine consecutive non rifanno il
    # /free (guard nel coordinatore). Best-effort, non blocca la submit se il coordinatore non è disponibile.
    try:
        from core import mem_coordinator
        mem_coordinator.free_for("image")
    except Exception:                                       # noqa: BLE001
        pass
    r = _post("/submit", {"module": module, "params": params})
    if "prompt_id" not in r:
        raise RuntimeError(r.get("error", "submit fallito"))
    return r["prompt_id"]


def history(prompt_id: str) -> dict:
    return _get(f"/history/{prompt_id}")


def run_and_wait(module: str, params: dict, *, poll: float = 1.0, timeout: float = 1800) -> dict:
    """Submit + polling fino a success/error. Ritorna l'entry job (con 'output')."""
    pid = submit(module, params)
    t0 = time.monotonic()
    _misses = 0
    while time.monotonic() - t0 < timeout:
        try:
            h = history(pid)                              # GET /history: può andare in timeout mentre il worker
            _misses = 0                                  # è nel decode/postprocess finale (server occupato)
        except Exception:                                # noqa: BLE001  — NON abortire per un singolo polling
            _misses += 1
            if _misses > 30:                             # ~30 polling falliti di fila = server davvero morto
                return {"status": "error", "error": "server nativo non risponde al polling /history"}
            time.sleep(poll)
            continue
        if h.get("status") in ("success", "error"):
            return h
        time.sleep(poll)
    return {"status": "timeout"}


def preview() -> dict | None:
    """Progresso + ultima anteprima del job in corso: {png, png_seq, step, total}. `png` può essere
    vuoto (progresso disponibile ma anteprima non ancora pronta); `png_seq` cresce ad ogni nuovo PNG
    (per ridisegnare solo i frame nuovi). Best-effort (non solleva)."""
    try:
        req = urllib.request.Request(BASE + "/preview")
        with urllib.request.urlopen(req, timeout=3) as r:
            data = r.read()
            return {"png": data,
                    "png_seq": int(r.headers.get("X-Preview-PngSeq", "0")),
                    "step": int(r.headers.get("X-Preview-Step", "0")),
                    "total": int(r.headers.get("X-Preview-Total", "0")),
                    "msg": urllib.parse.unquote(r.headers.get("X-Preview-Msg", ""))}
    except Exception:                                       # noqa: BLE001
        return None


def system_stats() -> dict:
    return _get("/system_stats")
