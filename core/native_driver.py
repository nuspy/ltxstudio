"""Client dell'InferenceServer nativo (lato app PyQt) — stessa semantica di comfy_driver.

L'app NON importa torch/diffusers: parla col server nativo via HTTP (come fa con ComfyUI),
così la GUI resta leggera. Usato dall'astrazione `core/backend.py`.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
import urllib.parse    # NB: DEVE stare qui in cima. Se importato DENTRO preview(), `urllib` diventa una
                       # variabile LOCALE dell'intera funzione → la 1ª riga urllib.request.Request(...) dà
                       # UnboundLocalError → preview() tornava sempre None (progress/anteprima mai aggiornati).

from core import loopback

loopback.install()      # connessioni verso 127.0.0.1 da una porta sorgente che il filtro di rete locale non scarta (vedi core/loopback.py)

BASE = "http://127.0.0.1:8189"

_LAUNCH_LOCK = threading.Lock()         # serializza i riavvii: N chiamanti → UN solo server, non N


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


def _say(log, msg: str) -> None:
    (log or print)(msg)


def ensure_up(*, timeout: float = 180.0, log=None) -> bool:
    """Garantisce che il server nativo risponda: se è CADUTO lo riavvia e attende che torni pronto.

    Bloccante e senza Qt → chiamabile da qualunque thread (GUI, worker, API, Film Studio). Il riavvio è
    serializzato dal lock: se dieci job partono insieme mentre il server è morto, parte UN solo server e
    tutti aspettano quello. Un server appena avviato non occupa VRAM (carica i pesi solo al primo job),
    quindi rialzarlo non ruba memoria a una generazione video in corso.
    """
    if is_up():
        return True
    with _LAUNCH_LOCK:
        if is_up():                                         # rialzato da un altro chiamante mentre attendevamo
            return True
        from core.native_launcher import launch, supervise_startup
        ok, msg = launch()
        _say(log, f"[nativo] server non risponde → RIAVVIO AUTOMATICO: {msg}")
        if not ok:
            return False
        t0 = time.monotonic()
        if supervise_startup(is_up, log=lambda text: _say(log, text), total_timeout=timeout):
            _say(log, f"[nativo] server di nuovo pronto ({int(time.monotonic() - t0)}s)")
            return True
        _say(log, f"[nativo] il server non è tornato su entro {int(timeout)}s")
        return False


def submit(module: str, params: dict) -> str:
    # Registro GPU unico (model-aware): prima di un job IMMAGINE decide se RIUSARE il modello residente
    # (nessun /free) o liberare i server di categoria diversa / a modello diverso se serve spazio. Rimpiazza
    # il coarse free_for('image'). Best-effort: non blocca mai la submit (fallback neutro interno).
    _tbtn = float(params.get("_t_button") or 0)
    _rel = (lambda: f"+{time.time()-_tbtn:.1f}s dal bottone" if _tbtn else "")
    _tb = time.monotonic()
    try:
        from core import gpu_registry
        # F5 unified-model-manager: need_mb REALE (cache misurata → header) — prima era sempre 0 e
        # il ramo size-aware del piano (sfratto minimo intra-categoria) non scattava mai.
        _model = str(params.get("model", ""))
        _need, _dtype = 0, ""
        try:
            from core import vram_estimate
            if _model and os.path.isabs(_model) and os.path.exists(_model):
                _fp8 = "fp8" in os.path.basename(_model).lower()
                _need, _src = vram_estimate.estimate_mb(_model, fp8_storage=_fp8)
                _dtype = "fp8" if _fp8 else ""
        except Exception:                                   # noqa: BLE001
            pass
        _plan = gpu_registry.before_job("image", arch=str(params.get("arch", "")),
                                        model=_model, dtype=_dtype, need_mb=_need, log=print)
        # GARANZIA DI SCARICO COMPLETO: il /free risponde PRIMA che il driver abbia davvero
        # restituito la VRAM (WDDM). Se il piano ha sfrattato qualcuno, si aspetta il rilascio
        # EFFETTIVO (NVML) prima del submit — altrimenti il child misura una VRAM ancora occupata
        # e degrada a offload/sticky a torto. Con action=reuse o nessuno sfratto: niente attesa.
        if _plan and _plan.get("free_ports"):
            from core import gpu_mem
            gpu_mem.wait_free_vram_mb(max(_need, 8000) + 2000, timeout_s=40, tag="image")
    except Exception:                                       # noqa: BLE001
        pass
    print(f"[step-dispatch] {_rel()} | before_job('image') = {time.monotonic()-_tb:.1f}s "
          f"(libera/attende la VRAM dell'altra categoria) → ora /submit al server", flush=True)
    payload = {"module": module, "params": params}
    try:
        r = _post("/submit", payload)
    except urllib.error.HTTPError:
        raise                                               # il server C'È e ha risposto (400/500): errore vero
    except OSError as e:                                    # porta chiusa / connessione rifiutata = processo morto
        if not ensure_up():
            raise RuntimeError(f"server nativo giù e riavvio automatico fallito ({e})") from e
        r = _post("/submit", payload)                       # ritenta una volta sul server appena rialzato
    if "prompt_id" not in r:
        raise RuntimeError(r.get("error", "submit fallito"))
    return r["prompt_id"]


def history(prompt_id: str) -> dict:
    return _get(f"/history/{prompt_id}")


def run_and_wait(module: str, params: dict, *, poll: float = 1.0, timeout: float = 1800,
                 retry_on_crash: bool = True, log=None) -> dict:
    """Come `_run_and_wait`, poi (task 21) incorpora nel file FINALE i metadati di generazione: il motore ha
    già scritto, firmato (stego) e marchiato (watermark) l'output, quindi questa è l'ultima scrittura."""
    res = _run_and_wait(module, params, poll=poll, timeout=timeout, retry_on_crash=retry_on_crash, log=log)
    if module != "segment":                                  # le maschere di segmentazione non sono generazioni
        try:
            from core import gen_metadata
            gen_metadata.embed_result(res, params)
        except Exception:                                    # noqa: BLE001
            pass
    return res


def _run_and_wait(module: str, params: dict, *, poll: float = 1.0, timeout: float = 1800,
                  retry_on_crash: bool = True, log=None) -> dict:
    """Submit + polling fino a success/error. Ritorna l'entry job (con 'output').

    Se il server MUORE a metà job (OOM, segfault), il job muore con lui: lo rialziamo e rilanciamo la
    generazione UNA sola volta (`retry_on_crash`), così un crash isolato non si vede in GUI. La condizione
    è `not is_up()`: un server vivo ma lento (history in timeout perché sta decodificando) NON deve mai far
    ripartire il job — sarebbe lavoro doppio.
    """
    # Architetture immagine con venv/worker DEDICATO (es. Krea 2: Qwen3-VL/diffusers 0.39, incompatibile con
    # l'inference_server) → bypassano il server e girano in subprocess (core/img_gen_job), come il video.
    if module == "image":
        try:
            from core import img_gen_job
            if img_gen_job.handles(params.get("arch")):
                return img_gen_job.run(params, timeout=int(timeout))
        except Exception as _e:                              # noqa: BLE001 — se manca il dispatch dedicato, prosegui col server
            _say(log, f"[nativo] dispatch immagine dedicato non disponibile ({_e}) -> server")
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
                if retry_on_crash and not is_up() and ensure_up(log=log):
                    _say(log, "[nativo] il server è morto durante il job → rilancio la generazione (1 volta)")
                    return _run_and_wait(module, params, poll=poll, timeout=timeout,
                                         retry_on_crash=False, log=log)
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
