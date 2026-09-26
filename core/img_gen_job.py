"""Dispatch IMMAGINI per architetture con venv/worker DEDICATO (fuori dall'inference_server).

Le immagini non hanno un chokepoint unico come il video (`ltx_video_job`): i chiamanti usano
`native_driver.run_and_wait("image", params)` con `params["arch"]`, e il dispatch per famiglia avviene
DENTRO l'inference_server (venv diffusers 0.33, transformers <5). Ma alcune architetture (es. **Krea 2**:
Qwen3-VL + diffusers 0.39) NON possono girare in quel venv. Per queste si usa lo STESSO pattern del video:
un subprocess one-shot in un venv dedicato, instradato da qui.

`native_driver.run_and_wait` intercetta `params["arch"]` presente in `_ARCH_RUNNER` e delega a `run()`.
Ritorna il formato del server: {"status": "success"|"error", "output": [png], "error": ...}.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(REPO / "output")

# arch immagine → (python.exe del venv, worker.py). Krea 2 riusa il venv LingBot (stesse deps).
_LINGBOT_PY = r"E:\AI\Tools\lingbot_venv311\Scripts\python.exe"
_DIFFUSERS_NEXT_PY = r"E:\AI\Tools\diffusers_next_venv\Scripts\python.exe"   # scripts/install_diffusers_next.py
_ARCH_RUNNER = {
    "krea2": (_LINGBOT_PY, str(REPO / "workers" / "krea2_infer.py")),
    # editing per istruzione (single/multi) — stesso worker, ramo edit scelto da params['arch']
    "krea2_edit": (_LINGBOT_PY, str(REPO / "workers" / "krea2_infer.py")),
    # Qwen-Image 2.1: t2i + edit fino a 10 riferimenti + RGBA, nel venv condiviso con MiniMax H3 (diffusers main)
    "qwen_image21": (_DIFFUSERS_NEXT_PY, str(REPO / "workers" / "qwen_image21_infer.py")),
}


# Server Krea 2 PERSISTENTE (:8191): se attivo, il pipe è già caldo → nessun reload per shot.
# Autostart opzionale (KREA2_SERVER_AUTOSTART=1).
# NB: SOLO il t2i puro ('krea2') passa dal server. L'EDIT ('krea2_edit') NO: eseguire più job Krea nello
# stesso processo (il server) corrompe uno stato globale torch/diffusers (attention dispatch native↔
# flash_varlen, autocast) che sopravvive anche allo sfratto del pipe → l'edit perde l'identità (drift, es.
# volto anime al posto del riferimento). Ogni edit gira quindi in un SUBPROCESS fresco (un solo job per
# processo = nessun accumulo di stato = coerenza garantita). Il t2i non ha coerenza da preservare → server ok.
KREA2_SERVER = os.environ.get("KREA2_SERVER", "http://127.0.0.1:8191")
# DISABILITATO (single job per processo): finché lo stato globale torch/diffusers non viene isolato, il
# server è OFF per default. Riabilitabile con KREA2_USE_SERVER=1 (solo per t2i puro, MAI per l'edit).
_SERVER_ARCHS = ("krea2",) if os.environ.get("KREA2_USE_SERVER") == "1" else ()


def handles(arch: str | None) -> bool:
    """True se questa arch immagine ha un worker dedicato (bypassa l'inference_server)."""
    return (arch or "").lower() in _ARCH_RUNNER


def _server_up() -> bool:
    """True se il server Krea 2 persistente risponde (pipe già in VRAM)."""
    import json as _j
    import urllib.request as _u
    try:
        with _u.urlopen(KREA2_SERVER + "/system_stats", timeout=1.5) as r:
            return bool(_j.loads(r.read()).get("up", False))
    except Exception:                                        # noqa: BLE001
        return False


def _submit_persistent(params: dict, out: str, timeout: int) -> dict:
    """Invia il job al server persistente e attende (poll /history). Il pipe Krea 2 resta caldo tra i job."""
    import json as _j
    import time as _t
    import urllib.request as _u
    data = _j.dumps({"params": params}).encode()
    req = _u.Request(KREA2_SERVER + "/submit", data=data, headers={"Content-Type": "application/json"})
    jid = _j.loads(_u.urlopen(req, timeout=30).read())["prompt_id"]
    t0 = _t.time()
    while _t.time() - t0 < (timeout or 1800):
        with _u.urlopen(KREA2_SERVER + f"/history/{jid}", timeout=10) as r:
            h = _j.loads(r.read())
        st = h.get("status")
        if st == "success":
            got = (h.get("output") or [out])[0]
            if got and os.path.exists(got) and got != out:
                import shutil
                shutil.copyfile(got, out)
            return {"status": "success", "output": [out if os.path.exists(out) else got]}
        if st in ("error", "unknown"):
            return {"status": "error", "error": h.get("error") or f"server Krea 2: stato {st}"}
        _t.sleep(1)
    return {"status": "error", "error": "timeout server Krea 2 persistente"}


def maybe_autostart_server() -> bool:
    """Avvia il server Krea 2 persistente (se non già su) quando KREA2_SERVER_AUTOSTART=1. Ritorna True se su.
    Modellato su auto_run per ltx2_server: usato dagli orchestratori (Lucky Director / Cinema Studio) prima di
    una raffica di shot, così Krea 2 si carica UNA volta invece che a ogni immagine."""
    if _server_up():
        return True
    if os.environ.get("KREA2_SERVER_AUTOSTART", "0") != "1":
        return False
    py = _LINGBOT_PY
    if not os.path.exists(py):
        return False
    try:
        import time as _t
        log = open(os.path.join(OUTPUT_DIR, "krea2_server_8191.log"), "a", encoding="utf-8")
        env = dict(os.environ)
        env["HF_HUB_OFFLINE"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        subprocess.Popen([py, "-m", "workers.krea2_server"], cwd=str(REPO), env=env,
                         stdout=log, stderr=subprocess.STDOUT)
        for _ in range(60):                                  # attesa avvio uvicorn (import pesanti)
            if _server_up():
                print("[img_gen_job] server Krea 2 persistente :8191 avviato", flush=True)
                return True
            _t.sleep(1)
    except Exception as _e:                                  # noqa: BLE001
        print(f"[img_gen_job] autostart server Krea 2 fallito ({_e}) → subprocess per-shot", flush=True)
    return False


def run(params: dict, *, timeout: int | None = 1800) -> dict:
    """Esegue un job immagine col worker dedicato. Ritorna {status, output:[png]} come il server.
    Se il server Krea 2 persistente (:8191) è su, instrada lì (pipe caldo); altrimenti subprocess one-shot."""
    arch = (params.get("arch") or "").lower()
    py, worker = _ARCH_RUNNER.get(arch, (None, None))
    if not py:
        return {"status": "error", "error": f"arch immagine senza worker dedicato: {arch}"}
    out = params.get("output") or os.path.join(OUTPUT_DIR, f"krea_{arch}_{os.getpid()}.png")
    params = dict(params, output=out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if not os.path.exists(py):
        installer = "install_diffusers_next.py" if py == _DIFFUSERS_NEXT_PY else "install_krea.py"
        return {"status": "error",
                "error": f"venv per '{arch}' mancante: {py}. Esegui scripts/{installer}"}
    # libera la VRAM delle altre categorie (video) prima del modello pesante
    try:
        from core import mem_coordinator as _mc
        _mc.free_for("image", timeout=20)
    except Exception:                                        # noqa: BLE001
        pass
    # Krea 2 è un modello IMMAGINE DIVERSO in un venv SEPARATO → deve avere la VRAM tutta per sé: libera anche
    # gli ALTRI motori immagine RESIDENTI (diffusers :8189 col figlio persistente + comfy), altrimenti la
    # residenza lascia FLUX/comfy in VRAM e Krea2 va in OOM caricando il Qwen3-VL (crash su 'Loading checkpoint
    # shards'). Single-GPU: un modello pesante alla volta.
    try:
        import urllib.request
        urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:8189/free", data=b"{}", headers={"Content-Type": "application/json"}),
            timeout=20).read()
    except Exception:                                        # noqa: BLE001
        pass
    try:
        from core import img_router
        img_router._comfy_child.free()
    except Exception:                                        # noqa: BLE001
        pass
    # GARANZIA DI SCARICO COMPLETO: i /free rispondono prima che il driver restituisca la VRAM
    # (WDDM). Krea2+Qwen3-VL sono ~20GB: se il load parte nel buco → shared-RAM/OOM. Se il server
    # :8191 ha già la pipe calda la VRAM è legittimamente SUA → attesa corta (solo il rilascio
    # degli altri), altrimenti attesa piena.
    try:
        from core import gpu_mem
        _hot = arch in _SERVER_ARCHS and _server_up()
        try:
            import urllib.request as _u2
            import json as _j2
            _hot = _hot and int(_j2.loads(_u2.urlopen(
                "http://127.0.0.1:8191/system_stats", timeout=3).read()).get("cached_pipelines", 0) or 0) > 0
        except Exception:                                    # noqa: BLE001
            _hot = False
        if _hot:
            gpu_mem.wait_free_vram_mb(6000, timeout_s=15, tag="krea2-res")
        else:
            gpu_mem.wait_free_vram_mb(22000, timeout_s=40, tag="krea2")
    except Exception:                                        # noqa: BLE001
        pass
    # PIPE CALDO: se il server persistente serve questa arch ed è su (o autostartabile), instrada lì.
    if arch in _SERVER_ARCHS and (_server_up() or maybe_autostart_server()):
        try:
            return _submit_persistent(params, out, timeout or 1800)
        except Exception as _e:                              # noqa: BLE001 — server morto a metà → fallback subprocess
            print(f"[img_gen_job] server Krea 2 ko ({_e}) → fallback subprocess", flush=True)
    jf = out + ".job.json"
    with open(jf, "w", encoding="utf-8") as fh:
        json.dump(params, fh)
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    try:
        r = subprocess.run([py, "-u", worker, jf], env=env, cwd=str(REPO),
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"timeout generazione immagine ({arch})"}
    finally:
        try:
            os.remove(jf)
        except OSError:
            pass
    if os.path.exists(out):
        return {"status": "success", "output": [out]}
    tail = ((r.stdout or "")[-800:] + (r.stderr or "")[-800:]).strip()
    return {"status": "error", "error": f"immagine non generata ({arch}):\n{tail}"}
