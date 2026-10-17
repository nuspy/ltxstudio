"""InferenceServer — backend nativo (sostituto di ComfyUI).

Server HTTP leggero (uvicorn) con la stessa semantica del driver ComfyUI (submit asincrono + polling
history) e una coda SERIALE. **Non tocca MAI torch**: ogni job viene eseguito in un PROCESSO FIGLIO isolato
(`job_runner`) che carica il modello, genera e muore.

Perché l'isolamento: PyTorch + safetensors non sono thread-safe dentro un server uvicorn di lunga durata su
Windows. Caricare un checkpoint mentre uvicorn serve richieste → access violation; caricarne un SECONDO
nello stesso processo (cambio modello) → segfault/MemoryError. Con un processo per job — che nasce, carica
UN modello e muore — entrambi i problemi spariscono alla radice (è come già gira il worker VIDEO). Costo:
il modello si ricarica ad ogni job (niente cache tra job); in cambio, zero crash.

Avvio:  <venv>/python -m inference_server.server   (porta 8189)
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import uuid

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = r"E:\AI\ltxstudio\output"
_JOB_DIR = os.path.join(PROJECT, "output", ".jobs")          # params/result/log per-job (IPC via file)
_PREVIEW_DIR = os.path.join(PROJECT, "output", ".preview")   # preview.png/preview.json scritti dal figlio
os.makedirs(_JOB_DIR, exist_ok=True)
os.makedirs(_PREVIEW_DIR, exist_ok=True)


def _tee_log() -> None:
    """Duplica stdout/stderr del server su output/server.log (oltre alla console)."""
    try:
        logdir = os.path.join(PROJECT, "output")
        os.makedirs(logdir, exist_ok=True)
        f = open(os.path.join(logdir, "server.log"), "a", buffering=1, encoding="utf-8", errors="replace")

        class _Tee:
            def __init__(self, *streams): self._s = [s for s in streams if s]
            def write(self, d):
                for s in self._s:
                    try: s.write(d)
                    except Exception: pass
            def flush(self):
                for s in self._s:
                    try: s.flush()
                    except Exception: pass
            def isatty(self):
                return False
            def fileno(self):
                for s in self._s:
                    try: return s.fileno()
                    except Exception: pass
                raise OSError("no fileno")
            def __getattr__(self, name):
                for s in self._s:
                    if hasattr(s, name): return getattr(s, name)
                raise AttributeError(name)
        sys.stdout = _Tee(sys.__stdout__, f)
        sys.stderr = _Tee(sys.__stderr__, f)
        print(f"[server] log su {os.path.join(logdir, 'server.log')}", flush=True)
    except Exception:                                       # noqa: BLE001
        pass


if os.environ.get("LTX_SERVER_NO_TEE") != "1":          # i test unitari importano il modulo senza dirottare stdout
    _tee_log()

# il server LEGGE la preview dai file scritti dal processo figlio (get_preview usa questa env)
os.environ["LTX2_SERVER_PREVIEW_DIR"] = _PREVIEW_DIR

from fastapi import FastAPI, Response
from pydantic import BaseModel

from inference_server import preview_bus                    # leggero: nessun import torch

app = FastAPI(title="LTX Studio — Native Inference Server")

# moduli validi (l'esecuzione avviene in job_runner, non qui — nessun handler/torch importato nel server)
_MODULES = {"image", "upscale", "segment", "video_upscale", "video_face", "video_fps"}
_jobs: dict[str, dict] = {}
_q: "queue.Queue[str]" = queue.Queue()


class SubmitReq(BaseModel):
    module: str
    params: dict


def _tail(path: str, n: int = 2000) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()[-n:]
    except Exception:                                       # noqa: BLE001
        return ""


# ─────────────── figlio PERSISTENTE — RESIDENZA del modello tra le gen (richiesta utente) ───────────────
# DEFAULT ON (unificazione motori, 22/07/2026): tiene UN figlio `persistent_runner` vivo tra i job → il
# modello resta CALDO tra le gen (niente reload da SSD ad ogni generazione). Riuso per job dello STESSO
# modello; al CAMBIO modello respawn di un figlio pulito. Il server resta senza torch → anti-crash intatto
# (EOF/timeout = respawn + FALLBACK PER-JOB automatico: nessuna regressione se il figlio muore).
# STORIA: era OFF per gli access-violation al 2° load su Windows (Ernie 16GB, Krea2) — la CAUSA era il
# mmap copy-on-write dei safetensors che impegna ~2× il checkpoint di COMMIT (pagefile), vedi la memoria
# pagefile-commit-mmap-crash. CURATA: persistent_runner patcha il load con core/st_direct (I/O diretto,
# commit-safe), FREE_DEMOTE parcheggia in RAM invece di distruggere, e i floor sono commit-aware.
# Opt-out: NATIVE_PERSISTENT_CHILD=0 (torna al per-job isolato puro).
_PERSIST = os.environ.get("NATIVE_PERSISTENT_CHILD", "1") != "0"
_RESPAWN_ON_CHANGE = os.environ.get("NATIVE_CHILD_RESPAWN", "change") != "never"


def _job_model_key(module: str, params: dict) -> str:
    """Chiave di RESIDENZA del job: dice QUALI PESI servono, non quale compito li usa (task 12: crea→modifica
    con lo stesso file = stessa chiave → il figlio NON viene ucciso). Definizione unica in core.model_identity,
    condivisa con persistent_runner e gpu_registry. Rollback: NATIVE_RESIDENCY_BY_TASK=1."""
    from core.model_identity import residency_key
    arch = str(params.get("arch", module) or module)
    return residency_key(arch, str(params.get("model", "") or params.get("model_gguf", "") or ""))


# Moduli che TENGONO un modello generativo residente nel figlio persistente. Gli altri (segment SAM2,
# upscale ESRGAN, video_*) sono AUSILIARI piccoli: prima passavano dal figlio persistente e, avendo una
# chiave diversa, UCCIDEVANO il modello immagini (flusso inpaint = segment → FLUX ricaricato due volte).
# Ora girano in un processo isolato (job_runner, già nel codice) e il figlio resta com'è; se manca VRAM
# per l'ausiliario il figlio viene PARCHEGGIATO in RAM (demote), mai ucciso. Rollback: NATIVE_HELPERS_ISOLATED=0.
_RESIDENT_MODULES = {"image"}
_HELPERS_ISOLATED = os.environ.get("NATIVE_HELPERS_ISOLATED", "1") != "0"
# VRAM libera MINIMA (MB) per far girare l'ausiliario accanto al modello residente: sotto → demote del figlio.
# Misurata a ogni job (NVML), non fissa per scheda; per-modulo perché SAM2+DINO pesa più di ESRGAN.
_HELPER_MIN_FREE_MB = {"segment": 6000, "upscale": 3000, "video_upscale": 6000, "video_face": 4000,
                       "video_fps": 3000}


def _helper_min_free_mb(module: str) -> int:
    env = os.environ.get("NATIVE_HELPER_MIN_FREE_MB")
    if env:
        try:
            return int(float(env))
        except ValueError:
            pass
    return int(_HELPER_MIN_FREE_MB.get(module, 4000))


def _use_persistent_child(module: str) -> bool:
    """True se il job va al figlio persistente (modello residente); False → processo isolato per-job."""
    if not (_PERSIST and _child is not None):
        return False
    return module in _RESIDENT_MODULES or not _HELPERS_ISOLATED


def _prepare_helper_vram(module: str, free_mb: float | None = None) -> bool:
    """Prima di un ausiliario isolato: se la VRAM libera non basta, PARCHEGGIA il modello residente (il
    figlio resta vivo, i pesi vanno in RAM e rientrano da lì). Ritorna True se ha fatto demote."""
    if _child is None or not _child._alive():
        return False
    if free_mb is None:
        try:
            from core.gpu_mem import free_vram_mb
            free_mb = free_vram_mb()
        except Exception:                                    # noqa: BLE001 — senza misura: non toccare nulla
            return False
    need = _helper_min_free_mb(module)
    if free_mb >= need:
        return False
    ok = _child.free(demote=True)
    print(f"[server] ausiliario '{module}': VRAM libera {int(free_mb)}MB < {need}MB → modello residente "
          f"{'PARCHEGGIATO in RAM' if ok else 'non parcheggiabile (job in corso)'}", flush=True)
    if ok:
        try:
            from core.gpu_mem import wait_free_vram_mb
            wait_free_vram_mb(need, timeout_s=20, tag=f"helper-{module}")
        except Exception:                                    # noqa: BLE001
            pass
    return bool(ok)


class _PersistentChild:
    """Un figlio `persistent_runner` vivo tra i job. Serializzato dal worker (coda seriale)."""

    _LOG = os.path.join(_JOB_DIR, "persistent_child.log")

    def __init__(self):
        self._proc = None
        self._reader = None
        self._logf = None
        self._lines: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()                        # serializza l'IPC (RUN vs STATS/FREE)
        self.model_key = "-"

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _spawn(self):
        # log PROPRIO del figlio (aperto UNA volta e tenuto aperto per tutta la sua vita): il figlio è
        # PERSISTENTE e sopravvive ai singoli job, quindi il suo stderr non può puntare a un logf per-job
        # che si chiude a fine job (→ write su fd chiuso → stallo/crash del figlio ai job successivi).
        try:
            self._logf = open(self._LOG, "a", encoding="utf-8", errors="replace")
        except Exception:                                    # noqa: BLE001
            self._logf = None
        self._proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "inference_server.persistent_runner"],
            cwd=PROJECT, env=dict(os.environ), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._logf, text=True, bufsize=1)
        self.model_key = "-"

        def _read():
            try:
                for ln in self._proc.stdout:
                    self._lines.put(ln.strip())
            except Exception:                                # noqa: BLE001
                pass
            self._lines.put("__EOF__")
        self._reader = threading.Thread(target=_read, daemon=True)
        self._reader.start()

    def _kill(self):
        if self._proc is not None:
            try:
                self._proc.stdin.write("QUIT\n"); self._proc.stdin.flush()
            except Exception:                                # noqa: BLE001
                pass
            try:
                self._proc.wait(timeout=5)
            except Exception:                                # noqa: BLE001
                try:
                    self._proc.kill()
                    # TerminateProcess (Windows) è ASINCRONO: senza wait() questa funzione tornava
                    # PRIMA che il driver recuperasse la VRAM del figlio → il chiamante misurava
                    # una VRAM ancora occupata (o caricava durante il rilascio).
                    self._proc.wait(timeout=10)
                except Exception:                            # noqa: BLE001
                    pass
        self._proc = None
        try:
            if self._logf:
                self._logf.close()
        except Exception:                                    # noqa: BLE001
            pass
        self._logf = None
        with self._lines.mutex:
            self._lines.queue.clear()

    def _send(self, line: str, wait_for: tuple, timeout: float):
        self._proc.stdin.write(line + "\n"); self._proc.stdin.flush()
        import time as _t
        t0 = _t.time()
        while _t.time() - t0 < timeout:
            try:
                out = self._lines.get(timeout=1.0)
            except Exception:                                # noqa: BLE001
                continue
            if out == "__EOF__":
                raise RuntimeError("figlio persistente morto (EOF)")
            if any(out.startswith(w) for w in wait_for):
                return out
        raise TimeoutError(f"figlio persistente: nessuna risposta a '{line}' entro {timeout}s")

    def run(self, module: str, params: dict, pf: str, rf: str, timeout: float = 1800) -> None:
        key = _job_model_key(module, params)
        with self._lock:
            # cambio di PESI (o figlio morto) → respawn pulito (evita il 2° load in-process). `key` è la chiave di
            # residenza (famiglia:file): crea→modifica con lo STESSO file non cambia chiave → nessun respawn.
            # NATIVE_CHILD_RESPAWN=never: nessun respawn al cambio (swap in-process, i parcheggi RAM restano
            # utilizzabili) — da abilitare solo dopo il test dal vivo; default "change" = comportamento storico.
            _changed = self.model_key not in ("-", key) and _RESPAWN_ON_CHANGE
            if not self._alive() or _changed:
                self._kill(); self._spawn()
            out = self._send(f"RUN {pf} {rf}", ("DONE", "ERR"), timeout)
            if out.startswith("DONE"):
                self.model_key = out[len("DONE "):].strip() or key

    def stats(self, timeout: float = 5) -> dict | None:
        # NON-BLOCCANTE: se un job è in corso (lock preso) ripiega sulla chiave cachata (niente race su stdin).
        if not self._alive():
            return None
        if not self._lock.acquire(blocking=False):
            return {"model_key": self.model_key, "components": [], "vram_mb": 0, "busy": True}
        try:
            sf = os.path.join(_JOB_DIR, "child_stats.json")
            self._send(f"STATS {sf}", ("STATS_OK",), timeout)
            with open(sf, encoding="utf-8") as f:
                return json.load(f)
        except Exception:                                    # noqa: BLE001
            return {"model_key": self.model_key, "components": [], "vram_mb": 0}
        finally:
            self._lock.release()

    def free(self, timeout: float = 10, demote: bool = False) -> bool:
        # /free = cambio-pipeline (un altro motore vuole la VRAM).
        # demote=True (unified-model-manager): il figlio RESTA VIVO e parcheggia i modelli in RAM
        # (FREE_DEMOTE → demote nel ModelManager/ComponentStore): se la pipeline successiva torna
        # alle immagini, restore da RAM senza SSD. I load sono commit-safe (st_direct nel runner),
        # che era la causa del segfault "2° load in-process". Fallback: kill come prima.
        # demote=False: KILLA il figlio (VRAM *e* RAM rilasciate all'OS; prossimo job = figlio fresco).
        # NON-BLOCCANTE sul lock del job: prima si bloccava fino a 1800s dietro il job in corso
        # mentre il client HTTP mollava a 8-15s e caricava COMUNQUE → ritorna False = "busy",
        # il chiamante lo vede nell'esito e decide (aspetta/riprova, mai caricare alla cieca).
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if demote and self._alive():
                try:
                    self._send("FREE_DEMOTE", ("FREED",), timeout)
                    return True
                except Exception:                            # noqa: BLE001 — figlio muto → kill
                    pass
            self._kill()
            self.model_key = "-"
            return True
        finally:
            self._lock.release()


_child = _PersistentChild() if _PERSIST else None


def _worker():
    """Coda seriale: per ogni job lancia un PROCESSO FIGLIO isolato (job_runner) e ne raccoglie il risultato.
    Il server non importa torch → non può crashare per il caricamento; se il figlio muore di crash nativo,
    lo rileviamo (result mancante + exit code) e restituiamo un errore chiaro col tail del suo log."""
    while True:
        jid = _q.get()
        job = _jobs.get(jid)
        if not job:
            _q.task_done()
            continue
        job["status"] = "running"
        preview_bus.reset()                                 # azzera l'anteprima del job precedente
        pf = os.path.join(_JOB_DIR, f"job_{jid}.json")
        rf = os.path.join(_JOB_DIR, f"res_{jid}.json")
        lf = os.path.join(_JOB_DIR, f"log_{jid}.txt")
        try:
            with open(pf, "w", encoding="utf-8") as f:
                json.dump({"module": job["module"], "params": job["params"]}, f)
            env = dict(os.environ)
            env["LTX2_PREVIEW_DIR"] = _PREVIEW_DIR          # il figlio scrive preview.png/json qui
            env.pop("LTX2_SERVER_PREVIEW_DIR", None)        # nel figlio preview_bus SCRIVE, non legge
            done_persistent = False
            proc = None
            if _use_persistent_child(job["module"]):
                # FIGLIO PERSISTENTE: riusa il modello caldo tra i job (stesso modello). Il figlio ha un log
                # PROPRIO (persistent_child.log). Fallback al per-job se non risponde/muore → nessuna regressione.
                try:
                    _child.run(job["module"], job["params"], pf, rf)
                    done_persistent = True
                except Exception as _pe:                     # noqa: BLE001
                    with open(lf, "a", encoding="utf-8", errors="replace") as logf:
                        logf.write(f"\n[server] figlio persistente ko ({_pe}) → fallback per-job\n")
            if not done_persistent:
                if _PERSIST and _child is not None and job["module"] not in _RESIDENT_MODULES:
                    _prepare_helper_vram(job["module"])      # ausiliario isolato: se manca VRAM, demote del residente
                with open(lf, "w", encoding="utf-8", errors="replace") as logf:
                    proc = subprocess.Popen(
                        [sys.executable, "-u", "-m", "inference_server.job_runner", pf, rf],
                        cwd=PROJECT, env=env, stdout=logf, stderr=logf)
                    job["_pid"] = proc.pid
                    proc.wait()
            if os.path.exists(rf):
                with open(rf, encoding="utf-8") as f:
                    res = json.load(f)
            else:                                           # figlio morto senza risultato = crash nativo
                # Col figlio PERSISTENTE `proc` non esiste (nessuna Popen per-job) e il log del crash sta
                # nel suo file, non in quello del job: leggere `proc.returncode` qui sollevava
                # UnboundLocalError, che sostituiva l'errore VERO e faceva sparire il log utile.
                _chi = "figlio persistente" if done_persistent else f"exit {proc.returncode}"
                _log = _tail(_PersistentChild._LOG) if done_persistent else _tail(lf)
                res = {"status": "error",
                       "error": f"job '{job['module']}' terminato senza risultato ({_chi}). "
                                f"Log:\n{_log}"}
            job.update(res)
            job["status"] = res.get("status", "success")
        except Exception as e:                              # noqa: BLE001
            job["status"] = "error"
            job["error"] = str(e)
        finally:
            for p in (pf, rf):
                try: os.remove(p)
                except Exception: pass                      # noqa: BLE001
            _q.task_done()


threading.Thread(target=_worker, daemon=True).start()


@app.post("/submit")
def submit(req: SubmitReq):
    if req.module not in _MODULES:
        return {"error": f"modulo non supportato: {req.module}"}
    jid = uuid.uuid4().hex
    _jobs[jid] = {"status": "pending", "module": req.module, "params": req.params, "output": []}
    _q.put(jid)
    return {"prompt_id": jid}


@app.get("/history/{jid}")
def history(jid: str):
    return _jobs.get(jid, {"status": "unknown"})


@app.get("/preview")
def preview():
    """Progresso + ultima anteprima per-step (letti dai file scritti dal processo figlio)."""
    import urllib.parse
    p = preview_bus.get_preview()
    return Response(content=p.get("png") or b"", media_type="image/png",
                    headers={"X-Preview-PngSeq": str(p["png_seq"]), "X-Preview-Step": str(p["step"]),
                             "X-Preview-Total": str(p["total"]),
                             "X-Preview-Msg": urllib.parse.quote(p.get("msg", ""))})


@app.get("/system_stats")
def system_stats():
    """VRAM via NVML/nvidia-smi — NIENTE torch.cuda nel server (nessun contesto CUDA, nessun crash)."""
    try:
        from core.gpu_mem import free_vram_mb
        free = free_vram_mb()
    except Exception:                                       # noqa: BLE001
        free = 0
    out = {"vram_free_mb": free, "loaded": [], "isolated": not _PERSIST}
    if _PERSIST and _child is not None:                     # figlio persistente → stato OSSERVABILE per il registro
        st = _child.stats()
        if st:
            out["model_key"] = st.get("model_key", _child.model_key)
            out["components"] = [c.get("key") for c in st.get("components", [])]
            out["model_vram_mb"] = st.get("vram_mb", 0)
            out["busy"] = bool(st.get("busy", False))       # job in corso: vram_mb=0 ma NON è "vuoto"
            out["loaded"] = out["components"]
        else:
            out["model_key"] = _child.model_key
    return out


class _FreeReq(BaseModel):
    mode: str = "demote"                                    # "demote" (parcheggia in RAM) | "destroy"


@app.post("/free")
def free(req: _FreeReq | None = None):
    # per-job isolato: niente da liberare (il figlio muore da sé). Con figlio persistente:
    # mode=demote → il figlio parcheggia i modelli in RAM e resta vivo (rientro veloce);
    # mode=destroy → kill del figlio (VRAM e RAM all'OS). Vedi mem_coordinator.
    mode = (req.mode if req else "demote").lower()
    if _PERSIST and _child is not None:
        ok = _child.free(demote=(mode != "destroy"))
        if not ok:                                          # job in corso: la VRAM resta al job
            return {"ok": False, "busy": True, "mode": mode}
        return {"ok": True, "busy": False, "freed": "persistent_child_store", "mode": mode}
    return {"ok": True, "busy": False, "mode": mode}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8189)
