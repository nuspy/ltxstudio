"""Router immagini: sceglie il MOTORE (comfy-native vs diffusers) secondo user_prefs.get_engine('image').

- 'comfy'  → workers/img_comfy_native.py (model_management ComfyUI, VRAM dinamica, scala 8→48GB; FLUX.2
             ~11x più veloce del block-swap diffusers). Solo t2i per ora; con FALLBACK automatico a
             diffusers per op non coperte (inpaint/kontext/integra) o in caso di errore → niente rotture.
- 'diffusers' → core.native_driver (inference_server :8189, image_flux.py). Path storico, intatto.

Default 'comfy'. Mantiene ENTRAMBI i percorsi (scelta in GUI/settings).
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

from core import native_driver, user_prefs

COMFY_PY = r"E:\AI\Comfy_Nexus.bak\ComfyUI\venv\Scripts\python.exe"
# ComfyUI RIMOSSO (cartelle rinominate .bau): il motore comfy non è più disponibile → si usa SOLO il nativo.
# Se un giorno ComfyUI tornasse, questo torna True da solo. Il memory-management va replicato NATIVO seguendo
# la logica di ComfyUI (model_management), non chiamando ComfyUI. Vedi [[native-backend-migration]].
_COMFY_AVAILABLE = os.path.exists(COMFY_PY)
REPO = str(Path(__file__).resolve().parent.parent)
OUTPUT_DIR = r"E:\AI\ltxstudio\output"
_DM_ROOT = r"E:\AI\Models\diffusion_models"

# componenti di default per arch (ComfyUI folder-relative): clip(+clip2), type, vae
_ARCH_COMPONENTS = {
    "flux2": {"clip1": "Flux/mistral_3_small_flux2.safetensors", "clip2": "", "clip_type": "flux2",
              "vae": "Flux/flux2-vae.safetensors"},
    "flux":  {"clip1": "clip_l.safetensors", "clip2": "t5xxl_fp16.safetensors", "clip_type": "flux",
              "vae": "ae.safetensors"},
}


def _dm_relative(model: str) -> str:
    if not model:
        return model
    m = model.replace("/", "\\")
    root = _DM_ROOT.rstrip("\\") + "\\"
    if m.lower().startswith(root.lower()):
        m = m[len(root):]
    return m.replace("\\", "/")


def _detect_arch(model_path: str) -> str:
    """Famiglia → arch comfy worker (flux2/flux). Default flux."""
    try:
        from core import model_detect
        fam = (model_detect.get(model_path) or {}).get("family", "")
        return "flux2" if fam == "flux2" else "flux"
    except Exception:                                          # noqa: BLE001
        m = (model_path or "").lower()
        return "flux2" if any(t in m for t in ("flux2", "flux_2", "flux 2")) else "flux"


class _ComfyChild:
    """Figlio comfy PERSISTENTE (residenza): `img_comfy_native --persist` vivo tra le gen → modelli CALDI nella
    cache di ComfyUI (niente reload dalla 2ª gen). IPC via **reader-thread** (thread di sfondo che legge stdout in
    una Queue) + `get(timeout)`: il thread chiamante NON si blocca mai su readline (la 1ª versione deadlockava la
    GUI se il figlio moriva in init). Serializzato dal lock. Fallback one-shot se muore/timeout."""

    def __init__(self):
        import threading
        self._proc = None
        self._q = None
        self._lock = threading.Lock()

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _reader(self, proc, q) -> None:
        try:
            for line in proc.stdout:                          # blocca QUI, in un thread di sfondo (non la GUI)
                q.put(line.rstrip("\n"))
        except Exception:                                     # noqa: BLE001
            pass
        q.put(None)                                           # sentinella EOF

    def _spawn(self, env: dict) -> None:
        import queue
        import threading
        self._q = queue.Queue()
        self._proc = subprocess.Popen(
            [COMFY_PY, os.path.join(REPO, "workers", "img_comfy_native.py"), "--persist"],
            cwd=REPO, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        threading.Thread(target=self._reader, args=(self._proc, self._q), daemon=True).start()
        self._wait_for(("READY",), 1200)                      # 1ª init ComfyUI (+eventuale load) può essere lunga
        print("[comfy] figlio persistente PRONTO (modelli caldi tra le gen)", flush=True)

    def _wait_for(self, prefixes: tuple, timeout: float) -> str:
        """Legge dalla Queue del reader-thread fino a una riga con prefisso atteso, con timeout e check-alive.
        Mai bloccante indefinito: se il figlio muore o non risponde → solleva → fallback one-shot."""
        import queue
        import time as _t
        t0 = _t.time()
        while _t.time() - t0 < timeout:
            try:
                line = self._q.get(timeout=2.0)
            except queue.Empty:
                if not self._alive():
                    self._proc = None
                    raise RuntimeError("comfy persistente morto (nessun output)")
                continue
            if line is None:
                self._proc = None
                raise RuntimeError("comfy persistente morto (EOF)")
            if line.startswith("ERR:"):
                raise RuntimeError(line[4:])
            for p in prefixes:
                if line.startswith(p):
                    return line
            # altre righe (READY atteso, o log [load]/[gen]) → passa oltre
        raise TimeoutError("timeout comfy persistente")

    def free(self) -> bool:
        """Killa il figlio comfy → libera la VRAM (coordinamento cross-motore quando gira diffusers/PuLID su
        single-GPU). Il prossimo job comfy respawna. Ritorna True se ha killato qualcosa."""
        with self._lock:
            if self._proc is None:
                return False
            try:
                self._proc.terminate(); self._proc.wait(timeout=8)
            except Exception:                                # noqa: BLE001
                try: self._proc.kill()
                except Exception: pass                       # noqa: BLE001
            self._proc = None
            print("[comfy] figlio persistente liberato (VRAM per l'altro motore)", flush=True)
            return True

    def run(self, pj: str, env: dict, timeout: int) -> str:
        with self._lock:
            if not self._alive():
                self._spawn(env)
            self._proc.stdin.write(pj + "\n"); self._proc.stdin.flush()
            line = self._wait_for(("OK:",), timeout)
            return line[3:].strip()


_comfy_child = _ComfyChild()


def _comfy_t2i(params: dict, timeout: int) -> dict:
    """Genera t2i col worker comfy-native. Figlio PERSISTENTE (modelli caldi) se COMFY_PERSISTENT!=0, con
    fallback al subprocess one-shot. Ritorna {status, output:[png]}."""
    model_abs = params.get("model", "")
    arch = _detect_arch(model_abs)
    comp = _ARCH_COMPONENTS.get(arch, _ARCH_COMPONENTS["flux"])
    p = {
        "arch": arch, "model": _dm_relative(model_abs),
        "clip1": params.get("encoder") or comp["clip1"], "clip2": comp["clip2"],
        "clip_type": comp["clip_type"], "vae": params.get("vae") or comp["vae"],
        "prompt": params.get("prompt", ""), "negative": params.get("negative", ""),
        "width": int(params.get("width", 1024)), "height": int(params.get("height", 1024)),
        "steps": int(params.get("steps", 20)), "guidance": float(params.get("guidance", 3.5)),
        "sampler": params.get("sampler") or "euler", "scheduler": params.get("scheduler") or "simple",
        "seed": int(params.get("seed", 0)),
        "output": params.get("output") or os.path.join(OUTPUT_DIR, f"comfy_{uuid.uuid4().hex[:8]}.png"),
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pj = os.path.join(OUTPUT_DIR, f"_iparams_{uuid.uuid4().hex[:8]}.json")
    Path(pj).write_text(json.dumps(p), encoding="utf-8")
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    # libera la VRAM del server diffusers :8189 (single-GPU: un modello pesante alla volta)
    try:
        import urllib.request
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8189/free", data=b"{}",
                               headers={"Content-Type": "application/json"}), timeout=20).read()
    except Exception:                                          # noqa: BLE001
        pass
    # FIGLIO PERSISTENTE (residenza): modelli caldi tra le gen. Ora con **reader-thread non-bloccante + timeout**
    # (validato con mock: no deadlock, riuso a caldo, free) → default ON. Fallback one-shot se il figlio muore.
    # Disattivabile: COMFY_PERSISTENT=0.
    if os.environ.get("COMFY_PERSISTENT", "1") != "0":
        try:
            out = _comfy_child.run(pj, env, timeout)
            return {"status": "success", "output": [out]}
        except Exception as _pe:                               # noqa: BLE001
            print(f"[comfy] figlio persistente ko ({_pe}) → fallback one-shot", flush=True)
        finally:
            try: os.remove(pj)
            except OSError: pass
    try:
        r = subprocess.run([COMFY_PY, os.path.join(REPO, "workers", "img_comfy_native.py"), pj],
                           cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "timeout comfy t2i"}
    finally:
        try:
            os.remove(pj)
        except OSError:
            pass
    out = (r.stdout or "") + (r.stderr or "")
    for line in out.splitlines():
        if line.startswith("OK:"):
            return {"status": "success", "output": [line[3:].strip()]}
    return {"status": "error", "error": out[-1500:]}


def generate(params: dict, timeout: int = 1800) -> dict:
    """Genera un'immagine instradando al motore scelto. t2i → comfy (se selezionato) con FALLBACK a
    diffusers; inpaint/kontext/integra/altri → diffusers (il worker comfy copre per ora solo t2i)."""
    arch = (params.get("task") or params.get("arch") or "flux").lower()
    is_t2i = not params.get("image_path") and arch in ("flux", "flux2", "zimage", "ernie")
    # DEFAULT 'diffusers' (nativo): ComfyUI è il ramo legacy/rotto e il suo /free svuotava la VRAM del server
    # nativo ad ogni t2i → ricaricava lo STESSO modello. Col nativo il ModelManager lo tiene in cache (nessun
    # reload per la medesima pipeline). Comfy solo se scelto ESPLICITAMENTE in Impostazioni.
    if _COMFY_AVAILABLE and user_prefs.get_engine("image", "diffusers") == "comfy" and is_t2i:
        res = _comfy_t2i(params, timeout)
        if res.get("status") == "success":
            return res
        print(f"[img_router] comfy fallito → fallback diffusers: {str(res.get('error',''))[:200]}", flush=True)
    # Path DIFFUSERS (non-comfy, o PuLID/inpaint, o comfy fallito): libera la VRAM del figlio comfy residente
    # → il modello diffusers non contende (single-GPU). Il prossimo t2i comfy respawna (cambio-pipeline).
    try:
        _comfy_child.free()
    except Exception:                                          # noqa: BLE001
        pass
    return native_driver.run_and_wait("image", params, timeout=timeout)
