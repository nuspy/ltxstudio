"""Avvio automatico del server NATIVO (InferenceServer, diffusers) se non già attivo.

Sostituisce `comfy_launcher` nella migrazione a zero-ComfyUI: l'app parla col backend nativo
su 127.0.0.1:8189. Se a startup non risponde, lanciamo `python -m inference_server.server`
(venv dedicato) in una finestra separata, con le ottimizzazioni attive (Sage attention auto +
FBCache). Il polling di MainWindow rileva quando è pronto.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

SERVER_PORT = 8189
VENV_PY = Path(r"E:\AI\Tools\inference_server_venv\Scripts\python.exe")
PROJECT = Path(__file__).resolve().parents[1]

# Un avvio normale apre la porta in 10-30 s (il 02/11 una volta 306 s, ma lavorando). Il blocco vero (self-pipe di asyncio perduta dal filtro
# di rete locale, vedi core/loopback.py) ha il processo FERMO e la porta mai aperta: si considera bloccato solo se, passato il tempo di sonda,
# la porta non e aperta E il processo non ha consumato CPU nell'ultima finestra.
STARTUP_PROBE_S = 60.0
IDLE_WINDOW_S = 20.0
IDLE_CPU_S = 1.0                  # meno di tanto CPU in una finestra = fermo
MAX_RESTARTS = 2
_server_process: subprocess.Popen | None = None      # il server avviato DA NOI: solo questo albero si chiude, mai altri processi


def launch() -> tuple[bool, str]:
    """Lancia l'InferenceServer nativo (porta 8189) in un processo indipendente.
    Ritorna (avviato?, messaggio). NON attende che sia pronto (ci pensa il polling)."""
    global _server_process
    if _server_process is not None and _server_process.poll() is None:
        return True, "Server nativo gia in avvio (stesso processo, nessun doppione)"
    py = VENV_PY if VENV_PY.exists() else Path(sys.executable)
    if not Path(py).exists():
        return False, f"venv nativo non trovato: {VENV_PY}"
    try:
        env = dict(os.environ)
        env.setdefault("NATIVE_ATTENTION", "auto")          # Sage se disponibile, else SDPA
        env.setdefault("NATIVE_FBCACHE", "1")               # First Block Cache ON: SCAGIONATO dalle macchie
        #   (A/B 11/07 a seed fisso: chiazze identiche con FBCache off, sage/sdpa, fp8/bf16, VAE fp32 e
        #   perfino con diffusers puro → è FLUX.1-dev sotto ~1MP, non la cache). Opt-out NATIVE_FBCACHE=0.
        # OFFLINE BLINDATO (weights-unification C): il server immagini era l'UNICO processo GPU
        # di lunga durata col network HF attivo (e perfino hf_transfer!) mentre il suo loader
        # dichiara "NON viene scaricato da HuggingFace". I download passano SOLO dal downloader
        # unico (tab Pesi / primo-uso con consenso). Override esplicito: NATIVE_HF_ONLINE=1.
        if env.get("NATIVE_HF_ONLINE") != "1":
            env.setdefault("HF_HUB_OFFLINE", "1")
        else:
            env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        # NB niente PYTORCH_CUDA_ALLOC_CONF=expandable_segments: su WINDOWS non è supportato (PyTorch lo
        # ignora con "expandable_segments not supported on this platform") → non è una via per la
        # frammentazione VRAM qui. L'accumulo di pipeline va evitato a monte, liberando (/free) prima di
        # cambiare arch/checkpoint. (landing 12/07)
        # Profilo hardware auto: calibra VRAM/RAM (offload, margine, risoluzione max) per la GPU corrente
        try:
            from core import hw_profiles
            prof = hw_profiles.detect_profile()
            for k, v in prof["env"].items():
                env.setdefault(k, v)                        # non sovrascrive override manuali utente
            print(f"[native] profilo HW: {prof['name']} (VRAM {prof['vram_gb']}GB, RAM {prof['ram_gb']}GB)", flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f"[native] profilo HW non applicato: {e}", flush=True)
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
        _server_process = subprocess.Popen([str(py), "-m", "inference_server.server"], cwd=str(PROJECT),
                                           env=env, creationflags=flags, close_fds=True)
        return True, "Server nativo avviato (porta 8189) — pronto tra ~10-30s"
    except Exception as e:                                   # noqa: BLE001  l'errore arriva al chiamante nel messaggio
        return False, f"avvio server nativo fallito: {e}"


def _process_tree(root) -> list:
    """Il processo `root` e tutti i suoi discendenti (il venv lancia il Python di base come figlio)."""
    import psutil
    try:
        parent = psutil.Process(root.pid)
        return [parent] + parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def _tree_cpu_seconds(root) -> float:
    import psutil
    total = 0.0
    for proc in _process_tree(root):
        try:
            times = proc.cpu_times()
            total += times.user + times.system
        except psutil.Error:
            continue                                         # processo sparito durante la lettura: non conta
    return total


def _is_listening(root, port: int = SERVER_PORT) -> bool:
    import psutil
    for proc in _process_tree(root):
        try:
            connections = proc.net_connections(kind="tcp") if hasattr(proc, "net_connections") else proc.connections(kind="tcp")
        except psutil.Error:
            continue
        if any(c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port for c in connections):
            return True
    return False


def kill_server_tree(log: Callable[[str], None] = print) -> int:
    """Chiude il server avviato da `launch()` e i suoi figli. Ritorna quanti processi ha chiuso. Non tocca nessun altro processo."""
    global _server_process
    import psutil
    root, _server_process = _server_process, None
    if root is None:
        return 0
    tree = _process_tree(root)
    for proc in tree:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            continue
        except psutil.Error as errore:
            log(f"[nativo] kill_server_tree: pid {proc.pid} non chiuso: {errore!r}")
    psutil.wait_procs(tree, timeout=10)
    return len(tree)


def is_startup_stuck(root, samples: list, now: float) -> bool:
    """Bloccato = porta non aperta e quasi nessuna CPU consumata nell'ultima finestra. `samples` = [(tempo, cpu_cumulata)] del processo."""
    if _is_listening(root):
        return False
    old = [cpu for moment, cpu in samples if now - moment >= IDLE_WINDOW_S]
    if not old:
        return False                                         # finestra non ancora completa
    return samples[-1][1] - old[-1] < IDLE_CPU_S


def supervise_startup(is_up: Callable[[], bool], *, log: Callable[[str], None] = print, probe_seconds: float = STARTUP_PROBE_S,
                      max_restarts: int = MAX_RESTARTS, total_timeout: float = 600.0, poll_seconds: float = 1.0) -> bool:
    """Attende che il server avviato da noi risponda. Se dopo `probe_seconds` e bloccato (vedi `is_startup_stuck`), chiude il SUO albero di
    processi e lo rilancia, al piu `max_restarts` volte. Ritorna True appena `is_up()` e vero. Bloccante: chiamarla da un thread."""
    start = time.monotonic()
    attempt_start = start
    restarts = 0
    samples: list = []
    while time.monotonic() - start < total_timeout:
        if is_up():
            return True
        root = _server_process
        now = time.monotonic()
        died = root is not None and root.poll() is not None
        stuck = False
        if root is not None and not died:
            samples.append((now, _tree_cpu_seconds(root)))
            stuck = now - attempt_start >= probe_seconds and is_startup_stuck(root, samples, now)
        if died or stuck:
            if restarts >= max_restarts:
                log(f"[nativo] il server non parte dopo {restarts} riavvii: mi fermo")
                return False
            why = "terminato da solo" if died else f"bloccato da {int(now - attempt_start)}s (porta {SERVER_PORT} mai aperta, nessuna CPU)"
            killed = kill_server_tree(log)
            restarts += 1
            log(f"[nativo] server {why}: chiuso l'albero ({killed} processi) e RILANCIO {restarts}/{max_restarts}")
            started, message = launch()
            if not started:
                log(f"[nativo] rilancio fallito: {message}")
                return False
            attempt_start, samples = time.monotonic(), []
        time.sleep(poll_seconds)
    log(f"[nativo] il server non e pronto entro {int(total_timeout)}s")
    return False


def launch_supervised_in_background(is_up: Callable[[], bool], log: Callable[[str], None] = print) -> tuple[bool, str]:
    """`launch()` + un thread che sorveglia l'avvio e rilancia se si blocca. Non attende: per l'avvio automatico dell'app."""
    started, message = launch()
    if started:
        threading.Thread(target=supervise_startup, args=(is_up,), kwargs={"log": log}, name="native-startup-watch", daemon=True).start()
    return started, message
