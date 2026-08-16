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
from pathlib import Path

VENV_PY = Path(r"E:\AI\Tools\inference_server_venv\Scripts\python.exe")
PROJECT = Path(__file__).resolve().parents[1]


def launch() -> tuple[bool, str]:
    """Lancia l'InferenceServer nativo (porta 8189) in un processo indipendente.
    Ritorna (avviato?, messaggio). NON attende che sia pronto (ci pensa il polling)."""
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
        subprocess.Popen([str(py), "-m", "inference_server.server"], cwd=str(PROJECT),
                         env=env, creationflags=flags, close_fds=True)
        return True, "Server nativo avviato (porta 8189) — pronto tra ~10-30s"
    except Exception as e:                                   # noqa: BLE001
        return False, f"avvio server nativo fallito: {e}"
