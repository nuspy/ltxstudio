"""Driver del SERVER VIDEO PERSISTENTE LTX-2.3 (workers/ltx2_server.py, porta 8190, venv ltx2).

L'app PyQt chiama generate(); il server tiene la pipeline in RAM (offload CPU) → nessun
ricaricamento da disco tra le generazioni. Avvio automatico del server se non attivo.
ZERO ComfyUI. Pattern: submit asincrono + polling (come comfy_driver/native_driver).
"""
from __future__ import annotations

import os
import subprocess
import time
import uuid
from pathlib import Path

import requests

LTX2_PY = r"E:\AI\Tools\ltx2_venv\Scripts\python.exe"
REPO = str(Path(__file__).resolve().parent.parent)
BASE = "http://127.0.0.1:8190"
OUTPUT_DIR = r"E:\AI\ltxstudio\output"
_CREATE_NEW_CONSOLE = 0x00000010  # Windows


def is_up() -> bool:
    try:
        requests.get(f"{BASE}/system_stats", timeout=2)
        return True
    except Exception:                                       # noqa: BLE001
        return False


def installed() -> bool:
    return os.path.exists(LTX2_PY) and os.path.exists(os.path.join(REPO, "workers", "ltx2_server.py"))


def ensure_server(wait_s: int = 120) -> bool:
    """Avvia il server video persistente se non è attivo (subprocess in console separata)."""
    if is_up():
        return True
    if not installed():
        return False
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.Popen([LTX2_PY, "-m", "workers.ltx2_server"], cwd=REPO, env=env,
                     creationflags=_CREATE_NEW_CONSOLE)
    for _ in range(wait_s // 2):
        if is_up():
            return True
        time.sleep(2)
    return False


def generate(params: dict, timeout: int = 3600) -> dict:
    """Genera un video. params: prompt, image_path?(i2v), num_frames, fps, steps, guidance,
    seed, model, quant, offload, loras=[{name,strength}], output?. Ritorna {status, output:[mp4]}."""
    if not ensure_server():
        return {"status": "error", "error": "server video LTX-2 non avviato (venv ltx2 mancante?)"}
    p = dict(params)
    p.setdefault("output", os.path.join(OUTPUT_DIR, f"ltx2_{uuid.uuid4().hex[:8]}.mp4"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        jid = requests.post(f"{BASE}/submit", json={"params": p}, timeout=30).json()["prompt_id"]
    except Exception as e:                                  # noqa: BLE001
        return {"status": "error", "error": f"submit fallito: {e}"}
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = requests.get(f"{BASE}/history/{jid}", timeout=10).json()
        except Exception:                                  # noqa: BLE001
            time.sleep(3); continue
        if h.get("status") == "success":
            return {"status": "success", "output": h.get("output", [])}
        if h.get("status") == "error":
            return {"status": "error", "error": h.get("error", "")[-2000:]}
        time.sleep(3)
    return {"status": "error", "error": "timeout generazione video"}
