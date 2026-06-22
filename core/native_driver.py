"""Client dell'InferenceServer nativo (lato app PyQt) — stessa semantica di comfy_driver.

L'app NON importa torch/diffusers: parla col server nativo via HTTP (come fa con ComfyUI),
così la GUI resta leggera. Usato dall'astrazione `core/backend.py`.
"""
from __future__ import annotations

import json
import time
import urllib.request

BASE = "http://127.0.0.1:8189"


def _post(path: str, payload: dict, timeout: int = 30) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def _get(path: str, timeout: int = 15) -> dict:
    return json.load(urllib.request.urlopen(BASE + path, timeout=timeout))


def is_up() -> bool:
    try:
        _get("/system_stats", timeout=3); return True
    except Exception:                                   # noqa: BLE001
        return False


def submit(module: str, params: dict) -> str:
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
    while time.monotonic() - t0 < timeout:
        h = history(pid)
        if h.get("status") in ("success", "error"):
            return h
        time.sleep(poll)
    return {"status": "timeout"}


def system_stats() -> dict:
    return _get("/system_stats")
