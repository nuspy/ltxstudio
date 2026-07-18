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
_ARCH_RUNNER = {
    "krea2": (_LINGBOT_PY, str(REPO / "workers" / "krea2_infer.py")),
    # editing per istruzione (single/multi) — stesso worker, ramo edit scelto da params['arch']
    "krea2_edit": (_LINGBOT_PY, str(REPO / "workers" / "krea2_infer.py")),
}


def handles(arch: str | None) -> bool:
    """True se questa arch immagine ha un worker dedicato (bypassa l'inference_server)."""
    return (arch or "").lower() in _ARCH_RUNNER


def run(params: dict, *, timeout: int | None = 1800) -> dict:
    """Esegue un job immagine col worker dedicato. Ritorna {status, output:[png]} come il server."""
    arch = (params.get("arch") or "").lower()
    py, worker = _ARCH_RUNNER.get(arch, (None, None))
    if not py:
        return {"status": "error", "error": f"arch immagine senza worker dedicato: {arch}"}
    out = params.get("output") or os.path.join(OUTPUT_DIR, f"krea_{arch}_{os.getpid()}.png")
    params = dict(params, output=out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if not os.path.exists(py):
        return {"status": "error",
                "error": f"venv per '{arch}' mancante: {py}. Esegui scripts/install_krea.py"}
    # libera la VRAM delle altre categorie (server immagini/video) prima del modello pesante
    try:
        from core import mem_coordinator as _mc
        _mc.free_for("image", timeout=20)
    except Exception:                                        # noqa: BLE001
        pass
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
