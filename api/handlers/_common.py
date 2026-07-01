"""Helper condivisi dagli handler REST: esecuzione worker video (ltx2), job immagine (:8189),
risoluzione file, parsing LoRA.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import uuid

from api.config import OUTPUT_DIR

LTX2_PY = r"E:\AI\Tools\ltx2_venv\Scripts\python.exe"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LTX2_WORKER = os.path.join(REPO, "workers", "ltx2_infer.py")


def out_path(prefix: str, ext: str = "mp4") -> str:
    """Nuovo path di output univoco sotto OUTPUT_DIR."""
    return str(OUTPUT_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}")


def parse_loras(loras) -> list[dict]:
    """Normalizza `loras` (lista di {name,strength} o stringa JSON) → lista di dict."""
    if not loras:
        return []
    if isinstance(loras, str):
        try:
            loras = json.loads(loras)
        except Exception:                                    # noqa: BLE001
            return []
    out = []
    for lo in loras or []:
        if isinstance(lo, dict) and lo.get("name"):
            out.append({"name": lo["name"], "strength": float(lo.get("strength", 1.0))})
    return out


def run_image(params: dict, *, timeout: float = 900) -> list[str]:
    """Esegue un job immagine sull'InferenceServer nativo (:8189). Ritorna i path di output."""
    from core import native_driver
    if not native_driver.is_up():
        raise RuntimeError("Backend nativo immagini (:8189) non attivo. Avvia l'InferenceServer "
                           "(python -m inference_server.server) o la GUI.")
    module = params.pop("_module", "image")
    res = native_driver.run_and_wait(module, params, timeout=timeout)
    if res.get("status") != "success":
        raise RuntimeError(res.get("error", "job immagine fallito")[:800])
    return list(res.get("output", []))


def run_ltx2(params: dict, job=None, *, timeout: float = 7200) -> list[str]:
    """Esegue workers/ltx2_infer.py (venv ltx2) con i params dati. Ritorna [output_path]."""
    if not os.path.isfile(LTX2_PY):
        raise RuntimeError(f"venv ltx2 mancante: {LTX2_PY}")
    params.setdefault("output", out_path("api_video"))
    jf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(params, jf); jf.close()
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    proc = subprocess.Popen([LTX2_PY, "-u", LTX2_WORKER, jf.name], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    result = None
    t0 = time.time()
    for line in proc.stdout:
        line = line.rstrip()
        if job is not None and (line.startswith("[") or line.startswith("OK:")):
            job.log(line[:200])
        if line.startswith("OK:"):
            result = line[3:].strip()
        if time.time() - t0 > timeout:
            proc.kill(); raise RuntimeError("timeout generazione video")
    proc.wait()
    if result and os.path.exists(result):
        return [result]
    raise RuntimeError("LTX-2 nativo: nessun output prodotto")


def run_job_obj(job_obj) -> list[str]:
    """Esegue un oggetto Job (core.music_service.Job / *_service.make_job) e ritorna l'output.
    Il Job scrive un file noto; il chiamante passa `expected` via attributo, oppure legge .out."""
    job_obj.run()
    out = getattr(job_obj, "out", None) or getattr(job_obj, "output", None)
    if out and os.path.exists(str(out)):
        return [str(out)]
    return []
