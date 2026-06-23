"""Launcher del motore VIDEO LTX-2.3 (pacchetto UFFICIALE Lightricks) — subprocess in venv isolato.

L'app PyQt chiama `generate(params)`; questo esegue `workers/ltx2_infer.py` nel venv dedicato
`E:\\AI\\Tools\\ltx2_venv` (torch 2.9.1, ltx-core+ltx-pipelines) e ritorna il path del video.
Stesso pattern dei moduli standalone (music_service / world_service). ZERO ComfyUI.

params accettati (→ workers/ltx2_infer.py):
  prompt, negative, seed, height, width, num_frames, fps, steps, guidance,
  image_path (i2v), loras=[{name,strength}], quant ("fp8_scaled_mm"|"fp8_cast"|"none"),
  output (opz., altrimenti auto)
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

LTX2_PY = r"E:\AI\Tools\ltx2_venv\Scripts\python.exe"
WORKER = str(Path(__file__).resolve().parent.parent / "workers" / "ltx2_infer.py")
OUTPUT_DIR = r"E:\AI\ltxstudio\output"


def is_available() -> bool:
    return os.path.exists(LTX2_PY) and os.path.exists(WORKER)


def generate(params: dict, timeout: int = 3600) -> dict:
    """Esegue il worker nel venv ltx2. Ritorna {status, output:[mp4], log} o {status:error, error}."""
    if not is_available():
        return {"status": "error",
                "error": f"motore video LTX-2 non installato (atteso {LTX2_PY})"}
    p = dict(params)
    p.setdefault("output", os.path.join(OUTPUT_DIR, f"ltx2_{uuid.uuid4().hex[:8]}.mp4"))
    p.setdefault("quant", "fp8_scaled_mm")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pf = os.path.join(tempfile.gettempdir(), f"ltx2_params_{uuid.uuid4().hex[:8]}.json")
    with open(pf, "w", encoding="utf-8") as fh:
        json.dump(p, fh)
    try:
        r = subprocess.run([LTX2_PY, WORKER, pf], capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "") + (r.stderr or "")
        for line in out.splitlines():
            if line.startswith("OK:"):
                return {"status": "success", "output": [line[3:].strip()], "log": out[-2000:]}
        return {"status": "error", "error": out[-2500:]}
    finally:
        try:
            os.remove(pf)
        except OSError:
            pass
