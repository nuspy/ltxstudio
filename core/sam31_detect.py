"""Ponte fra i backend di segmentazione e SAM 3.1 immagine (core/sam31_loader.py): testo -> riquadri / maschere, al posto di GroundingDINO.

Sam3Model esiste solo in transformers >= 5.17 (diffusers_next_venv): l'inference_server_venv (4.57) non lo ha. Per questo ogni chiamata lancia il worker
workers/sam31_detect_worker.py nel venv giusto e legge una riga `SAM31:{json}`. I punti non passano di qui: il modello immagine di SAM 3.1 e guidato da testo
e box, quindi clic e lazo restano di SAM2; la propagazione nel tempo resta di SAM2.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

WORKER_PY = r"E:\AI\Tools\diffusers_next_venv\Scripts\python.exe"
WORKER = str(Path(__file__).resolve().parents[1] / "workers" / "sam31_detect_worker.py")


def detect(image_path: str, text: str, device: str = "", threshold: float = 0.5, out_dir: str = "", python: str = "", runner=None) -> dict:
    """{'boxes': [[x0,y0,x1,y1]], 'scores': [...], 'masks': [png...], 'union': png|'' , 'presence', 'size', 'device'}; liste vuote se non trova nulla.
    `runner(cmd) -> (returncode, stdout, stderr)` iniettabile nei test."""
    out_dir = out_dir or tempfile.mkdtemp(prefix="sam31_")
    cmd = [python or WORKER_PY, "-u", WORKER, "--image", str(image_path), "--text", text.strip().rstrip("."), "--out-dir", out_dir,
           "--threshold", str(threshold)]
    if device:
        cmd += ["--device", device]
    if runner is None:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        rc, so, se = r.returncode, r.stdout or "", r.stderr or ""
    else:
        rc, so, se = runner(cmd)
    line = next((ln for ln in so.splitlines() if ln.startswith("SAM31:")), "")
    if not line:
        raise RuntimeError("SAM 3.1 non ha risposto" + (f" (codice {rc}): {(se or so).strip()[-400:]}" if (se or so).strip() else f" (codice {rc})"))
    return json.loads(line[len("SAM31:"):])


def boxes(image_path: str, text: str, device: str = "", threshold: float = 0.5, **kw) -> list[list[float]]:
    """Riquadri [x0, y0, x1, y1, punteggio] ordinati per punteggio decrescente (stesso formato di video_segment.detect_boxes)."""
    r = detect(image_path, text, device, threshold, **kw)
    out = [[float(v) for v in b[:4]] + [float(s)] for b, s in zip(r["boxes"], r["scores"])]
    return sorted(out, key=lambda b: -b[4])
