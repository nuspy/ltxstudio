"""Post-process VIDEO nativo — sostituisce ComfyUI/VHS per l'enhance video.

upscale per-frame (ESRGAN/spandrel, riusa image_upscale): estrae i frame con ffmpeg,
applica il modello a ogni frame, ricompone il video preservando FPS e AUDIO.
Indipendente dal motore di generazione (lavora su un video esistente) → testabile senza Gemma.
RIFE (FPS) / SeedVR2 / face-restore: handler separati (richiedono i loro pesi).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

VRAM_MB = 4000


def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or r"C:\Users\atain\scoop\shims\ffmpeg.exe"


def _ffprobe() -> str:
    return shutil.which("ffprobe") or r"C:\Users\atain\scoop\shims\ffprobe.exe"


def _fps(video: str) -> float:
    try:
        out = subprocess.run([_ffprobe(), "-v", "0", "-of", "csv=p=0", "-select_streams", "v:0",
                              "-show_entries", "stream=r_frame_rate", video],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        n, d = out.split("/") if "/" in out else (out, "1")
        return float(n) / float(d) if float(d) else 24.0
    except Exception:                                       # noqa: BLE001
        return 24.0


def run(params: dict, mm, output_dir: str) -> dict:
    """params: image_path (= video), model (ESRGAN), target_scale, seed."""
    import numpy as np
    import torch
    from PIL import Image
    from inference_server.backends.image_upscale import _resolve, _load, _upscale_tiled, _run_once

    video = params["image_path"]
    if not os.path.exists(video):
        return {"status": "error", "error": f"video non trovato: {video}"}
    mp = _resolve(params["model"])
    if not os.path.exists(mp):
        return {"status": "error", "error": f"modello upscale non trovato: {mp}"}
    key = f"upscale:{mp}"
    mm.register(key, lambda: _load(mp))
    model = mm.ensure(key, need_mb=VRAM_MB)
    fps = _fps(video)
    ff = _ffmpeg()

    with tempfile.TemporaryDirectory() as td:
        ind = os.path.join(td, "in"); outd = os.path.join(td, "out")
        os.makedirs(ind); os.makedirs(outd)
        # 1) estrai i frame
        subprocess.run([ff, "-y", "-i", video, os.path.join(ind, "f_%06d.png")],
                       capture_output=True, timeout=600)
        frames = sorted(Path(ind).glob("f_*.png"))
        if not frames:
            return {"status": "error", "error": "estrazione frame fallita"}
        # 2) upscale per-frame
        for fp in frames:
            img = Image.open(fp).convert("RGB")
            t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            if torch.cuda.is_available():
                t = t.cuda()
            out = _upscale_tiled(model, t) if max(img.size) > 1024 else _run_once(model, t)
            arr = (out.clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
            Image.fromarray(arr).save(os.path.join(outd, fp.name))
        # 3) ricomponi video + riporta l'audio originale
        os.makedirs(output_dir, exist_ok=True)
        path = str(Path(output_dir) / f"native_videoup_{params.get('seed', 0)}.mp4")
        subprocess.run([ff, "-y", "-framerate", f"{fps}", "-i", os.path.join(outd, "f_%06d.png"),
                        "-i", video, "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", path],
                       capture_output=True, timeout=900)
    return {"status": "success", "output": [path]}
