"""Handler UPSCALE nativo (ESRGAN/super-risoluzione via spandrel) — sostituisce ComfyUI.

spandrel carica i modelli .pth (RealESRGAN, UltraSharp, ecc.) e li esegue in-process.
Tiling automatico per immagini grandi (evita OOM). Niente nodi ComfyUI.
"""
from __future__ import annotations

import os
from pathlib import Path

UPSCALE_DIRS = [r"E:\AI\Models\Upscale", r"E:\AI\Models\upscale_models"]   # zero-comfy: solo E:\AI\Models
VRAM_MB = 4000


def _resolve(name: str) -> str:
    if os.path.isabs(name):
        return name
    for d in UPSCALE_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(UPSCALE_DIRS[0], name)


def _load(model_path: str, _arch: str = "upscale"):
    import torch
    from spandrel import ModelLoader
    desc = ModelLoader().load_from_file(model_path)
    model = desc.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, VRAM_MB


def _upscale_tiled(model, t, tile: int = 512, overlap: int = 32):
    """Upscale a tile per limitare la VRAM. t: (1,3,H,W) in [0,1] su device del modello."""
    import torch
    import torch.nn.functional as F
    scale = getattr(model, "scale", 4) or 4
    _, _, h, w = t.shape
    out = torch.zeros((1, 3, h * scale, w * scale), device=t.device, dtype=t.dtype)
    wsum = torch.zeros_like(out)
    for y in range(0, h, tile - overlap):
        for x in range(0, w, tile - overlap):
            y2, x2 = min(y + tile, h), min(x + tile, w)
            with torch.no_grad():
                ot = model(t[:, :, y:y2, x:x2])
            oy, ox = y * scale, x * scale
            out[:, :, oy:oy + ot.shape[2], ox:ox + ot.shape[3]] += ot
            wsum[:, :, oy:oy + ot.shape[2], ox:ox + ot.shape[3]] += 1
    return out / wsum.clamp(min=1)


def run(params: dict, mm, output_dir: str) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    mp = _resolve(params["model"])
    if not os.path.exists(mp):
        return {"status": "error", "error": f"modello upscale non trovato: {mp}"}
    key = f"upscale:{mp}"
    mm.register(key, lambda: _load(mp))
    model = mm.ensure(key, need_mb=VRAM_MB)

    img = Image.open(params["image_path"]).convert("RGB")
    t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    if torch.cuda.is_available():
        t = t.cuda()
    big = max(img.size) > 1024
    out = _upscale_tiled(model, t) if big else _run_once(model, t)
    arr = (out.clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
    res = Image.fromarray(arr)
    # ridimensiona al fattore target richiesto (l'ESRGAN scala ×N fisso)
    tgt = params.get("target_scale")
    if tgt:
        res = res.resize((int(img.width * tgt), int(img.height * tgt)), Image.LANCZOS)
    os.makedirs(output_dir, exist_ok=True)
    path = str(Path(output_dir) / f"native_upscale_{params.get('seed', 0)}.png")
    res.save(path)
    try:                                                     # #107 firma invisibile di provenienza (default ON) — PRIMA del watermark
        from core import stego
        if stego.enabled():
            stego.sign_file(path)
    except Exception:                                        # noqa: BLE001
        pass
    try:                                                     # #106 watermark obbligatorio (gated feature_matrix)
        from core import watermark
        path = watermark.apply_to_file(path)                 # forced → fail-closed (rimuove se non applicabile)
    except RuntimeError:
        return {"status": "error", "error": "watermark obbligatorio non applicabile"}
    except Exception:                                        # noqa: BLE001
        pass
    return {"status": "success", "output": [path]}


def _run_once(model, t):
    import torch
    with torch.no_grad():
        return model(t)
