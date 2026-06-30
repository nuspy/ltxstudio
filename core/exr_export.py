"""Export EXR/HDR (#29) — salva frame HDR lineari in OpenEXR. Usato dal path HDR IC-LoRA (ltx-pipelines
hdr_ic_lora restituisce un tensore HDR lineare [f,h,w,c]; il salvataggio EXR è responsabilità del caller).

Backend: imageio (plugin freeimage) o cv2 (entrambi presenti nel venv). Nessun download.
"""
from __future__ import annotations

import os

# DEVE stare PRIMA di qualsiasi import di cv2 (OpenCV legge la flag all'import, non a runtime):
# senza, cv2.imwrite('.exr') fallisce silenziosamente ("OpenEXR codec is disabled").
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def save_exr_frames(frames, out_dir: str, prefix: str = "hdr") -> list[str]:
    """frames: array/tensore [f,h,w,c] float lineare → un .exr per frame. Ritorna i path."""
    import numpy as np
    arr = frames.detach().cpu().numpy() if hasattr(frames, "detach") else np.asarray(frames)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, fr in enumerate(arr):
        p = os.path.join(out_dir, f"{prefix}_{i:04d}.exr")
        if not _write_exr(fr.astype("float32"), p):
            break
        paths.append(p)
    return paths


def _write_exr(frame, path: str) -> bool:
    try:
        import cv2
        # cv2 scrive EXR se OPENCV_IO_ENABLE_OPENEXR=1 (impostato qui), BGR float32
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import numpy as np
        bgr = frame[..., ::-1] if frame.shape[-1] == 3 else frame
        return bool(cv2.imwrite(path, np.ascontiguousarray(bgr.astype("float32"))))
    except Exception:                                       # noqa: BLE001
        pass
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, frame.astype("float32"))
        return True
    except Exception:                                       # noqa: BLE001
        return False


def tonemap_reinhard(frame):
    """Tonemap semplice (Reinhard) HDR lineare → sdr 0..1 per anteprima."""
    import numpy as np
    x = np.asarray(frame, dtype="float32")
    return x / (1.0 + x)
