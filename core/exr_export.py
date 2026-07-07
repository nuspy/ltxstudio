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


def video_to_exr(video: str, out_dir: str | None = None, *, linear: bool = True,
                 progress=None) -> str:
    """Esporta TUTTI i frame di un video in OpenEXR float32 (pipeline VFX: Nuke/Resolve/Blender).
    linear=True: de-gamma BT.709 → luce LINEARE (zscale transfer=linear, lo standard per il compositing);
    linear=False: i valori restano display-referred (gamma). Per un video HDR VERO, generarlo prima con
    la IC-LoRA HDR (LTX-2\\ltx-2.3-22b-ic-lora-hdr) nel tab Video, poi esportare da qui.
    Ritorna la cartella coi .exr. Richiede l'encoder exr di ffmpeg (verificato: ffmpeg 8.x full build)."""
    import subprocess
    if not os.path.isfile(video):
        raise FileNotFoundError(video)
    out_dir = out_dir or os.path.splitext(video)[0] + "_exr"
    os.makedirs(out_dir, exist_ok=True)
    vf = ["-vf", "zscale=transfer=linear:npl=100"] if linear else []
    if progress:
        progress(f"[exr] estrazione frame EXR ({'lineare' if linear else 'gamma'})…")
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", video, *vf,
                        "-pix_fmt", "gbrpf32le", "-c:v", "exr",
                        os.path.join(out_dir, "frame_%05d.exr"), "-y"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        # fallback: senza zscale (build ffmpeg senza filtro) → EXR comunque float, non linearizzato
        r2 = subprocess.run(["ffmpeg", "-v", "error", "-i", video,
                             "-pix_fmt", "gbrpf32le", "-c:v", "exr",
                             os.path.join(out_dir, "frame_%05d.exr"), "-y"],
                            capture_output=True, text=True)
        if r2.returncode != 0:
            raise RuntimeError(f"export EXR fallito: {(r.stderr or r2.stderr or '')[-400:]}")
        if progress:
            progress("[exr] nota: zscale assente → EXR non linearizzati (gamma display)")
    n = len([f for f in os.listdir(out_dir) if f.endswith(".exr")])
    if progress:
        progress(f"[exr] {n} frame EXR in {out_dir}")
    if not n:
        raise RuntimeError("export EXR: nessun frame prodotto")
    return out_dir
