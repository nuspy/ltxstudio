"""Face-swap universale post-process (InsightFace inswapper) — su immagine o video, qualsiasi sorgente.

Sostituisce il/i volto/i di un'immagine o video con il volto di una foto sorgente. Indipendente dal
modello generativo (LTX, Wan, FLUX…): è un passaggio FINALE. Riusa InsightFace (già installato) +
antelopev2 (già su disco) per il detect, + inswapper_128.onnx per lo swap. Gira su onnxruntime
(GPU se disponibile, altrimenti CPU) → non occupa la VRAM dei modelli diffusion.

API:
  available() -> bool
  swap_image(target, source_face, out) -> str
  swap_video(video, source_face, out, progress=) -> str   (ri-muxa l'audio originale)
"""
from __future__ import annotations

import os
import subprocess

INSIGHTFACE_ROOT = r"E:\AI\Models\insightface"
INSWAPPER = os.path.join(INSIGHTFACE_ROOT, "inswapper_128.onnx")

_CACHE: dict = {}


def _add_torch_cuda_dlls():
    """Rende le DLL CUDA/cuDNN incluse in torch (torch/lib) trovabili da onnxruntime-gpu,
    così CUDAExecutionProvider si inizializza senza un'installazione CUDA/cuDNN di sistema."""
    try:
        import torch
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib):
            os.add_dll_directory(lib)
    except Exception:                                   # noqa: BLE001
        pass


def available() -> bool:
    if not os.path.exists(INSWAPPER):
        return False
    try:
        import insightface  # noqa: F401
        return True
    except Exception:                                   # noqa: BLE001
        return False


def _providers():
    try:
        import onnxruntime as ort
        av = ort.get_available_providers()
        return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in av] or ["CPUExecutionProvider"]
    except Exception:                                   # noqa: BLE001
        return ["CPUExecutionProvider"]


def _load():
    if "app" in _CACHE:
        return _CACHE["app"], _CACHE["sw"]
    _add_torch_cuda_dlls()                              # cuDNN/cublas di torch → onnxruntime CUDA
    import insightface
    from insightface.app import FaceAnalysis
    prov = _providers()
    app = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_ROOT, providers=prov)
    app.prepare(ctx_id=0, det_size=(640, 640))
    sw = insightface.model_zoo.get_model(INSWAPPER, providers=prov)
    _CACHE["app"], _CACHE["sw"] = app, sw
    return app, sw


def _source_face(path: str):
    import cv2
    app, _ = _load()
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    faces = app.get(img)
    if not faces:
        raise RuntimeError("Nessun volto rilevato nella foto sorgente.")
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def swap_image(target_path: str, source_face_path: str, out_path: str) -> str:
    import cv2
    app, sw = _load()
    src = _source_face(source_face_path)
    img = cv2.imread(target_path)
    if img is None:
        raise FileNotFoundError(target_path)
    for f in app.get(img):
        img = sw.get(img, f, src, paste_back=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, img)
    try:  # #107 firma invisibile di provenienza (default ON) — PRIMA del watermark visibile
        from core import stego
        if stego.enabled():
            stego.sign_file(out_path)
    except Exception:                                   # noqa: BLE001
        pass
    # #106 watermark obbligatorio (gated feature_matrix): il face-swap è un post-process FINALE e
    # questo è l'UNICO punto che scrive la foto consegnata (GUI tab Restauro + REST /image/faceswap)
    # → si marchia qui, una sola volta. RuntimeError (forced non applicabile) → errore del job.
    try:
        from core import watermark
        out_path = watermark.apply_to_file(out_path)
    except RuntimeError:
        raise
    except Exception:                                   # noqa: BLE001
        pass
    return out_path


def swap_video(video_path: str, source_face_path: str, out_path: str, progress=None) -> str:
    """Face-swap su ogni frame del video; ri-muxa l'audio originale."""
    import cv2
    app, sw = _load()
    src = _source_face(source_face_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".noaudio.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        for f in app.get(fr):
            fr = sw.get(fr, f, src, paste_back=True)
        vw.write(fr); i += 1
        if progress and i % 10 == 0:
            progress(i, total)
    cap.release(); vw.release()
    # ri-muxa l'audio originale (se presente)
    subprocess.run(["ffmpeg", "-y", "-i", tmp, "-i", video_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0?", "-shortest", out_path], check=False)
    try:
        os.remove(tmp)
    except OSError:
        pass
    try:  # #107 firma invisibile di provenienza (default ON) — PRIMA del watermark visibile
        from core import stego
        if stego.enabled():
            stego.sign_file(out_path)
    except Exception:                                   # noqa: BLE001
        pass
    # #106 watermark obbligatorio: punto finale del face-swap video (GUI + REST /video/faceswap).
    # Se l'input era un video generato (già marchiato da ltx2_infer) la banda è baked nei frame e
    # questa è comunque l'unica applicazione a livello di QUESTO job. RuntimeError → errore del job.
    try:
        from core import watermark
        out_path = watermark.apply_to_file(out_path)
    except RuntimeError:
        raise
    except Exception:                                   # noqa: BLE001
        pass
    return out_path
