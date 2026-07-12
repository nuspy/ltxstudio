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
# insightface NON è nel venv GUI/API (Comfy) ma è nell'inference_server_venv → se manca localmente,
# il face-swap viene DELEGATO a un subprocess in quel venv (allineato al consolidamento #61).
_INF_PY = r"E:\AI\Tools\inference_server_venv\Scripts\python.exe"

_CACHE: dict = {}


def _has_insightface() -> bool:
    try:
        import insightface  # noqa: F401
        return True
    except Exception:                                   # noqa: BLE001
        return False


def _inf_has_insightface() -> bool:
    return os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(_INF_PY)),
                                      "Lib", "site-packages", "insightface"))


def _delegate(mode: str, target: str, source_face: str, out: str) -> str:
    """Esegue il face-swap nell'inference_server_venv (che ha insightface) via subprocess."""
    runner = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "scripts", "faceswap_run.py")
    r = subprocess.run([_INF_PY, "-u", runner, mode, target, source_face, out],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    line = next((l for l in (r.stdout or "").splitlines() if l.startswith("OK:")), "")
    if not line:
        # Estrai il MOTIVO vero (ultima riga d'errore), non la coda dei log: il tail 500 mostrava
        # il dump provider onnxruntime al posto di "Nessun volto rilevato" → diagnosi impossibile.
        _all = ((r.stderr or "") + "\n" + (r.stdout or "")).splitlines()
        _err = next((l for l in reversed(_all) if "Error" in l or "error" in l.lower()), "")
        raise RuntimeError("face-swap (subprocess inference_server_venv) fallito: "
                           + (_err.strip() or f"exit {r.returncode}"))
    return line[3:].strip()


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
    return _has_insightface() or _inf_has_insightface()   # locale, o delegabile all'inference_server_venv


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


def _restore_swapped_face(img):
    """Post-restore del volto swappato (GFPGAN/CodeFormer): inswapper lavora a 128×128 → il volto
    incollato è plasticoso/a bassa risoluzione (review utente 12/07). Il restauro riporta pelle e
    dettaglio alla risoluzione del target. Best-effort (senza gfpgan → immagine invariata).
    Opt-out: FACESWAP_RESTORE=0. Modello override: FACESWAP_RESTORE_MODEL."""
    if os.environ.get("FACESWAP_RESTORE", "1") != "1":
        return img
    try:
        from gfpgan import GFPGANer
        model = os.environ.get("FACESWAP_RESTORE_MODEL",
                               r"E:\AI\Models\facerestore_models\GFPGANv1.4.pth")
        if not os.path.exists(model):
            return img
        _is_cf = "codeformer" in os.path.basename(model).lower()
        try:
            restorer = GFPGANer(model_path=model, upscale=1,
                                arch="CodeFormer" if _is_cf else "clean", channel_multiplier=2)
        except Exception:                                # noqa: BLE001 — gfpgan vecchio senza arch CodeFormer
            restorer = GFPGANer(model_path=model, upscale=1, arch="clean", channel_multiplier=2)
        _, _, restored = restorer.enhance(img, has_aligned=False, only_center_face=False,
                                          paste_back=True)
        return restored if restored is not None else img
    except Exception as _e:                              # noqa: BLE001
        print(f"[faceswap] restauro volto saltato ({_e})", flush=True)
        return img


def _match_face_tone(swapped, original, faces):
    """COLOR-MATCH del volto swappato alla carnagione del TARGET (review 12/07: orecchie/collo
    restano originali → contrasto forte se la carnagione della sorgente differisce). Sposta media
    e deviazione LAB della regione volto swappata verso quelle della stessa regione PRE-swap
    (standard alla ReActor). Blending con falloff radiale: i bordi si fondono con la pelle attorno.
    Opt-out: FACESWAP_TONE_MATCH=0."""
    if os.environ.get("FACESWAP_TONE_MATCH", "1") != "1":
        return swapped
    import cv2
    import numpy as np
    out = swapped
    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        h, w = out.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        reg_s = out[y1:y2, x1:x2].astype(np.float32)
        reg_o = original[y1:y2, x1:x2].astype(np.float32)
        lab_s = cv2.cvtColor(reg_s / 255.0, cv2.COLOR_BGR2LAB)
        lab_o = cv2.cvtColor(reg_o / 255.0, cv2.COLOR_BGR2LAB)
        mean_s, std_s = lab_s.reshape(-1, 3).mean(0), lab_s.reshape(-1, 3).std(0) + 1e-6
        mean_o, std_o = lab_o.reshape(-1, 3).mean(0), lab_o.reshape(-1, 3).std(0) + 1e-6
        lab_m = (lab_s - mean_s) / std_s * std_o + mean_o
        reg_m = np.clip(cv2.cvtColor(lab_m, cv2.COLOR_LAB2BGR) * 255.0, 0, 255)
        # falloff radiale: pieno al centro del volto, 0 al bordo bbox → nessuna cucitura
        yy, xx = np.mgrid[0:y2 - y1, 0:x2 - x1].astype(np.float32)
        cy, cx = (y2 - y1) / 2.0, (x2 - x1) / 2.0
        d = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
        wgt = np.clip(1.2 - d, 0, 1)[..., None]
        out[y1:y2, x1:x2] = (reg_s * (1 - wgt) + reg_m * wgt).astype(np.uint8)
    return out


def swap_image(target_path: str, source_face_path: str, out_path: str) -> str:
    if not os.environ.get("FACESWAP_INPROC") and not _has_insightface():
        return _delegate("image", target_path, source_face_path, out_path)   # → inference_server_venv
    import cv2
    app, sw = _load()
    src = _source_face(source_face_path)
    img = cv2.imread(target_path)
    if img is None:
        raise FileNotFoundError(target_path)
    _orig = img.copy()
    _faces = app.get(img)
    for f in _faces:
        img = sw.get(img, f, src, paste_back=True)
    img = _match_face_tone(img, _orig, _faces)           # carnagione uniforme con orecchie/collo
    img = _restore_swapped_face(img)                     # anti-"plasticoso": restauro post-swap
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
    if not os.environ.get("FACESWAP_INPROC") and not _has_insightface():
        return _delegate("video", video_path, source_face_path, out_path)     # → inference_server_venv
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
