"""#58 Mask tracking VIDEO con SAM2 (video object segmentation) — la maschera SEGUE l'oggetto.

Usa transformers Sam2VideoModel/Sam2VideoProcessor (già nel venv, pesi sam2_hiera_large) + opz.
GroundingDINO per selezione a parole. Niente ComfyUI, niente nuove dipendenze.

Flusso:
  1. seleziona l'oggetto sul frame-0: punto (x,y) | box (x0,y0,x1,y1) | testo (GroundingDINO→box)
  2. SAM2 propaga la maschera su TUTTI i frame (propagate_in_video_iterator) → segue l'oggetto
  3. ritorna le maschere per-frame (uint8 0/255), pronte per:
     - inpaint mask-embedded (magenta) → ltx23_inpaint_masked
     - edit-anything localizzato

API:
  track_object(frames_dir|video, *, point=None, box=None, text="", dilate=0, device="cuda") -> list[mask_png]
"""
from __future__ import annotations

import os
from pathlib import Path

SEG_DIR = r"E:\AI\Models\Segmentation"
SAM2_DIR = os.path.join(SEG_DIR, "hf", "sam2-hiera-large")
GDINO_DIR = os.path.join(SEG_DIR, "hf", "grounding-dino-base")


def _frames_of(video: str, out_dir: str, fps: int = 0) -> tuple[list[str], float]:
    """Estrae i frame di un video in PNG (ffmpeg). Ritorna (lista_png, fps)."""
    import subprocess
    os.makedirs(out_dir, exist_ok=True)
    rate = []
    if fps:
        rate = ["-r", str(fps)]
    subprocess.run(["ffmpeg", "-v", "error", "-i", video, *rate,
                    os.path.join(out_dir, "%05d.png"), "-y"], check=True)
    frames = sorted(str(p) for p in Path(out_dir).glob("*.png"))
    # fps reale
    try:
        import json
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=r_frame_rate", "-of", "json", video], capture_output=True, text=True)
        rf = json.loads(r.stdout)["streams"][0]["r_frame_rate"]
        num, den = rf.split("/")
        real_fps = float(num) / float(den or 1)
    except Exception:                                       # noqa: BLE001
        real_fps = fps or 25.0
    return frames, real_fps


def _gdino_box(frame0_png: str, text: str, device: str):
    """GroundingDINO: testo → box [x0,y0,x1,y1] sul frame-0 (box col punteggio più alto)."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    proc = AutoProcessor.from_pretrained(GDINO_DIR)
    model = GroundingDinoForObjectDetection.from_pretrained(GDINO_DIR).to(device).eval()
    img = Image.open(frame0_png).convert("RGB")
    q = text if text.strip().endswith(".") else text.strip() + "."
    inp = proc(images=img, text=q, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp)
    res = proc.post_process_grounded_object_detection(out, inp["input_ids"],
                                                      threshold=0.35, target_sizes=[img.size[::-1]])[0]
    if len(res["boxes"]) == 0:
        return None
    best = int(res["scores"].argmax())
    return [float(x) for x in res["boxes"][best].tolist()]


def track_object(video: str, *, point=None, box=None, text: str = "", dilate: int = 0,
                 device: str = "cuda", work_dir: str = "") -> dict:
    """Traccia un oggetto nel video e ritorna le maschere per-frame.
    Selezione frame-0: `point` (x,y) | `box` (x0,y0,x1,y1) | `text` (GroundingDINO).
    Ritorna {frames:[png...], masks:[png...], fps:float}."""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    work_dir = work_dir or str(Path(video).with_suffix("")) + "_track"
    fr_dir = os.path.join(work_dir, "frames")
    mk_dir = os.path.join(work_dir, "masks")
    os.makedirs(mk_dir, exist_ok=True)
    frames, fps = _frames_of(video, fr_dir)
    if not frames:
        raise RuntimeError("Nessun frame estratto dal video.")

    if text and not (point or box):                        # selezione a parole sul frame-0
        box = _gdino_box(frames[0], text, device)
        if box is None:
            raise RuntimeError(f"GroundingDINO non ha trovato '{text}' nel primo frame.")

    proc = Sam2VideoProcessor.from_pretrained(SAM2_DIR)
    model = Sam2VideoModel.from_pretrained(SAM2_DIR).to(device).eval()
    # i frame vanno passati come ARRAY caricati (lista di path → il processor li decoderebbe come
    # video con torchvision.io.read_video, assente in questa build). Carico in np.uint8 [H,W,3].
    frames_np = [np.asarray(Image.open(f).convert("RGB")) for f in frames]
    sess = proc.init_video_session(video=frames_np, inference_device=device)
    add_kw = {"obj_ids": 1}
    if box is not None:
        add_kw["input_boxes"] = [[[float(b) for b in box]]]
    else:
        px, py = (point or (frames and 0))
        add_kw["input_points"] = [[[[float(px), float(py)]]]]
        add_kw["input_labels"] = [[[1]]]
    proc.add_inputs_to_inference_session(sess, frame_idx=0, **add_kw)

    H, W = frames_np[0].shape[:2]
    out_masks = []
    with torch.no_grad():
        for sam_out in model.propagate_in_video_iterator(sess, start_frame_idx=0):
            masks = proc.post_process_masks([sam_out.pred_masks], original_sizes=[(H, W)], binarize=True)[0]
            m = masks[0].squeeze().cpu().numpy().astype(np.uint8) * 255
            if dilate > 0:
                import cv2
                m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
            idx = int(getattr(sam_out, "frame_idx", len(out_masks)))
            mp = os.path.join(mk_dir, f"{idx:05d}.png")
            Image.fromarray(m).save(mp)
            out_masks.append(mp)
    out_masks.sort()
    return {"frames": frames, "masks": out_masks, "fps": fps}


def build_masked_guide(frames: list[str], masks: list[str], out_video: str, *, fps: float = 25.0,
                       mask_color=(255, 0, 255), feather: int = 0) -> str:
    """#59 Costruisce il GUIDE video per ltx23_inpaint_masked: i frame originali con l'area mascherata
    (SAM2-tracciata) dipinta di MAGENTA (255,0,255), come richiede la LoRA inpaint (mask embedded nel
    guide, non canale separato). Ritorna il path del video guide (mp4)."""
    import subprocess
    import numpy as np
    from PIL import Image
    tmp = str(Path(out_video).with_suffix("")) + "_gframes"
    os.makedirs(tmp, exist_ok=True)
    col = np.array(mask_color, np.uint8)
    n = min(len(frames), len(masks))
    for i in range(n):
        img = np.asarray(Image.open(frames[i]).convert("RGB")).copy()
        m = np.asarray(Image.open(masks[i]).convert("L"))
        sel = m > 127
        if feather > 0:
            import cv2
            a = (cv2.GaussianBlur(sel.astype(np.float32), (0, 0), feather))[..., None]
            img = (img * (1 - a) + col[None, None, :] * a).astype(np.uint8)
        else:
            img[sel] = col
        Image.fromarray(img).save(os.path.join(tmp, f"{i:05d}.png"))
    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", out_video, "-y"], check=True)
    return out_video
