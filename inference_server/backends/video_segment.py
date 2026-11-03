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
import sys as _sys
from pathlib import Path

# Lanciato come SCRIPT (`python video_segment.py ...`, tracking SAM2 di Modifica Video AI e «Trova persone») la radice del repo non e nel path: senza,
# `from core import ...` moriva con ModuleNotFoundError e il tracking falliva (giro C del 03/11, bug introdotto dal blocco SAM 3.1).
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

SEG_DIR = r"E:\AI\Models\Segmentation"
# FASE F weights-unification: dir DAL registro unico, fallback ai letterali storici.
try:
    from core import model_paths as _mp
    SAM2_DIR = str(_mp.path_of("sam2-hiera-large") or os.path.join(SEG_DIR, "hf", "sam2-hiera-large"))
    GDINO_DIR = str(_mp.path_of("grounding-dino-base") or os.path.join(SEG_DIR, "hf", "grounding-dino-base"))
except Exception:                                        # noqa: BLE001
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


def _gdino_box(frame0_png: str, text: str, device: str, engine: str | None = None):
    """GroundingDINO: testo → box [x0,y0,x1,y1] sul frame-0 (box col punteggio più alto). Con `engine`=SAM 3.1 il riquadro viene da SAM 3.1 (da testo)."""
    if engine == "sam31":
        from core import sam31_detect as _D
        bx = _D.boxes(frame0_png, text.strip().rstrip("."), device, 0.5)
        return bx[0][:4] if bx else None
    import torch
    from PIL import Image
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    # weights-unification C: errore CHIARO se i pesi mancano (prima: HFValidationError criptica).
    if not os.path.isdir(GDINO_DIR) or not os.path.isdir(SAM2_DIR):
        raise RuntimeError(
            f"Pesi segmentazione assenti ({GDINO_DIR} / {SAM2_DIR}): scaricali dal tab "
            "«Pesi modelli» (voci grounding-dino-base + sam2-hiera-large, ~1.8GB).")
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
                 device: str = "cuda", work_dir: str = "", engine: str | None = None) -> dict:
    """Traccia un oggetto nel video e ritorna le maschere per-frame.
    Selezione frame-0: `point` (x,y) | `box` (x0,y0,x1,y1) | `text` (GroundingDINO).
    `engine`: motore del menu «Segmentazione» (None = quello salvato); SAM 3.1 non utilizzabile -> RuntimeError chiara, mai ripiego silenzioso su SAM2.
    Ritorna {frames:[png...], masks:[png...], fps:float}."""
    from core import segmentation_engine as _SE
    _eng, _err = _SE.resolve_for_job(engine)
    if _err:
        raise RuntimeError(_err)
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
        box = _gdino_box(frames[0], text, device, _eng)
        if box is None:
            raise RuntimeError(f"{'SAM 3.1' if _eng == _SE.SAM31 else 'GroundingDINO'} non ha trovato '{text}' nel primo frame.")

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


def detect_boxes(frame_png: str, text: str, device: str, threshold: float = 0.35, engine: str | None = None) -> list[list[float]]:
    """GroundingDINO: testo -> TUTTI i riquadri [x0,y0,x1,y1,punteggio] sul frame (multi-box; `_gdino_box` ne da uno solo).
    Con `engine`=SAM 3.1 i riquadri vengono da SAM 3.1 (da testo, soglia 0,5)."""
    if engine == "sam31":
        from core import sam31_detect as _D
        return _D.boxes(frame_png, text.strip().rstrip("."), device, 0.5)
    import torch
    from PIL import Image
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    if not os.path.isdir(GDINO_DIR) or not os.path.isdir(SAM2_DIR):
        raise RuntimeError(f"Pesi segmentazione assenti ({GDINO_DIR} / {SAM2_DIR}): scaricali dal tab «Pesi modelli» (grounding-dino-base + sam2-hiera-large).")
    proc = AutoProcessor.from_pretrained(GDINO_DIR)
    model = GroundingDinoForObjectDetection.from_pretrained(GDINO_DIR).to(device).eval()
    img = Image.open(frame_png).convert("RGB")
    q = text if text.strip().endswith(".") else text.strip() + "."
    inp = proc(images=img, text=q, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp)
    res = proc.post_process_grounded_object_detection(out, inp["input_ids"], threshold=threshold, target_sizes=[img.size[::-1]])[0]
    boxes = [[float(v) for v in b.tolist()] + [float(sc)] for b, sc in zip(res["boxes"], res["scores"])]
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:                                      # noqa: BLE001
        pass
    return boxes


def find_people(video: str, work_dir: str, *, text: str = "person", max_people: int = 4, device: str = "cuda", engine: str | None = None) -> list:
    """«Trova persone» del tab «Scambia»: GroundingDINO multi-box sul frame-0 (senza doppioni) + una traccia SAM2 per riquadro. Scrive in `work_dir`:
    frames/, track_<k>/masks/*.png, thumb_<k>.jpg e tracks.json. Ritorna la lista di core.swap_project.Track (ordinata: la piu grande per prima).
    SAM 3.1 non utilizzabile -> RuntimeError chiara (mai ripiego silenzioso su SAM2)."""
    from core import segmentation_engine as _SE
    from core import swap_project as SP
    _eng, _err = _SE.resolve_for_job(engine)
    if _err:
        raise RuntimeError(_err)
    import numpy as np
    import torch
    from PIL import Image
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    os.makedirs(work_dir, exist_ok=True)
    frames, _fps = _frames_of(video, os.path.join(work_dir, "frames"))
    if not frames:
        raise RuntimeError("Nessun frame estratto dal video.")
    frames_np = [np.asarray(Image.open(f).convert("RGB")) for f in frames]
    H, W = frames_np[0].shape[:2]
    sproc = Sam2VideoProcessor.from_pretrained(SAM2_DIR)
    smodel = Sam2VideoModel.from_pretrained(SAM2_DIR).to(device).eval()

    def _detect(_v):
        return detect_boxes(frames[0], text, device, engine=_eng)

    def _track(_v, box, mask_dir):
        os.makedirs(mask_dir, exist_ok=True)
        sess = sproc.init_video_session(video=frames_np, inference_device=device)
        sproc.add_inputs_to_inference_session(sess, frame_idx=0, obj_ids=1, input_boxes=[[[float(b) for b in box[:4]]]])
        n, area = 0, 0.0
        with torch.no_grad():
            for out in smodel.propagate_in_video_iterator(sess, start_frame_idx=0):
                m = sproc.post_process_masks([out.pred_masks], original_sizes=[(H, W)], binarize=True)[0][0].squeeze().cpu().numpy().astype(np.uint8)
                idx = int(getattr(out, "frame_idx", n))
                Image.fromarray(m * 255).save(os.path.join(mask_dir, f"{idx:05d}.png"))
                area += float(m.mean())
                n += 1
        return {"frames": n, "area": area / max(1, n)}

    def _thumb(t):
        md = t.mask_dir
        names = sorted(os.listdir(md))
        best, best_a = None, -1.0
        for f in names[::max(1, len(names) // 24)]:
            a = float((np.asarray(Image.open(os.path.join(md, f))) > 127).mean())
            if a > best_a:
                best, best_a = f, a
        if best is None:
            return ""
        m = np.asarray(Image.open(os.path.join(md, best))) > 127
        ys, xs = np.where(m)
        if len(xs) == 0:
            return ""
        i = min(int(best.split(".")[0]), len(frames_np) - 1)
        crop = Image.fromarray(frames_np[i]).crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
        crop.thumbnail((256, 256))
        tp = os.path.join(work_dir, f"thumb_{t.id}.jpg")
        crop.save(tp, quality=88)
        return tp

    tracks = SP.find_people(video, work_dir, detect_boxes=_detect, track_box=_track, make_thumb=_thumb, max_people=max_people,
                            log=lambda m: print(m, flush=True))
    SP.save_tracks(tracks, os.path.join(work_dir, "tracks.json"))
    return tracks


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


def composite_inpaint(original: str, edited: str, masks: list[str], out_video: str, *, fps: float = 25.0,
                      feather: int = 6) -> str:
    """#59 Inpaint per COMPOSITING: out = mask*edited + (1-mask)*original per-frame (feather sui bordi).
    L'area mascherata (SAM2-tracciata) prende il contenuto del video EDITATO (edit-anything col nuovo
    soggetto); il resto resta ESATTAMENTE l'originale. Sfondo perfettamente preservato."""
    import subprocess
    import cv2
    import numpy as np
    from PIL import Image
    tmpd = str(Path(out_video).with_suffix("")) + "_cf"
    os.makedirs(tmpd, exist_ok=True)
    od, ed = tmpd + "_o", tmpd + "_e"
    for d, v in ((od, original), (ed, edited)):
        os.makedirs(d, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-i", v, os.path.join(d, "%05d.png"), "-y"], check=True)
    of = sorted(Path(od).glob("*.png")); ef = sorted(Path(ed).glob("*.png"))
    n = min(len(of), len(ef), len(masks))
    for i in range(n):
        orig = np.asarray(Image.open(of[i]).convert("RGB"), np.float32)
        edit = np.asarray(Image.open(ef[i]).convert("RGB").resize(orig.shape[1::-1]), np.float32)
        m = np.asarray(Image.open(masks[i]).convert("L").resize(orig.shape[1::-1]), np.float32) / 255.0
        if feather > 0:
            m = cv2.GaussianBlur(m, (0, 0), feather)
        m = m[..., None]
        comp = (edit * m + orig * (1 - m)).clip(0, 255).astype(np.uint8)
        Image.fromarray(comp).save(os.path.join(tmpd, f"{i:05d}.png"))
    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", str(fps), "-i", os.path.join(tmpd, "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14", out_video, "-y"], check=True)
    return out_video


if __name__ == "__main__":   # CLI: track oggetto + costruisci guide magenta. Stampa "GUIDE:<path>".
    import sys
    if len(sys.argv) >= 4 and sys.argv[1] == "--find-people":          # «Trova persone» (tab Scambia): --find-people <video> <work_dir> [testo]
        find_people(sys.argv[2], sys.argv[3], text=(sys.argv[4] if len(sys.argv) > 4 else "person"), device=os.environ.get("LTX_SEG_DEVICE", "cuda"))
        print("TRACKS:" + os.path.join(sys.argv[3], "tracks.json"), flush=True)
        sys.exit(0)
    if len(sys.argv) < 4:
        print("uso: video_segment.py <video> <text> <out_guide.mp4> [dilate]"); sys.exit(2)
    video, text, out_guide = sys.argv[1], sys.argv[2], sys.argv[3]
    dil = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    r = track_object(video, text=text, dilate=dil, device=os.environ.get("LTX_SEG_DEVICE", "cuda"))
    g = build_masked_guide(r["frames"], r["masks"], out_guide, fps=r["fps"])
    print(f"GUIDE:{g}")
