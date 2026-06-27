"""Handler SEGMENTAZIONE nativa (GroundingDINO testo→box + SAM2 box→mask) — sostituisce ComfyUI.

Niente custom_node comfyui_segment_anything: usa transformers in-process.
Pesi HF in E:\\AI\\Models\\Segmentation\\hf\\{grounding-dino-base, sam2-hiera-large}.
Stato: scritto per il PoC, da validare quando i pesi A.4 sono scaricati (API SAM2 transformers).
"""
from __future__ import annotations

import os
from pathlib import Path

SEG_DIR = r"E:\AI\Models\Segmentation\hf"
GDINO_DIR = os.path.join(SEG_DIR, "grounding-dino-base")
SAM2_DIR = os.path.join(SEG_DIR, "sam2-hiera-large")
VRAM_MB = 4000


def _load(_model: str = "", _arch: str = "segment"):
    import torch
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gproc = AutoProcessor.from_pretrained(GDINO_DIR)
    gmodel = GroundingDinoForObjectDetection.from_pretrained(GDINO_DIR).to(dev).eval()
    sproc = smodel = None
    try:
        from transformers import Sam2Processor, Sam2Model
        sproc = Sam2Processor.from_pretrained(SAM2_DIR)
        smodel = Sam2Model.from_pretrained(SAM2_DIR).to(dev).eval()
    except Exception as e:                                  # noqa: BLE001
        print(f"[segment] SAM2 non caricato ({e}) → solo box GroundingDINO", flush=True)
    return {"gproc": gproc, "gmodel": gmodel, "sproc": sproc, "smodel": smodel, "dev": dev}, VRAM_MB


def run(params: dict, mm, output_dir: str) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    key = "segment:gdino_sam2"
    mm.register(key, _load)
    m = mm.ensure(key, need_mb=VRAM_MB)
    gproc, gmodel, sproc, smodel, dev = m["gproc"], m["gmodel"], m["sproc"], m["smodel"], m["dev"]

    img = Image.open(params["image_path"]).convert("RGB")

    # MODALITÀ REGIONE (selezione magnetica da box/click): bbox dalla region mask → SAM2, no GroundingDINO
    if params.get("region_mask"):
        rm = np.array(Image.open(params["region_mask"]).convert("L"))
        ys, xs = np.where(rm > 127)
        if len(xs) == 0:
            return {"status": "error", "error": "region mask vuota"}
        box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        if sproc is not None and smodel is not None:
            sin = sproc(images=img, input_boxes=[[box]], return_tensors="pt").to(dev)
            with torch.no_grad():
                sout = smodel(**sin)
            masks = sproc.post_process_masks(sout.pred_masks.cpu(), sin["original_sizes"].cpu())[0]
            m_np = (masks.sum(0) > 0).numpy().astype("uint8") * 255 if masks.ndim == 3 \
                else (masks[0] > 0).numpy().astype("uint8") * 255
        else:
            m_np = np.zeros((img.height, img.width), dtype="uint8")
            x0, y0, x1, y1 = (int(v) for v in box)
            m_np[y0:y1, x0:x1] = 255
        os.makedirs(output_dir, exist_ok=True)
        path = str(Path(output_dir) / "native_segmask_region.png")
        Image.fromarray(m_np).save(path)
        return {"status": "success", "output": [path]}

    text = (params.get("text") or "object").strip()
    if not text.endswith("."):
        text += "."                                        # GroundingDINO vuole il punto finale
    thr = float(params.get("threshold", 0.3))

    gin = gproc(images=img, text=text, return_tensors="pt").to(dev)
    with torch.no_grad():
        gout = gmodel(**gin)
    res = gproc.post_process_grounded_object_detection(
        gout, gin["input_ids"], box_threshold=thr, text_threshold=0.25,
        target_sizes=[img.size[::-1]])[0]
    boxes = res["boxes"].cpu().tolist()
    if not boxes:
        return {"status": "error", "error": f"nessun oggetto '{text}' trovato (soglia {thr})"}

    if sproc is not None and smodel is not None:           # SAM2: box → maschera precisa
        sin = sproc(images=img, input_boxes=[boxes], return_tensors="pt").to(dev)
        with torch.no_grad():
            sout = smodel(**sin)
        masks = sproc.post_process_masks(sout.pred_masks.cpu(), sin["original_sizes"].cpu())[0]
        m_np = (masks.sum(0) > 0).numpy().astype("uint8") * 255 if masks.ndim == 3 \
            else (masks[0] > 0).numpy().astype("uint8") * 255
    else:                                                  # fallback: maschera dai box
        m_np = np.zeros((img.height, img.width), dtype="uint8")
        for x0, y0, x1, y1 in boxes:
            m_np[int(y0):int(y1), int(x0):int(x1)] = 255

    os.makedirs(output_dir, exist_ok=True)
    path = str(Path(output_dir) / "native_segmask.png")
    Image.fromarray(m_np).save(path)
    return {"status": "success", "output": [path]}
