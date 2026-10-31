"""Handler SEGMENTAZIONE nativa (GroundingDINO testo→box + SAM2 box→mask) — sostituisce ComfyUI.

Niente custom_node comfyui_segment_anything: usa transformers in-process.
Pesi HF in E:\\AI\\Models\\Segmentation\\hf\\{grounding-dino-base, sam2-hiera-large}.
Stato: scritto per il PoC, da validare quando i pesi A.4 sono scaricati (API SAM2 transformers).
"""
from __future__ import annotations

import os
from pathlib import Path

SEG_DIR = r"E:\AI\Models\Segmentation\hf"
# FASE F weights-unification: dir DAL registro unico, fallback ai letterali storici.
try:
    from core import model_paths as _mp
    GDINO_DIR = str(_mp.path_of("grounding-dino-base") or os.path.join(SEG_DIR, "grounding-dino-base"))
    SAM2_DIR = str(_mp.path_of("sam2-hiera-large") or os.path.join(SEG_DIR, "sam2-hiera-large"))
except Exception:                                        # noqa: BLE001
    GDINO_DIR = os.path.join(SEG_DIR, "grounding-dino-base")
    SAM2_DIR = os.path.join(SEG_DIR, "sam2-hiera-large")
VRAM_MB = 4000


def _mask_to_2d(masks, scores=None):
    """SAM2 → maschera 2D uint8*255. SAM2 restituisce 3 maschere per box (multimask: oggetto / sotto-parte /
    grezza≈box): unirle TUTTE (any) includerebbe la grezza → "quadrato". Con `scores` (iou_scores) si sceglie
    la MIGLIORE per box, poi unione TRA i box → contorno preciso dell'oggetto. Senza scores: fallback OR."""
    arr = masks > 0
    if arr.ndim == 4 and scores is not None:               # (n_box, n_mask, H, W) + scores (n_box, n_mask)
        # `post_process_masks(...)[0]` toglie la dimensione di BATCH alle maschere ma non agli
        # iou_scores, che restano (1, n_box, n_mask): con più di un box `scores[b]` era ancora 2-D e
        # `argmax()` (sul tensore appiattito) restituiva un indice fino a n_box*n_mask-1 → IndexError
        # sui 3 canali multimask. Si allinea la forma prima di scegliere, e se non torna si prende la
        # prima maschera invece di sollevare.
        s = scores
        while getattr(s, "ndim", 0) > 2 and int(s.shape[0]) == 1:
            s = s[0]
        n_box, n_mask = int(arr.shape[0]), int(arr.shape[1])
        per_box = getattr(s, "ndim", 0) == 2 and tuple(s.shape) == (n_box, n_mask) and n_mask > 1
        out = None
        for b in range(n_box):
            bi = int(s[b].argmax()) if per_box else 0
            mb = arr[b, bi]
            out = mb if out is None else (out | mb)
        if out is not None:
            return out.numpy().astype("uint8") * 255
    while arr.ndim > 2:                                     # ndim variabile (H,W / 1,H,W / …) → OR
        arr = arr.any(0)
    return arr.numpy().astype("uint8") * 255


def _load(_model: str = "", _arch: str = "segment"):
    import torch
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    # weights-unification C: prima un path assente finiva in from_pretrained → "HFValidationError:
    # Repo id must be in the form…" (criptico e fuorviante). Errore CHIARO con la voce di catalogo.
    if not os.path.isdir(GDINO_DIR):
        raise RuntimeError(f"GroundingDINO assente in {GDINO_DIR}: scaricalo dal tab «Pesi modelli» "
                           "(voce grounding-dino-base, ~0.9GB).")
    if not os.path.isdir(SAM2_DIR):
        raise RuntimeError(f"SAM2 assente in {SAM2_DIR}: scaricalo dal tab «Pesi modelli» "
                           "(voce sam2-hiera-large, ~0.9GB).")
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
    if params.get("region_point") and sproc is not None and smodel is not None:
        # CLIC = l'AI identifica e seleziona l'OGGETTO INTERO (punto 22): SAM2 con point prompt, multimask:
        # fra le 3 maschere (parte / sotto-oggetto / oggetto) si prende la PIU GRANDE fra quelle con IoU stimato
        # vicino al migliore (>= 0,8 x best), cosi non resta la «macchia» di colore sotto il clic.
        px, py = (float(v) for v in params["region_point"][:2])
        sin = sproc(images=img, input_points=[[[[px, py]]]], input_labels=[[[1]]], return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            sout = smodel(**sin, multimask_output=True)
        masks = sproc.post_process_masks(sout.pred_masks.cpu(), sin["original_sizes"].cpu())[0]
        arr = (masks > 0)
        while arr.ndim > 3:                                   # (n_punti, n_mask, H, W) -> (n_mask, H, W)
            arr = arr[0]
        sc = sout.iou_scores.cpu().float().reshape(-1)[-arr.shape[0]:]
        # Prova su CPU (31/10): per un clic sull'«occhio» di un oggetto SAM2 da [parte 0,985 | oggetto 0,06 | oggetto 0,07]: gli iou
        # stimati non indicano l'oggetto intero. Regola: fra le maschere che non invadono lo sfondo (<= 60 % della foto) si prende la PIU GRANDE.
        areas = [int(arr[i].sum()) for i in range(arr.shape[0])]
        lim = 0.6 * arr.shape[-1] * arr.shape[-2]
        cands = [i for i in range(arr.shape[0]) if 0 < areas[i] <= lim]
        pick = max(cands, key=lambda i: areas[i]) if cands else int(sc.argmax())
        m_np = arr[pick].numpy().astype("uint8") * 255
        os.makedirs(output_dir, exist_ok=True)
        path = str(Path(output_dir) / "native_segmask_point.png")
        _mout = Image.fromarray(m_np)
        if _mout.size != img.size:
            _mout = _mout.resize(img.size, Image.NEAREST)
        _mout.save(path)
        return {"status": "success", "output": [path]}
    if params.get("region_mask"):
        rm = np.array(Image.open(params["region_mask"]).convert("L"))
        ys, xs = np.where(rm > 127)
        if len(xs) == 0:
            return {"status": "error", "error": "region mask vuota"}
        box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        if sproc is not None and smodel is not None:
            sin = sproc(images=img, input_boxes=[[box]], return_tensors="pt").to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                sout = smodel(**sin)
            masks = sproc.post_process_masks(sout.pred_masks.cpu(), sin["original_sizes"].cpu())[0]
            m_np = _mask_to_2d(masks, sout.iou_scores.cpu() if hasattr(sout, "iou_scores") else None)
        else:
            m_np = np.zeros((img.height, img.width), dtype="uint8")
            x0, y0, x1, y1 = (int(v) for v in box)
            m_np[y0:y1, x0:x1] = 255
        os.makedirs(output_dir, exist_ok=True)
        path = str(Path(output_dir) / "native_segmask_region.png")
        _mout = Image.fromarray(m_np)
        if _mout.size != img.size:                           # la maschera DEVE combaciare con la foto
            _mout = _mout.resize(img.size, Image.NEAREST)
        _mout.save(path)
        return {"status": "success", "output": [path]}

    text = (params.get("text") or "object").strip()
    if not text.endswith("."):
        text += "."                                        # GroundingDINO vuole il punto finale
    thr = float(params.get("threshold", 0.3))

    gin = gproc(images=img, text=text, return_tensors="pt").to(dev)
    # autocast fp16 (audit acceleratori): DINO/SAM2 giravano in fp32 puro → +20-30% sui tensor-core
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
        gout = gmodel(**gin)
    # transformers recenti hanno rinominato box_threshold→threshold in post_process_grounded_object_detection
    try:
        res = gproc.post_process_grounded_object_detection(
            gout, gin["input_ids"], threshold=thr, text_threshold=0.25,
            target_sizes=[img.size[::-1]])[0]
    except TypeError:
        res = gproc.post_process_grounded_object_detection(
            gout, gin["input_ids"], box_threshold=thr, text_threshold=0.25,
            target_sizes=[img.size[::-1]])[0]
    boxes = res["boxes"].cpu().tolist()
    if not boxes:
        return {"status": "error", "error": f"nessun oggetto '{text}' trovato (soglia {thr})"}

    # use_sam=False → solo GroundingDINO (maschera rettangolare dai box, niente raffinamento SAM2):
    # selezionabile dall'utente ("solo groundingdino" vs "groundingdino+sam").
    _use_sam = params.get("use_sam", True)
    if _use_sam and sproc is not None and smodel is not None:   # SAM2: box → maschera precisa
        sin = sproc(images=img, input_boxes=[boxes], return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            sout = smodel(**sin)
        masks = sproc.post_process_masks(sout.pred_masks.cpu(), sin["original_sizes"].cpu())[0]
        m_np = _mask_to_2d(masks, sout.iou_scores.cpu() if hasattr(sout, 'iou_scores') else None)
    else:                                                  # solo box GroundingDINO (o SAM non disponibile)
        m_np = np.zeros((img.height, img.width), dtype="uint8")
        for x0, y0, x1, y1 in boxes:
            m_np[int(y0):int(y1), int(x0):int(x1)] = 255

    os.makedirs(output_dir, exist_ok=True)
    path = str(Path(output_dir) / "native_segmask.png")
    _mout = Image.fromarray(m_np)
    if _mout.size != img.size:                               # la maschera DEVE combaciare con la foto
        _mout = _mout.resize(img.size, Image.NEAREST)
    _mout.save(path)
    return {"status": "success", "output": [path]}
