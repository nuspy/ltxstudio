"""SAM 3.1 (detector per immagini: prompt di testo e/o box -> maschere) con ``transformers.Sam3Model``.

Pesi: ``E:\\AI\\Models\\Segmentation\\sam3.1`` = repo ``unsloth/sam3.1`` gia' nel formato transformers (``model.safetensors``
BF16 1,68 GB + config/processor/tokenizer).  Nessuna conversione, nessun pacchetto ``sam3`` di Meta, nessun ComfyUI.

Uso tipico (CPU, float32)::

    model, processor = load_image_model(r"E:\\AI\\Models\\Segmentation\\sam3.1")
    r = segment_image(model, processor, "foto.jpg", text="person")         # r["masks"] (N,H,W) bool, r["boxes"], r["scores"]
    rs = segment_video_frames(model, processor, [f0, f1, f2], text="person")  # lista di dict, uno per fotogramma

Note sulla precisione su CPU: ``load_image_model`` default float32 (``dtype=torch.float32``): i pesi BF16 del file vengono
promossi alla lettura.  BF16 su CPU e' esposto ma NON e' il default: vedi la nota in ``load_image_model``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

DEFAULT_MODEL_DIR = Path(r"E:\AI\Models\Segmentation\sam3.1")


def weights_ready(model_dir: str | Path = DEFAULT_MODEL_DIR) -> bool:
    """True se ``model.safetensors`` esiste completo (il download scrive prima ``model.safetensors.part``)."""
    return (Path(model_dir) / "model.safetensors").is_file()


def load_image_model(model_dir: str | Path = DEFAULT_MODEL_DIR, device: str = "cpu", dtype: torch.dtype = torch.float32):
    """Carica ``Sam3Model`` + ``Sam3Processor`` dalla cartella (formato transformers).  Ritorna ``(model, processor)``.

    ``dtype``: float32 di default.  Misurato su CPU (24 thread, frame 1280x704): float32 ~8 s/immagine, RAM di picco
    ~5,6 GiB; ``torch.bfloat16`` (pesi puri, SENZA autocast) funziona e da' lo stesso risultato (IoU 0,9988, punteggi
    entro 0,003) ma e' ~30x piu' lento (~250 s/immagine): su CPU non conviene.  Con ``device="cuda"`` si puo' usare
    ``torch.bfloat16`` (formato nativo del file).  Il modello e' in ``eval()`` e senza gradienti.
    """
    from transformers import Sam3Model, Sam3Processor

    model_dir = Path(model_dir)
    if not weights_ready(model_dir):
        raise FileNotFoundError(f"manca {model_dir / 'model.safetensors'} (download non completato?)")
    model = Sam3Model.from_pretrained(str(model_dir), dtype=dtype)
    model = model.to(device).eval().requires_grad_(False)
    processor = Sam3Processor.from_pretrained(str(model_dir))
    return model, processor


# --------------------------------------------------------------------------------------------------------------
def _to_pil(image):
    """Percorso, ``PIL.Image`` o array numpy HxWx3 (RGB uint8) -> ``PIL.Image`` RGB."""
    import numpy as np
    from PIL import Image

    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image.convert("RGB")


def _unpack(results: dict, out, H: int, W: int) -> dict[str, Any]:
    import numpy as np

    presence = float(out.presence_logits.float().sigmoid().reshape(-1)[0]) if out.presence_logits is not None else 1.0
    return {
        "masks": results["masks"].cpu().numpy().astype(bool),        # (N, H, W)
        "boxes": results["boxes"].float().cpu().numpy().astype(np.float32),  # (N, 4) xyxy pixel
        "scores": results["scores"].float().cpu().numpy().astype(np.float32),  # (N,)
        "presence": presence,   # probabilita' che il concetto compaia nell'immagine (0..1)
        "size": (H, W),
    }


def _forward(model, processor, img, text, boxes, box_labels, threshold, mask_threshold, text_cache=None):
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    W, H = img.size
    kw: dict[str, Any] = {}
    if boxes:
        kw["input_boxes"] = [[[float(v) for v in b] for b in boxes]]
        kw["input_boxes_labels"] = [list(box_labels) if box_labels is not None else [1] * len(boxes)]
    if text_cache is not None and not boxes:
        # testo gia' codificato (uguale per tutti i fotogrammi): si codifica solo l'immagine
        inputs = processor(images=img, return_tensors="pt")
        pv = inputs["pixel_values"].to(dev, dt)
        with torch.inference_mode():
            out = model(pixel_values=pv, text_embeds=text_cache["emb"], attention_mask=text_cache["mask"])
    else:
        inputs = processor(images=img, text=text or None, return_tensors="pt", **kw)
        inputs = {k: (v.to(dev, dt) if v.is_floating_point() else v.to(dev))
                  for k, v in inputs.items() if k != "original_sizes" and hasattr(v, "to")}
        with torch.inference_mode():
            out = model(**inputs)
    res = processor.post_process_instance_segmentation(
        out, threshold=threshold, mask_threshold=mask_threshold, target_sizes=[(H, W)])[0]
    return _unpack(res, out, H, W)


def segment_image(model, processor, image, text: str | None = "person", boxes: Sequence[Sequence[float]] | None = None,
                  threshold: float = 0.5, mask_threshold: float = 0.5, box_labels: Sequence[int] | None = None) -> dict[str, Any]:
    """Segmenta UN'immagine con prompt di testo e/o box.

    ``image``: percorso, ``PIL.Image`` o array RGB.  ``boxes``: lista di ``[x1, y1, x2, y2]`` in pixel dell'immagine
    originale; ``box_labels``: 1 = positiva (default), 0 = negativa.  Con sole box il testo e' "visual" (convenzione
    transformers).  Ritorna ``masks`` (N,H,W) bool numpy alla risoluzione originale, ``boxes`` (N,4) xyxy pixel,
    ``scores`` (N,) (sigmoide query x sigmoide presenza), ``presence`` (0..1), ``size`` (H, W).
    """
    if not text and not boxes:
        raise ValueError("serve almeno un testo o una box")
    return _forward(model, processor, _to_pil(image), text, boxes, box_labels, threshold, mask_threshold)


def segment_video_frames(model, processor, frames: Iterable[Any], text: str, threshold: float = 0.5,
                         mask_threshold: float = 0.5) -> list[dict[str, Any]]:
    """Applica lo stesso prompt di testo a una lista di fotogrammi chiave (percorsi, PIL o array RGB).

    Serve a sostituire GroundingDINO nel tracking: SAM 3 trova l'oggetto dal testo sui keyframe, un tracker
    (SAM 2) propaga.  Il testo e' codificato UNA volta sola; ogni fotogramma e' indipendente (nessuno stato tra
    fotogrammi).  Ritorna una lista di dict come ``segment_image`` (stesso ordine dei fotogrammi).
    """
    if not text:
        raise ValueError("serve un testo")
    ti = processor.tokenizer(text, return_tensors="pt", padding="max_length", max_length=32)
    dev = next(model.parameters()).device
    with torch.inference_mode():
        emb = model.get_text_features(input_ids=ti["input_ids"].to(dev), attention_mask=ti["attention_mask"].to(dev),
                                      return_dict=True)
    cache = {"emb": emb, "mask": ti["attention_mask"].to(dev)}
    return [_forward(model, processor, _to_pil(f), text, None, None, threshold, mask_threshold, text_cache=cache)
            for f in frames]
