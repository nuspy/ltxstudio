"""SAM 3.1 (Meta) immagine: testo -> riquadri / maschere. Gira SOLO tramite il lanciatore E:\\AI\\Tools\\sam3_run.sh (diffusers_next_venv, -P -s, PYTHONPATH su
sam3_src revisionato e sam3_deps), mai col python dell'app. Inferenza senza rete: il checkpoint e locale (build_sam3_image_model non scarica nulla se
checkpoint_path e dato e load_from_HF=False). Il checkpoint si apre qui con torch.load(weights_only=True) o safetensors (copia Comfy-Org): mai col caricatore di Meta.

  bash E:/AI/Tools/sam3_run.sh sam31_image_text.py --image foto.png --text person --out-dir DIR [--checkpoint FILE] [--threshold 0.5] [--device cpu|cuda]

Stampa una riga `SAM31:{json}` (stesso formato di prima: boxes xyxy pixel, scores, masks PNG, union, presence, size, device).
Copia di lavoro: E:\\AI\\Tools\\sam3_jobs (la fonte nel repo e workers/sam3_jobs).
"""
import argparse
import json
import os
import sys

DEFAULT_CKPT = r"E:\AI\Models\Segmentation\sam3.1\sam3.1_multiplex.pt"


def load_state(path):
    """Dizionario di tensori float32 dal checkpoint: .pt di Meta (weights_only=True) oppure .safetensors (Comfy-Org, fp16 -> float32)."""
    import torch
    if str(path).lower().endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
        return {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return ck["model"] if isinstance(ck.get("model"), dict) else ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="")
    a = ap.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_sam3_image_model(device="cpu", eval_mode=True, checkpoint_path=None, load_from_HF=False)
    sd = load_state(a.checkpoint)
    det = {k[len("detector."):]: v for k, v in sd.items() if k.startswith("detector.")}
    missing, unexpected = model.load_state_dict(det, strict=False)
    print(f"[sam31] checkpoint {os.path.basename(a.checkpoint)}: {len(det)} chiavi detector, mancanti {len(missing)}, in piu {len(unexpected)}", file=sys.stderr, flush=True)
    del sd, det
    if dev.startswith("cuda"):
        model = model.cuda()
    proc = Sam3Processor(model, device=dev, confidence_threshold=a.threshold)
    img = Image.open(a.image).convert("RGB")
    state = proc.set_image(img)
    state = proc.set_text_prompt(a.text.strip().rstrip("."), state)
    masks = state["masks"].squeeze(1).cpu().numpy().astype(bool) if "masks" in state else np.zeros((0, img.height, img.width), bool)
    boxes = state["boxes"].float().cpu().numpy() if "boxes" in state else np.zeros((0, 4), "float32")
    scores = state["scores"].float().cpu().numpy() if "scores" in state else np.zeros((0,), "float32")
    os.makedirs(a.out_dir, exist_ok=True)
    paths = []
    for i, m in enumerate(masks):
        p = os.path.join(a.out_dir, f"sam31_mask_{i:02d}.png")
        Image.fromarray(m.astype("uint8") * 255).save(p)
        paths.append(p)
    union = ""
    if len(masks):
        union = os.path.join(a.out_dir, "sam31_union.png")
        Image.fromarray(np.any(masks, axis=0).astype("uint8") * 255).save(union)
    out = {"boxes": [[float(v) for v in b] for b in boxes], "scores": [float(s) for s in scores], "masks": paths, "union": union,
           "presence": 1.0, "size": [img.height, img.width], "device": dev, "missing": len(missing), "unexpected": len(unexpected)}
    print("SAM31:" + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
