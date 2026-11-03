"""Worker SAM 3.1 (immagine): testo -> riquadri / maschere. Gira nel venv condiviso E:/AI/Tools/diffusers_next_venv (transformers >= 5.17 ha Sam3Model;
l'inference_server_venv, transformers 4.57, no). Un processo per chiamata: carica il modello (core/sam31_loader.py), scrive le maschere in --out-dir e stampa
una riga `SAM31:{json}` con riquadri, punteggi e percorsi. Mai usato dalla GUI.

  python sam31_detect_worker.py --image foto.png --text person --out-dir DIR [--threshold 0.5] [--device cuda|cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="")
    a = ap.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from core import sam31_loader as L
    from core import segmentation_engine as SE
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.startswith("cuda") else torch.float32
    model, processor = L.load_image_model(Path(SE.models_dir()), dev, dtype)
    r = L.segment_image(model, processor, a.image, text=a.text, threshold=a.threshold)
    os.makedirs(a.out_dir, exist_ok=True)
    paths = []
    for i, m in enumerate(r["masks"]):
        p = os.path.join(a.out_dir, f"sam31_mask_{i:02d}.png")
        Image.fromarray(m.astype("uint8") * 255).save(p)
        paths.append(p)
    union = ""
    if len(r["masks"]):
        union = os.path.join(a.out_dir, "sam31_union.png")
        Image.fromarray(np.any(r["masks"], axis=0).astype("uint8") * 255).save(union)
    out = {"boxes": [[float(v) for v in b] for b in r["boxes"]], "scores": [float(s) for s in r["scores"]], "masks": paths, "union": union,
           "presence": float(r.get("presence", 1.0)), "size": [int(r["size"][0]), int(r["size"][1])], "device": dev}
    print("SAM31:" + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
