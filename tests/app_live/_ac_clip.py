"""Worker CLIP dei controlli automatici (gira nel venv MMAudio: torch CPU + open_clip, nessuna GPU, nessun download).
Ingresso (stdin JSON): {"items": [{"image": p, "mask": p|None, "pad": 0.5, "texts": [..]}]}  ->  riga `@@JSON@@{"scores": [[cos...]]}`."""
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
MODEL = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"


def main():
    job = json.loads(sys.stdin.read())
    import numpy as np
    import open_clip
    import torch
    from PIL import Image
    m, _, pre = open_clip.create_model_and_transforms(MODEL)
    tok = open_clip.get_tokenizer(MODEL)
    m.eval()
    out = []
    with torch.no_grad():
        for it in job["items"]:
            im = Image.open(it["image"]).convert("RGB")
            if it.get("mask"):
                a = np.asarray(Image.open(it["mask"]).convert("L").resize(im.size)) > 127
                ys, xs = np.where(a)
                if len(xs):
                    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
                    s = int(max(x1 - x0, y1 - y0) * (1 + float(it.get("pad", 0.5)))) or 64
                    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                    im = im.crop((max(0, cx - s // 2), max(0, cy - s // 2), min(im.width, cx + s // 2), min(im.height, cy + s // 2)))
            fi = m.encode_image(pre(im).unsqueeze(0))
            fi = fi / fi.norm(dim=-1, keepdim=True)
            ft = m.encode_text(tok(it["texts"]))
            ft = ft / ft.norm(dim=-1, keepdim=True)
            out.append([round(float(x), 4) for x in (fi @ ft.T)[0]])
    print("@@JSON@@" + json.dumps({"scores": out}))


if __name__ == "__main__":
    main()
