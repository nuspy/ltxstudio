"""Confronto Meta (.pt) vs Comfy-Org (.safetensors fp16) CARICATI dal codice di Meta: stesso fotogramma, stesso testo, maschere e punteggi del detector.
Gira SOLO tramite il lanciatore E:\\AI\\Tools\\sam3_run.sh. Il confronto tensore per tensore (senza codice di Meta) sta in scripts/sam31_compare_checkpoints.py.

  bash E:/AI/Tools/sam3_run.sh sam31_compare_ckpts_image.py --image frame.png --text person [--device cpu]
"""
import argparse
import json
import os

META = r"E:\AI\Models\Segmentation\sam3.1\sam3.1_multiplex.pt"
COMFY = r"E:\AI\Models\Segmentation\sam3.1_multiplex\sam3.1_multiplex_fp16.safetensors"


def load_state(path):
    import torch
    if path.lower().endswith(".safetensors"):
        from safetensors.torch import load_file
        return {k: (v.float() if v.is_floating_point() else v) for k, v in load_file(path).items()}
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return ck["model"] if isinstance(ck.get("model"), dict) else ck


def run(path, image, text, dev, thr):
    import numpy as np
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(device="cpu", eval_mode=True, checkpoint_path=None, load_from_HF=False)
    sd = load_state(path)
    model.load_state_dict({k[len("detector."):]: v for k, v in sd.items() if k.startswith("detector.")}, strict=False)
    del sd
    if dev.startswith("cuda"):
        model = model.cuda()
    proc = Sam3Processor(model, device=dev, confidence_threshold=thr)
    st = proc.set_text_prompt(text, proc.set_image(image))
    masks = st["masks"].squeeze(1).cpu().numpy().astype(bool)
    return masks, st["boxes"].float().cpu().numpy(), st["scores"].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--text", default="person")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    import numpy as np
    from PIL import Image
    img = Image.open(a.image).convert("RGB")
    mm, bm, sm = run(META, img, a.text, a.device, a.threshold)
    mc, bc, sc = run(COMFY, img, a.text, a.device, a.threshold)
    out = {"meta": {"n": int(len(sm)), "scores": [round(float(s), 4) for s in sm]}, "comfy": {"n": int(len(sc)), "scores": [round(float(s), 4) for s in sc]}}
    if len(sm) and len(sc):
        order_m, order_c = np.argsort(-sm), np.argsort(-sc)
        ious = []
        for i, j in zip(order_m, order_c):
            u = (mm[i] | mc[j]).sum()
            ious.append(float((mm[i] & mc[j]).sum() / u) if u else 1.0)
        out["iou_per_oggetto_ordinati_per_punteggio"] = [round(x, 4) for x in ious]
        out["max_diff_punteggio"] = float(max(abs(float(sm[i]) - float(sc[j])) for i, j in zip(order_m, order_c)))
    print("SAM31CMP:" + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
