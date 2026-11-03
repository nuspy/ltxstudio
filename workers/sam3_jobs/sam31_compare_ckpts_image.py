"""Confronto Meta (.pt) vs Comfy-Org (.safetensors fp16) CARICATI dal codice di Meta: stesso fotogramma, stesso testo, maschere e punteggi del detector.
Gira SOLO tramite il lanciatore E:\\AI\\Tools\\sam3_run.sh. Il confronto tensore per tensore (senza codice di Meta) sta in scripts/sam31_compare_checkpoints.py.

  bash E:/AI/Tools/sam3_run.sh sam31_compare_ckpts_image.py --image frame.png --text person [--device cpu]
"""
import argparse
import json
import os
import sys

if any(a == "cpu" for a in sys.argv[1:]) and "--device" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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


def patch_fp32_cpu():
    """Su CPU il codice di Meta forza matmul bfloat16 (perflib.fused.addmm_act: kernel pensati per GPU, sotto autocast bf16) e con pesi float32 va in errore di dtype.
    Qui si sostituisce SOLO quella funzione con la versione float32 equivalente (linear + attivazione): stessi numeri, niente bf16 lento sulla CPU."""
    import sys
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def addmm_act(activation, linear, mat1):
        y = linear(mat1)
        if isinstance(activation, type) and issubclass(activation, nn.Module):
            return activation()(y)
        if activation is F.relu:
            return F.relu(y)
        if activation is F.gelu:
            return F.gelu(y)
        raise ValueError(f"Unexpected activation {activation}")
    n = 0
    for name, mod in list(sys.modules.items()):
        if name.startswith("sam3") and getattr(mod, "addmm_act", None) is not None:
            mod.addmm_act = addmm_act
            n += 1
    # Il codice di Meta scrive device="cuda" fisso in molti punti (decoder, posizioni, ecc.): in modalita CPU ogni richiesta di device cuda diventa cpu.
    def _cpu_kw(fn):
        def w(*a, **k):
            d = k.get("device")
            if d is not None and str(d).startswith("cuda"):
                k["device"] = "cpu"
            if k.get("dtype") in (torch.bfloat16, torch.float16):          # CPU: tutto in float32 (il codice di Meta presume l'autocast bf16 della GPU)
                k["dtype"] = torch.float32
            return fn(*a, **k)
        return w
    for nm in ("arange", "zeros", "ones", "empty", "full", "tensor", "as_tensor", "linspace", "rand", "randn", "zeros_like", "ones_like", "eye", "randperm", "empty_like"):
        if hasattr(torch, nm):
            setattr(torch, nm, _cpu_kw(getattr(torch, nm)))
    import types
    torch.cuda.get_device_properties = lambda *a, **k: types.SimpleNamespace(major=0, minor=0, name="cpu", total_memory=0, multi_processor_count=1)   # lettura al momento dell'import
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.Tensor.pin_memory = lambda self, *a, **k: self
    torch.Tensor.bfloat16 = lambda self, *a, **k: self.float()
    torch.Tensor.half = lambda self, *a, **k: self.float()
    nn.Module.cuda = lambda self, *a, **k: self
    _to = torch.Tensor.to

    def _tensor_to(self, *a, **k):
        a = tuple("cpu" if isinstance(x, (str, torch.device)) and str(x).startswith("cuda") else (torch.float32 if isinstance(x, torch.dtype) and x in (torch.bfloat16, torch.float16) else x) for x in a)
        if k.get("dtype") in (torch.bfloat16, torch.float16):
            k["dtype"] = torch.float32
        if str(k.get("device", "")).startswith("cuda"):
            k["device"] = "cpu"
        return _to(self, *a, **k)
    torch.Tensor.to = _tensor_to
    # La cache delle posizioni del builder viene precalcolata con device="cuda" scritto fisso (position_encoding.py:55): su CPU si rimanda al primo uso.
    import sam3.model_builder as mb
    from sam3.model.position_encoding import PositionEmbeddingSine
    mb._create_position_encoding = lambda precompute_resolution=None: PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=10000,
                                                                                            precompute_resolution=None)
    return n


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
    import sam3.model.vitdet  # noqa: F401  (i moduli di Meta devono essere importati prima del patch)
    import sam3.model_builder  # noqa: F401
    if a.device == "cpu":
        patch_fp32_cpu()
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
