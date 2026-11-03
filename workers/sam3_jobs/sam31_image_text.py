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

if any(a == "cpu" for a in sys.argv[1:]) and "--device" in sys.argv:        # --device cpu: nessuna GPU, nemmeno un contesto CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
    if dev == "cpu":
        print(f"[sam31] CPU: addmm_act float32 in {patch_fp32_cpu()} moduli", file=sys.stderr, flush=True)
    model = build_sam3_image_model(device="cpu", eval_mode=True, checkpoint_path=None, load_from_HF=False)
    sd = load_state(a.checkpoint)
    det = {k[len("detector."):]: v for k, v in sd.items() if k.startswith("detector.")}
    missing, unexpected = model.load_state_dict(det, strict=False)
    print(f"[sam31] checkpoint {os.path.basename(a.checkpoint)}: {len(det)} chiavi detector, mancanti {len(missing)} {list(missing)[:6]}, in piu {len(unexpected)}", file=sys.stderr, flush=True)
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
