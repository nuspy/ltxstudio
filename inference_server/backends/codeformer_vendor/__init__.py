"""CodeFormer nativo (arch vendorizzata dal custom node facerestore_cf, come da regola
"replica la soluzione ComfyUI in nativo"): la libreria gfpgan NON sa caricare i pesi
CodeFormer (arch VQGAN+transformer ≠ GFPGANv1Clean → Error(s) in loading state_dict).

build_codeformer(path) → rete pronta su CUDA; restore con net(t, w=fidelity, adain=True)[0].
"""
from __future__ import annotations


def build_codeformer(model_path: str, device: str = "cuda"):
    import torch
    from .codeformer_arch import CodeFormer
    net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                     connect_list=["32", "64", "128", "256"])
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("params_ema") or ckpt.get("params") or ckpt
    net.load_state_dict(sd, strict=True)
    return net.eval().to(device)
