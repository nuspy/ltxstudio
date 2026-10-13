"""Worker GENERICO di dequantizzazione comfy_quant → bf16 standard. Eseguito nel venv ComfyUI
(`E:\\AI\\Comfy_Nexus\\ComfyUI\\venv`, ha comfy.quant_ops + kernel CUDA fp8/nvfp4/mxfp8).

Converte un safetensors quantizzato di ComfyUI (chiavi .comfy_quant/.weight_scale[/_2]) in un
safetensors bf16 standard. Opzionale --remap gemma per riportare il naming Gemma3 a quello che
ltx-core/transformers si aspetta. Usato da core/dequant_comfy.py (automatico).

Uso: <comfy_venv>/python scripts/dequant_comfy_worker.py --src IN.safetensors --dst OUT.safetensors [--remap gemma|none]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, r"E:/AI/Comfy_Nexus/ComfyUI")
import comfy.quant_ops as q  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


def remap_gemma(k: str) -> str:
    if k.startswith(("model.layers.", "model.embed_tokens.", "model.norm.")):
        return "model.model.language_model." + k[len("model."):]
    if k.startswith("vision_model."):
        return "model.model.vision_tower." + k
    if k.startswith("multi_modal_projector."):
        return "model.model." + k
    return "model.model." + k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--remap", default="none", choices=["none", "gemma"])
    a = ap.parse_args()
    rm = remap_gemma if a.remap == "gemma" else (lambda k: k)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    f = safe_open(a.src, framework="pt")
    keys = list(f.keys())
    qfmt = {k[: -len(".comfy_quant")]: json.loads(bytes(f.get_tensor(k).tolist()))["format"]
            for k in keys if k.endswith(".comfy_quant")}
    FP8 = q.get_layout_class("TensorCoreFP8E4M3Layout")
    NV = q.get_layout_class("TensorCoreNVFP4Layout")
    out = {}
    nq = 0
    for k in keys:
        if k.endswith((".comfy_quant", ".weight_scale", ".weight_scale_2")):
            continue
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        if base in qfmt:
            w = f.get_tensor(k).to(dev)
            if qfmt[base] == "float8_e4m3fn":
                p = FP8.Params(scale=f.get_tensor(base + ".weight_scale").to(dev),
                               orig_dtype=torch.bfloat16, orig_shape=tuple(w.shape))
                d = FP8.dequantize(w, p)
            else:  # nvfp4
                p = NV.Params(scale=f.get_tensor(base + ".weight_scale_2").to(dev),
                              block_scale=f.get_tensor(base + ".weight_scale").to(dev),
                              orig_dtype=torch.bfloat16,
                              orig_shape=(w.shape[0], w.shape[1] * 2), transposed=False)
                d = NV.dequantize(w, p)
            out[rm(k)] = d.to(torch.bfloat16).cpu().contiguous()
            nq += 1
            del w, d
        else:
            out[rm(k)] = f.get_tensor(k).to(torch.bfloat16).contiguous()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.dst) or ".", exist_ok=True)
    save_file(out, a.dst, metadata={"format": "pt"})
    print(f"DONE dequant={nq} total={len(out)} -> {a.dst}", flush=True)


if __name__ == "__main__":
    main()
