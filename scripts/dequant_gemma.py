"""WORKAROUND: dequantizza il Gemma-3-12B di ComfyUI (gemma3, comfy_quant fp8+nvfp4) in formato
bf16 standard transformers caricabile da ltx-core — ZERO download.

Eseguito nel venv ComfyUI (`E:\\AI\\Comfy_Nexus\\ComfyUI\\venv`) che ha comfy.quant_ops + i kernel
CUDA di dequant. Rimappa il naming al formato Gemma3ForConditionalGeneration e salva in gemma-3-ltx.
"""
import glob
import json
import os
import sys

import torch

sys.path.insert(0, r"E:/AI/Comfy_Nexus/ComfyUI")
import comfy.quant_ops as q  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

SRC = r"E:\AI\Models\TextEncoders\gemma3\model.safetensors"
DST_DIR = r"E:\AI\Models\TextEncoders\gemma-3-ltx"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def remap(k: str) -> str:
    if k.startswith(("model.layers.", "model.embed_tokens.", "model.norm.")):
        return "model.model.language_model." + k[len("model."):]
    if k.startswith("vision_model."):
        return "model.model.vision_tower." + k
    if k.startswith("multi_modal_projector."):
        return "model.model." + k
    return "model.model." + k


def main() -> None:
    f = safe_open(SRC, framework="pt")
    keys = list(f.keys())
    qfmt = {k[: -len(".comfy_quant")]: json.loads(bytes(f.get_tensor(k).tolist()))["format"]
            for k in keys if k.endswith(".comfy_quant")}
    FP8 = q.get_layout_class("TensorCoreFP8E4M3Layout")
    NV = q.get_layout_class("TensorCoreNVFP4Layout")
    out = {}
    n_fp8 = n_nv = 0
    for k in keys:
        if k.endswith((".comfy_quant", ".weight_scale", ".weight_scale_2")):
            continue
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        if base in qfmt:
            w = f.get_tensor(k).to(DEV)
            if qfmt[base] == "float8_e4m3fn":
                p = FP8.Params(scale=f.get_tensor(base + ".weight_scale").to(DEV),
                               orig_dtype=torch.bfloat16, orig_shape=tuple(w.shape))
                d = FP8.dequantize(w, p); n_fp8 += 1
            else:  # nvfp4
                p = NV.Params(scale=f.get_tensor(base + ".weight_scale_2").to(DEV),
                              block_scale=f.get_tensor(base + ".weight_scale").to(DEV),
                              orig_dtype=torch.bfloat16,
                              orig_shape=(w.shape[0], w.shape[1] * 2), transposed=False)
                d = NV.dequantize(w, p); n_nv += 1
            out[remap(k)] = d.to(torch.bfloat16).cpu().contiguous()
            del w, d
        else:
            out[remap(k)] = f.get_tensor(k).to(torch.bfloat16).contiguous()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"dequant: fp8={n_fp8} nvfp4={n_nv} | tensori totali={len(out)}", flush=True)

    os.makedirs(DST_DIR, exist_ok=True)
    for p in glob.glob(os.path.join(DST_DIR, "model-*.safetensors")):
        os.remove(p)
    idx = os.path.join(DST_DIR, "model.safetensors.index.json")
    if os.path.exists(idx):
        os.remove(idx)
    save_file(out, os.path.join(DST_DIR, "model.safetensors"), metadata={"format": "pt"})
    print("DONE -> " + DST_DIR, flush=True)


if __name__ == "__main__":
    main()
