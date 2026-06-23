"""Converte il Gemma-3-12B di ComfyUI (comfy_quant fp8 + naming vecchio) in formato standard
transformers (bf16) caricabile da ltx-core (GemmaTextEncoder → Gemma3ForConditionalGeneration).

- dequant:  weight(fp8) * weight_scale  ->  bf16 ; scarta .comfy_quant / .weight_scale
- rimappa:  model.layers/embed_tokens/norm -> model.model.language_model.*
            vision_model.* -> model.model.vision_tower.vision_model.*
            multi_modal_projector.* -> model.model.multi_modal_projector.*

Uso: <ltx2_venv>/python scripts/convert_gemma_comfy.py
"""
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = r"E:\AI\Models\TextEncoders\gemma3"
DST = r"E:\AI\Models\TextEncoders\gemma3_std"


def remap(k: str) -> str:
    if k.startswith(("model.layers.", "model.embed_tokens.", "model.norm.")):
        return "model.model.language_model." + k[len("model."):]
    if k.startswith("vision_model."):
        return "model.model.vision_tower." + k
    if k.startswith("multi_modal_projector."):
        return "model.model." + k
    return "model.model." + k


def main() -> None:
    src_file = os.path.join(SRC, "model.safetensors")
    with safe_open(src_file, framework="pt") as f:
        keys = list(f.keys())
        scales = {k[: -len(".weight_scale")]: f.get_tensor(k)
                  for k in keys if k.endswith(".weight_scale")}
        print(f"chiavi: {len(keys)} | scale (quantizzati): {len(scales)}", flush=True)
        out = {}
        n_dq = 0
        for k in keys:
            if k.endswith(".weight_scale") or k.endswith(".comfy_quant"):
                continue
            t = f.get_tensor(k)
            base = k[: -len(".weight")] if k.endswith(".weight") else None
            if base is not None and base in scales:
                t = t.to(torch.bfloat16) * scales[base].to(torch.bfloat16)
                n_dq += 1
            else:
                t = t.to(torch.bfloat16)
            out[remap(k)] = t.contiguous()
    print(f"dequantizzati {n_dq} tensori; totale output {len(out)}", flush=True)

    os.makedirs(DST, exist_ok=True)
    save_file(out, os.path.join(DST, "model.safetensors"), metadata={"format": "pt"})
    for fn in os.listdir(SRC):
        if fn == "model.safetensors":
            continue
        if fn.endswith((".json", ".model")) or "token" in fn.lower():
            shutil.copy(os.path.join(SRC, fn), os.path.join(DST, fn))
    print(f"DONE → {DST}", flush=True)


if __name__ == "__main__":
    main()
