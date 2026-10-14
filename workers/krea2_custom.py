"""Krea 2: transformer da FILE SINGOLO (finetune community, es. Krea Noct V4 int8 ConvRot) + LoRA su Linear quantizzati.

Le cartelle ufficiali Krea-2-Turbo/Raw sono snapshot diffusers in fp32 (~52 GB di transformer). I finetune
community arrivano invece come single-file ComfyUI; core/native_convert («krea2_dit») li porta ai nomi diffusers
tenendo i Linear int8 compressi (+ marcatore `.convrot_group`). Qui il transformer si costruisce vuoto dalla config
dello snapshot di base e si riempie con `_qlinear` (dequant bf16 un layer alla volta nel forward): text encoder,
VAE e scheduler restano quelli dello snapshot. PEFT non sa avvolgere i moduli quantizzati → le LoRA si agganciano
«al volo» con `_qlinear.apply_lora`.
"""
from __future__ import annotations

import os
import re
import sys

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:      # `import _qlinear` anche dal worker
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ComfyUI (nomi del codice originale Krea) → Krea2Transformer2DModel di diffusers. Verificato tensore per tensore
# sull'header di KreaNoct_V4 contro lo snapshot ufficiale Krea-2-Turbo (13/10/2026).
_PREFIX = (
    ("blocks.", "transformer_blocks."), ("txtfusion.", "text_fusion."),
    ("first.", "img_in."), ("last.linear.", "final_layer.linear."),
    ("last.norm.scale", "final_layer.norm.weight"), ("last.modulation.lin", "final_layer.scale_shift_table"),
    ("tmlp.0.", "time_embed.linear_1."), ("tmlp.2.", "time_embed.linear_2."), ("tproj.1.", "time_mod_proj."),
    ("txtmlp.0.scale", "txt_in.norm.weight"), ("txtmlp.1.", "txt_in.linear_1."), ("txtmlp.3.", "txt_in.linear_2."),
)
_INNER = (
    (".attn.wq.", ".attn.to_q."), (".attn.wk.", ".attn.to_k."), (".attn.wv.", ".attn.to_v."),
    (".attn.wo.", ".attn.to_out.0."), (".attn.gate.", ".attn.to_gate."),
    (".attn.qknorm.qnorm.scale", ".attn.norm_q.weight"), (".attn.qknorm.knorm.scale", ".attn.norm_k.weight"),
    (".mlp.", ".ff."), (".prenorm.scale", ".norm1.weight"), (".postnorm.scale", ".norm2.weight"),
    (".mod.lin", ".scale_shift_table"),
)
_BLOCK_MOD = re.compile(r"^blocks\.\d+\.mod\.lin$")


def dit_key(k: str) -> str:
    """Nome ComfyUI → nome diffusers (chiavi già diffusers passano invariate)."""
    for pre in ("model.diffusion_model.", "diffusion_model."):
        if k.startswith(pre):
            k = k[len(pre):]
    for a, b in _PREFIX:
        if k.startswith(a):
            k = b + k[len(a):]
            break
    for a, b in _INNER:
        k = k.replace(a, b)
    return k


def is_block_modulation(k: str) -> bool:
    """`blocks.N.mod.lin` è piatto [6·dim]: in diffusers è `scale_shift_table` [6, dim]."""
    return bool(_BLOCK_MOD.match(k))


_KOHYA = re.compile(r"^lora_unet_(.+?)\.(lora_down\.weight|lora_up\.weight|alpha|lokr_\w+|diff|diff_b)$")


def _kohya_module(m: str) -> str:
    """`blocks_12_attn_gate` → `blocks.12.attn.gate`, `txtfusion_refiner_blocks_0_mlp_up` → `txtfusion.refiner_blocks.0.mlp.up`."""
    m = re.sub(r"^txtfusion_", "txtfusion.", m)
    m = re.sub(r"(blocks|layerwise_blocks|refiner_blocks)_(\d+)_", r"\1.\2.", m)
    return re.sub(r"\.(attn|mlp)_", r".\1.", m)


def lora_key(k: str) -> str:
    """Chiave LoRA ComfyUI/ai-toolkit/kohya → nome del modulo diffusers (suffisso lora_A/lora_B/alpha invariato)."""
    m = _KOHYA.match(k)
    if m:
        k = f"{_kohya_module(m.group(1))}.{m.group(2)}"
    for pre in ("transformer.", "model.diffusion_model.", "diffusion_model."):
        if k.startswith(pre):
            k = k[len(pre):]
            break
    return dit_key(k)


def build_transformer(dit_path: str, config_dir: str, device: str = "cuda"):
    """Krea2Transformer2DModel vuoto dalla config dello snapshot di base, riempito dal file nativo
    (int8 ConvRot / fp8 / bf16) con i Linear quantizzati di `_qlinear`."""
    import torch
    from accelerate import init_empty_weights
    from diffusers import Krea2Transformer2DModel
    import _qlinear as Q
    with init_empty_weights():
        tr = Krea2Transformer2DModel.from_config(Krea2Transformer2DModel.load_config(config_dir))
    sd = Q.quantize_from_state_dict(tr, Q.read_safetensors(dit_path, device))
    miss, unex = Q.load_into(tr, sd, device=device, dtype=torch.bfloat16,
                             keep_fp32=("norm",))      # = _keep_in_fp32_modules di Krea2Transformer2DModel
    del sd
    if miss or unex:
        raise RuntimeError(f"Krea 2 single-file {os.path.basename(dit_path)}: chiavi mancanti {miss[:6]} "
                           f"/ inattese {unex[:6]}")
    return tr.to(device).eval()


def build_text_encoder(te_path: str, config_dir: str, device: str = "cpu"):
    """Qwen3VLModel (text encoder di Krea 2) da file nativo «krea2_te»: Linear fp8 compressi (RowFp8Linear), resto
    bf16. Caricato su `device` (CPU di default): lo sposta la pipeline col suo placement (GPU o cpu_offload)."""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3VLModel
    import _qlinear as Q
    with init_empty_weights():
        te = Qwen3VLModel._from_config(AutoConfig.from_pretrained(config_dir), torch_dtype=torch.bfloat16)
    sd = Q.quantize_from_state_dict(te, Q.read_safetensors(te_path, device))
    miss, unex = Q.load_into(te, sd, device=device, dtype=torch.bfloat16)
    del sd
    if miss or unex:
        raise RuntimeError(f"text encoder Krea 2 {os.path.basename(te_path)}: chiavi mancanti {miss[:6]} "
                           f"/ inattese {unex[:6]}")
    return te.eval()


def apply_loras(tr, loras: list[tuple[str, float]], log=print) -> int:
    """LoRA (nomi ComfyUI o diffusers) agganciate ai Linear del transformer, quantizzati o no."""
    from safetensors.torch import load_file
    import _qlinear as Q
    n = 0
    for path, w in loras:
        sd = {lora_key(k): v for k, v in load_file(path).items()}
        got = Q.apply_lora(tr, sd, float(w))
        log(f"[krea2] LoRA {os.path.basename(path)} @{w}: {got} moduli")
        if not got:
            raise RuntimeError(f"LoRA {os.path.basename(path)}: nessun modulo Krea 2 corrispondente")
        n += got
    return n
